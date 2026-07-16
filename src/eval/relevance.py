"""Proxy relevance labelling for offline NDCG evaluation.

Assigns integer relevance labels (0/1/2) to scholarship rows based on
keyword overlap, text-similarity thresholds, and hard profile-match signals.
The labels are used as a proxy for human judgement in the absence of ground-
truth click or award data.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING, Literal

import pandas as pd

from src.text_utils import normalize_list as _normalize_list
from src.text_utils import normalize_text as _normalize_text

if TYPE_CHECKING:
    from src.eval.golden_students import GoldenStudent


@dataclass(frozen=True, slots=True)
class RelevanceConfig:
    """Configuration for the proxy relevance labelling heuristic.

    Attributes:
        label_mode: ``"hybrid"`` uses text similarity as a tiebreaker;
            ``"no_similarity"`` relies on keyword overlap only.
        tfidf_threshold: Minimum TF-IDF cosine similarity to assign label ≥ 1.
        embed_threshold: Minimum embedding cosine similarity to assign label ≥ 1.
        strict_requires_all_of: Profile fields that must all match for label 2.
        require_major_match_for_label2: When ``True`` (default), label 2 requires
            an *explicit* major match — a scholarship that lists no
            ``majors_allowed`` no longer passes the major check vacuously. This
            stops unrestricted scholarships (open to all majors) from earning the
            top relevance label on keyword overlap alone. Set ``False`` to restore
            the older "compatible-or-unrestricted" behavior.
    """

    label_mode: Literal["hybrid", "no_similarity"] = "hybrid"
    tfidf_threshold: float = 0.12
    embed_threshold: float = 0.30
    strict_requires_all_of: tuple[str, ...] = ("major", "state", "education_level")
    require_major_match_for_label2: bool = True


DEFAULT_RELEVANCE_CONFIG = RelevanceConfig()


def _keyword_overlap_positive(row: pd.Series) -> bool:
    overlap = row.get("keyword_overlap")
    if overlap is None or pd.isna(overlap):
        return False
    return float(overlap) > 0.0


def _strict_profile_match(
    row: pd.Series,
    student: GoldenStudent,
    cfg: RelevanceConfig,
) -> bool:
    profile = student.profile
    majors_allowed = _normalize_list(row.get("majors_allowed"))
    states_allowed = _normalize_list(row.get("states_allowed"))
    scholarship_edu = _normalize_text(row.get("education_level"))
    profile_major = _normalize_text(profile.major)
    if cfg.require_major_match_for_label2:
        major_match = bool(majors_allowed) and (profile_major in majors_allowed)
    else:
        major_match = (not majors_allowed) or (profile_major in majors_allowed)
    checks: dict[str, bool] = {
        "major": major_match,
        "state": (not states_allowed) or (_normalize_text(profile.state) in states_allowed),
        "education_level": (not scholarship_edu)
        or (scholarship_edu == _normalize_text(profile.education_level)),
    }
    return all(checks.get(field, False) for field in cfg.strict_requires_all_of)


def get_similarity_threshold(similarity_mode: str, cfg: RelevanceConfig) -> float:
    """Return the text-similarity threshold for the given mode from ``cfg``.

    Args:
        similarity_mode: Either ``"tfidf"`` or ``"embeddings"``.
        cfg: Relevance configuration holding the per-mode thresholds.

    Raises:
        ValueError: For unsupported similarity modes.
    """
    if similarity_mode == "tfidf":
        return cfg.tfidf_threshold
    if similarity_mode == "embeddings":
        return cfg.embed_threshold
    raise ValueError(f"Unsupported similarity mode: {similarity_mode}")


def _text_similarity(row: pd.Series) -> float:
    value = row.get("text_sim")
    if value is None or pd.isna(value):
        value = row.get("tfidf_sim")
    if value is None or pd.isna(value):
        value = row.get("embed_sim")
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def proxy_relevance_label(
    row: pd.Series,
    student: GoldenStudent,
    *,
    similarity_mode: str,
    cfg: RelevanceConfig = DEFAULT_RELEVANCE_CONFIG,
) -> int:
    """Assign a proxy relevance label (0, 1, or 2) to a single scholarship row.

    Label 2 (highly relevant): keyword overlap AND strict profile match. By
        default the strict match requires an explicit major match (see
        ``RelevanceConfig.require_major_match_for_label2``), so scholarships open
        to all majors do not earn label 2 on keyword overlap alone.
    Label 1 (relevant): keyword overlap OR text similarity above threshold.
    Label 0 (not relevant): neither condition met.

    Args:
        row: A scored scholarship row from the Stage 2 output DataFrame.
        student: The golden student being evaluated.
        similarity_mode: Determines which threshold to apply (``"tfidf"`` or
            ``"embeddings"``).
        cfg: Labelling configuration.

    Returns:
        Integer label in {0, 1, 2}.
    """
    keyword_overlap_positive = _keyword_overlap_positive(row)
    strict_match = _strict_profile_match(row, student, cfg)

    if strict_match and keyword_overlap_positive:
        return 2

    if cfg.label_mode == "no_similarity":
        return 1 if keyword_overlap_positive else 0

    if keyword_overlap_positive or _text_similarity(row) >= get_similarity_threshold(similarity_mode, cfg):
        return 1

    return 0


def proxy_relevance_labels(
    frame: pd.DataFrame,
    student: GoldenStudent,
    *,
    similarity_mode: str,
    cfg: RelevanceConfig = DEFAULT_RELEVANCE_CONFIG,
) -> list[int]:
    """Return a list of proxy relevance labels for every row in ``frame``.

    Wraps :func:`proxy_relevance_label` for bulk application to a scored
    scholarship DataFrame.
    """
    return [
        proxy_relevance_label(row, student, similarity_mode=similarity_mode, cfg=cfg)
        for _, row in frame.iterrows()
    ]


def calibrate_similarity_threshold(
    eligible_frames: list[pd.DataFrame],
    *,
    target_share: float = 0.25,
) -> float | None:
    """Estimate a similarity threshold that marks approximately ``target_share`` of rows as relevant.

    Computes the (1 − target_share) quantile of ``text_sim`` values from rows
    with zero keyword overlap, then returns the median across profiles.

    Args:
        eligible_frames: List of eligible-scholarship DataFrames, one per profile.
        target_share: Desired fraction of rows to label as relevant via similarity.

    Returns:
        Calibrated threshold float, or ``None`` if no usable frames are provided.
    """
    if not eligible_frames:
        return None

    profile_thresholds: list[float] = []
    quantile = max(0.0, min(1.0, 1.0 - target_share))

    for frame in eligible_frames:
        if frame.empty or "text_sim" not in frame.columns:
            continue
        if "keyword_overlap" in frame.columns:
            zero_keyword = frame.loc[~frame["keyword_overlap"].fillna(0.0).gt(0.0), "text_sim"].dropna()
        else:
            zero_keyword = frame["text_sim"].dropna()
        if zero_keyword.empty:
            continue
        profile_thresholds.append(float(zero_keyword.quantile(quantile, interpolation="linear")))

    if not profile_thresholds:
        return None

    return float(median(profile_thresholds))
