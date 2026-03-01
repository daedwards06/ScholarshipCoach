from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from src.eval.golden_students import GoldenStudent


@dataclass(frozen=True, slots=True)
class RelevanceConfig:
    label_mode: Literal["hybrid", "no_similarity"] = "hybrid"
    tfidf_threshold: float = 0.12
    embed_threshold: float = 0.30
    strict_requires_all_of: tuple[str, ...] = ("major", "state", "education_level")


DEFAULT_RELEVANCE_CONFIG = RelevanceConfig()


def _normalize_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _normalize_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            normalized = _normalize_text(item)
            if normalized:
                result.append(normalized)
        return result
    normalized = _normalize_text(value)
    return [normalized] if normalized else []


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
    checks: dict[str, bool] = {
        "major": (not majors_allowed) or (_normalize_text(profile.major) in majors_allowed),
        "state": (not states_allowed) or (_normalize_text(profile.state) in states_allowed),
        "education_level": (scholarship_edu is None)
        or (scholarship_edu == _normalize_text(profile.education_level)),
    }
    return all(checks.get(field, False) for field in cfg.strict_requires_all_of)


def get_similarity_threshold(similarity_mode: str, cfg: RelevanceConfig) -> float:
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
    return [
        proxy_relevance_label(row, student, similarity_mode=similarity_mode, cfg=cfg)
        for _, row in frame.iterrows()
    ]


def calibrate_similarity_threshold(
    eligible_frames: list[pd.DataFrame],
    *,
    target_share: float = 0.25,
) -> float | None:
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
