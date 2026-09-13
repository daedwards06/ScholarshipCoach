"""Decision-aware reranking with urgency and expected value signals.

Stage 3 adjusts Stage 2 scores by adding an urgency boost (exponential decay
over days to deadline) and an expected-value signal (optionally from the win
probability model) before producing the final ranked order.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from src.types import ProfileLike

import numpy as np
import pandas as pd

from src.rank.weights import Stage3Weights
from src.text_utils import coerce_text
from src.win_model.features import FEATURE_COLUMNS, build_pair_features
from src.win_model.infer import load_latest_model, load_model, predict_p_win

# Marginal effort cost per requirement, on top of a base cost of 1.0. One essay
# costs 0.5 so a row carrying only ``essay_required`` keeps its historic 1.5.
ESSAY_EFFORT = 0.5
LETTER_EFFORT = 0.25
EXTRA_EFFORT = 0.25


def _resolve_deadline(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None or pd.isna(value):
        return pd.NaT
    return pd.Timestamp(value)


def _effective_deadline(row: pd.Series, today: date) -> pd.Timestamp | pd.NaT:
    """Return the date the student would actually apply against.

    A recurring award reaches Stage 3 with its past cycle date still in
    ``deadline``; the timeline supplies the next cycle in ``projected_deadline``,
    and urgency has to be measured against that instead.
    """
    deadline = _resolve_deadline(row.get("deadline"))
    if not pd.isna(deadline) and deadline.date() >= today:
        return deadline
    projected = _resolve_deadline(row.get("projected_deadline"))
    if not pd.isna(projected):
        return projected
    return deadline


def _compute_days_to_deadline(df: pd.DataFrame, today: date) -> np.ndarray:
    values: list[float] = []
    for _, row in df.iterrows():
        deadline = _effective_deadline(row, today)
        if pd.isna(deadline):
            values.append(np.nan)
            continue
        delta_days = (deadline.date() - today).days
        values.append(float(delta_days))
    return np.array(values, dtype=float)


def _compute_urgency_boost(days_to_deadline: np.ndarray) -> np.ndarray:
    urgency = np.zeros(days_to_deadline.shape[0], dtype=float)
    # A deadline already past is not urgent, it is gone: clamping negative days
    # to zero would score it exp(0) = 1.0, the maximum boost. Recurring awards
    # reach Stage 3 with a past deadline because Stage 1 keeps them for the
    # timeline to bucket, so the past case has to be excluded explicitly.
    upcoming_mask = ~np.isnan(days_to_deadline) & (days_to_deadline >= 0.0)
    if not np.any(upcoming_mask):
        return urgency

    urgency_values = np.exp(-days_to_deadline[upcoming_mask] / 30.0)
    urgency[upcoming_mask] = np.clip(urgency_values, 0.0, 1.0)
    return urgency


def _requirements_mapping(row: pd.Series) -> Mapping[str, Any] | None:
    raw = row.get("requirements")
    if isinstance(raw, Mapping) and raw:
        return raw
    return None


def _count_list(value: Any) -> int:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return 0
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return 0
    return sum(1 for item in value if str(item).strip())


def _count_int(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0
    return max(count, 0)


def effort_counts(row: pd.Series) -> dict[str, int]:
    """Return the ``essays``/``letters``/``extras`` an award asks for.

    Counts come from the curated ``requirements`` object when it is present;
    a scraped row without one falls back to the ``essay_required`` boolean, so
    the numbers stay honest instead of implying a count nobody published.
    """
    requirements = _requirements_mapping(row)
    if requirements is None:
        essays = 1 if bool(row.get("essay_required")) else 0
        return {"essays": essays, "letters": 0, "extras": 0}

    essays = _count_list(requirements.get("essay_prompts"))
    if not essays and requirements.get("essay") is True:
        essays = 1
    letters = _count_int(requirements.get("recommendation_letters"))
    extras = sum(
        1
        for key in ("transcript", "fafsa", "video_or_portfolio", "interview")
        if requirements.get(key) is True
    )
    return {"essays": essays, "letters": letters, "extras": extras}


def _compute_effort_cost(df: pd.DataFrame) -> np.ndarray:
    costs: list[float] = []
    for _, row in df.iterrows():
        counts = effort_counts(row)
        costs.append(
            1.0
            + (ESSAY_EFFORT * counts["essays"])
            + (LETTER_EFFORT * counts["letters"])
            + (EXTRA_EFFORT * counts["extras"])
        )
    return np.array(costs, dtype=float)


def is_local_award(row: pd.Series) -> bool:
    """Whether this award draws from a small applicant pool.

    Either a person confirmed it locally, or the sponsor restricts it to named
    counties or states.  A national award lists no states at all, so an empty
    list is not a restriction.
    """
    if coerce_text(row.get("trust")) == "verified_local":
        return True
    return bool(_count_list(row.get("counties_allowed")) or _count_list(row.get("states_allowed")))


def _compute_local_award(df: pd.DataFrame) -> np.ndarray:
    return np.array([is_local_award(row) for _, row in df.iterrows()], dtype=bool)


def _resolve_amount_for_ev(row: pd.Series) -> float:
    amount_max = row.get("amount_max")
    if amount_max is not None and not pd.isna(amount_max):
        return float(amount_max)
    amount_min = row.get("amount_min")
    if amount_min is not None and not pd.isna(amount_min):
        return float(amount_min)
    return 0.0


def _normalize_minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=float)

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    span = maximum - minimum
    if span <= 0.0:
        return np.zeros(values.shape[0], dtype=float)
    return np.clip((values - minimum) / span, 0.0, 1.0)


def rerank_stage3(
    scored_df: pd.DataFrame,
    today: date | None = None,
    *,
    profile: ProfileLike | dict[str, Any] | None = None,
    timeline_bucket: str | None = "now",
    weights: Stage3Weights | None = None,
    use_win_model: bool = False,
    win_model_path: Path | None = None,
    win_model: object | None = None,
) -> pd.DataFrame:
    """Rerank Stage 2 results using urgency, expected value, and optional win-probability signals.

    Adds ``days_to_deadline``, ``urgency_boost``, ``effort_cost``,
    ``local_award``, ``ev_proxy``, ``ev_proxy_norm``, ``final_score``, and
    (when win model is active) ``p_win``, ``expected_value``, and
    ``expected_value_norm`` columns.

    Args:
        scored_df: DataFrame with a ``stage2_score`` column from Stage 2.
        today: Reference date for urgency computation; defaults to ``date.today()``.
        profile: Student profile required when ``use_win_model=True``.
        timeline_bucket: Bucket to rank, applied only when the frame carries a
            ``timeline_bucket`` column; ``None`` ranks every bucket.
        weights: Stage 3 scoring weights; uses ``Stage3Weights.baseline()`` if ``None``.
        use_win_model: Whether to load and run the win probability model.
        win_model_path: Explicit path to a ``.joblib`` model artifact.
        win_model: Pre-loaded model object (skips disk load when provided).

    Returns:
        Copy of ``scored_df`` with reranking columns added and rows sorted by
        ``final_score`` descending (ties broken by deadline then scholarship ID).

    Raises:
        ValueError: If ``stage2_score`` column is missing or ``profile`` is
            absent when ``use_win_model=True``.
    """
    effective_today = today or date.today()
    reranked_df = scored_df.copy()
    active_weights = weights or Stage3Weights.baseline()

    if "stage2_score" not in reranked_df.columns:
        raise ValueError("Stage 3 rerank requires a 'stage2_score' column.")

    if timeline_bucket is not None and "timeline_bucket" in reranked_df.columns:
        reranked_df = reranked_df[reranked_df["timeline_bucket"].eq(timeline_bucket)].copy()

    days_to_deadline = _compute_days_to_deadline(reranked_df, effective_today)
    urgency_boost = _compute_urgency_boost(days_to_deadline)
    effort_cost = _compute_effort_cost(reranked_df)
    local_award = _compute_local_award(reranked_df)
    amounts = np.array([_resolve_amount_for_ev(row) for _, row in reranked_df.iterrows()], dtype=float)
    ev_proxy = amounts / np.clip(effort_cost, 1e-9, None)
    ev_proxy_norm = _normalize_minmax(ev_proxy)

    ev_signal = ev_proxy_norm
    p_win: np.ndarray | None = None
    expected_value: np.ndarray | None = None
    expected_value_norm: np.ndarray | None = None
    if use_win_model and not reranked_df.empty:
        if profile is None:
            raise ValueError("Stage 3 win-model rerank requires a profile object.")
        active_model = win_model
        if active_model is None:
            active_model = load_model(win_model_path) if win_model_path is not None else load_latest_model()
        feature_rows = [
            build_pair_features(
                profile,
                row,
                stage2_row=row,
                today=effective_today,
            )
            for _, row in reranked_df.iterrows()
        ]
        features_df = pd.DataFrame(feature_rows, columns=list(FEATURE_COLUMNS))
        p_win = predict_p_win(active_model, features_df)
        expected_value = p_win * amounts
        expected_value_norm = _normalize_minmax(expected_value)
        ev_signal = expected_value_norm

    stage2_score = pd.to_numeric(reranked_df["stage2_score"], errors="coerce").fillna(0.0).to_numpy()
    final_score = (
        (active_weights.stage2 * stage2_score)
        + (active_weights.urgency * urgency_boost)
        + (active_weights.ev * ev_signal)
        + (active_weights.local_boost * local_award.astype(float))
    )

    reranked_df["days_to_deadline"] = days_to_deadline
    reranked_df["urgency_boost"] = urgency_boost
    reranked_df["effort_cost"] = effort_cost
    reranked_df["local_award"] = local_award
    reranked_df["ev_proxy"] = ev_proxy
    reranked_df["ev_proxy_norm"] = ev_proxy_norm
    if use_win_model:
        reranked_df["p_win"] = (
            p_win if p_win is not None else np.zeros(reranked_df.shape[0], dtype=float)
        )
        reranked_df["expected_value"] = (
            expected_value if expected_value is not None else np.zeros(reranked_df.shape[0], dtype=float)
        )
        reranked_df["expected_value_norm"] = (
            expected_value_norm
            if expected_value_norm is not None
            else np.zeros(reranked_df.shape[0], dtype=float)
        )
    reranked_df["final_score"] = final_score

    reranked_df["_deadline_sort"] = pd.to_datetime(reranked_df.get("deadline"), errors="coerce")
    reranked_df = reranked_df.sort_values(
        by=["final_score", "_deadline_sort", "scholarship_id"],
        ascending=[False, True, True],
        na_position="last",
        kind="mergesort",
    ).drop(columns=["_deadline_sort"])

    return reranked_df.reset_index(drop=True)
