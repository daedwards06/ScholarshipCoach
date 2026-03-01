from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.rank.weights import Stage3Weights
from src.win_model.features import FEATURE_COLUMNS, build_pair_features
from src.win_model.infer import load_latest_model, load_model, predict_p_win


def _resolve_deadline(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None or pd.isna(value):
        return pd.NaT
    return pd.Timestamp(value)


def _compute_days_to_deadline(df: pd.DataFrame, today: date) -> np.ndarray:
    values: list[float] = []
    for _, row in df.iterrows():
        deadline = _resolve_deadline(row.get("deadline"))
        if pd.isna(deadline):
            values.append(np.nan)
            continue
        delta_days = (deadline.date() - today).days
        values.append(float(delta_days))
    return np.array(values, dtype=float)


def _compute_urgency_boost(days_to_deadline: np.ndarray) -> np.ndarray:
    urgency = np.zeros(days_to_deadline.shape[0], dtype=float)
    valid_mask = ~np.isnan(days_to_deadline)
    if not np.any(valid_mask):
        return urgency

    bounded_days = np.maximum(days_to_deadline[valid_mask], 0.0)
    urgency_values = np.exp(-bounded_days / 30.0)
    urgency[valid_mask] = np.clip(urgency_values, 0.0, 1.0)
    return urgency


def _compute_effort_cost(df: pd.DataFrame) -> np.ndarray:
    costs: list[float] = []
    for _, row in df.iterrows():
        if bool(row.get("essay_required")):
            costs.append(1.5)
        else:
            costs.append(1.0)
    return np.array(costs, dtype=float)


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
    profile: Any | None = None,
    weights: Stage3Weights | None = None,
    use_win_model: bool = False,
    win_model_path: Path | None = None,
    win_model: object | None = None,
) -> pd.DataFrame:
    effective_today = today or date.today()
    reranked_df = scored_df.copy()
    active_weights = weights or Stage3Weights.baseline()

    if "stage2_score" not in reranked_df.columns:
        raise ValueError("Stage 3 rerank requires a 'stage2_score' column.")

    days_to_deadline = _compute_days_to_deadline(reranked_df, effective_today)
    urgency_boost = _compute_urgency_boost(days_to_deadline)
    effort_cost = _compute_effort_cost(reranked_df)
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
    )

    reranked_df["days_to_deadline"] = days_to_deadline
    reranked_df["urgency_boost"] = urgency_boost
    reranked_df["effort_cost"] = effort_cost
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
