from __future__ import annotations

from typing import Any

import pandas as pd


def format_amount_range(amount_min: Any, amount_max: Any) -> str:
    min_value = _coerce_amount(amount_min)
    max_value = _coerce_amount(amount_max)
    if min_value is None and max_value is None:
        return "Unknown"
    if min_value is None:
        return f"Up to ${max_value:,.0f}"
    if max_value is None:
        return f"${min_value:,.0f}+"
    if abs(min_value - max_value) < 1e-9:
        return f"${min_value:,.0f}"
    return f"${min_value:,.0f} - ${max_value:,.0f}"


def explain_ranked_row(row: pd.Series, *, max_signals: int = 3) -> list[str]:
    text_similarity = (
        _coerce_float(row.get("text_sim"))
        or _coerce_float(row.get("tfidf_sim"))
        or _coerce_float(row.get("embed_sim"))
        or 0.0
    )
    signal_scores = [
        (
            float(text_similarity),
            "Strong match to your goals/keywords",
        ),
        (
            float(_coerce_float(row.get("amount_utility")) or 0.0),
            "High award amount",
        ),
        (
            float(_coerce_float(row.get("keyword_overlap")) or 0.0),
            "High direct keyword overlap",
        ),
        (
            float(_coerce_float(row.get("urgency_boost")) or 0.0),
            "Deadline soon, boosted for urgency",
        ),
        (
            float(_coerce_float(row.get("expected_value_norm")) or _coerce_float(row.get("ev_proxy_norm")) or 0.0),
            "Strong expected-value proxy",
        ),
    ]
    if not bool(row.get("essay_required")):
        signal_scores.append((0.4, "Lower effort (no essay)"))

    ranked = [label for score, label in sorted(signal_scores, key=lambda item: item[0], reverse=True) if score > 0]
    if not ranked:
        return ["Balanced profile fit after scoring"]
    return ranked[:max_signals]


def reasons_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if str(item).strip())
    return str(value)


def _coerce_amount(value: Any) -> float | None:
    coerced = _coerce_float(value)
    if coerced is None:
        return None
    return max(coerced, 0.0)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric
