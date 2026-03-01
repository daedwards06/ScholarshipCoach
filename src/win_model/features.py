from __future__ import annotations

import math
from datetime import date
from typing import Any

import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "gpa",
    "min_gpa",
    "gpa_above_min",
    "keyword_overlap",
    "text_sim",
    "days_to_deadline",
    "amount_value",
    "amount_log",
    "major_match",
    "state_match",
    "education_level_match",
    "essay_required",
    "source_is_scholarship_america",
)


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        return text or None
    text = str(value).strip().lower()
    return text or None


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = _normalize_text(value)
        return [normalized] if normalized else []
    if isinstance(value, (list, tuple, set)):
        values = [_normalize_text(item) for item in value]
        return [item for item in values if item]
    return []


def _profile_value(profile: Any, field_name: str) -> Any:
    if isinstance(profile, dict):
        return profile.get(field_name)
    return getattr(profile, field_name, None)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _resolve_amount_value(row: pd.Series) -> float:
    amount_max = _coerce_float(row.get("amount_max"), default=math.nan)
    if math.isfinite(amount_max):
        return max(amount_max, 0.0)
    amount_min = _coerce_float(row.get("amount_min"), default=math.nan)
    if math.isfinite(amount_min):
        return max(amount_min, 0.0)
    return 0.0


def build_pair_features(
    profile: Any,
    scholarship_row: pd.Series | dict[str, Any],
    stage2_row: pd.Series | dict[str, Any] | None = None,
    today: date | None = None,
) -> dict[str, float]:
    row = scholarship_row if isinstance(scholarship_row, pd.Series) else pd.Series(scholarship_row)
    stage2 = stage2_row if isinstance(stage2_row, pd.Series) else pd.Series(stage2_row or {})

    effective_today = today or _profile_value(profile, "today") or date.today()
    gpa = min(max(_coerce_float(_profile_value(profile, "gpa")), 0.0), 4.0)
    min_gpa = max(_coerce_float(row.get("min_gpa")), 0.0)
    gpa_above_min = max(gpa - min_gpa, 0.0)

    deadline = pd.to_datetime(row.get("deadline"), errors="coerce")
    if pd.isna(deadline):
        days_to_deadline = 365.0
    else:
        delta_days = float((deadline.date() - effective_today).days)
        days_to_deadline = float(min(max(delta_days, 0.0), 365.0))

    amount_value = _resolve_amount_value(row)
    amount_log = math.log1p(max(amount_value, 0.0))

    profile_major = _normalize_text(_profile_value(profile, "major"))
    profile_state = _normalize_text(_profile_value(profile, "state"))
    profile_level = _normalize_text(_profile_value(profile, "education_level"))

    majors_allowed = _normalize_list(row.get("majors_allowed"))
    states_allowed = _normalize_list(row.get("states_allowed"))
    scholarship_level = _normalize_text(row.get("education_level"))
    source_text = _normalize_text(row.get("source")) or ""

    features = {
        "gpa": gpa,
        "min_gpa": min_gpa,
        "gpa_above_min": gpa_above_min,
        "keyword_overlap": max(_coerce_float(stage2.get("keyword_overlap")), 0.0),
        "text_sim": max(_coerce_float(stage2.get("text_sim")), 0.0),
        "days_to_deadline": days_to_deadline,
        "amount_value": amount_value,
        "amount_log": amount_log,
        "major_match": 1.0 if (not majors_allowed or profile_major in majors_allowed) else 0.0,
        "state_match": 1.0 if (not states_allowed or profile_state in states_allowed) else 0.0,
        "education_level_match": 1.0
        if (not scholarship_level or scholarship_level == profile_level)
        else 0.0,
        "essay_required": 1.0 if bool(row.get("essay_required")) else 0.0,
        "source_is_scholarship_america": 1.0 if "scholarship_america" in source_text else 0.0,
    }
    return {name: float(features[name]) for name in FEATURE_COLUMNS}
