from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app import modes
from app.helpers import csv_list, csv_text
from scripts.run_ingest import get_latest_snapshot_path
from src.profile.grade_levels import (
    grade_label_to_levels,
    infer_graduation_year,
    levels_to_grade_label,
)
from src.profile.store import (
    DEFAULT_STUDENT_ID,
    default_profile,
    load_profile_or_demo,
    profile_path,
    to_stage1_profile,
)
from src.rank.stage1_eligibility import StudentProfile
from src.rank.weights import Stage2Weights, Stage3Weights
from src.store import repo
from src.store.db import open_db

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROFILE_PATH = profile_path(DEFAULT_STUDENT_ID)
BEST_WEIGHTS_LATEST_PATH = PROCESSED_DIR / "best_weights_latest.json"
WEIGHTS_PROFILE_PATHS = {
    "Latest": BEST_WEIGHTS_LATEST_PATH,
    "Relevance": PROCESSED_DIR / "best_weights_relevance.json",
    "Blended": PROCESSED_DIR / "best_weights_blended.json",
    "Pareto": PROCESSED_DIR / "best_weights_pareto.json",
}

WEIGHTS_PROFILE_OPTIONS = tuple(WEIGHTS_PROFILE_PATHS.keys()) + ("Custom file path",)
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
PREFER_NOT_TO_SAY = "Prefer not to say"
TRISTATE_OPTIONS = (PREFER_NOT_TO_SAY, "Yes", "No")
GENDER_OPTIONS = (PREFER_NOT_TO_SAY, "Female", "Male", "Nonbinary")
_LIST_WIDGET_FIELDS = (
    "heritage",
    "parent_employers",
    "memberships",
    "intended_colleges",
    "extracurriculars",
)


def _tristate_label(value: bool | None) -> str:
    if value is None:
        return PREFER_NOT_TO_SAY
    return "Yes" if value else "No"


def _tristate_value(label: str) -> bool | None:
    if label == "Yes":
        return True
    if label == "No":
        return False
    return None


def _gender_label(value: str | None) -> str:
    if not value:
        return PREFER_NOT_TO_SAY
    return value.capitalize() if value.capitalize() in GENDER_OPTIONS else PREFER_NOT_TO_SAY


def _gender_value(label: str) -> str | None:
    return None if label == PREFER_NOT_TO_SAY else label.casefold()


def _positive_or_none(value: Any) -> int | None:
    number = int(value or 0)
    return number if number > 0 else None


def ensure_session_state() -> None:
    if "profile" not in st.session_state:
        profile, is_demo = load_profile_or_demo()
        if is_demo:
            # Demo data seeds the default slot so Save writes where Load reads.
            profile["student_id"] = DEFAULT_STUDENT_ID
        st.session_state.profile = profile
        st.session_state.profile_is_demo = is_demo
    st.session_state.setdefault("profile_is_demo", False)
    if "latest_snapshot_path" not in st.session_state:
        st.session_state.latest_snapshot_path = None
    if "latest_delta_summary" not in st.session_state:
        st.session_state.latest_delta_summary = None
    if "ingest_report" not in st.session_state:
        st.session_state.ingest_report = None
    if "eligible_df" not in st.session_state:
        st.session_state.eligible_df = None
    if "scored_df" not in st.session_state:
        st.session_state.scored_df = None
    if "final_df" not in st.session_state:
        st.session_state.final_df = None
    if "ineligible_df" not in st.session_state:
        st.session_state.ineligible_df = None
    if "use_win_model" not in st.session_state:
        st.session_state.use_win_model = False
    if "include_unconfirmed" not in st.session_state:
        st.session_state.include_unconfirmed = False
    st.session_state.setdefault("similarity_mode", "tfidf")
    st.session_state.setdefault("embedding_model_name", DEFAULT_MODEL_NAME)
    st.session_state.setdefault("weights_profile", "Latest")
    st.session_state.setdefault("custom_weights_path", "")
    _sync_widget_defaults_from_profile(st.session_state.profile)


def _widget_values_from_profile(profile: dict[str, Any]) -> dict[str, Any]:
    iso_override = str(profile.get("today_override") or date.today().isoformat())
    values: dict[str, Any] = {
        "profile_name": str(profile.get("name") or ""),
        "profile_gpa": float(profile.get("gpa") or 0.0),
        "profile_state": str(profile.get("state") or ""),
        "profile_county": str(profile.get("county") or ""),
        "profile_high_school": str(profile.get("high_school") or ""),
        "profile_major": str(profile.get("major") or ""),
        "profile_grade_label": levels_to_grade_label(
            profile.get("education_level"), profile.get("grade_level")
        ),
        "profile_graduation_year": int(profile.get("graduation_year") or 0),
        "profile_citizenship": str(profile.get("citizenship") or ""),
        "profile_gender": _gender_label(profile.get("gender")),
        "profile_religion": str(profile.get("religion") or ""),
        "profile_service_hours": int(profile.get("service_hours") or 0),
        "profile_sat": int(profile.get("sat") or 0),
        "profile_act": int(profile.get("act") or 0),
        "profile_essay_ready": bool(profile.get("essay_ready", False)),
        "profile_keywords_csv": csv_text(profile.get("profile_keywords")),
        "profile_goals": str(profile.get("goals") or ""),
        "profile_today_override": date.fromisoformat(iso_override),
        "profile_use_today_override": bool(profile.get("use_today_override", False)),
    }
    for field_name in ("financial_need", "first_gen", "military_family", "disability"):
        values[f"profile_{field_name}"] = _tristate_label(profile.get(field_name))
    for field_name in _LIST_WIDGET_FIELDS:
        values[f"profile_{field_name}_csv"] = csv_text(profile.get(field_name))
    return values


def _sync_widget_defaults_from_profile(profile: dict[str, Any]) -> None:
    for key, value in _widget_values_from_profile(profile).items():
        st.session_state.setdefault(key, value)


def _resolve_weights_profile_path(profile_name: str, custom_path: str = "") -> Path | None:
    if profile_name == "Custom file path":
        raw_path = custom_path.strip()
        if not raw_path:
            return None
        candidate = Path(raw_path)
        return candidate if candidate.is_absolute() else ROOT_DIR / candidate

    if profile_name == "Latest":
        if not BEST_WEIGHTS_LATEST_PATH.exists():
            return None
        pointer_payload = json.loads(BEST_WEIGHTS_LATEST_PATH.read_text(encoding="utf-8"))
        raw_path = str(pointer_payload.get("path") or "").strip()
        if not raw_path:
            return None
        candidate = Path(raw_path)
        return candidate if candidate.is_absolute() else ROOT_DIR / candidate

    return WEIGHTS_PROFILE_PATHS.get(profile_name)


def load_weights_profile(profile_name: str, custom_path: str = "") -> dict[str, Any] | None:
    weights_path = _resolve_weights_profile_path(profile_name, custom_path)
    if weights_path is None or not weights_path.exists():
        return None

    payload = json.loads(weights_path.read_text(encoding="utf-8"))
    stage2_weights = Stage2Weights.from_mapping(payload.get("stage2_weights"))
    stage3_weights = Stage3Weights.from_mapping(payload.get("stage3_weights"))
    amount_utility_mode = str(payload.get("amount_utility_mode") or "log")
    if amount_utility_mode not in {"linear", "log"}:
        raise ValueError(f"{weights_path.name} has an invalid amount_utility_mode.")

    return {
        "stage2_weights": stage2_weights,
        "stage3_weights": stage3_weights,
        "amount_utility_mode": amount_utility_mode,
        "snapshot_used": str(payload.get("snapshot_used") or ""),
        "timestamp": str(payload.get("timestamp") or ""),
        "use_win_model": bool(payload.get("use_win_model", False)),
        "objective": str(payload.get("objective") or ""),
        "config_id": str(payload.get("config_id") or ""),
        "source_path": str(weights_path),
        "source_name": weights_path.name,
        "selected_profile": profile_name,
    }


def apply_profile_to_widgets(profile: dict[str, Any]) -> None:
    for key, value in _widget_values_from_profile(profile).items():
        st.session_state[key] = value


def profile_from_widgets() -> dict[str, Any]:
    today_override = st.session_state.get("profile_today_override", date.today())
    grade_label = str(st.session_state.get("profile_grade_label") or "")
    education_level, grade_level = grade_label_to_levels(grade_label)
    graduation_year = _positive_or_none(st.session_state.get("profile_graduation_year"))

    profile = default_profile()
    profile.update(
        {
            "student_id": str(
                st.session_state.get("profile", {}).get("student_id") or DEFAULT_STUDENT_ID
            ),
            "name": str(st.session_state.get("profile_name") or ""),
            "gpa": float(st.session_state.get("profile_gpa") or 0.0),
            "state": str(st.session_state.get("profile_state") or ""),
            "county": str(st.session_state.get("profile_county") or ""),
            "high_school": str(st.session_state.get("profile_high_school") or ""),
            "major": str(st.session_state.get("profile_major") or ""),
            "education_level": education_level or "",
            "grade_level": grade_level or "",
            "graduation_year": graduation_year
            or infer_graduation_year(grade_level, _widget_today()),
            "citizenship": str(st.session_state.get("profile_citizenship") or ""),
            "gender": _gender_value(str(st.session_state.get("profile_gender") or "")),
            "religion": str(st.session_state.get("profile_religion") or "").strip() or None,
            "service_hours": _positive_or_none(st.session_state.get("profile_service_hours")),
            "sat": _positive_or_none(st.session_state.get("profile_sat")),
            "act": _positive_or_none(st.session_state.get("profile_act")),
            "essay_ready": bool(st.session_state.get("profile_essay_ready", False)),
            "profile_keywords": csv_list(st.session_state.get("profile_keywords_csv")),
            "goals": str(st.session_state.get("profile_goals") or ""),
            "today_override": today_override.isoformat(),
            "use_today_override": bool(
                st.session_state.get("profile_use_today_override", False)
            ),
        }
    )
    for field_name in ("financial_need", "first_gen", "military_family", "disability"):
        profile[field_name] = _tristate_value(
            str(st.session_state.get(f"profile_{field_name}") or PREFER_NOT_TO_SAY)
        )
    for field_name in _LIST_WIDGET_FIELDS:
        profile[field_name] = csv_list(st.session_state.get(f"profile_{field_name}_csv"))
    return profile


def _widget_today() -> date:
    if st.session_state.get("profile_use_today_override", False):
        return st.session_state.get("profile_today_override", date.today())
    return date.today()


def effective_today(profile: dict[str, Any]) -> date:
    if profile.get("use_today_override"):
        return date.fromisoformat(str(profile.get("today_override")))
    return date.today()


def active_snapshot_path() -> str | None:
    """Return the snapshot the app should read, remembering it for later reruns."""
    snapshot_path_text = st.session_state.latest_snapshot_path
    if snapshot_path_text is None:
        try:
            latest = get_latest_snapshot_path()
        except FileNotFoundError:
            latest = None
        if latest is not None:
            snapshot_path_text = str(latest.resolve())
            st.session_state.latest_snapshot_path = snapshot_path_text
    if snapshot_path_text is None or not Path(snapshot_path_text).exists():
        return None
    return str(snapshot_path_text)


def build_stage1_profile(profile: dict[str, Any]) -> StudentProfile:
    return to_stage1_profile(profile, today=effective_today(profile))


@st.cache_data(show_spinner=False)
def load_snapshot_cached(path_text: str) -> pd.DataFrame:
    return pd.read_parquet(Path(path_text))


def current_student_id() -> str:
    return str(st.session_state.profile.get("student_id") or DEFAULT_STUDENT_ID)


def ensure_student(conn: Any) -> str:
    """Make sure the profile has a row to hang applications off, and return it."""
    student_id = current_student_id()
    if repo.get_student(conn, student_id) is None:
        repo.upsert_student(conn, student_id, str(st.session_state.profile.get("name") or ""))
    return student_id


def current_mode() -> modes.Mode:
    return modes.normalize_mode(st.session_state.get(modes.MODE_STATE_KEY))


def operator_enabled() -> bool:
    try:
        with open_db() as conn:
            return repo.get_flag(conn, modes.OPERATOR_ENABLED_SETTING)
    except Exception:
        return False
