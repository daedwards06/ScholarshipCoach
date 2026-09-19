from __future__ import annotations

import calendar
import json
import re
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app import modes
from app.helpers import (
    explain_ranked_row,
    format_amount_range,
    phone_width_css,
    reasons_to_text,
    unverified_to_text,
)
from scripts.run_ingest import get_latest_snapshot_path, run_ingest
from src.catalog import inbox
from src.catalog import entry as catalog_entry
from src.embeddings.cache import ensure_embedding_store_for_df
from src.eval.golden_students import get_golden_students
from src.ingest.prefill import prefill_from_url
from src.ingest.sources.curated_catalog import CuratedCatalogSource
from src.normalize.catalog_schema import RECORDS_DIR
from src.profile.grade_levels import (
    GRADE_LABEL_TO_LEVELS,
    GRADE_LABELS,
    GRADE_SEQUENCE,
    grade_label_to_levels,
    infer_graduation_year,
    levels_to_grade_label,
)
from src.profile.store import (
    DEFAULT_STUDENT_ID,
    default_profile,
    load_profile,
    load_profile_or_demo,
    profile_path,
    save_profile,
    to_stage1_profile,
    to_stage2_profile,
)
from src.rank.stage1_eligibility import StudentProfile, apply_eligibility_filter
from src.rank.stage2_scoring import score_stage2
from src.rank.stage3_rerank import rerank_stage3
from src.rank.timeline import (
    TIMELINE_BUCKET_LABELS,
    TIMELINE_BUCKETS,
    classify_timeline,
)
from src.rank.weights import Stage2Weights, Stage3Weights
from src.rank.whatif import WHATIF_FIELD_LABELS, WhatIfAward, whatif_eligibility
from src.store import calendar_feed, essays, milestones, money, repo, tracker
from src.store.db import open_db
from src.text_utils import coerce_text
from src.win_model.infer import get_latest_model_path, load_model
from src.win_model.train import train_win_model

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
SNAPSHOT_DATE_RE = re.compile(r"scholarships_snapshot_(\d{8})\.parquet$")

# Sections whose surfaces are built by later Phase 3 tasks. Listing them now is
# deliberate: the nav shows the family the whole product, not just what runs.
_PENDING_SECTION_NOTES = {
    "outcomes": "Results per application: awarded amounts and renewal terms.",
}
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# The timeline shows catalog awards alongside the family's own dates. A whole
# eligible catalog would bury them, so only the nearest deadlines in each
# bucket are drawn.
TIMELINE_AWARDS_PER_BUCKET = 40
CALENDAR_FILE_NAME = "scholarship_coach.ics"

CURATED_SOURCE_NAME = CuratedCatalogSource.name
CATALOG_PAGE_KEY = "catalog_page"
CATALOG_PAGE_REQUEST_KEY = "catalog_page_request"
CATALOG_FORM_KEY = "catalog_form"
CATALOG_EDIT_KEY = "catalog_editing_proposal"
CATALOG_NOTICE_KEY = "catalog_notice"
CATALOG_PAGES: tuple[str, ...] = ("add", "inbox")
CATALOG_PAGE_LABELS = {"add": "Add an award", "inbox": "Inbox"}

# A catalog field is unknown until the page says otherwise, which is a different
# thing from the profile's "prefer not to say".
CATALOG_TRISTATE_OPTIONS = ("Not stated", "Yes", "No")

GRADE_LEVEL_LABELS = {
    level: label
    for label, (_, level) in GRADE_LABEL_TO_LEVELS.items()
    if level is not None
}

REQUIREMENT_LABELS = {
    "essay": "Essay",
    "transcript": "Transcript",
    "fafsa": "FAFSA",
    "video_or_portfolio": "Video or portfolio",
    "interview": "Interview",
}

TRUST_LABELS = {
    "verified_local": "Verified by us",
    "structured_feed": "Structured feed",
    "aggregator": "Aggregator",
    "unverified": "Unverified",
}

PROPOSAL_KIND_LABELS = {
    "prefill": "From a pasted URL",
    "feed": "From a feed",
    "reverify": "Re-verification found a change",
    "manual": "Entered by hand",
}

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


def _csv_text(values: Any) -> str:
    return ", ".join(str(value) for value in (values or []))


def _csv_list(raw: Any) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _positive_or_none(value: Any) -> int | None:
    number = int(value or 0)
    return number if number > 0 else None


def _ensure_session_state() -> None:
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
        "profile_keywords_csv": _csv_text(profile.get("profile_keywords")),
        "profile_goals": str(profile.get("goals") or ""),
        "profile_today_override": date.fromisoformat(iso_override),
        "profile_use_today_override": bool(profile.get("use_today_override", False)),
    }
    for field_name in ("financial_need", "first_gen", "military_family", "disability"):
        values[f"profile_{field_name}"] = _tristate_label(profile.get(field_name))
    for field_name in _LIST_WIDGET_FIELDS:
        values[f"profile_{field_name}_csv"] = _csv_text(profile.get(field_name))
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


def _load_weights_profile(profile_name: str, custom_path: str = "") -> dict[str, Any] | None:
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


def _load_latest_win_model_info() -> dict[str, Any] | None:
    try:
        model_path = get_latest_model_path()
        model = load_model(model_path)
    except FileNotFoundError:
        return None
    except Exception as exc:
        return {
            "path": "",
            "timestamp": "unavailable",
            "error": str(exc),
            "roc_auc": None,
            "brier_score": None,
            "log_loss": None,
        }

    summary = getattr(model, "training_summary", {}) or {}
    metrics = summary.get("metrics", {})
    return {
        "path": str(model_path),
        "timestamp": model_path.stem.replace("win_model_", ""),
        "roc_auc": metrics.get("roc_auc"),
        "brier_score": metrics.get("brier_score"),
        "log_loss": metrics.get("log_loss"),
    }


def _apply_profile_to_widgets(profile: dict[str, Any]) -> None:
    for key, value in _widget_values_from_profile(profile).items():
        st.session_state[key] = value


def _profile_from_widgets() -> dict[str, Any]:
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
            "profile_keywords": _csv_list(st.session_state.get("profile_keywords_csv")),
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
        profile[field_name] = _csv_list(st.session_state.get(f"profile_{field_name}_csv"))
    return profile


def _widget_today() -> date:
    if st.session_state.get("profile_use_today_override", False):
        return st.session_state.get("profile_today_override", date.today())
    return date.today()


def _effective_today(profile: dict[str, Any]) -> date:
    if profile.get("use_today_override"):
        return date.fromisoformat(str(profile.get("today_override")))
    return date.today()


def _active_snapshot_path() -> str | None:
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


def _build_stage1_profile(profile: dict[str, Any]) -> StudentProfile:
    return to_stage1_profile(profile, today=_effective_today(profile))


def _build_stage2_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return to_stage2_profile(profile)


def _extract_snapshot_date(snapshot_path: Path) -> str | None:
    match = SNAPSHOT_DATE_RE.match(snapshot_path.name)
    if not match:
        return None
    stamp = match.group(1)
    return f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}"


def _changes_path_for_snapshot(snapshot_path: Path) -> Path | None:
    snapshot_date = _extract_snapshot_date(snapshot_path)
    if snapshot_date is None:
        return None
    stamp = snapshot_date.replace("-", "")
    candidate = snapshot_path.parent / f"changes_{stamp}.json"
    if candidate.exists():
        return candidate
    return None


@st.cache_data(show_spinner=False)
def _load_snapshot_cached(path_text: str) -> pd.DataFrame:
    return pd.read_parquet(Path(path_text))


@st.cache_data(show_spinner=False)
def _load_delta_cached(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        return {"added": [], "removed": [], "changed": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_rank_filters(
    df: pd.DataFrame,
    *,
    search_term: str,
    deadline_within_days: int,
    min_amount: float,
    essay_required_only: bool,
    today_value: date,
) -> pd.DataFrame:
    filtered = df.copy()
    if search_term:
        pattern = re.escape(search_term.strip())
        searchable = (
            filtered["title"].fillna("")
            + " "
            + filtered["sponsor"].fillna("")
            + " "
            + filtered["description"].fillna("")
        )
        filtered = filtered[searchable.str.contains(pattern, case=False, regex=True)]

    if deadline_within_days > 0:
        deadline_ts = pd.to_datetime(filtered["deadline"], errors="coerce")
        day_diff = (deadline_ts - pd.Timestamp(today_value)).dt.days
        filtered = filtered[(day_diff >= 0) & (day_diff <= deadline_within_days)]

    if min_amount > 0:
        amount_series = pd.to_numeric(filtered["amount_max"], errors="coerce").fillna(
            pd.to_numeric(filtered["amount_min"], errors="coerce")
        )
        filtered = filtered[amount_series.fillna(0.0) >= min_amount]

    if essay_required_only:
        filtered = filtered[filtered["essay_required"].fillna(False).eq(False)]

    return filtered


def _source_health_rows(source_info: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the report's per-source health blocks, disabled connectors included."""
    rows: list[dict[str, Any]] = []
    for item in source_info.get("details", []):
        health = item.get("health") or {}
        rows.append(
            {
                "source": item.get("source"),
                "status": item.get("status"),
                "records_this_run": health.get("records_this_run", item.get("records", 0)),
                "records_prior_run": health.get("records_prior_run"),
                "zero_record_regression": bool(health.get("zero_record_regression")),
                "note": item.get("error"),
            }
        )
    for item in source_info.get("disabled", []):
        rows.append(
            {
                "source": item.get("source"),
                "status": "disabled",
                "records_this_run": None,
                "records_prior_run": None,
                "zero_record_regression": False,
                "note": item.get("note"),
            }
        )
    return rows


def _display_ingest_summary(report: dict[str, Any]) -> None:
    st.subheader("Ingest Report")
    source_info = report.get("sources", {})
    record_info = report.get("records", {})
    delta_counts = report.get("delta_counts", {})
    st.write(
        {
            "duration_seconds": report.get("duration_seconds"),
            "run_date": report.get("run_date"),
            "sources_succeeded": source_info.get("succeeded_count", 0),
            "sources_failed": source_info.get("failed_count", 0),
            "records_snapshot_total": record_info.get("snapshot_total", 0),
            "delta_added": delta_counts.get("added", 0),
            "delta_removed": delta_counts.get("removed", 0),
            "delta_changed": delta_counts.get("changed", 0),
        }
    )
    regressions = source_info.get("zero_record_regressions") or []
    if regressions:
        st.error(
            "Zero-record regression: "
            + ", ".join(str(name) for name in regressions)
            + " returned records on the prior run and none on this one."
        )

    health_rows = _source_health_rows(source_info)
    if health_rows:
        st.caption("Source health")
        st.dataframe(pd.DataFrame(health_rows), use_container_width=True)

    failed_details = [
        item for item in source_info.get("details", []) if item.get("status") != "succeeded"
    ]
    if failed_details:
        st.warning("Some sources failed during ingest.")
        st.dataframe(pd.DataFrame(failed_details), use_container_width=True)
    elif not regressions:
        st.success("All sources succeeded.")


def _weights_display_payload(
    stage2_weights: Stage2Weights,
    stage3_weights: Stage3Weights,
    amount_utility_mode: str,
) -> dict[str, Any]:
    return {
        "stage2_weights": stage2_weights.to_dict(),
        "stage3_weights": stage3_weights.to_dict(),
        "amount_utility_mode": amount_utility_mode,
    }


# Synthetic win-model outputs. A student reads "52% chance of winning" as a
# fact, so these stay behind operator mode.
_WIN_MODEL_COLUMNS = ("p_win", "expected_value", "expected_value_norm")


def _topk_win_model_summary(df: pd.DataFrame) -> dict[str, float] | None:
    if "p_win" not in df.columns or "expected_value" not in df.columns:
        return None
    p_win = pd.to_numeric(df["p_win"], errors="coerce").dropna()
    expected_value = pd.to_numeric(df["expected_value"], errors="coerce").dropna()
    if p_win.empty or expected_value.empty:
        return None
    return {
        "mean_p_win": float(p_win.mean()),
        "median_p_win": float(p_win.median()),
        "mean_expected_value": float(expected_value.mean()),
        "median_expected_value": float(expected_value.median()),
    }


def _calculate_days_until_deadline(deadline_str: str, today: date) -> int | None:
    try:
        deadline = pd.to_datetime(deadline_str)
        return (deadline.date() - today).days
    except (ValueError, TypeError, AttributeError):
        return None


def _timeline_deadline(row: pd.Series, today: date) -> tuple[str, bool]:
    """Return the deadline to display and whether it is a projected cycle date."""
    deadline = coerce_text(row.get("deadline"))
    days_until = _calculate_days_until_deadline(deadline, today)
    if deadline and days_until is not None and days_until >= 0:
        return deadline, False
    projected = row.get("projected_deadline")
    if projected is not None and not pd.isna(projected):
        return pd.Timestamp(projected).date().isoformat(), True
    return deadline, False


def _get_urgency_indicator(days_until_deadline: int | None) -> tuple[str, str]:
    if days_until_deadline is None:
        return ("⚠️ Unknown deadline", "#666666")
    if days_until_deadline < 0:
        return ("⏰ Passed", "#888888")
    if days_until_deadline <= 7:
        return ("🔴 URGENT (≤7 days)", "#FF4444")
    if days_until_deadline <= 30:
        return ("🟡 Soon (≤30 days)", "#FFAA00")
    return ("🟢 Later", "#00AA00")


def _student_id() -> str:
    return str(st.session_state.profile.get("student_id") or DEFAULT_STUDENT_ID)


def _ensure_student(conn: Any) -> str:
    """Make sure the profile has a row to hang applications off, and return it."""
    student_id = _student_id()
    if repo.get_student(conn, student_id) is None:
        repo.upsert_student(conn, student_id, str(st.session_state.profile.get("name") or ""))
    return student_id


def _row_catalog_id(row: pd.Series) -> str:
    """The stable id to track an award under, falling back to the snapshot id."""
    return coerce_text(row.get("catalog_id")) or coerce_text(row.get("scholarship_id"))


def _row_requirements(row: pd.Series) -> dict[str, Any]:
    """The award's ``requirements`` object, rebuilt for pre-catalog sources."""
    raw = row.get("requirements")
    if isinstance(raw, Mapping) and raw:
        return dict(raw)
    essay_required = row.get("essay_required")
    essay_prompt = coerce_text(row.get("essay_prompt"))
    return {
        "essay": None if essay_required is None or pd.isna(essay_required) else bool(essay_required),
        "essay_prompts": [essay_prompt] if essay_prompt else [],
    }


def _saved_catalog_ids(student_id: str) -> set[str]:
    """Catalog ids the student already tracks, so cards can say so."""
    try:
        with open_db() as conn:
            return {
                application.catalog_id
                for application in repo.list_applications(conn, student_id)
            }
    except Exception:
        return set()


def _save_award_from_card(row: pd.Series, today_value: date) -> None:
    catalog_id = _row_catalog_id(row)
    if not catalog_id:
        st.error("This award has no id to track it by.")
        return
    deadline, _ = _timeline_deadline(row, today_value)
    try:
        with open_db() as conn:
            student_id = _ensure_student(conn)
            _, created = tracker.save_award(
                conn,
                student_id,
                catalog_id,
                title=coerce_text(row.get("title")) or catalog_id,
                source_url=coerce_text(row.get("source_url")),
                deadline=deadline or None,
                requirements=_row_requirements(row),
            )
    except Exception as exc:
        st.error(f"Could not save this award: {exc}")
        return
    st.toast("Saved to My Applications." if created else "Already in My Applications.")
    st.rerun()


def _render_scholarship_card(
    row: pd.Series,
    today_value: date,
    saved_ids: set[str] | None = None,
    *,
    operator_mode: bool = False,
) -> None:
    title = (
        coerce_text(row.get("title")) or coerce_text(row.get("scholarship_id")) or "Untitled"
    )
    sponsor = coerce_text(row.get("sponsor"))
    deadline, deadline_is_projected = _timeline_deadline(row, today_value)
    bucket = coerce_text(row.get("timeline_bucket"))
    amount_str = format_amount_range(row.get("amount_min"), row.get("amount_max"))
    amount_not_published = pd.isna(row.get("amount_min")) and pd.isna(row.get("amount_max"))
    source_url = coerce_text(row.get("source_url"))

    days_until = _calculate_days_until_deadline(deadline, today_value)
    urgency_text, urgency_color = _get_urgency_indicator(days_until)

    with st.container(border=True):
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.markdown(f"**{title}**")
            if sponsor:
                st.caption(sponsor)
        with col2:
            st.markdown(f"<div style='text-align: right; font-size: 0.85em;'>{urgency_text}</div>", unsafe_allow_html=True)

        col_amt, col_deadline = st.columns(2)
        with col_amt:
            st.text(f"Award: {amount_str}")
            if amount_not_published:
                st.caption("Amount not published — ranked on fit alone")
        with col_deadline:
            if deadline and deadline_is_projected:
                st.text(f"Next deadline: ~{deadline}")
                st.caption("Projected from this award's usual cycle")
            else:
                st.text(f"Deadline: {deadline if deadline else 'Unknown'}")
        if bucket:
            st.caption(f"Timeline: {TIMELINE_BUCKET_LABELS.get(bucket, bucket)}")

        st.markdown("**Why this matches you:**")
        for explanation in explain_ranked_row(row, operator_mode=operator_mode):
            st.markdown(f"• {explanation}")

        confirm_text = unverified_to_text(row.get("unverified_axes"))
        if confirm_text:
            st.warning(f"Confirm you meet: {confirm_text}")

        catalog_id = _row_catalog_id(row)
        already_saved = catalog_id in (saved_ids or set())
        col_save, col_apply = st.columns([0.25, 0.75])
        with col_save:
            if already_saved:
                st.button(
                    "✓ Saved",
                    key=f"save_{catalog_id}",
                    disabled=True,
                    help="Already under My Applications.",
                )
            elif st.button("Save", key=f"save_{catalog_id}", type="secondary"):
                _save_award_from_card(row, today_value)
        with col_apply:
            if source_url:
                st.link_button("Apply at Source", source_url, use_container_width=False)

        with st.expander("Signal details", expanded=False):
            component_columns = [
                "text_sim",
                "tfidf_sim",
                "embed_sim",
                "amount_utility",
                "keyword_overlap",
                "effort_penalty",
                "urgency_boost",
                "ev_proxy_norm",
                "final_score",
            ]
            if operator_mode:
                component_columns.extend(_WIN_MODEL_COLUMNS)
            component_values = {
                column: float(row.get(column))
                for column in component_columns
                if column in row and pd.notna(row.get(column))
            }
            st.json(component_values)


def _render_profile_sidebar() -> None:
    st.header("Your Profile")

    if st.session_state.get("profile_is_demo", False):
        st.info(
            "Demo profile — fictional data. Edit the fields and press Save Profile to "
            "store your own under data/private/."
        )

    st.subheader("Academic", divider=False)
    st.text_input("Full Name", key="profile_name", placeholder="Your name (optional)")
    st.number_input("GPA", min_value=0.0, max_value=4.0, step=0.01, key="profile_gpa",
                   help="Your current GPA (0.0-4.0)")
    st.text_input("State", key="profile_state", placeholder="e.g., NC")
    st.text_input("County", key="profile_county",
                 placeholder="e.g., Guilford",
                 help="Many local awards are county-restricted")
    st.text_input("High School", key="profile_high_school",
                 placeholder="e.g., Northside High School")
    st.text_input("Intended Major", key="profile_major", placeholder="e.g., Computer Science")
    st.selectbox("Grade Level",
                options=list(GRADE_LABELS),
                key="profile_grade_label",
                help="Your current or upcoming grade level")
    st.number_input("Graduation Year", min_value=0, max_value=2100, step=1,
                   key="profile_graduation_year",
                   help="Leave at 0 to estimate it from your grade level")
    st.number_input("SAT", min_value=0, max_value=1600, step=10, key="profile_sat",
                   help="Leave at 0 if you have not taken it")
    st.number_input("ACT", min_value=0, max_value=36, step=1, key="profile_act",
                   help="Leave at 0 if you have not taken it")

    st.subheader("About you", divider=False)
    st.caption("Every field here is optional and only used to match restricted awards.")
    st.text_input("Citizenship", key="profile_citizenship", placeholder="e.g., U.S. Citizen")
    st.selectbox("Financial need", options=list(TRISTATE_OPTIONS),
                key="profile_financial_need",
                help="Whether you qualify for need-based aid")
    st.selectbox("First-generation college student", options=list(TRISTATE_OPTIONS),
                key="profile_first_gen")
    st.selectbox("Gender", options=list(GENDER_OPTIONS), key="profile_gender")
    st.text_input("Heritage / background", key="profile_heritage_csv",
                 placeholder="e.g., Hispanic, Cherokee (comma-separated)")
    st.selectbox("Military family", options=list(TRISTATE_OPTIONS),
                key="profile_military_family")
    st.selectbox("Disability", options=list(TRISTATE_OPTIONS), key="profile_disability")
    st.text_input("Religion", key="profile_religion", placeholder="Optional")
    st.text_input("Parent employers", key="profile_parent_employers_csv",
                 placeholder="e.g., Duke Energy (comma-separated)",
                 help="Some awards are restricted to employees' children")

    st.subheader("Activities", divider=False)
    st.text_input(
        "Interests / Keywords",
        key="profile_keywords_csv",
        placeholder="e.g., robotics, leadership, community service (comma-separated)",
        help="Topics you're passionate about—we'll match scholarships to these"
    )
    st.text_input("Extracurriculars", key="profile_extracurriculars_csv",
                 placeholder="e.g., robotics team, marching band (comma-separated)")
    st.text_input("Memberships", key="profile_memberships_csv",
                 placeholder="e.g., 4-H, National Honor Society (comma-separated)")
    st.number_input("Community service hours", min_value=0, max_value=10000, step=5,
                   key="profile_service_hours")
    st.checkbox("I have an essay draft ready", key="profile_essay_ready")
    st.text_area("Your Goals", key="profile_goals", height=120,
                placeholder="Tell us about your goals, dreams, or what matters to you",
                help="We use this to find scholarships that align with your aspirations")

    st.subheader("Colleges", divider=False)
    st.text_input("Intended colleges", key="profile_intended_colleges_csv",
                 placeholder="e.g., NC State, UNC Charlotte (comma-separated)")

    st.subheader("Advanced", divider=False)
    st.checkbox("Use custom date", key="profile_use_today_override", help="Override today's date for testing")
    st.date_input("Custom date", key="profile_today_override")

    save_col, load_col = st.columns(2)
    if save_col.button("Save Profile", use_container_width=True):
        profile = _profile_from_widgets()
        saved_path = save_profile(profile)
        st.session_state.profile = profile
        st.session_state.profile_is_demo = False
        st.success(f"Saved to {saved_path}")
    if load_col.button("Load Profile", use_container_width=True):
        loaded = load_profile()
        if loaded is None:
            st.warning(f"No profile file found at {PROFILE_PATH}")
        else:
            st.session_state.profile = loaded
            st.session_state.profile_is_demo = False
            _apply_profile_to_widgets(loaded)
            st.success("Profile loaded.")
            st.rerun()


def _render_operator_sidebar() -> None:
    tuned_weights_payload: dict[str, Any] | None = None
    tuned_weights_error: str | None = None

    with st.expander("Advanced / Operator", expanded=False):
        st.subheader("Similarity")
        similarity_label = st.selectbox(
            "Similarity mode",
            options=("TF-IDF", "Embeddings"),
            index=0 if st.session_state.get("similarity_mode", "tfidf") == "tfidf" else 1,
        )
        st.session_state.similarity_mode = "tfidf" if similarity_label == "TF-IDF" else "embeddings"
        st.session_state.embedding_model_name = st.selectbox(
            "Model",
            options=(DEFAULT_MODEL_NAME,),
            index=0,
        )

        st.subheader("Catalog Trust")
        st.checkbox("Include unconfirmed records", key="include_unconfirmed")
        st.caption(
            "Off: only confirmed records and trusted structured feeds are eligible "
            "(`TRUST_UNCONFIRMED`). On: unconfirmed and aggregator records rank too, "
            "for pipeline inspection only."
        )

        st.subheader("Ranking Weights")
        selected_weights_profile = st.selectbox(
            "Weights profile",
            options=WEIGHTS_PROFILE_OPTIONS,
            index=WEIGHTS_PROFILE_OPTIONS.index(st.session_state.get("weights_profile", "Latest")),
            key="weights_profile",
        )
        custom_weights_path = ""
        if selected_weights_profile == "Custom file path":
            custom_weights_path = st.text_input(
                "Custom weights JSON path",
                key="custom_weights_path",
                placeholder="data/processed/best_weights_relevance.json",
            )
        try:
            tuned_weights_payload = _load_weights_profile(selected_weights_profile, custom_weights_path)
        except Exception as exc:
            tuned_weights_error = str(exc)

        if tuned_weights_error:
            st.warning(f"Could not load selected weights profile: {tuned_weights_error}")
        elif tuned_weights_payload is None:
            st.caption("Selected weights profile is unavailable. Baseline weights will be used.")
        else:
            objective_label = tuned_weights_payload.get("objective") or "unspecified"
            st.caption(
                f"Loaded `{selected_weights_profile}` weights from {tuned_weights_payload['source_name']} "
                f"(objective: {objective_label})."
            )
            if tuned_weights_payload.get("use_win_model"):
                st.caption("This tuned weights file was generated with the win model enabled.")

        st.subheader("Win Probability Model")
        latest_win_model_info = _load_latest_win_model_info()
        if st.button("Train/Refresh Win Model", use_container_width=True):
            try:
                latest_snapshot = get_latest_snapshot_path()
                snapshot_df = pd.read_parquet(latest_snapshot)
                training_info = train_win_model(
                    snapshot_df,
                    get_golden_students(),
                    PROCESSED_DIR / "win_model",
                    seed=0,
                )
                latest_win_model_info = _load_latest_win_model_info()
                metrics = training_info["metrics"]
                st.success(
                    f"Win model trained. AUC={metrics['roc_auc']:.4f} Brier={metrics['brier_score']:.4f}"
                )
            except FileNotFoundError:
                st.warning("No saved snapshot found. Load or ingest a snapshot before training the win model.")
            except Exception as exc:
                st.error(f"Win model training failed: {exc}")
        st.checkbox("Use Win Model in Ranking", key="use_win_model")
        if latest_win_model_info is None:
            st.caption("No trained win model found yet.")
        else:
            st.caption(f"Latest model: {latest_win_model_info['timestamp']}")
            if latest_win_model_info.get("error"):
                st.warning(f"Could not load win model details: {latest_win_model_info['error']}")
            st.write(
                {
                    "roc_auc": latest_win_model_info.get("roc_auc"),
                    "brier_score": latest_win_model_info.get("brier_score"),
                    "log_loss": latest_win_model_info.get("log_loss"),
                }
            )

        st.subheader("Data Update")
        update_col, latest_col = st.columns(2)
        if update_col.button("Run Update (Ingest)", use_container_width=True):
            try:
                report = run_ingest(date=None)
                st.session_state.ingest_report = report
                st.session_state.latest_snapshot_path = report["artifact_paths"]["snapshot"]
                st.session_state.latest_delta_summary = report["delta_counts"]
                st.success("Ingest completed.")
            except Exception as exc:
                st.error(f"Ingest failed: {exc}")

        if latest_col.button("Use Latest Snapshot", use_container_width=True):
            try:
                latest = get_latest_snapshot_path()
            except FileNotFoundError:
                st.warning("No snapshot found. Click 'Run Update (Ingest)' first.")
                st.session_state.latest_snapshot_path = None
            else:
                st.session_state.latest_snapshot_path = str(latest.resolve())
                delta_path = _changes_path_for_snapshot(latest)
                if delta_path is not None:
                    delta_payload = _load_delta_cached(str(delta_path.resolve()))
                    st.session_state.latest_delta_summary = {
                        "added": len(delta_payload.get("added", [])),
                        "removed": len(delta_payload.get("removed", [])),
                        "changed": len(delta_payload.get("changed", [])),
                    }
                st.success(f"Loaded latest snapshot: {latest.name}")

        if st.session_state.ingest_report:
            _display_ingest_summary(st.session_state.ingest_report)
        if st.session_state.latest_delta_summary:
            st.subheader("Delta Summary")
            st.write(st.session_state.latest_delta_summary)


def _render_find_section() -> None:
    tuned_weights_payload = None
    try:
        tuned_weights_payload = _load_weights_profile(
            str(st.session_state.get("weights_profile") or "Latest"),
            str(st.session_state.get("custom_weights_path") or ""),
        )
    except Exception:
        tuned_weights_payload = None

    if tuned_weights_payload is not None:
        active_stage2_weights = tuned_weights_payload["stage2_weights"]
        active_stage3_weights = tuned_weights_payload["stage3_weights"]
        active_amount_utility_mode = tuned_weights_payload["amount_utility_mode"]
        active_weights_label = str(tuned_weights_payload.get("selected_profile") or "tuned")
    else:
        active_stage2_weights = Stage2Weights.baseline()
        active_stage3_weights = Stage3Weights.baseline()
        active_amount_utility_mode = "log"
        active_weights_label = "baseline"

    snapshot_path_text = _active_snapshot_path()
    has_snapshot = snapshot_path_text is not None
    if snapshot_path_text is not None:
        st.info(f"Active snapshot: {Path(snapshot_path_text).name}")
    else:
        st.info(
            "No snapshot available. Use 'Run Update (Ingest)' or 'Use Latest Snapshot' to start."
        )

    st.header("Pipeline Execution")
    st.caption(f"Active ranking weights: {active_weights_label}")
    similarity_mode = str(st.session_state.get("similarity_mode") or "tfidf")
    model_name = str(st.session_state.get("embedding_model_name") or DEFAULT_MODEL_NAME)
    operator_mode = _current_mode() == "operator"
    use_win_model = operator_mode and bool(st.session_state.get("use_win_model"))
    include_unconfirmed = operator_mode and bool(st.session_state.get("include_unconfirmed"))
    if operator_mode and tuned_weights_payload is not None and tuned_weights_payload.get("use_win_model") and not use_win_model:
        st.warning(
            "The selected weights profile was tuned with the win model enabled, but 'Use Win Model in Ranking' is off."
        )
    top_n = st.slider("Top-N results", min_value=10, max_value=100, value=25, step=5)
    search_term = st.text_input("Search title/sponsor/description")
    timeline_choice = st.selectbox(
        "Timeline",
        ["All", *TIMELINE_BUCKETS],
        index=1,
        format_func=lambda bucket: (
            "All" if bucket == "All" else TIMELINE_BUCKET_LABELS.get(bucket, bucket)
        ),
    )
    deadline_within_days = st.slider("Deadline within X days (0 = no filter)", 0, 365, 0, 5)
    min_amount = st.number_input("Minimum amount", min_value=0.0, value=0.0, step=500.0)
    no_essay_only = st.checkbox("Only no-essay scholarships", value=False)

    if st.button("Run Scholarship Coach", disabled=not has_snapshot, type="primary"):
        try:
            snapshot_df = _load_snapshot_cached(snapshot_path_text)
            if similarity_mode == "embeddings":
                if "embedding_key" not in snapshot_df.columns:
                    st.info(
                        "This snapshot predates persisted embedding keys. "
                        "Embedding keys and vectors will be computed locally and cached for this run."
                    )
                with st.spinner("Preparing cached scholarship embeddings..."):
                    snapshot_df = ensure_embedding_store_for_df(
                        snapshot_df,
                        model_name,
                        processed_dir=PROCESSED_DIR,
                    )
            stage1_profile = _build_stage1_profile(st.session_state.profile)
            stage2_profile = _build_stage2_profile(st.session_state.profile)

            eligible_df, ineligible_df = apply_eligibility_filter(
                snapshot_df, stage1_profile, include_unconfirmed=include_unconfirmed
            )
            eligible_df = classify_timeline(eligible_df, stage1_profile)
            scored_df = score_stage2(
                eligible_df,
                stage2_profile,
                weights=active_stage2_weights,
                amount_utility_mode=active_amount_utility_mode,
                similarity_mode=similarity_mode,
                model_name=model_name,
                processed_dir=PROCESSED_DIR,
            )
            final_df = rerank_stage3(
                scored_df,
                today=stage1_profile.today,
                profile=stage1_profile,
                timeline_bucket=None if timeline_choice == "All" else timeline_choice,
                weights=active_stage3_weights,
                use_win_model=use_win_model,
            )

            st.session_state.eligible_df = eligible_df
            st.session_state.scored_df = scored_df
            st.session_state.final_df = final_df
            st.session_state.ineligible_df = ineligible_df
            st.success(
                f"Pipeline complete: eligible={len(eligible_df)} ineligible={len(ineligible_df)}"
            )
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")

    final_df: pd.DataFrame | None = st.session_state.final_df
    if isinstance(final_df, pd.DataFrame):
        filtered_ranked = _apply_rank_filters(
            final_df,
            search_term=search_term,
            deadline_within_days=deadline_within_days,
            min_amount=float(min_amount),
            essay_required_only=no_essay_only,
            today_value=_effective_today(st.session_state.profile),
        )
        top_df = filtered_ranked.head(top_n).copy()
        top_df["amount"] = top_df.apply(
            lambda row: format_amount_range(row.get("amount_min"), row.get("amount_max")),
            axis=1,
        )

        st.subheader(f"Top Ranked Scholarships ({len(top_df)} shown)")
        win_summary = _topk_win_model_summary(top_df) if operator_mode else None
        if win_summary is not None:
            with st.expander("📊 Win Model Summary", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean P(Win)", f"{round(win_summary['mean_p_win'], 4)}")
                    st.metric("Mean Expected Value", f"${round(win_summary['mean_expected_value'], 2):,.0f}")
                with col2:
                    st.metric("Median P(Win)", f"{round(win_summary['median_p_win'], 4)}")
                    st.metric("Median Expected Value", f"${round(win_summary['median_expected_value'], 2):,.0f}")

        today_for_cards = _effective_today(st.session_state.profile)
        saved_ids = _saved_catalog_ids(_student_id())
        for _, row in top_df.iterrows():
            _render_scholarship_card(
                row, today_for_cards, saved_ids, operator_mode=operator_mode
            )

    ineligible_df: pd.DataFrame | None = st.session_state.ineligible_df
    if isinstance(ineligible_df, pd.DataFrame):
        st.subheader("Excluded Scholarships (Stage 1 Reasons)")
        reason_codes = sorted(
            {
                str(reason)
                for reasons in ineligible_df["reasons"].tolist()
                for reason in (reasons or [])
                if str(reason).strip()
            }
        )
        reason_filter = st.selectbox("Filter by reason code", ["All", *reason_codes], index=0)
        excluded_filtered = ineligible_df.copy()
        if reason_filter != "All":
            excluded_filtered = excluded_filtered[
                excluded_filtered["reasons"].apply(lambda reasons: reason_filter in (reasons or []))
            ]

        st.write(f"**{len(excluded_filtered)} scholarship{'s' if len(excluded_filtered) != 1 else ''} excluded** by eligibility filter")

        for _, row in excluded_filtered.iterrows():
            title = (
                coerce_text(row.get("title"))
                or coerce_text(row.get("scholarship_id"))
                or "Untitled"
            )
            deadline = coerce_text(row.get("deadline"), default="Unknown")
            amount_str = format_amount_range(row.get("amount_min"), row.get("amount_max"))
            reasons = row.get("reasons")
            reasons_text = reasons_to_text(reasons)

            with st.container(border=True):
                col1, col2 = st.columns([0.7, 0.3])
                with col1:
                    st.markdown(f"**{title}**")
                with col2:
                    st.text(f"Deadline: {deadline}")

                col_amt, col_reason = st.columns(2)
                with col_amt:
                    st.text(f"Award: {amount_str}")
                with col_reason:
                    st.text(f"Reason: {reasons_text}")


def _current_mode() -> modes.Mode:
    return modes.normalize_mode(st.session_state.get(modes.MODE_STATE_KEY))


def _essay_choice_labels(entries: list[essays.BankEntry]) -> dict[int, str]:
    return {
        entry.essay.id: (
            f"{entry.essay.title} · {entry.theme_label} · {entry.essay.word_count} words"
        )
        for entry in entries
    }


def _render_prompt_slot(
    conn: Any, slot: essays.PromptSlot, entries: list[essays.BankEntry], can_edit: bool
) -> None:
    """The essay picker under an award's essay prompt."""
    if not entries:
        st.caption("No essays in the bank yet — write one under Essays.")
        return
    if not can_edit:
        st.caption(slot.essay_title or "No essay linked yet.")
        return

    labels = _essay_choice_labels(entries)
    options = [0, *labels]
    chosen = st.selectbox(
        "Essay for this prompt",
        options=options,
        index=options.index(slot.essay_id) if slot.essay_id in labels else 0,
        format_func=lambda value: (
            "— no essay linked —" if value == 0 else labels[int(value)]
        ),
        key=f"slot_essay_{slot.item_id}",
        label_visibility="collapsed",
    )
    if int(chosen) == (slot.essay_id or 0):
        return
    if int(chosen) == 0:
        essays.clear_prompt(conn, slot)
    else:
        essays.use_essay_for_prompt(conn, int(chosen), slot)
    st.rerun()


def _render_checklist(conn: Any, application: repo.Application) -> None:
    items = repo.list_checklist_items(conn, application.id)
    slots = {slot.item_id: slot for slot in essays.prompt_slots(conn, application.id)}
    entries = essays.essay_bank(conn, application.student_id) if slots else []
    can_edit_essays = modes.can_edit_essays(_current_mode())
    if items:
        for item in items:
            col_done, col_due = st.columns([0.75, 0.25])
            with col_done:
                checked = st.checkbox(item.label, value=item.done, key=f"chk_{item.id}")
            with col_due:
                due_value = None
                if item.due_on:
                    try:
                        due_value = date.fromisoformat(item.due_on[:10])
                    except ValueError:
                        due_value = None
                due_choice = st.date_input(
                    "Due",
                    value=due_value,
                    key=f"chk_due_{item.id}",
                    format="YYYY-MM-DD",
                    label_visibility="collapsed",
                )
            new_due = due_choice.isoformat() if isinstance(due_choice, date) else None
            if checked != item.done or new_due != (item.due_on[:10] if item.due_on else None):
                repo.update_checklist_item(conn, item.id, done=checked, due_on=new_due)
                st.rerun()
            slot = slots.get(item.id)
            if slot is not None:
                _render_prompt_slot(conn, slot, entries, can_edit_essays)
    else:
        st.caption("No requirements recorded for this award — add what it asks for below.")

    new_label = st.text_input(
        "Add a step", key=f"chk_new_{application.id}", placeholder="e.g. Ask Ms. Perez for a letter"
    )
    if st.button("Add step", key=f"chk_add_{application.id}") and new_label.strip():
        tracker.add_checklist_item(conn, application.id, new_label)
        st.session_state[f"chk_new_{application.id}"] = ""
        st.rerun()


def _render_outcome_form(conn: Any, application: repo.Application, today_value: date) -> None:
    outcome = repo.get_outcome(conn, application.id)
    st.markdown("**Result**")
    col_result, col_amount = st.columns(2)
    with col_result:
        result = st.selectbox(
            "Result",
            options=tracker.DECIDED_STATUSES,
            index=(
                tracker.DECIDED_STATUSES.index(outcome.result)
                if outcome is not None and outcome.result in tracker.DECIDED_STATUSES
                else 0
            ),
            format_func=lambda name: tracker.STATUS_LABELS[str(name)],
            key=f"outcome_result_{application.id}",
        )
    with col_amount:
        amount = st.number_input(
            "Amount awarded",
            min_value=0.0,
            step=500.0,
            value=float(outcome.amount_awarded or 0.0) if outcome is not None else 0.0,
            key=f"outcome_amount_{application.id}",
        )
    paid_to = st.text_input(
        "Paid to",
        value=outcome.paid_to if outcome is not None else "",
        key=f"outcome_paid_{application.id}",
        placeholder="School, or the student",
    )
    renewal = st.text_input(
        "Renewal terms",
        value=outcome.renewal_terms if outcome is not None else "",
        key=f"outcome_renewal_{application.id}",
        placeholder="e.g. renewable 4 years at 3.0 GPA",
    )
    if st.button("Record result", key=f"outcome_save_{application.id}"):
        try:
            tracker.record_outcome(
                conn,
                application.id,
                str(result),
                amount_awarded=float(amount) or None,
                paid_to=paid_to,
                renewal_terms=renewal,
                today=today_value,
            )
        except tracker.TransitionError as exc:
            st.error(str(exc))
        else:
            st.rerun()


def _render_application_detail(
    conn: Any, application: repo.Application, today_value: date
) -> None:
    status = tracker.normalize_status(application.status)
    done, total = tracker.checklist_progress(repo.list_checklist_items(conn, application.id))
    progress = f" — {done}/{total} done" if total else ""
    header = f"{application.title or application.catalog_id} · {tracker.STATUS_LABELS[status]}{progress}"

    with st.expander(header, expanded=False):
        if application.deadline:
            days_until = _calculate_days_until_deadline(application.deadline, today_value)
            urgency_text, _ = _get_urgency_indicator(days_until)
            st.caption(f"Deadline: {application.deadline} · {urgency_text}")
        if application.submitted_on:
            st.caption(f"Submitted {application.submitted_on}")
        if application.source_url:
            st.link_button("Apply at Source", application.source_url)

        choices = tracker.next_statuses(status)
        chosen = st.selectbox(
            "Status",
            options=choices,
            index=0,
            format_func=lambda name: tracker.STATUS_LABELS[str(name)],
            key=f"app_status_{application.id}",
        )
        if str(chosen) != status:
            try:
                tracker.set_status(conn, application.id, str(chosen), today=today_value)
            except tracker.TransitionError as exc:
                st.error(str(exc))
            else:
                st.rerun()

        _render_checklist(conn, application)

        notes = st.text_area(
            "Notes", value=application.notes, key=f"app_notes_{application.id}", height=90
        )
        if notes != application.notes:
            repo.update_application(conn, application.id, notes=notes)

        if status in ("submitted", *tracker.DECIDED_STATUSES):
            _render_outcome_form(conn, application, today_value)

        if st.button("Remove from my list", key=f"app_delete_{application.id}"):
            repo.delete_application(conn, application.id)
            st.rerun()


def _render_applications_section() -> None:
    st.subheader(modes.SECTION_LABELS["applications"])
    today_value = _effective_today(st.session_state.profile)
    try:
        with open_db() as conn:
            student_id = _ensure_student(conn)
            applications = repo.list_applications(conn, student_id)
            if not applications:
                st.info(
                    "Nothing saved yet. Open Find Scholarships and press Save on a card."
                )
                return

            counts = {status: 0 for status in tracker.APPLICATION_STATUSES}
            for application in applications:
                counts[tracker.normalize_status(application.status)] += 1
            st.caption(
                " · ".join(
                    f"{tracker.STATUS_LABELS[status]}: {count}"
                    for status, count in counts.items()
                    if count
                )
            )

            status_filter = st.selectbox(
                "Show",
                options=("all", *tracker.APPLICATION_STATUSES),
                format_func=lambda name: (
                    "All" if name == "all" else tracker.STATUS_LABELS[str(name)]
                ),
                key="applications_status_filter",
            )
            for application in applications:
                if status_filter != "all":
                    if tracker.normalize_status(application.status) != status_filter:
                        continue
                _render_application_detail(conn, application, today_value)
    except Exception as exc:
        st.error(f"Could not open the family database: {exc}")


def _render_this_week_section() -> None:
    st.subheader(modes.SECTION_LABELS["this_week"])
    today_value = _effective_today(st.session_state.profile)
    st.caption(
        f"Due on or before {today_value + timedelta(days=tracker.THIS_WEEK_DAYS)}, "
        "plus anything already overdue."
    )
    try:
        with open_db() as conn:
            student_id = _ensure_student(conn)
            due_items = tracker.this_week(conn, student_id, today=today_value)
    except Exception as exc:
        st.error(f"Could not open the family database: {exc}")
        return

    if not due_items:
        st.success("Nothing due in the next two weeks.")
        return

    for due in due_items:
        urgency_text, _ = _get_urgency_indicator(due.days_until)
        with st.container(border=True):
            col_what, col_when = st.columns([0.7, 0.3])
            with col_what:
                st.markdown(f"**{due.label}**")
                st.caption(due.award_title)
            with col_when:
                st.text(due.due_on)
                st.caption(f"{urgency_text} · {tracker.STATUS_LABELS[due.status]}")


def _render_new_essay_form(conn: Any, student_id: str) -> None:
    with st.form("new_essay", clear_on_submit=True):
        title = st.text_input("Title", placeholder="e.g. The summer I rebuilt the robot")
        theme = st.selectbox(
            "Theme",
            options=repo.ESSAY_THEMES,
            format_func=lambda name: repo.ESSAY_THEME_LABELS[str(name)],
        )
        body = st.text_area("Draft", height=160)
        if st.form_submit_button("Add essay") and title.strip():
            repo.create_essay(conn, student_id, title.strip(), body, str(theme))
            st.rerun()


def _reuse_caption(count: int) -> str:
    if not count:
        return "not used yet"
    return f"used in {count} application{'s' if count != 1 else ''}"


def _render_essay_detail(conn: Any, entry: essays.BankEntry, can_edit: bool) -> None:
    essay = entry.essay
    header = (
        f"{essay.title} · {entry.theme_label} · {essay.word_count} words "
        f"· {_reuse_caption(entry.reuse_count)}"
    )
    with st.expander(header, expanded=False):
        if entry.used_by:
            st.caption("Used in: " + ", ".join(entry.used_by))

        if can_edit:
            title = st.text_input("Title", value=essay.title, key=f"essay_title_{essay.id}")
            theme = st.selectbox(
                "Theme",
                options=repo.ESSAY_THEMES,
                index=repo.ESSAY_THEMES.index(repo.normalize_theme(essay.theme)),
                format_func=lambda name: repo.ESSAY_THEME_LABELS[str(name)],
                key=f"essay_theme_{essay.id}",
            )
            body = st.text_area(
                "Draft", value=essay.body, height=240, key=f"essay_body_{essay.id}"
            )
            st.caption(f"{repo.word_count(body)} words")
            col_save, col_delete = st.columns([0.7, 0.3])
            with col_save:
                if st.button("Save draft", key=f"essay_save_{essay.id}"):
                    repo.update_essay(
                        conn,
                        essay.id,
                        title=title.strip() or essay.title,
                        body=body,
                        theme=str(theme),
                    )
                    st.rerun()
            with col_delete:
                if st.button("Delete essay", key=f"essay_delete_{essay.id}"):
                    repo.delete_essay(conn, essay.id)
                    st.rerun()
        else:
            st.markdown(essay.body or "_Nothing written yet._")

        earlier = repo.list_essay_versions(conn, essay.id)[1:]
        if earlier and st.checkbox(
            f"Earlier drafts ({len(earlier)})", key=f"essay_versions_{essay.id}"
        ):
            for version in earlier:
                st.caption(f"Saved {version.created_at} · {version.word_count} words")
                st.text_area(
                    "Earlier draft",
                    value=version.body,
                    height=140,
                    disabled=True,
                    key=f"essay_version_{version.id}",
                    label_visibility="collapsed",
                )


def _render_open_prompts(
    conn: Any, student_id: str, entries: list[essays.BankEntry], can_edit: bool
) -> None:
    open_slots: list[tuple[str, essays.PromptSlot]] = []
    for application in repo.list_applications(conn, student_id):
        if tracker.normalize_status(application.status) not in tracker.OPEN_STATUSES:
            continue
        title = application.title or application.catalog_id
        open_slots.extend(
            (title, slot) for slot in essays.open_prompt_slots(conn, application.id)
        )

    st.markdown("**Prompts waiting for an essay**")
    if not open_slots:
        st.caption("Every prompt on your open applications has an essay.")
        return
    for award_title, slot in open_slots:
        with st.container(border=True):
            st.markdown(f"**{slot.prompt}**")
            st.caption(award_title)
            _render_prompt_slot(conn, slot, entries, can_edit)


def _render_essays_section() -> None:
    st.subheader(modes.SECTION_LABELS["essays"])
    can_edit = modes.can_edit_essays(_current_mode())
    if not can_edit:
        st.caption("Parent view reads the essay bank. Switch to Student to edit.")
    try:
        with open_db() as conn:
            student_id = _ensure_student(conn)
            entries = essays.essay_bank(conn, student_id)
            if entries:
                counts = essays.theme_counts(entries)
                st.caption(
                    " · ".join(
                        f"{repo.ESSAY_THEME_LABELS[theme]}: {count}"
                        for theme, count in counts.items()
                        if count
                    )
                )
            else:
                st.info(
                    "No essays yet. A few themed drafts answer most prompts — start "
                    "with a challenge story and one on why your major."
                )

            if can_edit:
                with st.expander("Add an essay", expanded=not entries):
                    _render_new_essay_form(conn, student_id)

            for entry in entries:
                _render_essay_detail(conn, entry, can_edit)

            _render_open_prompts(conn, student_id, entries, can_edit)
    except Exception as exc:
        st.error(f"Could not open the family database: {exc}")


def _render_new_recommender_form(conn: Any, student_id: str) -> None:
    with st.form("new_recommender", clear_on_submit=True):
        name = st.text_input("Name", placeholder="e.g. Ms. Rivera")
        col_role, col_email = st.columns(2)
        with col_role:
            role = st.text_input("Role", placeholder="AP Physics teacher")
        with col_email:
            email = st.text_input("Email")
        if st.form_submit_button("Add recommender") and name.strip():
            repo.create_recommender(conn, student_id, name.strip(), role, email)
            st.rerun()


def _render_recommendation_request(
    conn: Any, request: repo.RecommendationRequest, award_title: str, today_value: date
) -> None:
    status = tracker.normalize_request_status(request.status)
    with st.container(border=True):
        st.markdown(f"**{award_title}**")
        col_status, col_due = st.columns(2)
        with col_status:
            chosen = st.selectbox(
                "Status",
                options=tracker.next_request_statuses(status),
                index=0,
                format_func=lambda name: tracker.REQUEST_STATUS_LABELS[str(name)],
                key=f"req_status_{request.id}",
            )
        with col_due:
            due_value = None
            if request.due_on:
                try:
                    due_value = date.fromisoformat(request.due_on[:10])
                except ValueError:
                    due_value = None
            due_choice = st.date_input(
                "Letter due", value=due_value, key=f"req_due_{request.id}", format="YYYY-MM-DD"
            )
        stamps = [
            text
            for text in (
                f"Asked {request.asked_on}" if request.asked_on else "",
                f"Received {request.received_on}" if request.received_on else "",
            )
            if text
        ]
        if stamps:
            st.caption(" · ".join(stamps))

        new_due = due_choice.isoformat() if isinstance(due_choice, date) else None
        if new_due != (request.due_on[:10] if request.due_on else None):
            repo.update_recommendation_request(conn, request.id, due_on=new_due)
            st.rerun()
        if str(chosen) != status:
            try:
                tracker.set_request_status(conn, request.id, str(chosen), today=today_value)
            except tracker.TransitionError as exc:
                st.error(str(exc))
            else:
                st.rerun()
        if st.button("Remove request", key=f"req_delete_{request.id}"):
            repo.delete_recommendation_request(conn, request.id)
            st.rerun()


def _render_recommender_detail(
    conn: Any,
    recommender: repo.Recommender,
    applications: list[repo.Application],
    today_value: date,
) -> None:
    requests = repo.list_recommendation_requests(conn, recommender_id=recommender.id)
    outstanding = sum(
        1
        for request in requests
        if tracker.normalize_request_status(request.status) in tracker.OPEN_REQUEST_STATUSES
    )
    titles = {
        application.id: application.title or application.catalog_id
        for application in applications
    }
    header = recommender.name
    if recommender.role:
        header += f" · {recommender.role}"
    header += f" · {outstanding} outstanding" if outstanding else " · all in"

    with st.expander(header, expanded=False):
        col_role, col_email = st.columns(2)
        with col_role:
            role = st.text_input("Role", value=recommender.role, key=f"rec_role_{recommender.id}")
        with col_email:
            email = st.text_input(
                "Email", value=recommender.email, key=f"rec_email_{recommender.id}"
            )
        if (role, email) != (recommender.role, recommender.email):
            repo.update_recommender(conn, recommender.id, role=role, email=email)

        for request in requests:
            _render_recommendation_request(
                conn, request, titles.get(request.application_id, "This award"), today_value
            )

        remaining = [
            application
            for application in applications
            if application.id not in {request.application_id for request in requests}
        ]
        if remaining:
            chosen = st.selectbox(
                "Ask for a letter for",
                options=[application.id for application in remaining],
                format_func=lambda value: titles.get(int(value), ""),
                key=f"rec_new_request_{recommender.id}",
            )
            if st.button("Add request", key=f"rec_add_request_{recommender.id}"):
                repo.create_recommendation_request(conn, recommender.id, int(chosen))
                st.rerun()

        if st.button("Remove recommender", key=f"rec_delete_{recommender.id}"):
            repo.delete_recommender(conn, recommender.id)
            st.rerun()


def _render_recommenders_section() -> None:
    st.subheader(modes.SECTION_LABELS["recommenders"])
    today_value = _effective_today(st.session_state.profile)
    try:
        with open_db() as conn:
            student_id = _ensure_student(conn)
            people = repo.list_recommenders(conn, student_id)
            applications = repo.list_applications(conn, student_id)
            if not people:
                st.info(
                    "No recommenders yet. Ask early — a teacher writing ten letters "
                    "needs weeks, not days."
                )
            with st.expander("Add a recommender", expanded=not people):
                _render_new_recommender_form(conn, student_id)
            if people and not applications:
                st.caption("Save an award first, then ask a recommender for its letter.")
            for recommender in people:
                _render_recommender_detail(conn, recommender, applications, today_value)
    except Exception as exc:
        st.error(f"Could not open the family database: {exc}")


def _award_calendar_events(today_value: date) -> list[calendar_feed.CalendarEvent]:
    """Catalog awards as calendar events, keeping the bucket ranking gave them."""
    eligible_df = st.session_state.get("eligible_df")
    if not isinstance(eligible_df, pd.DataFrame) or eligible_df.empty:
        return []

    per_bucket: dict[str, list[calendar_feed.CalendarEvent]] = {}
    for _, row in eligible_df.iterrows():
        bucket = coerce_text(row.get("timeline_bucket")) or "now"
        if bucket not in calendar_feed.CALENDAR_BUCKETS:
            continue
        deadline_text, is_projected = _timeline_deadline(row, today_value)
        try:
            starts_on = date.fromisoformat(str(deadline_text)[:10])
        except ValueError:
            continue
        catalog_id = _row_catalog_id(row) or coerce_text(row.get("title"))
        per_bucket.setdefault(bucket, []).append(
            calendar_feed.CalendarEvent(
                uid=f"award-{catalog_id}",
                kind="award",
                title=coerce_text(row.get("title")) or catalog_id,
                starts_on=starts_on,
                detail=coerce_text(row.get("sponsor")),
                note="Projected from this award's usual cycle" if is_projected else "",
                bucket=bucket,
            )
        )

    events: list[calendar_feed.CalendarEvent] = []
    for bucket_events in per_bucket.values():
        events.extend(
            calendar_feed.sort_events(bucket_events)[:TIMELINE_AWARDS_PER_BUCKET]
        )
    return events


def _render_calendar_event(event: calendar_feed.CalendarEvent, today_value: date) -> None:
    when = event.starts_on.strftime("%a %d")
    if event.ends_on is not None and event.ends_on != event.starts_on:
        when = f"{when} – {event.ends_on.strftime('%a %d')}"
    urgency_text, _ = _get_urgency_indicator(event.days_until(today_value))

    col_when, col_what = st.columns([0.18, 0.82])
    with col_when:
        st.markdown(f"**{when}**")
    with col_what:
        st.markdown(f"{event.kind_label}: {event.title}")
        caption = " · ".join(part for part in (event.detail, event.note) if part)
        if caption:
            st.caption(caption)
        if event.bucket == "now":
            st.caption(urgency_text)


def _render_timeline_bucket(
    events: list[calendar_feed.CalendarEvent], today_value: date
) -> None:
    if not events:
        st.info("Nothing scheduled in this stretch yet.")
        return

    for year_end, year_events in calendar_feed.group_by_school_year(events).items():
        st.markdown(f"**{milestones.school_year_label(year_end)} school year**")
        for (year, month), month_events in calendar_feed.group_by_month(year_events).items():
            with st.container(border=True):
                st.markdown(f"##### {calendar.month_name[month]} {year}")
                for event in month_events:
                    _render_calendar_event(event, today_value)


def _render_timeline_section() -> None:
    st.subheader(modes.SECTION_LABELS["timeline"])
    today_value = _effective_today(st.session_state.profile)
    grade_level = str(st.session_state.profile.get("grade_level") or "")

    try:
        with open_db() as conn:
            student_id = _ensure_student(conn)
            events = calendar_feed.family_events(
                conn, student_id, grade_level=grade_level or None, today=today_value
            )
    except Exception as exc:
        st.error(f"Could not open the family database: {exc}")
        return

    family_only = list(events)
    events = calendar_feed.sort_events([*events, *_award_calendar_events(today_value)])

    st.caption(
        "Deadlines, tasks, letters and milestones by month. Milestone dates are typical, "
        "not guaranteed — confirm each one with the college or agency."
    )
    if st.session_state.get("eligible_df") is None:
        st.caption("Run the pipeline under Find Scholarships to add catalog award cycles here.")

    st.download_button(
        "Download calendar (.ics)",
        data=calendar_feed.to_ics(family_only),
        file_name=CALENDAR_FILE_NAME,
        mime="text/calendar",
        help="Saved applications, their tasks and letters, and the milestones that apply.",
    )

    tabs = st.tabs(
        [
            TIMELINE_BUCKET_LABELS.get(bucket, bucket)
            for bucket in calendar_feed.CALENDAR_BUCKETS
        ]
    )
    for tab, bucket in zip(tabs, calendar_feed.CALENDAR_BUCKETS):
        with tab:
            _render_timeline_bucket(
                calendar_feed.events_in_bucket(events, bucket), today_value
            )


def _money_text(value: float | None) -> str:
    return "—" if value is None else f"${value:,.0f}"


def _award_count_text(count: int) -> str:
    return f"{count} award" if count == 1 else f"{count} awards"


def _render_new_college_form(conn: Any, student_id: str) -> None:
    with st.form("new_college", clear_on_submit=True):
        name = st.text_input("College", placeholder="e.g. NC State University")
        col_state, col_deadline = st.columns(2)
        with col_state:
            in_state = st.checkbox("In state")
        with col_deadline:
            deadline_type = st.selectbox(
                "Deadline type",
                options=repo.COLLEGE_DEADLINE_TYPES,
                format_func=lambda name: repo.COLLEGE_DEADLINE_TYPE_LABELS[str(name)],
            )
        col_sticker, col_net = st.columns(2)
        with col_sticker:
            sticker = st.number_input("Sticker price", min_value=0.0, step=1000.0, value=0.0)
        with col_net:
            net = st.number_input(
                "Net price estimate",
                min_value=0.0,
                step=1000.0,
                value=0.0,
                help="From the school's own net price calculator, not a published average.",
            )
        if st.form_submit_button("Add college") and name.strip():
            repo.create_college(
                conn,
                student_id,
                name.strip(),
                cost_of_attendance=float(sticker) or None,
                in_state=bool(in_state),
                net_price_estimate=float(net) or None,
                deadline_type=str(deadline_type),
            )
            st.rerun()


def _render_college_detail(conn: Any, cost: money.CollegeCost) -> None:
    where = "In state" if cost.in_state else "Out of state"
    header = f"{cost.name} · {where} · net {_money_text(cost.net_price)}"
    with st.expander(header, expanded=False):
        st.caption(
            f"Sticker {_money_text(cost.sticker_price)} · "
            f"{repo.COLLEGE_DEADLINE_TYPE_LABELS[cost.deadline_type]} · "
            f"still to cover {_money_text(cost.remaining)}"
        )
        college_id = cost.college_id

        col_state, col_deadline = st.columns(2)
        with col_state:
            in_state = st.checkbox(
                "In state", value=cost.in_state, key=f"college_in_state_{college_id}"
            )
        with col_deadline:
            deadline_type = st.selectbox(
                "Deadline type",
                options=repo.COLLEGE_DEADLINE_TYPES,
                index=repo.COLLEGE_DEADLINE_TYPES.index(cost.deadline_type),
                format_func=lambda name: repo.COLLEGE_DEADLINE_TYPE_LABELS[str(name)],
                key=f"college_deadline_{college_id}",
            )

        col_sticker, col_net = st.columns(2)
        with col_sticker:
            sticker = st.number_input(
                "Sticker price",
                min_value=0.0,
                step=1000.0,
                value=float(cost.sticker_price or 0.0),
                key=f"college_sticker_{college_id}",
            )
        with col_net:
            net = st.number_input(
                "Net price estimate",
                min_value=0.0,
                step=1000.0,
                value=float(cost.net_price_estimate or 0.0),
                key=f"college_net_{college_id}",
                help="From the school's own net price calculator.",
            )

        merit = st.text_area(
            "Merit aid notes",
            value=cost.merit_aid_notes,
            height=68,
            key=f"college_merit_{college_id}",
            placeholder="e.g. Park Scholarship, separate application due October",
        )
        policy = st.text_area(
            "Outside-award policy",
            value=cost.outside_award_policy,
            height=68,
            key=f"college_policy_{college_id}",
            help=(
                "What the school does with a scholarship won elsewhere: reduce loans "
                "and work-study first, or displace its own grant dollar for dollar."
            ),
            placeholder="e.g. reduces loans first, then institutional grant",
        )

        col_save, col_remove = st.columns(2)
        with col_save:
            if st.button("Save", key=f"college_save_{college_id}"):
                repo.update_college(
                    conn,
                    college_id,
                    in_state=bool(in_state),
                    deadline_type=str(deadline_type),
                    cost_of_attendance=float(sticker) or None,
                    net_price_estimate=float(net) or None,
                    merit_aid_notes=merit,
                    outside_award_policy=policy,
                )
                st.rerun()
        with col_remove:
            if st.button("Remove", key=f"college_remove_{college_id}"):
                repo.delete_college(conn, college_id)
                st.rerun()


def _render_money_summary(summary: money.MoneySummary) -> None:
    col_won, col_awards, col_colleges = st.columns(3)
    col_won.metric("Total won", _money_text(summary.total_won))
    col_awards.metric("Awards won", str(summary.award_count))
    col_colleges.metric("Colleges tracked", str(len(summary.colleges)))

    if summary.by_year:
        st.markdown("**Won by school year**")
        for row in summary.by_year:
            st.markdown(
                f"{row.label}: {_money_text(row.total)} · {_award_count_text(row.count)}"
            )

    if summary.renewals:
        st.markdown("**Renewal conditions to keep**")
        for renewal in summary.renewals:
            with st.container(border=True):
                st.markdown(f"**{renewal.award_title}** — {_money_text(renewal.amount)}")
                st.caption(renewal.terms)


def _render_colleges_money_section() -> None:
    st.subheader(modes.SECTION_LABELS["colleges_money"])
    if _current_mode() == "student":
        st.info("Colleges and money live in Parent view.")
        return

    try:
        with open_db() as conn:
            student_id = _ensure_student(conn)
            colleges = repo.list_colleges(conn, student_id)
            titles = {
                application.id: application.title or application.catalog_id
                for application in repo.list_applications(conn, student_id)
            }
            summary = money.money_summary(
                colleges, repo.list_outcomes(conn, student_id), titles
            )

            _render_money_summary(summary)
            st.divider()
            st.caption(
                "Net price is one year; total won is everything to date, so "
                "\"still to cover\" is a first-year figure. Check each school's "
                "outside-award policy before counting a scholarship against it."
            )
            with st.expander("Add a college", expanded=not colleges):
                _render_new_college_form(conn, student_id)
            for cost in summary.colleges:
                _render_college_detail(conn, cost)
    except Exception as exc:
        st.error(f"Could not open the family database: {exc}")


def _catalog_tristate_label(value: bool | None) -> str:
    if value is None:
        return CATALOG_TRISTATE_OPTIONS[0]
    return "Yes" if value else "No"


def _catalog_tristate_value(label: str) -> bool | None:
    if label == "Yes":
        return True
    if label == "No":
        return False
    return None


def _tristate_select(label: str, value: bool | None, *, help: str | None = None) -> bool | None:
    choice = st.selectbox(
        label,
        options=CATALOG_TRISTATE_OPTIONS,
        index=CATALOG_TRISTATE_OPTIONS.index(_catalog_tristate_label(value)),
        help=help,
    )
    return _catalog_tristate_value(str(choice))


def _month_select(label: str, value: int | None) -> int | None:
    options = (0, *catalog_entry.MONTH_OPTIONS)
    choice = st.selectbox(
        label,
        options=options,
        index=options.index(value) if value in options else 0,
        format_func=lambda number: "—" if not number else calendar.month_name[int(number)],
    )
    return int(choice) or None


def _with_existing(options: tuple[str, ...], selected: list[str]) -> list[str]:
    """Offer the vocabulary plus whatever this record already carries.

    Editing a record must never silently drop a value just because it predates
    the dropdown's list.
    """
    return list(dict.fromkeys([*options, *selected]))


def _catalog_form_values() -> dict[str, Any]:
    values = st.session_state.get(CATALOG_FORM_KEY)
    if not isinstance(values, dict):
        values = catalog_entry.blank_form()
        st.session_state[CATALOG_FORM_KEY] = values
    return values


def _load_catalog_form(values: dict[str, Any], *, proposal_id: str | None = None) -> None:
    st.session_state[CATALOG_FORM_KEY] = values
    st.session_state[CATALOG_EDIT_KEY] = proposal_id
    # The page selector owns its key, so a switch asked for from the inbox has
    # to wait until just before that widget is drawn on the next run.
    st.session_state[CATALOG_PAGE_REQUEST_KEY] = "add"


def _set_catalog_notice(level: str, message: str) -> None:
    st.session_state[CATALOG_NOTICE_KEY] = (level, message)


def _render_catalog_notice() -> None:
    notice = st.session_state.pop(CATALOG_NOTICE_KEY, None)
    if not notice:
        return
    level, message = notice
    {"success": st.success, "warning": st.warning, "error": st.error}.get(level, st.info)(message)


def _render_prefill_input() -> None:
    col_url, col_fetch = st.columns([0.75, 0.25], vertical_alignment="bottom")
    with col_url:
        url = st.text_input(
            "Scholarship URL",
            key="catalog_prefill_url",
            placeholder="https://example.org/scholarship",
            help="Paste a listing to prefill the form. Leave it empty to type an award in by hand.",
        )
    with col_fetch:
        fetch = st.button("Read the page", use_container_width=True)

    if not fetch:
        return
    if not url.strip():
        st.warning("Paste a URL first, or fill the form in by hand below.")
        return

    with st.spinner("Reading the page…"):
        result = prefill_from_url(url.strip())
    _load_catalog_form(catalog_entry.form_from_prefill(result.to_form_dict()))
    if result.error:
        _set_catalog_notice(
            "warning",
            f"Could not read that page ({result.error}). The URL is filled in; the rest is by hand.",
        )
    else:
        _set_catalog_notice("success", "Read the page. Check every field before confirming.")
    st.rerun()


def _render_prefill_candidates(values: dict[str, Any]) -> None:
    deadlines = list(values.get("deadline_candidates") or [])
    amounts = list(values.get("amount_candidates") or [])
    if not deadlines and not amounts:
        return

    with st.container(border=True):
        st.caption("Found on the page. Pick one, or ignore these and type your own.")
        if deadlines:
            col_pick, col_use = st.columns([0.7, 0.3], vertical_alignment="bottom")
            choice = col_pick.selectbox("Dates on the page", options=deadlines)
            if col_use.button("Use as deadline", use_container_width=True):
                values["deadline"] = str(choice)
                st.rerun()
        if amounts:
            col_pick, col_min, col_max = st.columns([0.5, 0.25, 0.25], vertical_alignment="bottom")
            amount = col_pick.selectbox(
                "Amounts on the page",
                options=amounts,
                format_func=lambda value: f"${float(value):,.0f}",
            )
            if col_min.button("Use as minimum", use_container_width=True):
                values["amount_min"] = float(amount)
                st.rerun()
            if col_max.button("Use as maximum", use_container_width=True):
                values["amount_max"] = float(amount)
                st.rerun()

        confidence = values.get("confidence") or {}
        if confidence:
            st.caption(
                "Extractor confidence — "
                + ", ".join(
                    f"{name} {float(score):.2f}" for name, score in sorted(confidence.items())
                )
            )


def _award_form_values(values: dict[str, Any]) -> dict[str, Any]:
    """Draw the entry form and return the values it was submitted with.

    Returns an empty dict while the form has not been submitted.  The widgets
    carry no keys on purpose: the defaults come from ``values``, so loading a
    prefill or a proposal into the form redraws it with the new content.
    """
    form = dict(values)
    with st.form("catalog_entry_form"):
        st.markdown("**The award**")
        col_title, col_id = st.columns(2)
        form["title"] = col_title.text_input("Title", value=values["title"])
        form["catalog_id"] = col_id.text_input(
            "Catalog ID",
            value=values["catalog_id"],
            help="Stable lowercase slug; it is this award's identity across cycles. Leave it to derive one from the title.",
        )
        col_sponsor, col_url = st.columns(2)
        form["sponsor"] = col_sponsor.text_input("Sponsor", value=values["sponsor"])
        form["source_url"] = col_url.text_input("Source URL", value=values["source_url"])
        form["description"] = st.text_area("Description", value=values["description"], height=80)
        form["eligibility_text"] = st.text_area(
            "Eligibility text",
            value=values["eligibility_text"],
            height=80,
            help="The page's own words about who may apply. Stage 2 scores against this.",
        )

        st.markdown("**Money and dates**")
        col_min, col_max, col_status = st.columns(3)
        form["amount_min"] = col_min.number_input(
            "Amount minimum", min_value=0.0, step=500.0, value=values["amount_min"]
        )
        form["amount_max"] = col_max.number_input(
            "Amount maximum", min_value=0.0, step=500.0, value=values["amount_max"]
        )
        status_options = catalog_entry.status_options()
        form["status"] = col_status.selectbox(
            "Status",
            options=status_options,
            index=status_options.index(values["status"]) if values["status"] in status_options else 0,
        )
        col_deadline, col_opens, col_month = st.columns(3)
        with col_deadline:
            deadline = st.date_input(
                "Deadline",
                value=date.fromisoformat(values["deadline"]) if values["deadline"] else None,
                format="YYYY-MM-DD",
            )
            form["deadline"] = deadline
        with col_opens:
            form["cycle_opens_month"] = _month_select("Opens in", values["cycle_opens_month"])
        with col_month:
            form["cycle_deadline_month"] = _month_select(
                "Deadline month", values["cycle_deadline_month"]
            )
        form["cycle_recurring"] = _tristate_select(
            "Runs every year", values["cycle_recurring"], help="Drives the projected next deadline."
        )

        st.markdown("**Who it is for**")
        col_level, col_grades = st.columns(2)
        level_options = ("", *catalog_entry.EDUCATION_LEVEL_OPTIONS)
        form["education_level"] = col_level.selectbox(
            "Education level",
            options=level_options,
            index=level_options.index(values["education_level"])
            if values["education_level"] in level_options
            else 0,
            format_func=lambda name: str(name) or "Any",
        )
        form["grade_levels"] = col_grades.multiselect(
            "Grade levels",
            options=catalog_entry.GRADE_LEVEL_OPTIONS,
            default=values["grade_levels"],
            format_func=lambda level: GRADE_LEVEL_LABELS.get(str(level), str(level)),
        )
        form["majors_allowed"] = st.multiselect(
            "Majors allowed (empty means any)",
            options=_with_existing(catalog_entry.MAJOR_OPTIONS, values["majors_allowed"]),
            default=values["majors_allowed"],
            format_func=lambda major: str(major).title(),
        )
        col_states, col_counties = st.columns(2)
        form["states_allowed"] = col_states.multiselect(
            "States allowed (empty means national)",
            options=catalog_entry.STATE_OPTIONS,
            default=values["states_allowed"],
        )
        form["counties_allowed"] = col_counties.multiselect(
            "NC counties allowed",
            options=_with_existing(catalog_entry.COUNTY_OPTIONS, values["counties_allowed"]),
            default=values["counties_allowed"],
        )
        col_gpa, col_sat, col_act = st.columns(3)
        form["min_gpa"] = col_gpa.number_input(
            "Minimum GPA", min_value=0.0, max_value=5.0, step=0.1, value=values["min_gpa"]
        )
        form["sat"] = col_sat.number_input(
            "Minimum SAT", min_value=400, max_value=1600, step=10, value=values["sat"]
        )
        form["act"] = col_act.number_input(
            "Minimum ACT", min_value=1, max_value=36, step=1, value=values["act"]
        )
        col_citizen, col_gender, col_religion = st.columns(3)
        form["citizenship"] = col_citizen.text_input("Citizenship", value=values["citizenship"])
        gender_options = ("", *catalog_entry.gender_options())
        form["gender"] = col_gender.selectbox(
            "Gender",
            options=gender_options,
            index=gender_options.index(values["gender"]) if values["gender"] in gender_options else 0,
            format_func=lambda name: str(name).title() or "Not stated",
        )
        form["religion"] = col_religion.text_input("Religion", value=values["religion"])
        col_need, col_first_gen, col_military, col_disability = st.columns(4)
        with col_need:
            form["need_based"] = _tristate_select("Need based", values["need_based"])
        with col_first_gen:
            form["first_gen_only"] = _tristate_select("First generation only", values["first_gen_only"])
        with col_military:
            form["military_family"] = _tristate_select("Military family", values["military_family"])
        with col_disability:
            form["disability"] = _tristate_select("Disability", values["disability"])
        col_heritage, col_employer, col_membership = st.columns(3)
        form["heritage"] = col_heritage.text_input(
            "Heritage (comma separated)", value=_csv_text(values["heritage"])
        )
        form["employer_restricted"] = col_employer.text_input(
            "Employers (comma separated)", value=_csv_text(values["employer_restricted"])
        )
        form["membership_required"] = col_membership.text_input(
            "Memberships (comma separated)", value=_csv_text(values["membership_required"])
        )

        st.markdown("**What it asks for**")
        requirement_columns = st.columns(len(catalog_entry.REQUIREMENT_FLAGS))
        evidence = values.get("requirement_evidence") or {}
        for column, flag in zip(requirement_columns, catalog_entry.REQUIREMENT_FLAGS, strict=True):
            with column:
                form[f"req_{flag}"] = _tristate_select(
                    REQUIREMENT_LABELS[flag],
                    values[f"req_{flag}"],
                    help=str(evidence.get(flag) or "") or None,
                )
        col_letters, col_renewal = st.columns(2)
        form["recommendation_letters"] = col_letters.number_input(
            "Recommendation letters",
            min_value=0,
            max_value=10,
            step=1,
            value=values["recommendation_letters"],
        )
        form["renewal_terms"] = col_renewal.text_input(
            "Renewal terms", value=values["renewal_terms"]
        )
        form["essay_prompts"] = st.text_area(
            "Essay prompts (one per line)",
            value="\n".join(values["essay_prompts"]),
            height=68,
        )
        form["keywords"] = st.text_input(
            "Keywords (comma separated)", value=_csv_text(values["keywords"])
        )

        st.markdown("**Where it came from**")
        col_trust, col_kind = st.columns(2)
        trust_options = catalog_entry.trust_options()
        form["trust"] = col_trust.selectbox(
            "Trust",
            options=trust_options,
            index=trust_options.index(values["trust"]) if values["trust"] in trust_options else 0,
            format_func=lambda name: TRUST_LABELS.get(str(name), str(name)),
            help="verified_local means a person opened the URL and confirmed this record today.",
        )
        source_kinds = catalog_entry.source_kind_options()
        form["source_kind"] = col_kind.selectbox(
            "Source kind",
            options=source_kinds,
            index=source_kinds.index(values["source_kind"]) if values["source_kind"] in source_kinds else 0,
            format_func=lambda name: str(name).replace("_", " ").capitalize(),
        )
        col_verified, col_by = st.columns(2)
        with col_verified:
            verified_on = st.date_input(
                "Verified on",
                value=date.fromisoformat(values["verified_on"]) if values["verified_on"] else None,
                format="YYYY-MM-DD",
            )
            form["verified_on"] = verified_on
        form["verified_by"] = col_by.text_input("Verified by", value=values["verified_by"])
        form["notes"] = st.text_area("Notes", value=values["notes"], height=68)

        submitted = st.form_submit_button("Confirm and add to the catalog", type="primary")

    if not submitted:
        return {}
    # Back to the shapes blank_form() uses, so a rejected submission redraws
    # with what was typed rather than with a date object or a split string.
    for field_name in ("heritage", "employer_restricted", "membership_required", "keywords"):
        form[field_name] = _csv_list(form[field_name])
    form["essay_prompts"] = [
        line.strip() for line in str(form["essay_prompts"]).splitlines() if line.strip()
    ]
    for field_name in ("deadline", "verified_on"):
        value = form[field_name]
        form[field_name] = value.isoformat() if isinstance(value, date) else ""
    return form


def _confirm_award_record(record: dict[str, Any], proposal_id: str | None) -> Path:
    """Write a confirmed record, always through the inbox's ``confirm``.

    Hand entry proposes and immediately confirms so that ``records/`` keeps a
    single writer; an edited proposal is confirmed in place so it leaves the
    queue instead of lingering after its award is in the catalog.
    """
    if proposal_id:
        return inbox.confirm(proposal_id, edits=record)
    proposal = inbox.propose(record, kind="manual", notes="Entered by hand on the catalog page.")
    return inbox.confirm(proposal.proposal_id)


def _render_add_award_page() -> None:
    editing = st.session_state.get(CATALOG_EDIT_KEY)
    if editing:
        st.info(f"Editing proposal `{editing}`. Confirming clears it from the inbox.")
    _render_prefill_input()
    values = _catalog_form_values()
    _render_prefill_candidates(values)

    submitted = _award_form_values(values)
    if not submitted:
        return

    # Keep what was typed on screen when validation fails.
    st.session_state[CATALOG_FORM_KEY] = submitted
    record, errors = catalog_entry.validate_form(submitted)
    if errors:
        st.error("This is not a valid catalog record yet:")
        for message in errors:
            st.markdown(f"- {message}")
        return

    try:
        path = _confirm_award_record(record, editing)
    except inbox.ProposalError as exc:
        st.error(f"Could not add the award: {exc}")
        return

    st.session_state[CATALOG_FORM_KEY] = catalog_entry.blank_form()
    st.session_state[CATALOG_EDIT_KEY] = None
    _set_catalog_notice(
        "success",
        f"Added **{record['title']}** to the catalog ({Path(path).name}). "
        "Rebuild the snapshot to rank it.",
    )
    st.rerun()


def _proposal_edit_record(proposal: inbox.Proposal) -> dict[str, Any]:
    """The record to edit: a reverify proposal carries only the fields it changes."""
    existing: dict[str, Any] = {}
    catalog_id = proposal.catalog_id
    if catalog_id:
        path = RECORDS_DIR / f"{catalog_id}.json"
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8-sig"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                loaded = None
            if isinstance(loaded, dict):
                existing = loaded
    return {**existing, **proposal.record}


def _render_proposal(proposal: inbox.Proposal) -> None:
    label = proposal.title or proposal.catalog_id or proposal.proposal_id
    with st.expander(f"{label} · {proposal.created_on}", expanded=False):
        if proposal.notes:
            st.caption(proposal.notes)
        rows = catalog_entry.diff_rows(proposal.diff)
        if rows:
            st.dataframe(
                pd.DataFrame(
                    [{"Field": row.field, "In the catalog": row.old, "Proposed": row.new} for row in rows]
                ),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("Nothing in the catalog to compare against — this would be a new award.")
            st.json(proposal.record, expanded=False)

        proposal_id = proposal.proposal_id
        col_confirm, col_edit = st.columns(2)
        if col_confirm.button("Confirm", key=f"inbox_confirm_{proposal_id}", use_container_width=True):
            try:
                path = inbox.confirm(proposal_id)
            except inbox.ProposalError as exc:
                st.error(f"{exc}")
                st.caption("Use “Edit and confirm” to fill in what the record is missing.")
            else:
                _set_catalog_notice("success", f"Confirmed {Path(path).name}. Rebuild the snapshot to rank it.")
                st.rerun()
        if col_edit.button("Edit and confirm", key=f"inbox_edit_{proposal_id}", use_container_width=True):
            _load_catalog_form(
                catalog_entry.form_from_record(_proposal_edit_record(proposal)),
                proposal_id=proposal_id,
            )
            st.rerun()

        reason = st.text_input("Reason to reject", key=f"inbox_reason_{proposal_id}")
        if st.button("Reject", key=f"inbox_reject_{proposal_id}"):
            if not reason.strip():
                st.warning("A rejection needs a reason — a later pass reads it.")
            else:
                inbox.reject(proposal_id, reason)
                _set_catalog_notice("success", f"Rejected {proposal_id}.")
                st.rerun()


def _render_inbox_page() -> None:
    try:
        proposals = inbox.list_proposals()
    except Exception as exc:
        st.error(f"Could not read the inbox: {exc}")
        return

    if not proposals:
        st.info("Nothing waiting. Feeds and re-verification runs leave their proposals here.")
        return

    st.caption(
        f"{len(proposals)} waiting. Nothing reaches the catalog until it is confirmed here."
    )
    for kind in inbox.PROPOSAL_KINDS:
        matching = [proposal for proposal in proposals if proposal.kind == kind]
        if not matching:
            continue
        st.markdown(f"**{PROPOSAL_KIND_LABELS.get(kind, kind)}** ({len(matching)})")
        for proposal in matching:
            _render_proposal(proposal)


def _render_rebuild_snapshot() -> None:
    with st.container(border=True):
        st.markdown("**Rebuild snapshot**")
        st.caption(
            "Runs the curated catalog only — no scrapers, no network — so an award "
            "confirmed a moment ago is rankable now."
        )
        if not st.button("Rebuild snapshot"):
            return
        try:
            with st.spinner("Rebuilding from the curated catalog…"):
                report = run_ingest(date=None, only_sources=[CURATED_SOURCE_NAME])
        except Exception as exc:
            st.error(f"Rebuild failed: {exc}")
            return

        snapshot = report["artifact_paths"]["snapshot"]
        if snapshot is None:
            st.error(
                report["artifact_notes"]["snapshot_skip_reason"] or "No snapshot was written."
            )
            if report["artifact_notes"]["snapshot_blocked"]:
                st.caption(
                    "The prior snapshot is untouched. Nothing was lost — but the curated "
                    "records just confirmed are not rankable until this is resolved."
                )
            return
        st.session_state.ingest_report = report
        st.session_state.latest_snapshot_path = snapshot
        st.session_state.latest_delta_summary = report["delta_counts"]
        carried = report["records"]["carried_forward"]
        curated_total = report["records"]["snapshot_total"] - sum(carried.values())
        st.success(
            f"Snapshot rebuilt with {curated_total} curated awards "
            f"and {report['records']['snapshot_total']} records in total."
        )
        if carried:
            st.caption(
                "Carried forward from the prior snapshot: "
                + ", ".join(f"{source} ({count})" for source, count in carried.items())
            )


def _render_catalog_inbox_section() -> None:
    st.subheader(modes.SECTION_LABELS["catalog_inbox"])
    if _current_mode() == "student":
        st.info("Adding and reviewing awards lives in Parent view.")
        return

    _render_catalog_notice()
    st.session_state.setdefault(CATALOG_PAGE_KEY, CATALOG_PAGES[0])
    requested = st.session_state.pop(CATALOG_PAGE_REQUEST_KEY, None)
    if requested in CATALOG_PAGES:
        st.session_state[CATALOG_PAGE_KEY] = requested
    page = str(
        st.radio(
            "Catalog page",
            options=CATALOG_PAGES,
            format_func=lambda name: CATALOG_PAGE_LABELS[str(name)],
            horizontal=True,
            label_visibility="collapsed",
            key=CATALOG_PAGE_KEY,
        )
    )
    if page == "inbox":
        _render_inbox_page()
    else:
        _render_add_award_page()
    st.divider()
    _render_rebuild_snapshot()


def _operator_enabled() -> bool:
    try:
        with open_db() as conn:
            return repo.get_flag(conn, modes.OPERATOR_ENABLED_SETTING)
    except Exception:
        return False


def _milestone_window_text(milestone: milestones.Milestone) -> str:
    start = f"{calendar.month_abbr[milestone.month]} {milestone.day}"
    if milestone.end_month is None or milestone.end_day is None:
        return start
    return f"{start} – {calendar.month_abbr[milestone.end_month]} {milestone.end_day}"


def _render_milestone_settings(conn: Any) -> None:
    st.markdown("**Milestones**")
    st.caption(
        "The general planning dates every family shares. Hide the ones that do not apply, "
        "and add your own — a district scholarship night, a counselor's deadline."
    )

    overrides = {str(row["id"]): dict(row) for row in milestones.load_overrides(conn)}
    defaults = milestones.load_default_milestones()
    default_ids = {milestone.id for milestone in defaults}
    resolved = milestones.load_milestones(conn)
    shown_ids = {milestone.id for milestone in resolved}
    changed = False

    for milestone in defaults:
        visible = milestone.id in shown_ids
        choice = st.checkbox(
            f"{milestone.title} ({_milestone_window_text(milestone)})",
            value=visible,
            key=f"milestone_show_{milestone.id}",
            help=milestone.note or None,
        )
        if choice == visible:
            continue
        if choice:
            overrides.pop(milestone.id, None)
        else:
            overrides[milestone.id] = {"id": milestone.id, "hidden": True}
        changed = True

    for milestone in [row for row in resolved if row.id not in default_ids]:
        col_label, col_remove = st.columns([0.8, 0.2])
        with col_label:
            st.text(f"{milestone.title} ({_milestone_window_text(milestone)})")
        with col_remove:
            if st.button("Remove", key=f"milestone_remove_{milestone.id}"):
                overrides.pop(milestone.id, None)
                changed = True

    if changed:
        milestones.save_overrides(conn, list(overrides.values()))
        st.rerun()

    with st.form("new_milestone", clear_on_submit=True):
        st.caption("Add a family milestone")
        title = st.text_input("Title", placeholder="e.g. District scholarship night")
        col_month, col_day = st.columns(2)
        with col_month:
            month = st.selectbox(
                "Month",
                options=list(range(1, 13)),
                format_func=lambda number: calendar.month_name[int(number)],
            )
        with col_day:
            day = st.number_input("Day", min_value=1, max_value=31, value=1, step=1)
        grades = st.multiselect(
            "Grades it applies to (leave empty for every year)", options=list(GRADE_SEQUENCE)
        )
        note = st.text_area("Note", height=68)
        if st.form_submit_button("Add milestone") and title.strip():
            identifier = milestones.family_milestone_id(title, taken=shown_ids | set(overrides))
            overrides[identifier] = {
                "id": identifier,
                "title": title.strip(),
                "month": int(month),
                "day": int(day),
                "grade_levels": list(grades),
                "note": note.strip(),
            }
            milestones.save_overrides(conn, list(overrides.values()))
            st.rerun()


def _render_what_if_award_list(
    title: str, awards: list[WhatIfAward], *, reason_caption: str
) -> None:
    if not awards:
        return
    st.markdown(f"**{title}**")
    for award in awards:
        with st.container(border=True):
            st.markdown(f"**{award.title or award.scholarship_id}**")
            bucket_label = TIMELINE_BUCKET_LABELS.get(
                award.timeline_bucket, award.timeline_bucket
            )
            columns = st.columns(2)
            with columns[0]:
                st.caption(f"{_money_text(award.amount)} · {bucket_label}")
            with columns[1]:
                if award.reason_text:
                    st.caption(f"{reason_caption} {award.reason_text}")


def _render_what_if_section() -> None:
    st.subheader(modes.SECTION_LABELS["what_if"])
    st.caption(
        "Change one thing about the profile and see which awards open up. "
        "Nothing here is saved — the stored profile is untouched."
    )

    snapshot_path_text = _active_snapshot_path()
    if snapshot_path_text is None:
        st.info("No snapshot available yet. Run an update under Find Scholarships first.")
        return

    stage1_profile = _build_stage1_profile(st.session_state.profile)

    col_gpa, col_hours = st.columns(2)
    with col_gpa:
        gpa = st.slider(
            WHATIF_FIELD_LABELS["gpa"],
            min_value=0.0,
            max_value=4.0,
            value=float(stage1_profile.gpa or 0.0),
            step=0.05,
        )
    with col_hours:
        service_hours = st.slider(
            WHATIF_FIELD_LABELS["service_hours"],
            min_value=0,
            max_value=500,
            value=int(stage1_profile.service_hours or 0),
            step=10,
        )

    col_sat, col_act = st.columns(2)
    with col_sat:
        sat = st.number_input(
            WHATIF_FIELD_LABELS["sat"],
            min_value=0,
            max_value=1600,
            value=int(stage1_profile.sat or 0),
            step=10,
            help="0 means no score yet.",
        )
    with col_act:
        act = st.number_input(
            WHATIF_FIELD_LABELS["act"],
            min_value=0,
            max_value=36,
            value=int(stage1_profile.act or 0),
            step=1,
            help="0 means no score yet.",
        )

    col_first_gen, col_need = st.columns(2)
    with col_first_gen:
        first_gen = st.checkbox(
            WHATIF_FIELD_LABELS["first_gen"], value=bool(stage1_profile.first_gen)
        )
    with col_need:
        financial_need = st.checkbox(
            WHATIF_FIELD_LABELS["financial_need"], value=bool(stage1_profile.financial_need)
        )

    overrides: dict[str, Any] = {
        "gpa": float(gpa),
        "service_hours": int(service_hours),
        "sat": int(sat) or None,
        "act": int(act) or None,
        "first_gen": bool(first_gen),
        "financial_need": bool(financial_need),
    }

    try:
        snapshot_df = _load_snapshot_cached(snapshot_path_text)
        summary = whatif_eligibility(snapshot_df, stage1_profile, overrides)
    except Exception as exc:
        st.error(f"What-if failed: {exc}")
        return

    if not summary.overrides:
        st.info("Nothing changed yet — move a slider or flip a checkbox above.")
        return

    changed_text = ", ".join(
        f"{WHATIF_FIELD_LABELS.get(name, name)} → {value}"
        for name, value in summary.overrides.items()
    )
    st.caption(f"Asking: {changed_text}")

    col_opened, col_closed, col_dollars = st.columns(3)
    col_opened.metric("Newly eligible", len(summary.newly_eligible))
    col_closed.metric("No longer eligible", len(summary.newly_ineligible))
    col_dollars.metric("Dollars unlocked", _money_text(summary.dollars_unlocked))

    if summary.needs_confirmation:
        st.caption(
            f"{summary.needs_confirmation} more award(s) fit this profile but are waiting "
            "on confirmation — nothing you can change opens them."
        )

    if summary.is_noop:
        st.info("That change does not open or close any award in this catalog.")
        return

    if summary.dollars_by_bucket:
        st.markdown("**Dollars unlocked by timeline**")
        for bucket, dollars in summary.dollars_by_bucket.items():
            st.caption(
                f"{TIMELINE_BUCKET_LABELS.get(bucket, bucket)}: {_money_text(dollars)}"
            )

    _render_what_if_award_list(
        "Opens up", summary.newly_eligible, reason_caption="Clears:"
    )
    _render_what_if_award_list(
        "Closes off", summary.newly_ineligible, reason_caption="Blocked by:"
    )


def _render_settings_section() -> None:
    st.subheader(modes.SECTION_LABELS["settings"])
    try:
        with open_db() as conn:
            enabled = repo.get_flag(conn, modes.OPERATOR_ENABLED_SETTING)
            choice = st.checkbox(
                "Show operator tools",
                value=enabled,
                help="Adds an Operator view with the ranking pipeline, ingest and win model controls.",
            )
            if choice != enabled:
                repo.set_flag(conn, modes.OPERATOR_ENABLED_SETTING, choice)
                st.rerun()

            st.divider()
            _render_milestone_settings(conn)
    except Exception as exc:
        st.error(f"Could not open the family database: {exc}")


def _render_pending_section(section: str) -> None:
    st.subheader(modes.SECTION_LABELS[section])
    st.info(_PENDING_SECTION_NOTES[section])


def main() -> None:
    st.set_page_config(page_title="Scholarship Coach", layout="wide")
    st.markdown(phone_width_css(), unsafe_allow_html=True)
    st.title("Scholarship Coach")
    st.caption("Find scholarships matched to your profile")

    _ensure_session_state()

    operator_enabled = _operator_enabled()
    mode = modes.render_mode_selector(operator_enabled=operator_enabled)
    section = modes.render_section_selector(mode)

    with st.sidebar:
        _render_profile_sidebar()
        if modes.show_operator_tools(mode, operator_enabled):
            _render_operator_sidebar()

    st.session_state.profile = _profile_from_widgets()

    if section == "find":
        _render_find_section()
    elif section == "this_week":
        _render_this_week_section()
    elif section == "applications":
        _render_applications_section()
    elif section == "essays":
        _render_essays_section()
    elif section == "recommenders":
        _render_recommenders_section()
    elif section == "timeline":
        _render_timeline_section()
    elif section == "colleges_money":
        _render_colleges_money_section()
    elif section == "catalog_inbox":
        _render_catalog_inbox_section()
    elif section == "what_if":
        _render_what_if_section()
    elif section == "settings":
        _render_settings_section()
    else:
        _render_pending_section(section)


if __name__ == "__main__":
    main()
