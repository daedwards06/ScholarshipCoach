from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app import modes
from app.helpers import explain_ranked_row, format_amount_range, reasons_to_text
from scripts.run_ingest import get_latest_snapshot_path, run_ingest
from src.embeddings.cache import ensure_embedding_store_for_df
from src.eval.golden_students import get_golden_students
from src.profile.grade_levels import (
    GRADE_LABELS,
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
from src.store import repo, tracker
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
    "essays": "Your essay bank, themed and reused across prompts.",
    "recommenders": "Who you asked for letters, when, and what is still outstanding.",
    "catalog_inbox": "Add an award from a URL, and review proposed catalog changes.",
    "timeline": "Deadlines, milestones and letter due dates by month.",
    "colleges_money": "College list, net price estimates, and what has been won.",
    "outcomes": "Results per application: awarded amounts and renewal terms.",
}
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
    row: pd.Series, today_value: date, saved_ids: set[str] | None = None
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
        for explanation in explain_ranked_row(row):
            st.markdown(f"• {explanation}")

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
                "p_win",
                "expected_value",
                "expected_value_norm",
                "final_score",
            ]
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

    snapshot_path_text = st.session_state.latest_snapshot_path
    if snapshot_path_text is None:
        latest = None
        try:
            latest = get_latest_snapshot_path()
        except FileNotFoundError:
            latest = None
        if latest is not None:
            snapshot_path_text = str(latest.resolve())
            st.session_state.latest_snapshot_path = snapshot_path_text

    has_snapshot = snapshot_path_text is not None and Path(snapshot_path_text).exists()
    if has_snapshot:
        st.info(f"Active snapshot: {Path(snapshot_path_text).name}")
    else:
        st.info(
            "No snapshot available. Use 'Run Update (Ingest)' or 'Use Latest Snapshot' to start."
        )

    st.header("Pipeline Execution")
    st.caption(f"Active ranking weights: {active_weights_label}")
    similarity_mode = str(st.session_state.get("similarity_mode") or "tfidf")
    model_name = str(st.session_state.get("embedding_model_name") or DEFAULT_MODEL_NAME)
    use_win_model = bool(st.session_state.get("use_win_model"))
    if tuned_weights_payload is not None and tuned_weights_payload.get("use_win_model") and not use_win_model:
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

            eligible_df, ineligible_df = apply_eligibility_filter(snapshot_df, stage1_profile)
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
        win_summary = _topk_win_model_summary(top_df)
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
            _render_scholarship_card(row, today_for_cards, saved_ids)

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


def _render_checklist(conn: Any, application: repo.Application) -> None:
    items = repo.list_checklist_items(conn, application.id)
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


def _operator_enabled() -> bool:
    try:
        with open_db() as conn:
            return repo.get_flag(conn, modes.OPERATOR_ENABLED_SETTING)
    except Exception:
        return False


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
    except Exception as exc:
        st.error(f"Could not open the family database: {exc}")


def _render_pending_section(section: str) -> None:
    st.subheader(modes.SECTION_LABELS[section])
    st.info(_PENDING_SECTION_NOTES[section])


def main() -> None:
    st.set_page_config(page_title="Scholarship Coach", layout="wide")
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
    elif section == "settings":
        _render_settings_section()
    else:
        _render_pending_section(section)


if __name__ == "__main__":
    main()
