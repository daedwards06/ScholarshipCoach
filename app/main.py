from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.helpers import explain_ranked_row, format_amount_range, reasons_to_text
from scripts.run_ingest import get_latest_snapshot_path, run_ingest
from src.embeddings.cache import ensure_embedding_store_for_df
from src.rank.stage1_eligibility import StudentProfile, apply_eligibility_filter
from src.rank.stage2_scoring import score_stage2
from src.rank.stage3_rerank import rerank_stage3
from src.rank.weights import Stage2Weights, Stage3Weights

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROFILE_PATH = PROCESSED_DIR / "student_profile.json"
BEST_WEIGHTS_PATH = PROCESSED_DIR / "best_weights.json"
SNAPSHOT_DATE_RE = re.compile(r"scholarships_snapshot_(\d{8})\.parquet$")
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def _default_profile() -> dict[str, Any]:
    return {
        "name": "",
        "gpa": 0.0,
        "state": "",
        "major": "",
        "education_level": "",
        "citizenship": "",
        "profile_keywords": [],
        "goals": "",
        "today_override": date.today().isoformat(),
        "use_today_override": False,
    }


def _ensure_session_state() -> None:
    if "profile" not in st.session_state:
        st.session_state.profile = _default_profile()
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
    st.session_state.setdefault("similarity_mode", "tfidf")
    st.session_state.setdefault("embedding_model_name", DEFAULT_MODEL_NAME)
    _sync_widget_defaults_from_profile(st.session_state.profile)


def _sync_widget_defaults_from_profile(profile: dict[str, Any]) -> None:
    st.session_state.setdefault("profile_name", str(profile.get("name") or ""))
    st.session_state.setdefault("profile_gpa", float(profile.get("gpa") or 0.0))
    st.session_state.setdefault("profile_state", str(profile.get("state") or ""))
    st.session_state.setdefault("profile_major", str(profile.get("major") or ""))
    st.session_state.setdefault(
        "profile_education_level", str(profile.get("education_level") or "")
    )
    st.session_state.setdefault("profile_citizenship", str(profile.get("citizenship") or ""))
    keywords = profile.get("profile_keywords") or []
    st.session_state.setdefault(
        "profile_keywords_csv", ", ".join(str(keyword) for keyword in keywords)
    )
    st.session_state.setdefault("profile_goals", str(profile.get("goals") or ""))
    iso_override = str(profile.get("today_override") or date.today().isoformat())
    st.session_state.setdefault("profile_today_override", date.fromisoformat(iso_override))
    st.session_state.setdefault(
        "profile_use_today_override", bool(profile.get("use_today_override", False))
    )


def _load_profile_from_disk() -> dict[str, Any] | None:
    if not PROFILE_PATH.exists():
        return None
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    defaults = _default_profile()
    defaults.update(payload)
    return defaults


def _save_profile_to_disk(profile: dict[str, Any]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")


def _load_best_weights() -> dict[str, Any] | None:
    if not BEST_WEIGHTS_PATH.exists():
        return None

    payload = json.loads(BEST_WEIGHTS_PATH.read_text(encoding="utf-8"))
    stage2_weights = Stage2Weights.from_mapping(payload.get("stage2_weights"))
    stage3_weights = Stage3Weights.from_mapping(payload.get("stage3_weights"))
    amount_utility_mode = str(payload.get("amount_utility_mode") or "log")
    if amount_utility_mode not in {"linear", "log"}:
        raise ValueError("best_weights.json has an invalid amount_utility_mode.")

    return {
        "stage2_weights": stage2_weights,
        "stage3_weights": stage3_weights,
        "amount_utility_mode": amount_utility_mode,
        "snapshot_used": str(payload.get("snapshot_used") or ""),
        "timestamp": str(payload.get("timestamp") or ""),
    }


def _apply_profile_to_widgets(profile: dict[str, Any]) -> None:
    st.session_state.profile_name = str(profile.get("name") or "")
    st.session_state.profile_gpa = float(profile.get("gpa") or 0.0)
    st.session_state.profile_state = str(profile.get("state") or "")
    st.session_state.profile_major = str(profile.get("major") or "")
    st.session_state.profile_education_level = str(profile.get("education_level") or "")
    st.session_state.profile_citizenship = str(profile.get("citizenship") or "")
    keywords = profile.get("profile_keywords") or []
    st.session_state.profile_keywords_csv = ", ".join(str(keyword) for keyword in keywords)
    st.session_state.profile_goals = str(profile.get("goals") or "")
    st.session_state.profile_use_today_override = bool(profile.get("use_today_override", False))
    iso_override = str(profile.get("today_override") or date.today().isoformat())
    st.session_state.profile_today_override = date.fromisoformat(iso_override)


def _profile_from_widgets() -> dict[str, Any]:
    keywords = [
        keyword.strip()
        for keyword in str(st.session_state.get("profile_keywords_csv") or "").split(",")
        if keyword.strip()
    ]
    today_override = st.session_state.get("profile_today_override", date.today())
    return {
        "name": str(st.session_state.get("profile_name") or ""),
        "gpa": float(st.session_state.get("profile_gpa") or 0.0),
        "state": str(st.session_state.get("profile_state") or ""),
        "major": str(st.session_state.get("profile_major") or ""),
        "education_level": str(st.session_state.get("profile_education_level") or ""),
        "citizenship": str(st.session_state.get("profile_citizenship") or ""),
        "profile_keywords": keywords,
        "goals": str(st.session_state.get("profile_goals") or ""),
        "today_override": today_override.isoformat(),
        "use_today_override": bool(st.session_state.get("profile_use_today_override", False)),
    }


def _effective_today(profile: dict[str, Any]) -> date:
    if profile.get("use_today_override"):
        return date.fromisoformat(str(profile.get("today_override")))
    return date.today()


def _build_stage1_profile(profile: dict[str, Any]) -> StudentProfile:
    return StudentProfile(
        gpa=float(profile.get("gpa") or 0.0),
        state=profile.get("state"),
        major=profile.get("major"),
        education_level=profile.get("education_level"),
        citizenship=profile.get("citizenship"),
        today=_effective_today(profile),
    )


def _build_stage2_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "major": profile.get("major"),
        "keywords": profile.get("profile_keywords") or [],
        "interests": profile.get("profile_keywords") or [],
        "goals": profile.get("goals") or "",
        "extracurriculars": [],
    }


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
    failed_details = [
        item for item in source_info.get("details", []) if item.get("status") != "succeeded"
    ]
    if failed_details:
        st.warning("Some sources failed during ingest.")
        st.dataframe(pd.DataFrame(failed_details), use_container_width=True)
    else:
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


def main() -> None:
    st.set_page_config(page_title="Scholarship Coach", layout="wide")
    st.title("Scholarship Coach")
    st.caption("Ingest -> Snapshot/Delta -> Stage 1 -> Stage 2 -> Stage 3")

    _ensure_session_state()

    with st.sidebar:
        st.header("Student Profile")
        st.text_input("name (optional)", key="profile_name")
        st.number_input("gpa", min_value=0.0, max_value=4.0, step=0.01, key="profile_gpa")
        st.text_input("state", key="profile_state")
        st.text_input("major", key="profile_major")
        st.text_input("education_level", key="profile_education_level")
        st.text_input("citizenship", key="profile_citizenship")
        st.text_input(
            "profile_keywords (comma-separated)",
            key="profile_keywords_csv",
        )
        st.text_area("goals/free_text", key="profile_goals", height=120)
        st.checkbox("Use today override", key="profile_use_today_override")
        st.date_input("today override", key="profile_today_override")

        save_col, load_col = st.columns(2)
        if save_col.button("Save Profile", use_container_width=True):
            profile = _profile_from_widgets()
            _save_profile_to_disk(profile)
            st.session_state.profile = profile
            st.success(f"Saved to {PROFILE_PATH}")
        if load_col.button("Load Profile", use_container_width=True):
            loaded = _load_profile_from_disk()
            if loaded is None:
                st.warning(f"No profile file found at {PROFILE_PATH}")
            else:
                st.session_state.profile = loaded
                _apply_profile_to_widgets(loaded)
                st.success("Profile loaded.")
                st.rerun()

        tuned_weights_payload: dict[str, Any] | None = None
        tuned_weights_error: str | None = None
        try:
            tuned_weights_payload = _load_best_weights()
        except Exception as exc:
            tuned_weights_error = str(exc)

        st.divider()
        st.header("Similarity")
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

        st.divider()
        st.header("Ranking Weights")
        if "use_tuned_weights" not in st.session_state:
            st.session_state.use_tuned_weights = tuned_weights_payload is not None
        use_tuned_weights = st.checkbox(
            "Use tuned weights (best_weights.json)",
            key="use_tuned_weights",
            disabled=tuned_weights_payload is None,
        )
        if tuned_weights_error:
            st.warning(f"Could not load {BEST_WEIGHTS_PATH.name}: {tuned_weights_error}")
        elif tuned_weights_payload is None:
            st.caption(f"No tuned weights found at {BEST_WEIGHTS_PATH}. Using baseline weights.")
        else:
            st.caption(f"Loaded tuned weights from {BEST_WEIGHTS_PATH.name}.")

    st.session_state.profile = _profile_from_widgets()

    tuned_weights_payload = None
    try:
        tuned_weights_payload = _load_best_weights()
    except Exception:
        tuned_weights_payload = None

    use_tuned_weights = bool(st.session_state.get("use_tuned_weights")) and tuned_weights_payload is not None
    if use_tuned_weights and tuned_weights_payload is not None:
        active_stage2_weights = tuned_weights_payload["stage2_weights"]
        active_stage3_weights = tuned_weights_payload["stage3_weights"]
        active_amount_utility_mode = tuned_weights_payload["amount_utility_mode"]
        active_weights_label = "tuned"
    else:
        active_stage2_weights = Stage2Weights.baseline()
        active_stage3_weights = Stage3Weights.baseline()
        active_amount_utility_mode = "log"
        active_weights_label = "baseline"

    st.header("Data Update + Status")
    update_col, latest_col = st.columns(2)
    if update_col.button("Run Update (Ingest)", type="primary", use_container_width=True):
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
    status_payload: dict[str, Any] = {"similarity_mode": similarity_mode}
    if similarity_mode == "embeddings":
        status_payload["model_name"] = model_name
    st.write(status_payload)
    st.json(
        _weights_display_payload(
            active_stage2_weights,
            active_stage3_weights,
            active_amount_utility_mode,
        )
    )
    top_n = st.slider("Top-N results", min_value=10, max_value=100, value=25, step=5)
    search_term = st.text_input("Search title/sponsor/description")
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
                weights=active_stage3_weights,
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
        table_columns = [
            "scholarship_id",
            "title",
            "sponsor",
            "deadline",
            "amount",
            "final_score",
            "source_url",
        ]
        available_columns = [column for column in table_columns if column in top_df.columns]
        st.dataframe(top_df[available_columns], use_container_width=True)

        st.subheader("Ranked Detail + Explainability")
        for _, row in top_df.iterrows():
            title = str(row.get("title") or row.get("scholarship_id"))
            with st.expander(title):
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
                component_values = {
                    column: float(row.get(column))
                    for column in component_columns
                    if column in row and pd.notna(row.get(column))
                }
                st.write("Component scores")
                st.json(component_values)
                st.write(
                    {
                        "deadline": str(row.get("deadline") or ""),
                        "amount": format_amount_range(row.get("amount_min"), row.get("amount_max")),
                        "sponsor": str(row.get("sponsor") or ""),
                        "source_url": str(row.get("source_url") or ""),
                    }
                )
                st.write("Why ranked")
                for explanation in explain_ranked_row(row):
                    st.write(f"- {explanation}")

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

        excluded_filtered["amount"] = excluded_filtered.apply(
            lambda row: format_amount_range(row.get("amount_min"), row.get("amount_max")),
            axis=1,
        )
        excluded_filtered["reasons_text"] = excluded_filtered["reasons"].apply(reasons_to_text)
        excluded_columns = ["title", "deadline", "amount", "reasons_text"]
        available_excluded = [
            column for column in excluded_columns if column in excluded_filtered.columns
        ]
        st.dataframe(excluded_filtered[available_excluded], use_container_width=True)


if __name__ == "__main__":
    main()
