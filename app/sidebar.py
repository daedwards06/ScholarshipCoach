from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app import state
from scripts.run_ingest import get_latest_snapshot_path, run_ingest
from src.eval.golden_students import get_golden_students
from src.win_model.infer import get_latest_model_path, load_model
from src.win_model.train import train_win_model

SNAPSHOT_DATE_RE = re.compile(r"scholarships_snapshot_(\d{8})\.parquet$")


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
def _load_delta_cached(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        return {"added": [], "removed": [], "changed": []}
    return json.loads(path.read_text(encoding="utf-8"))


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
        item
        for item in source_info.get("details", [])
        if item.get("status") not in {"succeeded", "skipped"}
    ]
    if failed_details:
        st.warning("Some sources failed during ingest.")
        st.dataframe(pd.DataFrame(failed_details), use_container_width=True)
    elif not regressions:
        st.success("All sources succeeded.")


def render_operator_sidebar() -> None:
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
            options=(state.DEFAULT_MODEL_NAME,),
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
            options=state.WEIGHTS_PROFILE_OPTIONS,
            index=state.WEIGHTS_PROFILE_OPTIONS.index(st.session_state.get("weights_profile", "Latest")),
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
            tuned_weights_payload = state.load_weights_profile(selected_weights_profile, custom_weights_path)
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
                    state.PROCESSED_DIR / "win_model",
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
