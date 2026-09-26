from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import streamlit as st

from app import state
from app.helpers import (
    award_count_text,
    days_until_deadline,
    explain_ranked_row,
    format_amount_range,
    needs_date_awards,
    reasons_to_text,
    row_catalog_id,
    timeline_deadline,
    unverified_to_text,
    urgency_indicator,
)
from src.embeddings.cache import ensure_embedding_store_for_df
from src.profile.store import to_stage2_profile
from src.rank.stage1_eligibility import apply_eligibility_filter
from src.rank.stage2_scoring import score_stage2
from src.rank.stage3_rerank import rerank_stage3
from src.rank.timeline import TIMELINE_BUCKET_LABELS, TIMELINE_BUCKETS, classify_timeline
from src.rank.weights import Stage2Weights, Stage3Weights
from src.store import repo, tracker
from src.store.db import open_db
from src.text_utils import coerce_text


def _build_stage2_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return to_stage2_profile(profile)


def _apply_rank_filters(
    df: pd.DataFrame,
    *,
    search_term: str,
    deadline_within_days: int,
    min_amount: float,
    no_essay_only: bool,
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

    if no_essay_only:
        filtered = filtered[filtered["essay_required"].fillna(False).eq(False)]

    return filtered


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
    catalog_id = row_catalog_id(row)
    if not catalog_id:
        st.error("This award has no id to track it by.")
        return
    deadline, _ = timeline_deadline(row, today_value)
    try:
        with open_db() as conn:
            student_id = state.ensure_student(conn)
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
    deadline, deadline_is_projected = timeline_deadline(row, today_value)
    bucket = coerce_text(row.get("timeline_bucket"))
    amount_str = format_amount_range(row.get("amount_min"), row.get("amount_max"))
    amount_not_published = pd.isna(row.get("amount_min")) and pd.isna(row.get("amount_max"))
    source_url = coerce_text(row.get("source_url"))

    days_until = days_until_deadline(deadline, today_value)
    urgency_text, urgency_color = urgency_indicator(days_until)

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
            elif deadline:
                st.text(f"Deadline: {deadline}")
            else:
                st.text("Deadline not on record")
                st.caption("Check the source before you plan around it")
        if bucket:
            st.caption(f"Timeline: {TIMELINE_BUCKET_LABELS.get(bucket, bucket)}")

        st.markdown("**Why this matches you:**")
        for explanation in explain_ranked_row(row, operator_mode=operator_mode):
            st.markdown(f"• {explanation}")

        confirm_text = unverified_to_text(row.get("unverified_axes"))
        if confirm_text:
            st.warning(f"Confirm you meet: {confirm_text}")

        catalog_id = row_catalog_id(row)
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


def render() -> None:
    tuned_weights_payload = None
    try:
        tuned_weights_payload = state.load_weights_profile(
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

    snapshot_path_text = state.active_snapshot_path()
    has_snapshot = snapshot_path_text is not None
    if snapshot_path_text is not None:
        st.info(f"Active snapshot: {Path(snapshot_path_text).name}")
    else:
        st.info(
            "No snapshot available. Use 'Run Update (Ingest)' or 'Use Latest Snapshot' to start."
        )

    st.header("Pipeline Execution")
    st.caption(f"Active ranking weights: {active_weights_label}")
    similarity_mode: Literal["tfidf", "embeddings"] = (
        "embeddings" if st.session_state.get("similarity_mode") == "embeddings" else "tfidf"
    )
    model_name = str(st.session_state.get("embedding_model_name") or state.DEFAULT_MODEL_NAME)
    operator_mode = state.current_mode() == "operator"
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

    run_clicked = st.button("Run Scholarship Coach", disabled=not has_snapshot, type="primary")
    if run_clicked and snapshot_path_text is not None:
        try:
            snapshot_df = state.load_snapshot_cached(snapshot_path_text)
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
                        processed_dir=state.PROCESSED_DIR,
                    )
            stage1_profile = state.build_stage1_profile(st.session_state.profile)
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
                processed_dir=state.PROCESSED_DIR,
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

    final_df = st.session_state.final_df
    if isinstance(final_df, pd.DataFrame):
        filtered_ranked = _apply_rank_filters(
            final_df,
            search_term=search_term,
            deadline_within_days=deadline_within_days,
            min_amount=float(min_amount),
            no_essay_only=no_essay_only,
            today_value=state.effective_today(st.session_state.profile),
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

        today_for_cards = state.effective_today(st.session_state.profile)
        saved_ids = _saved_catalog_ids(state.current_student_id())
        for _, row in top_df.iterrows():
            _render_scholarship_card(
                row, today_for_cards, saved_ids, operator_mode=operator_mode
            )

        waiting = needs_date_awards(st.session_state.get("eligible_df"))
        if not waiting.empty and timeline_choice != "needs_date":
            st.caption(
                f"{award_count_text(len(waiting))} you also qualify for have no deadline "
                "on record. They are under Timeline → "
                f"{TIMELINE_BUCKET_LABELS['needs_date']}."
            )

    ineligible_df = st.session_state.ineligible_df
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
