from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import streamlit as st

from app import identity, modes, state
from app.components import render_empty_state
from app.helpers import (
    award_count_text,
    badge_markdown,
    days_until_deadline,
    deadline_badge,
    effort_badges,
    fit_reasons,
    format_amount_range,
    friendly_date,
    markdown_safe,
    needs_date_awards,
    reasons_to_text,
    row_catalog_id,
    timeline_deadline,
    unverified_to_text,
)
from app.sections.todos import author_for, viewer_role
from src.embeddings.cache import ensure_embedding_store_for_df
from src.profile.store import to_stage1_profile, to_stage2_profile
from src.rank.kinds import scholarships_only
from src.rank.stage1_eligibility import apply_eligibility_filter
from src.rank.stage2_scoring import score_stage2
from src.rank.stage3_rerank import is_local_award, rerank_stage3
from src.rank.timeline import TIMELINE_BUCKET_LABELS, TIMELINE_BUCKETS, classify_timeline
from src.rank.weights import Stage2Weights, Stage3Weights
from src.store import repo, todos, tracker
from src.store.db import open_db
from src.text_utils import coerce_text

PAGE_SIZE = 25
SHOWN_STATE_KEY = "find_shown"

# "Open now" includes the awards that name no grade: they are open, but she has
# to check the grade before applying, and the card says so.
OPEN_NOW = "open"
OPEN_NOW_BUCKETS: tuple[str, ...] = ("now", "check_grade")
WHEN_OPTIONS: tuple[str, ...] = (
    OPEN_NOW,
    *(bucket for bucket in TIMELINE_BUCKETS if bucket not in OPEN_NOW_BUCKETS),
    "all",
)


def when_label(choice: str) -> str:
    if choice == OPEN_NOW:
        return "Open now"
    if choice == "all":
        return "Any time"
    return TIMELINE_BUCKET_LABELS.get(choice, choice)


def when_buckets(choice: str) -> tuple[str, ...] | None:
    """The timeline buckets a "When can she apply" choice ranks; ``None`` is all."""
    if choice == OPEN_NOW:
        return OPEN_NOW_BUCKETS
    if choice == "all":
        return None
    return (choice,)


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


def filters_to_relax(
    *,
    search_term: str,
    when_choice: str,
    deadline_within_days: int,
    min_amount: float,
    no_essay_only: bool,
) -> list[str]:
    """The filters in use, named as the Filters panel names them."""
    active = []
    if search_term.strip():
        active.append("search")
    if when_choice != "all":
        active.append("when she can apply")
    if deadline_within_days > 0:
        active.append("deadline window")
    if min_amount > 0:
        active.append("minimum amount")
    if no_essay_only:
        active.append("no-essay only")
    return active


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


@st.cache_data(show_spinner=False, max_entries=16)
def rank_snapshot(
    snapshot_path_text: str,
    profile: dict[str, Any],
    today_value: date,
    buckets: tuple[str, ...] | None,
    stage2_weights: dict[str, float],
    stage3_weights: dict[str, float],
    amount_utility_mode: Literal["linear", "log"],
    similarity_mode: Literal["tfidf", "embeddings"],
    model_name: str,
    include_unconfirmed: bool,
    use_win_model: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stages 1-3 over the snapshot: ``(eligible, ineligible, ranked)``.

    Cached on the profile, snapshot and date, so opening Find shows results at
    once after the first run and re-ranks only when one of them changes.
    Programs are left out from the start, so they never show on Timeline or in
    the no-deadline count either: they live on Opportunities.
    """
    snapshot_df = scholarships_only(state.load_snapshot_cached(snapshot_path_text))
    if similarity_mode == "embeddings":
        snapshot_df = ensure_embedding_store_for_df(
            snapshot_df, model_name, processed_dir=state.PROCESSED_DIR
        )
    stage1_profile = to_stage1_profile(profile, today=today_value)
    eligible_df, ineligible_df = apply_eligibility_filter(
        snapshot_df, stage1_profile, include_unconfirmed=include_unconfirmed
    )
    eligible_df = classify_timeline(eligible_df, stage1_profile)
    scored_df = score_stage2(
        eligible_df,
        _build_stage2_profile(profile),
        weights=Stage2Weights.from_mapping(stage2_weights),
        amount_utility_mode=amount_utility_mode,
        similarity_mode=similarity_mode,
        model_name=model_name,
        processed_dir=state.PROCESSED_DIR,
    )
    final_df = rerank_stage3(
        scored_df,
        today=stage1_profile.today,
        profile=stage1_profile,
        timeline_bucket=buckets,
        weights=Stage3Weights.from_mapping(stage3_weights),
        use_win_model=use_win_model,
    )
    return eligible_df, ineligible_df, final_df


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


def _tracked_catalog_ids(student_id: str) -> tuple[set[str], set[str]]:
    """``(saved, suggested)``: catalog ids already saved, and already suggested to her."""
    try:
        with open_db() as conn:
            saved = {
                application.catalog_id
                for application in repo.list_applications(conn, student_id)
            }
            suggested = {
                todo.source_ref
                for todo in todos.list_todos(conn, student_id)
                if todo.source_kind == "application"
            }
    except Exception:
        return set(), set()
    return saved, suggested


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
    st.toast("Saved to Applications." if created else "Already in Applications.")
    st.rerun()


def _suggest_award_to_her(row: pd.Series, today_value: date) -> None:
    """Put the award on her To-dos, dated by its deadline when it has a real one."""
    catalog_id = row_catalog_id(row)
    if not catalog_id:
        st.error("This award has no id to suggest it by.")
        return
    deadline, projected = timeline_deadline(row, today_value)
    days_until = days_until_deadline(deadline, today_value)
    due_on = deadline[:10] if days_until is not None and days_until >= 0 and not projected else None
    role = viewer_role(state.current_mode())
    created_by, created_role = author_for(identity.current_identity(), role)
    try:
        with open_db() as conn:
            student_id = state.ensure_student(conn)
            if todos.find_todo(conn, student_id, "application", catalog_id) is None:
                todos.create_todo(
                    conn,
                    student_id,
                    f"Look at {coerce_text(row.get('title')) or catalog_id}",
                    notes=coerce_text(row.get("source_url")),
                    due_on=due_on,
                    assignee="student",
                    created_by=created_by,
                    created_role=created_role,
                    source_kind="application",
                    source_ref=catalog_id,
                )
    except Exception as exc:
        st.error(f"Could not add the to-do: {exc}")
        return
    st.toast("Added to her to-dos.")
    st.rerun()


def award_badges(row: pd.Series, today_value: date) -> list[str]:
    """The card's badge row: deadline, when it opens, effort, local, what to confirm."""
    deadline, projected = timeline_deadline(row, today_value)
    days_until = days_until_deadline(deadline, today_value)
    deadline_date = None if days_until is None else today_value + timedelta(days=days_until)
    when = deadline_badge(days_until, projected=projected, deadline=deadline_date, owed=False)
    badges = [badge_markdown(when.label, when.icon, when.color)]

    bucket = coerce_text(row.get("timeline_bucket"))
    if bucket and bucket not in (*OPEN_NOW_BUCKETS, "needs_date"):
        badges.append(
            badge_markdown(
                TIMELINE_BUCKET_LABELS.get(bucket, bucket), ":material/schedule:", "gray"
            )
        )
    badges.extend(
        badge_markdown(label, ":material/edit_note:", "gray") for label in effort_badges(row)
    )
    if is_local_award(row):
        badges.append(badge_markdown("Local", ":material/location_on:", "violet"))
    confirm_text = unverified_to_text(row.get("unverified_axes"))
    if confirm_text:
        badges.append(
            badge_markdown(f"Confirm you meet: {confirm_text}", ":material/warning:", "yellow")
        )
    return badges


def _render_award_details(row: pd.Series, today_value: date, *, operator_mode: bool) -> None:
    # The reasons live here, not on the card face: at 400px they wrap to two
    # lines, and the badges already carry effort, local and what to confirm.
    with st.expander("Why it fits & details"):
        st.markdown(
            "**Why it fits:** "
            + " · ".join(fit_reasons(row, operator_mode=operator_mode))
        )
        description = coerce_text(row.get("description"))
        if description:
            st.markdown(markdown_safe(description))
        deadline, projected = timeline_deadline(row, today_value)
        deadline_date = None
        days_until = days_until_deadline(deadline, today_value)
        if days_until is not None:
            deadline_date = today_value + timedelta(days=days_until)
        if deadline_date is not None and projected:
            st.caption(
                f"Next deadline ~{friendly_date(deadline_date, today_value)}, "
                "projected from this award's usual cycle."
            )
        elif deadline_date is not None:
            st.caption(f"Deadline {friendly_date(deadline_date, today_value)}.")
        else:
            st.caption("No deadline on record. Check the source before planning around it.")
        if pd.isna(row.get("amount_min")) and pd.isna(row.get("amount_max")):
            st.caption("Amount not published, so it is ranked on fit alone.")
        if operator_mode:
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
                *_WIN_MODEL_COLUMNS,
            ]
            st.json(
                {
                    column: float(row.get(column))
                    for column in component_columns
                    if column in row and pd.notna(row.get(column))
                }
            )


def _render_award_card(
    row: pd.Series,
    today_value: date,
    saved_ids: set[str],
    suggested_ids: set[str],
    *,
    operator_mode: bool = False,
) -> None:
    """MASTER "Award card" in its Parent form: the money is the headline."""
    title = (
        coerce_text(row.get("title")) or coerce_text(row.get("scholarship_id")) or "Untitled"
    )
    sponsor = coerce_text(row.get("sponsor"))
    source_url = coerce_text(row.get("source_url"))
    catalog_id = row_catalog_id(row)
    already_saved = catalog_id in saved_ids

    amount = format_amount_range(row.get("amount_min"), row.get("amount_max"))
    with st.container(border=True, key=f"award_{catalog_id}", gap="xsmall"):
        # Title and sponsor share one element: at 400px every element costs a gap.
        sponsor_text = markdown_safe(sponsor).replace("[", "(").replace("]", ")")
        sponsor_line = f"  \n:small[{sponsor_text}]" if sponsor else ""
        st.markdown(f"**{markdown_safe(title)}**{sponsor_line}")
        st.markdown(f"### {markdown_safe(amount)}")
        st.markdown(" ".join(award_badges(row, today_value)))
        with st.container(horizontal=True):
            if already_saved:
                st.badge("Saved", icon=":material/check_circle:", color="green")
            elif st.button("Save", key=f"save_{catalog_id}", type="primary"):
                _save_award_from_card(row, today_value)
            if source_url:
                st.link_button(
                    "Apply",
                    source_url,
                    icon=":material/open_in_new:",
                    type="primary" if already_saved else "secondary",
                )
            if catalog_id in suggested_ids:
                st.badge("Suggested", icon=":material/send:", color="violet")
            elif st.button("Suggest to her", key=f"suggest_{catalog_id}"):
                _suggest_award_to_her(row, today_value)
        _render_award_details(row, today_value, operator_mode=operator_mode)


def _render_excluded(ineligible_df: pd.DataFrame) -> None:
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
        reasons_text = reasons_to_text(row.get("reasons"))

        with st.container(border=True):
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                st.markdown(f"**{markdown_safe(title)}**")
            with col2:
                st.text(f"Deadline: {deadline}")

            col_amt, col_reason = st.columns(2)
            with col_amt:
                st.text(f"Award: {amount_str}")
            with col_reason:
                st.text(f"Reason: {reasons_text}")


def render() -> None:
    modes.render_page_header("find")
    operator_mode = state.current_mode() == "operator"
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

    similarity_mode: Literal["tfidf", "embeddings"] = (
        "embeddings" if st.session_state.get("similarity_mode") == "embeddings" else "tfidf"
    )
    model_name = str(st.session_state.get("embedding_model_name") or state.DEFAULT_MODEL_NAME)
    use_win_model = operator_mode and bool(st.session_state.get("use_win_model"))
    include_unconfirmed = operator_mode and bool(st.session_state.get("include_unconfirmed"))
    snapshot_path_text = state.active_snapshot_path()

    top_n = PAGE_SIZE
    if operator_mode:
        if snapshot_path_text is not None:
            st.info(f"Active snapshot: {Path(snapshot_path_text).name}")
        else:
            st.info(
                "No snapshot available. Use 'Run Update (Ingest)' or 'Use Latest Snapshot' to start."
            )
        st.header("Pipeline Execution")
        st.caption(f"Active ranking weights: {active_weights_label}")
        if tuned_weights_payload is not None and tuned_weights_payload.get("use_win_model") and not use_win_model:
            st.warning(
                "The selected weights profile was tuned with the win model enabled, but 'Use Win Model in Ranking' is off."
            )
        top_n = st.slider("Top-N results", min_value=10, max_value=100, value=25, step=5)

    if snapshot_path_text is None:
        render_empty_state(
            "Matches show here once the award catalog has been built.",
            next_up="Add or confirm awards in Catalog & Inbox, then rebuild the catalog.",
            link_page=st.session_state.get(modes.NAV_PAGES_STATE_KEY, {}).get("catalog_inbox"),
            link_label="Open Catalog & Inbox",
        )
        return

    with st.expander("Filters", icon=":material/tune:"):
        search_term = st.text_input("Search title, sponsor or description")
        when_choice = str(
            st.selectbox("When can she apply", WHEN_OPTIONS, index=0, format_func=when_label)
        )
        deadline_within_days = (
            st.slider("Deadline within X days (0 = no filter)", 0, 365, 0, 5)
            if operator_mode
            else 0
        )
        min_amount = float(
            st.number_input("Minimum amount", min_value=0.0, value=0.0, step=500.0)
        )
        no_essay_only = st.checkbox("No essay required", value=False)

    if similarity_mode == "embeddings":
        snapshot_columns = state.load_snapshot_cached(snapshot_path_text).columns
        if "embedding_key" not in snapshot_columns:
            st.info(
                "This snapshot predates persisted embedding keys. "
                "Embedding keys and vectors will be computed locally and cached for this run."
            )
    today_value = state.effective_today(st.session_state.profile)
    try:
        with st.spinner("Finding matches…"):
            eligible_df, ineligible_df, final_df = rank_snapshot(
                snapshot_path_text,
                dict(st.session_state.profile),
                today_value,
                when_buckets(when_choice),
                active_stage2_weights.to_dict(),
                active_stage3_weights.to_dict(),
                active_amount_utility_mode,
                similarity_mode,
                model_name,
                include_unconfirmed,
                use_win_model,
            )
    except Exception as exc:
        st.error(f"Could not rank the awards: {exc}")
        return

    # Timeline and What If read the last ranking from here.
    st.session_state.eligible_df = eligible_df
    st.session_state.final_df = final_df
    st.session_state.ineligible_df = ineligible_df

    filtered = _apply_rank_filters(
        final_df,
        search_term=search_term,
        deadline_within_days=deadline_within_days,
        min_amount=min_amount,
        no_essay_only=no_essay_only,
        today_value=today_value,
    )
    if operator_mode:
        st.caption(f"eligible={len(eligible_df)} ineligible={len(ineligible_df)}")

    count = len(filtered)
    st.subheader(f"{count} match{'es' if count != 1 else ''} for her")
    if filtered.empty:
        relax = filters_to_relax(
            search_term=search_term,
            when_choice=when_choice,
            deadline_within_days=deadline_within_days,
            min_amount=min_amount,
            no_essay_only=no_essay_only,
        )
        render_empty_state(
            "No awards match these filters."
            + (f" Try relaxing: {', '.join(relax)}." if relax else ""),
        )
    else:
        shown = top_n if operator_mode else int(st.session_state.get(SHOWN_STATE_KEY, PAGE_SIZE))
        top_df = filtered.head(shown)

        win_summary = _topk_win_model_summary(top_df) if operator_mode else None
        if win_summary is not None:
            with st.expander(
                "Win model summary", icon=":material/analytics:", expanded=False
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean P(Win)", f"{round(win_summary['mean_p_win'], 4)}")
                    st.metric("Mean Expected Value", f"${round(win_summary['mean_expected_value'], 2):,.0f}")
                with col2:
                    st.metric("Median P(Win)", f"{round(win_summary['median_p_win'], 4)}")
                    st.metric("Median Expected Value", f"${round(win_summary['median_expected_value'], 2):,.0f}")

        saved_ids, suggested_ids = _tracked_catalog_ids(state.current_student_id())
        for _, row in top_df.iterrows():
            _render_award_card(
                row, today_value, saved_ids, suggested_ids, operator_mode=operator_mode
            )
        if not operator_mode and count > shown:
            if st.button(f"Show more ({count - shown} left)", key="find_show_more"):
                st.session_state[SHOWN_STATE_KEY] = shown + PAGE_SIZE
                st.rerun()

    waiting = needs_date_awards(eligible_df)
    if not waiting.empty and when_choice != "needs_date":
        st.caption(
            f"{award_count_text(len(waiting))} she also qualifies for "
            f"{'has' if len(waiting) == 1 else 'have'} no deadline on record. "
            "They are under Timeline → "
            f"{TIMELINE_BUCKET_LABELS['needs_date']}."
        )

    if operator_mode and not ineligible_df.empty:
        _render_excluded(ineligible_df)
