from __future__ import annotations

from typing import Any

import streamlit as st

from app import modes, state
from app.helpers import money_text
from src.rank.timeline import TIMELINE_BUCKET_LABELS
from src.rank.whatif import WHATIF_FIELD_LABELS, WhatIfAward, whatif_eligibility


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
                st.caption(f"{money_text(award.amount)} · {bucket_label}")
            with columns[1]:
                if award.reason_text:
                    st.caption(f"{reason_caption} {award.reason_text}")


def render() -> None:
    st.subheader(modes.SECTION_LABELS["what_if"])
    st.caption(
        "Change one thing about the profile and see which awards open up. "
        "Nothing here is saved — the stored profile is untouched."
    )

    snapshot_path_text = state.active_snapshot_path()
    if snapshot_path_text is None:
        st.info("No snapshot available yet. Run an update under Find Scholarships first.")
        return

    stage1_profile = state.build_stage1_profile(st.session_state.profile)

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
        snapshot_df = state.load_snapshot_cached(snapshot_path_text)
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
    col_dollars.metric("Dollars unlocked", money_text(summary.dollars_unlocked))

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
                f"{TIMELINE_BUCKET_LABELS.get(bucket, bucket)}: {money_text(dollars)}"
            )

    _render_what_if_award_list(
        "Opens up", summary.newly_eligible, reason_caption="Clears:"
    )
    _render_what_if_award_list(
        "Closes off", summary.newly_ineligible, reason_caption="Blocked by:"
    )
