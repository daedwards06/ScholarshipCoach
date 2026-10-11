from __future__ import annotations

from typing import Any

import streamlit as st

from app import modes, state
from app.helpers import BadgeColor, markdown_safe, money_text
from app.sections import todos as todos_page
from src.rank.stage1_eligibility import StudentProfile
from src.rank.whatif import WHATIF_FIELD_LABELS, WhatIfAward, WhatIfSummary, whatif_eligibility
from src.store import repo, requirements
from src.store.db import open_db
from src.store.repo import CollegeRequirement

# My Path links here with ``?goal=<requirement id>`` to start at that target.
GOAL_QUERY_PARAM = "goal"

# Award cards shown before the rest fold into an expander, so the controls stay near.
SHOWN_AWARDS = 3

_GOAL_BADGES: dict[bool, tuple[str, BadgeColor]] = {
    True: ("Reached", "green"),
    False: ("Not yet", "gray"),
}


def award_amount_text(amount: float | None) -> str:
    """An award's listed amount; an award with no amount never reads as "$0"."""
    return "Amount not listed" if not amount else money_text(amount)


def goals_heading(role: str) -> str:
    return "Your goals" if role == "student" else "Her goals"


def preset_values(
    items: list[CollegeRequirement], goal_param: str | None
) -> dict[str, float]:
    """The starting value My Path's link asks for: ``{"gpa": 3.5}`` for a GPA row."""
    try:
        goal_id = int(str(goal_param or ""))
    except ValueError:
        return {}
    for item in items:
        if item.id != goal_id:
            continue
        measure = requirements.goal_measure(item)
        target = requirements.target_number(item.target)
        if measure is not None and target is not None:
            return {measure: target}
    return {}


def asked_overrides(profile: StudentProfile, values: dict[str, Any]) -> dict[str, Any]:
    """Drop a control still at the value a blank profile field starts it at.

    A widget cannot show "blank", so an unset GPA or service-hours field
    starts at 0; leaving it there is not a question about 0.
    """
    return {
        name: value
        for name, value in values.items()
        if not (getattr(profile, name, None) is None and not value)
    }


def _load_goal_rows() -> list[CollegeRequirement]:
    with open_db() as conn:
        college = repo.first_target_college(conn, state.ensure_student(conn))
        return [] if college is None else repo.list_requirements(conn, college.id)


def _render_goals(
    items: list[CollegeRequirement], values: dict[str, float | None], role: str
) -> None:
    goals = requirements.goal_reaches(items, values)
    if not goals:
        return
    st.subheader(goals_heading(role))
    for goal in goals:
        label, color = _GOAL_BADGES[goal.reached]
        with st.container(horizontal=True, gap="small", vertical_alignment="center"):
            st.badge(label, color=color)
            st.markdown(markdown_safe(requirements.goal_reach_text(goal)))


def _render_award(award: WhatIfAward, caption: str) -> None:
    with st.container(border=True, gap="small"):
        st.markdown(f"**{markdown_safe(award.title or award.scholarship_id)}**")
        st.caption(f"{award_amount_text(award.amount)} · {award.when_text}")
        if caption:
            st.caption(caption)


def _render_opened(awards: list[WhatIfAward]) -> None:
    for award in awards[:SHOWN_AWARDS]:
        _render_award(award, award.opened_by_text)
    rest = awards[SHOWN_AWARDS:]
    if rest:
        with st.expander(f"{len(rest)} more"):
            for award in rest:
                _render_award(award, award.opened_by_text)


def _render_awards(summary: WhatIfSummary) -> None:
    if not summary.overrides:
        st.caption("Move a slider below to see which awards open up.")
        return
    if summary.is_noop:
        st.caption("That change doesn't open or close any award.")
    else:
        count = len(summary.newly_eligible)
        if count:
            st.subheader("Awards that open up")
            listed = summary.dollars_unlocked
            dollars = f" · {money_text(listed)} listed" if listed else ""
            st.caption(f"{count} award{'s' if count != 1 else ''}{dollars}")
            _render_opened(summary.newly_eligible)
        if summary.newly_ineligible:
            with st.expander(f"Awards that close ({len(summary.newly_ineligible)})"):
                for award in summary.newly_ineligible:
                    reason = f"Closes because: {award.reason_text}" if award.reason_text else ""
                    _render_award(award, reason)
    if summary.needs_confirmation:
        st.caption(
            f"{summary.needs_confirmation} more award(s) fit but are waiting on "
            "confirmation; nothing you change here opens them."
        )


def render() -> None:
    modes.render_page_header("what_if")
    role = todos_page.viewer_role(state.current_mode())
    stage1_profile = state.build_stage1_profile(st.session_state.profile)

    try:
        items = _load_goal_rows()
    except Exception as exc:
        items = []
        st.caption(f"Could not load the school's goals: {exc}")
    preset = preset_values(items, st.query_params.get(GOAL_QUERY_PARAM))

    # Results are drawn into these before the controls so they lead on a phone.
    goals_area = st.container()
    awards_area = st.container()

    st.markdown("**Try a change**")
    col_gpa, col_hours = st.columns(2)
    with col_gpa:
        gpa = st.slider(
            WHATIF_FIELD_LABELS["gpa"],
            min_value=0.0,
            max_value=4.0,
            value=float(preset.get("gpa", stage1_profile.gpa or 0.0)),
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
            value=int(preset.get("sat", stage1_profile.sat or 0)),
            step=10,
            help="0 means no score yet.",
        )
    with col_act:
        act = st.number_input(
            WHATIF_FIELD_LABELS["act"],
            min_value=0,
            max_value=36,
            value=int(preset.get("act", stage1_profile.act or 0)),
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

    with goals_area:
        _render_goals(
            items, {"gpa": float(gpa), "sat": float(sat), "act": float(act)}, role
        )

    overrides = asked_overrides(
        stage1_profile,
        {
            "gpa": float(gpa),
            "service_hours": int(service_hours),
            "sat": int(sat) or None,
            "act": int(act) or None,
            "first_gen": bool(first_gen),
            "financial_need": bool(financial_need),
        },
    )

    with awards_area:
        snapshot_path_text = state.active_snapshot_path()
        if snapshot_path_text is None:
            st.caption("No award list yet, so only the goals above can be checked.")
            return
        try:
            snapshot_df = state.load_snapshot_cached(snapshot_path_text)
            summary = whatif_eligibility(snapshot_df, stage1_profile, overrides)
        except Exception as exc:
            st.error(f"What-if failed: {exc}")
            return
        _render_awards(summary)
