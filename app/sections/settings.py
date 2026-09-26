from __future__ import annotations

import calendar
from typing import Any

import streamlit as st

from app import modes
from src.profile.grade_levels import GRADE_SEQUENCE
from src.store import milestones, repo
from src.store.db import open_db


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


def render() -> None:
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
