from __future__ import annotations

from datetime import timedelta

import streamlit as st

from app import modes, state
from app.helpers import deadline_badge, friendly_date, parse_date
from src.store import tracker
from src.store.db import open_db


def render() -> None:
    st.subheader(modes.SECTION_LABELS["this_week"])
    today_value = state.effective_today(st.session_state.profile)
    st.caption(
        "Due on or before "
        f"{friendly_date(today_value + timedelta(days=tracker.THIS_WEEK_DAYS), today_value)}, "
        "plus anything already overdue."
    )
    try:
        with open_db() as conn:
            student_id = state.ensure_student(conn)
            due_items = tracker.this_week(conn, student_id, today=today_value)
    except Exception as exc:
        st.error(f"Could not open the family database: {exc}")
        return

    if not due_items:
        st.success("Nothing due in the next two weeks.")
        return

    for due in due_items:
        badge = deadline_badge(due.days_until, deadline=parse_date(due.due_on))
        with st.container(border=True):
            col_what, col_when = st.columns([0.6, 0.4])
            with col_what:
                st.markdown(f"**{due.label}**")
                st.caption(due.award_title)
            with col_when:
                st.badge(badge.label, icon=badge.icon, color=badge.color)
                st.caption(tracker.STATUS_LABELS[due.status])
