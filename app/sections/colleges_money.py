from __future__ import annotations

from typing import Any

import streamlit as st

from app import modes, state
from app.helpers import award_count_text, money_text
from src.store import money, repo
from src.store.db import open_db


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
    header = f"{cost.name} · {where} · net {money_text(cost.net_price)}"
    with st.expander(header, expanded=False):
        st.caption(
            f"Sticker {money_text(cost.sticker_price)} · "
            f"{repo.COLLEGE_DEADLINE_TYPE_LABELS[cost.deadline_type]} · "
            f"still to cover {money_text(cost.remaining)}"
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
    col_won.metric("Total won", money_text(summary.total_won))
    col_awards.metric("Awards won", str(summary.award_count))
    col_colleges.metric("Colleges tracked", str(len(summary.colleges)))

    if summary.by_year:
        st.markdown("**Won by school year**")
        for row in summary.by_year:
            st.markdown(
                f"{row.label}: {money_text(row.total)} · {award_count_text(row.count)}"
            )

    if summary.renewals:
        st.markdown("**Renewal conditions to keep**")
        for renewal in summary.renewals:
            with st.container(border=True):
                st.markdown(f"**{renewal.award_title}** — {money_text(renewal.amount)}")
                st.caption(renewal.terms)


def render() -> None:
    st.subheader(modes.SECTION_LABELS["colleges_money"])
    if state.current_mode() == "student":
        st.info("Colleges and money live in Parent view.")
        return

    try:
        with open_db() as conn:
            student_id = state.ensure_student(conn)
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
