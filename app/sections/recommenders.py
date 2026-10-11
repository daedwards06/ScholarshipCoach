from __future__ import annotations

from datetime import date
from typing import Any

import streamlit as st

from app import identity, modes, state
from app.helpers import BadgeColor
from app.sections import todos as todos_page
from src.store import repo, todos, tracker
from src.store.db import open_db


def letter_counts(requests: list[repo.RecommendationRequest]) -> tuple[int, int]:
    """``(in, waiting)``: letters received, and letters planned or asked."""
    statuses = [tracker.normalize_request_status(request.status) for request in requests]
    received = sum(status == "received" for status in statuses)
    waiting = sum(status in tracker.OPEN_REQUEST_STATUSES for status in statuses)
    return received, waiting


def letter_badge(requests: list[repo.RecommendationRequest]) -> tuple[str, BadgeColor]:
    """The badge label and color for one person's letters."""
    received, waiting = letter_counts(requests)
    if not received and not waiting:
        return "No letters yet", "gray"
    return f"{received} in / {waiting} waiting", "blue" if waiting else "green"


def _sync_letter_todo(conn: Any, student_id: str, request_id: int, today_value: date) -> None:
    created_by, created_role = todos_page.author_for(
        identity.current_identity(), todos_page.viewer_role(state.current_mode())
    )
    todos.sync_letter_todo(
        conn,
        student_id,
        request_id,
        created_by=created_by,
        created_role=created_role,
        today=today_value,
    )


def _render_new_recommender_form(conn: Any, student_id: str) -> None:
    with st.form("new_recommender", clear_on_submit=True):
        name = st.text_input("Name", placeholder="e.g. Ms. Rivera")
        col_role, col_email = st.columns(2)
        with col_role:
            role = st.text_input("Role", placeholder="AP Physics teacher")
        with col_email:
            email = st.text_input("Email")
        if st.form_submit_button("Add recommender") and name.strip():
            repo.create_recommender(conn, student_id, name.strip(), role, email)
            st.rerun()


def _render_recommendation_request(
    conn: Any,
    student_id: str,
    request: repo.RecommendationRequest,
    award_title: str,
    today_value: date,
) -> None:
    status = tracker.normalize_request_status(request.status)
    with st.container(border=True):
        st.markdown(f"**{award_title}**")
        col_status, col_due = st.columns(2)
        with col_status:
            chosen = st.selectbox(
                "Status",
                options=tracker.next_request_statuses(status),
                index=0,
                format_func=lambda name: tracker.REQUEST_STATUS_LABELS[str(name)],
                key=f"req_status_{request.id}",
            )
        with col_due:
            due_value = None
            if request.due_on:
                try:
                    due_value = date.fromisoformat(request.due_on[:10])
                except ValueError:
                    due_value = None
            due_choice = st.date_input(
                "Letter due", value=due_value, key=f"req_due_{request.id}", format="YYYY-MM-DD"
            )
        stamps = [
            text
            for text in (
                f"Asked {request.asked_on}" if request.asked_on else "",
                f"Received {request.received_on}" if request.received_on else "",
            )
            if text
        ]
        if stamps:
            st.caption(" · ".join(stamps))

        new_due = due_choice.isoformat() if isinstance(due_choice, date) else None
        if new_due != (request.due_on[:10] if request.due_on else None):
            repo.update_recommendation_request(conn, request.id, due_on=new_due)
            _sync_letter_todo(conn, student_id, request.id, today_value)
            st.rerun()
        if str(chosen) != status:
            try:
                tracker.set_request_status(conn, request.id, str(chosen), today=today_value)
            except tracker.TransitionError as exc:
                st.error(str(exc))
            else:
                _sync_letter_todo(conn, student_id, request.id, today_value)
                st.rerun()
        if new_due and tracker.normalize_request_status(request.status) == "planned":
            st.caption("On her to-dos as an ask by this date.")
        if st.button("Remove request", key=f"req_delete_{request.id}"):
            repo.delete_recommendation_request(conn, request.id)
            _sync_letter_todo(conn, student_id, request.id, today_value)
            st.rerun()


def _render_recommender_detail(
    conn: Any,
    recommender: repo.Recommender,
    applications: list[repo.Application],
    today_value: date,
) -> None:
    requests = repo.list_recommendation_requests(conn, recommender_id=recommender.id)
    titles = {
        application.id: application.title or application.catalog_id
        for application in applications
    }
    label, color = letter_badge(requests)

    with st.container(border=True, key=f"recommender_{recommender.id}"):
        with st.container(horizontal=True, gap="small", vertical_alignment="center"):
            st.markdown(f"**{recommender.name}**")
            if recommender.role:
                st.caption(recommender.role)
            st.badge(label, icon=":material/mail:", color=color)
        _render_recommender_letters(
            conn, recommender, requests, applications, titles, today_value
        )


def _render_recommender_letters(
    conn: Any,
    recommender: repo.Recommender,
    requests: list[repo.RecommendationRequest],
    applications: list[repo.Application],
    titles: dict[int, str],
    today_value: date,
) -> None:
    with st.expander("Letters and contact", expanded=False):
        col_role, col_email = st.columns(2)
        with col_role:
            role = st.text_input("Role", value=recommender.role, key=f"rec_role_{recommender.id}")
        with col_email:
            email = st.text_input(
                "Email", value=recommender.email, key=f"rec_email_{recommender.id}"
            )
        if (role, email) != (recommender.role, recommender.email):
            repo.update_recommender(conn, recommender.id, role=role, email=email)

        for request in requests:
            _render_recommendation_request(
                conn,
                recommender.student_id,
                request,
                titles.get(request.application_id, "This award"),
                today_value,
            )

        remaining = [
            application
            for application in applications
            if application.id not in {request.application_id for request in requests}
        ]
        if remaining:
            chosen = st.selectbox(
                "Ask for a letter for",
                options=[application.id for application in remaining],
                format_func=lambda value: titles.get(int(value), ""),
                key=f"rec_new_request_{recommender.id}",
            )
            if st.button("Add request", key=f"rec_add_request_{recommender.id}"):
                repo.create_recommendation_request(conn, recommender.id, int(chosen))
                st.rerun()

        if st.button("Remove recommender", key=f"rec_delete_{recommender.id}"):
            repo.delete_recommender(conn, recommender.id)
            for request in requests:
                _sync_letter_todo(conn, recommender.student_id, request.id, today_value)
            st.rerun()


def render() -> None:
    modes.render_page_header("recommenders")
    today_value = state.effective_today(st.session_state.profile)
    try:
        with open_db() as conn:
            student_id = state.ensure_student(conn)
            people = repo.list_recommenders(conn, student_id)
            applications = repo.list_applications(conn, student_id)
            if not people:
                st.info(
                    "No recommenders yet. Ask early — a teacher writing ten letters "
                    "needs weeks, not days."
                )
            with st.expander("Add a recommender", expanded=not people):
                _render_new_recommender_form(conn, student_id)
            if people and not applications:
                st.caption("Save an award first, then ask a recommender for its letter.")
            for recommender in people:
                _render_recommender_detail(conn, recommender, applications, today_value)
    except Exception as exc:
        st.error(f"Could not open the family database: {exc}")
