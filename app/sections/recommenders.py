from __future__ import annotations

from datetime import date
from typing import Any

import streamlit as st

from app import modes, state
from src.store import repo, tracker
from src.store.db import open_db


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
    conn: Any, request: repo.RecommendationRequest, award_title: str, today_value: date
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
            st.rerun()
        if str(chosen) != status:
            try:
                tracker.set_request_status(conn, request.id, str(chosen), today=today_value)
            except tracker.TransitionError as exc:
                st.error(str(exc))
            else:
                st.rerun()
        if st.button("Remove request", key=f"req_delete_{request.id}"):
            repo.delete_recommendation_request(conn, request.id)
            st.rerun()


def _render_recommender_detail(
    conn: Any,
    recommender: repo.Recommender,
    applications: list[repo.Application],
    today_value: date,
) -> None:
    requests = repo.list_recommendation_requests(conn, recommender_id=recommender.id)
    outstanding = sum(
        1
        for request in requests
        if tracker.normalize_request_status(request.status) in tracker.OPEN_REQUEST_STATUSES
    )
    titles = {
        application.id: application.title or application.catalog_id
        for application in applications
    }
    header = recommender.name
    if recommender.role:
        header += f" · {recommender.role}"
    header += f" · {outstanding} outstanding" if outstanding else " · all in"

    with st.expander(header, expanded=False):
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
                conn, request, titles.get(request.application_id, "This award"), today_value
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
            st.rerun()


def render() -> None:
    st.subheader(modes.SECTION_LABELS["recommenders"])
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
