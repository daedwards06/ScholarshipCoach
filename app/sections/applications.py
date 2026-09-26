from __future__ import annotations

from datetime import date
from typing import Any

import streamlit as st

from app import modes, state
from app.helpers import days_until_deadline, urgency_indicator
from app.sections.essays import render_prompt_slot
from src.store import essays, repo, tracker
from src.store.db import open_db


def _render_checklist(conn: Any, application: repo.Application) -> None:
    items = repo.list_checklist_items(conn, application.id)
    slots = {slot.item_id: slot for slot in essays.prompt_slots(conn, application.id)}
    entries = essays.essay_bank(conn, application.student_id) if slots else []
    can_edit_essays = modes.can_edit_essays(state.current_mode())
    if items:
        for item in items:
            col_done, col_due = st.columns([0.75, 0.25])
            with col_done:
                checked = st.checkbox(item.label, value=item.done, key=f"chk_{item.id}")
            with col_due:
                due_value = None
                if item.due_on:
                    try:
                        due_value = date.fromisoformat(item.due_on[:10])
                    except ValueError:
                        due_value = None
                due_choice = st.date_input(
                    "Due",
                    value=due_value,
                    key=f"chk_due_{item.id}",
                    format="YYYY-MM-DD",
                    label_visibility="collapsed",
                )
            new_due = due_choice.isoformat() if isinstance(due_choice, date) else None
            if checked != item.done or new_due != (item.due_on[:10] if item.due_on else None):
                repo.update_checklist_item(conn, item.id, done=checked, due_on=new_due)
                st.rerun()
            slot = slots.get(item.id)
            if slot is not None:
                render_prompt_slot(conn, slot, entries, can_edit_essays)
    else:
        st.caption("No requirements recorded for this award — add what it asks for below.")

    new_label = st.text_input(
        "Add a step", key=f"chk_new_{application.id}", placeholder="e.g. Ask Ms. Perez for a letter"
    )
    if st.button("Add step", key=f"chk_add_{application.id}") and new_label.strip():
        tracker.add_checklist_item(conn, application.id, new_label)
        st.session_state[f"chk_new_{application.id}"] = ""
        st.rerun()


def _render_outcome_form(conn: Any, application: repo.Application, today_value: date) -> None:
    outcome = repo.get_outcome(conn, application.id)
    st.markdown("**Result**")
    col_result, col_amount = st.columns(2)
    with col_result:
        result = st.selectbox(
            "Result",
            options=tracker.DECIDED_STATUSES,
            index=(
                tracker.DECIDED_STATUSES.index(outcome.result)
                if outcome is not None and outcome.result in tracker.DECIDED_STATUSES
                else 0
            ),
            format_func=lambda name: tracker.STATUS_LABELS[str(name)],
            key=f"outcome_result_{application.id}",
        )
    with col_amount:
        amount = st.number_input(
            "Amount awarded",
            min_value=0.0,
            step=500.0,
            value=float(outcome.amount_awarded or 0.0) if outcome is not None else 0.0,
            key=f"outcome_amount_{application.id}",
        )
    paid_to = st.text_input(
        "Paid to",
        value=outcome.paid_to if outcome is not None else "",
        key=f"outcome_paid_{application.id}",
        placeholder="School, or the student",
    )
    renewal = st.text_input(
        "Renewal terms",
        value=outcome.renewal_terms if outcome is not None else "",
        key=f"outcome_renewal_{application.id}",
        placeholder="e.g. renewable 4 years at 3.0 GPA",
    )
    if st.button("Record result", key=f"outcome_save_{application.id}"):
        try:
            tracker.record_outcome(
                conn,
                application.id,
                str(result),
                amount_awarded=float(amount) or None,
                paid_to=paid_to,
                renewal_terms=renewal,
                today=today_value,
            )
        except tracker.TransitionError as exc:
            st.error(str(exc))
        else:
            st.rerun()


def _render_application_detail(
    conn: Any, application: repo.Application, today_value: date
) -> None:
    status = tracker.normalize_status(application.status)
    done, total = tracker.checklist_progress(repo.list_checklist_items(conn, application.id))
    progress = f" — {done}/{total} done" if total else ""
    header = f"{application.title or application.catalog_id} · {tracker.STATUS_LABELS[status]}{progress}"

    with st.expander(header, expanded=False):
        if application.deadline:
            days_until = days_until_deadline(application.deadline, today_value)
            urgency_text, _ = urgency_indicator(days_until)
            st.caption(f"Deadline: {application.deadline} · {urgency_text}")
        if application.submitted_on:
            st.caption(f"Submitted {application.submitted_on}")
        if application.source_url:
            st.link_button("Apply at Source", application.source_url)

        choices = tracker.next_statuses(status)
        chosen = st.selectbox(
            "Status",
            options=choices,
            index=0,
            format_func=lambda name: tracker.STATUS_LABELS[str(name)],
            key=f"app_status_{application.id}",
        )
        if str(chosen) != status:
            try:
                tracker.set_status(conn, application.id, str(chosen), today=today_value)
            except tracker.TransitionError as exc:
                st.error(str(exc))
            else:
                st.rerun()

        _render_checklist(conn, application)

        notes = st.text_area(
            "Notes", value=application.notes, key=f"app_notes_{application.id}", height=90
        )
        if notes != application.notes:
            repo.update_application(conn, application.id, notes=notes)

        if status in ("submitted", *tracker.DECIDED_STATUSES):
            _render_outcome_form(conn, application, today_value)

        if st.button("Remove from my list", key=f"app_delete_{application.id}"):
            repo.delete_application(conn, application.id)
            st.rerun()


def render() -> None:
    st.subheader(modes.SECTION_LABELS["applications"])
    today_value = state.effective_today(st.session_state.profile)
    try:
        with open_db() as conn:
            student_id = state.ensure_student(conn)
            applications = repo.list_applications(conn, student_id)
            if not applications:
                st.info(
                    "Nothing saved yet. Open Find Scholarships and press Save on a card."
                )
                return

            counts = {status: 0 for status in tracker.APPLICATION_STATUSES}
            for application in applications:
                counts[tracker.normalize_status(application.status)] += 1
            st.caption(
                " · ".join(
                    f"{tracker.STATUS_LABELS[status]}: {count}"
                    for status, count in counts.items()
                    if count
                )
            )

            status_filter = st.selectbox(
                "Show",
                options=("all", *tracker.APPLICATION_STATUSES),
                format_func=lambda name: (
                    "All" if name == "all" else tracker.STATUS_LABELS[str(name)]
                ),
                key="applications_status_filter",
            )
            for application in applications:
                if status_filter != "all":
                    if tracker.normalize_status(application.status) != status_filter:
                        continue
                _render_application_detail(conn, application, today_value)
    except Exception as exc:
        st.error(f"Could not open the family database: {exc}")
