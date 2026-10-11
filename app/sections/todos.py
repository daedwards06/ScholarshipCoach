from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import streamlit as st

from app import identity, modes, state
from app.components import render_empty_state
from app.helpers import deadline_badge, friendly_date, parse_date
from src.store import requirements, todos
from src.store.db import open_db

ERROR_STATE_KEY = "todos_error"

GROUP_LABELS: dict[str, str] = {
    "overdue": "Overdue",
    "this_week": "This week",
    "coming_up": "Coming up",
    "someday": "Someday",
}

# Done rows stay visible this long, so a parent can see what she finished (P3).
DONE_VISIBLE_DAYS = 14

ASSIGNEE_CHOICES: dict[str, dict[str, str]] = {
    "student": {"student": "Me", "parent": "Parents", "family": "Family"},
    "parent": {"student": "Her", "parent": "Parents", "family": "Family"},
}

_ASSIGNEE_LABELS: dict[str, dict[str, str]] = {
    "student": {"student": "For you", "parent": "For your parents", "family": "For the family"},
    "parent": {"student": "For her", "parent": "For parents", "family": "For the family"},
}

# Where a live tracker row (by kind) or a generated to-do (by source) links back
# to, when the mode can see that page.
_SOURCE_PAGES: dict[str, str] = {
    "deadline": "applications",
    "checklist": "applications",
    "letter": "recommenders",
    "requirement": "my_path",
    "opportunity": "opportunities",
}

# The page the empty state points to.
_EMPTY_STATE_LINKS: dict[str, tuple[str, str]] = {
    "student": ("opportunities", "Browse opportunities"),
    "parent": ("find", "Find scholarships"),
}


def viewer_role(mode: modes.Mode) -> str:
    return "student" if modes.normalize_mode(mode) == "student" else "parent"


def assignee_label(assignee: str, role: str) -> str:
    return _ASSIGNEE_LABELS[role][todos.normalize_assignee(assignee)]


def author_label(row: todos.TodoRow, role: str, viewer_name: str = "") -> str:
    """Who a row came from, in the reader's words ("from you", "from Dana")."""
    if row.kind != "todo":
        return ""
    if row.source_kind == "milestone":
        return "planning date"
    if row.source_kind == "requirement":
        return "school requirement"
    if row.created_by:
        if viewer_name and row.created_by == viewer_name:
            return "from you"
        return f"from {row.created_by.split()[0]}"
    if row.created_role == role:
        return "from you"
    if row.created_role == "student":
        return "from her"
    if row.created_role == "parent":
        return "from your parents"
    return ""


def author_for(who: identity.Identity | None, role: str) -> tuple[str, str]:
    """``(created_by, created_role)`` for a new row: the signed-in name, else just the mode."""
    return (who.name if who is not None else "", role)


def group_rows(rows: list[todos.TodoRow], today: date) -> dict[str, list[todos.TodoRow]]:
    groups: dict[str, list[todos.TodoRow]] = {bucket: [] for bucket in todos.TODO_BUCKETS}
    for row in rows:
        groups[todos.row_bucket(row, today)].append(row)
    return groups


def next_up_text(groups: dict[str, list[todos.TodoRow]], today: date) -> str | None:
    for row in (*groups["coming_up"], *groups["later"]):
        due = parse_date(row.due_on)
        if due is not None:
            return f"{row.title}, {friendly_date(due, today)}"
    return None


def _recently_done(rows: list[todos.TodoRow], today: date) -> list[todos.TodoRow]:
    cutoff = today - timedelta(days=DONE_VISIBLE_DAYS)
    return [row for row in rows if (parse_date(row.done_on) or today) >= cutoff]


def _tick(row: todos.TodoRow, widget_key: str, done_by: str, today: date) -> None:
    try:
        with open_db() as conn:
            todos.set_row_done(
                conn, row, bool(st.session_state.get(widget_key)), done_by=done_by, today=today
            )
    except Exception as exc:
        st.session_state[ERROR_STATE_KEY] = f"Could not save that tick: {exc}"


def render_row(
    row: todos.TodoRow,
    *,
    mode: modes.Mode,
    role: str,
    viewer_name: str,
    today: date,
    here: str = "todos",
) -> None:
    # The done state is part of the key so a tick made elsewhere is never
    # overridden by this widget's remembered value.
    widget_key = f"tick_{row.key}_{int(row.done)}"
    with st.container(border=True, key=row.key, gap="small"):
        st.checkbox(
            row.title,
            value=row.done,
            key=widget_key,
            on_change=_tick,
            args=(row, widget_key, viewer_name, today),
            help=(row.detail or None) if row.kind == "todo" else None,
        )
        with st.container(horizontal=True, gap="small", vertical_alignment="center"):
            due = parse_date(row.due_on)
            if due is not None and not row.done:
                badge = deadline_badge((due - today).days, deadline=due)
                st.badge(badge.label, icon=badge.icon, color=badge.color)
            parts = [assignee_label(row.assignee, role), author_label(row, role, viewer_name)]
            if row.kind != "todo":
                parts.append(row.detail)
            st.caption(" · ".join(part for part in parts if part))
            target = _SOURCE_PAGES.get(row.source_kind if row.kind == "todo" else row.kind)
            pages = st.session_state.get(modes.NAV_PAGES_STATE_KEY, {})
            if target and target != here and modes.can_view(mode, target) and target in pages:
                st.page_link(pages[target], label=modes.SECTION_LABELS[target])


def _render_add_form(*, role: str, who: identity.Identity | None, expanded: bool) -> None:
    choices = ASSIGNEE_CHOICES[role]
    with st.expander("Add a to-do", icon=":material/add:", expanded=expanded):
        with st.form("add_todo", clear_on_submit=True, border=False):
            title = st.text_input("To-do", placeholder="Register for the PSAT")
            due = st.date_input("Date (optional)", value=None, format="MM/DD/YYYY")
            assignee = st.radio(
                "For",
                options=list(choices),
                format_func=lambda value: choices[value],
                horizontal=True,
            )
            if st.form_submit_button("Add to-do", type="primary") and title.strip():
                created_by, created_role = author_for(who, role)
                try:
                    with open_db() as conn:
                        todos.create_todo(
                            conn,
                            state.current_student_id(),
                            title.strip(),
                            due_on=due.isoformat() if isinstance(due, date) else None,
                            assignee=str(assignee),
                            created_by=created_by,
                            created_role=created_role,
                        )
                except Exception as exc:
                    st.error(f"Could not add the to-do: {exc}")
                    return
                st.rerun()


def _render_empty_state(mode: modes.Mode, role: str, next_up: str | None) -> None:
    section, label = _EMPTY_STATE_LINKS[role]
    pages: dict[str, Any] = st.session_state.get(modes.NAV_PAGES_STATE_KEY, {})
    link = pages.get(section) if modes.can_view(mode, section) else None
    render_empty_state(
        "Nothing to do this week.",
        next_up=next_up,
        link_page=link,
        link_label=label if link is not None else None,
    )


def render() -> None:
    mode = state.current_mode()
    role = viewer_role(mode)
    who = identity.current_identity()
    viewer_name = who.name if who is not None else ""
    today_value = state.effective_today(st.session_state.profile)
    grade_level = str(st.session_state.profile.get("grade_level") or "")

    modes.render_page_header("todos")
    try:
        with open_db() as conn:
            student_id = state.ensure_student(conn)
            todos.sync_milestone_todos(conn, student_id, grade_level or None, today=today_value)
            requirements.sync_requirement_todos(conn, student_id, today=today_value)
            rows = todos.assemble_rows(conn, student_id, role, today=today_value)
    except Exception as exc:
        st.error(f"Could not open the family database: {exc}")
        return

    error = st.session_state.pop(ERROR_STATE_KEY, None)
    if error:
        st.error(error)

    groups = group_rows(rows, today_value)
    open_rows = [row for row in rows if not row.done]
    _render_add_form(role=role, who=who, expanded=not open_rows)

    def draw(row: todos.TodoRow) -> None:
        render_row(row, mode=mode, role=role, viewer_name=viewer_name, today=today_value)

    if not groups["overdue"] and not groups["this_week"]:
        _render_empty_state(mode, role, next_up_text(groups, today_value))

    for bucket, label in GROUP_LABELS.items():
        if not groups[bucket]:
            continue
        st.markdown(f"**{label}**")
        for row in groups[bucket]:
            draw(row)

    if groups["later"]:
        with st.expander(f"Later ({len(groups['later'])})"):
            for row in groups["later"]:
                draw(row)

    done = _recently_done(groups["done"], today_value)
    if done:
        with st.expander(f"Done recently ({len(done)})"):
            for row in done:
                draw(row)
