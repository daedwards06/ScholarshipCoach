from __future__ import annotations

from datetime import date
from typing import Any

import streamlit as st

from app import identity, modes, state
from app.components import render_empty_state
from app.helpers import BadgeColor, markdown_safe
from app.sections import todos as todos_page
from app.sections.what_if import GOAL_QUERY_PARAM
from src.store import repo, requirements, todos
from src.store.db import open_db
from src.store.repo import College, CollegeRequirement

ERROR_STATE_KEY = "my_path_error"

COMING_UP_COUNT = 3

STATUS_BADGE_COLORS: dict[str, BadgeColor] = {
    "not_started": "gray",
    "in_progress": "blue",
    "met": "green",
    "not_needed": "gray",
}

STATUS_BADGE_ICONS: dict[str, str] = {
    "not_started": ":material/radio_button_unchecked:",
    "in_progress": ":material/pending:",
    "met": ":material/check_circle:",
    "not_needed": ":material/remove_circle:",
}

_EMPTY_MESSAGES: dict[str, str] = {
    "student": "Your family hasn't added a school yet.",
    "parent": "No first target school yet. Add one in Colleges & Money and mark it first.",
}

_NO_REQUIREMENTS_MESSAGES: dict[str, str] = {
    "student": "Your family hasn't added what this school asks for yet.",
    "parent": "No requirements yet. Add them under this school in Colleges & Money.",
}

# Where the empty state points.
_EMPTY_STATE_LINKS: dict[str, tuple[str, str]] = {
    "student": ("opportunities", "Browse opportunities"),
    "parent": ("colleges_money", "Open Colleges & Money"),
}


def path_heading(college_name: str, role: str) -> str:
    whose = "Your" if role == "student" else "Her"
    return f"{whose} path to {college_name}"


def can_set_status(requirement: CollegeRequirement, mode: modes.Mode) -> bool:
    """She sets status on her own course and test rows; Parent mode only reads."""
    return (
        modes.normalize_mode(mode) == "student"
        and requirement.category in requirements.STUDENT_STATUS_CATEGORIES
        and requirement.status != "not_needed"
    )


def coming_up(rows: list[todos.TodoRow], count: int = COMING_UP_COUNT) -> list[todos.TodoRow]:
    """The next ``count`` open dated rows, overdue first."""
    dated = [row for row in rows if not row.done and row.due_on]
    dated.sort(key=lambda row: (row.due_on or "", row.title.casefold()))
    return dated[:count]


def _pages() -> dict[str, Any]:
    pages: dict[str, Any] = st.session_state.get(modes.NAV_PAGES_STATE_KEY, {})
    return pages


def _page_link(
    mode: modes.Mode,
    section: str,
    label: str | None = None,
    query_params: dict[str, str] | None = None,
) -> None:
    pages = _pages()
    if modes.can_view(mode, section) and section in pages:
        st.page_link(pages[section], label=label, query_params=query_params)


def what_if_params(requirement: CollegeRequirement) -> dict[str, str]:
    """The link's query, so What If opens at this row's target."""
    return {GOAL_QUERY_PARAM: str(requirement.id)}


def _set_status(requirement_id: int, widget_key: str, today: date) -> None:
    status = st.session_state.get(widget_key)
    if status is None:
        return
    try:
        with open_db() as conn:
            repo.update_requirement(conn, requirement_id, status=str(status))
            requirements.sync_requirement_todos(conn, state.current_student_id(), today=today)
    except Exception as exc:
        st.session_state[ERROR_STATE_KEY] = f"Could not save that status: {exc}"


def _render_requirement(
    requirement: CollegeRequirement, *, mode: modes.Mode, gpa: float | None, today: date
) -> None:
    with st.container(border=True, key=f"req_{requirement.id}", gap="small"):
        st.markdown(f"**{markdown_safe(requirement.label)}**")
        parts = [
            requirements.CATEGORY_LABELS[requirement.category],
            markdown_safe(requirement.target),
            requirements.due_by_text(requirement.due_by, today),
        ]
        with st.container(horizontal=True, gap="small", vertical_alignment="center"):
            st.badge(
                requirements.STATUS_LABELS[requirement.status],
                icon=STATUS_BADGE_ICONS[requirement.status],
                color=STATUS_BADGE_COLORS[requirement.status],
            )
            st.caption(" · ".join(part for part in parts if part))

        gap = requirements.gpa_gap_text(requirement, gpa)
        if gap:
            with st.container(horizontal=True, gap="small", vertical_alignment="center"):
                st.markdown(gap)
                _page_link(mode, "what_if", "Try What If", what_if_params(requirement))
        if requirement.notes:
            st.caption(markdown_safe(requirement.notes))

        if can_set_status(requirement, mode):
            # The status is part of the key so a change made elsewhere (a tick
            # on To-dos) is never overridden by this widget's remembered value.
            widget_key = f"req_status_{requirement.id}_{requirement.status}"
            st.segmented_control(
                "Status",
                options=requirements.STUDENT_STATUSES,
                default=requirement.status,
                format_func=lambda value: requirements.STATUS_LABELS[str(value)],
                key=widget_key,
                on_change=_set_status,
                args=(requirement.id, widget_key, today),
                label_visibility="collapsed",
            )


def _render_path(
    college: College,
    items: list[CollegeRequirement],
    *,
    mode: modes.Mode,
    role: str,
    gpa: float | None,
    grade_level: str | None,
    today: date,
) -> None:
    st.subheader(markdown_safe(path_heading(college.name, role)))
    if not items:
        _render_empty(mode, role, _NO_REQUIREMENTS_MESSAGES[role])
        return

    shown = requirements.with_effective_status(items, gpa)
    met, counted = repo.requirement_progress(shown)
    st.markdown(requirements.progress_text(met, counted))
    st.progress(met / counted if counted else 0.0)

    groups = requirements.group_requirements(shown, grade_level, today)
    for group, label in requirements.PATH_GROUP_LABELS.items():
        found = groups[group]
        if not found:
            continue
        if group == "done":
            with st.expander(f"{label} ({len(found)})"):
                for requirement in found:
                    _render_requirement(requirement, mode=mode, gpa=gpa, today=today)
            continue
        st.markdown(f"**{label}**")
        for requirement in found:
            _render_requirement(requirement, mode=mode, gpa=gpa, today=today)


def _render_empty(mode: modes.Mode, role: str, message: str) -> None:
    section, label = _EMPTY_STATE_LINKS[role]
    pages = _pages()
    link = pages.get(section) if modes.can_view(mode, section) else None
    render_empty_state(
        message,
        link_page=link,
        link_label=label if link is not None else None,
    )


def _render_coming_up(
    rows: list[todos.TodoRow], *, mode: modes.Mode, role: str, viewer_name: str, today: date
) -> None:
    upcoming = coming_up(rows)
    if not upcoming:
        return
    st.markdown("**Coming up**")
    for row in upcoming:
        todos_page.render_row(
            row, mode=mode, role=role, viewer_name=viewer_name, today=today, here="my_path"
        )


def render() -> None:
    mode = state.current_mode()
    role = todos_page.viewer_role(mode)
    who = identity.current_identity()
    viewer_name = who.name if who is not None else ""
    profile = st.session_state.profile
    today_value = state.effective_today(profile)
    grade_level = str(profile.get("grade_level") or "") or None
    gpa = float(profile.get("gpa") or 0.0) or None

    modes.render_page_header("my_path")
    if role == "parent":
        st.badge("Her view", icon=":material/visibility:", color="gray")

    try:
        with open_db() as conn:
            student_id = state.ensure_student(conn)
            requirements.sync_requirement_todos(conn, student_id, today=today_value)
            college = repo.first_target_college(conn, student_id)
            items = [] if college is None else repo.list_requirements(conn, college.id)
            rows = todos.assemble_rows(conn, student_id, role, today=today_value)
    except Exception as exc:
        st.error(f"Could not open the family database: {exc}")
        return

    error = st.session_state.pop(ERROR_STATE_KEY, None)
    if error:
        st.error(error)

    if college is None:
        _render_empty(mode, role, _EMPTY_MESSAGES[role])
    else:
        _render_path(
            college,
            items,
            mode=mode,
            role=role,
            gpa=gpa,
            grade_level=grade_level,
            today=today_value,
        )

    _render_coming_up(rows, mode=mode, role=role, viewer_name=viewer_name, today=today_value)

    if mode == "student":
        modes.render_home_links(mode, "my_path")
    else:
        _page_link(mode, "todos", "All to-dos")
