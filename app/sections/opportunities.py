from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from app import identity, modes, state
from app.components import render_empty_state
from app.helpers import (
    badge_markdown,
    format_amount_range,
    friendly_date,
    markdown_safe,
    row_catalog_id,
    unverified_to_text,
)
from app.sections import todos as todos_page
from src.profile.grade_levels import grade_level_rank
from src.profile.store import to_stage1_profile
from src.rank.opportunities import OPEN_NOW, OPENS_LATER, PASSED, classify_opportunities
from src.store import todos
from src.store.db import open_db
from src.text_utils import coerce_text, normalize_list

GROUP_LABELS: dict[str, str] = {
    OPEN_NOW: "Open now",
    OPENS_LATER: "Opens later",
    PASSED: "Passed this year",
}

KIND_FILTERS: dict[str, str] = {"all": "All", "program": "Programs", "scholarship": "Awards"}

_GRADE_NAMES: dict[str, str] = {
    "9": "9th grade",
    "10": "10th grade",
    "11": "11th grade",
    "12": "12th grade",
    "college_1": "college year 1",
    "college_2": "college year 2",
    "college_3": "college year 3",
    "college_4": "college year 4",
}


_HIGH_SCHOOL_GRADES = frozenset({"9", "10", "11", "12"})


def grade_name(grade: str | None) -> str:
    return _GRADE_NAMES.get(str(grade or ""), str(grade or ""))


def grade_badge_text(grade_levels: Any, student_grade: str | None) -> str:
    """"Open to 10th grade" when her grade is listed, else the grades it is for."""
    grades = normalize_list(grade_levels)
    if student_grade and student_grade in grades:
        return f"Open to {grade_name(student_grade)}"
    if grades and all(grade in _HIGH_SCHOOL_GRADES for grade in grades):
        ordinals = [grade_name(grade).removesuffix(" grade") for grade in grades]
        if len(ordinals) > 2:
            return f"For {ordinals[0]}–{ordinals[-1]} grade"
        return f"For {' and '.join(ordinals)} grade"
    return "For " + " and ".join(grade_name(grade) for grade in grades)


def money_badge_text(row: pd.Series) -> str:
    """A program's cost ("Free", "Costs $500") or an award's amount."""
    if bool(row.get("is_program")):
        cost = row.get("cost")
        if cost is None or pd.isna(cost):
            return "Cost not listed"
        return "Free" if float(cost) == 0 else f"Costs ${float(cost):,.0f}"
    return format_amount_range(row.get("amount_min"), row.get("amount_max"))


def _day(value: Any, today: date) -> str:
    return friendly_date(value, today, weekday=False)


def headline(row: pd.Series, today: date) -> str:
    """The card's h3: the next step and its date, never the money."""
    group = row.get("opportunity_group")
    apply_by = row.get("apply_by")
    projected = bool(row.get("apply_by_projected"))
    opens_on = row.get("opens_on")
    opens_grade = row.get("opens_grade")
    due = None
    if isinstance(apply_by, date):
        # "~" marks a date projected from the usual cycle, not one the sponsor posted.
        due = f"~{_day(apply_by, today)}" if projected else _day(apply_by, today)

    if group == PASSED:
        return f"Passed · next round {due}" if due and projected else "Passed this year"
    if group == OPENS_LATER:
        if opens_grade:
            return f"Opens in {grade_name(opens_grade)}"
        if isinstance(opens_on, date):
            opened = f"Opens {_day(opens_on, today)}"
            return f"{opened} · due {due}" if due else opened
        if due:
            return f"Not open yet · due {due}"
        return "Date not posted yet"
    if due:
        return f"Apply by {due}"
    return "Open now · date not posted"


def is_local(row: pd.Series) -> bool:
    """Limited to named counties. Find's "Local" (any confirmed record) would
    mark every program here, since the family confirms each one."""
    return bool(normalize_list(row.get("counties_allowed")))


def fit_line(row: pd.Series) -> str:
    """Up to three plain reasons it is for her, from the axes it restricts on."""
    reasons = []
    if normalize_list(row.get("majors_allowed")):
        reasons.append("Fits your major")
    if is_local(row):
        reasons.append("Local to your county")
    states = [str(state_code).upper() for state_code in normalize_list(row.get("states_allowed"))]
    if states and len(states) <= 2:
        reasons.append(f"For {' and '.join(states)} students")
    return " · ".join(reasons[:3])


# A reminder for a round nobody has posted yet comes this long before its usual deadline.
WATCH_LEAD_DAYS = 30


def _years_later(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year + years)
    except ValueError:  # Feb 29 into a common year
        return day.replace(year=day.year + years, day=28)


def todo_plan(row: pd.Series, today: date, student_grade: str | None = None) -> tuple[str, str | None]:
    """``(title, due_on)`` for the to-do a card adds.

    Only a deadline the sponsor posted becomes "Apply to …" with that date.
    Otherwise the to-do is a reminder to look again when the round usually
    opens (or a month before its usual deadline), moved forward to the year of
    a later grade, so an annual program she cannot do yet comes back in time.
    """
    name = coerce_text(row.get("title")) or row_catalog_id(row)
    apply_by = row.get("apply_by")
    projected = bool(row.get("apply_by_projected"))
    opens_on = row.get("opens_on")
    opens_grade = coerce_text(row.get("opens_grade")) or None

    years = 0
    if opens_grade:
        later, now = grade_level_rank(opens_grade), grade_level_rank(student_grade)
        if later is None or now is None:
            return f"Apply to {name} in {grade_name(opens_grade)}", None
        years = later - now
    suffix = f" ({grade_name(opens_grade)})" if opens_grade else ""

    if not opens_grade and isinstance(apply_by, date) and not projected and apply_by >= today:
        return f"Apply to {name}", apply_by.isoformat()
    if isinstance(opens_on, date):
        return f"Watch for {name} to open{suffix}", _years_later(opens_on, years).isoformat()
    if isinstance(apply_by, date):
        watch = _years_later(apply_by, years) - timedelta(days=WATCH_LEAD_DAYS)
        return f"Check the dates for {name}{suffix}", max(watch, today).isoformat()
    if opens_grade:
        return f"Apply to {name} in {grade_name(opens_grade)}", None
    return f"Apply to {name}", None


def apply_filters(
    df: pd.DataFrame, *, kind: str = "all", free_only: bool = False, local_only: bool = False
) -> pd.DataFrame:
    filtered = df
    if kind == "program":
        filtered = filtered[filtered["is_program"].astype(bool)]
    elif kind == "scholarship":
        filtered = filtered[~filtered["is_program"].astype(bool)]
    if free_only:
        filtered = filtered[filtered.apply(lambda row: money_badge_text(row) == "Free", axis=1)]
    if local_only:
        filtered = filtered[filtered.apply(is_local, axis=1).astype(bool)]
    return filtered


@st.cache_data(show_spinner=False, max_entries=8)
def opportunities_frame(
    snapshot_path_text: str, profile: dict[str, Any], today_value: date
) -> pd.DataFrame:
    snapshot_df = state.load_snapshot_cached(snapshot_path_text)
    return classify_opportunities(
        snapshot_df, to_stage1_profile(profile, today=today_value), today_value
    )


def _opportunity_todos(student_id: str) -> dict[str, todos.Todo]:
    try:
        with open_db() as conn:
            found = todos.list_todos(conn, student_id)
    except Exception:
        return {}
    return {todo.source_ref: todo for todo in found if todo.source_kind == "opportunity"}


def _add_to_todos(row: pd.Series, role: str, today: date, student_grade: str | None = None) -> None:
    catalog_id = row_catalog_id(row)
    title, due_on = todo_plan(row, today, student_grade)
    created_by, created_role = todos_page.author_for(identity.current_identity(), role)
    try:
        with open_db() as conn:
            student_id = state.ensure_student(conn)
            if todos.find_todo(conn, student_id, "opportunity", catalog_id) is None:
                todos.create_todo(
                    conn,
                    student_id,
                    title,
                    notes=coerce_text(row.get("source_url")),
                    due_on=due_on,
                    assignee="student",
                    created_by=created_by,
                    created_role=created_role,
                    source_kind="opportunity",
                    source_ref=catalog_id,
                )
    except Exception as exc:
        st.error(f"Could not add the to-do: {exc}")
        return
    st.toast("Added to your to-dos." if role == "student" else "Added to her to-dos.")
    st.rerun()


def card_badges(row: pd.Series, student_grade: str | None, todo: todos.Todo | None, role: str) -> list[str]:
    badges = []
    # A record for a later grade says so in its headline ("Opens in 11th grade").
    if student_grade and student_grade in normalize_list(row.get("grade_levels")):
        badges.append(
            badge_markdown(grade_badge_text(row.get("grade_levels"), student_grade), ":material/school:", "blue")
        )
    # Two "$" in one badge would be typeset as LaTeX.
    badges.append(badge_markdown(markdown_safe(money_badge_text(row)), ":material/payments:", "gray"))
    if is_local(row):
        badges.append(badge_markdown("Local", ":material/location_on:", "violet"))
    if todo is not None and todo.created_role == "parent":
        label = "From your family" if role == "student" else "Suggested"
        badges.append(badge_markdown(label, ":material/send:", "violet"))
    confirm_text = unverified_to_text(row.get("unverified_axes"))
    if confirm_text:
        badges.append(badge_markdown(f"Check: {confirm_text}", ":material/warning:", "yellow"))
    return badges


def _render_card(
    row: pd.Series,
    *,
    student_grade: str | None,
    todo: todos.Todo | None,
    role: str,
    today: date,
) -> None:
    """Student-mode "Opportunity card": the headline is the next step and its date."""
    catalog_id = row_catalog_id(row)
    title = coerce_text(row.get("title")) or catalog_id or "Untitled"
    sponsor = coerce_text(row.get("sponsor"))
    source_url = coerce_text(row.get("source_url"))
    kind_badge = (
        ":gray-badge[:material/event: Program]"
        if bool(row.get("is_program"))
        else ":gray-badge[:material/workspace_premium: Award]"
    )
    with st.container(border=True, key=f"opp_{catalog_id}", gap="xsmall"):
        sponsor_text = markdown_safe(sponsor).replace("[", "(").replace("]", ")")
        sponsor_line = f"  \n:small[{sponsor_text}]" if sponsor else ""
        st.markdown(
            f"**{markdown_safe(title)}** {kind_badge}{sponsor_line}"
        )
        st.markdown(f"### {markdown_safe(headline(row, today))}")
        st.markdown(" ".join(card_badges(row, student_grade, todo, role)))
        with st.container(horizontal=True):
            if todo is not None:
                label = "On my to-dos" if role == "student" else "On her to-dos"
                st.badge(label, icon=":material/check_circle:", color="green")
            elif st.button(
                "Add to my to-dos" if role == "student" else "Suggest to her",
                key=f"opp_add_{catalog_id}",
                type="primary" if role == "student" else "secondary",
            ):
                _add_to_todos(row, role, today, student_grade)
            if source_url:
                st.link_button(
                    "Apply",
                    source_url,
                    icon=":material/open_in_new:",
                    type="primary" if todo is not None else "secondary",
                )
        description = coerce_text(row.get("description"))
        eligibility = coerce_text(row.get("eligibility_text"))
        reasons = fit_line(row)
        # The reasons sit in here, as on Find's card: on the face they wrap at
        # 400px and push the card past its height budget.
        if description or eligibility or reasons:
            with st.expander("Why it fits & more"):
                if reasons:
                    st.markdown(f"**Why it fits:** {reasons}")
                if description:
                    st.markdown(markdown_safe(description))
                if eligibility:
                    st.caption(f"Who can apply: {markdown_safe(eligibility)}")


def _nav_page(mode: modes.Mode, section: str) -> Any:
    pages: dict[str, Any] = st.session_state.get(modes.NAV_PAGES_STATE_KEY, {})
    return pages.get(section) if modes.can_view(mode, section) else None


def render() -> None:
    mode = state.current_mode()
    role = todos_page.viewer_role(mode)
    profile = st.session_state.profile
    today_value = state.effective_today(profile)
    student_grade = str(profile.get("grade_level") or "") or None

    modes.render_page_header("opportunities")
    if role == "parent":
        st.badge("Her view", icon=":material/visibility:", color="gray")

    snapshot_path_text = state.active_snapshot_path()
    if snapshot_path_text is None:
        link = _nav_page(mode, "catalog_inbox")
        render_empty_state(
            "Programs and awards show here once your family has built the list.",
            link_page=link,
            link_label="Open Catalog & Inbox" if link is not None else None,
        )
        return
    if student_grade is None:
        link = _nav_page(mode, "profile")
        render_empty_state(
            "Add your grade to see what's open to you.",
            link_page=link,
            link_label="Add your grade" if link is not None else None,
        )
        return

    try:
        frame = opportunities_frame(snapshot_path_text, dict(profile), today_value)
    except Exception as exc:
        st.error(f"Could not load opportunities: {exc}")
        return

    with st.expander("Filters", icon=":material/tune:"):
        kind = st.segmented_control(
            "Show",
            options=list(KIND_FILTERS),
            default="all",
            format_func=lambda value: KIND_FILTERS[str(value)],
            key="opp_kind",
        )
        free_only = st.checkbox("Free programs only", key="opp_free")
        local_only = st.checkbox("Local only", key="opp_local")
    shown = apply_filters(
        frame, kind=str(kind or "all"), free_only=free_only, local_only=local_only
    )

    attached = _opportunity_todos(state.current_student_id())

    def draw(row: pd.Series) -> None:
        _render_card(
            row,
            student_grade=student_grade,
            todo=attached.get(row_catalog_id(row)),
            role=role,
            today=today_value,
        )

    groups = {
        group: shown[shown["opportunity_group"] == group]
        for group in (OPEN_NOW, OPENS_LATER, PASSED)
    }
    st.markdown(f"**{GROUP_LABELS[OPEN_NOW]}**")
    if groups[OPEN_NOW].empty:
        later = groups[OPENS_LATER]
        next_up = None
        if not later.empty:
            first = later.iloc[0]
            next_up = f"{coerce_text(first.get('title'))} · {headline(first, today_value)}"
        link = _nav_page(mode, "catalog_inbox")
        render_empty_state(
            "Nothing is open to your grade right now."
            if role == "student"
            else "Nothing is open to her grade right now.",
            next_up=next_up,
            link_page=link,
            link_label="Add a program" if link is not None else None,
        )
    for _, row in groups[OPEN_NOW].iterrows():
        draw(row)

    if not groups[OPENS_LATER].empty:
        st.markdown(f"**{GROUP_LABELS[OPENS_LATER]}**")
        for _, row in groups[OPENS_LATER].iterrows():
            draw(row)

    if not groups[PASSED].empty:
        with st.expander(f"{GROUP_LABELS[PASSED]} ({len(groups[PASSED])})"):
            for _, row in groups[PASSED].iterrows():
                draw(row)
