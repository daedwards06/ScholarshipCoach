from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from app import modes, state
from app.helpers import award_count_text, money_text
from src.store import money, repo, requirements
from src.store.db import open_db
from src.store.repo import CollegeRequirement

REQUIREMENT_FIELDS: tuple[str, ...] = (
    "category",
    "label",
    "target",
    "due_by",
    "status",
    "notes",
    "source_url",
    "verified_on",
)


@dataclass
class RequirementEdits:
    """What one save of the requirements table changes."""

    creates: list[dict[str, Any]] = field(default_factory=list)
    updates: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    deletes: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NaT


def _text(value: Any) -> str:
    return "" if _blank(value) else str(value).strip()


def _day_text(value: Any) -> str | None:
    if _blank(value):
        return None
    if isinstance(value, date):
        return value.isoformat()[:10]
    try:
        return date.fromisoformat(str(value).strip()[:10]).isoformat()
    except ValueError:
        return None


def requirement_records(items: list[CollegeRequirement]) -> list[dict[str, Any]]:
    """The editor's rows: one per requirement, in order, ``order`` counting from 1."""
    return [
        {
            "id": item.id,
            "order": index,
            "category": item.category,
            "label": item.label,
            "target": item.target,
            "due_by": item.due_by,
            "status": item.status,
            "notes": item.notes,
            "source_url": item.source_url,
            "verified_on": (
                None if item.verified_on is None else date.fromisoformat(item.verified_on[:10])
            ),
        }
        for index, item in enumerate(items, start=1)
    ]


def requirement_edits(
    existing: list[CollegeRequirement], records: list[dict[str, Any]]
) -> RequirementEdits:
    """Turn the edited table into creates, updates and deletes.

    Rows are numbered by their ``order`` (ties keep table order), so retyping
    one number reorders the list.  A row whose "Due by" is neither a grade nor
    a date is refused rather than filed under This semester by accident.
    """
    edits = RequirementEdits()
    by_id = {item.id: item for item in existing}
    kept: set[int] = set()

    def order_key(pair: tuple[int, dict[str, Any]]) -> tuple[float, int]:
        index, record = pair
        order: Any = record.get("order")
        return (float("inf") if _blank(order) else float(order), index)

    for position, (_, record) in enumerate(sorted(enumerate(records), key=order_key)):
        raw_id: Any = record.get("id")
        row_id = None if _blank(raw_id) else int(raw_id)
        if row_id is not None:
            kept.add(row_id)
        label = _text(record.get("label"))
        if not label:
            continue
        due_by = requirements.normalize_due_by(record.get("due_by"))
        if due_by is None:
            edits.errors.append(
                f"{label}: due by must be a grade (9-12) or a date like 2027-03-01."
            )
            continue
        values: dict[str, Any] = {
            "category": repo.normalize_requirement_category(record.get("category")),
            "label": label,
            "target": _text(record.get("target")),
            "due_by": due_by,
            "status": repo.normalize_requirement_status(record.get("status")),
            "notes": _text(record.get("notes")),
            "source_url": _text(record.get("source_url")),
            "verified_on": _day_text(record.get("verified_on")),
            "position": position,
        }
        current = None if row_id is None else by_id.get(row_id)
        if current is None:
            edits.creates.append(values)
            continue
        changes = {name: value for name, value in values.items() if getattr(current, name) != value}
        if changes:
            edits.updates.append((current.id, changes))

    edits.deletes = [item.id for item in existing if item.id not in kept]
    return edits


def _apply_requirement_edits(conn: Any, college_id: int, edits: RequirementEdits) -> None:
    for row_id in edits.deletes:
        repo.delete_requirement(conn, row_id)
    for row_id, changes in edits.updates:
        repo.update_requirement(conn, row_id, **changes)
    for values in edits.creates:
        repo.create_requirement(
            conn,
            college_id,
            values["label"],
            values["category"],
            target=values["target"],
            due_by=values["due_by"],
            status=values["status"],
            notes=values["notes"],
            source_url=values["source_url"],
            verified_on=values["verified_on"],
            position=values["position"],
        )


def _render_requirements(conn: Any, student_id: str, college_id: int) -> None:
    items = repo.list_requirements(conn, college_id)
    st.markdown("**Requirements**")
    st.caption(
        "What this school asks for; her first target's rows are what My Path shows her. "
        "Due by is a grade (11) or a date (2027-03-01), and dated rows also land on her "
        "To-dos. Set Verified on each time you re-check the school's own page."
    )
    with st.form(f"requirements_{college_id}", border=False):
        frame = pd.DataFrame(
            requirement_records(items), columns=["id", "order", *REQUIREMENT_FIELDS]
        )
        # The editor keeps its pending edits under its key, so the key changes
        # with the stored rows or a save would replay added rows on top of them.
        stamp = max((item.updated_at for item in items), default="")
        edited = st.data_editor(
            frame,
            key=f"requirements_editor_{college_id}_{len(items)}_{stamp}",
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "id": None,
                "order": st.column_config.NumberColumn(
                    "Order", min_value=1, step=1, width="small"
                ),
                "category": st.column_config.SelectboxColumn(
                    "Category",
                    options=repo.REQUIREMENT_CATEGORIES,
                    format_func=lambda value: requirements.CATEGORY_LABELS[str(value)],
                    default=repo.DEFAULT_REQUIREMENT_CATEGORY,
                    required=True,
                ),
                "label": st.column_config.TextColumn("Requirement", required=True),
                "target": st.column_config.TextColumn("Target", help="e.g. 4 units, 3.0, 1200"),
                "due_by": st.column_config.TextColumn(
                    "Due by", help="A grade (9-12) or a date (2027-03-01)"
                ),
                "status": st.column_config.SelectboxColumn(
                    "Status",
                    options=repo.REQUIREMENT_STATUSES,
                    format_func=lambda value: requirements.STATUS_LABELS[str(value)],
                    default=repo.DEFAULT_REQUIREMENT_STATUS,
                    required=True,
                ),
                "notes": st.column_config.TextColumn(
                    "Notes", help="Anything unclear goes here, with status Not started."
                ),
                "source_url": st.column_config.LinkColumn("Source"),
                "verified_on": st.column_config.DateColumn("Verified on", format="YYYY-MM-DD"),
            },
        )
        if st.form_submit_button("Save requirements"):
            edits = requirement_edits(items, edited.to_dict("records"))
            if edits.errors:
                for message in edits.errors:
                    st.error(message)
                return
            _apply_requirement_edits(conn, college_id, edits)
            requirements.sync_requirement_todos(conn, student_id, today=date.today())
            st.rerun()


def _render_new_college_form(conn: Any, student_id: str) -> None:
    with st.form("new_college", clear_on_submit=True):
        name = st.text_input("College", placeholder="e.g. NC State University")
        first = st.checkbox("Her first target school", help="My Path is built around this one.")
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
            college = repo.create_college(
                conn,
                student_id,
                name.strip(),
                cost_of_attendance=float(sticker) or None,
                in_state=bool(in_state),
                net_price_estimate=float(net) or None,
                deadline_type=str(deadline_type),
            )
            if first:
                repo.set_first_target(conn, student_id, college.id)
            st.rerun()


def _render_college_detail(
    conn: Any, student_id: str, cost: money.CollegeCost, *, first_target: bool
) -> None:
    where = "In state" if cost.in_state else "Out of state"
    header = f"{cost.name} · {where} · net {money_text(cost.net_price)}"
    if first_target:
        header = f"{header} · first target"
    with st.expander(header, expanded=False):
        st.caption(
            f"Sticker {money_text(cost.sticker_price)} · "
            f"{repo.COLLEGE_DEADLINE_TYPE_LABELS[cost.deadline_type]} · "
            f"still to cover {money_text(cost.remaining)}"
        )
        college_id = cost.college_id

        first = st.checkbox(
            "Her first target school",
            value=first_target,
            key=f"college_first_{college_id}",
            help="My Path is built around this one. Marking it unmarks any other.",
        )
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
                if first != first_target:
                    repo.set_first_target(conn, student_id, college_id if first else None)
                st.rerun()
        with col_remove:
            if st.button("Remove", key=f"college_remove_{college_id}"):
                repo.delete_college(conn, college_id)
                st.rerun()

        st.divider()
        _render_requirements(conn, student_id, college_id)


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
    modes.render_page_header("colleges_money")
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
            first_ids = {college.id for college in colleges if college.priority == 1}
            for cost in sorted(summary.colleges, key=lambda cost: cost.college_id not in first_ids):
                _render_college_detail(
                    conn, student_id, cost, first_target=cost.college_id in first_ids
                )
    except Exception as exc:
        st.error(f"Could not open the family database: {exc}")
