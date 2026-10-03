"""Shared to-dos: one list the student and her parents both write to.

A row records who it is for (``assignee``), who added it (``created_by``, the
Tailscale login when one is known, and ``created_role``, the mode it was added
from) and who ticked it off.  ``source_kind`` / ``source_ref`` point a row back
at whatever generated it, so the To-dos page can tick a checklist step or a
requirement along with the row.

The SQL lives here rather than in :mod:`src.store.repo` only to keep that
module from growing further; it uses the same helpers and conventions.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date

from src.store.db import utc_now
from src.store.repo import UNSET, _delete, _fetch_one, _insert, _set, _update

ASSIGNEES: tuple[str, ...] = ("student", "parent", "family")
DEFAULT_ASSIGNEE = "student"

SOURCE_KINDS: tuple[str, ...] = (
    "manual",
    "application",
    "checklist",
    "letter",
    "requirement",
    "opportunity",
    "milestone",
)
DEFAULT_SOURCE_KIND = "manual"

# The To-dos page groups, in display order.  "later" is dated but past the
# Coming up window; "someday" has no date at all.
TODO_BUCKETS: tuple[str, ...] = (
    "overdue",
    "this_week",
    "coming_up",
    "later",
    "someday",
    "done",
)
THIS_WEEK_DAYS = 7
COMING_UP_DAYS = 60


def normalize_assignee(value: object) -> str:
    text = str(value or "").strip().casefold()
    return text if text in ASSIGNEES else DEFAULT_ASSIGNEE


def normalize_source_kind(value: object) -> str:
    text = str(value or "").strip().casefold()
    return text if text in SOURCE_KINDS else DEFAULT_SOURCE_KIND


@dataclass(slots=True)
class Todo:
    id: int
    student_id: str
    title: str
    notes: str = ""
    due_on: str | None = None
    assignee: str = DEFAULT_ASSIGNEE
    created_by: str = ""
    created_role: str = ""
    done_on: str | None = None
    done_by: str = ""
    source_kind: str = DEFAULT_SOURCE_KIND
    source_ref: str = ""
    created_at: str = ""
    updated_at: str = ""

    @property
    def done(self) -> bool:
        return self.done_on is not None


def _parse_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


def todo_bucket(todo: Todo, today: date) -> str:
    """Which To-dos group a row belongs in on ``today``.

    An unparseable date is treated as no date: the row still shows, under
    Someday, rather than vanishing.
    """
    if todo.done:
        return "done"
    due = _parse_date(todo.due_on)
    if due is None:
        return "someday"
    days = (due - today).days
    if days < 0:
        return "overdue"
    if days <= THIS_WEEK_DAYS:
        return "this_week"
    if days <= COMING_UP_DAYS:
        return "coming_up"
    return "later"


def _optional(row: sqlite3.Row, column: str) -> str | None:
    value = row[column]
    return None if value is None else str(value)


def _to_todo(row: sqlite3.Row) -> Todo:
    return Todo(
        id=int(row["id"]),
        student_id=str(row["student_id"]),
        title=str(row["title"]),
        notes=str(row["notes"]),
        due_on=_optional(row, "due_on"),
        assignee=normalize_assignee(row["assignee"]),
        created_by=str(row["created_by"]),
        created_role=str(row["created_role"]),
        done_on=_optional(row, "done_on"),
        done_by=str(row["done_by"]),
        source_kind=normalize_source_kind(row["source_kind"]),
        source_ref=str(row["source_ref"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def create_todo(
    conn: sqlite3.Connection,
    student_id: str,
    title: str,
    *,
    notes: str = "",
    due_on: str | None = None,
    assignee: str = DEFAULT_ASSIGNEE,
    created_by: str = "",
    created_role: str = "",
    source_kind: str = DEFAULT_SOURCE_KIND,
    source_ref: str = "",
) -> Todo:
    now = utc_now()
    todo = Todo(
        id=0,
        student_id=student_id,
        title=title,
        notes=notes,
        due_on=due_on,
        assignee=normalize_assignee(assignee),
        created_by=created_by,
        created_role=created_role,
        source_kind=normalize_source_kind(source_kind),
        source_ref=source_ref,
        created_at=now,
        updated_at=now,
    )
    todo.id = _insert(
        conn,
        "todos",
        {
            "student_id": todo.student_id,
            "title": todo.title,
            "notes": todo.notes,
            "due_on": todo.due_on,
            "assignee": todo.assignee,
            "created_by": todo.created_by,
            "created_role": todo.created_role,
            "done_on": None,
            "done_by": "",
            "source_kind": todo.source_kind,
            "source_ref": todo.source_ref,
            "created_at": now,
            "updated_at": now,
        },
    )
    return todo


def get_todo(conn: sqlite3.Connection, todo_id: int) -> Todo | None:
    row = _fetch_one(conn, "SELECT * FROM todos WHERE id = ?", (todo_id,))
    return None if row is None else _to_todo(row)


def find_todo(
    conn: sqlite3.Connection, student_id: str, source_kind: str, source_ref: str
) -> Todo | None:
    """The row generated from one source, so a generator does not add it twice."""
    row = _fetch_one(
        conn,
        "SELECT * FROM todos WHERE student_id = ? AND source_kind = ? AND source_ref = ? "
        "ORDER BY id LIMIT 1",
        (student_id, normalize_source_kind(source_kind), source_ref),
    )
    return None if row is None else _to_todo(row)


def list_todos(
    conn: sqlite3.Connection,
    student_id: str,
    *,
    assignees: tuple[str, ...] | None = None,
    include_done: bool = True,
) -> list[Todo]:
    """A student's to-dos, dated first by date, then undated, oldest first."""
    clauses = ["student_id = ?"]
    params: list[object] = [student_id]
    if assignees is not None:
        wanted = tuple(normalize_assignee(value) for value in assignees)
        clauses.append(f"assignee IN ({', '.join('?' for _ in wanted)})")
        params.extend(wanted)
    if not include_done:
        clauses.append("done_on IS NULL")
    rows = conn.execute(
        f"SELECT * FROM todos WHERE {' AND '.join(clauses)} "
        "ORDER BY due_on IS NULL, due_on, created_at, id",
        tuple(params),
    ).fetchall()
    return [_to_todo(row) for row in rows]


def update_todo(
    conn: sqlite3.Connection,
    todo_id: int,
    *,
    title: str = UNSET,
    notes: str = UNSET,
    due_on: str | None = UNSET,
    assignee: str = UNSET,
) -> bool:
    changes = _set(title=title, notes=notes, due_on=due_on, assignee=assignee)
    if "assignee" in changes:
        changes["assignee"] = normalize_assignee(changes["assignee"])
    return _update(conn, "todos", todo_id, changes)


def set_todo_done(
    conn: sqlite3.Connection,
    todo_id: int,
    done: bool,
    *,
    done_by: str = "",
    today: date | None = None,
) -> bool:
    """Tick a row off (recording who and when) or un-tick it."""
    changes: dict[str, str | None]
    if done:
        changes = {"done_on": (today or date.today()).isoformat(), "done_by": done_by}
    else:
        changes = {"done_on": None, "done_by": ""}
    return _update(conn, "todos", todo_id, changes)


def delete_todo(conn: sqlite3.Connection, todo_id: int) -> bool:
    return _delete(conn, "todos", todo_id)
