"""Shared to-dos: one list the student and her parents both write to.

A row records who it is for (``assignee``), who added it (``created_by``, the
Tailscale display name when one is known, and ``created_role``, the mode it was
added from) and who ticked it off.  ``source_kind`` / ``source_ref`` point a row back
at whatever generated it, so the To-dos page can tick a checklist step or a
requirement along with the row.

The SQL lives here rather than in :mod:`src.store.repo` only to keep that
module from growing further; it uses the same helpers and conventions.

The To-dos page shows more than this table.  Open application work from
:func:`src.store.tracker.this_week` joins it as live rows (:class:`TodoRow`)
rather than copies, so ticking a checklist step here ticks it in the
application, and ticking it there takes it off this list.  Planning milestones
for her grade are the other way round: they are written in as ordinary rows,
once per school year, because a calendar date has nothing of its own to tick.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date

from src.store import milestones as milestone_store
from src.store import repo, tracker
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
    return date_bucket(todo.due_on, today, done=todo.done)


def date_bucket(due_on: str | None, today: date, *, done: bool = False) -> str:
    if done:
        return "done"
    due = _parse_date(due_on)
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


# -- the assembled list -------------------------------------------------------

# Who a live tracker row is for: the deadline is the family's, the steps and
# the letters are hers to do.
_TRACKER_ASSIGNEES: dict[str, str] = {
    "deadline": "family",
    "checklist": "student",
    "letter": "student",
}

_TRACKER_SOURCE_KINDS: dict[str, str] = {
    "deadline": "application",
    "checklist": "checklist",
    "letter": "letter",
}

# A letter row ticked moves the request one step along.
_LETTER_NEXT_STEP: dict[str, str] = {"planned": "asked", "asked": "received"}


@dataclass(frozen=True, slots=True)
class TodoRow:
    """One line on the To-dos page: a stored to-do or live tracker work.

    ``kind`` is ``todo`` for a row of this table, else the tracker item kind
    (``deadline``, ``checklist``, ``letter``); ``ref_id`` is the id ticking
    acts on in that kind's own table.
    """

    key: str
    kind: str
    ref_id: int
    title: str
    due_on: str | None = None
    assignee: str = DEFAULT_ASSIGNEE
    detail: str = ""
    source_kind: str = DEFAULT_SOURCE_KIND
    source_ref: str = ""
    created_by: str = ""
    created_role: str = ""
    done_on: str | None = None
    done_by: str = ""
    application_id: int | None = None

    @property
    def done(self) -> bool:
        return self.done_on is not None


def visible_to(todo: Todo, viewer_role: str) -> bool:
    """Parents see every row; she sees hers, the family's, and what she asked of them."""
    if viewer_role != "student":
        return True
    return todo.assignee in ("student", "family") or todo.created_role == "student"


def row_from_todo(todo: Todo) -> TodoRow:
    return TodoRow(
        key=f"todo_{todo.id}",
        kind="todo",
        ref_id=todo.id,
        title=todo.title,
        due_on=todo.due_on,
        assignee=todo.assignee,
        detail=todo.notes,
        source_kind=todo.source_kind,
        source_ref=todo.source_ref,
        created_by=todo.created_by,
        created_role=todo.created_role,
        done_on=todo.done_on,
        done_by=todo.done_by,
    )


def row_from_due_item(due: tracker.DueItem) -> TodoRow:
    if due.kind == "deadline":
        title, detail = f"Submit {due.award_title}", "Application deadline"
    else:
        title, detail = due.label, due.award_title
    return TodoRow(
        key=f"todo_{due.kind}_{due.ref_id}",
        kind=due.kind,
        ref_id=due.ref_id,
        title=title,
        due_on=due.due_on,
        assignee=_TRACKER_ASSIGNEES.get(due.kind, DEFAULT_ASSIGNEE),
        detail=detail,
        source_kind=_TRACKER_SOURCE_KINDS.get(due.kind, DEFAULT_SOURCE_KIND),
        application_id=due.application_id,
    )


def row_bucket(row: TodoRow, today: date) -> str:
    return date_bucket(row.due_on, today, done=row.done)


def assemble_rows(
    conn: sqlite3.Connection, student_id: str, viewer_role: str, *, today: date
) -> list[TodoRow]:
    """Everything the To-dos page lists for ``viewer_role``, dated rows first."""
    stored = list_todos(conn, student_id)
    rows = [row_from_todo(todo) for todo in stored if visible_to(todo, viewer_role)]
    # A letter with an open "Ask ... by ..." row is that row until she has asked.
    asking = {todo.source_ref for todo in stored if todo.source_kind == "letter" and not todo.done}
    rows.extend(
        row_from_due_item(due)
        for due in tracker.this_week(conn, student_id, today=today, within_days=COMING_UP_DAYS)
        if not (due.kind == "letter" and str(due.ref_id) in asking)
    )
    rows.sort(key=lambda row: (row.due_on is None, row.due_on or "", row.title.casefold()))
    return rows


def set_row_done(
    conn: sqlite3.Connection,
    row: TodoRow,
    done: bool,
    *,
    done_by: str = "",
    today: date | None = None,
) -> bool:
    """Tick a row, acting on its source.

    A checklist step is ticked in the application.  A deadline row ticked means
    the application went in; a letter row moves the request one step (planned
    to asked, asked to received).  Tracker rows leave the list once their work
    is done, so only stored rows and checklist steps can be un-ticked here.
    A requirement's row ticks the requirement too (met, or back to in
    progress), so My Path and the list agree, and ticking an "Ask ... by ..."
    row marks that letter asked.
    """
    if row.kind == "todo":
        if row.source_kind == "requirement" and row.source_ref.isdigit():
            repo.update_requirement(
                conn, int(row.source_ref), status="met" if done else "in_progress"
            )
        if done and row.source_kind == "letter" and row.source_ref.isdigit():
            request = repo.get_recommendation_request(conn, int(row.source_ref))
            if request is not None and tracker.normalize_request_status(request.status) == "planned":
                tracker.set_request_status(conn, request.id, "asked", today=today)
        return set_todo_done(conn, row.ref_id, done, done_by=done_by, today=today)
    if row.kind == "checklist":
        return repo.update_checklist_item(conn, row.ref_id, done=done)
    if not done:
        return False
    if row.kind == "deadline":
        tracker.set_status(conn, row.ref_id, "submitted", today=today)
        return True
    if row.kind == "letter":
        request = repo.get_recommendation_request(conn, row.ref_id)
        step = (
            None
            if request is None
            else _LETTER_NEXT_STEP.get(tracker.normalize_request_status(request.status))
        )
        if step is None:
            return False
        tracker.set_request_status(conn, row.ref_id, step, today=today)
        return True
    return False


ESSAY_REF_PREFIX = "essay:"


def essay_source_ref(essay_id: int) -> str:
    """``source_ref`` of a parent's suggestion on an essay (``source_kind`` is manual)."""
    return f"{ESSAY_REF_PREFIX}{essay_id}"


def is_essay_ref(source_ref: str) -> bool:
    return source_ref.startswith(ESSAY_REF_PREFIX)


def suggest_essay_change(
    conn: sqlite3.Connection,
    essay: repo.Essay,
    suggestion: str,
    *,
    created_by: str = "",
    created_role: str = "parent",
) -> Todo:
    """A parent's note on a shared essay, as a to-do for her that links the essay."""
    return create_todo(
        conn,
        essay.student_id,
        f"Look at a suggestion on “{essay.title}”",
        notes=suggestion.strip(),
        assignee="student",
        created_by=created_by,
        created_role=created_role,
        source_ref=essay_source_ref(essay.id),
    )


def delete_open_essay_suggestions(conn: sqlite3.Connection, essay_id: int) -> int:
    """Drop the open suggestions on an essay; ticked ones stay as history."""
    cursor = conn.execute(
        "DELETE FROM todos WHERE source_kind = ? AND source_ref = ? AND done_on IS NULL",
        (DEFAULT_SOURCE_KIND, essay_source_ref(essay_id)),
    )
    conn.commit()
    return cursor.rowcount


def letter_todo_title(who: str, due: date) -> str:
    return f"Ask {who} by {due:%b} {due.day}"


def sync_letter_todo(
    conn: sqlite3.Connection,
    student_id: str,
    request_id: int,
    *,
    created_by: str = "",
    created_role: str = "",
    today: date | None = None,
) -> Todo | None:
    """Keep her "Ask ... by ..." to-do in step with one letter request.

    A planned request with a due date has one open row, renamed and redated as
    the request changes.  Once she has asked it is ticked; a request that loses
    its date or is removed takes its open row with it.
    """
    existing = find_todo(conn, student_id, "letter", str(request_id))
    request = repo.get_recommendation_request(conn, request_id)
    due = None if request is None else _parse_date(request.due_on)
    if request is None or due is None:
        if existing is not None and not existing.done:
            delete_todo(conn, existing.id)
        return None
    if tracker.normalize_request_status(request.status) != "planned":
        if existing is not None and not existing.done:
            set_todo_done(conn, existing.id, True, today=today)
        return existing

    recommender = repo.get_recommender(conn, request.recommender_id)
    title = letter_todo_title(recommender.name if recommender else "your recommender", due)
    application = repo.get_application(conn, request.application_id)
    notes = (
        f"Letter for {application.title or application.catalog_id}" if application else ""
    )
    if existing is None:
        return create_todo(
            conn,
            student_id,
            title,
            notes=notes,
            due_on=due.isoformat(),
            assignee="student",
            created_by=created_by,
            created_role=created_role,
            source_kind="letter",
            source_ref=str(request_id),
        )
    update_todo(conn, existing.id, title=title, notes=notes, due_on=due.isoformat())
    if existing.done:
        set_todo_done(conn, existing.id, False)
    return get_todo(conn, existing.id)


def milestone_source_ref(milestone_id: str, year_end: int) -> str:
    return f"{milestone_id}:{year_end}"


def sync_milestone_todos(
    conn: sqlite3.Connection,
    student_id: str,
    grade_level: str | None,
    *,
    today: date,
    milestones: list[milestone_store.Milestone] | None = None,
) -> list[Todo]:
    """Write in her grade's milestones that fall due in the Coming up window.

    Each one is written once per school year (``source_ref`` carries the year),
    so a tick or an edit sticks and next year's lands on its own.  A window
    counts by its last day.  With no grade on the profile nothing is written:
    every milestone would apply, and a FAFSA row is noise to a sophomore.
    """
    if not str(grade_level or "").strip():
        return []
    resolved = milestones if milestones is not None else milestone_store.load_milestones(conn)
    current = milestone_store.current_school_year(today)
    created: list[Todo] = []
    for year_end in (current, current + 1):
        for found in milestone_store.occurrences_for_school_year(
            resolved, grade_level, year_end, today=today
        ):
            due = found.ends_on or found.starts_on
            if not 0 <= (due - today).days <= COMING_UP_DAYS:
                continue
            ref = milestone_source_ref(found.milestone.id, year_end)
            if find_todo(conn, student_id, "milestone", ref) is not None:
                continue
            created.append(
                create_todo(
                    conn,
                    student_id,
                    found.milestone.title,
                    notes=found.milestone.note,
                    due_on=due.isoformat(),
                    assignee="student",
                    source_kind="milestone",
                    source_ref=ref,
                )
            )
    return created
