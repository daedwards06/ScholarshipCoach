"""Typed reads and writes over the family database.

Every statement that touches ``coach.db`` lives here.  The app layer calls
functions that take and return dataclasses, so a schema change is a change to
this module and its migrations rather than a hunt through Streamlit callbacks.

``None`` is a real value everywhere (an unset amount, a cleared due date), so
the update functions use a sentinel instead: an argument you do not pass
leaves its column alone, and passing ``None`` writes ``NULL``.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.store.db import utc_now

UNSET: Any = object()


@dataclass(slots=True)
class Student:
    student_id: str
    name: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class Application:
    id: int
    student_id: str
    catalog_id: str
    status: str = "saved"
    notes: str = ""
    title: str = ""
    source_url: str = ""
    deadline: str | None = None
    submitted_on: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class ChecklistItem:
    id: int
    application_id: int
    label: str
    done: bool = False
    due_on: str | None = None
    position: int = 0
    created_at: str = ""
    updated_at: str = ""


ESSAY_THEMES: tuple[str, ...] = (
    "challenge",
    "leadership",
    "why_major",
    "community",
    "identity",
    "other",
)

ESSAY_THEME_LABELS: dict[str, str] = {
    "challenge": "Challenge",
    "leadership": "Leadership",
    "why_major": "Why this major",
    "community": "Community",
    "identity": "Identity",
    "other": "Other",
}

DEFAULT_ESSAY_THEME = "other"


def normalize_theme(value: object) -> str:
    """Coerce anything to a known essay theme, falling back to ``other``."""
    text = str(value or "").strip().casefold().replace(" ", "_")
    return text if text in ESSAY_THEMES else DEFAULT_ESSAY_THEME


@dataclass(slots=True)
class Essay:
    id: int
    student_id: str
    title: str
    body: str = ""
    theme: str = DEFAULT_ESSAY_THEME
    word_count: int = 0
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class EssayVersion:
    """One saved draft of an essay, kept so earlier text can be read back."""

    id: int
    essay_id: int
    title: str
    body: str = ""
    word_count: int = 0
    created_at: str = ""


@dataclass(slots=True)
class EssayLink:
    id: int
    essay_id: int
    application_id: int
    prompt: str = ""
    created_at: str = ""


@dataclass(slots=True)
class Recommender:
    id: int
    student_id: str
    name: str
    role: str = ""
    email: str = ""
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class RecommendationRequest:
    id: int
    recommender_id: int
    application_id: int
    status: str = "planned"
    asked_on: str | None = None
    due_on: str | None = None
    received_on: str | None = None
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class Outcome:
    id: int
    application_id: int
    result: str = "pending"
    amount_awarded: float | None = None
    paid_to: str = ""
    renewal_terms: str = ""
    decided_on: str | None = None
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""


COLLEGE_DEADLINE_TYPES: tuple[str, ...] = (
    "",
    "early_decision",
    "early_action",
    "regular",
    "rolling",
    "priority",
)

COLLEGE_DEADLINE_TYPE_LABELS: dict[str, str] = {
    "": "Not set",
    "early_decision": "Early decision",
    "early_action": "Early action",
    "regular": "Regular decision",
    "rolling": "Rolling",
    "priority": "Priority",
}


def normalize_deadline_type(value: object) -> str:
    text = str(value or "").strip().casefold().replace(" ", "_")
    return text if text in COLLEGE_DEADLINE_TYPES else ""


@dataclass(slots=True)
class College:
    id: int
    student_id: str
    name: str
    status: str = "considering"
    cost_of_attendance: float | None = None
    aid_offered: float | None = None
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""
    in_state: bool = False
    net_price_estimate: float | None = None
    merit_aid_notes: str = ""
    outside_award_policy: str = ""
    deadline_type: str = ""


def _set(**values: Any) -> dict[str, Any]:
    """Drop the arguments the caller did not pass."""
    return {name: value for name, value in values.items() if value is not UNSET}


def _insert(conn: sqlite3.Connection, table: str, values: Mapping[str, Any]) -> int:
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    cursor = conn.execute(
        f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", tuple(values.values())
    )
    conn.commit()
    return int(cursor.lastrowid or 0)


def _update(
    conn: sqlite3.Connection, table: str, row_id: int, changes: Mapping[str, Any]
) -> bool:
    if not changes:
        return False
    assignments = ", ".join(f"{column} = ?" for column in changes)
    cursor = conn.execute(
        f"UPDATE {table} SET {assignments}, updated_at = ? WHERE id = ?",
        (*changes.values(), utc_now(), row_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def _delete(conn: sqlite3.Connection, table: str, row_id: int) -> bool:
    cursor = conn.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
    conn.commit()
    return cursor.rowcount > 0


def _fetch_one(
    conn: sqlite3.Connection, sql: str, params: tuple[Any, ...]
) -> sqlite3.Row | None:
    row: sqlite3.Row | None = conn.execute(sql, params).fetchone()
    return row


# -- students ---------------------------------------------------------------


def _to_student(row: sqlite3.Row) -> Student:
    return Student(
        student_id=str(row["student_id"]),
        name=str(row["name"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def upsert_student(conn: sqlite3.Connection, student_id: str, name: str = "") -> Student:
    """Create ``student_id`` or update its name, and return the stored row."""
    now = utc_now()
    conn.execute(
        "INSERT INTO students (student_id, name, created_at, updated_at) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT (student_id) DO UPDATE SET "
        "name = excluded.name, updated_at = excluded.updated_at",
        (student_id, name, now, now),
    )
    conn.commit()
    stored = get_student(conn, student_id)
    if stored is None:  # pragma: no cover - the insert above guarantees a row
        raise RuntimeError(f"Student {student_id!r} vanished after upsert.")
    return stored


def get_student(conn: sqlite3.Connection, student_id: str) -> Student | None:
    row = _fetch_one(conn, "SELECT * FROM students WHERE student_id = ?", (student_id,))
    return None if row is None else _to_student(row)


def list_students(conn: sqlite3.Connection) -> list[Student]:
    rows = conn.execute("SELECT * FROM students ORDER BY student_id").fetchall()
    return [_to_student(row) for row in rows]


def delete_student(conn: sqlite3.Connection, student_id: str) -> bool:
    """Delete a student and, by cascade, everything that belongs to them."""
    cursor = conn.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
    conn.commit()
    return cursor.rowcount > 0


# -- applications -----------------------------------------------------------


def _to_application(row: sqlite3.Row) -> Application:
    deadline = row["deadline"]
    submitted_on = row["submitted_on"]
    return Application(
        id=int(row["id"]),
        student_id=str(row["student_id"]),
        catalog_id=str(row["catalog_id"]),
        status=str(row["status"]),
        notes=str(row["notes"]),
        title=str(row["title"]),
        source_url=str(row["source_url"]),
        deadline=None if deadline is None else str(deadline),
        submitted_on=None if submitted_on is None else str(submitted_on),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def create_application(
    conn: sqlite3.Connection,
    student_id: str,
    catalog_id: str,
    status: str = "saved",
    notes: str = "",
    title: str = "",
    source_url: str = "",
    deadline: str | None = None,
) -> Application:
    now = utc_now()
    row_id = _insert(
        conn,
        "applications",
        {
            "student_id": student_id,
            "catalog_id": catalog_id,
            "status": status,
            "notes": notes,
            "title": title,
            "source_url": source_url,
            "deadline": deadline,
            "submitted_on": None,
            "created_at": now,
            "updated_at": now,
        },
    )
    return Application(
        row_id,
        student_id,
        catalog_id,
        status,
        notes,
        title,
        source_url,
        deadline,
        None,
        now,
        now,
    )


def get_application(conn: sqlite3.Connection, application_id: int) -> Application | None:
    row = _fetch_one(conn, "SELECT * FROM applications WHERE id = ?", (application_id,))
    return None if row is None else _to_application(row)


def find_application(
    conn: sqlite3.Connection, student_id: str, catalog_id: str
) -> Application | None:
    """Return the student's application for one award, if they saved it."""
    row = _fetch_one(
        conn,
        "SELECT * FROM applications WHERE student_id = ? AND catalog_id = ?",
        (student_id, catalog_id),
    )
    return None if row is None else _to_application(row)


def list_applications(
    conn: sqlite3.Connection, student_id: str, status: str | None = None
) -> list[Application]:
    if status is None:
        rows = conn.execute(
            "SELECT * FROM applications WHERE student_id = ? ORDER BY created_at, id",
            (student_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM applications WHERE student_id = ? AND status = ? "
            "ORDER BY created_at, id",
            (student_id, status),
        ).fetchall()
    return [_to_application(row) for row in rows]


def update_application(
    conn: sqlite3.Connection,
    application_id: int,
    *,
    status: str = UNSET,
    notes: str = UNSET,
    title: str = UNSET,
    source_url: str = UNSET,
    deadline: str | None = UNSET,
    submitted_on: str | None = UNSET,
) -> bool:
    return _update(
        conn,
        "applications",
        application_id,
        _set(
            status=status,
            notes=notes,
            title=title,
            source_url=source_url,
            deadline=deadline,
            submitted_on=submitted_on,
        ),
    )


def delete_application(conn: sqlite3.Connection, application_id: int) -> bool:
    return _delete(conn, "applications", application_id)


# -- checklist items --------------------------------------------------------


def _to_checklist_item(row: sqlite3.Row) -> ChecklistItem:
    due_on = row["due_on"]
    return ChecklistItem(
        id=int(row["id"]),
        application_id=int(row["application_id"]),
        label=str(row["label"]),
        done=bool(row["done"]),
        due_on=None if due_on is None else str(due_on),
        position=int(row["position"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def create_checklist_item(
    conn: sqlite3.Connection,
    application_id: int,
    label: str,
    due_on: str | None = None,
    position: int = 0,
    done: bool = False,
) -> ChecklistItem:
    now = utc_now()
    row_id = _insert(
        conn,
        "checklist_items",
        {
            "application_id": application_id,
            "label": label,
            "done": int(done),
            "due_on": due_on,
            "position": position,
            "created_at": now,
            "updated_at": now,
        },
    )
    return ChecklistItem(row_id, application_id, label, done, due_on, position, now, now)


def get_checklist_item(conn: sqlite3.Connection, item_id: int) -> ChecklistItem | None:
    row = _fetch_one(conn, "SELECT * FROM checklist_items WHERE id = ?", (item_id,))
    return None if row is None else _to_checklist_item(row)


def list_checklist_items(conn: sqlite3.Connection, application_id: int) -> list[ChecklistItem]:
    rows = conn.execute(
        "SELECT * FROM checklist_items WHERE application_id = ? ORDER BY position, id",
        (application_id,),
    ).fetchall()
    return [_to_checklist_item(row) for row in rows]


def update_checklist_item(
    conn: sqlite3.Connection,
    item_id: int,
    *,
    label: str = UNSET,
    done: bool = UNSET,
    due_on: str | None = UNSET,
    position: int = UNSET,
) -> bool:
    changes = _set(label=label, done=done, due_on=due_on, position=position)
    if "done" in changes:
        changes["done"] = int(bool(changes["done"]))
    return _update(conn, "checklist_items", item_id, changes)


def delete_checklist_item(conn: sqlite3.Connection, item_id: int) -> bool:
    return _delete(conn, "checklist_items", item_id)


# -- essays -----------------------------------------------------------------


def _to_essay(row: sqlite3.Row) -> Essay:
    return Essay(
        id=int(row["id"]),
        student_id=str(row["student_id"]),
        title=str(row["title"]),
        body=str(row["body"]),
        theme=str(row["theme"]),
        word_count=int(row["word_count"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _to_essay_version(row: sqlite3.Row) -> EssayVersion:
    return EssayVersion(
        id=int(row["id"]),
        essay_id=int(row["essay_id"]),
        title=str(row["title"]),
        body=str(row["body"]),
        word_count=int(row["word_count"]),
        created_at=str(row["created_at"]),
    )


def word_count(body: str) -> int:
    """Count words the way a word limit does: whitespace-separated tokens."""
    return len(body.split())


def _record_essay_version(
    conn: sqlite3.Connection, essay_id: int, title: str, body: str, when: str
) -> None:
    _insert(
        conn,
        "essay_versions",
        {
            "essay_id": essay_id,
            "title": title,
            "body": body,
            "word_count": word_count(body),
            "created_at": when,
        },
    )


def create_essay(
    conn: sqlite3.Connection,
    student_id: str,
    title: str,
    body: str = "",
    theme: str = DEFAULT_ESSAY_THEME,
) -> Essay:
    now = utc_now()
    words = word_count(body)
    tag = normalize_theme(theme)
    row_id = _insert(
        conn,
        "essays",
        {
            "student_id": student_id,
            "title": title,
            "body": body,
            "theme": tag,
            "word_count": words,
            "created_at": now,
            "updated_at": now,
        },
    )
    _record_essay_version(conn, row_id, title, body, now)
    return Essay(row_id, student_id, title, body, tag, words, now, now)


def get_essay(conn: sqlite3.Connection, essay_id: int) -> Essay | None:
    row = _fetch_one(conn, "SELECT * FROM essays WHERE id = ?", (essay_id,))
    return None if row is None else _to_essay(row)


def list_essays(conn: sqlite3.Connection, student_id: str) -> list[Essay]:
    rows = conn.execute(
        "SELECT * FROM essays WHERE student_id = ? ORDER BY updated_at DESC, id",
        (student_id,),
    ).fetchall()
    return [_to_essay(row) for row in rows]


def update_essay(
    conn: sqlite3.Connection,
    essay_id: int,
    *,
    title: str = UNSET,
    body: str = UNSET,
    theme: str = UNSET,
) -> bool:
    """Save an edit, recording a version row whenever the text itself changes.

    A theme change is filing, not writing, so it does not add a version.
    """
    changes = _set(title=title, body=body, theme=theme)
    if "theme" in changes:
        changes["theme"] = normalize_theme(changes["theme"])
    if "body" in changes:
        changes["word_count"] = word_count(str(changes["body"]))

    current = get_essay(conn, essay_id)
    if current is None:
        return False
    new_title = str(changes.get("title", current.title))
    new_body = str(changes.get("body", current.body))
    text_changed = (new_title, new_body) != (current.title, current.body)

    updated = _update(conn, "essays", essay_id, changes)
    if updated and text_changed:
        _record_essay_version(conn, essay_id, new_title, new_body, utc_now())
    return updated


def list_essay_versions(conn: sqlite3.Connection, essay_id: int) -> list[EssayVersion]:
    """Every saved draft of an essay, newest first."""
    rows = conn.execute(
        "SELECT * FROM essay_versions WHERE essay_id = ? ORDER BY id DESC", (essay_id,)
    ).fetchall()
    return [_to_essay_version(row) for row in rows]


def delete_essay(conn: sqlite3.Connection, essay_id: int) -> bool:
    return _delete(conn, "essays", essay_id)


# -- essay links ------------------------------------------------------------


def _to_essay_link(row: sqlite3.Row) -> EssayLink:
    return EssayLink(
        id=int(row["id"]),
        essay_id=int(row["essay_id"]),
        application_id=int(row["application_id"]),
        prompt=str(row["prompt"]),
        created_at=str(row["created_at"]),
    )


def link_essay(
    conn: sqlite3.Connection, essay_id: int, application_id: int, prompt: str = ""
) -> EssayLink:
    """Attach an essay to an award's prompt, replacing any earlier pairing."""
    now = utc_now()
    conn.execute(
        "INSERT INTO essay_links (essay_id, application_id, prompt, created_at) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT (essay_id, application_id) DO UPDATE SET prompt = excluded.prompt",
        (essay_id, application_id, prompt, now),
    )
    conn.commit()
    row = _fetch_one(
        conn,
        "SELECT * FROM essay_links WHERE essay_id = ? AND application_id = ?",
        (essay_id, application_id),
    )
    if row is None:  # pragma: no cover - the insert above guarantees a row
        raise RuntimeError("Essay link vanished after insert.")
    return _to_essay_link(row)


def list_essay_links(
    conn: sqlite3.Connection,
    *,
    essay_id: int | None = None,
    application_id: int | None = None,
) -> list[EssayLink]:
    clauses: list[str] = []
    params: list[Any] = []
    if essay_id is not None:
        clauses.append("essay_id = ?")
        params.append(essay_id)
    if application_id is not None:
        clauses.append("application_id = ?")
        params.append(application_id)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM essay_links{where} ORDER BY id", tuple(params)
    ).fetchall()
    return [_to_essay_link(row) for row in rows]


def unlink_essay(conn: sqlite3.Connection, essay_id: int, application_id: int) -> bool:
    cursor = conn.execute(
        "DELETE FROM essay_links WHERE essay_id = ? AND application_id = ?",
        (essay_id, application_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def essay_reuse_counts(conn: sqlite3.Connection, student_id: str) -> dict[int, int]:
    """How many applications each of a student's essays is linked to."""
    rows = conn.execute(
        "SELECT e.id AS essay_id, COUNT(l.id) AS uses "
        "FROM essays AS e LEFT JOIN essay_links AS l ON l.essay_id = e.id "
        "WHERE e.student_id = ? GROUP BY e.id",
        (student_id,),
    ).fetchall()
    return {int(row["essay_id"]): int(row["uses"]) for row in rows}


# -- recommenders -----------------------------------------------------------


def _to_recommender(row: sqlite3.Row) -> Recommender:
    return Recommender(
        id=int(row["id"]),
        student_id=str(row["student_id"]),
        name=str(row["name"]),
        role=str(row["role"]),
        email=str(row["email"]),
        notes=str(row["notes"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def create_recommender(
    conn: sqlite3.Connection,
    student_id: str,
    name: str,
    role: str = "",
    email: str = "",
    notes: str = "",
) -> Recommender:
    now = utc_now()
    row_id = _insert(
        conn,
        "recommenders",
        {
            "student_id": student_id,
            "name": name,
            "role": role,
            "email": email,
            "notes": notes,
            "created_at": now,
            "updated_at": now,
        },
    )
    return Recommender(row_id, student_id, name, role, email, notes, now, now)


def get_recommender(conn: sqlite3.Connection, recommender_id: int) -> Recommender | None:
    row = _fetch_one(conn, "SELECT * FROM recommenders WHERE id = ?", (recommender_id,))
    return None if row is None else _to_recommender(row)


def list_recommenders(conn: sqlite3.Connection, student_id: str) -> list[Recommender]:
    rows = conn.execute(
        "SELECT * FROM recommenders WHERE student_id = ? ORDER BY name, id", (student_id,)
    ).fetchall()
    return [_to_recommender(row) for row in rows]


def update_recommender(
    conn: sqlite3.Connection,
    recommender_id: int,
    *,
    name: str = UNSET,
    role: str = UNSET,
    email: str = UNSET,
    notes: str = UNSET,
) -> bool:
    return _update(
        conn,
        "recommenders",
        recommender_id,
        _set(name=name, role=role, email=email, notes=notes),
    )


def delete_recommender(conn: sqlite3.Connection, recommender_id: int) -> bool:
    return _delete(conn, "recommenders", recommender_id)


# -- recommendation requests ------------------------------------------------


def _to_recommendation_request(row: sqlite3.Row) -> RecommendationRequest:
    def _optional(column: str) -> str | None:
        value = row[column]
        return None if value is None else str(value)

    return RecommendationRequest(
        id=int(row["id"]),
        recommender_id=int(row["recommender_id"]),
        application_id=int(row["application_id"]),
        status=str(row["status"]),
        asked_on=_optional("asked_on"),
        due_on=_optional("due_on"),
        received_on=_optional("received_on"),
        notes=str(row["notes"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def create_recommendation_request(
    conn: sqlite3.Connection,
    recommender_id: int,
    application_id: int,
    status: str = "planned",
    asked_on: str | None = None,
    due_on: str | None = None,
    notes: str = "",
) -> RecommendationRequest:
    now = utc_now()
    row_id = _insert(
        conn,
        "recommendation_requests",
        {
            "recommender_id": recommender_id,
            "application_id": application_id,
            "status": status,
            "asked_on": asked_on,
            "due_on": due_on,
            "received_on": None,
            "notes": notes,
            "created_at": now,
            "updated_at": now,
        },
    )
    return RecommendationRequest(
        row_id,
        recommender_id,
        application_id,
        status,
        asked_on,
        due_on,
        None,
        notes,
        now,
        now,
    )


def get_recommendation_request(
    conn: sqlite3.Connection, request_id: int
) -> RecommendationRequest | None:
    row = _fetch_one(conn, "SELECT * FROM recommendation_requests WHERE id = ?", (request_id,))
    return None if row is None else _to_recommendation_request(row)


def list_recommendation_requests(
    conn: sqlite3.Connection,
    *,
    recommender_id: int | None = None,
    application_id: int | None = None,
) -> list[RecommendationRequest]:
    clauses: list[str] = []
    params: list[Any] = []
    if recommender_id is not None:
        clauses.append("recommender_id = ?")
        params.append(recommender_id)
    if application_id is not None:
        clauses.append("application_id = ?")
        params.append(application_id)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM recommendation_requests{where} ORDER BY id", tuple(params)
    ).fetchall()
    return [_to_recommendation_request(row) for row in rows]


def list_student_recommendation_requests(
    conn: sqlite3.Connection, student_id: str
) -> list[RecommendationRequest]:
    """Every request a student has open, across recommenders and applications."""
    rows = conn.execute(
        "SELECT r.* FROM recommendation_requests AS r "
        "JOIN recommenders AS p ON p.id = r.recommender_id "
        "WHERE p.student_id = ? ORDER BY r.due_on IS NULL, r.due_on, r.id",
        (student_id,),
    ).fetchall()
    return [_to_recommendation_request(row) for row in rows]


def update_recommendation_request(
    conn: sqlite3.Connection,
    request_id: int,
    *,
    status: str = UNSET,
    asked_on: str | None = UNSET,
    due_on: str | None = UNSET,
    received_on: str | None = UNSET,
    notes: str = UNSET,
) -> bool:
    return _update(
        conn,
        "recommendation_requests",
        request_id,
        _set(
            status=status,
            asked_on=asked_on,
            due_on=due_on,
            received_on=received_on,
            notes=notes,
        ),
    )


def delete_recommendation_request(conn: sqlite3.Connection, request_id: int) -> bool:
    return _delete(conn, "recommendation_requests", request_id)


# -- outcomes ---------------------------------------------------------------


def _to_outcome(row: sqlite3.Row) -> Outcome:
    amount = row["amount_awarded"]
    decided_on = row["decided_on"]
    return Outcome(
        id=int(row["id"]),
        application_id=int(row["application_id"]),
        result=str(row["result"]),
        amount_awarded=None if amount is None else float(amount),
        paid_to=str(row["paid_to"]),
        renewal_terms=str(row["renewal_terms"]),
        decided_on=None if decided_on is None else str(decided_on),
        notes=str(row["notes"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def set_outcome(
    conn: sqlite3.Connection,
    application_id: int,
    result: str = "pending",
    amount_awarded: float | None = None,
    paid_to: str = "",
    renewal_terms: str = "",
    decided_on: str | None = None,
    notes: str = "",
) -> Outcome:
    """Record the result of an application, replacing any earlier result."""
    now = utc_now()
    conn.execute(
        "INSERT INTO outcomes (application_id, result, amount_awarded, paid_to, "
        "renewal_terms, decided_on, notes, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (application_id) DO UPDATE SET "
        "result = excluded.result, amount_awarded = excluded.amount_awarded, "
        "paid_to = excluded.paid_to, renewal_terms = excluded.renewal_terms, "
        "decided_on = excluded.decided_on, notes = excluded.notes, "
        "updated_at = excluded.updated_at",
        (
            application_id,
            result,
            amount_awarded,
            paid_to,
            renewal_terms,
            decided_on,
            notes,
            now,
            now,
        ),
    )
    conn.commit()
    stored = get_outcome(conn, application_id)
    if stored is None:  # pragma: no cover - the insert above guarantees a row
        raise RuntimeError(f"Outcome for application {application_id} vanished after upsert.")
    return stored


def get_outcome(conn: sqlite3.Connection, application_id: int) -> Outcome | None:
    row = _fetch_one(conn, "SELECT * FROM outcomes WHERE application_id = ?", (application_id,))
    return None if row is None else _to_outcome(row)


def list_outcomes(conn: sqlite3.Connection, student_id: str) -> list[Outcome]:
    rows = conn.execute(
        "SELECT outcomes.* FROM outcomes "
        "JOIN applications ON applications.id = outcomes.application_id "
        "WHERE applications.student_id = ? ORDER BY outcomes.id",
        (student_id,),
    ).fetchall()
    return [_to_outcome(row) for row in rows]


def delete_outcome(conn: sqlite3.Connection, application_id: int) -> bool:
    cursor = conn.execute("DELETE FROM outcomes WHERE application_id = ?", (application_id,))
    conn.commit()
    return cursor.rowcount > 0


# -- colleges ---------------------------------------------------------------


def _to_college(row: sqlite3.Row) -> College:
    cost = row["cost_of_attendance"]
    aid = row["aid_offered"]
    estimate = row["net_price_estimate"]
    return College(
        id=int(row["id"]),
        student_id=str(row["student_id"]),
        name=str(row["name"]),
        status=str(row["status"]),
        cost_of_attendance=None if cost is None else float(cost),
        aid_offered=None if aid is None else float(aid),
        notes=str(row["notes"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        in_state=bool(row["in_state"]),
        net_price_estimate=None if estimate is None else float(estimate),
        merit_aid_notes=str(row["merit_aid_notes"]),
        outside_award_policy=str(row["outside_award_policy"]),
        deadline_type=normalize_deadline_type(row["deadline_type"]),
    )


def create_college(
    conn: sqlite3.Connection,
    student_id: str,
    name: str,
    status: str = "considering",
    cost_of_attendance: float | None = None,
    aid_offered: float | None = None,
    notes: str = "",
    *,
    in_state: bool = False,
    net_price_estimate: float | None = None,
    merit_aid_notes: str = "",
    outside_award_policy: str = "",
    deadline_type: str = "",
) -> College:
    now = utc_now()
    deadline = normalize_deadline_type(deadline_type)
    row_id = _insert(
        conn,
        "colleges",
        {
            "student_id": student_id,
            "name": name,
            "status": status,
            "cost_of_attendance": cost_of_attendance,
            "aid_offered": aid_offered,
            "notes": notes,
            "created_at": now,
            "updated_at": now,
            "in_state": int(in_state),
            "net_price_estimate": net_price_estimate,
            "merit_aid_notes": merit_aid_notes,
            "outside_award_policy": outside_award_policy,
            "deadline_type": deadline,
        },
    )
    return College(
        row_id,
        student_id,
        name,
        status,
        cost_of_attendance,
        aid_offered,
        notes,
        now,
        now,
        in_state=in_state,
        net_price_estimate=net_price_estimate,
        merit_aid_notes=merit_aid_notes,
        outside_award_policy=outside_award_policy,
        deadline_type=deadline,
    )


def get_college(conn: sqlite3.Connection, college_id: int) -> College | None:
    row = _fetch_one(conn, "SELECT * FROM colleges WHERE id = ?", (college_id,))
    return None if row is None else _to_college(row)


def list_colleges(conn: sqlite3.Connection, student_id: str) -> list[College]:
    rows = conn.execute(
        "SELECT * FROM colleges WHERE student_id = ? ORDER BY name, id", (student_id,)
    ).fetchall()
    return [_to_college(row) for row in rows]


def update_college(
    conn: sqlite3.Connection,
    college_id: int,
    *,
    name: str = UNSET,
    status: str = UNSET,
    cost_of_attendance: float | None = UNSET,
    aid_offered: float | None = UNSET,
    notes: str = UNSET,
    in_state: bool = UNSET,
    net_price_estimate: float | None = UNSET,
    merit_aid_notes: str = UNSET,
    outside_award_policy: str = UNSET,
    deadline_type: str = UNSET,
) -> bool:
    return _update(
        conn,
        "colleges",
        college_id,
        _set(
            name=name,
            status=status,
            cost_of_attendance=cost_of_attendance,
            aid_offered=aid_offered,
            notes=notes,
            in_state=in_state if in_state is UNSET else int(in_state),
            net_price_estimate=net_price_estimate,
            merit_aid_notes=merit_aid_notes,
            outside_award_policy=outside_award_policy,
            deadline_type=(
                deadline_type
                if deadline_type is UNSET
                else normalize_deadline_type(deadline_type)
            ),
        ),
    )


def delete_college(conn: sqlite3.Connection, college_id: int) -> bool:
    return _delete(conn, "colleges", college_id)


def net_price(college: College) -> float | None:
    """What this school actually costs, or ``None`` when nothing says.

    A hand-entered estimate from the school's own net price calculator wins:
    it already accounts for aid no public number predicts.  Otherwise fall
    back to sticker minus the aid on offer.
    """
    if college.net_price_estimate is not None:
        return college.net_price_estimate
    if college.cost_of_attendance is None or college.aid_offered is None:
        return None
    return college.cost_of_attendance - college.aid_offered


# -- settings ---------------------------------------------------------------


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value, "
        "updated_at = excluded.updated_at",
        (key, value, utc_now()),
    )
    conn.commit()


def get_setting(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = _fetch_one(conn, "SELECT value FROM settings WHERE key = ?", (key,))
    return default if row is None else str(row["value"])


def set_flag(conn: sqlite3.Connection, key: str, value: bool) -> None:
    set_setting(conn, key, "true" if value else "false")


def get_flag(conn: sqlite3.Connection, key: str, default: bool = False) -> bool:
    """Read a setting written by :func:`set_flag`."""
    value = get_setting(conn, key)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def all_settings(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT key, value FROM settings ORDER BY key").fetchall()
    return {str(row["key"]): str(row["value"]) for row in rows}


def delete_setting(conn: sqlite3.Connection, key: str) -> bool:
    cursor = conn.execute("DELETE FROM settings WHERE key = ?", (key,))
    conn.commit()
    return cursor.rowcount > 0
