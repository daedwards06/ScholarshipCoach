"""Activities & Honors: what she applied to, got into, and does.

A program from the catalog is a row here from the moment she applies (ticking
its "Apply to …" to-do), and its result moves along a small graph in the
tracker's style: applied, then accepted, waitlisted or not accepted, and once
accepted, completed or declined.  Each status can step back one, because a
misclick should not be permanent.  A row is one application cycle (``term``
starts as the school year), so a program she was not accepted to can be
applied to again next year as a new row.

Clubs, awards and jobs she enters by hand have no ``catalog_id`` and are only
``active`` or ``completed``.

Programs never become applications, outcomes or win-model rows: those are the
money views, and this list is hers.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta

from src.profile.grade_levels import school_year_end
from src.store.db import utc_now
from src.store.repo import UNSET, _delete, _fetch_one, _insert, _set, _update
from src.store.tracker import TransitionError

KINDS: tuple[str, ...] = ("program", "club", "award", "job", "volunteer", "other")

KIND_LABELS: dict[str, str] = {
    "program": "Program",
    "club": "Club",
    "award": "Award or honor",
    "job": "Job",
    "volunteer": "Volunteering",
    "other": "Other",
}

DEFAULT_KIND = "other"

STATUSES: tuple[str, ...] = (
    "applied",
    "accepted",
    "waitlisted",
    "not_accepted",
    "declined",
    "completed",
    "active",
)

STATUS_LABELS: dict[str, str] = {
    "applied": "Applied",
    "accepted": "Accepted",
    "waitlisted": "Waitlisted",
    "not_accepted": "Not accepted",
    "declined": "Declined",
    "completed": "Completed",
    "active": "Active",
}

# A catalog program's result.  Each status's last entry is its step back,
# except applied, which has nothing before it.
PROGRAM_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "applied": ("accepted", "waitlisted", "not_accepted"),
    "waitlisted": ("accepted", "not_accepted", "applied"),
    "accepted": ("completed", "declined", "applied"),
    "not_accepted": ("applied",),
    "declined": ("accepted",),
    "completed": ("accepted",),
}

HAND_ENTERED_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "active": ("completed",),
    "completed": ("active",),
}

HAND_ENTERED_STATUSES: tuple[str, ...] = tuple(HAND_ENTERED_TRANSITIONS)

# What goes on the brag sheet, and what moves a program to Opportunities' Yours.
HONORS_STATUSES: tuple[str, ...] = ("accepted", "completed", "active")
YOURS_STATUSES: tuple[str, ...] = ("accepted", "completed")


def normalize_kind(value: object) -> str:
    text = str(value or "").strip().casefold()
    return text if text in KINDS else DEFAULT_KIND


def normalize_status(value: object, default: str = "active") -> str:
    text = str(value or "").strip().casefold().replace(" ", "_")
    return text if text in STATUSES else default


@dataclass(slots=True)
class Experience:
    id: int
    student_id: str
    title: str
    catalog_id: str | None = None
    organization: str = ""
    kind: str = DEFAULT_KIND
    status: str = "active"
    starts_on: str | None = None
    ends_on: str | None = None
    term: str = ""
    hours_per_week: float | None = None
    notes: str = ""
    created_by: str = ""
    created_role: str = ""
    created_at: str = ""
    updated_at: str = ""

    @property
    def from_catalog(self) -> bool:
        return self.catalog_id is not None


def _transitions(experience: Experience) -> dict[str, tuple[str, ...]]:
    return PROGRAM_TRANSITIONS if experience.from_catalog else HAND_ENTERED_TRANSITIONS


def can_transition(experience: Experience, target: object) -> bool:
    """True when ``experience`` may move to ``target`` (staying put always may)."""
    end = normalize_status(target, default="")
    graph = _transitions(experience)
    return end == experience.status or end in graph.get(experience.status, ())


def next_statuses(experience: Experience) -> tuple[str, ...]:
    """The statuses a picker should offer, current first."""
    return (experience.status, *_transitions(experience).get(experience.status, ()))


def _optional(row: sqlite3.Row, column: str) -> str | None:
    value = row[column]
    return None if value is None else str(value)


def _to_experience(row: sqlite3.Row) -> Experience:
    hours = row["hours_per_week"]
    return Experience(
        id=int(row["id"]),
        student_id=str(row["student_id"]),
        title=str(row["title"]),
        catalog_id=_optional(row, "catalog_id"),
        organization=str(row["organization"]),
        kind=normalize_kind(row["kind"]),
        status=normalize_status(row["status"]),
        starts_on=_optional(row, "starts_on"),
        ends_on=_optional(row, "ends_on"),
        term=str(row["term"]),
        hours_per_week=None if hours is None else float(hours),
        notes=str(row["notes"]),
        created_by=str(row["created_by"]),
        created_role=str(row["created_role"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def create_experience(
    conn: sqlite3.Connection,
    student_id: str,
    title: str,
    *,
    catalog_id: str | None = None,
    organization: str = "",
    kind: str | None = None,
    status: str | None = None,
    starts_on: str | None = None,
    ends_on: str | None = None,
    term: str = "",
    hours_per_week: float | None = None,
    notes: str = "",
    created_by: str = "",
    created_role: str = "",
) -> Experience:
    """Add a row; a catalog program starts ``applied``, anything else ``active``.

    Without a ``kind``, a catalog row is a program and anything else "other".
    """
    default = "applied" if catalog_id is not None else "active"
    allowed = PROGRAM_TRANSITIONS if catalog_id is not None else HAND_ENTERED_TRANSITIONS
    resolved = normalize_status(status, default=default)
    if resolved not in allowed:
        resolved = default
    now = utc_now()
    values = {
        "student_id": student_id,
        "catalog_id": catalog_id,
        "title": title,
        "organization": organization,
        "kind": normalize_kind(kind or ("program" if catalog_id is not None else DEFAULT_KIND)),
        "status": resolved,
        "starts_on": starts_on,
        "ends_on": ends_on,
        "term": term,
        "hours_per_week": hours_per_week,
        "notes": notes,
        "created_by": created_by,
        "created_role": created_role,
        "created_at": now,
        "updated_at": now,
    }
    experience_id = _insert(conn, "experiences", values)
    found = get_experience(conn, experience_id)
    assert found is not None
    return found


def get_experience(conn: sqlite3.Connection, experience_id: int) -> Experience | None:
    row = _fetch_one(conn, "SELECT * FROM experiences WHERE id = ?", (experience_id,))
    return None if row is None else _to_experience(row)


def list_experiences(conn: sqlite3.Connection, student_id: str) -> list[Experience]:
    """Every row for a student, oldest first."""
    rows = conn.execute(
        "SELECT * FROM experiences WHERE student_id = ? ORDER BY created_at, id",
        (student_id,),
    ).fetchall()
    return [_to_experience(row) for row in rows]


def latest_by_catalog(experiences: Iterable[Experience]) -> dict[str, Experience]:
    """Each catalog program's most recent cycle, by catalog ID."""
    latest: dict[str, Experience] = {}
    for experience in experiences:
        if experience.catalog_id is None:
            continue
        current = latest.get(experience.catalog_id)
        if current is None or (experience.created_at, experience.id) > (
            current.created_at,
            current.id,
        ):
            latest[experience.catalog_id] = experience
    return latest


def latest_for_catalog(
    conn: sqlite3.Connection, student_id: str, catalog_id: str
) -> Experience | None:
    row = _fetch_one(
        conn,
        "SELECT * FROM experiences WHERE student_id = ? AND catalog_id = ? "
        "ORDER BY created_at DESC, id DESC LIMIT 1",
        (student_id, catalog_id),
    )
    return None if row is None else _to_experience(row)


def update_experience(
    conn: sqlite3.Connection,
    experience_id: int,
    *,
    title: str = UNSET,
    organization: str = UNSET,
    kind: str = UNSET,
    starts_on: str | None = UNSET,
    ends_on: str | None = UNSET,
    term: str = UNSET,
    hours_per_week: float | None = UNSET,
    notes: str = UNSET,
) -> bool:
    """Edit the details; the status only changes through :func:`set_status`."""
    changes = _set(
        title=title,
        organization=organization,
        kind=kind,
        starts_on=starts_on,
        ends_on=ends_on,
        term=term,
        hours_per_week=hours_per_week,
        notes=notes,
    )
    if "kind" in changes:
        changes["kind"] = normalize_kind(changes["kind"])
    return _update(conn, "experiences", experience_id, changes)


def delete_experience(conn: sqlite3.Connection, experience_id: int) -> bool:
    return _delete(conn, "experiences", experience_id)


def set_status(conn: sqlite3.Connection, experience_id: int, target: str) -> Experience:
    """Move a row along its graph, refusing a move the graph does not allow."""
    experience = get_experience(conn, experience_id)
    if experience is None:
        raise LookupError(f"No experience {experience_id}")
    status = normalize_status(target, default="")
    if not can_transition(experience, status):
        raise TransitionError(
            f"Cannot move {experience.title!r} from {experience.status} to {target}"
        )
    if status != experience.status:
        _update(conn, "experiences", experience_id, {"status": status})
    found = get_experience(conn, experience_id)
    assert found is not None
    return found


def school_year_label(today: date) -> str:
    """``2026–27``: a new application's term until she names one."""
    end = school_year_end(today)
    return f"{end - 1}–{end % 100:02d}"


def _created_on(experience: Experience) -> date | None:
    try:
        return date.fromisoformat(experience.created_at[:10])
    except ValueError:
        return None


def record_applied(
    conn: sqlite3.Connection,
    student_id: str,
    catalog_id: str,
    title: str,
    *,
    today: date,
    organization: str = "",
    created_by: str = "",
    created_role: str = "",
) -> Experience:
    """The row for "she applied to this program", written once per cycle.

    Ticking the apply to-do twice, or un-ticking and ticking it again, returns
    the row already there.  Only a program she was not accepted to, or turned
    down, in an earlier school year starts a new row.
    """
    latest = latest_for_catalog(conn, student_id, catalog_id)
    term = school_year_label(today)
    if latest is not None:
        closed = latest.status in ("not_accepted", "declined")
        created = _created_on(latest)
        earlier_year = created is not None and school_year_end(created) < school_year_end(today)
        if not (closed and earlier_year) or latest.term == term:
            return latest
    return create_experience(
        conn,
        student_id,
        title,
        catalog_id=catalog_id,
        organization=organization,
        kind="program",
        status="applied",
        term=term,
        created_by=created_by,
        created_role=created_role,
    )


def undo_applied(conn: sqlite3.Connection, student_id: str, catalog_id: str) -> bool:
    """Un-ticking the apply to-do drops the row, but only while it is still ``applied``."""
    latest = latest_for_catalog(conn, student_id, catalog_id)
    if latest is None or latest.status != "applied":
        return False
    return delete_experience(conn, latest.id)


def apply_again_due(experience: Experience, today: date) -> bool:
    """A program she was not accepted to, from an earlier school year."""
    created = _created_on(experience)
    return (
        experience.status == "not_accepted"
        and created is not None
        and school_year_end(created) < school_year_end(today)
    )


def _sort_day(experience: Experience) -> str:
    return experience.starts_on or experience.created_at[:10]


def honors_list(experiences: Iterable[Experience]) -> list[Experience]:
    """What she got into or does (accepted, completed, active), newest first."""
    shown = [e for e in experiences if e.status in HONORS_STATUSES]
    shown.sort(key=lambda e: (_sort_day(e), e.id), reverse=True)
    return shown


def recently_accepted(
    experiences: Iterable[Experience], today: date, days: int
) -> list[Experience]:
    """Programs marked accepted within the last ``days`` days, for a parent's week."""
    cutoff = (today - timedelta(days=days)).isoformat()
    return [
        e
        for e in experiences
        if e.from_catalog and e.status == "accepted" and e.updated_at[:10] >= cutoff
    ]


def _day_text(value: str | None) -> str:
    try:
        day = date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return ""
    return f"{day:%b} {day.day}, {day.year}"


def dates_text(experience: Experience) -> str:
    """``Jun 15, 2027 – Jul 10, 2027``, either end alone, or empty."""
    start, end = _day_text(experience.starts_on), _day_text(experience.ends_on)
    if start and end:
        return f"{start} – {end}"
    if start:
        return f"From {start}"
    return f"Until {end}" if end else ""


def hours_text(hours: float | None) -> str:
    if not hours:
        return ""
    return f"{hours:g} hours/week"


def summary_line(experience: Experience) -> str:
    """``Example University · Program · Accepted · Spring 2027``."""
    parts = [
        experience.organization,
        KIND_LABELS[experience.kind],
        STATUS_LABELS[experience.status],
        experience.term,
    ]
    return " · ".join(part for part in parts if part)


def honors_text(experiences: Iterable[Experience], *, name: str = "", today: date) -> str:
    """The plain-text brag sheet for counselors and applications."""
    heading = f"Activities & Honors: {name}" if name else "Activities & Honors"
    lines = [heading, f"Updated {_day_text(today.isoformat())}", ""]
    rows = honors_list(experiences)
    if not rows:
        lines.append("Nothing listed yet.")
    for experience in rows:
        lines.append(experience.title)
        lines.append(summary_line(experience))
        detail = " · ".join(
            part
            for part in (dates_text(experience), hours_text(experience.hours_per_week))
            if part
        )
        if detail:
            lines.append(detail)
        if experience.notes.strip():
            lines.append(experience.notes.strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
