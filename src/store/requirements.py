"""My Path: a first target school's requirements, placed against her school years.

A requirement's ``due_by`` is either a grade ("11") or an ISO date, because
"take precalculus by junior year" has no date until a course schedule exists.
This module turns either into one of the My Path groups relative to the grade
on her profile, reads a GPA target against her GPA, and keeps dated
requirements on the shared To-dos list.

Everything here is generic: requirement categories and grade arithmetic.  The
school, its rules and its thresholds only ever arrive from ``coach.db``.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import replace
from datetime import date

from src.profile.grade_levels import grade_level_rank, school_year_end
from src.store import repo, todos
from src.store.repo import CollegeRequirement

# My Path groups, in display order.  A date or grade in the current school year
# (or already past) is "this semester": it is what she should be working on now.
PATH_GROUPS: tuple[str, ...] = ("this_semester", "next_year", "senior_year", "done")

PATH_GROUP_LABELS: dict[str, str] = {
    "this_semester": "This semester",
    "next_year": "Next year",
    "senior_year": "Senior year",
    "done": "Done",
}

DONE_STATUSES: frozenset[str] = frozenset({"met", "not_needed"})

# Categories whose status she sets herself; the rest are the parents' to keep.
STUDENT_STATUS_CATEGORIES: frozenset[str] = frozenset({"course", "test"})

STUDENT_STATUSES: tuple[str, ...] = ("not_started", "in_progress", "met")

STATUS_LABELS: dict[str, str] = {
    "not_started": "Not started",
    "in_progress": "In progress",
    "met": "Met",
    "not_needed": "Not needed",
}

CATEGORY_LABELS: dict[str, str] = {
    "course": "Course",
    "gpa": "GPA",
    "test": "Test",
    "application": "Application",
    "scholarship": "Scholarship",
    "program": "Program",
    "other": "Other",
}

HIGH_SCHOOL_GRADES: tuple[str, ...] = ("9", "10", "11", "12")

_GRADE_PATTERN = re.compile(r"(?:grade\s*)?(9|10|11|12)(?:th)?(?:\s*grade)?")
_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def parse_due_by(value: object) -> tuple[str | None, date | None]:
    """``(grade, day)`` from a stored ``due_by``; at most one is set."""
    text = str(value or "").strip().casefold()
    if not text:
        return None, None
    try:
        return None, date.fromisoformat(text[:10])
    except ValueError:
        pass
    match = _GRADE_PATTERN.fullmatch(text)
    return (match.group(1), None) if match else (None, None)


def normalize_due_by(value: object) -> str | None:
    """The stored spelling of a typed ``due_by`` ("11" or an ISO date), ``""`` for blank.

    ``None`` means the text is neither, so the form can refuse it rather than
    file the row under This semester by accident.
    """
    if not str(value or "").strip():
        return ""
    grade, day = parse_due_by(value)
    if day is not None:
        return day.isoformat()
    return grade


def grade_ordinal(grade: str) -> str:
    return f"{grade}th"


def due_by_text(due_by: str, today: date) -> str:
    """"By 11th grade" or "By Mar 1, 2027"; empty when there is no due-by."""
    grade, day = parse_due_by(due_by)
    if grade is not None:
        return f"By {grade_ordinal(grade)} grade"
    if day is not None:
        when = f"{day:%b} {day.day}"
        if school_year_end(day) != school_year_end(today):
            when = f"{when}, {day.year}"
        return f"By {when}"
    return ""


def due_school_year(due_by: str, grade_level: str | None, today: date) -> int | None:
    """The school year (by its ending year) a requirement falls due in.

    A grade converts through her current grade, so it is ``None`` when the
    profile has no grade.
    """
    grade, day = parse_due_by(due_by)
    if day is not None:
        return school_year_end(day)
    if grade is None:
        return None
    current = grade_level_rank(grade_level)
    target = grade_level_rank(grade)
    if current is None or target is None:
        return None
    return school_year_end(today) + (target - current)


def requirement_group(
    requirement: CollegeRequirement, grade_level: str | None, today: date
) -> str:
    """Which My Path group a requirement belongs in.

    Undated rows (a GPA to hold, a course with no grade on it) belong to now.
    With no grade on the profile, a senior-year grade still reads as senior
    year and any other grade as now, rather than guessing how far off it is.
    """
    if requirement.status in DONE_STATUSES:
        return "done"
    year = due_school_year(requirement.due_by, grade_level, today)
    if year is None:
        grade, _ = parse_due_by(requirement.due_by)
        return "senior_year" if grade == "12" else "this_semester"
    ahead = year - school_year_end(today)
    if ahead <= 0:
        return "this_semester"
    if ahead == 1:
        return "next_year"
    return "senior_year"


def target_number(target: str) -> float | None:
    """The first number in a target ("GPA 3.0+" -> 3.0)."""
    match = _NUMBER_PATTERN.search(str(target or ""))
    return float(match.group(0)) if match else None


def gpa_text(value: float) -> str:
    """3.25 -> "3.25", 3.5 -> "3.5", 3.0 -> "3.0"."""
    text = f"{value:.2f}"
    return text[:-1] if text.endswith("0") else text


def gpa_gap(requirement: CollegeRequirement, gpa: float | None) -> float | None:
    """How far her GPA is below a GPA row's target (0 when at or above it)."""
    if requirement.category != "gpa" or not gpa:
        return None
    want = target_number(requirement.target)
    if want is None:
        return None
    return max(0.0, round(want - gpa, 2))


def gpa_gap_text(requirement: CollegeRequirement, gpa: float | None) -> str | None:
    """"You're at 3.25 · 0.25 to go" for a GPA row; ``None`` when it cannot say."""
    gap = gpa_gap(requirement, gpa)
    if gap is None or gpa is None:
        return None
    if gap == 0:
        return f"You're at {gpa_text(gpa)} · you meet this"
    return f"You're at {gpa_text(gpa)} · {gpa_text(gap)} to go"


def effective_status(requirement: CollegeRequirement, gpa: float | None) -> str:
    """The status My Path shows: a GPA row follows her profile GPA when it can."""
    if requirement.status == "not_needed":
        return requirement.status
    gap = gpa_gap(requirement, gpa)
    if gap is None:
        return requirement.status
    return "met" if gap == 0 else "in_progress"


def with_effective_status(
    requirements: list[CollegeRequirement], gpa: float | None
) -> list[CollegeRequirement]:
    return [replace(item, status=effective_status(item, gpa)) for item in requirements]


def group_requirements(
    requirements: list[CollegeRequirement], grade_level: str | None, today: date
) -> dict[str, list[CollegeRequirement]]:
    groups: dict[str, list[CollegeRequirement]] = {group: [] for group in PATH_GROUPS}
    for requirement in requirements:
        groups[requirement_group(requirement, grade_level, today)].append(requirement)
    return groups


def progress_text(met: int, counted: int) -> str:
    return f"{met} of {counted} on track"


# -- requirement rows on the To-dos list ---------------------------------------


def requirement_source_ref(requirement_id: int) -> str:
    return str(requirement_id)


def sync_requirement_todos(
    conn: sqlite3.Connection, student_id: str, *, today: date
) -> None:
    """Keep one to-do per dated requirement of her first target school.

    The requirement is the source of truth: its label and date are copied
    onto the row, and the row is done exactly when the requirement is met or
    not needed.  Open rows whose requirement lost its date, was removed, or
    belongs to a school that is no longer the first target are dropped; done
    ones stay as history.
    """
    college = repo.first_target_college(conn, student_id)
    dated: dict[str, tuple[CollegeRequirement, date]] = {}
    if college is not None:
        for requirement in repo.list_requirements(conn, college.id):
            _, day = parse_due_by(requirement.due_by)
            if day is not None:
                dated[requirement_source_ref(requirement.id)] = (requirement, day)

    for todo in todos.list_todos(conn, student_id):
        if todo.source_kind == "requirement" and todo.source_ref not in dated and not todo.done:
            todos.delete_todo(conn, todo.id)

    for ref, (requirement, day) in dated.items():
        done = requirement.status in DONE_STATUSES
        existing = todos.find_todo(conn, student_id, "requirement", ref)
        if existing is None:
            if not done:
                todos.create_todo(
                    conn,
                    student_id,
                    requirement.label,
                    notes=requirement.notes,
                    due_on=day.isoformat(),
                    assignee="student",
                    source_kind="requirement",
                    source_ref=ref,
                )
            continue
        if existing.title != requirement.label or existing.due_on != day.isoformat():
            todos.update_todo(conn, existing.id, title=requirement.label, due_on=day.isoformat())
        if existing.done != done:
            todos.set_todo_done(conn, existing.id, done, today=today)
