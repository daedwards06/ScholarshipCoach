"""Planning milestones: the dates that are true for everyone, plus the family's edits.

An award deadline belongs to a sponsor.  A milestone belongs to the calendar --
the FAFSA opening, the PSAT window, the early-action cluster -- and it is the
same date for every family, which is why the general set ships in
``data/milestones.json`` and is committed.

Two things make a milestone usable.  First, applicability: the FAFSA matters to
a senior and not to a freshman, so each milestone names the grades it speaks to
and an empty list means everyone.  Second, the year: a milestone stores a month
and a day, not a date, and resolves against a school year, so the same row
serves this year and the ones after it.

A family's own dates -- a district scholarship night, a counselor deadline --
and their corrections to the general ones live in the ``settings`` table as one
JSON overlay keyed by milestone id.  The shipped file stays pristine, so a
later correction to it reaches families who did not override that row.
"""
from __future__ import annotations

import calendar
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from src.profile.grade_levels import grade_level_in_school_year, school_year_end
from src.store import repo

ROOT_DIR = Path(__file__).resolve().parents[2]
MILESTONES_PATH = ROOT_DIR / "data" / "milestones.json"

FAMILY_MILESTONES_SETTING = "family_milestones"

DEFAULT_SOURCE = "default"
FAMILY_SOURCE = "family"


@dataclass(frozen=True, slots=True)
class Milestone:
    """A recurring planning date, stored as a month and day rather than a date."""

    id: str
    title: str
    month: int
    day: int
    end_month: int | None = None
    end_day: int | None = None
    grade_levels: tuple[str, ...] = ()
    note: str = ""
    source: str = DEFAULT_SOURCE

    @property
    def is_window(self) -> bool:
        return self.end_month is not None


@dataclass(frozen=True, slots=True)
class MilestoneOccurrence:
    """A milestone landed on a specific school year."""

    milestone: Milestone
    starts_on: date
    ends_on: date | None = None

    @property
    def is_window(self) -> bool:
        return self.ends_on is not None and self.ends_on != self.starts_on


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _month(value: Any) -> int | None:
    try:
        month = int(value)
    except (TypeError, ValueError):
        return None
    return month if 1 <= month <= 12 else None


def _day(value: Any) -> int | None:
    try:
        day = int(value)
    except (TypeError, ValueError):
        return None
    return day if 1 <= day <= 31 else None


def _grade_levels(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(_clean(grade) for grade in value if _clean(grade))


def _safe_date(year: int, month: int, day: int) -> date:
    """Build a date, clamping the day to the month (Feb 29 in a common year)."""
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def milestone_from_mapping(
    payload: Mapping[str, Any], source: str = DEFAULT_SOURCE
) -> Milestone | None:
    """Build a milestone from a JSON object, or ``None`` if it is unusable.

    A malformed row is dropped rather than raised on: one bad hand-edited entry
    in the family's settings must not take the whole timeline down with it.
    """
    identifier = _clean(payload.get("id"))
    title = _clean(payload.get("title"))
    month = _month(payload.get("month"))
    day = _day(payload.get("day"))
    if not identifier or not title or month is None or day is None:
        return None

    end_month = _month(payload.get("end_month"))
    end_day = _day(payload.get("end_day"))
    if end_month is None:
        end_day = None
    elif end_day is None:
        end_day = 1

    return Milestone(
        id=identifier,
        title=title,
        month=month,
        day=day,
        end_month=end_month,
        end_day=end_day,
        grade_levels=_grade_levels(payload.get("grade_levels")),
        note=_clean(payload.get("note")),
        source=source,
    )


def milestone_to_mapping(milestone: Milestone) -> dict[str, Any]:
    """Render a milestone back to the JSON shape the settings overlay stores."""
    payload: dict[str, Any] = {
        "id": milestone.id,
        "title": milestone.title,
        "month": milestone.month,
        "day": milestone.day,
        "grade_levels": list(milestone.grade_levels),
        "note": milestone.note,
    }
    if milestone.end_month is not None:
        payload["end_month"] = milestone.end_month
        payload["end_day"] = milestone.end_day
    return payload


def load_default_milestones(path: Path | None = None) -> list[Milestone]:
    """Read the committed general milestones, skipping any malformed row."""
    source_path = path or MILESTONES_PATH
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    rows = payload.get("milestones") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        return []
    parsed = [
        milestone_from_mapping(row, DEFAULT_SOURCE) for row in rows if isinstance(row, Mapping)
    ]
    return [milestone for milestone in parsed if milestone is not None]


def load_overrides(conn: Any) -> list[dict[str, Any]]:
    """Read the family's milestone overlay out of ``settings``."""
    raw = repo.get_setting(conn, FAMILY_MILESTONES_SETTING)
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(payload, list):
        return []
    return [dict(row) for row in payload if isinstance(row, Mapping) and _clean(row.get("id"))]


def save_overrides(conn: Any, overrides: Iterable[Mapping[str, Any]]) -> None:
    """Write the family's milestone overlay, replacing whatever was there."""
    rows = [dict(row) for row in overrides if _clean(row.get("id"))]
    repo.set_setting(conn, FAMILY_MILESTONES_SETTING, json.dumps(rows, ensure_ascii=False))


def apply_overrides(
    defaults: Iterable[Milestone], overrides: Iterable[Mapping[str, Any]]
) -> list[Milestone]:
    """Overlay the family's edits on the general set, keeping the general order.

    An override matching a shipped id edits that milestone in place; one with
    ``"hidden": true`` removes it; one with a new id is the family's own
    milestone and is appended.
    """
    resolved = list(defaults)
    by_id = {milestone.id: index for index, milestone in enumerate(resolved)}
    dropped: set[str] = set()

    for override in overrides:
        identifier = _clean(override.get("id"))
        if not identifier:
            continue
        if override.get("hidden") is True:
            dropped.add(identifier)
            continue
        dropped.discard(identifier)
        index = by_id.get(identifier)
        if index is None:
            added = milestone_from_mapping(override, FAMILY_SOURCE)
            if added is not None:
                by_id[identifier] = len(resolved)
                resolved.append(added)
            continue
        merged = {**milestone_to_mapping(resolved[index]), **dict(override)}
        edited = milestone_from_mapping(merged, FAMILY_SOURCE)
        if edited is not None:
            resolved[index] = edited

    return [milestone for milestone in resolved if milestone.id not in dropped]


def load_milestones(conn: Any | None = None, path: Path | None = None) -> list[Milestone]:
    """The milestones this family plans against: the general set plus their edits."""
    defaults = load_default_milestones(path)
    if conn is None:
        return defaults
    return apply_overrides(defaults, load_overrides(conn))


def applies_to_grade(milestone: Milestone, grade_level: str | None) -> bool:
    """True when a milestone speaks to ``grade_level``; no grades means everyone."""
    if not milestone.grade_levels:
        return True
    grade = _clean(grade_level).casefold()
    return bool(grade) and grade in {level.casefold() for level in milestone.grade_levels}


def milestones_for_grade(
    milestones: Iterable[Milestone], grade_level: str | None
) -> list[Milestone]:
    return [milestone for milestone in milestones if applies_to_grade(milestone, grade_level)]


def occurrence(milestone: Milestone, year_end: int) -> MilestoneOccurrence:
    """Land ``milestone`` on the school year ending in ``year_end``.

    July onward belongs to the front half of the school year, so an October
    date in the 2026-27 year falls in calendar 2026 and a March date in 2027.
    A window whose end month precedes its start month crosses the new year.
    """
    start_year = year_end - 1 if milestone.month >= 7 else year_end
    starts_on = _safe_date(start_year, milestone.month, milestone.day)

    if milestone.end_month is None or milestone.end_day is None:
        return MilestoneOccurrence(milestone=milestone, starts_on=starts_on)

    end_year = start_year + 1 if milestone.end_month < milestone.month else start_year
    ends_on = _safe_date(end_year, milestone.end_month, milestone.end_day)
    if ends_on < starts_on:
        ends_on = starts_on
    return MilestoneOccurrence(milestone=milestone, starts_on=starts_on, ends_on=ends_on)


def occurrences_for_school_year(
    milestones: Iterable[Milestone],
    grade_level: str | None,
    year_end: int,
    today: date | None = None,
) -> list[MilestoneOccurrence]:
    """The milestones that apply in one school year, dated and in date order.

    Applicability is judged against the grade the student will be *in* that
    year, not the grade they are in today -- that is the point of planning
    ahead.  A year the student has already graduated out of returns nothing.
    """
    reference = today or date.today()
    if _clean(grade_level):
        grade_then = grade_level_in_school_year(grade_level, year_end, today=reference)
        if grade_then is None:
            return []
        applicable = milestones_for_grade(milestones, grade_then)
    else:
        # No grade on the profile is not the same as being in no grade: filtering
        # every grade-scoped milestone away would leave the page blank.
        applicable = list(milestones)

    dated = [occurrence(milestone, year_end) for milestone in applicable]
    dated.sort(key=lambda item: (item.starts_on, item.milestone.title))
    return dated


def family_milestone_id(title: str, taken: Iterable[str] = ()) -> str:
    """A stable, readable id for a milestone the family adds themselves."""
    slug = "".join(
        character if character.isalnum() else "_" for character in _clean(title).casefold()
    ).strip("_")
    base = f"{FAMILY_SOURCE}_{slug or 'milestone'}"
    used = set(taken)
    if base not in used:
        return base
    suffix = 2
    while f"{base}_{suffix}" in used:
        suffix += 1
    return f"{base}_{suffix}"


def school_year_label(year_end: int) -> str:
    """Render a school year the way a family says it: ``2026-27``."""
    return f"{year_end - 1}-{year_end % 100:02d}"


def current_school_year(today: date | None = None) -> int:
    return school_year_end(today or date.today())
