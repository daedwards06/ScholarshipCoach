"""Timeline bucketing for eligible scholarships.

Most awards repeat on the same calendar, so a listing that is closed today is
usually a target for a later cycle rather than a dead link.  This module places
each Stage 1 survivor in one of ``now``, ``next_cycle``, ``senior_year``,
``expired``, or ``not_applicable`` and, for a recurring award whose listed
deadline has passed, projects the date of its next cycle.
"""
from __future__ import annotations

import calendar
from datetime import date
from typing import Any

import pandas as pd

from src.profile.grade_levels import grade_level_rank
from src.rank.stage1_eligibility import StudentProfile
from src.text_utils import normalize_list as _normalize_list
from src.text_utils import normalize_text as _normalize_text

TIMELINE_BUCKETS: tuple[str, ...] = (
    "now",
    "next_cycle",
    "senior_year",
    "expired",
    "not_applicable",
)

TIMELINE_BUCKET_LABELS: dict[str, str] = {
    "now": "Apply now",
    "next_cycle": "Next cycle",
    "senior_year": "A later school year",
    "expired": "Closed for good",
    "not_applicable": "Not applicable",
}


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in {"true", "yes", "1"}:
            return True
        if text in {"false", "no", "0"}:
            return False
        return None
    return bool(value)


def _mapping_field(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _row_deadline(row: pd.Series) -> date | None:
    deadline = row.get("deadline")
    if deadline is None or pd.isna(deadline):
        return None
    return pd.Timestamp(deadline).date()


def _row_recurring(row: pd.Series) -> bool | None:
    recurring = _coerce_bool(_mapping_field(row.get("cycle")).get("recurring"))
    if recurring is None:
        recurring = _coerce_bool(row.get("is_recurring"))
    return recurring


def _row_deadline_month(row: pd.Series) -> int | None:
    month = _mapping_field(row.get("cycle")).get("deadline_month")
    if month is None or pd.isna(month):
        return None
    try:
        value = int(month)
    except (TypeError, ValueError):
        return None
    return value if 1 <= value <= 12 else None


def _safe_date(year: int, month: int, day: int) -> date:
    """Build a date, clamping the day to the month (Feb 29 in a common year)."""
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def project_next_deadline(
    deadline: date | None, deadline_month: int | None, today: date
) -> date | None:
    """Return the next occurrence of a recurring award's deadline on or after ``today``.

    The cycle's ``deadline_month`` is the authority on the month; the day comes
    from the listed deadline when the two agree, and otherwise the first of the
    month, since a cycle month alone does not say which day.
    """
    if deadline is None and deadline_month is None:
        return None

    month = deadline_month if deadline_month is not None else deadline.month
    day = deadline.day if deadline is not None and deadline.month == month else 1

    candidate = _safe_date(today.year, month, day)
    if candidate < today:
        candidate = _safe_date(today.year + 1, month, day)
    return candidate


def _school_year_end(value: date) -> int:
    """Return the calendar year the school year containing ``value`` ends in."""
    return value.year + 1 if value.month >= 7 else value.year


def _grade_rank_at(student_rank: int, today: date, when: date) -> int:
    """Advance the student's grade rank by the school years between the two dates."""
    return student_rank + (_school_year_end(when) - _school_year_end(today))


def _row_bucket(
    row: pd.Series, profile: StudentProfile, today: date
) -> tuple[str, date | None]:
    deadline = _row_deadline(row)
    recurring = _row_recurring(row)
    status = _normalize_text(row.get("status"))
    deadline_is_past = deadline is not None and deadline < today
    open_deadline = deadline is not None and deadline >= today
    closed = status == "closed" and not open_deadline

    projected: date | None = None
    if recurring and (deadline_is_past or deadline is None or closed):
        projected = project_next_deadline(deadline, _row_deadline_month(row), today)

    if deadline_is_past or closed:
        if not recurring:
            return "expired", None
        bucket = "next_cycle"
    else:
        bucket = "now"

    grade_levels_allowed = _normalize_list(row.get("grade_levels"))
    student_rank = grade_level_rank(profile.grade_level)
    allowed_ranks = [
        rank
        for rank in (grade_level_rank(grade) for grade in grade_levels_allowed)
        if rank is not None
    ]
    if student_rank is None or not allowed_ranks:
        return bucket, projected

    acts_on = projected or deadline or today
    rank_then = _grade_rank_at(student_rank, today, acts_on)
    if rank_then > max(allowed_ranks):
        # Stage 1 already dropped awards the student has aged out of today, so
        # this is the student ageing out before the next cycle comes around.
        return "not_applicable", projected
    if rank_then < min(allowed_ranks):
        return "senior_year", projected
    return bucket, projected


def classify_timeline(
    df: pd.DataFrame, profile: StudentProfile, today: date | None = None
) -> pd.DataFrame:
    """Add ``timeline_bucket`` and ``projected_deadline`` columns to ``df``.

    Args:
        df: Eligible scholarships from Stage 1.
        profile: Student profile supplying ``grade_level`` and ``today``.
        today: Reference date; defaults to the profile's ``today``, then today.

    Returns:
        Copy of ``df`` with ``timeline_bucket`` (one of
        :data:`TIMELINE_BUCKETS`) and ``projected_deadline`` (the next cycle
        date for a recurring award whose listed deadline has passed, otherwise
        ``NaT``).
    """
    effective_today = today or profile.today or date.today()

    classified_df = df.copy()
    evaluations = [
        _row_bucket(row=row, profile=profile, today=effective_today)
        for _, row in classified_df.iterrows()
    ]
    classified_df["timeline_bucket"] = pd.Series(
        [bucket for bucket, _ in evaluations], index=classified_df.index, dtype=object
    )
    classified_df["projected_deadline"] = pd.to_datetime(
        pd.Series(
            [projected for _, projected in evaluations],
            index=classified_df.index,
            dtype=object,
        ),
        errors="coerce",
    )
    return classified_df
