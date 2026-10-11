"""Programs and awards a student can act on, grouped by when she can apply.

The Opportunities page is not a ranking: Stage 3 scores on money and a summer
program has none.  Every record that passes Stage 1 *and names her grade* (or
a later one) is listed by its application date in one of three groups:

* ``open_now``: applications are open for this school year.
* ``opens_later``: applications open later this year, in a later grade, or on
  a date nobody has posted yet.
* ``passed``: this school year's deadline has gone by; the next round is shown
  when the award recurs.

A record that names no grade is never shown to her (R2): most of those are for
seniors or college students, and a student surface must not guess.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from src.profile.grade_levels import GRADE_SEQUENCE, grade_level_rank, school_year_end
from src.rank.kinds import program_mask
from src.rank.stage1_eligibility import (
    SENIOR_GRADE_RANK,
    StudentProfile,
    apply_eligibility_filter,
)
from src.rank.timeline import classify_timeline
from src.text_utils import normalize_list, normalize_text

OPEN_NOW = "open_now"
OPENS_LATER = "opens_later"
PASSED = "passed"
OPPORTUNITY_GROUPS: tuple[str, ...] = (OPEN_NOW, OPENS_LATER, PASSED)

# A sponsor saying it has not opened yet outranks a date that looks open.
_NOT_OPEN_STATUSES = frozenset({"upcoming", "closed"})


def explicit_grade_mask(df: pd.DataFrame) -> pd.Series:
    """Rows that name at least one grade."""
    if "grade_levels" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df["grade_levels"].map(lambda value: bool(normalize_list(value))).astype(bool)


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError):
        return None


def _opens_month(row: pd.Series) -> int | None:
    cycle = row.get("cycle")
    month = cycle.get("opens_month") if isinstance(cycle, dict) else None
    try:
        value = int(month) if month is not None and not pd.isna(month) else None
    except (TypeError, ValueError):
        return None
    return value if value is not None and 1 <= value <= 12 else None


def opening_date(opens_month: int | None, apply_by: date | None) -> date | None:
    """The first of ``opens_month`` before ``apply_by``: when this round opens."""
    if opens_month is None or apply_by is None:
        return None
    year = apply_by.year if opens_month <= apply_by.month else apply_by.year - 1
    return date(year, opens_month, 1)


def opens_in_grade(grade_levels: Any, student_rank: int) -> str | None:
    """The first grade after the student's that the record is open to."""
    later = sorted(
        rank
        for rank in (grade_level_rank(grade) for grade in normalize_list(grade_levels))
        if rank is not None and rank > student_rank
    )
    return GRADE_SEQUENCE[later[0]] if later else None


def _classify_row(row: pd.Series, student_rank: int, today: date) -> dict[str, Any] | None:
    bucket = str(row.get("timeline_bucket") or "")
    if bucket == "check_grade":
        return None

    deadline = _as_date(row.get("deadline"))
    projected = _as_date(row.get("projected_deadline"))
    passed_this_year = (
        deadline is not None
        and deadline < today
        and school_year_end(deadline) == school_year_end(today)
    )
    if bucket == "not_applicable":
        # She is too old for the next round, but missing this one is still news.
        if not passed_this_year:
            return None
        projected = None

    if deadline is not None and deadline >= today:
        apply_by, is_projected = deadline, False
    elif projected is not None:
        apply_by, is_projected = projected, True
    else:
        apply_by, is_projected = None, False

    opens_on = opening_date(_opens_month(row), apply_by)
    opens_grade: str | None = None
    status = normalize_text(row.get("status"))

    if bucket in {"expired", "not_applicable"} or passed_this_year:
        group = PASSED
    elif bucket == "senior_year":
        opens_grade = opens_in_grade(row.get("grade_levels"), student_rank)
        rank = grade_level_rank(opens_grade)
        if rank is None or rank > SENIOR_GRADE_RANK:
            # A college-years award is not part of her high-school path.
            return None
        group = OPENS_LATER
    elif bucket == "needs_date":
        group = OPENS_LATER
    elif status in _NOT_OPEN_STATUSES or (opens_on is not None and opens_on > today):
        group = OPENS_LATER
    elif bucket == "next_cycle" and (opens_on is None or opens_on > today):
        # Last round's date has gone by and nobody has said the new one is open.
        group = OPENS_LATER
    else:
        group = OPEN_NOW

    return {
        "opportunity_group": group,
        "apply_by": apply_by,
        "apply_by_projected": is_projected,
        "opens_on": opens_on if opens_on is not None and opens_on > today else None,
        "opens_grade": opens_grade,
    }


def classify_opportunities(
    snapshot_df: pd.DataFrame, profile: StudentProfile, today: date | None = None
) -> pd.DataFrame:
    """Stage 1, then the explicit-grade filter, then the three groups.

    Returns the surviving rows with ``opportunity_group``, ``apply_by`` (the
    date to apply by this round, real or projected), ``apply_by_projected``,
    ``opens_on`` (a future opening date), ``opens_grade`` (the later grade it
    opens in), ``is_program`` and Stage 1's ``unverified_axes``, sorted by
    group and then by date.  Empty when the profile has no grade.
    """
    effective_today = today or profile.today or date.today()
    student_rank = grade_level_rank(profile.grade_level)
    columns = [
        *snapshot_df.columns,
        "reasons",
        "unverified_axes",
        "timeline_bucket",
        "projected_deadline",
        "opportunity_group",
        "apply_by",
        "apply_by_projected",
        "opens_on",
        "opens_grade",
        "is_program",
    ]
    if student_rank is None or snapshot_df.empty:
        return pd.DataFrame(columns=columns)

    eligible_df, _ = apply_eligibility_filter(snapshot_df, profile)
    eligible_df = eligible_df[explicit_grade_mask(eligible_df)]
    if eligible_df.empty:
        return pd.DataFrame(columns=columns)
    classified = classify_timeline(eligible_df, profile, effective_today)

    rows = []
    for index, row in classified.iterrows():
        extra = _classify_row(row, student_rank, effective_today)
        if extra is not None:
            rows.append((index, extra))
    if not rows:
        return pd.DataFrame(columns=columns)

    kept = classified.loc[[index for index, _ in rows]].copy()
    for name in ("opportunity_group", "apply_by", "apply_by_projected", "opens_on", "opens_grade"):
        kept[name] = pd.Series([extra[name] for _, extra in rows], index=kept.index, dtype=object)
    kept["is_program"] = program_mask(kept)

    order = {group: position for position, group in enumerate(OPPORTUNITY_GROUPS)}
    later_grade = kept["opens_grade"].map(lambda grade: grade_level_rank(grade) or 0)
    sort_date = [
        opens_on or apply_by or date.max
        for opens_on, apply_by in zip(kept["opens_on"], kept["apply_by"], strict=True)
    ]
    kept = kept.assign(
        _group=kept["opportunity_group"].map(order),
        _grade=later_grade,
        _date=sort_date,
        _title=kept["title"].fillna("").astype(str).str.casefold(),
    )
    kept = kept.sort_values(["_group", "_grade", "_date", "_title"], kind="mergesort")
    return kept.drop(columns=["_group", "_grade", "_date", "_title"]).reset_index(drop=True)
