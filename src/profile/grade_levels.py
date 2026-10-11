"""Grade-level vocabulary shared by the UI, the profile store, and Stage 1.

The sidebar shows human labels ("High School Senior"); the catalog stores
``grade_levels`` as ``9|10|11|12|college_1..4`` and Stage 1 matches on the
education-level vocabulary ("high school", "undergraduate").  This module is
the single place those three spellings meet.
"""
from __future__ import annotations

from datetime import date

GRADE_LABEL_TO_LEVELS: dict[str, tuple[str | None, str | None]] = {
    "": (None, None),
    "High School Freshman": ("high school", "9"),
    "High School Sophomore": ("high school", "10"),
    "High School Junior": ("high school", "11"),
    "High School Senior": ("high school", "12"),
    "College Freshman": ("undergraduate", "college_1"),
    "College Sophomore": ("undergraduate", "college_2"),
    "College Junior": ("undergraduate", "college_3"),
    "College Senior": ("undergraduate", "college_4"),
}

GRADE_LABELS: tuple[str, ...] = tuple(GRADE_LABEL_TO_LEVELS)

# Labels written by the pre-Task-1.2 sidebar, whose bare class years meant
# college years and were stored directly as ``education_level``.
_LEGACY_LABELS: dict[str, str] = {
    "freshman": "College Freshman",
    "sophomore": "College Sophomore",
    "junior": "College Junior",
    "senior": "College Senior",
    "high school senior": "High School Senior",
}

_LEVELS_TO_GRADE_LABEL: dict[str | None, str] = {
    grade_level: label
    for label, (_, grade_level) in GRADE_LABEL_TO_LEVELS.items()
    if grade_level is not None
}

# The school sequence in order. Stage 1 uses it to tell an award the student has
# already passed (a grade-9 award for a college sophomore) from one still ahead
# of them (a seniors-only award), which the timeline buckets rather than filters.
GRADE_SEQUENCE: tuple[str, ...] = (
    "9",
    "10",
    "11",
    "12",
    "college_1",
    "college_2",
    "college_3",
    "college_4",
)

# Years of school remaining after the school year in which the grade is spent.
_YEARS_TO_GRADUATION: dict[str, int] = {
    "9": 3,
    "10": 2,
    "11": 1,
    "12": 0,
    "college_1": 3,
    "college_2": 2,
    "college_3": 1,
    "college_4": 0,
}


def resolve_grade_label(label: str | None) -> str:
    """Return the canonical grade label for ``label`` ("" when unrecognized)."""
    raw = (label or "").strip()
    if raw in GRADE_LABEL_TO_LEVELS:
        return raw
    return _LEGACY_LABELS.get(raw.casefold(), "")


def grade_label_to_levels(label: str | None) -> tuple[str | None, str | None]:
    """Map a UI grade label to ``(education_level, grade_level)``.

    Unrecognized labels map to ``(None, None)`` rather than raising: a profile
    saved by an older build must still load.
    """
    return GRADE_LABEL_TO_LEVELS[resolve_grade_label(label)]


def levels_to_grade_label(
    education_level: str | None, grade_level: str | None
) -> str:
    """Return the UI grade label for a stored profile ("" when unknown).

    Falls back to the ``education_level`` string when no ``grade_level`` is
    stored, so profiles written before this vocabulary existed still select
    the right option.
    """
    label = _LEVELS_TO_GRADE_LABEL.get((grade_level or "").strip() or None)
    if label is not None:
        return label
    return resolve_grade_label(education_level)


def school_year_end(value: date) -> int:
    """Return the calendar year the school year containing ``value`` ends in.

    A school year runs July through June, so anything from July on belongs to
    the year that ends the following spring.
    """
    return value.year + 1 if value.month >= 7 else value.year


def grade_level_rank(grade_level: str | None) -> int | None:
    """Return the position of ``grade_level`` in :data:`GRADE_SEQUENCE`.

    Unrecognized or missing values return ``None`` so callers can treat the
    grade as unknown instead of guessing a position.
    """
    key = (grade_level or "").strip().casefold()
    try:
        return GRADE_SEQUENCE.index(key)
    except ValueError:
        return None


def grade_level_in_school_year(
    grade_level: str | None, year_end: int, today: date | None = None
) -> str | None:
    """Return the grade the student is in during the school year ending ``year_end``.

    Returns ``None`` when the grade is unknown, when the year is before the
    student started, or when they have already graduated by then.
    """
    rank = grade_level_rank(grade_level)
    if rank is None:
        return None
    reference = today or date.today()
    rank_then = rank + (year_end - school_year_end(reference))
    if not 0 <= rank_then < len(GRADE_SEQUENCE):
        return None
    return GRADE_SEQUENCE[rank_then]


def infer_graduation_year(grade_level: str | None, today: date | None = None) -> int | None:
    """Estimate the graduation year for ``grade_level`` as of ``today``.

    Assumes a school year ending in the spring, so a grade entered in the fall
    graduates in the following calendar year.
    """
    remaining = _YEARS_TO_GRADUATION.get((grade_level or "").strip())
    if remaining is None:
        return None
    reference = today or date.today()
    return school_year_end(reference) + remaining


_SENIOR_RANK = GRADE_SEQUENCE.index("12")


def _is_college(rank: int) -> bool:
    return rank > _SENIOR_RANK


def education_level_for_grade(grade_level: str | None) -> str | None:
    """Stage 1's education level for a grade ("high school" or "undergraduate")."""
    rank = grade_level_rank(grade_level)
    if rank is None:
        return None
    return "undergraduate" if _is_college(rank) else "high school"


def grade_for_graduation_year(
    graduation_year: int | None, grade_level: str | None, today: date | None = None
) -> str | None:
    """The grade ``graduation_year`` implies as of ``today``.

    The graduation year is read on the same track as ``grade_level``: a high
    school year for grades 9-12, a college year for college grades.  Past the
    end of high school the grade carries on into college years.  ``None`` when
    either input is unknown or the student has not started the track yet.
    """
    rank = grade_level_rank(grade_level)
    if graduation_year is None or rank is None:
        return None
    last = len(GRADE_SEQUENCE) - 1 if _is_college(rank) else _SENIOR_RANK
    first = _SENIOR_RANK + 1 if _is_college(rank) else 0
    implied = last - (graduation_year - school_year_end(today or date.today()))
    if implied < first:
        return None
    return GRADE_SEQUENCE[min(implied, len(GRADE_SEQUENCE) - 1)]


def reconcile_grade(
    grade_level: str | None,
    graduation_year: int | None,
    previous_grade: str | None,
    previous_year: int | None,
    today: date | None = None,
) -> tuple[str, int | None]:
    """Return the ``(grade_level, graduation_year)`` pair to keep, consistent with each other.

    The graduation year is the anchor, so the grade rolls forward every July
    with nobody remembering to change it.  A grade someone just changed by
    hand wins and re-anchors the year; a changed year moves the grade.
    """
    grade = (grade_level or "").strip()
    if not grade:
        return "", graduation_year
    if grade != (previous_grade or "").strip() or graduation_year is None:
        return grade, infer_graduation_year(grade, today)
    current = grade_for_graduation_year(graduation_year, grade, today)
    if current is None:
        return grade, infer_graduation_year(grade, today)
    current_rank = grade_level_rank(current)
    grade_rank = grade_level_rank(grade)
    if current_rank is not None and grade_rank is not None and (
        _is_college(current_rank) != _is_college(grade_rank)
    ):
        # Rolled past high school: the old year was a high-school graduation.
        return current, infer_graduation_year(current, today)
    return current, graduation_year
