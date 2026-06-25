"""Hard eligibility filtering with reason codes.

Applies deadline, GPA, state, major, education level, and citizenship checks
to a scholarship DataFrame and splits it into eligible and ineligible sets.
Each ineligible row is annotated with the list of reason codes that disqualified it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from src.text_utils import normalize_list as _normalize_list
from src.text_utils import normalize_text as _normalize_text


@dataclass(slots=True)
class StudentProfile:
    """Mutable student profile used for Stage 1 eligibility checks.

    All fields are optional so callers can supply only the attributes they know.
    ``today`` is used as the reference date for deadline comparisons; it
    defaults to ``date.today()`` if not set.
    """

    gpa: float | None = None
    state: str | None = None
    major: str | None = None
    education_level: str | None = None
    citizenship: str | None = None
    today: date | None = None


def _row_reasons(row: pd.Series, profile: StudentProfile, today: date) -> list[str]:
    reasons: list[str] = []

    deadline = row.get("deadline")
    if deadline is not None and not pd.isna(deadline):
        deadline_date = pd.Timestamp(deadline).date()
        if deadline_date < today:
            reasons.append("DEADLINE_PASSED")

    min_gpa = row.get("min_gpa")
    if profile.gpa is not None and min_gpa is not None and not pd.isna(min_gpa):
        if profile.gpa < float(min_gpa):
            reasons.append("GPA_BELOW_MIN")

    states_allowed = _normalize_list(row.get("states_allowed"))
    profile_state = _normalize_text(profile.state)
    if states_allowed and profile_state not in states_allowed:
        reasons.append("STATE_NOT_ALLOWED")

    majors_allowed = _normalize_list(row.get("majors_allowed"))
    profile_major = _normalize_text(profile.major)
    if majors_allowed and profile_major not in majors_allowed:
        reasons.append("MAJOR_NOT_ALLOWED")

    scholarship_education_level = _normalize_text(row.get("education_level"))
    profile_education_level = _normalize_text(profile.education_level)
    if scholarship_education_level and scholarship_education_level != profile_education_level:
        reasons.append("EDUCATION_LEVEL_MISMATCH")

    scholarship_citizenship = _normalize_text(row.get("citizenship"))
    profile_citizenship = _normalize_text(profile.citizenship)
    if scholarship_citizenship and scholarship_citizenship != profile_citizenship:
        reasons.append("CITIZENSHIP_MISMATCH")

    return reasons


def apply_eligibility_filter(
    df: pd.DataFrame, profile: StudentProfile
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split scholarships into eligible and ineligible sets based on hard filters.

    Applies deadline, GPA, state, major, education level, and citizenship checks.
    Each ineligible scholarship is tagged with all applicable reason codes.

    Args:
        df: Scholarship DataFrame with normalized columns.
        profile: Student profile with eligibility attributes.

    Returns:
        Tuple of ``(eligible_df, ineligible_df)``.  The ineligible frame includes
        a ``reasons`` column with lists of reason codes such as
        ``"GPA_BELOW_MIN"`` and ``"STATE_NOT_ALLOWED"``.
    """
    effective_today = profile.today or date.today()

    with_reasons_df = df.copy()
    with_reasons_df["reasons"] = with_reasons_df.apply(
        lambda row: _row_reasons(row=row, profile=profile, today=effective_today),
        axis=1,
    )

    is_ineligible = with_reasons_df["reasons"].map(bool)
    ineligible_df = with_reasons_df[is_ineligible].copy()
    eligible_df = with_reasons_df[~is_ineligible].copy()

    return eligible_df, ineligible_df
