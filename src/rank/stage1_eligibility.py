"""Hard eligibility filtering with reason codes.

Applies record trust, deadline, status, GPA, geography, major, education and
grade level, citizenship, and the curated restriction axes (need, first
generation, gender, heritage, military family, disability, religion, employer,
membership, test scores) to a scholarship DataFrame, splitting it into eligible
and ineligible sets.  Each ineligible row is annotated with the reason codes
that disqualified it; each eligible row carries the axes its profile could not
answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from src.profile.grade_levels import grade_level_rank
from src.rank.taxonomy import education_level_matches, majors_match
from src.text_utils import normalize_list as _normalize_list
from src.text_utils import normalize_text as _normalize_text

# Only a confirmed record, or one from a trusted structured feed, counts as
# eligible: an aggregator scrape is a lead, not an award a family should apply
# to.  A missing or null ``trust`` passes, because pre-catalog snapshots and
# test fixtures carry no such column.
TRUST_UNCONFIRMED_CODE = "TRUST_UNCONFIRMED"
UNCONFIRMED_TRUST_VALUES: frozenset[str] = frozenset({"unverified", "aggregator"})

# Restriction axes a profile may leave unanswered, and the phrasing the app uses
# after "Confirm you meet:".  An axis listed here passes Stage 1 when the profile
# is silent, so the student confirms it instead of the filter guessing.
UNVERIFIED_AXIS_LABELS: dict[str, str] = {
    "county": "county residency",
    "grade_level": "grade level",
    "financial_need": "financial need",
    "first_gen": "first-generation status",
    "gender": "gender requirement",
    "heritage": "heritage requirement",
    "military_family": "military family connection",
    "disability": "disability requirement",
    "religion": "religious affiliation",
    "parent_employers": "employer connection",
    "memberships": "membership requirement",
    "test_scores": "test score minimum",
}


@dataclass(slots=True)
class StudentProfile:
    """Mutable student profile used for Stage 1 eligibility checks.

    All fields are optional so callers can supply only the attributes they know.
    ``today`` is used as the reference date for deadline comparisons; it
    defaults to ``date.today()`` if not set. When ``strict_education_level`` is
    ``True`` the education check requires an exact level match instead of
    allowing adjacency (e.g. high school ↔ undergraduate).

    A restriction axis left at ``None`` means "the student has not answered",
    which is different from a stated "no": the rules must not disqualify on an
    unanswered axis.
    """

    gpa: float | None = None
    state: str | None = None
    major: str | None = None
    education_level: str | None = None
    citizenship: str | None = None
    today: date | None = None
    strict_education_level: bool = False

    student_id: str | None = None
    graduation_year: int | None = None
    grade_level: str | None = None
    county: str | None = None
    high_school: str | None = None
    financial_need: bool | None = None
    first_gen: bool | None = None
    gender: str | None = None
    heritage: list[str] = field(default_factory=list)
    military_family: bool | None = None
    disability: bool | None = None
    religion: str | None = None
    parent_employers: list[str] = field(default_factory=list)
    memberships: list[str] = field(default_factory=list)
    service_hours: int | None = None
    sat: int | None = None
    act: int | None = None
    intended_colleges: list[str] = field(default_factory=list)
    essay_ready: bool = False


def _coerce_bool(value: Any) -> bool | None:
    """Return a tri-state bool: ``None`` when the value is missing or unreadable."""
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


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _mapping_field(value: Any) -> dict[str, Any]:
    """Return a nested object column (``cycle``, ``min_test_scores``) as a dict."""
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


def _normalize_county(value: Any) -> str:
    """Normalize a county name so "Wake" and "Wake County" compare equal."""
    text = _normalize_text(value)
    if text.endswith(" county"):
        return text[: -len(" county")].strip()
    return text


def _names_overlap(required: list[str], student: list[str]) -> bool:
    """Return whether any required organization name matches a student's.

    Employer and membership names are free text ("Duke Energy" vs. "Duke Energy
    Corporation"), so containment in either direction counts as a match rather
    than only exact equality.
    """
    return any(
        required_name == student_name
        or required_name in student_name
        or student_name in required_name
        for required_name in required
        for student_name in student
    )


def _tristate_reason(
    required: Any,
    answer: bool | None,
    *,
    axis: str,
    code: str,
    reasons: list[str],
    unverified: list[str],
) -> None:
    """Apply one boolean restriction axis (need, first gen, military, disability)."""
    if _coerce_bool(required) is not True:
        return
    if answer is None:
        unverified.append(axis)
    elif not answer:
        reasons.append(code)


def _row_reasons(
    row: pd.Series,
    profile: StudentProfile,
    today: date,
    *,
    include_unconfirmed: bool = False,
) -> tuple[list[str], list[str]]:
    """Return ``(reason_codes, unverified_axes)`` for one scholarship row."""
    reasons: list[str] = []
    unverified: list[str] = []

    if not include_unconfirmed:
        trust = _normalize_text(row.get("trust"))
        if trust in UNCONFIRMED_TRUST_VALUES:
            reasons.append(TRUST_UNCONFIRMED_CODE)

    deadline_date = _row_deadline(row)
    recurring = _row_recurring(row)
    # A recurring award's listed deadline is a cycle date, not an expiry: it is
    # bucketed by the timeline instead of being filtered out here.
    if deadline_date is not None and deadline_date < today and not recurring:
        reasons.append("DEADLINE_PASSED")

    status = _normalize_text(row.get("status"))
    has_future_deadline = deadline_date is not None and deadline_date >= today
    if status == "closed" and not recurring and not has_future_deadline:
        reasons.append("STATUS_CLOSED_NONRECURRING")

    min_gpa = row.get("min_gpa")
    if profile.gpa is not None and min_gpa is not None and not pd.isna(min_gpa):
        if profile.gpa < float(min_gpa):
            reasons.append("GPA_BELOW_MIN")

    states_allowed = _normalize_list(row.get("states_allowed"))
    profile_state = _normalize_text(profile.state)
    if states_allowed and profile_state not in states_allowed:
        reasons.append("STATE_NOT_ALLOWED")

    counties_allowed = [
        _normalize_county(county) for county in _normalize_list(row.get("counties_allowed"))
    ]
    if counties_allowed:
        profile_county = _normalize_county(profile.county)
        if not profile_county:
            unverified.append("county")
        elif profile_county not in counties_allowed:
            reasons.append("COUNTY_NOT_ALLOWED")

    majors_allowed = _normalize_list(row.get("majors_allowed"))
    if majors_allowed and not majors_match(profile.major, majors_allowed):
        reasons.append("MAJOR_NOT_ALLOWED")

    scholarship_education_level = _normalize_text(row.get("education_level"))
    if scholarship_education_level and not education_level_matches(
        profile.education_level,
        scholarship_education_level,
        strict=profile.strict_education_level,
    ):
        reasons.append("EDUCATION_LEVEL_MISMATCH")

    grade_levels_allowed = _normalize_list(row.get("grade_levels"))
    if grade_levels_allowed:
        profile_grade = _normalize_text(profile.grade_level)
        student_rank = grade_level_rank(profile_grade)
        allowed_ranks = [
            rank
            for rank in (grade_level_rank(grade) for grade in grade_levels_allowed)
            if rank is not None
        ]
        if not profile_grade:
            unverified.append("grade_level")
        elif student_rank is not None and allowed_ranks and max(allowed_ranks) < student_rank:
            # Only an award the student has already aged out of is a mismatch; an
            # award aimed at a later grade is a future target, not a rejection.
            reasons.append("GRADE_LEVEL_MISMATCH")

    scholarship_citizenship = _normalize_text(row.get("citizenship"))
    profile_citizenship = _normalize_text(profile.citizenship)
    if scholarship_citizenship and scholarship_citizenship != profile_citizenship:
        reasons.append("CITIZENSHIP_MISMATCH")

    _tristate_reason(
        row.get("need_based"),
        profile.financial_need,
        axis="financial_need",
        code="NEED_BASED_NOT_MET",
        reasons=reasons,
        unverified=unverified,
    )
    _tristate_reason(
        row.get("first_gen_only"),
        profile.first_gen,
        axis="first_gen",
        code="FIRST_GEN_ONLY",
        reasons=reasons,
        unverified=unverified,
    )

    required_gender = _normalize_text(row.get("gender"))
    if required_gender and required_gender != "any":
        profile_gender = _normalize_text(profile.gender)
        if not profile_gender:
            unverified.append("gender")
        elif profile_gender != required_gender:
            reasons.append("GENDER_RESTRICTED")

    required_heritage = _normalize_list(row.get("heritage"))
    if required_heritage:
        profile_heritage = _normalize_list(profile.heritage)
        if not profile_heritage:
            unverified.append("heritage")
        elif not set(required_heritage) & set(profile_heritage):
            reasons.append("HERITAGE_RESTRICTED")

    _tristate_reason(
        row.get("military_family"),
        profile.military_family,
        axis="military_family",
        code="MILITARY_FAMILY_ONLY",
        reasons=reasons,
        unverified=unverified,
    )
    _tristate_reason(
        row.get("disability"),
        profile.disability,
        axis="disability",
        code="DISABILITY_RESTRICTED",
        reasons=reasons,
        unverified=unverified,
    )

    required_religion = _normalize_text(row.get("religion"))
    if required_religion:
        profile_religion = _normalize_text(profile.religion)
        if not profile_religion:
            unverified.append("religion")
        elif profile_religion != required_religion:
            reasons.append("RELIGION_RESTRICTED")

    required_employers = _normalize_list(row.get("employer_restricted"))
    if required_employers:
        profile_employers = _normalize_list(profile.parent_employers)
        if not profile_employers:
            unverified.append("parent_employers")
        elif not _names_overlap(required_employers, profile_employers):
            reasons.append("EMPLOYER_RESTRICTED")

    required_memberships = _normalize_list(row.get("membership_required"))
    if required_memberships:
        profile_memberships = _normalize_list(profile.memberships)
        if not profile_memberships:
            unverified.append("memberships")
        elif not _names_overlap(required_memberships, profile_memberships):
            reasons.append("MEMBERSHIP_REQUIRED")

    min_test_scores = _mapping_field(row.get("min_test_scores"))
    min_sat = _coerce_int(min_test_scores.get("sat"))
    min_act = _coerce_int(min_test_scores.get("act"))
    if min_sat is not None or min_act is not None:
        # An award stating both minimums accepts either test, so one score at or
        # above its own minimum clears the axis.
        comparisons = []
        if min_sat is not None and profile.sat is not None:
            comparisons.append(profile.sat >= min_sat)
        if min_act is not None and profile.act is not None:
            comparisons.append(profile.act >= min_act)
        if not comparisons:
            unverified.append("test_scores")
        elif not any(comparisons):
            reasons.append("TEST_SCORE_BELOW_MIN")

    amount_max = row.get("amount_max")
    amount_min = row.get("amount_min")
    has_amount_max = amount_max is not None and not pd.isna(amount_max)
    has_amount_min = amount_min is not None and not pd.isna(amount_min)
    if has_amount_max:
        resolved_amount = float(amount_max)
    elif has_amount_min:
        resolved_amount = float(amount_min)
    else:
        resolved_amount = None
    if resolved_amount is not None and resolved_amount <= 0.0:
        # A stated non-positive amount is a real disqualifier; an *unknown*
        # amount (both fields null) is a data-quality gap, not ineligibility,
        # so it passes through with zero amount utility (Stage 2/3) instead.
        reasons.append("AMOUNT_ZERO")

    return reasons, unverified


def apply_eligibility_filter(
    df: pd.DataFrame, profile: StudentProfile, *, include_unconfirmed: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split scholarships into eligible and ineligible sets based on hard filters.

    Applies the trust, deadline, status, GPA, geography, major, education and
    grade level, citizenship, restriction-axis, test-score, and amount checks.
    Each ineligible scholarship is tagged with all applicable reason codes; a
    restriction the profile cannot answer never disqualifies a row, it is
    recorded on ``unverified_axes`` for the student to confirm.

    Args:
        df: Scholarship DataFrame with normalized columns.
        profile: Student profile with eligibility attributes.
        include_unconfirmed: When ``True``, skip the trust check so an
            unconfirmed or aggregator record can be inspected in the pipeline.
            Operator-only: the family-facing path always enforces it.

    Returns:
        Tuple of ``(eligible_df, ineligible_df)``.  Both frames carry a
        ``reasons`` column with lists of reason codes such as
        ``"GPA_BELOW_MIN"``, ``"STATE_NOT_ALLOWED"`` and
        ``"TRUST_UNCONFIRMED"``, and an
        ``unverified_axes`` column with lists of axis keys such as
        ``"financial_need"`` (see :data:`UNVERIFIED_AXIS_LABELS`).
    """
    effective_today = profile.today or date.today()

    with_reasons_df = df.copy()
    evaluations = [
        _row_reasons(
            row=row,
            profile=profile,
            today=effective_today,
            include_unconfirmed=include_unconfirmed,
        )
        for _, row in with_reasons_df.iterrows()
    ]
    with_reasons_df["reasons"] = pd.Series(
        [reasons for reasons, _ in evaluations],
        index=with_reasons_df.index,
        dtype=object,
    )
    with_reasons_df["unverified_axes"] = pd.Series(
        [unverified for _, unverified in evaluations],
        index=with_reasons_df.index,
        dtype=object,
    )

    is_ineligible = with_reasons_df["reasons"].map(bool).astype(bool)
    ineligible_df = with_reasons_df[is_ineligible].copy()
    eligible_df = with_reasons_df[~is_ineligible].copy()

    return eligible_df, ineligible_df
