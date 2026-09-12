from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.helpers import unverified_to_text
from src.rank.stage1_eligibility import StudentProfile, apply_eligibility_filter
from src.rank.taxonomy import education_level_matches, majors_match


def _row(**kwargs) -> dict:
    base = {
        "scholarship_id": "test",
        "deadline": date(2026, 3, 1),
        "min_gpa": None,
        "states_allowed": [],
        "majors_allowed": [],
        "education_level": None,
        "citizenship": None,
        "amount_max": 5000.0,
        "amount_min": None,
    }
    base.update(kwargs)
    return base


@pytest.mark.parametrize(
    "override,expected_reason",
    [
        ({"deadline": date(2026, 2, 1)}, "DEADLINE_PASSED"),
        ({"min_gpa": 3.8}, "GPA_BELOW_MIN"),
        ({"states_allowed": ["NV", "AZ"]}, "STATE_NOT_ALLOWED"),
        ({"majors_allowed": ["History"]}, "MAJOR_NOT_ALLOWED"),
        ({"education_level": "Graduate"}, "EDUCATION_LEVEL_MISMATCH"),
        ({"citizenship": "Canada"}, "CITIZENSHIP_MISMATCH"),
        ({"amount_max": 0.0}, "AMOUNT_ZERO"),
        ({"amount_min": -100.0, "amount_max": None}, "AMOUNT_ZERO"),
    ],
)
def test_each_reason_code_is_emitted_individually(
    sample_profile: StudentProfile,
    override: dict,
    expected_reason: str,
) -> None:
    df = pd.DataFrame([_row(**override)])
    _, ineligible_df = apply_eligibility_filter(df=df, profile=sample_profile)
    assert ineligible_df.iloc[0]["reasons"] == [expected_reason]


def test_apply_eligibility_filter_emits_reason_codes_for_each_rule() -> None:
    today = date(2026, 2, 22)
    profile = StudentProfile(
        gpa=3.2,
        state="CA",
        major="Computer Science",
        education_level="Undergraduate",
        citizenship="US",
        today=today,
    )

    df = pd.DataFrame(
        [
            {
                "scholarship_id": "deadline",
                "deadline": date(2026, 2, 1),
                "min_gpa": None,
                "states_allowed": [],
                "majors_allowed": [],
                "education_level": None,
                "citizenship": None,
                "amount_max": 5000.0,
            },
            {
                "scholarship_id": "gpa",
                "deadline": date(2026, 3, 1),
                "min_gpa": 3.8,
                "states_allowed": [],
                "majors_allowed": [],
                "education_level": None,
                "citizenship": None,
                "amount_max": 5000.0,
            },
            {
                "scholarship_id": "state",
                "deadline": date(2026, 3, 1),
                "min_gpa": None,
                "states_allowed": ["NV", "AZ"],
                "majors_allowed": [],
                "education_level": None,
                "citizenship": None,
                "amount_max": 5000.0,
            },
            {
                "scholarship_id": "major",
                "deadline": date(2026, 3, 1),
                "min_gpa": None,
                "states_allowed": [],
                "majors_allowed": ["History"],
                "education_level": None,
                "citizenship": None,
                "amount_max": 5000.0,
            },
            {
                "scholarship_id": "education",
                "deadline": date(2026, 3, 1),
                "min_gpa": None,
                "states_allowed": [],
                "majors_allowed": [],
                "education_level": "Graduate",
                "citizenship": None,
                "amount_max": 5000.0,
            },
            {
                "scholarship_id": "citizenship",
                "deadline": date(2026, 3, 1),
                "min_gpa": None,
                "states_allowed": [],
                "majors_allowed": [],
                "education_level": None,
                "citizenship": "Canada",
                "amount_max": 5000.0,
            },
            {
                "scholarship_id": "eligible",
                "deadline": date(2026, 3, 1),
                "min_gpa": 3.0,
                "states_allowed": ["CA", "WA"],
                "majors_allowed": ["Computer Science", "Math"],
                "education_level": "Undergraduate",
                "citizenship": "US",
                "amount_max": 5000.0,
            },
        ]
    )

    eligible_df, ineligible_df = apply_eligibility_filter(df=df, profile=profile)
    reasons_by_id = {
        row["scholarship_id"]: row["reasons"] for _, row in ineligible_df.iterrows()
    }

    assert reasons_by_id["deadline"] == ["DEADLINE_PASSED"]
    assert reasons_by_id["gpa"] == ["GPA_BELOW_MIN"]
    assert reasons_by_id["state"] == ["STATE_NOT_ALLOWED"]
    assert reasons_by_id["major"] == ["MAJOR_NOT_ALLOWED"]
    assert reasons_by_id["education"] == ["EDUCATION_LEVEL_MISMATCH"]
    assert reasons_by_id["citizenship"] == ["CITIZENSHIP_MISMATCH"]
    assert eligible_df["scholarship_id"].tolist() == ["eligible"]
    assert "reasons" in ineligible_df.columns


def test_case_insensitive_matching_does_not_reject() -> None:
    profile = StudentProfile(
        gpa=3.6,
        state="ca",
        major="computer science",
        education_level="undergraduate",
        citizenship="us",
        today=date(2026, 2, 22),
    )
    df = pd.DataFrame(
        [
            {
                "scholarship_id": "case-ok",
                "deadline": date(2026, 4, 1),
                "min_gpa": 3.0,
                "states_allowed": ["CA"],
                "majors_allowed": ["Computer Science"],
                "education_level": "UNDERGRADUATE",
                "citizenship": "US",
                "amount_max": 5000.0,
            }
        ]
    )

    eligible_df, ineligible_df = apply_eligibility_filter(df=df, profile=profile)

    assert eligible_df["scholarship_id"].tolist() == ["case-ok"]
    assert ineligible_df.empty


def test_apply_eligibility_filter_collects_multiple_reasons() -> None:
    profile = StudentProfile(
        gpa=2.5,
        state="CA",
        major="Computer Science",
        education_level="Undergraduate",
        citizenship="US",
        today=date(2026, 2, 22),
    )
    df = pd.DataFrame(
        [
            {
                "scholarship_id": "multi",
                "deadline": date(2026, 1, 1),
                "min_gpa": 3.8,
                "states_allowed": ["NV"],
                "majors_allowed": ["History"],
                "education_level": "Graduate",
                "citizenship": "Canada",
                "amount_max": 5000.0,
            }
        ]
    )

    eligible_df, ineligible_df = apply_eligibility_filter(df=df, profile=profile)

    assert eligible_df.empty
    assert ineligible_df["scholarship_id"].tolist() == ["multi"]
    assert ineligible_df.iloc[0]["reasons"] == [
        "DEADLINE_PASSED",
        "GPA_BELOW_MIN",
        "STATE_NOT_ALLOWED",
        "MAJOR_NOT_ALLOWED",
        "EDUCATION_LEVEL_MISMATCH",
        "CITIZENSHIP_MISMATCH",
    ]


def test_family_match_passes_for_related_major(sample_profile: StudentProfile) -> None:
    """A Computer Science profile matches Computer Engineering / STEM awards."""
    df = pd.DataFrame(
        [
            _row(scholarship_id="ce", majors_allowed=["Computer Engineering"]),
            _row(scholarship_id="stem", majors_allowed=["STEM"]),
            _row(scholarship_id="eng", majors_allowed=["Engineering"]),
        ]
    )
    eligible_df, ineligible_df = apply_eligibility_filter(df=df, profile=sample_profile)

    assert set(eligible_df["scholarship_id"]) == {"ce", "stem", "eng"}
    assert ineligible_df.empty


def test_unrelated_major_still_fails(sample_profile: StudentProfile) -> None:
    """A Computer Science profile is rejected from an unrelated (nursing) award."""
    df = pd.DataFrame([_row(scholarship_id="health", majors_allowed=["Nursing"])])
    _, ineligible_df = apply_eligibility_filter(df=df, profile=sample_profile)

    assert ineligible_df.iloc[0]["reasons"] == ["MAJOR_NOT_ALLOWED"]


def test_unknown_major_falls_back_to_exact_match() -> None:
    """A major absent from the taxonomy only matches by exact string."""
    profile = StudentProfile(major="Underwater Basket Weaving", today=date(2026, 2, 22))
    assert majors_match("Underwater Basket Weaving", ["underwater basket weaving"]) is True
    assert majors_match("Underwater Basket Weaving", ["Engineering"]) is False
    # No restriction (empty allowed list / empty scholarship level) always passes.
    assert majors_match("Underwater Basket Weaving", []) is True
    assert education_level_matches("high school", None) is True
    df = pd.DataFrame([_row(scholarship_id="niche", majors_allowed=["Engineering"])])
    _, ineligible_df = apply_eligibility_filter(df=df, profile=profile)
    assert ineligible_df.iloc[0]["reasons"] == ["MAJOR_NOT_ALLOWED"]


def test_high_school_matches_undergraduate_by_adjacency() -> None:
    """A college-bound high-school profile passes an undergraduate-labeled award."""
    profile = StudentProfile(
        major="Computer Science",
        education_level="high school",
        today=date(2026, 2, 22),
    )
    df = pd.DataFrame(
        [
            _row(scholarship_id="ug", education_level="Undergraduate"),
            _row(scholarship_id="grad", education_level="Graduate"),
        ]
    )
    eligible_df, ineligible_df = apply_eligibility_filter(df=df, profile=profile)

    assert eligible_df["scholarship_id"].tolist() == ["ug"]
    assert ineligible_df.iloc[0]["scholarship_id"] == "grad"
    assert ineligible_df.iloc[0]["reasons"] == ["EDUCATION_LEVEL_MISMATCH"]


def test_strict_education_level_restores_exact_behavior() -> None:
    """The strict escape hatch rejects the high-school ↔ undergraduate adjacency."""
    profile = StudentProfile(
        major="Computer Science",
        education_level="high school",
        strict_education_level=True,
        today=date(2026, 2, 22),
    )
    df = pd.DataFrame([_row(scholarship_id="ug", education_level="Undergraduate")])
    eligible_df, ineligible_df = apply_eligibility_filter(df=df, profile=profile)

    assert eligible_df.empty
    assert ineligible_df.iloc[0]["reasons"] == ["EDUCATION_LEVEL_MISMATCH"]
    assert education_level_matches("high school", "Undergraduate") is True
    assert education_level_matches("high school", "Undergraduate", strict=True) is False


def test_stated_zero_amount_filtered_but_unknown_amount_passes(
    sample_profile: StudentProfile,
) -> None:
    """A stated $0 amount is filtered as AMOUNT_ZERO; an *unknown* amount passes Stage 1."""
    df = pd.DataFrame(
        [
            _row(scholarship_id="zero-amount", amount_max=0.0, amount_min=None),
            _row(scholarship_id="unknown-amount", amount_max=None, amount_min=None),
            _row(scholarship_id="positive-amount", amount_max=5000.0),
        ]
    )
    eligible_df, ineligible_df = apply_eligibility_filter(df=df, profile=sample_profile)

    assert set(eligible_df["scholarship_id"]) == {"unknown-amount", "positive-amount"}
    assert ineligible_df["scholarship_id"].tolist() == ["zero-amount"]
    assert ineligible_df.iloc[0]["reasons"] == ["AMOUNT_ZERO"]
    # An unknown-amount row carries no reason code — it is eligible, not disqualified.
    unknown_row = eligible_df[eligible_df["scholarship_id"] == "unknown-amount"].iloc[0]
    assert unknown_row["reasons"] == []


def _axis_row(**kwargs) -> dict:
    """A row that clears every legacy rule, so only the new axes can fire."""
    base = _row(
        scholarship_id="axis",
        status="open",
        cycle={"recurring": None, "opens_month": None, "deadline_month": None},
        grade_levels=[],
        counties_allowed=[],
        need_based=None,
        first_gen_only=None,
        gender=None,
        heritage=[],
        military_family=None,
        disability=None,
        religion=None,
        employer_restricted=[],
        membership_required=[],
        min_test_scores={"sat": None, "act": None},
    )
    base.update(kwargs)
    return base


def _answered_profile(**kwargs) -> StudentProfile:
    """A profile that answers every restriction axis (so nothing is unverified)."""
    base = {
        "gpa": 3.5,
        "state": "CA",
        "major": "Computer Science",
        "education_level": "Undergraduate",
        "citizenship": "US",
        "today": date(2026, 2, 22),
        "grade_level": "college_2",
        "county": "Wake",
        "financial_need": True,
        "first_gen": True,
        "gender": "female",
        "heritage": ["Hispanic"],
        "military_family": True,
        "disability": True,
        "religion": "Methodist",
        "parent_employers": ["Duke Energy"],
        "memberships": ["4-H"],
        "sat": 1400,
        "act": 31,
    }
    base.update(kwargs)
    return StudentProfile(**base)


_AXIS_CASES = [
    (
        "COUNTY_NOT_ALLOWED",
        {"counties_allowed": ["Durham County"]},
        {"county": "Wake"},
        {"county": "Durham"},
        "county",
    ),
    (
        "GRADE_LEVEL_MISMATCH",
        {"grade_levels": ["9", "10"]},
        {"grade_level": "college_2"},
        {"grade_level": "10"},
        "grade_level",
    ),
    ("NEED_BASED_NOT_MET", {"need_based": True}, {"financial_need": False}, {}, "financial_need"),
    ("FIRST_GEN_ONLY", {"first_gen_only": True}, {"first_gen": False}, {}, "first_gen"),
    ("GENDER_RESTRICTED", {"gender": "female"}, {"gender": "male"}, {}, "gender"),
    ("HERITAGE_RESTRICTED", {"heritage": ["Hispanic"]}, {"heritage": ["Korean"]}, {}, "heritage"),
    (
        "MILITARY_FAMILY_ONLY",
        {"military_family": True},
        {"military_family": False},
        {},
        "military_family",
    ),
    ("DISABILITY_RESTRICTED", {"disability": True}, {"disability": False}, {}, "disability"),
    ("RELIGION_RESTRICTED", {"religion": "Methodist"}, {"religion": "Baptist"}, {}, "religion"),
    (
        "EMPLOYER_RESTRICTED",
        {"employer_restricted": ["Duke Energy"]},
        {"parent_employers": ["Wells Fargo"]},
        {},
        "parent_employers",
    ),
    (
        "MEMBERSHIP_REQUIRED",
        {"membership_required": ["4-H"]},
        {"memberships": ["Key Club"]},
        {},
        "memberships",
    ),
    (
        "TEST_SCORE_BELOW_MIN",
        {"min_test_scores": {"sat": 1500, "act": 34}},
        {"sat": 1100, "act": 24},
        {"sat": 1550, "act": 35},
        "test_scores",
    ),
]


@pytest.mark.parametrize(
    "expected_reason,row_override,failing_profile,passing_profile,axis",
    _AXIS_CASES,
    ids=[case[0] for case in _AXIS_CASES],
)
def test_each_new_axis_fires_clears_and_records_unverified(
    expected_reason: str,
    row_override: dict,
    failing_profile: dict,
    passing_profile: dict,
    axis: str,
) -> None:
    """Each new axis disqualifies on a stated "no", clears on a "yes", and is
    recorded as unverified (not disqualifying) when the profile is silent."""
    df = pd.DataFrame([_axis_row(**row_override)])

    _, ineligible_df = apply_eligibility_filter(
        df=df, profile=_answered_profile(**failing_profile)
    )
    assert ineligible_df.iloc[0]["reasons"] == [expected_reason]
    assert ineligible_df.iloc[0]["unverified_axes"] == []

    eligible_df, _ = apply_eligibility_filter(
        df=df, profile=_answered_profile(**passing_profile)
    )
    assert eligible_df["scholarship_id"].tolist() == ["axis"]
    assert eligible_df.iloc[0]["unverified_axes"] == []

    unanswered = {
        key: ([] if isinstance(value, list) else None) for key, value in failing_profile.items()
    }
    unverified_eligible_df, unverified_ineligible_df = apply_eligibility_filter(
        df=df, profile=_answered_profile(**unanswered)
    )
    assert unverified_ineligible_df.empty
    assert unverified_eligible_df.iloc[0]["reasons"] == []
    assert unverified_eligible_df.iloc[0]["unverified_axes"] == [axis]


def test_closed_nonrecurring_is_ineligible_but_closed_recurring_passes(
    sample_profile: StudentProfile,
) -> None:
    """A closed one-off award is dead; a closed recurring one is next year's target."""
    df = pd.DataFrame(
        [
            _axis_row(
                scholarship_id="closed-oneoff",
                status="closed",
                deadline=date(2026, 1, 15),
                cycle={"recurring": False, "opens_month": None, "deadline_month": 1},
            ),
            _axis_row(
                scholarship_id="closed-recurring",
                status="closed",
                deadline=date(2026, 1, 15),
                cycle={"recurring": True, "opens_month": None, "deadline_month": 1},
            ),
            _axis_row(
                scholarship_id="closed-next-cycle-dated",
                status="closed",
                deadline=date(2026, 12, 1),
                cycle={"recurring": False, "opens_month": None, "deadline_month": 12},
            ),
        ]
    )

    eligible_df, ineligible_df = apply_eligibility_filter(df=df, profile=sample_profile)

    assert set(eligible_df["scholarship_id"]) == {
        "closed-recurring",
        "closed-next-cycle-dated",
    }
    assert ineligible_df["scholarship_id"].tolist() == ["closed-oneoff"]
    assert ineligible_df.iloc[0]["reasons"] == [
        "DEADLINE_PASSED",
        "STATUS_CLOSED_NONRECURRING",
    ]


def test_future_grade_award_passes_and_past_grade_award_fails() -> None:
    """A seniors-only award survives Stage 1 for a sophomore (the timeline buckets it)."""
    profile = _answered_profile(grade_level="10")
    df = pd.DataFrame(
        [
            _axis_row(scholarship_id="seniors-only", grade_levels=["12"]),
            _axis_row(scholarship_id="current", grade_levels=["10", "11"]),
            _axis_row(scholarship_id="already-past", grade_levels=["9"]),
        ]
    )

    eligible_df, ineligible_df = apply_eligibility_filter(df=df, profile=profile)

    assert set(eligible_df["scholarship_id"]) == {"seniors-only", "current"}
    assert ineligible_df["scholarship_id"].tolist() == ["already-past"]
    assert ineligible_df.iloc[0]["reasons"] == ["GRADE_LEVEL_MISMATCH"]


def test_either_test_score_at_minimum_clears_the_axis() -> None:
    """An award listing both minimums accepts whichever test the student has."""
    df = pd.DataFrame([_axis_row(min_test_scores={"sat": 1500, "act": 30})])

    act_only_eligible, _ = apply_eligibility_filter(
        df=df, profile=_answered_profile(sat=None, act=32)
    )
    assert act_only_eligible["scholarship_id"].tolist() == ["axis"]

    _, both_low_ineligible = apply_eligibility_filter(
        df=df, profile=_answered_profile(sat=1200, act=25)
    )
    assert both_low_ineligible.iloc[0]["reasons"] == ["TEST_SCORE_BELOW_MIN"]


def test_unverified_axes_accumulate_and_never_disqualify() -> None:
    """A profile that answers nothing still passes, carrying every open axis."""
    profile = StudentProfile(
        gpa=3.5,
        state="CA",
        major="Computer Science",
        education_level="Undergraduate",
        citizenship="US",
        today=date(2026, 2, 22),
    )
    df = pd.DataFrame(
        [
            _axis_row(
                counties_allowed=["Wake"],
                grade_levels=["college_2"],
                need_based=True,
                first_gen_only=True,
                gender="female",
                heritage=["Hispanic"],
                military_family=True,
                disability=True,
                religion="Methodist",
                employer_restricted=["Duke Energy"],
                membership_required=["4-H"],
                min_test_scores={"sat": 1500, "act": 34},
            )
        ]
    )

    eligible_df, ineligible_df = apply_eligibility_filter(df=df, profile=profile)

    assert ineligible_df.empty
    assert eligible_df.iloc[0]["reasons"] == []
    assert eligible_df.iloc[0]["unverified_axes"] == [
        "county",
        "grade_level",
        "financial_need",
        "first_gen",
        "gender",
        "heritage",
        "military_family",
        "disability",
        "religion",
        "parent_employers",
        "memberships",
        "test_scores",
    ]
    assert unverified_to_text(eligible_df.iloc[0]["unverified_axes"]).startswith(
        "county residency, grade level, financial need"
    )


def test_unverified_axes_column_present_on_both_frames(
    sample_profile: StudentProfile,
) -> None:
    """The new column ships on the eligible and the ineligible frame alike."""
    df = pd.DataFrame(
        [
            _axis_row(scholarship_id="ok"),
            _axis_row(scholarship_id="no", min_gpa=3.9),
        ]
    )

    eligible_df, ineligible_df = apply_eligibility_filter(df=df, profile=sample_profile)

    assert "unverified_axes" in eligible_df.columns
    assert "unverified_axes" in ineligible_df.columns
    assert len(eligible_df) == 1 and len(ineligible_df) == 1


def test_employer_and_membership_names_match_on_containment() -> None:
    """Free-text organization names match without exact spelling agreement."""
    df = pd.DataFrame(
        [
            _axis_row(
                employer_restricted=["Duke Energy"],
                membership_required=["4-H"],
            )
        ]
    )
    profile = _answered_profile(
        parent_employers=["Duke Energy Corporation"], memberships=["4-H Club of Wake County"]
    )

    eligible_df, ineligible_df = apply_eligibility_filter(df=df, profile=profile)

    assert ineligible_df.empty
    assert eligible_df.iloc[0]["reasons"] == []


def test_unverified_to_text_renders_labels_and_handles_missing() -> None:
    assert unverified_to_text(["financial_need"]) == "financial need"
    assert (
        unverified_to_text(["county", "test_scores"])
        == "county residency, test score minimum"
    )
    assert unverified_to_text([]) == ""
    assert unverified_to_text(None) == ""
    # An axis with no label yet still shows, spelled out rather than dropped.
    assert unverified_to_text(["future_axis"]) == "future axis"
