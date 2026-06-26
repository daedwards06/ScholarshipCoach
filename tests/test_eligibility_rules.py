from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.rank.stage1_eligibility import StudentProfile, apply_eligibility_filter


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
        ({"amount_max": 0.0}, "AMOUNT_MISSING_OR_ZERO"),
        ({"amount_max": None}, "AMOUNT_MISSING_OR_ZERO"),
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


def test_zero_amount_filtered_below_positive_amount(sample_profile: StudentProfile) -> None:
    """A $0-amount scholarship is filtered in Stage 1; a $5000 one passes through."""
    df = pd.DataFrame(
        [
            _row(scholarship_id="zero-amount", amount_max=0.0, amount_min=None),
            _row(scholarship_id="missing-amount", amount_max=None, amount_min=None),
            _row(scholarship_id="positive-amount", amount_max=5000.0),
        ]
    )
    eligible_df, ineligible_df = apply_eligibility_filter(df=df, profile=sample_profile)

    assert eligible_df["scholarship_id"].tolist() == ["positive-amount"]
    ineligible_ids = set(ineligible_df["scholarship_id"])
    assert ineligible_ids == {"zero-amount", "missing-amount"}
    for _, row in ineligible_df.iterrows():
        assert "AMOUNT_MISSING_OR_ZERO" in row["reasons"]
