from __future__ import annotations

from datetime import date

import pandas as pd

from src.rank.stage1_eligibility import StudentProfile, apply_eligibility_filter


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
            },
            {
                "scholarship_id": "gpa",
                "deadline": date(2026, 3, 1),
                "min_gpa": 3.8,
                "states_allowed": [],
                "majors_allowed": [],
                "education_level": None,
                "citizenship": None,
            },
            {
                "scholarship_id": "state",
                "deadline": date(2026, 3, 1),
                "min_gpa": None,
                "states_allowed": ["NV", "AZ"],
                "majors_allowed": [],
                "education_level": None,
                "citizenship": None,
            },
            {
                "scholarship_id": "major",
                "deadline": date(2026, 3, 1),
                "min_gpa": None,
                "states_allowed": [],
                "majors_allowed": ["History"],
                "education_level": None,
                "citizenship": None,
            },
            {
                "scholarship_id": "education",
                "deadline": date(2026, 3, 1),
                "min_gpa": None,
                "states_allowed": [],
                "majors_allowed": [],
                "education_level": "Graduate",
                "citizenship": None,
            },
            {
                "scholarship_id": "citizenship",
                "deadline": date(2026, 3, 1),
                "min_gpa": None,
                "states_allowed": [],
                "majors_allowed": [],
                "education_level": None,
                "citizenship": "Canada",
            },
            {
                "scholarship_id": "eligible",
                "deadline": date(2026, 3, 1),
                "min_gpa": 3.0,
                "states_allowed": ["CA", "WA"],
                "majors_allowed": ["Computer Science", "Math"],
                "education_level": "Undergraduate",
                "citizenship": "US",
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
