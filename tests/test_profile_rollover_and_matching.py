from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from app import state
from src.profile.grade_levels import grade_for_graduation_year, reconcile_grade
from src.profile.store import load_profile, normalize_profile, with_current_grade
from src.rank.stage1_eligibility import (
    StudentProfile,
    apply_eligibility_filter,
    canonical_citizenship,
    citizenship_accepts,
)

FALL_2026 = date(2026, 10, 11)


# -- grade rolls forward from the graduation year -----------------------------


@pytest.mark.parametrize(
    ("today", "grade"),
    [
        (date(2026, 10, 11), "10"),
        (date(2027, 6, 30), "10"),
        (date(2027, 7, 1), "11"),
        (date(2028, 9, 1), "12"),
        (date(2029, 8, 1), "college_1"),
    ],
)
def test_grade_follows_a_class_of_2029_graduation_year(today: date, grade: str) -> None:
    assert grade_for_graduation_year(2029, "10", today) == grade


def test_graduation_year_is_read_on_the_grade_track() -> None:
    # 2029 for a college sophomore is a college graduation.
    assert grade_for_graduation_year(2029, "college_2", FALL_2026) == "college_2"
    assert grade_for_graduation_year(2032, "10", FALL_2026) is None  # not started yet
    assert grade_for_graduation_year(None, "10", FALL_2026) is None


def test_reconcile_rolls_forward_unless_someone_changed_the_grade() -> None:
    next_summer = date(2027, 8, 1)
    assert reconcile_grade("10", 2029, "10", 2029, next_summer) == ("11", 2029)
    # A grade set by hand wins and moves the graduation year.
    assert reconcile_grade("11", 2029, "10", 2029, FALL_2026) == ("11", 2028)
    # A changed graduation year moves the grade.
    assert reconcile_grade("10", 2028, "10", 2029, FALL_2026) == ("11", 2028)
    # No year yet: estimated from the grade.
    assert reconcile_grade("10", None, "", None, FALL_2026) == ("10", 2029)
    # Past high school the year is re-anchored on college.
    assert reconcile_grade("12", 2029, "12", 2029, date(2029, 8, 1)) == ("college_1", 2033)
    assert reconcile_grade("", 2029, "10", 2029, FALL_2026) == ("", 2029)


def test_a_saved_profile_loads_in_its_current_grade(tmp_path: Path) -> None:
    profile = normalize_profile({"grade_level": "10", "education_level": "high school",
                                 "graduation_year": 2029})
    assert with_current_grade(profile, date(2028, 9, 1))["grade_level"] == "12"

    students = tmp_path / "students"
    students.mkdir()
    (students / "student_1.json").write_text(json.dumps(
        {**profile, "use_today_override": True, "today_override": "2027-07-15"}
    ))
    loaded = load_profile(students_dir=students)
    assert loaded is not None
    assert (loaded["grade_level"], loaded["graduation_year"]) == ("11", 2029)


# -- state and citizenship never fail silently --------------------------------


def test_profile_state_and_citizenship_are_stored_as_the_catalog_spells_them() -> None:
    profile = normalize_profile({"state": "north carolina", "citizenship": "U.S. citizen"})
    assert (profile["state"], profile["citizenship"]) == ("NC", "US")
    assert normalize_profile({"citizenship": "green card holder"})["citizenship"] == "Permanent Resident"
    assert normalize_profile({})["citizenship"] == ""


def test_citizenship_spellings_and_either_or_awards() -> None:
    assert canonical_citizenship("United States") == "US"
    assert citizenship_accepts("US", "u.s. citizen")
    assert not citizenship_accepts("US", "Permanent Resident")
    assert citizenship_accepts("U.S. citizen or permanent resident", "Permanent Resident")
    assert citizenship_accepts("US citizen or permanent resident", "US")


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"scholarship_id": "nc-only", "states_allowed": ["NC"], "citizenship": None},
            {"scholarship_id": "citizens", "states_allowed": [], "citizenship": "US"},
        ]
    )


def test_blank_state_or_citizenship_asks_instead_of_rejecting() -> None:
    eligible, ineligible = apply_eligibility_filter(_rows(), StudentProfile(today=FALL_2026))
    assert ineligible.empty
    axes = dict(zip(eligible["scholarship_id"], eligible["unverified_axes"], strict=True))
    assert axes["nc-only"] == ["state"]
    assert axes["citizens"] == ["citizenship"]


def test_a_stated_state_or_citizenship_still_rejects_a_mismatch() -> None:
    profile = StudentProfile(state="SC", citizenship="Permanent Resident", today=FALL_2026)
    _, ineligible = apply_eligibility_filter(_rows(), profile)
    reasons = dict(zip(ineligible["scholarship_id"], ineligible["reasons"], strict=True))
    assert reasons == {"nc-only": ["STATE_NOT_ALLOWED"], "citizens": ["CITIZENSHIP_MISMATCH"]}


def test_profile_widgets_offer_choices_matching_stage1() -> None:
    assert "NC" in state.STATE_CODE_OPTIONS
    assert state.state_label("NC") == "NC — North Carolina"
    assert state.state_label("") == "Not set"
    assert state._state_widget_value("North Carolina") == "NC"
    assert state._state_widget_value("Narnia") == ""
    assert state._citizenship_widget_value("u.s. citizen") == "US"
    assert state._citizenship_widget_value("Canadian") == state.OTHER_CITIZENSHIP
    assert state._citizenship_widget_value("") == ""
