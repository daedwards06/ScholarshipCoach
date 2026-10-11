from __future__ import annotations

import pytest

from app.sections import my_path
from app.sections import what_if as page
from src.rank.stage1_eligibility import StudentProfile
from src.store import requirements
from src.store.repo import CollegeRequirement


def _req(**values: object) -> CollegeRequirement:
    fields: dict = {"id": 1, "college_id": 1, "label": "x"}
    fields.update(values)
    return CollegeRequirement(**fields)


PROGRAM_MINIMUM = _req(id=1, label="Program minimum", category="gpa", target="3.5")
FULL_COST = _req(id=2, label="Full-cost scholarship", category="gpa", target="3.75+")
SAT_GOAL = _req(id=3, label="SAT for merit aid", category="test", target="1200")
ACT_GOAL = _req(id=4, label="Admission test", category="test", target="ACT 26")


@pytest.mark.parametrize(
    ("value", "reached", "text"),
    [
        (3.25, False, "3.25: 0.25 short of Program minimum"),
        (3.5, True, "3.50: reaches Program minimum"),
        (3.8, True, "3.80: reaches Program minimum"),
    ],
)
def test_gpa_goal_below_at_and_above_threshold(value: float, reached: bool, text: str) -> None:
    [goal] = requirements.goal_reaches([PROGRAM_MINIMUM], {"gpa": value})
    assert goal.reached is reached
    assert requirements.goal_reach_text(goal) == text


@pytest.mark.parametrize(
    ("requirement", "values", "reached", "text"),
    [
        (SAT_GOAL, {"sat": 1150}, False, "1150: 50 short of SAT for merit aid"),
        (SAT_GOAL, {"sat": 1200}, True, "1200: reaches SAT for merit aid"),
        (SAT_GOAL, {"sat": 1350}, True, "1350: reaches SAT for merit aid"),
        (ACT_GOAL, {"act": 24}, False, "24: 2 short of Admission test"),
        (ACT_GOAL, {"act": 26}, True, "26: reaches Admission test"),
        (ACT_GOAL, {"act": 30}, True, "30: reaches Admission test"),
    ],
)
def test_test_goal_below_at_and_above_threshold(
    requirement: CollegeRequirement, values: dict, reached: bool, text: str
) -> None:
    [goal] = requirements.goal_reaches([requirement], values)
    assert goal.reached is reached
    assert requirements.goal_reach_text(goal) == text


def test_one_value_reads_against_every_goal() -> None:
    goals = requirements.goal_reaches([PROGRAM_MINIMUM, FULL_COST], {"gpa": 3.5})
    assert [requirements.goal_reach_text(goal) for goal in goals] == [
        "3.50: reaches Program minimum",
        "3.50: 0.25 short of Full-cost scholarship",
    ]


def test_no_score_yet_is_not_reached() -> None:
    [goal] = requirements.goal_reaches([SAT_GOAL], {"sat": 0})
    assert not goal.reached
    assert requirements.goal_reach_text(goal) == "No SAT yet for SAT for merit aid"


@pytest.mark.parametrize(
    ("requirement", "measure"),
    [
        (PROGRAM_MINIMUM, "gpa"),
        (SAT_GOAL, "sat"),
        (ACT_GOAL, "act"),
        (_req(category="test", target="28"), "act"),
        (_req(label="SAT", category="test", target="30"), "sat"),
        (_req(label="Take the PSAT", category="test", target="1100"), None),
        (_req(label="Take the PSAT", category="test", target=""), None),
        (_req(category="gpa", target="strong"), None),
        (_req(category="course", target="1 unit"), None),
    ],
)
def test_goal_measure(requirement: CollegeRequirement, measure: str | None) -> None:
    assert requirements.goal_measure(requirement) == measure


def test_not_needed_rows_are_not_goals() -> None:
    skipped = _req(category="gpa", target="3.0", status="not_needed")
    assert requirements.goal_reaches([skipped], {"gpa": 3.25}) == []


def test_unknown_amount_reads_as_not_listed() -> None:
    assert page.award_amount_text(None) == "Amount not listed"
    assert page.award_amount_text(0.0) == "Amount not listed"
    assert page.award_amount_text(2500.0) == "$2,500"


def test_my_path_link_presets_the_target() -> None:
    params = my_path.what_if_params(FULL_COST)
    assert page.preset_values([PROGRAM_MINIMUM, FULL_COST], params[page.GOAL_QUERY_PARAM]) == {
        "gpa": 3.75
    }
    assert page.preset_values([ACT_GOAL], "4") == {"act": 26.0}


@pytest.mark.parametrize("goal", [None, "", "abc", "99"])
def test_a_missing_or_stale_goal_presets_nothing(goal: str | None) -> None:
    assert page.preset_values([PROGRAM_MINIMUM], goal) == {}


def test_goals_heading_speaks_to_the_reader() -> None:
    assert page.goals_heading("student") == "Your goals"
    assert page.goals_heading("parent") == "Her goals"


def test_controls_left_at_a_blank_fields_start_ask_nothing() -> None:
    profile = StudentProfile(gpa=3.25, service_hours=None, sat=None, first_gen=None)
    asked = page.asked_overrides(
        profile, {"gpa": 3.25, "service_hours": 0, "sat": None, "first_gen": False}
    )
    assert asked == {"gpa": 3.25}
    assert page.asked_overrides(profile, {"service_hours": 20}) == {"service_hours": 20}
