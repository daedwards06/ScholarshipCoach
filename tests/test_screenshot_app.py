from __future__ import annotations

from pathlib import Path

import pytest

from app.modes import nav_sections
from scripts import screenshot_app
from src.profile.store import with_current_grade


def test_plan_covers_every_section_of_both_modes_at_both_widths_and_themes() -> None:
    shots = screenshot_app.plan_captures(screenshot_app.selected_themes("both"))

    student, parent = nav_sections("student"), nav_sections("parent")
    assert len(shots) == (len(student) + len(parent)) * 2 * 2
    assert {shot.width for shot in shots} == {400, 1366}
    assert {shot.theme for shot in shots} == {"light", "dark"}
    assert {s.section for s in shots if s.mode == "parent"} == set(parent)
    assert {s.section for s in shots if s.mode == "student"} == set(student)
    assert "profile" in student
    assert len({shot.filename for shot in shots}) == len(shots)


def test_plan_filters_sections() -> None:
    shots = screenshot_app.plan_captures(("dark",), {"timeline"})

    assert [shot.filename for shot in shots] == [
        "400_parent_timeline_dark.png",
        "1366_parent_timeline_dark.png",
    ]


def test_section_url_opens_the_page_by_its_url_path() -> None:
    url = screenshot_app.section_url("http://127.0.0.1:8501", "parent", "colleges_money")

    assert url == "http://127.0.0.1:8501/colleges-money?shot_mode=parent"


def test_section_url_opens_the_default_page_at_the_root() -> None:
    url = screenshot_app.section_url("http://127.0.0.1:8501", "student", "my_path")

    assert url == "http://127.0.0.1:8501/?shot_mode=student"
    assert screenshot_app.section_url("http://127.0.0.1:8501", "student", "todos") == (
        "http://127.0.0.1:8501/to-dos?shot_mode=student"
    )


def test_scratch_db_is_the_default_and_output_is_outside_the_repo() -> None:
    args = screenshot_app.parse_args([])

    assert args.scratch_db is True
    assert screenshot_app.ROOT_DIR not in Path(args.out).resolve().parents


def test_seed_refuses_the_live_database() -> None:
    with pytest.raises(SystemExit):
        screenshot_app.parse_args(["--live-db", "--seed"])


def test_journeys_walk_every_plan_journey_once() -> None:
    ids = [journey.id for journey in screenshot_app.journeys()]

    assert sorted(ids) == ["P1", "P2", "P3", "P4", "S1", "S2", "S3", "S4", "S5", "S6"]
    for journey in screenshot_app.journeys():
        assert journey.who == ("student" if journey.id.startswith("S") else "parent")


def test_journey_budgets_match_the_plan() -> None:
    budgets = {
        journey.id: screenshot_app.budget_text(journey.budget)
        for journey in screenshot_app.journeys()
    }

    assert budgets == {
        "S1": "0", "S2": "1", "S3": "2", "S4": "2", "S5": "1", "S6": "2",
        "P1": "3 + nav", "P2": "1 + form", "P3": "1", "P4": "0 + form",
    }


def _journey(budget: screenshot_app.Budget, *kinds: str) -> screenshot_app.Journey:
    """A student journey of ``kinds`` steps, plus a look at the parent's view."""
    steps = [screenshot_app.Step(kind, kind, lambda walker: None) for kind in kinds]
    steps.append(screenshot_app.Step("parent view", "check", lambda walker: None, mode="parent"))
    return screenshot_app.Journey("X1", "student", "test", budget, tuple(steps))


def test_a_nav_the_budget_allows_is_not_charged_as_taps() -> None:
    journey = _journey(screenshot_app.Budget(3, navs=1), "open", "nav", "tap", "tap")
    cost = screenshot_app.journey_cost(journey)

    assert cost == screenshot_app.Cost(taps=2, navs=1, forms=0)
    assert screenshot_app.within_budget(cost, journey.budget)
    assert screenshot_app.cost_text(cost, journey.budget) == "2 + nav"


def test_a_nav_beyond_the_budget_costs_three_taps() -> None:
    journey = _journey(screenshot_app.Budget(1, forms=1), "open", "form", "tap", "nav", "tap")
    cost = screenshot_app.journey_cost(journey)

    assert screenshot_app.charged_taps(cost, journey.budget) == 5
    assert not screenshot_app.within_budget(cost, journey.budget)
    assert screenshot_app.cost_text(cost, journey.budget) == "5 + form"


def test_typing_and_the_other_persons_view_cost_nothing() -> None:
    journey = _journey(screenshot_app.Budget(2), "open", "tap", "type", "link")

    assert screenshot_app.journey_cost(journey).taps == 2


def test_every_step_writes_its_own_png() -> None:
    names = [
        screenshot_app.step_filename(journey, index, step)
        for journey in screenshot_app.journeys()
        for index, step in enumerate(journey.steps, start=1)
    ]

    assert len(set(names)) == len(names)
    assert names[0] == "journey_S1_01_open-the-app.png"


def test_journeys_refuse_the_live_database() -> None:
    with pytest.raises(SystemExit):
        screenshot_app.parse_args(["--journeys", "--live-db"])


def test_journeys_use_the_demo_profile_as_a_10th_grader() -> None:
    profile = screenshot_app.demo_profile_in_grade("10")

    assert profile["grade_level"] == "10"
    assert profile["student_id"] == screenshot_app.DEFAULT_STUDENT_ID
    assert profile["name"].endswith("(demo)")
    assert with_current_grade(profile)["grade_level"] == "10"
