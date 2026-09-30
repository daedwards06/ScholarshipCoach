from __future__ import annotations

from pathlib import Path

import pytest

from app.modes import nav_sections
from scripts import screenshot_app


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
    url = screenshot_app.section_url("http://127.0.0.1:8501", "student", "this_week")

    assert url == "http://127.0.0.1:8501/?shot_mode=student"


def test_scratch_db_is_the_default_and_output_is_outside_the_repo() -> None:
    args = screenshot_app.parse_args([])

    assert args.scratch_db is True
    assert screenshot_app.ROOT_DIR not in Path(args.out).resolve().parents


def test_seed_refuses_the_live_database() -> None:
    with pytest.raises(SystemExit):
        screenshot_app.parse_args(["--live-db", "--seed"])
