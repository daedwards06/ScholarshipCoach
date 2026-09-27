from __future__ import annotations

from pathlib import Path

import pytest

from app.modes import PARENT_SECTIONS, STUDENT_SECTIONS
from scripts import screenshot_app


def test_plan_covers_every_section_of_both_modes_at_both_widths_and_themes() -> None:
    shots = screenshot_app.plan_captures(screenshot_app.selected_themes("both"))

    per_theme_and_width = len(STUDENT_SECTIONS) + len(PARENT_SECTIONS)
    assert len(shots) == per_theme_and_width * 2 * 2
    assert {shot.width for shot in shots} == {400, 1366}
    assert {shot.theme for shot in shots} == {"light", "dark"}
    assert {s.section for s in shots if s.mode == "parent"} == set(PARENT_SECTIONS)
    assert {s.section for s in shots if s.mode == "student"} == set(STUDENT_SECTIONS)
    assert len({shot.filename for shot in shots}) == len(shots)


def test_plan_filters_sections() -> None:
    shots = screenshot_app.plan_captures(("dark",), {"timeline"})

    assert [shot.filename for shot in shots] == [
        "400_parent_timeline_dark.png",
        "1366_parent_timeline_dark.png",
    ]


def test_section_url_carries_mode_and_section() -> None:
    url = screenshot_app.section_url("http://127.0.0.1:8501", "parent", "colleges_money")

    assert url == "http://127.0.0.1:8501/?shot_mode=parent&shot_section=colleges_money"


def test_scratch_db_is_the_default_and_output_is_outside_the_repo() -> None:
    args = screenshot_app.parse_args([])

    assert args.scratch_db is True
    assert screenshot_app.ROOT_DIR not in Path(args.out).resolve().parents


def test_seed_refuses_the_live_database() -> None:
    with pytest.raises(SystemExit):
        screenshot_app.parse_args(["--live-db", "--seed"])
