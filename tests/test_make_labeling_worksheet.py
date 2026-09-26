from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from scripts.make_labeling_worksheet import (
    DEFAULT_OUTPUT_DIR,
    PRIVATE_OUTPUT_DIR,
    WORKSHEET_COLUMNS,
    _format_amount,
    build_worksheet,
    find_stored_student,
    get_stored_student,
    resolve_output_dir,
)
from src.profile import store as profile_store
from src.eval.golden_students import GoldenStudent
from src.rank.stage1_eligibility import StudentProfile


def _student() -> GoldenStudent:
    return GoldenStudent(
        student_id="test_profile",
        description="Test profile",
        profile=StudentProfile(
            gpa=3.5,
            state="NC",
            major="Computer Science",
            education_level="high school",
            citizenship="US",
            today=date(2026, 2, 22),
        ),
        interests=("programming",),
        keywords=("stem",),
        extracurriculars=("robotics club",),
        goals="Study CS.",
    )


def _snapshot() -> pd.DataFrame:
    # Two eligible rows (no constraints) and one ineligible (GPA too high).
    return pd.DataFrame(
        [
            {
                "scholarship_id": "sch_a",
                "title": "Alpha STEM Award",
                "sponsor": "Alpha",
                "source_url": "https://example.org/alpha",
                "description": "  Supports STEM   students   pursuing engineering.  ",
                "deadline": pd.Timestamp("2027-01-01"),
                "amount_min": 1000.0,
                "amount_max": 5000.0,
                "min_gpa": None,
                "states_allowed": [],
                "majors_allowed": [],
                "education_level": None,
                "citizenship": None,
            },
            {
                "scholarship_id": "sch_b",
                "title": "Beta Grant",
                "sponsor": "Beta",
                "description": None,
                "deadline": pd.NaT,
                "amount_min": None,
                "amount_max": 2500.0,
                "min_gpa": None,
                "states_allowed": [],
                "majors_allowed": [],
                "education_level": None,
                "citizenship": None,
            },
            {
                "scholarship_id": "sch_c",
                "title": "Gamma Prize",
                "sponsor": "Gamma",
                "description": "High bar.",
                "deadline": pd.Timestamp("2027-01-01"),
                "amount_min": 2000.0,
                "amount_max": 2000.0,
                "min_gpa": 3.9,
                "states_allowed": [],
                "majors_allowed": [],
                "education_level": None,
                "citizenship": None,
            },
        ]
    )


def test_build_worksheet_has_expected_columns_and_empty_labels() -> None:
    worksheet = build_worksheet(_snapshot(), _student(), n=10, seed=0)

    assert list(worksheet.columns) == WORKSHEET_COLUMNS
    # Only the two eligible rows are sampled; the GPA-blocked row is excluded.
    assert len(worksheet) == 2
    assert set(worksheet["scholarship_id"]) == {"sch_a", "sch_b"}
    assert (worksheet["label"] == "").all()
    assert (worksheet["profile_id"] == "test_profile").all()


def test_build_worksheet_caps_at_eligible_count_and_is_deterministic() -> None:
    snapshot = _snapshot()
    student = _student()

    first = build_worksheet(snapshot, student, n=50, seed=7)
    second = build_worksheet(snapshot, student, n=50, seed=7)

    # n exceeds the eligible-set size, so the worksheet caps at the 2 eligible rows.
    assert len(first) == 2
    pd.testing.assert_frame_equal(first, second)


def test_build_worksheet_formats_amount_and_snippet() -> None:
    worksheet = build_worksheet(_snapshot(), _student(), n=10, seed=0)
    by_id = worksheet.set_index("scholarship_id")

    assert by_id.loc["sch_a", "amount"] == "$1,000-$5,000"
    assert by_id.loc["sch_b", "amount"] == "$2,500"
    # Whitespace is collapsed in the description snippet.
    assert by_id.loc["sch_a", "description_snippet"] == "Supports STEM students pursuing engineering."
    assert by_id.loc["sch_b", "description_snippet"] == ""
    assert by_id.loc["sch_a", "deadline"] == "2027-01-01"
    assert by_id.loc["sch_b", "deadline"] == ""
    # source_url is carried through when present and blank-safe when absent.
    assert by_id.loc["sch_a", "source_url"] == "https://example.org/alpha"
    assert by_id.loc["sch_b", "source_url"] == ""


def test_format_amount_handles_unknown_and_single_value() -> None:
    assert _format_amount(pd.Series({"amount_min": None, "amount_max": None})) == "Unknown"
    assert _format_amount(pd.Series({"amount_min": None, "amount_max": 2500.0})) == "$2,500"
    assert _format_amount(pd.Series({"amount_min": 3000.0, "amount_max": 3000.0})) == "$3,000"


def test_build_worksheet_empty_when_no_eligible_rows() -> None:
    student = _student()
    # Force ineligibility by requiring a GPA above the profile's.
    snapshot = _snapshot()
    snapshot["min_gpa"] = 3.99

    worksheet = build_worksheet(snapshot, student, n=10, seed=0)

    assert worksheet.empty
    assert list(worksheet.columns) == WORKSHEET_COLUMNS


def test_build_worksheet_top_ranked_returns_ranked_prefix() -> None:
    worksheet = build_worksheet(_snapshot(), _student(), n=1, top_ranked=True)

    assert len(worksheet) == 1
    # The ranked prefix is the pipeline's own order, not a random draw.
    assert worksheet.iloc[0]["scholarship_id"] in {"sch_a", "sch_b"}
    assert (worksheet["label"] == "").all()


def test_get_stored_student_falls_back_to_demo_profile() -> None:
    subject, is_demo = get_stored_student("no_such_student")

    assert is_demo is True
    assert subject.student_id
    assert isinstance(subject.profile, StudentProfile)
    assert "major" in subject.as_stage2_profile()


def test_stored_student_subject_works_as_worksheet_subject() -> None:
    subject, _ = get_stored_student("no_such_student")
    permissive = _snapshot()
    permissive["min_gpa"] = None
    permissive["education_level"] = None

    worksheet = build_worksheet(permissive, subject, n=10, seed=0)

    assert list(worksheet.columns) == WORKSHEET_COLUMNS
    assert (worksheet["profile_id"] == subject.student_id).all()


def test_stored_student_worksheet_defaults_to_the_private_dir() -> None:
    assert resolve_output_dir(None, stored_student=True) == PRIVATE_OUTPUT_DIR
    assert resolve_output_dir(None, stored_student=False) == DEFAULT_OUTPUT_DIR
    assert PRIVATE_OUTPUT_DIR.parts[-2:] == ("private", "eval")


def test_explicit_output_dir_wins_over_the_default(tmp_path: Path) -> None:
    assert resolve_output_dir(tmp_path, stored_student=True) == tmp_path


def test_find_stored_student_reads_a_private_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(profile_store, "STUDENTS_DIR", tmp_path)
    profile = profile_store.load_demo_profile()
    profile["student_id"] = "student_1"
    (tmp_path / "student_1.json").write_text(json.dumps(profile), encoding="utf-8")

    subject = find_stored_student("student_1")

    assert subject is not None
    assert subject.student_id == "student_1"
    assert isinstance(subject.profile, StudentProfile)


def test_find_stored_student_never_substitutes_the_demo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(profile_store, "STUDENTS_DIR", tmp_path)
    demo_id = str(profile_store.load_demo_profile()["student_id"])

    assert find_stored_student("student_1") is None
    assert find_stored_student("not a valid id!") is None
    demo = find_stored_student(demo_id)
    assert demo is not None and demo.student_id == demo_id
