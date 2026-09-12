from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.eval.golden_students import get_golden_students
from src.profile.grade_levels import (
    GRADE_LABELS,
    grade_label_to_levels,
    infer_graduation_year,
    levels_to_grade_label,
)
from src.profile.store import (
    DEFAULT_STUDENT_ID,
    DEMO_PROFILE_PATH,
    default_profile,
    list_students,
    load_demo_profile,
    load_profile,
    load_profile_or_demo,
    normalize_profile,
    save_profile,
    to_stage1_profile,
    to_stage2_profile,
)


def _filled_profile() -> dict:
    profile = default_profile()
    profile.update(
        {
            "student_id": "student_1",
            "name": "Test Student",
            "gpa": 3.25,
            "state": "NC",
            "county": "Guilford",
            "high_school": "Northside High School",
            "major": "Computer Science",
            "education_level": "undergraduate",
            "grade_level": "college_2",
            "graduation_year": 2029,
            "citizenship": "US",
            "financial_need": True,
            "first_gen": False,
            "gender": "female",
            "heritage": ["Cherokee"],
            "military_family": None,
            "disability": None,
            "religion": "Methodist",
            "parent_employers": ["Duke Energy"],
            "memberships": ["4-H"],
            "service_hours": 90,
            "sat": 1280,
            "act": 27,
            "intended_colleges": ["NC State University"],
            "essay_ready": True,
            "extracurriculars": ["robotics team"],
            "profile_keywords": ["robotics", "ai"],
            "goals": "Build assistive technology.",
        }
    )
    return profile


def test_profile_round_trip(tmp_path: Path) -> None:
    original = _filled_profile()
    written = save_profile(original, students_dir=tmp_path)

    assert written == tmp_path / "student_1.json"
    loaded = load_profile("student_1", students_dir=tmp_path)
    assert loaded == normalize_profile(original)


def test_save_profile_creates_missing_directory(tmp_path: Path) -> None:
    students_dir = tmp_path / "private" / "students"
    save_profile(_filled_profile(), students_dir=students_dir)
    assert (students_dir / "student_1.json").exists()


def test_list_students_is_sorted_and_empty_when_absent(tmp_path: Path) -> None:
    assert list_students(students_dir=tmp_path / "missing") == []

    save_profile(_filled_profile(), student_id="zoe", students_dir=tmp_path)
    save_profile(_filled_profile(), student_id="alex", students_dir=tmp_path)
    assert list_students(students_dir=tmp_path) == ["alex", "zoe"]


def test_load_profile_returns_none_when_absent(tmp_path: Path) -> None:
    assert load_profile("nobody", students_dir=tmp_path) is None


@pytest.mark.parametrize("student_id", ["../escape", "has space", "a" * 65])
def test_save_profile_rejects_unsafe_student_id(tmp_path: Path, student_id: str) -> None:
    with pytest.raises(ValueError, match="Invalid student_id"):
        save_profile(_filled_profile(), student_id=student_id, students_dir=tmp_path)


def test_save_profile_falls_back_to_the_profiles_own_id(tmp_path: Path) -> None:
    profile = _filled_profile()
    profile["student_id"] = "student_2"
    assert save_profile(profile, student_id="", students_dir=tmp_path).stem == "student_2"


def test_demo_profile_is_committed_and_valid() -> None:
    assert DEMO_PROFILE_PATH.exists()
    demo = load_demo_profile()
    assert demo["state"] == "NC"
    assert demo["education_level"] == "high school"
    assert demo["grade_level"] == "12"
    assert demo["extracurriculars"]


def test_demo_fallback_when_no_private_profile(tmp_path: Path) -> None:
    profile, is_demo = load_profile_or_demo(students_dir=tmp_path)
    assert is_demo is True
    assert profile == load_demo_profile()

    save_profile(_filled_profile(), students_dir=tmp_path)
    profile, is_demo = load_profile_or_demo(students_dir=tmp_path)
    assert is_demo is False
    assert profile["name"] == "Test Student"


def test_demo_profile_file_parses_as_json() -> None:
    payload = json.loads(DEMO_PROFILE_PATH.read_text(encoding="utf-8"))
    assert payload["student_id"] == "student_demo"


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("", (None, None)),
        ("High School Freshman", ("high school", "9")),
        ("High School Senior", ("high school", "12")),
        ("College Freshman", ("undergraduate", "college_1")),
        ("College Senior", ("undergraduate", "college_4")),
        ("Not A Grade", (None, None)),
    ],
)
def test_grade_label_mapping(label: str, expected: tuple[str | None, str | None]) -> None:
    assert grade_label_to_levels(label) == expected


def test_every_grade_label_maps_to_stage1_vocabulary() -> None:
    for label in GRADE_LABELS:
        education_level, _ = grade_label_to_levels(label)
        assert education_level in {None, "high school", "undergraduate"}


def test_levels_round_trip_to_label() -> None:
    for label in GRADE_LABELS:
        education_level, grade_level = grade_label_to_levels(label)
        assert levels_to_grade_label(education_level, grade_level) == label


def test_legacy_education_level_labels_are_migrated() -> None:
    profile = normalize_profile({"education_level": "High School Senior"})
    assert profile["education_level"] == "high school"
    assert profile["grade_level"] == "12"

    legacy_college = normalize_profile({"education_level": "Sophomore"})
    assert legacy_college["education_level"] == "undergraduate"
    assert legacy_college["grade_level"] == "college_2"


def test_normalize_profile_coerces_loose_values() -> None:
    profile = normalize_profile(
        {
            "gpa": "3.5",
            "heritage": "Cherokee, Hispanic",
            "sat": "1300",
            "financial_need": "yes",
            "first_gen": "no",
            "gender": "  ",
            "unknown_future_field": 7,
        }
    )
    assert profile["gpa"] == 3.5
    assert profile["heritage"] == ["Cherokee", "Hispanic"]
    assert profile["sat"] == 1300
    assert profile["financial_need"] is True
    assert profile["first_gen"] is False
    assert profile["gender"] is None
    assert profile["unknown_future_field"] == 7


def test_default_profile_leaves_identity_axes_unanswered() -> None:
    profile = default_profile()
    for axis in ("financial_need", "first_gen", "gender", "military_family", "disability"):
        assert profile[axis] is None


def test_to_stage1_profile_carries_new_axes() -> None:
    stage1 = to_stage1_profile(_filled_profile(), today=date(2026, 9, 12))

    assert stage1.gpa == 3.25
    assert stage1.state == "NC"
    assert stage1.today == date(2026, 9, 12)
    assert stage1.county == "Guilford"
    assert stage1.grade_level == "college_2"
    assert stage1.graduation_year == 2029
    assert stage1.financial_need is True
    assert stage1.first_gen is False
    assert stage1.heritage == ["Cherokee"]
    assert stage1.memberships == ["4-H"]
    assert stage1.sat == 1280
    assert stage1.act == 27
    assert stage1.intended_colleges == ["NC State University"]
    assert stage1.essay_ready is True


def test_to_stage2_profile_passes_extracurriculars() -> None:
    stage2 = to_stage2_profile(_filled_profile())
    assert stage2["extracurriculars"] == ["robotics team"]
    assert stage2["keywords"] == ["robotics", "ai"]
    assert stage2["goals"] == "Build assistive technology."


@pytest.mark.parametrize(
    ("grade_level", "today", "expected"),
    [
        ("12", date(2026, 9, 12), 2027),
        ("12", date(2026, 3, 1), 2026),
        ("10", date(2026, 9, 12), 2029),
        ("college_1", date(2026, 9, 12), 2030),
        (None, date(2026, 9, 12), None),
    ],
)
def test_infer_graduation_year(grade_level: str | None, today: date, expected: int | None) -> None:
    assert infer_graduation_year(grade_level, today) == expected


def test_golden_students_unchanged_by_new_profile_fields() -> None:
    students = get_golden_students()
    assert students

    for student in students:
        profile = student.profile
        assert profile.education_level is not None
        # Every axis added in Task 1.2 is optional and unanswered for fixtures.
        assert profile.grade_level is None
        assert profile.county is None
        assert profile.financial_need is None
        assert profile.heritage == []
        assert profile.intended_colleges == []
        assert profile.essay_ready is False
        assert student.as_stage2_profile()["extracurriculars"] == list(student.extracurriculars)


def test_golden_student_profiles_do_not_share_mutable_defaults() -> None:
    first, second = get_golden_students()[:2]
    first.profile.heritage.append("mutated")
    assert second.profile.heritage == []
    first.profile.heritage.clear()


def test_default_student_id_is_the_app_slot() -> None:
    assert DEFAULT_STUDENT_ID == "student_1"
