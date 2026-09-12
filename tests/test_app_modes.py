from __future__ import annotations

import pytest

from app import modes


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("student", "student"),
        ("Parent", "parent"),
        ("  OPERATOR ", "operator"),
        ("", "student"),
        (None, "student"),
        ("admin", "student"),
    ],
)
def test_normalize_mode_falls_back_to_student(raw: object, expected: str) -> None:
    assert modes.normalize_mode(raw) == expected


def test_available_modes_hides_operator_unless_enabled() -> None:
    assert modes.available_modes(False) == ("student", "parent")
    assert modes.available_modes(True) == ("student", "parent", "operator")


def test_student_sections_are_the_four_daily_surfaces_plus_find() -> None:
    sections = modes.sections_for_mode("student")
    assert sections == ("this_week", "find", "applications", "essays", "recommenders")


def test_parent_sections_extend_student_sections() -> None:
    student = modes.sections_for_mode("student")
    parent = modes.sections_for_mode("parent")
    assert parent[: len(student)] == student
    assert parent[len(student) :] == (
        "catalog_inbox",
        "timeline",
        "colleges_money",
        "outcomes",
        "settings",
    )


def test_operator_sees_the_same_sections_as_parent() -> None:
    assert modes.sections_for_mode("operator") == modes.sections_for_mode("parent")


@pytest.mark.parametrize("section", modes.PARENT_ONLY_SECTIONS)
def test_parent_only_sections_hidden_from_student(section: str) -> None:
    assert not modes.can_view("student", section)
    assert modes.can_view("parent", section)


@pytest.mark.parametrize("section", modes.STUDENT_SECTIONS)
def test_student_sections_visible_in_every_mode(section: str) -> None:
    assert modes.can_view("student", section)
    assert modes.can_view("parent", section)
    assert modes.can_view("operator", section)


def test_every_section_has_a_label() -> None:
    assert set(modes.SECTION_LABELS) == set(modes.PARENT_SECTIONS)


def test_parent_reads_essays_student_writes_them() -> None:
    assert modes.essays_read_only("parent")
    assert not modes.can_edit_essays("parent")
    for mode in ("student", "operator"):
        assert not modes.essays_read_only(mode)
        assert modes.can_edit_essays(mode)


def test_operator_tools_need_both_the_mode_and_the_setting() -> None:
    assert modes.show_operator_tools("operator", True)
    assert not modes.show_operator_tools("operator", False)
    assert not modes.show_operator_tools("parent", True)
    assert not modes.show_operator_tools("student", True)


@pytest.mark.parametrize(
    ("secrets", "expected"),
    [
        (None, None),
        ({}, None),
        ({"parent_pin": ""}, None),
        ({"parent_pin": "   "}, None),
        ({"parent_pin": " 4321 "}, "4321"),
        ({"parent_pin": 4321}, "4321"),
        ({"other": "4321"}, None),
    ],
)
def test_parent_pin_reads_optional_secret(secrets: dict | None, expected: str | None) -> None:
    assert modes.parent_pin(secrets) == expected


def test_pin_required_only_above_student_and_only_with_a_pin() -> None:
    assert modes.pin_required("parent", "4321")
    assert modes.pin_required("operator", "4321")
    assert not modes.pin_required("student", "4321")
    assert not modes.pin_required("parent", None)
    assert not modes.pin_required("parent", "")


def test_check_pin_accepts_anything_when_no_pin_is_configured() -> None:
    assert modes.check_pin(None, None)
    assert modes.check_pin("", "whatever")


def test_check_pin_compares_the_configured_pin() -> None:
    assert modes.check_pin("4321", "4321")
    assert modes.check_pin("4321", " 4321 ")
    assert not modes.check_pin("4321", "1234")
    assert not modes.check_pin("4321", "")
    assert not modes.check_pin("4321", None)


def test_resolve_mode_grants_parent_when_no_pin_is_set() -> None:
    assert modes.resolve_mode("parent", operator_enabled=False) == "parent"


def test_resolve_mode_withholds_operator_until_the_setting_is_on() -> None:
    assert modes.resolve_mode("operator", operator_enabled=False) == "student"
    assert modes.resolve_mode("operator", operator_enabled=True) == "operator"


def test_resolve_mode_locks_parent_behind_the_pin() -> None:
    assert modes.resolve_mode("parent", operator_enabled=False, pin="4321") == "student"
    granted = modes.resolve_mode("parent", operator_enabled=False, pin="4321", unlocked=True)
    assert granted == "parent"


def test_resolve_mode_never_locks_student_out() -> None:
    assert modes.resolve_mode("student", operator_enabled=False, pin="4321") == "student"


def test_resolve_section_defaults_to_the_first_section_of_the_mode() -> None:
    assert modes.resolve_section(None, "student") == "this_week"
    assert modes.resolve_section("nonsense", "parent") == "this_week"


def test_resolve_section_drops_a_section_the_mode_cannot_see() -> None:
    assert modes.resolve_section("settings", "parent") == "settings"
    assert modes.resolve_section("settings", "student") == "this_week"
