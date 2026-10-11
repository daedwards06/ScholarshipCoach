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


def test_student_nav_is_top_pages_plus_more() -> None:
    assert modes.pages_for_mode("student") == {
        "": ("todos", "essays"),
        "More": ("what_if", "recommenders", "profile"),
    }


def test_parent_nav_is_her_path_money_and_family_tools() -> None:
    assert modes.pages_for_mode("parent") == {
        "Her path": ("todos", "essays", "what_if", "recommenders"),
        "Money": ("find", "applications", "colleges_money", "timeline", "outcomes"),
        "Family tools": ("catalog_inbox", "settings", "profile"),
    }


def test_operator_sees_the_same_pages_as_parent() -> None:
    assert modes.pages_for_mode("operator") == modes.pages_for_mode("parent")


@pytest.mark.parametrize("section", ["find", "applications"])
def test_scholarship_tracker_is_a_parent_page(section: str) -> None:
    assert not modes.can_view("student", section)
    assert modes.can_view("parent", section)
    assert modes.resolve_section(section, "student") == "todos"


def test_tracker_labels_in_parent() -> None:
    assert modes.SECTION_LABELS["find"] == "Find Scholarships"
    assert modes.SECTION_LABELS["applications"] == "Applications"


def test_every_student_page_is_also_in_parent() -> None:
    assert set(modes.nav_sections("student")) <= set(modes.nav_sections("parent"))


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
    assert set(modes.SECTION_LABELS) == set(modes.nav_sections("parent"))


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
    assert modes.resolve_section(None, "student") == "todos"
    assert modes.resolve_section("nonsense", "parent") == "todos"


def test_resolve_section_drops_a_section_the_mode_cannot_see() -> None:
    assert modes.resolve_section("settings", "parent") == "settings"
    assert modes.resolve_section("settings", "student") == "todos"


@pytest.mark.parametrize("mode", ["student", "parent", "operator"])
def test_every_page_has_a_label_icon_and_unique_url_path(mode: modes.Mode) -> None:
    sections = modes.nav_sections(mode)
    assert all(sections for sections in modes.pages_for_mode(mode).values())
    assert len(set(sections)) == len(sections)
    paths = [modes.section_url_path(section) for section in sections]
    assert len(set(paths)) == len(paths)
    for section, path in zip(sections, paths, strict=True):
        assert modes.SECTION_LABELS[section]
        assert modes.SECTION_ICONS[section].startswith(":material/")
        assert path and "/" not in path and "_" not in path


def test_every_nav_page_has_a_purpose_caption() -> None:
    assert set(modes.SECTION_CAPTIONS) == set(modes.nav_sections("parent"))


def test_this_week_is_retired_for_todos() -> None:
    assert "this_week" not in modes.SECTION_LABELS
    assert modes.section_url_path("todos") == "to-dos"
    assert "this-week" in modes.RETIRED_URL_PATHS


@pytest.mark.parametrize("section", modes.PARENT_ONLY_SECTIONS)
def test_parent_only_pages_absent_from_student_nav(section: str) -> None:
    assert section not in modes.nav_sections("student")
    assert section in modes.nav_sections("parent")


def test_profile_page_is_in_every_mode() -> None:
    for mode in ("student", "parent", "operator"):
        assert modes.can_view(mode, "profile")


def test_pin_still_gates_the_parent_pages() -> None:
    granted = modes.resolve_mode("parent", operator_enabled=False, pin="4321")
    assert granted == "student"
    assert "settings" not in modes.nav_sections(granted)


def test_layout_is_centered_for_student_and_wide_otherwise() -> None:
    assert modes.layout_for_mode("student") == "centered"
    assert modes.layout_for_mode("parent") == "wide"
    assert modes.layout_for_mode("operator") == "wide"


def test_every_nav_section_has_a_renderer() -> None:
    from app.main import SECTION_RENDERERS

    assert set(SECTION_RENDERERS) == set(modes.SECTION_LABELS)
