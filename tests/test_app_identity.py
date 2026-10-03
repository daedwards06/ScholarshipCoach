from __future__ import annotations

from typing import Any

import pytest

from app import identity, modes
from app.identity import Identity

FAMILY = {"Kid@Example.com": "student", "parent@example.com": "Parent"}


def _headers(login: str | None = None, name: str | None = None) -> dict[str, str]:
    headers = {"Host": "coach.example.ts.net"}
    if login is not None:
        headers["Tailscale-User-Login"] = login
    if name is not None:
        headers["Tailscale-User-Name"] = name
    return headers


def test_mapped_student_login_becomes_a_student_identity() -> None:
    who = identity.role_for_headers(_headers("kid@example.com", "Kid"), FAMILY)
    assert who == Identity(login="kid@example.com", name="Kid", role="student")


def test_logins_and_roles_are_compared_case_folded() -> None:
    who = identity.role_for_headers(_headers("PARENT@Example.COM", "A Parent"), FAMILY)
    assert who is not None
    assert who.role == "parent"
    assert who.login == "parent@example.com"


def test_header_names_are_matched_case_insensitively() -> None:
    headers = {"tailscale-user-login": "kid@example.com"}
    who = identity.role_for_headers(headers, FAMILY)
    assert who is not None and who.role == "student"


def test_name_falls_back_to_the_login() -> None:
    who = identity.role_for_headers(_headers("kid@example.com"), FAMILY)
    assert who is not None and who.name == "kid@example.com"


@pytest.mark.parametrize(
    "headers",
    [None, {}, _headers(), _headers(""), _headers("   "), _headers(None, "Kid")],
)
def test_missing_login_header_means_no_identity(headers: dict[str, str] | None) -> None:
    assert identity.role_for_headers(headers, FAMILY) is None


def test_unmapped_login_falls_back_to_the_pin() -> None:
    assert identity.role_for_headers(_headers("guest@example.com"), FAMILY) is None


@pytest.mark.parametrize("family_map", [None, {}])
def test_missing_family_table_means_no_identity(family_map: dict[str, Any] | None) -> None:
    assert identity.role_for_headers(_headers("kid@example.com"), family_map) is None


def test_unknown_role_in_the_family_table_is_ignored() -> None:
    assert identity.role_for_headers(_headers("x@example.com"), {"x@example.com": "admin"}) is None


def test_family_map_from_secrets_without_a_secrets_file_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoSecrets:
        def get(self, key: str) -> Any:
            raise FileNotFoundError("no secrets.toml")

    monkeypatch.setattr(identity.st, "secrets", NoSecrets())
    assert identity.family_map_from_secrets() == {}


def test_family_map_from_secrets_reads_the_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(identity.st, "secrets", {"parent_pin": "4321", "family": FAMILY})
    assert identity.family_map_from_secrets() == FAMILY
    monkeypatch.setattr(identity.st, "secrets", {"parent_pin": "4321"})
    assert identity.family_map_from_secrets() == {}


def test_header_presence_reports_names_not_values() -> None:
    presence = identity.header_presence(_headers("kid@example.com"))
    assert presence == {"Tailscale-User-Login": True, "Tailscale-User-Name": False}


def test_activity_counts_for_the_login_else_the_role() -> None:
    who = Identity(login="kid@example.com", name="Kid", role="student")
    assert identity.activity_key(who, "student") == "kid@example.com"
    assert identity.activity_key(None, "student") == "student"
    assert identity.activity_key(None, "parent") == "parent"
    assert identity.activity_key(None, "operator") == "parent"


STUDENT = Identity(login="kid@example.com", name="Kid", role="student")
PARENT = Identity(login="parent@example.com", name="A Parent", role="parent")


@pytest.mark.parametrize("requested", ["student", "parent", "operator"])
def test_student_login_cannot_reach_parent_even_with_the_pin_unlocked(requested: str) -> None:
    granted = modes.resolve_mode(
        requested, operator_enabled=True, pin="4321", unlocked=True, identity=STUDENT
    )
    assert granted == "student"


def test_parent_login_skips_the_pin() -> None:
    granted = modes.resolve_mode("parent", operator_enabled=False, pin="4321", identity=PARENT)
    assert granted == "parent"


def test_parent_login_can_still_preview_student_mode() -> None:
    assert modes.resolve_mode("student", operator_enabled=False, identity=PARENT) == "student"


def test_parent_login_reaches_operator_only_when_enabled() -> None:
    assert modes.resolve_mode("operator", operator_enabled=True, pin="4321", identity=PARENT) == (
        "operator"
    )
    assert modes.resolve_mode("operator", operator_enabled=False, identity=PARENT) == "parent"


def test_no_identity_keeps_the_pin_path() -> None:
    assert modes.resolve_mode("parent", operator_enabled=False, pin="4321", identity=None) == (
        "student"
    )
