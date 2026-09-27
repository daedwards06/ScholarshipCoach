from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.helpers import deadline_badge, friendly_date, parse_date

TODAY = date(2026, 10, 6)


def _badge(days: int | None, **kwargs: object):  # type: ignore[no-untyped-def]
    deadline = None if days is None else TODAY + timedelta(days=days)
    return deadline_badge(days, deadline=deadline, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "days,expected",
    [
        (-5, ("Overdue · was Oct 1", ":material/error:", "red")),
        (0, ("Due Tue, Oct 6 · today", ":material/schedule:", "orange")),
        (4, ("Due Sat, Oct 10 · 4 days", ":material/schedule:", "orange")),
        (7, ("Due Tue, Oct 13 · 7 days", ":material/schedule:", "orange")),
        (8, ("Due Oct 14 · 8 days", ":material/event:", "blue")),
        (30, ("Due Nov 5 · 30 days", ":material/event:", "blue")),
        (31, ("Due Nov 6", ":material/event:", "gray")),
        (None, ("Date not posted yet", ":material/help:", "yellow")),
    ],
)
def test_deadline_states(days: int | None, expected: tuple[str, str, str]) -> None:
    assert tuple(_badge(days)) == expected


def test_projected_deadline_is_gray() -> None:
    assert tuple(_badge(177, projected=True)) == (
        "Usually ~Apr 1",
        ":material/update:",
        "gray",
    )


def test_past_date_nobody_owes_is_passed_not_overdue() -> None:
    assert tuple(_badge(-5, owed=False)) == ("Passed", ":material/history:", "gray")


@pytest.mark.parametrize("days", [-3, 0, 1, 5, 7, 12, 30, 90])
def test_milestone_is_never_red_orange_or_urgent(days: int) -> None:
    badge = _badge(days, kind="milestone")
    assert badge.color not in ("red", "orange")
    assert "urgent" not in badge.label.lower()


def test_milestone_states() -> None:
    assert tuple(_badge(4, kind="milestone")) == (
        "Sat, Oct 10 · 4 days",
        ":material/flag:",
        "blue",
    )
    assert tuple(_badge(-3, kind="milestone")) == ("Passed", ":material/history:", "gray")


def test_badges_without_a_date_still_read() -> None:
    assert deadline_badge(0).label == "Due today"
    assert deadline_badge(3).label == "Due in 3 days"
    assert deadline_badge(12).label == "Due in 12 days"
    assert deadline_badge(-1).label == "Overdue"


def test_friendly_date_drops_the_year_only_inside_the_school_year() -> None:
    assert friendly_date(date(2026, 10, 10), TODAY) == "Sat, Oct 10"
    assert friendly_date(date(2027, 6, 30), TODAY) == "Wed, Jun 30"
    assert friendly_date(date(2027, 7, 1), TODAY) == "Jul 1, 2027"
    assert friendly_date(date(2026, 6, 30), TODAY) == "Jun 30, 2026"
    assert friendly_date(date(2026, 10, 10), TODAY, weekday=False) == "Oct 10"


def test_parse_date() -> None:
    assert parse_date("2026-10-10T00:00:00") == date(2026, 10, 10)
    assert parse_date("") is None
    assert parse_date(None) is None
