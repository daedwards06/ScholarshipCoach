from __future__ import annotations

from datetime import date

import pandas as pd

from app.sections.find import (
    OPEN_NOW,
    WHEN_OPTIONS,
    award_badges,
    filters_to_relax,
    when_buckets,
    when_label,
)

TODAY = date(2026, 10, 10)


def test_open_now_ranks_the_grade_to_check_awards_too() -> None:
    assert WHEN_OPTIONS[0] == OPEN_NOW
    assert when_buckets(OPEN_NOW) == ("now", "check_grade")
    assert when_buckets("senior_year") == ("senior_year",)
    assert when_buckets("all") is None
    assert "check_grade" not in WHEN_OPTIONS and "now" not in WHEN_OPTIONS


def test_no_when_option_is_labelled_apply_now() -> None:
    assert "Apply now" not in {when_label(choice) for choice in WHEN_OPTIONS}


def test_unscoped_award_card_asks_to_confirm_grade_and_never_says_apply_now() -> None:
    row = pd.Series(
        {
            "deadline": "2026-11-01",
            "timeline_bucket": "check_grade",
            "unverified_axes": ["grade_level"],
            "essay_required": False,
        }
    )

    badges = " ".join(award_badges(row, TODAY))

    assert ":yellow-badge[:material/warning: Confirm you meet: grade level]" in badges
    assert "Apply now" not in badges


def test_later_award_card_says_when_it_opens() -> None:
    row = pd.Series(
        {"deadline": "2027-03-01", "timeline_bucket": "senior_year", "essay_required": False}
    )

    assert ":gray-badge[:material/schedule: A later school year]" in award_badges(row, TODAY)


def test_empty_results_name_the_filters_to_relax() -> None:
    assert filters_to_relax(
        search_term=" robotics ",
        when_choice=OPEN_NOW,
        deadline_within_days=0,
        min_amount=1000.0,
        no_essay_only=True,
    ) == ["search", "when she can apply", "minimum amount", "no-essay only"]
    assert (
        filters_to_relax(
            search_term="",
            when_choice="all",
            deadline_within_days=0,
            min_amount=0.0,
            no_essay_only=False,
        )
        == []
    )
