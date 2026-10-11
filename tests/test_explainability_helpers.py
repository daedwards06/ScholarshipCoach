from __future__ import annotations

import pandas as pd
import pytest

from app.helpers import (
    badge_markdown,
    effort_to_text,
    explain_ranked_row,
    fit_reasons,
    markdown_safe,
    format_amount_range,
    needs_date_awards,
)


def test_explain_ranked_row_is_stable_and_prioritizes_strong_signals() -> None:
    row = pd.Series(
        {
            "tfidf_sim": 0.92,
            "amount_utility": 0.80,
            "keyword_overlap": 0.60,
            "urgency_boost": 0.20,
            "ev_proxy_norm": 0.40,
            "essay_required": False,
        }
    )

    assert explain_ranked_row(row) == [
        "Strong match to your goals/keywords",
        "High award amount",
        "High direct keyword overlap",
    ]


@pytest.mark.parametrize(
    "amount_min,amount_max,expected",
    [
        (None, None, "Amount not listed"),
        (None, 5000, "Up to $5,000"),
        (2500, None, "$2,500+"),
        (4000, 4000, "$4,000"),
        (1000, 5000, "$1,000 - $5,000"),
        (0, 0, "$0"),
    ],
)
def test_format_amount_range(
    amount_min: float | None, amount_max: float | None, expected: str
) -> None:
    assert format_amount_range(amount_min, amount_max) == expected


def test_explain_ranked_row_hides_expected_value_outside_operator_mode() -> None:
    row = pd.Series(
        {
            "tfidf_sim": 0.10,
            "amount_utility": 0.0,
            "keyword_overlap": 0.0,
            "urgency_boost": 0.0,
            "expected_value_norm": 0.99,
            "essay_required": True,
        }
    )

    assert explain_ranked_row(row) == ["Strong match to your goals/keywords", "1 essay"]
    assert explain_ranked_row(row, operator_mode=True) == [
        "Strong expected-value proxy",
        "Strong match to your goals/keywords",
        "1 essay",
    ]


def test_explain_ranked_row_adds_local_and_effort_lines() -> None:
    row = pd.Series(
        {
            "tfidf_sim": 0.92,
            "amount_utility": 0.80,
            "keyword_overlap": 0.60,
            "urgency_boost": 0.20,
            "essay_required": True,
            "trust": "verified_local",
            "requirements": {
                "essay_prompts": ["Why computer science?", "Describe a setback."],
                "recommendation_letters": 1,
            },
        }
    )

    assert explain_ranked_row(row) == [
        "Strong match to your goals/keywords",
        "High award amount",
        "High direct keyword overlap",
        "Local award, smaller applicant pool",
        "2 essays, 1 letter",
    ]


@pytest.mark.parametrize(
    "requirements,expected",
    [
        ({"essay_prompts": ["A"]}, "1 essay"),
        ({"essay_prompts": ["A", "B"], "recommendation_letters": 2}, "2 essays, 2 letters"),
        ({"recommendation_letters": 1, "transcript": True}, "1 letter, 1 extra item"),
        ({"essay": True, "fafsa": True, "interview": True}, "1 essay, 2 extra items"),
        ({"essay_prompts": [], "recommendation_letters": 0}, ""),
    ],
)
def test_effort_to_text_renders_counts(
    requirements: dict[str, object], expected: str
) -> None:
    assert effort_to_text(pd.Series({"requirements": requirements})) == expected


def test_needs_date_awards_are_the_undated_ones_biggest_first() -> None:
    df = pd.DataFrame(
        [
            {"scholarship_id": "small", "timeline_bucket": "needs_date", "amount_max": 500},
            {"scholarship_id": "dated", "timeline_bucket": "now", "amount_max": 9000},
            {"scholarship_id": "big", "timeline_bucket": "needs_date", "amount_max": 5000},
            {"scholarship_id": "unpriced", "timeline_bucket": "needs_date", "amount_max": None},
        ]
    )

    assert needs_date_awards(df)["scholarship_id"].tolist() == ["big", "small", "unpriced"]


def test_needs_date_awards_falls_back_to_the_minimum_amount() -> None:
    df = pd.DataFrame(
        [
            {"scholarship_id": "floor", "timeline_bucket": "needs_date", "amount_min": 2000},
            {"scholarship_id": "ceiling", "timeline_bucket": "needs_date", "amount_max": 3000},
        ]
    )

    assert needs_date_awards(df)["scholarship_id"].tolist() == ["ceiling", "floor"]


def test_needs_date_awards_handles_frames_without_the_column() -> None:
    assert needs_date_awards(pd.DataFrame([{"scholarship_id": "a"}])).empty
    assert needs_date_awards(None).empty


def test_deadline_reason_needs_the_deadline_within_two_weeks() -> None:
    # F9: a 40-day deadline still earns e^(-40/30) of urgency boost, and the
    # card used to call it "soon" next to a "Later" badge.
    far = pd.Series(
        {"tfidf_sim": 0.1, "urgency_boost": 0.26, "days_to_deadline": 40, "essay_required": True}
    )
    near = pd.Series(
        {"tfidf_sim": 0.1, "urgency_boost": 0.72, "days_to_deadline": 10, "essay_required": True}
    )

    assert "Due within two weeks" not in explain_ranked_row(far)
    assert explain_ranked_row(near)[0] == "Due within two weeks"


def test_deadline_reason_falls_back_to_the_boost_without_a_day_count() -> None:
    assert "Due within two weeks" not in explain_ranked_row(
        pd.Series({"urgency_boost": 0.26, "essay_required": True})
    )
    assert "Due within two weeks" in explain_ranked_row(
        pd.Series({"urgency_boost": 0.72, "essay_required": True})
    )


def test_fit_reasons_leave_local_and_effort_to_the_badges() -> None:
    row = pd.Series(
        {
            "text_sim": 0.8,
            "essay_required": True,
            "essay_prompts": ["Why CS?"],
            "local_award": True,
        }
    )

    assert fit_reasons(row) == ["Strong match to your goals/keywords"]
    assert explain_ranked_row(row)[: len(fit_reasons(row))] == fit_reasons(row)


def test_badge_markdown_is_one_inline_badge() -> None:
    assert (
        badge_markdown("Due Nov 5", ":material/event:", "gray")
        == ":gray-badge[:material/event: Due Nov 5]"
    )
    assert badge_markdown("a [b]", ":material/flag:", "blue").endswith("a (b)]")


def test_markdown_safe_keeps_dollar_amounts_out_of_latex() -> None:
    assert markdown_safe("$2,000 - $4,000") == r"\$2,000 - \$4,000"
