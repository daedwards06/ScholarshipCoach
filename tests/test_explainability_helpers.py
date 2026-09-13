from __future__ import annotations

import pandas as pd
import pytest

from app.helpers import effort_to_text, explain_ranked_row, format_amount_range


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
        (None, None, "Unknown"),
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
