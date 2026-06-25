from __future__ import annotations

import pandas as pd
import pytest

from app.helpers import explain_ranked_row, format_amount_range


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
