from __future__ import annotations

import pandas as pd

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


def test_format_amount_range_handles_missing_and_equal_values() -> None:
    assert format_amount_range(None, None) == "Unknown"
    assert format_amount_range(None, 5000) == "Up to $5,000"
    assert format_amount_range(2500, None) == "$2,500+"
    assert format_amount_range(4000, 4000) == "$4,000"
