from __future__ import annotations

import pandas as pd

from src.rank.stage2_scoring import _compute_amount_utility


def test_log_amount_utility_reduces_extreme_outlier_dominance() -> None:
    df = pd.DataFrame(
        [
            {"amount_min": None, "amount_max": 1_000.0},
            {"amount_min": None, "amount_max": 1_000_000.0},
        ]
    )

    linear = _compute_amount_utility(df, mode="linear")
    log_scaled = _compute_amount_utility(df, mode="log")

    assert linear[1] == 1.0
    assert log_scaled[1] == 1.0
    assert log_scaled[0] > linear[0]
