from __future__ import annotations

from datetime import date

import pandas as pd

from src.rank.stage3_rerank import rerank_stage3


def test_rerank_stage3_boosts_closer_deadline_when_otherwise_equal() -> None:
    df = pd.DataFrame(
        [
            {
                "scholarship_id": "later",
                "stage2_score": 0.80,
                "deadline": date(2026, 4, 15),
                "amount_min": 1000.0,
                "amount_max": 5000.0,
                "essay_required": False,
            },
            {
                "scholarship_id": "sooner",
                "stage2_score": 0.80,
                "deadline": date(2026, 2, 28),
                "amount_min": 1000.0,
                "amount_max": 5000.0,
                "essay_required": False,
            },
        ]
    )

    reranked_df = rerank_stage3(df, today=date(2026, 2, 22))

    assert reranked_df["scholarship_id"].tolist()[0] == "sooner"
    assert reranked_df.loc[0, "urgency_boost"] > reranked_df.loc[1, "urgency_boost"]
    assert reranked_df.loc[0, "days_to_deadline"] < reranked_df.loc[1, "days_to_deadline"]


def test_rerank_stage3_higher_amount_has_higher_ev_proxy() -> None:
    df = pd.DataFrame(
        [
            {
                "scholarship_id": "lower-amount",
                "stage2_score": 0.60,
                "deadline": date(2026, 3, 10),
                "amount_min": 1000.0,
                "amount_max": 3000.0,
                "essay_required": False,
            },
            {
                "scholarship_id": "higher-amount",
                "stage2_score": 0.60,
                "deadline": date(2026, 3, 10),
                "amount_min": 1000.0,
                "amount_max": 7000.0,
                "essay_required": False,
            },
        ]
    )

    reranked_df = rerank_stage3(df, today=date(2026, 2, 22)).set_index("scholarship_id")

    assert reranked_df.loc["higher-amount", "ev_proxy"] > reranked_df.loc["lower-amount", "ev_proxy"]
    assert reranked_df.loc["higher-amount", "ev_proxy_norm"] > reranked_df.loc[
        "lower-amount", "ev_proxy_norm"
    ]


def test_rerank_stage3_uses_deterministic_tie_breaking() -> None:
    df = pd.DataFrame(
        [
            {
                "scholarship_id": "c-id",
                "stage2_score": 0.75,
                "deadline": date(2026, 3, 15),
                "amount_min": 2000.0,
                "amount_max": 2000.0,
                "essay_required": False,
            },
            {
                "scholarship_id": "a-id",
                "stage2_score": 0.75,
                "deadline": date(2026, 3, 1),
                "amount_min": 2000.0,
                "amount_max": 2000.0,
                "essay_required": False,
            },
            {
                "scholarship_id": "b-id",
                "stage2_score": 0.75,
                "deadline": date(2026, 3, 1),
                "amount_min": 2000.0,
                "amount_max": 2000.0,
                "essay_required": False,
            },
        ]
    )

    reranked_df = rerank_stage3(df, today=date(2026, 2, 22))

    assert reranked_df["scholarship_id"].tolist() == ["a-id", "b-id", "c-id"]
