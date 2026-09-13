from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.rank.stage3_rerank import effort_counts, is_local_award, rerank_stage3
from src.rank.weights import Stage3Weights

_TODAY = date(2026, 2, 22)


def _two_row_df(*, sooner_days: int, later_days: int, stage2_score: float = 0.80) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scholarship_id": "later",
                "stage2_score": stage2_score,
                "deadline": _TODAY + timedelta(days=later_days),
                "amount_min": 1000.0,
                "amount_max": 5000.0,
                "essay_required": False,
            },
            {
                "scholarship_id": "sooner",
                "stage2_score": stage2_score,
                "deadline": _TODAY + timedelta(days=sooner_days),
                "amount_min": 1000.0,
                "amount_max": 5000.0,
                "essay_required": False,
            },
        ]
    )


@pytest.mark.parametrize(
    "sooner_days,later_days",
    [
        (6, 52),    # ~1 week vs ~7 weeks
        (14, 90),   # 2 weeks vs ~3 months
        (1, 30),    # tomorrow vs 1 month
        (30, 180),  # 1 month vs 6 months
    ],
)
def test_rerank_stage3_boosts_closer_deadline(sooner_days: int, later_days: int) -> None:
    df = _two_row_df(sooner_days=sooner_days, later_days=later_days)
    reranked_df = rerank_stage3(df, today=_TODAY)

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

    reranked_df = rerank_stage3(df, today=_TODAY).set_index("scholarship_id")

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

    reranked_df = rerank_stage3(df, today=_TODAY)

    assert reranked_df["scholarship_id"].tolist() == ["a-id", "b-id", "c-id"]


def test_past_deadline_scores_no_urgency_instead_of_maximum() -> None:
    """A recurring award whose deadline has passed must not outrank a live one.

    Stage 1 keeps recurring awards past their listed deadline so the timeline can
    bucket them, so Stage 3 sees negative days-to-deadline and must score them as
    gone rather than as maximally urgent.
    """
    df = pd.DataFrame(
        [
            {
                "scholarship_id": "stale-recurring",
                "stage2_score": 0.80,
                "deadline": _TODAY - timedelta(days=240),
                "amount_min": 1000.0,
                "amount_max": 5000.0,
                "essay_required": False,
            },
            {
                "scholarship_id": "live-soon",
                "stage2_score": 0.80,
                "deadline": _TODAY + timedelta(days=19),
                "amount_min": 1000.0,
                "amount_max": 5000.0,
                "essay_required": False,
            },
        ]
    )

    reranked_df = rerank_stage3(df, today=_TODAY)
    urgency_by_id = dict(
        zip(reranked_df["scholarship_id"], reranked_df["urgency_boost"], strict=True)
    )

    assert urgency_by_id["stale-recurring"] == 0.0
    assert urgency_by_id["live-soon"] > 0.0
    assert reranked_df["scholarship_id"].tolist()[0] == "live-soon"


def _requirements_row(
    scholarship_id: str, requirements: dict[str, object] | None, **extra: object
) -> dict[str, object]:
    row: dict[str, object] = {
        "scholarship_id": scholarship_id,
        "stage2_score": 0.80,
        "deadline": _TODAY + timedelta(days=30),
        "amount_min": 1000.0,
        "amount_max": 5000.0,
        "essay_required": False,
        "requirements": requirements,
    }
    row.update(extra)
    return row


def test_effort_counts_use_requirements_when_present() -> None:
    row = pd.Series(
        {
            "essay_required": False,
            "requirements": {
                "essay": True,
                "essay_prompts": ["Why engineering?", "Describe a setback."],
                "recommendation_letters": 2,
                "transcript": True,
                "fafsa": None,
                "video_or_portfolio": False,
                "interview": True,
            },
        }
    )

    assert effort_counts(row) == {"essays": 2, "letters": 2, "extras": 2}


@pytest.mark.parametrize(
    "essay_required,expected",
    [(True, {"essays": 1, "letters": 0, "extras": 0}), (False, {"essays": 0, "letters": 0, "extras": 0})],
)
def test_effort_counts_fall_back_to_essay_required(
    essay_required: bool, expected: dict[str, int]
) -> None:
    row = pd.Series({"essay_required": essay_required, "requirements": None})

    assert effort_counts(row) == expected


def test_effort_cost_grows_with_requirement_counts_and_preserves_boolean_fallback() -> None:
    df = pd.DataFrame(
        [
            _requirements_row("no-requirements", None),
            _requirements_row("essay-boolean-only", None, essay_required=True),
            _requirements_row("one-essay", {"essay_prompts": ["Why engineering?"]}),
            _requirements_row(
                "two-essays-two-letters",
                {"essay_prompts": ["A", "B"], "recommendation_letters": 2},
            ),
        ]
    )

    reranked_df = rerank_stage3(df, today=_TODAY).set_index("scholarship_id")

    assert reranked_df.loc["no-requirements", "effort_cost"] == pytest.approx(1.0)
    assert reranked_df.loc["essay-boolean-only", "effort_cost"] == pytest.approx(1.5)
    assert reranked_df.loc["one-essay", "effort_cost"] == pytest.approx(1.5)
    assert reranked_df.loc["two-essays-two-letters", "effort_cost"] == pytest.approx(2.5)


@pytest.mark.parametrize(
    "extra,expected",
    [
        ({"trust": "verified_local"}, True),
        ({"counties_allowed": ["Guilford"]}, True),
        ({"states_allowed": ["NC"]}, True),
        ({"trust": "aggregator", "counties_allowed": [], "states_allowed": []}, False),
        ({}, False),
    ],
)
def test_is_local_award_flags_small_pools(extra: dict[str, object], expected: bool) -> None:
    assert is_local_award(pd.Series(extra)) is expected


def test_local_boost_lifts_a_restricted_award_over_an_identical_national_one() -> None:
    df = pd.DataFrame(
        [
            _requirements_row("national", None, states_allowed=[], trust="aggregator"),
            _requirements_row("local", None, states_allowed=["NC"], trust="verified_local"),
        ]
    )

    reranked_df = rerank_stage3(df, today=_TODAY).set_index("scholarship_id")

    assert reranked_df.index.tolist()[0] == "local"
    assert bool(reranked_df.loc["local", "local_award"]) is True
    assert bool(reranked_df.loc["national", "local_award"]) is False
    assert reranked_df.loc["local", "final_score"] - reranked_df.loc[
        "national", "final_score"
    ] == pytest.approx(Stage3Weights.baseline().local_boost)


def test_local_boost_of_zero_leaves_final_score_unchanged() -> None:
    df = pd.DataFrame(
        [
            _requirements_row("national", None),
            _requirements_row("local", None, trust="verified_local"),
        ]
    )
    weights = Stage3Weights(stage2=0.80, urgency=0.15, ev=0.05, local_boost=0.0)

    reranked_df = rerank_stage3(df, today=_TODAY, weights=weights).set_index("scholarship_id")

    assert reranked_df.loc["local", "final_score"] == pytest.approx(
        reranked_df.loc["national", "final_score"]
    )
