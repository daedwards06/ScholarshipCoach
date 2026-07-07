"""Direct, hand-computed unit tests for src/eval/metrics.py.

These pin the exact arithmetic behind the project's headline numbers (NDCG@k,
Coverage@k, eligibility precision, amount stats) and lock in the post-cleanup
API contract: compute_ndcg_at_k returns float | None (never "N/A"), and
ranking_stability returns a dict and never raises.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.eval.metrics import (
    amount_distribution_stats,
    compute_ndcg_at_k,
    coverage_at_k,
    eligibility_precision,
    ranking_stability,
)

# DCG helper values used to hand-derive the NDCG expectations below.
_LOG2_3 = math.log2(3.0)  # discount for rank position 2 (index 1)


def _reasons_frame(reason_lists: list[list[str]]) -> pd.DataFrame:
    return pd.DataFrame({"reasons": reason_lists}) if reason_lists else pd.DataFrame(
        {"reasons": pd.Series([], dtype="object")}
    )


# --------------------------------------------------------------------------- #
# compute_ndcg_at_k
# --------------------------------------------------------------------------- #
def test_ndcg_perfect_ordering_is_one() -> None:
    # Labels already in ideal order -> DCG == IDCG -> NDCG == 1.0.
    assert compute_ndcg_at_k({"p": [2, 1, 0]}, k=3) == pytest.approx(1.0)


def test_ndcg_imperfect_ordering_matches_hand_computed_dcg() -> None:
    # observed [0, 2, 1]: DCG = 0/1 + 3/log2(3) + 1/log2(4) = 3/log2(3) + 0.5
    # ideal    [2, 1, 0]: IDCG = 3/1 + 1/log2(3) + 0 = 3 + 1/log2(3)
    dcg = 3.0 / _LOG2_3 + 0.5
    idcg = 3.0 + 1.0 / _LOG2_3
    assert compute_ndcg_at_k({"p": [0, 2, 1]}, k=3) == pytest.approx(dcg / idcg)


def test_ndcg_all_zero_labels_is_zero() -> None:
    # IDCG == 0 -> the profile contributes 0.0 (not skipped, not None).
    assert compute_ndcg_at_k({"p": [0, 0, 0]}, k=3) == pytest.approx(0.0)


def test_ndcg_k_greater_than_label_count_does_not_error() -> None:
    # k exceeds the list length: truncation is a no-op, ordering is ideal.
    assert compute_ndcg_at_k({"p": [2, 1]}, k=5) == pytest.approx(1.0)


def test_ndcg_k_smaller_than_labels_truncates() -> None:
    # observed@2 [2, 0]: DCG = 3. ideal@2 [2, 1]: IDCG = 3 + 1/log2(3).
    idcg = 3.0 + 1.0 / _LOG2_3
    assert compute_ndcg_at_k({"p": [2, 0, 1]}, k=2) == pytest.approx(3.0 / idcg)


def test_ndcg_averages_across_profiles() -> None:
    perfect = 1.0
    imperfect = (3.0 / _LOG2_3 + 0.5) / (3.0 + 1.0 / _LOG2_3)
    result = compute_ndcg_at_k({"a": [2, 1, 0], "b": [0, 2, 1]}, k=3)
    assert result == pytest.approx((perfect + imperfect) / 2.0)


def test_ndcg_returns_none_when_no_labels() -> None:
    assert compute_ndcg_at_k(None, k=3) is None
    assert compute_ndcg_at_k({}, k=3) is None


def test_ndcg_returns_none_when_all_label_lists_empty() -> None:
    assert compute_ndcg_at_k({"a": [], "b": []}, k=3) is None


# --------------------------------------------------------------------------- #
# coverage_at_k
# --------------------------------------------------------------------------- #
def test_coverage_counts_unique_ids_across_profiles() -> None:
    topk = {
        "p1": [{"scholarship_id": "a"}, {"scholarship_id": "b"}],
        "p2": [{"scholarship_id": "a"}, {"scholarship_id": "c"}],
    }
    result = coverage_at_k(topk, k=2)
    assert result["unique_recommended_count"] == 3
    assert result["total_recommended"] == 4
    assert result["coverage_at_k"] == pytest.approx(0.75)


def test_coverage_respects_k_cutoff() -> None:
    topk = {
        "p1": [{"scholarship_id": "a"}, {"scholarship_id": "b"}],
        "p2": [{"scholarship_id": "a"}, {"scholarship_id": "c"}],
    }
    result = coverage_at_k(topk, k=1)
    assert result["unique_recommended_count"] == 1
    assert result["total_recommended"] == 2
    assert result["coverage_at_k"] == pytest.approx(0.5)


def test_coverage_counts_missing_id_in_total_only() -> None:
    topk = {"p1": [{"scholarship_id": "a"}, {"scholarship_id": ""}, {}]}
    result = coverage_at_k(topk, k=3)
    assert result["unique_recommended_count"] == 1
    assert result["total_recommended"] == 3
    assert result["coverage_at_k"] == pytest.approx(1.0 / 3.0)


def test_coverage_empty_is_zero() -> None:
    assert coverage_at_k({}, k=5)["coverage_at_k"] == 0.0


# --------------------------------------------------------------------------- #
# eligibility_precision
# --------------------------------------------------------------------------- #
def test_eligibility_precision_and_reason_breakdown() -> None:
    per_profile_results = [
        {
            "eligible_df": pd.DataFrame({"scholarship_id": ["a", "b"]}),
            "ineligible_df": _reasons_frame([["MIN_GPA"]]),
        },
        {
            "eligible_df": pd.DataFrame({"scholarship_id": ["c"]}),
            "ineligible_df": _reasons_frame(
                [["MIN_GPA"], ["STATE_NOT_ALLOWED", "MAJOR_NOT_ALLOWED"]]
            ),
        },
    ]
    result = eligibility_precision(per_profile_results)
    assert result["eligible_count"] == 3
    assert result["total_count"] == 6
    assert result["eligibility_precision"] == pytest.approx(0.5)
    assert result["ineligible_reason_breakdown"] == {
        "MAJOR_NOT_ALLOWED": 1,
        "MIN_GPA": 2,
        "STATE_NOT_ALLOWED": 1,
    }


def test_eligibility_precision_ignores_non_list_reasons() -> None:
    per_profile_results = [
        {
            "eligible_df": pd.DataFrame({"scholarship_id": ["a"]}),
            "ineligible_df": _reasons_frame([["MIN_GPA"], "not-a-list", [""], [None]]),
        }
    ]
    result = eligibility_precision(per_profile_results)
    assert result["ineligible_reason_breakdown"] == {"MIN_GPA": 1}


def test_eligibility_precision_zero_when_no_rows() -> None:
    per_profile_results = [
        {
            "eligible_df": pd.DataFrame({"scholarship_id": []}),
            "ineligible_df": _reasons_frame([]),
        }
    ]
    result = eligibility_precision(per_profile_results)
    assert result["total_count"] == 0
    assert result["eligibility_precision"] == 0.0


# --------------------------------------------------------------------------- #
# amount_distribution_stats
# --------------------------------------------------------------------------- #
def test_amount_stats_skip_nan_and_none() -> None:
    topk = {
        "p1": [
            {"amount_max": 1000.0},
            {"amount_max": float("nan")},
            {"amount_max": None},
            {"amount_max": 3000.0},
        ]
    }
    result = amount_distribution_stats(topk, k=10)
    assert result["count"] == 2
    assert result["mean"] == pytest.approx(2000.0)
    assert result["median"] == pytest.approx(2000.0)
    assert result["max"] == pytest.approx(3000.0)


def test_amount_stats_empty_returns_zeros() -> None:
    result = amount_distribution_stats({"p1": [{"amount_max": None}]}, k=10)
    assert result == {"count": 0, "mean": 0.0, "median": 0.0, "max": 0.0}


# --------------------------------------------------------------------------- #
# ranking_stability
# --------------------------------------------------------------------------- #
def test_ranking_stability_identical_runs() -> None:
    run = {"p1": ["a", "b"], "p2": ["c"]}
    result = ranking_stability(run, dict(run))
    assert result == {"is_stable": True, "mismatches": []}


def test_ranking_stability_reports_mismatches_without_raising() -> None:
    run_one = {"p1": ["a", "b"], "p2": ["c"]}
    run_two = {"p1": ["b", "a"], "p2": ["c"]}
    result = ranking_stability(run_one, run_two)
    assert result["is_stable"] is False
    assert result["mismatches"] == [
        {"profile_id": "p1", "run_one": ["a", "b"], "run_two": ["b", "a"]}
    ]
