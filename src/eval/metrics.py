"""Offline evaluation metrics for ranking quality.

Provides NDCG@k, Coverage@k, eligibility precision, amount distribution
statistics, and ranking stability checks used by the golden-student evaluation
harness.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import pandas as pd


def eligibility_precision(
    per_profile_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute the fraction of scholarships that pass Stage 1 eligibility across all profiles.

    Args:
        per_profile_results: Each dict must contain ``eligible_df`` and
            ``ineligible_df`` DataFrames as produced by the evaluation harness.

    Returns:
        Dict with ``eligible_count``, ``total_count``, ``eligibility_precision``
        (float in [0, 1]), and ``ineligible_reason_breakdown`` (reason → count).
    """
    total_count = 0
    eligible_count = 0
    reason_counter: Counter[str] = Counter()

    for result in per_profile_results:
        eligible_df: pd.DataFrame = result["eligible_df"]
        ineligible_df: pd.DataFrame = result["ineligible_df"]
        eligible_count += int(len(eligible_df))
        total_count += int(len(eligible_df) + len(ineligible_df))

        if "reasons" not in ineligible_df.columns or ineligible_df.empty:
            continue
        for reasons in ineligible_df["reasons"]:
            if not isinstance(reasons, list):
                continue
            for reason in reasons:
                if isinstance(reason, str) and reason:
                    reason_counter[reason] += 1

    precision = (eligible_count / total_count) if total_count > 0 else 0.0
    return {
        "eligible_count": eligible_count,
        "total_count": total_count,
        "eligibility_precision": precision,
        "ineligible_reason_breakdown": dict(sorted(reason_counter.items())),
    }


def coverage_at_k(
    per_profile_topk: dict[str, list[dict[str, Any]]],
    k: int,
) -> dict[str, Any]:
    """Compute the cross-profile diversity ratio of the top-k recommendations.

    This is NOT catalog coverage (recommended items / total catalog). It is
    ``unique recommended scholarships / total recommended slots`` across every
    profile: a diversity ratio measuring how much the top-k lists overlap. A
    value near 1.0 means each profile is served largely distinct scholarships;
    a low value means the same few scholarships dominate many profiles' lists.

    Args:
        per_profile_topk: Maps profile ID → list of ranked scholarship dicts.
        k: Cutoff rank.

    Returns:
        Dict with ``unique_recommended_count``, ``total_recommended``, and
        ``coverage_at_k`` (float in [0, 1]).
    """
    unique_ids: set[str] = set()
    total_recommended = 0
    for recs in per_profile_topk.values():
        for rec in recs[:k]:
            scholarship_id = rec.get("scholarship_id")
            if isinstance(scholarship_id, str) and scholarship_id:
                unique_ids.add(scholarship_id)
            total_recommended += 1

    return {
        "k": k,
        "unique_recommended_count": len(unique_ids),
        "total_recommended": total_recommended,
        "coverage_at_k": (len(unique_ids) / total_recommended) if total_recommended else 0.0,
    }


def amount_distribution_stats(
    per_profile_topk: dict[str, list[dict[str, Any]]],
    k: int,
) -> dict[str, Any]:
    """Compute descriptive statistics for award amounts in the top-k recommendations.

    Args:
        per_profile_topk: Maps profile ID → list of ranked scholarship dicts.
        k: Cutoff rank.

    Returns:
        Dict with ``count``, ``mean``, ``median``, and ``max`` dollar amounts.
    """
    values: list[float] = []
    for recs in per_profile_topk.values():
        for rec in recs[:k]:
            amount = rec.get("amount_max")
            if amount is None or pd.isna(amount):
                continue
            values.append(float(amount))

    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "max": 0.0}

    series = pd.Series(values, dtype="float64")
    return {
        "count": len(values),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "max": float(series.max()),
    }


def ranking_stability(
    run_one: dict[str, list[str]],
    run_two: dict[str, list[str]],
) -> dict[str, Any]:
    """Assert that two ranking runs produce identical ordered lists.

    Args:
        run_one: Maps profile ID → ordered list of scholarship IDs.
        run_two: Second run to compare against ``run_one``.

    Returns:
        Dict with ``is_stable`` (bool) and ``mismatches`` (empty list when stable).

    Raises:
        AssertionError: If any profile produces a different ordering between runs.
    """
    mismatches: list[dict[str, Any]] = []
    for profile_id in sorted(set(run_one) | set(run_two)):
        ids_one = run_one.get(profile_id, [])
        ids_two = run_two.get(profile_id, [])
        if ids_one != ids_two:
            mismatches.append(
                {
                    "profile_id": profile_id,
                    "run_one": ids_one,
                    "run_two": ids_two,
                }
            )

    is_stable = len(mismatches) == 0
    if not is_stable:
        raise AssertionError(f"Ranking stability check failed: {mismatches}")

    return {"is_stable": is_stable, "mismatches": mismatches}


def compute_ndcg_at_k(
    relevance_labels: dict[str, list[int]] | None,
    k: int,
) -> float | str:
    """Compute mean NDCG@k across all profiles.

    Args:
        relevance_labels: Maps profile ID → ordered list of integer relevance
            labels (0, 1, or 2) for the recommended scholarships.
        k: Cutoff rank.

    Returns:
        Mean NDCG@k as a float, or ``"N/A"`` if no labels are provided.
    """
    if not relevance_labels:
        return "N/A"

    ndcg_values: list[float] = []
    for labels in relevance_labels.values():
        if not labels:
            continue
        observed = labels[:k]
        ideal = sorted(labels, reverse=True)[:k]
        dcg = _dcg(observed)
        idcg = _dcg(ideal)
        ndcg_values.append((dcg / idcg) if idcg > 0 else 0.0)

    if not ndcg_values:
        return "N/A"
    return float(sum(ndcg_values) / len(ndcg_values))


def _dcg(labels: list[int]) -> float:
    total = 0.0
    for index, rel in enumerate(labels):
        gain = (2**max(rel, 0)) - 1
        discount = math.log2(index + 2.0)
        total += gain / discount
    return total
