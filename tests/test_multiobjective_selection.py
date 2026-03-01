from __future__ import annotations

from scripts.tune_weights import (
    _pareto_front,
    _pareto_knee_point,
    _resolve_objective_weights,
    _with_blended_scores,
)


def _result(config_id: str, ndcg: float, coverage: float, ev_norm: float) -> dict[str, object]:
    return {
        "config_id": config_id,
        "metrics": {
            "ndcg_at_k": ndcg,
            "coverage_at_k": coverage,
            "avg_expected_value_norm_topk": ev_norm,
        },
    }


def test_blended_objective_picks_highest_score() -> None:
    results = [
        _result("a", 0.70, 0.40, 0.20),
        _result("b", 0.60, 0.90, 0.90),
        _result("c", 0.80, 0.30, 0.10),
    ]

    weights = _resolve_objective_weights(
        "ndcg=0.7,coverage=0.2,ev=0.1",
        use_win_model=True,
    )
    assert weights is not None

    _with_blended_scores(results, objective_weights=weights)

    best = sorted(results, key=lambda result: -float(result["objective_score"]))[0]
    assert best["config_id"] == "b"


def test_pareto_front_excludes_dominated_configs_and_knee_is_deterministic() -> None:
    results = [
        _result("dominated", 0.50, 0.50, 0.50),
        _result("high-ndcg", 0.90, 0.40, 0.40),
        _result("balanced", 0.80, 0.80, 0.80),
        _result("high-coverage", 0.40, 0.90, 0.40),
    ]

    front = _pareto_front(results, metric_names=["ndcg", "coverage", "ev"])

    assert [result["config_id"] for result in front] == [
        "high-ndcg",
        "balanced",
        "high-coverage",
    ]

    knee = _pareto_knee_point(front, metric_names=["ndcg", "coverage", "ev"])
    assert knee["config_id"] == "balanced"
