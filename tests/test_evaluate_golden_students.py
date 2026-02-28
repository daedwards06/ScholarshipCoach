from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_golden_students import _load_weight_overrides, _markdown_report


def test_load_weight_overrides_reads_best_weights_payload(tmp_path: Path) -> None:
    weights_path = tmp_path / "best_weights.json"
    weights_path.write_text(
        json.dumps(
            {
                "amount_utility_mode": "log",
                "stage2_weights": {
                    "tfidf": 0.7,
                    "amount": 0.05,
                    "keyword": 0.15,
                    "effort": 0.1,
                },
                "stage3_weights": {
                    "stage2": 0.95,
                    "urgency": 0.05,
                    "ev": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )

    stage2_weights, stage3_weights, amount_utility_mode, resolved_path = _load_weight_overrides(weights_path)

    assert amount_utility_mode == "log"
    assert resolved_path == weights_path
    assert stage2_weights is not None
    assert stage2_weights.to_dict() == {
        "tfidf": 0.7,
        "amount": 0.05,
        "keyword": 0.15,
        "effort": 0.1,
    }
    assert stage3_weights is not None
    assert stage3_weights.to_dict() == {
        "stage2": 0.95,
        "urgency": 0.05,
        "ev": 0.0,
    }


def test_load_weight_overrides_rejects_invalid_amount_utility_mode(tmp_path: Path) -> None:
    weights_path = tmp_path / "bad_weights.json"
    weights_path.write_text(
        json.dumps(
            {
                "amount_utility_mode": "invalid",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported amount_utility_mode"):
        _load_weight_overrides(weights_path)


def test_markdown_report_records_weight_source() -> None:
    report = _markdown_report(
        snapshot_path=Path("snapshot.parquet"),
        snapshot_count=1,
        generated_at="2026-02-28T00:00:00Z",
        weights_path=Path("data/processed/best_weights.json"),
        metrics={
            "eligibility": {
                "eligibility_precision": 1.0,
                "eligible_count": 1,
                "total_count": 1,
                "ineligible_reason_breakdown": {},
            },
            "coverage_at_k": {
                "k": 1,
                "coverage_at_k": 1.0,
                "unique_recommended_count": 1,
            },
            "amount_distribution_topk": {
                "mean": 1000.0,
                "median": 1000.0,
                "max": 1000.0,
            },
            "ranking_stability": {
                "is_stable": True,
            },
            "ndcg_at_k": {
                "k": 1,
                "value": 1.0,
            },
        },
        students=[],
        per_profile_topk={},
    )

    assert f"- Weights file: `{Path('data/processed/best_weights.json')}`" in report
