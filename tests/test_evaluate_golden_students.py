from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.evaluate_golden_students import (
    _cross_label_check,
    _load_weight_overrides,
    _markdown_report,
)
from src.eval.golden_students import get_golden_students
from src.eval.relevance import RelevanceConfig


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
        "text_sim": 0.7,
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
            "proxy_relevance": {
                "label_mode": "hybrid",
                "text_similarity_thresholds": {
                    "tfidf": 0.12,
                    "embeddings": 0.30,
                },
                "active_text_similarity_threshold": 0.12,
                "calibration": None,
            },
        },
        students=[],
        per_profile_topk={},
    )

    assert f"- Weights file: `{Path('data/processed/best_weights.json')}`" in report
    assert "- Label mode: `hybrid`" in report
    assert "- TF-IDF relevance threshold: 0.1200" in report
    assert "- Embeddings relevance threshold: 0.3000" in report


def test_cross_label_check_scores_both_label_modes() -> None:
    student = next(s for s in get_golden_students() if s.student_id == "golden_ca_cs_ug_us")
    # Row 0: similarity-only relevance (label 1 under hybrid, 0 under no_similarity),
    # ranked ABOVE Row 1: a strict + keyword match (label 2 under both). The mis-order
    # makes NDCG diverge between the two heuristics.
    reranked_df = pd.DataFrame(
        [
            {
                "keyword_overlap": 0.0,
                "text_sim": 0.9,
                "majors_allowed": ["Nursing"],
                "states_allowed": ["CA"],
                "education_level": "Undergraduate",
            },
            {
                "keyword_overlap": 1.0,
                "text_sim": 0.9,
                "majors_allowed": ["Computer Science"],
                "states_allowed": ["CA"],
                "education_level": "Undergraduate",
            },
        ]
    )
    per_profile_results = [{"student_id": student.student_id, "k": 2, "reranked_df": reranked_df}]

    result = _cross_label_check(
        [student],
        per_profile_results,
        similarity_mode="tfidf",
        relevance_config=RelevanceConfig(),
        k=2,
        enabled=True,
    )

    assert result is not None
    ndcg = result["ndcg_by_label_mode"]
    assert set(ndcg) == {"hybrid", "no_similarity"}
    assert ndcg["hybrid"] > ndcg["no_similarity"]


def test_cross_label_check_disabled_returns_none() -> None:
    result = _cross_label_check(
        [],
        [],
        similarity_mode="tfidf",
        relevance_config=RelevanceConfig(),
        k=0,
        enabled=False,
    )
    assert result is None


def test_markdown_report_renders_cross_label_section() -> None:
    report = _markdown_report(
        snapshot_path=Path("snapshot.parquet"),
        snapshot_count=1,
        generated_at="2026-02-28T00:00:00Z",
        weights_path=None,
        metrics={
            "eligibility": {
                "eligibility_precision": 1.0,
                "eligible_count": 1,
                "total_count": 1,
                "ineligible_reason_breakdown": {},
            },
            "coverage_at_k": {"k": 1, "coverage_at_k": 1.0, "unique_recommended_count": 1},
            "amount_distribution_topk": {"mean": 0.0, "median": 0.0, "max": 0.0},
            "ranking_stability": {"is_stable": True},
            "ndcg_at_k": {"k": 1, "value": 1.0},
            "proxy_relevance": {
                "label_mode": "hybrid",
                "text_similarity_thresholds": {"tfidf": 0.12, "embeddings": 0.30},
                "active_text_similarity_threshold": 0.12,
                "calibration": None,
            },
        },
        students=[],
        per_profile_topk={},
        cross_label={
            "k": 10,
            "similarity_mode": "tfidf",
            "ndcg_by_label_mode": {"hybrid": 0.61, "no_similarity": 0.54},
        },
    )

    assert "## Cross-Label NDCG Check" in report
    assert "| hybrid | 0.6100 |" in report
    assert "| no_similarity | 0.5400 |" in report
