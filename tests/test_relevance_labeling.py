from __future__ import annotations

import pandas as pd
import pytest

from src.eval.golden_students import GoldenStudent
from src.eval.relevance import RelevanceConfig, proxy_relevance_label, proxy_relevance_labels


def _open_row(*, keyword_overlap: float = 0.0, text_sim: float = 0.0) -> pd.Series:
    """Scholarship open to all (no field restrictions) with controllable similarity signals."""
    return pd.Series(
        {
            "majors_allowed": [],
            "states_allowed": [],
            "education_level": None,
            "keyword_overlap": keyword_overlap,
            "text_sim": text_sim,
        }
    )


def test_hybrid_mode_uses_mode_specific_thresholds(sample_golden_student: GoldenStudent) -> None:
    row = pd.Series(
        {
            "majors_allowed": ["Computer Science"],
            "states_allowed": ["CA"],
            "education_level": "Undergraduate",
            "keyword_overlap": 0.0,
            "text_sim": 0.2,
        }
    )
    cfg = RelevanceConfig(label_mode="hybrid", tfidf_threshold=0.12, embed_threshold=0.30)

    assert proxy_relevance_label(row, sample_golden_student, similarity_mode="tfidf", cfg=cfg) == 1
    assert proxy_relevance_label(row, sample_golden_student, similarity_mode="embeddings", cfg=cfg) == 0


def test_no_similarity_mode_ignores_similarity(sample_golden_student: GoldenStudent) -> None:
    cfg = RelevanceConfig(label_mode="no_similarity", tfidf_threshold=0.01, embed_threshold=0.01)

    high_similarity_only = pd.Series(
        {
            "majors_allowed": ["History"],
            "states_allowed": ["NY"],
            "education_level": "Graduate",
            "keyword_overlap": 0.0,
            "text_sim": 0.99,
        }
    )
    keyword_only = pd.Series(
        {
            "majors_allowed": ["History"],
            "states_allowed": ["NY"],
            "education_level": "Graduate",
            "keyword_overlap": 2.0,
            "text_sim": 0.0,
        }
    )

    assert proxy_relevance_label(high_similarity_only, sample_golden_student, similarity_mode="embeddings", cfg=cfg) == 0
    assert proxy_relevance_label(keyword_only, sample_golden_student, similarity_mode="embeddings", cfg=cfg) == 1


@pytest.mark.parametrize(
    "text_sim,expected_label",
    [
        (0.11, 0),   # just below tfidf_threshold=0.12
        (0.12, 1),   # exactly at threshold
        (0.13, 1),   # just above threshold
        (0.00, 0),   # well below threshold
        (0.50, 1),   # well above threshold
    ],
)
def test_tfidf_threshold_boundary(
    sample_golden_student: GoldenStudent,
    text_sim: float,
    expected_label: int,
) -> None:
    row = _open_row(keyword_overlap=0.0, text_sim=text_sim)
    cfg = RelevanceConfig(label_mode="hybrid", tfidf_threshold=0.12, embed_threshold=0.30)

    assert proxy_relevance_label(row, sample_golden_student, similarity_mode="tfidf", cfg=cfg) == expected_label


def test_label2_requires_positive_major_match_by_default(
    sample_golden_student: GoldenStudent,
) -> None:
    # Scholarship open to all majors (no restriction) with keyword overlap.
    row = _open_row(keyword_overlap=1.0)

    # Default: an unrestricted scholarship cannot earn label 2 on keyword overlap
    # alone; it caps at 1.
    default_cfg = RelevanceConfig()
    assert proxy_relevance_label(row, sample_golden_student, similarity_mode="tfidf", cfg=default_cfg) == 1

    # Escape hatch restores the older vacuous-match behavior (label 2).
    legacy_cfg = RelevanceConfig(require_major_match_for_label2=False)
    assert proxy_relevance_label(row, sample_golden_student, similarity_mode="tfidf", cfg=legacy_cfg) == 2


def test_label2_positive_major_match_still_scores_two(
    sample_golden_student: GoldenStudent,
) -> None:
    # Scholarship that explicitly targets the student's major (no state/edu limit).
    row = pd.Series(
        {
            "majors_allowed": ["Computer Science"],
            "states_allowed": [],
            "education_level": None,
            "keyword_overlap": 1.0,
            "text_sim": 0.0,
        }
    )

    assert proxy_relevance_label(row, sample_golden_student, similarity_mode="tfidf", cfg=RelevanceConfig()) == 2


def test_proxy_relevance_labels_are_deterministic_for_tiny_frame(
    sample_golden_student: GoldenStudent,
) -> None:
    cfg = RelevanceConfig(label_mode="hybrid", tfidf_threshold=0.12, embed_threshold=0.30)
    frame = pd.DataFrame(
        [
            {
                "majors_allowed": ["Computer Science"],
                "states_allowed": ["CA"],
                "education_level": "Undergraduate",
                "keyword_overlap": 1.0,
                "text_sim": 0.05,
            },
            {
                "majors_allowed": ["History"],
                "states_allowed": ["NY"],
                "education_level": "Graduate",
                "keyword_overlap": 0.0,
                "text_sim": 0.20,
            },
            {
                "majors_allowed": ["History"],
                "states_allowed": ["NY"],
                "education_level": "Graduate",
                "keyword_overlap": 0.0,
                "text_sim": 0.05,
            },
        ]
    )

    first = proxy_relevance_labels(frame, sample_golden_student, similarity_mode="tfidf", cfg=cfg)
    second = proxy_relevance_labels(frame, sample_golden_student, similarity_mode="tfidf", cfg=cfg)

    assert first == [2, 1, 0]
    assert second == first
