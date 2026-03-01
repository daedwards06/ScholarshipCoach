from __future__ import annotations

from datetime import date

import pandas as pd

from src.eval.golden_students import GoldenStudent
from src.eval.relevance import RelevanceConfig, proxy_relevance_label, proxy_relevance_labels
from src.rank.stage1_eligibility import StudentProfile


def _student() -> GoldenStudent:
    return GoldenStudent(
        student_id="synthetic",
        description="Synthetic profile for relevance tests.",
        profile=StudentProfile(
            gpa=3.5,
            state="CA",
            major="Computer Science",
            education_level="Undergraduate",
            citizenship="US",
            today=date(2026, 2, 28),
        ),
        interests=(),
        keywords=(),
        extracurriculars=(),
        goals="",
    )


def test_hybrid_mode_uses_mode_specific_thresholds() -> None:
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

    assert proxy_relevance_label(row, _student(), similarity_mode="tfidf", cfg=cfg) == 1
    assert proxy_relevance_label(row, _student(), similarity_mode="embeddings", cfg=cfg) == 0


def test_no_similarity_mode_ignores_similarity() -> None:
    student = _student()
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

    assert proxy_relevance_label(high_similarity_only, student, similarity_mode="embeddings", cfg=cfg) == 0
    assert proxy_relevance_label(keyword_only, student, similarity_mode="embeddings", cfg=cfg) == 1


def test_proxy_relevance_labels_are_deterministic_for_tiny_frame() -> None:
    student = _student()
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

    first = proxy_relevance_labels(frame, student, similarity_mode="tfidf", cfg=cfg)
    second = proxy_relevance_labels(frame, student, similarity_mode="tfidf", cfg=cfg)

    assert first == [2, 1, 0]
    assert second == first
