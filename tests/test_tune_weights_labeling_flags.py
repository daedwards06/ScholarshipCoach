from __future__ import annotations

from datetime import date

import pandas as pd

from scripts.tune_weights import ProfileCache, _run_config_once, generate_candidate_configs, parse_args
from src.eval.golden_students import GoldenStudent
from src.eval.relevance import RelevanceConfig
from src.rank.stage1_eligibility import StudentProfile


def _student() -> GoldenStudent:
    return GoldenStudent(
        student_id="synthetic",
        description="Synthetic profile for tune_weights labeling tests.",
        profile=StudentProfile(
            gpa=3.7,
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


def _profile_cache() -> ProfileCache:
    component_df = pd.DataFrame(
        [
            {
                "scholarship_id": "sim-only",
                "amount_max": 5000.0,
                "sponsor": "Synthetic Sponsor",
                "source": "synthetic",
                "text_sim": 0.95,
                "keyword_overlap": 0.0,
                "amount_utility": 0.1,
                "effort_penalty": 0.0,
                "urgency_boost": 0.0,
                "ev_proxy_norm": 0.0,
                "deadline": pd.NaT,
                "_deadline_sort": pd.NaT,
                "majors_allowed": ["History"],
                "states_allowed": ["NY"],
                "education_level": "Graduate",
            }
        ]
    )
    return ProfileCache(
        student=_student(),
        eligible_df=component_df.copy(),
        ineligible_df=pd.DataFrame(),
        component_df=component_df,
    )


def test_parse_args_accepts_labeling_flags() -> None:
    args = parse_args(
        [
            "--label-mode",
            "no_similarity",
            "--tfidf-threshold",
            "0.22",
            "--embed-threshold",
            "0.44",
            "--calibrate-thresholds",
        ]
    )

    assert args.label_mode == "no_similarity"
    assert args.tfidf_threshold == 0.22
    assert args.embed_threshold == 0.44
    assert args.calibrate_thresholds is True


def test_run_config_once_uses_label_mode_and_is_deterministic() -> None:
    cache = _profile_cache()
    config = generate_candidate_configs(max_configs=1)[0]

    hybrid = _run_config_once(
        [cache],
        config=config,
        k=1,
        use_win_model=False,
        similarity_mode="embeddings",
        relevance_config=RelevanceConfig(
            label_mode="hybrid",
            tfidf_threshold=0.12,
            embed_threshold=0.30,
        ),
    )
    no_similarity = _run_config_once(
        [cache],
        config=config,
        k=1,
        use_win_model=False,
        similarity_mode="embeddings",
        relevance_config=RelevanceConfig(
            label_mode="no_similarity",
            tfidf_threshold=0.12,
            embed_threshold=0.30,
        ),
    )
    repeat = _run_config_once(
        [cache],
        config=config,
        k=1,
        use_win_model=False,
        similarity_mode="embeddings",
        relevance_config=RelevanceConfig(
            label_mode="hybrid",
            tfidf_threshold=0.12,
            embed_threshold=0.30,
        ),
    )

    assert hybrid["relevance_labels"]["synthetic"] == [1]
    assert no_similarity["relevance_labels"]["synthetic"] == [0]
    assert hybrid["relevance_labels"] != no_similarity["relevance_labels"]
    assert repeat["relevance_labels"] == hybrid["relevance_labels"]
    assert repeat["ordered_ids"] == hybrid["ordered_ids"]
