from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.rank.stage1_eligibility import StudentProfile
from src.win_model.features import FEATURE_COLUMNS, build_pair_features


def test_build_pair_features_returns_stable_expected_values() -> None:
    profile = StudentProfile(
        gpa=3.8,
        state="CA",
        major="Computer Science",
        education_level="Undergraduate",
        today=date(2026, 2, 22),
    )
    scholarship_row = pd.Series(
        {
            "source": "scholarship_america_live",
            "deadline": date(2026, 3, 4),
            "amount_max": 5000.0,
            "min_gpa": 3.2,
            "majors_allowed": ["Computer Science", "Engineering"],
            "states_allowed": ["CA", "NV"],
            "education_level": "Undergraduate",
            "essay_required": True,
        }
    )
    stage2_row = pd.Series({"keyword_overlap": 0.5, "text_sim": 0.75})

    features = build_pair_features(
        profile,
        scholarship_row,
        stage2_row=stage2_row,
        today=date(2026, 2, 22),
    )

    assert tuple(features.keys()) == FEATURE_COLUMNS
    assert features["gpa"] == 3.8
    assert features["min_gpa"] == 3.2
    assert features["gpa_above_min"] == pytest.approx(0.6)
    assert features["keyword_overlap"] == 0.5
    assert features["text_sim"] == 0.75
    assert features["days_to_deadline"] == 10.0
    assert features["amount_value"] == 5000.0
    assert features["major_match"] == 1.0
    assert features["state_match"] == 1.0
    assert features["education_level_match"] == 1.0
    assert features["essay_required"] == 1.0
    assert features["source_is_scholarship_america"] == 1.0
