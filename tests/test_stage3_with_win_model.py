from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.rank.stage1_eligibility import StudentProfile
from src.rank.stage3_rerank import rerank_stage3
from src.rank.weights import Stage3Weights


class _FixedProbabilityModel:
    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        p = self._probabilities[: len(features_df)]
        return np.column_stack([1.0 - p, p])


def test_rerank_stage3_uses_expected_value_when_win_model_enabled() -> None:
    df = pd.DataFrame(
        [
            {
                "scholarship_id": "a-id",
                "stage2_score": 0.5,
                "text_sim": 0.5,
                "keyword_overlap": 0.1,
                "deadline": date(2026, 3, 1),
                "amount_min": 5000.0,
                "amount_max": 5000.0,
                "essay_required": False,
                "majors_allowed": [],
                "states_allowed": [],
                "education_level": "Undergraduate",
                "source": "manual",
            },
            {
                "scholarship_id": "b-id",
                "stage2_score": 0.5,
                "text_sim": 0.5,
                "keyword_overlap": 0.1,
                "deadline": date(2026, 3, 1),
                "amount_min": 5000.0,
                "amount_max": 5000.0,
                "essay_required": False,
                "majors_allowed": [],
                "states_allowed": [],
                "education_level": "Undergraduate",
                "source": "manual",
            },
        ]
    )
    profile = StudentProfile(
        gpa=3.5,
        state="CA",
        major="Computer Science",
        education_level="Undergraduate",
        today=date(2026, 2, 22),
    )
    weights = Stage3Weights(stage2=0.0, urgency=0.0, ev=1.0)

    baseline = rerank_stage3(df, today=profile.today, weights=weights)
    with_win_model = rerank_stage3(
        df,
        today=profile.today,
        profile=profile,
        weights=weights,
        use_win_model=True,
        win_model=_FixedProbabilityModel([0.1, 0.9]),
    )

    assert baseline["scholarship_id"].tolist() == ["a-id", "b-id"]
    assert with_win_model["scholarship_id"].tolist() == ["b-id", "a-id"]
    assert "p_win" in with_win_model.columns
    assert "expected_value_norm" in with_win_model.columns
