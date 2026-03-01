from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.golden_students import get_golden_students
from src.win_model.infer import load_model, predict_p_win
from src.win_model.synthetic import generate_synthetic_training_data
from src.win_model.train import train_win_model


def _tiny_snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scholarship_id": "s1",
                "source": "scholarship_america_live",
                "title": "California STEM Award",
                "sponsor": "STEM Fund",
                "description": "Support for computer science students in California.",
                "eligibility_text": "Computer Science undergraduate in CA.",
                "essay_prompt": "Describe your project.",
                "deadline": date(2026, 3, 15),
                "amount_min": 1000.0,
                "amount_max": 5000.0,
                "states_allowed": ["CA"],
                "majors_allowed": ["Computer Science"],
                "min_gpa": 3.0,
                "education_level": "Undergraduate",
                "essay_required": True,
            },
            {
                "scholarship_id": "s2",
                "source": "manual",
                "title": "Healthcare Service Grant",
                "sponsor": "Care Foundation",
                "description": "Nursing students serving rural communities.",
                "eligibility_text": "Open to undergraduate nursing majors.",
                "essay_prompt": None,
                "deadline": date(2026, 4, 20),
                "amount_min": 2000.0,
                "amount_max": 8000.0,
                "states_allowed": ["TX", "OK"],
                "majors_allowed": ["Nursing"],
                "min_gpa": 3.2,
                "education_level": "Undergraduate",
                "essay_required": False,
            },
            {
                "scholarship_id": "s3",
                "source": "manual",
                "title": "Graduate Climate Fellowship",
                "sponsor": "Green Lab",
                "description": "Environmental research support for graduate students.",
                "eligibility_text": "Graduate environmental science focus.",
                "essay_prompt": "Summarize your climate work.",
                "deadline": date(2026, 5, 30),
                "amount_min": 3000.0,
                "amount_max": 12000.0,
                "states_allowed": [],
                "majors_allowed": ["Environmental Science"],
                "min_gpa": 3.5,
                "education_level": "Graduate",
                "essay_required": True,
            },
        ]
    )


def test_train_and_infer_win_model_are_deterministic(tmp_path: Path) -> None:
    snapshot_df = _tiny_snapshot()
    students = get_golden_students()[:3]
    first_out = tmp_path / "first"
    second_out = tmp_path / "second"

    first_result = train_win_model(snapshot_df, students, first_out, seed=0)
    second_result = train_win_model(snapshot_df, students, second_out, seed=0)

    assert Path(first_result["model_path"]).exists()
    assert Path(second_result["model_path"]).exists()

    X_df, _, _ = generate_synthetic_training_data(snapshot_df, students, n_samples=64, seed=0)
    first_model = load_model(Path(first_result["model_path"]))
    second_model = load_model(Path(second_result["model_path"]))

    first_pred = predict_p_win(first_model, X_df)
    second_pred = predict_p_win(second_model, X_df)

    assert np.all((first_pred >= 0.0) & (first_pred <= 1.0))
    assert np.allclose(first_pred, second_pred)
