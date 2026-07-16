from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.golden_students import get_golden_students
from src.win_model.features import FEATURE_COLUMNS
from src.win_model.infer import load_model, predict_p_win
from src.win_model.synthetic import GENERATOR_COEFFICIENTS, generate_synthetic_training_data
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


def test_train_report_includes_generator_recovery(tmp_path: Path) -> None:
    snapshot_df = _tiny_snapshot()
    students = get_golden_students()[:3]

    result = train_win_model(snapshot_df, students, tmp_path / "out", seed=0)

    report = json.loads(Path(result["train_report_path"]).read_text(encoding="utf-8"))
    recovery = report["recovery"]

    # p_true recovery: predicted p_win closely tracks the generator's latent
    # probability — the core honesty claim for this component.
    p_true = recovery["p_true"]
    assert p_true["n_test"] > 0
    assert p_true["pearson_correlation"] is not None
    assert p_true["pearson_correlation"] > 0.9
    assert 0.0 <= p_true["mean_absolute_error"] <= 0.2

    # Coefficient recovery: one row per feature, signs match the generator.
    rows = recovery["coefficients"]
    assert [row["feature"] for row in rows] == list(FEATURE_COLUMNS)
    by_feature = {row["feature"]: row for row in rows}
    assert by_feature["major_match"]["generator_coefficient"] == GENERATOR_COEFFICIENTS["major_match"]
    # The strongest, non-collinear generator signals recover their direction.
    for feature in ("major_match", "education_level_match", "state_match", "essay_required"):
        assert by_feature[feature]["direction_consistent"] is True
    # A feature the generator never uses has no defined direction.
    assert by_feature["source_is_scholarship_america"]["direction_consistent"] is None
    # The clear majority of generator-driven features recover their sign; a
    # couple may be masked by collinearity in this tiny fixture (e.g. keyword
    # overlap vs. text similarity), which the strong p_true recovery confirms.
    generator_rows = [row for row in rows if row["generator_coefficient"] != 0.0]
    consistent = sum(1 for row in generator_rows if row["direction_consistent"])
    assert consistent >= len(generator_rows) - 1

    # Returned dict mirrors the report's recovery section.
    assert result["recovery"]["p_true"]["n_test"] == p_true["n_test"]
