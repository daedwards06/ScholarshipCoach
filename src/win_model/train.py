"""Win-probability model training pipeline.

Generates synthetic training data from golden profiles and a snapshot,
trains a calibrated logistic regression, writes the ``WinModelArtifact``
to disk, and returns a structured training report.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.win_model.features import FEATURE_COLUMNS
from src.win_model.infer import WinModelArtifact
from src.win_model.synthetic import (
    GENERATOR_COEFFICIENTS,
    GENERATOR_INTERCEPT,
    generate_synthetic_training_data,
)


def _calibration_bins(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == bins - 1:
            mask = (y_pred >= lower) & (y_pred <= upper)
        else:
            mask = (y_pred >= lower) & (y_pred < upper)
        if not np.any(mask):
            rows.append(
                {
                    "bin": index,
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "predicted_prob_mean": None,
                    "observed_rate": None,
                }
            )
            continue
        rows.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "count": int(np.sum(mask)),
                "predicted_prob_mean": float(np.mean(y_pred[mask])),
                "observed_rate": float(np.mean(y_true[mask])),
            }
        )
    return rows


def _p_true_recovery(predicted: np.ndarray, p_true: np.ndarray) -> dict[str, Any]:
    """Compare predicted ``p_win`` against the generator's latent ``p_true``.

    Reports Pearson correlation and mean absolute error on the held-out test
    split. This is the core honesty check: a well-calibrated pipeline over a
    known generator should recover ``p_true`` closely.
    """
    predicted = np.asarray(predicted, dtype=float)
    p_true = np.asarray(p_true, dtype=float)
    if predicted.size < 2 or np.std(predicted) == 0.0 or np.std(p_true) == 0.0:
        correlation: float | None = None
    else:
        correlation = float(np.corrcoef(predicted, p_true)[0, 1])
    return {
        "pearson_correlation": correlation,
        "mean_absolute_error": float(np.mean(np.abs(predicted - p_true))),
        "predicted_mean": float(np.mean(predicted)),
        "p_true_mean": float(np.mean(p_true)),
        "n_test": int(predicted.size),
    }


def _coefficient_recovery(base_model: Pipeline) -> list[dict[str, Any]]:
    """Compare learned logistic coefficients against generator coefficients.

    The base classifier is fit on standardised features, so learned magnitudes
    are not directly comparable to the generator's raw-scale coefficients; the
    *sign* is, since ``StandardScaler`` rescales by a positive standard
    deviation. Features absent from the generator (coefficient 0) have no
    defined direction and are reported with ``direction_consistent=None``.
    """
    classifier = base_model.named_steps["classifier"]
    learned = np.asarray(classifier.coef_[0], dtype=float)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(FEATURE_COLUMNS):
        learned_coef = float(learned[index])
        generator_coef = float(GENERATOR_COEFFICIENTS.get(name, 0.0))
        if generator_coef == 0.0:
            direction_consistent: bool | None = None
        else:
            direction_consistent = bool(np.sign(learned_coef) == np.sign(generator_coef))
        rows.append(
            {
                "feature": name,
                "learned_coefficient": learned_coef,
                "generator_coefficient": generator_coef,
                "direction_consistent": direction_consistent,
            }
        )
    return rows


def train_win_model(
    snapshot_df: pd.DataFrame,
    golden_profiles: list[Any],
    out_dir: Path,
    seed: int = 0,
) -> dict[str, Any]:
    """Train, calibrate, evaluate, and persist a win-probability model.

    Uses 80/20 train–test split with an additional 25% held out from train
    for Platt scaling calibration.

    Args:
        snapshot_df: Scholarship snapshot used to generate synthetic training pairs.
        golden_profiles: List of student profiles for pair generation.
        out_dir: Directory where the model artifact and training report are written.
        seed: Random seed passed to data generation and sklearn for reproducibility.

    Returns:
        Dict with ``model_path``, ``latest_model_pointer``, ``train_report_path``,
        ``metrics`` (ROC-AUC, Brier score, log loss, positive rate), and
        ``recovery`` (generator ``p_true`` correlation/MAE and per-feature
        learned-vs-generator coefficient directions).
    """
    out_dir = out_dir if out_dir.is_absolute() else out_dir.resolve()
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    X_df, y, meta_df = generate_synthetic_training_data(
        snapshot_df,
        golden_profiles,
        seed=seed,
    )
    p_true_all = meta_df["p_true"].to_numpy()
    X_train_full, X_test, y_train_full, y_test, _, p_true_test = train_test_split(
        X_df,
        y,
        p_true_all,
        test_size=0.2,
        random_state=0,
        stratify=y,
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=0,
        stratify=y_train_full,
    )

    base_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=0,
                ),
            ),
        ]
    )
    base_model.fit(X_train.loc[:, list(FEATURE_COLUMNS)], y_train)

    calibration_input = base_model.predict_proba(X_cal.loc[:, list(FEATURE_COLUMNS)])[:, 1].reshape(-1, 1)
    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    )
    calibrator.fit(calibration_input, y_cal)

    test_raw = base_model.predict_proba(X_test.loc[:, list(FEATURE_COLUMNS)])[:, 1]
    test_pred = calibrator.predict_proba(test_raw.reshape(-1, 1))[:, 1]
    test_pred = np.clip(np.asarray(test_pred, dtype=float), 0.0, 1.0)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_pred)),
        "brier_score": float(brier_score_loss(y_test, test_pred)),
        "log_loss": float(log_loss(y_test, test_pred, labels=[0, 1])),
        "positive_rate": float(np.mean(y)),
        "n_samples": int(len(y)),
        "calibration_bins": _calibration_bins(y_test, test_pred, bins=10),
    }

    recovery = {
        "description": (
            "Honesty check: the win model is a calibration/EV pipeline over a "
            "known synthetic generator. These fields show it recovers the "
            "generator's latent p_true and coefficient directions."
        ),
        "generator_intercept": GENERATOR_INTERCEPT,
        "p_true": _p_true_recovery(test_pred, p_true_test),
        "coefficients": _coefficient_recovery(base_model),
    }

    artifact = WinModelArtifact(
        model=base_model,
        calibrator=calibrator,
        feature_names=FEATURE_COLUMNS,
        training_summary={
            "metrics": metrics,
            "seed": int(seed),
            "feature_names": list(FEATURE_COLUMNS),
            "train_count": int(len(X_train)),
            "calibration_count": int(len(X_cal)),
            "test_count": int(len(X_test)),
        },
    )

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"win_model_{timestamp}.joblib"
    latest_pointer = out_dir / "latest_model.txt"
    report_path = out_dir / f"train_report_{timestamp}.json"

    joblib.dump(artifact, model_path)
    latest_pointer.write_text(str(model_path.resolve()), encoding="utf-8")

    report_payload = {
        "generated_at": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": int(seed),
        "model_path": str(model_path.resolve()),
        "latest_model_pointer": str(latest_pointer.resolve()),
        "feature_names": list(FEATURE_COLUMNS),
        "metrics": metrics,
        "recovery": recovery,
        "split_sizes": {
            "train": int(len(X_train)),
            "calibration": int(len(X_cal)),
            "test": int(len(X_test)),
        },
        "meta_preview": meta_df.head(25).to_dict(orient="records"),
    }
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "model_path": str(model_path.resolve()),
        "latest_model_pointer": str(latest_pointer.resolve()),
        "train_report_path": str(report_path.resolve()),
        "metrics": metrics,
        "recovery": recovery,
    }
