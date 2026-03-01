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
from src.win_model.synthetic import generate_synthetic_training_data


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


def train_win_model(
    snapshot_df: pd.DataFrame,
    golden_profiles: list[Any],
    out_dir: Path,
    seed: int = 0,
) -> dict[str, Any]:
    out_dir = out_dir if out_dir.is_absolute() else out_dir.resolve()
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    X_df, y, meta_df = generate_synthetic_training_data(
        snapshot_df,
        golden_profiles,
        seed=seed,
    )
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_df,
        y,
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
    }
