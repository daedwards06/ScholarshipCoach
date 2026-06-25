"""Win-probability model loading and inference utilities.

Provides ``WinModelArtifact`` (the serialised model + calibrator container)
and helper functions to load the latest artifact from disk and run inference
against a feature DataFrame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.win_model.features import FEATURE_COLUMNS

DEFAULT_WIN_MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "win_model"
LATEST_MODEL_POINTER = DEFAULT_WIN_MODEL_DIR / "latest_model.txt"


@dataclass(slots=True)
class WinModelArtifact:
    """Serialisable container for the trained win-probability model and its calibrator.

    Attributes:
        model: Fitted sklearn ``Pipeline`` (scaler + logistic regressor).
        calibrator: Fitted ``LogisticRegression`` that maps raw probabilities to
            calibrated probabilities.
        feature_names: Ordered tuple of feature column names used during training.
        training_summary: Optional dict of training metadata and evaluation metrics.
    """

    model: Any
    calibrator: Any
    feature_names: tuple[str, ...] = field(default_factory=lambda: FEATURE_COLUMNS)
    training_summary: dict[str, Any] | None = None

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """Run calibrated win-probability inference and return a (n, 2) probability array.

        Columns are [P(loss), P(win)] with values clipped to [0, 1].
        """
        ordered = features_df.loc[:, list(self.feature_names)]
        raw = np.asarray(self.model.predict_proba(ordered)[:, 1], dtype=float)
        if self.calibrator is None:
            calibrated = raw
        else:
            calibrated = np.asarray(
                self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1],
                dtype=float,
            )
        calibrated = np.nan_to_num(calibrated, nan=0.0, posinf=1.0, neginf=0.0)
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])


def get_latest_model_path() -> Path:
    """Resolve the path to the most recently trained win-model artifact.

    Raises:
        FileNotFoundError: If the ``latest_model.txt`` pointer or the model
            file it references does not exist.
    """
    if not LATEST_MODEL_POINTER.exists():
        raise FileNotFoundError(f"No win model pointer found at '{LATEST_MODEL_POINTER}'.")
    model_path = Path(LATEST_MODEL_POINTER.read_text(encoding="utf-8").strip())
    if not model_path.is_absolute():
        model_path = (DEFAULT_WIN_MODEL_DIR / model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Win model path '{model_path}' from latest_model.txt does not exist.")
    return model_path


def load_model(path: Path) -> Any:
    """Load a ``WinModelArtifact`` from ``path`` using joblib."""
    resolved = path if path.is_absolute() else path.resolve()
    return joblib.load(resolved)


def load_latest_model() -> Any:
    """Load the most recently trained win model, resolving the pointer file."""
    return load_model(get_latest_model_path())


def predict_p_win(model: Any, features_df: pd.DataFrame) -> np.ndarray:
    """Run the model and return win-probability scores clipped to [0, 1].

    Args:
        model: A ``WinModelArtifact`` or any object with a ``predict_proba`` method.
        features_df: DataFrame with columns matching ``FEATURE_COLUMNS``.

    Returns:
        Float64 array of win probabilities, one per row.
    """
    probabilities = np.asarray(model.predict_proba(features_df)[:, 1], dtype=float)
    probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(probabilities, 0.0, 1.0)
