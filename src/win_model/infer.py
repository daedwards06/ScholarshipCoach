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
    model: Any
    calibrator: Any
    feature_names: tuple[str, ...] = field(default_factory=lambda: FEATURE_COLUMNS)
    training_summary: dict[str, Any] | None = None

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
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
    if not LATEST_MODEL_POINTER.exists():
        raise FileNotFoundError(f"No win model pointer found at '{LATEST_MODEL_POINTER}'.")
    model_path = Path(LATEST_MODEL_POINTER.read_text(encoding="utf-8").strip())
    if not model_path.is_absolute():
        model_path = (DEFAULT_WIN_MODEL_DIR / model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Win model path '{model_path}' from latest_model.txt does not exist.")
    return model_path


def load_model(path: Path) -> Any:
    resolved = path if path.is_absolute() else path.resolve()
    return joblib.load(resolved)


def load_latest_model() -> Any:
    return load_model(get_latest_model_path())


def predict_p_win(model: Any, features_df: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features_df)[:, 1], dtype=float)
    probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(probabilities, 0.0, 1.0)
