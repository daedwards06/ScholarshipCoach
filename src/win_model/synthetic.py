"""Synthetic training-data generation for the win-probability model.

Randomly pairs golden student profiles with snapshot scholarships and assigns
binary win labels using a logistic model over human-interpretable features.
The synthetic labels encode intuitive domain rules (major match boosts win
probability, essay requirement reduces it, etc.) so the trained model learns
a sensible inductive bias even without real award outcome data.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.rank.stage2_scoring import build_scholarship_text, build_student_profile_text
from src.win_model.features import FEATURE_COLUMNS, build_pair_features

# Single source of truth for the synthetic label generator. The win model is a
# calibration/recovery demonstration: it should provably recover these known
# coefficients (directionally) and the resulting ``p_true``. Keys are feature
# names in ``FEATURE_COLUMNS``; values are the coefficient applied to that
# feature (after the clip/scale transforms below) inside the logit. Features
# absent here contribute nothing to the generator.
GENERATOR_INTERCEPT: float = -2.2
GENERATOR_COEFFICIENTS: dict[str, float] = {
    "major_match": 1.0,
    "education_level_match": 0.7,
    "state_match": 0.5,
    "text_sim": 0.6,
    "keyword_overlap": 0.6,
    "gpa_above_min": 0.4,
    "essay_required": -0.4,
    "days_to_deadline": -0.25,
    "amount_log": -0.35,
}


def _student_stage2_profile(student: Any) -> dict[str, Any]:
    if hasattr(student, "as_stage2_profile"):
        payload = dict(student.as_stage2_profile())
    elif isinstance(student, dict):
        payload = dict(student)
    else:
        payload = {}

    profile_obj = getattr(student, "profile", student)
    payload.setdefault("major", getattr(profile_obj, "major", payload.get("major")))
    return payload


def _student_profile(student: Any) -> Any:
    return getattr(student, "profile", student)


def _student_id(student: Any, index: int) -> str:
    return str(getattr(student, "student_id", f"profile_{index}"))


def _token_set(text: str) -> set[str]:
    tokens = [token for token in text.lower().split() if token]
    return set(tokens)


def _simple_text_similarity(student: Any, scholarship_row: pd.Series) -> float:
    student_text = build_student_profile_text(_student_stage2_profile(student))
    scholarship_text = build_scholarship_text(scholarship_row)
    left = _token_set(student_text)
    right = _token_set(scholarship_text)
    if not left or not right:
        return 0.0
    return float(len(left.intersection(right)) / len(left.union(right)))


def _simple_keyword_overlap(student: Any, scholarship_row: pd.Series) -> float:
    stage2_profile = _student_stage2_profile(student)
    student_keywords = _token_set(
        " ".join(
            [
                *[str(value) for value in stage2_profile.get("keywords", [])],
                *[str(value) for value in stage2_profile.get("interests", [])],
            ]
        )
    )
    _kw = scholarship_row.get("keywords")
    _kw_iter: Any = _kw if hasattr(_kw, "__iter__") and not isinstance(_kw, str) else []
    scholarship_tokens = _token_set(
        " ".join(
            [
                str(scholarship_row.get("title") or ""),
                " ".join(str(value) for value in _kw_iter),
            ]
        )
    )
    if not student_keywords:
        return 0.0
    return float(len(student_keywords.intersection(scholarship_tokens)) / len(student_keywords))


def generate_synthetic_training_data(
    snapshot_df: pd.DataFrame,
    golden_profiles: list[Any],
    n_samples: int = 8000,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Generate synthetic (features, labels) pairs for win-model training.

    Randomly samples ``n_samples`` (student, scholarship) pairs and assigns
    binary win labels via a hand-tuned logistic model that rewards major match,
    keyword overlap, and text similarity while penalising essay requirements.

    Args:
        snapshot_df: Scholarship snapshot DataFrame used as the candidate pool.
        golden_profiles: List of ``GoldenStudent`` (or equivalent) objects.
        n_samples: Number of training samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(X_df, y, meta_df)`` where ``X_df`` has ``FEATURE_COLUMNS``,
        ``y`` is a binary label array, and ``meta_df`` contains profile/
        scholarship IDs and the true latent probability.

    Raises:
        ValueError: If ``snapshot_df`` is empty, ``golden_profiles`` is empty,
            or ``n_samples < 1``.
    """
    if snapshot_df.empty:
        raise ValueError("Synthetic training requires at least one scholarship row.")
    if not golden_profiles:
        raise ValueError("Synthetic training requires at least one golden profile.")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    rng = np.random.RandomState(seed)
    scholarship_positions = rng.randint(0, len(snapshot_df), size=n_samples)
    profile_positions = rng.randint(0, len(golden_profiles), size=n_samples)

    feature_rows: list[dict[str, float]] = []
    meta_rows: list[dict[str, Any]] = []

    for sample_index in range(n_samples):
        scholarship_row = snapshot_df.iloc[int(scholarship_positions[sample_index])]
        student = golden_profiles[int(profile_positions[sample_index])]
        stage2_row = {
            "keyword_overlap": _simple_keyword_overlap(student, scholarship_row),
            "text_sim": _simple_text_similarity(student, scholarship_row),
        }
        pair_features = build_pair_features(
            _student_profile(student),
            scholarship_row,
            stage2_row=stage2_row,
            today=getattr(_student_profile(student), "today", None),
        )
        feature_rows.append(pair_features)
        meta_rows.append(
            {
                "profile_id": _student_id(student, int(profile_positions[sample_index])),
                "scholarship_id": str(scholarship_row.get("scholarship_id") or ""),
            }
        )

    X_df = pd.DataFrame(feature_rows, columns=list(FEATURE_COLUMNS))
    max_amount_log = float(X_df["amount_log"].max()) if not X_df.empty else 1.0
    if max_amount_log <= 0.0:
        max_amount_log = 1.0

    clipped_text = X_df["text_sim"].clip(lower=0.0, upper=1.0)
    clipped_keyword = (X_df["keyword_overlap"].clip(lower=0.0, upper=3.0) / 3.0).clip(lower=0.0, upper=1.0)
    clipped_gpa = X_df["gpa_above_min"].clip(lower=0.0, upper=1.0)
    clipped_deadline = (X_df["days_to_deadline"] / 365.0).clip(lower=0.0, upper=1.0)
    clipped_amount = (X_df["amount_log"] / max_amount_log).clip(lower=0.0, upper=1.0)

    logits = (
        GENERATOR_INTERCEPT
        + (GENERATOR_COEFFICIENTS["major_match"] * X_df["major_match"])
        + (GENERATOR_COEFFICIENTS["education_level_match"] * X_df["education_level_match"])
        + (GENERATOR_COEFFICIENTS["state_match"] * X_df["state_match"])
        + (GENERATOR_COEFFICIENTS["text_sim"] * clipped_text)
        + (GENERATOR_COEFFICIENTS["keyword_overlap"] * clipped_keyword)
        + (GENERATOR_COEFFICIENTS["gpa_above_min"] * clipped_gpa)
        + (GENERATOR_COEFFICIENTS["essay_required"] * X_df["essay_required"])
        + (GENERATOR_COEFFICIENTS["days_to_deadline"] * clipped_deadline)
        + (GENERATOR_COEFFICIENTS["amount_log"] * clipped_amount)
    )
    p_true = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float)))
    y = rng.binomial(1, p_true, size=n_samples).astype(int)

    meta_df = pd.DataFrame(meta_rows)
    meta_df["p_true"] = p_true
    return X_df, y, meta_df
