"""Loading human relevance labels for offline evaluation.

Human labels live in a CSV with ``profile_id``, ``scholarship_id``, and ``label``
(0/1/2) columns — the same schema the labeling worksheet emits. They provide a
non-circular check on the proxy labels, which share features with the ranker.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_human_labels(path: Path) -> dict[str, dict[str, int]]:
    """Load human relevance labels into ``{profile_id: {scholarship_id: label}}``.

    The CSV must have ``profile_id``, ``scholarship_id``, and ``label`` columns.
    Rows with a blank/missing profile, scholarship, or label, or a label outside
    {0, 1, 2}, are skipped so a partially filled worksheet still parses cleanly.

    Args:
        path: Path to the human-labels CSV.

    Returns:
        Nested mapping of profile id → scholarship id → integer label.

    Raises:
        ValueError: If a required column is missing.
    """
    frame = pd.read_csv(path, dtype=str)
    required = {"profile_id", "scholarship_id", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Human-labels CSV '{path}' is missing columns: {sorted(missing)}.")

    labels: dict[str, dict[str, int]] = {}
    for _, row in frame.iterrows():
        raw_profile = row["profile_id"]
        raw_scholarship = row["scholarship_id"]
        raw_label = row.get("label")
        if pd.isna(raw_profile) or pd.isna(raw_scholarship) or raw_label is None or pd.isna(raw_label):
            continue
        if str(raw_label).strip() == "":
            continue
        try:
            label = int(float(str(raw_label).strip()))
        except (TypeError, ValueError):
            continue
        if label not in (0, 1, 2):
            continue
        profile_id = str(raw_profile).strip()
        scholarship_id = str(raw_scholarship).strip()
        if not profile_id or not scholarship_id:
            continue
        labels.setdefault(profile_id, {})[scholarship_id] = label
    return labels
