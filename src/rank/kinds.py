"""What a catalog record is: a scholarship (money) or a program (something to do).

Stage 3 scores on amount, expected value and deadline urgency, so a $0 summer
program would always sink; programs are listed by grade and date on the
Opportunities page instead and never enter scholarship ranking.  A snapshot
built before ``kind`` existed, and every feed row, has no value, which means
scholarship.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.text_utils import normalize_text

SCHOLARSHIP_KIND = "scholarship"
PROGRAM_KIND = "program"


def record_kind(value: Any) -> str:
    """Normalize a ``kind`` cell; anything but ``program`` is a scholarship."""
    try:
        if pd.isna(value):
            return SCHOLARSHIP_KIND
    except (TypeError, ValueError):
        pass
    return PROGRAM_KIND if normalize_text(value) == PROGRAM_KIND else SCHOLARSHIP_KIND


def program_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of the program rows in ``df`` (all False without a ``kind`` column)."""
    if "kind" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df["kind"].map(lambda value: record_kind(value) == PROGRAM_KIND).astype(bool)


def scholarships_only(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` without its program rows, for everything that ranks money."""
    mask = program_mask(df)
    if not mask.any():
        return df
    return df.loc[~mask]
