"""Shared text normalization utilities.

These functions replace the private ``_normalize_text`` / ``_normalize_list``
helpers that were previously copy-pasted across seven source modules.  All
callers should import from here instead of defining their own variants.

Notable behavioral contract
----------------------------
- ``normalize_text`` lowercases and strips the value, returning ``default``
  (empty string by default) when the input is ``None``, a pandas NA, or
  reduces to an empty string after stripping.
- ``normalize_list`` normalises every element of a list-like input; strings
  are treated as single-element lists; ``None`` yields ``[]``.
- ``canonical_id`` has its own thin wrapper (``_normalize_text``) that also
  collapses internal whitespace to keep SHA-1 hashes stable across ingests.
"""
from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


def normalize_text(value: Any, *, default: str = "") -> str:
    """Normalize a text value: coerce to string, lowercase, and strip whitespace.

    Returns ``default`` when the input is ``None``, a pandas NA/NaT, or
    produces an empty string after stripping.

    Args:
        value: The value to normalize.  Strings are stripped and lowercased
               directly; numeric/boolean scalars are converted via ``str()``
               first; pandas NA values are treated as missing.
        default: The value to return for ``None`` / empty inputs.  Defaults
                 to ``""``.

    Returns:
        Normalized lowercase stripped string, or ``default`` if the result
        would be empty.

    Examples:
        >>> normalize_text("  Hello World  ")
        'hello world'
        >>> normalize_text(None)
        ''
        >>> normalize_text(None, default="unknown")
        'unknown'
    """
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized if normalized else default
    # Handle pandas NA / NaT / np.nan without a hard numpy dependency.
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    normalized = str(value).strip().lower()
    return normalized if normalized else default


def normalize_list(value: Any) -> list[str]:
    """Normalize a list-like value of strings: coerce, lowercase, strip each element.

    Returns an empty list for ``None`` inputs.  A bare string is treated as a
    single-element list.  Any other ``Iterable`` has each element normalized
    via :func:`normalize_text`; empty results are filtered out.

    Args:
        value: The value to normalize.  Can be ``None``, a ``str``, or any
               ``Iterable`` (``list``, ``tuple``, ``set``, …).

    Returns:
        A list of normalized non-empty strings.

    Examples:
        >>> normalize_list(["NC", "CA", " TX "])
        ['nc', 'ca', 'tx']
        >>> normalize_list(None)
        []
        >>> normalize_list("Computer Science")
        ['computer science']
    """
    if value is None:
        return []
    if isinstance(value, str):
        normalized = normalize_text(value)
        return [normalized] if normalized else []
    if isinstance(value, Iterable):
        return [item for item in (normalize_text(v) for v in value) if item]
    return []
