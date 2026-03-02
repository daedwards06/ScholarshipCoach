"""Tests for src.text_utils shared normalization functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.text_utils import normalize_list, normalize_text


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


def test_normalize_text_strips_and_lowercases():
    assert normalize_text("  Hello World  ") == "hello world"


def test_normalize_text_none_returns_default():
    assert normalize_text(None) == ""


def test_normalize_text_none_custom_default():
    assert normalize_text(None, default="unknown") == "unknown"


def test_normalize_text_empty_string_returns_default():
    assert normalize_text("") == ""


def test_normalize_text_whitespace_only_returns_default():
    assert normalize_text("   ") == ""


def test_normalize_text_whitespace_custom_default():
    assert normalize_text("   ", default="fallback") == "fallback"


def test_normalize_text_integer():
    assert normalize_text(42) == "42"


def test_normalize_text_float():
    assert normalize_text(3.14) == "3.14"


def test_normalize_text_bool_true():
    assert normalize_text(True) == "true"


def test_normalize_text_list_coerces_to_string():
    # A plain list is coerced via str(); not a common usage, but should not raise.
    result = normalize_text(["a", "b"])
    assert isinstance(result, str)


@pytest.mark.parametrize("na_value", [None, np.nan, pd.NA, pd.NaT])
def test_normalize_text_pandas_na_values(na_value):
    """All pandas/numpy NA-like values must return the default."""
    assert normalize_text(na_value) == ""


def test_normalize_text_preserves_internal_spacing():
    """normalize_text strips leading/trailing but does NOT collapse internal spaces."""
    assert normalize_text("  hello   world  ") == "hello   world"


def test_normalize_text_already_normalized():
    assert normalize_text("already clean") == "already clean"


# ---------------------------------------------------------------------------
# normalize_list
# ---------------------------------------------------------------------------


def test_normalize_list_none_returns_empty():
    assert normalize_list(None) == []


def test_normalize_list_empty_list():
    assert normalize_list([]) == []


def test_normalize_list_plain_list():
    assert normalize_list(["NC", "CA", " TX "]) == ["nc", "ca", "tx"]


def test_normalize_list_tuple():
    assert normalize_list(("Computer Science", "Engineering")) == [
        "computer science",
        "engineering",
    ]


def test_normalize_list_set():
    result = normalize_list({"Alpha", "Beta"})
    assert sorted(result) == ["alpha", "beta"]


def test_normalize_list_string_becomes_single_element():
    assert normalize_list("Computer Science") == ["computer science"]


def test_normalize_list_string_empty_after_strip():
    assert normalize_list("   ") == []


def test_normalize_list_filters_empty_elements():
    assert normalize_list(["NC", "", "  ", "CA"]) == ["nc", "ca"]


def test_normalize_list_with_none_elements():
    """None elements inside a list are filtered out (normalize_text returns "")."""
    assert normalize_list(["NC", None, "CA"]) == ["nc", "ca"]


def test_normalize_list_with_mixed_types():
    """Numeric elements are coerced to strings."""
    result = normalize_list([1, "two", None])
    assert result == ["1", "two"]
