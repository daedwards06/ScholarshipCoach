from __future__ import annotations

from typing import Any

import pandas as pd

from src.rank.stage1_eligibility import UNVERIFIED_AXIS_LABELS
from src.rank.stage3_rerank import effort_counts, is_local_award
from src.text_utils import coerce_text


# Streamlit stacks ``st.columns`` to full width below this width on its own, so
# the rules below only have to fix what stacking does not: tap targets, the
# 16px input floor that stops iOS zooming on focus, and page padding.
PHONE_BREAKPOINT_PX = 640

# 44px is the smallest comfortable tap target on a phone.
_MIN_TAP_TARGET_REM = 2.75

# Below 16px, mobile Safari zooms the page when an input takes focus and does
# not zoom back out -- which is how the essay editor becomes unusable.
_MIN_INPUT_FONT_REM = 1.0


def phone_width_css() -> str:
    """Return the stylesheet that makes the student surfaces usable at ~400px.

    Streamlit class names are not a public API, so every rule here is an
    improvement on a layout that already works without it: if a selector stops
    matching, the page degrades to Streamlit's own responsive behaviour.
    """
    return f"""<style>
@media (max-width: {PHONE_BREAKPOINT_PX}px) {{
  [data-testid="stMainBlockContainer"] {{
    padding: 1.5rem 1rem 4rem;
  }}
  [data-testid="stButton"] button,
  [data-testid="stFormSubmitButton"] button,
  [data-testid="stLinkButton"] a,
  [data-testid="stDownloadButton"] button {{
    width: 100%;
    min-height: {_MIN_TAP_TARGET_REM}rem;
  }}
  [data-testid="stTextInput"] input,
  [data-testid="stTextArea"] textarea,
  [data-testid="stNumberInput"] input,
  [data-testid="stDateInput"] input {{
    font-size: {_MIN_INPUT_FONT_REM}rem;
  }}
  [data-testid="stExpander"] summary {{
    min-height: {_MIN_TAP_TARGET_REM}rem;
  }}
  [data-testid="stJson"],
  [data-testid="stMainBlockContainer"] pre {{
    max-width: 100%;
    overflow-x: auto;
  }}
}}
</style>"""


def format_amount_range(amount_min: Any, amount_max: Any) -> str:
    min_value = _coerce_amount(amount_min)
    max_value = _coerce_amount(amount_max)
    if min_value is None and max_value is None:
        return "Unknown"
    if min_value is None:
        return f"Up to ${max_value:,.0f}"
    if max_value is None:
        return f"${min_value:,.0f}+"
    if abs(min_value - max_value) < 1e-9:
        return f"${min_value:,.0f}"
    return f"${min_value:,.0f} - ${max_value:,.0f}"


def needs_date_awards(df: Any) -> pd.DataFrame:
    """The ``needs_date`` awards in ``df``, biggest award first.

    Nobody can sort these by deadline -- that is the whole point of the bucket
    -- so the money decides which one is worth looking up first.
    """
    if not isinstance(df, pd.DataFrame) or "timeline_bucket" not in df.columns:
        return pd.DataFrame()
    waiting = df[df["timeline_bucket"].astype("string").eq("needs_date")].copy()
    if waiting.empty:
        return waiting
    worth = waiting.apply(
        lambda row: _coerce_amount(row.get("amount_max"))
        or _coerce_amount(row.get("amount_min"))
        or 0.0,
        axis=1,
    )
    return waiting.assign(_worth=worth).sort_values("_worth", ascending=False).drop(
        columns="_worth"
    )


def effort_to_text(row: pd.Series) -> str:
    """Render an award's effort as plain counts, or "" when it asks for nothing.

    A family can act on "2 essays, 1 letter"; it cannot act on an effort
    penalty of 0.31.
    """
    counts = effort_counts(row)
    parts = [
        _pluralize(counts["essays"], "essay", "essays"),
        _pluralize(counts["letters"], "letter", "letters"),
        _pluralize(counts["extras"], "extra item", "extra items"),
    ]
    return ", ".join(part for part in parts if part)


def explain_ranked_row(
    row: pd.Series, *, max_signals: int = 3, operator_mode: bool = False
) -> list[str]:
    """Explain a ranked row in signals a student can act on.

    The expected-value line is operator-only: it is a synthetic estimate, and a
    teen reads it as a promise.
    """
    text_similarity = (
        _coerce_float(row.get("text_sim"))
        or _coerce_float(row.get("tfidf_sim"))
        or _coerce_float(row.get("embed_sim"))
        or 0.0
    )
    signal_scores = [
        (
            float(text_similarity),
            "Strong match to your goals/keywords",
        ),
        (
            float(_coerce_float(row.get("amount_utility")) or 0.0),
            "High award amount",
        ),
        (
            float(_coerce_float(row.get("keyword_overlap")) or 0.0),
            "High direct keyword overlap",
        ),
        (
            float(_coerce_float(row.get("urgency_boost")) or 0.0),
            "Deadline soon, boosted for urgency",
        ),
    ]
    if operator_mode:
        signal_scores.append(
            (
                float(
                    _coerce_float(row.get("expected_value_norm"))
                    or _coerce_float(row.get("ev_proxy_norm"))
                    or 0.0
                ),
                "Strong expected-value proxy",
            )
        )
    if not bool(row.get("essay_required")):
        signal_scores.append((0.4, "Lower effort (no essay)"))

    ranked = [label for score, label in sorted(signal_scores, key=lambda item: item[0], reverse=True) if score > 0]
    lines = ranked[:max_signals] if ranked else ["Balanced profile fit after scoring"]

    if is_local_award(row):
        lines.append("Local award, smaller applicant pool")
    effort_text = effort_to_text(row)
    if effort_text:
        lines.append(effort_text)
    return lines


def reasons_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value if str(item).strip())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        return reasons_to_text(value.tolist())
    return coerce_text(value)


def unverified_to_text(value: Any) -> str:
    """Render Stage 1 ``unverified_axes`` as the text after "Confirm you meet:".

    Unknown axis keys fall back to their own spelling rather than being dropped,
    so a new rule is visible in the UI before it has a label.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        axes = [str(item).strip() for item in value if str(item).strip()]
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        return unverified_to_text(value.tolist())
    else:
        text = coerce_text(value)
        axes = [text] if text else []
    return ", ".join(
        UNVERIFIED_AXIS_LABELS.get(axis, axis.replace("_", " ")) for axis in axes
    )


def _pluralize(count: int, singular: str, plural: str) -> str:
    if count <= 0:
        return ""
    return f"{count} {singular if count == 1 else plural}"


def _coerce_amount(value: Any) -> float | None:
    coerced = _coerce_float(value)
    if coerced is None:
        return None
    return max(coerced, 0.0)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric
