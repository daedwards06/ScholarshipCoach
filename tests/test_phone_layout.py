from __future__ import annotations

import re

from app.helpers import PHONE_BREAKPOINT_PX, phone_width_css


def test_breakpoint_matches_streamlits_own_column_stacking_width() -> None:
    # Streamlit stacks st.columns below 640px; a different breakpoint here would
    # style one layout while the columns are still in the other.
    assert PHONE_BREAKPOINT_PX == 640


def test_css_is_scoped_to_the_phone_breakpoint() -> None:
    css = phone_width_css()
    assert f"@media (max-width: {PHONE_BREAKPOINT_PX}px)" in css
    # Every rule sits inside the one media query, so desktop is untouched.
    assert css.count("@media") == 1


def test_css_is_a_single_style_block_with_no_script() -> None:
    css = phone_width_css()
    assert css.startswith("<style>")
    assert css.rstrip().endswith("</style>")
    assert css.count("<style>") == 1
    assert "<script" not in css.lower()


def test_tap_targets_are_at_least_44px() -> None:
    css = phone_width_css()
    heights = [float(value) for value in re.findall(r"min-height: ([\d.]+)rem", css)]
    assert heights
    assert min(heights) * 16 >= 44


def test_text_inputs_stay_at_16px_so_mobile_safari_does_not_zoom() -> None:
    css = phone_width_css()
    sizes = [float(value) for value in re.findall(r"font-size: ([\d.]+)rem", css)]
    assert sizes
    assert min(sizes) * 16 >= 16


def test_buttons_and_essay_editor_are_styled_for_phone_width() -> None:
    css = phone_width_css()
    for testid in ("stButton", "stFormSubmitButton", "stLinkButton", "stTextArea"):
        assert f'[data-testid="{testid}"]' in css
