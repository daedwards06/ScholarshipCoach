from __future__ import annotations

import re

from app.helpers import AWARD_CARD_SELECTOR, PHONE_BREAKPOINT_PX, phone_width_css


def test_breakpoint_matches_streamlits_own_column_stacking_width() -> None:
    # Streamlit stacks st.columns below 640px; a different breakpoint here would
    # style one layout while the columns are still in the other.
    assert PHONE_BREAKPOINT_PX == 640


def test_css_is_scoped_to_the_phone_breakpoint() -> None:
    css = phone_width_css()
    assert f"@media (max-width: {PHONE_BREAKPOINT_PX}px)" in css
    # Every phone rule sits inside the one media query, so desktop is untouched;
    # the caption contrast fix is the only rule outside it.
    assert css.count("@media") == 1
    outside = css.split("@media")[0]
    assert outside.count("{") == 1
    assert '[data-testid="stCaptionContainer"]' in outside


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


def test_award_card_buttons_stay_in_one_row_at_phone_width() -> None:
    css = phone_width_css()
    phone_rules = css.split("@media")[1]
    # The full-width rule still applies everywhere else...
    assert "width: 100%;" in phone_rules
    # ...and award cards override it after it, so Save / Apply / Suggest share a row.
    scoped = phone_rules[phone_rules.index(AWARD_CARD_SELECTOR) :]
    assert phone_rules.index("width: 100%;") < phone_rules.index(AWARD_CARD_SELECTOR)
    assert f'{AWARD_CARD_SELECTOR} [data-testid="stButton"] button' in scoped
    assert f'{AWARD_CARD_SELECTOR} [data-testid="stLinkButton"] a' in scoped
    assert "width: auto;" in scoped


def test_award_card_selector_matches_the_keyed_card_container() -> None:
    # st.container(key="award_<id>") renders with the class "st-key-award_<id>".
    assert AWARD_CARD_SELECTOR == '[class*="st-key-award_"]'
