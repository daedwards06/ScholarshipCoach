from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
from streamlit import config as st_config

from app.helpers import CAPTION_OPACITY

CONFIG_PATH = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
ROOT_DIR = CONFIG_PATH.parents[1]
THEMES = ("light", "dark")
STATUS_COLORS = ("red", "orange", "yellow", "green", "blue", "violet", "gray")
WHITE = "#FFFFFF"


def _load() -> dict[str, Any]:
    with CONFIG_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _theme(name: str) -> dict[str, Any]:
    return _load()["theme"][name]


def _luminance(hex_color: str) -> float:
    channels = [int(hex_color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(foreground: str, background: str) -> float:
    lighter, darker = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def test_contrast_function_matches_known_values() -> None:
    assert contrast("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert contrast("#777777", "#FFFFFF") == pytest.approx(4.48, abs=0.01)


@pytest.mark.parametrize("name", THEMES)
def test_body_and_muted_text_meet_4_5(name: str) -> None:
    theme = _theme(name)
    for surface in ("backgroundColor", "secondaryBackgroundColor"):
        assert contrast(theme["textColor"], theme[surface]) >= 4.5
        assert contrast(theme["grayTextColor"], theme[surface]) >= 4.5


def _blend(foreground: str, background: str, alpha: float) -> str:
    fg, bg = (
        [int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
        for color in (foreground, background)
    )
    return "#" + "".join(f"{round(alpha * f + (1 - alpha) * b):02X}" for f, b in zip(fg, bg))


@pytest.mark.parametrize("name", THEMES)
def test_captions_meet_4_5_at_the_injected_opacity(name: str) -> None:
    # st.caption is body text at reduced opacity, not grayTextColor.
    theme = _theme(name)
    for surface in (
        theme["backgroundColor"],
        theme["secondaryBackgroundColor"],
        theme["sidebar"]["backgroundColor"],
    ):
        caption = _blend(theme["textColor"], surface, CAPTION_OPACITY)
        assert contrast(caption, surface) >= 4.5


@pytest.mark.parametrize("name", THEMES)
def test_white_button_label_on_primary_meets_4_5(name: str) -> None:
    # Streamlit paints primary-button labels white in both themes.
    assert contrast(WHITE, _theme(name)["primaryColor"]) >= 4.5


@pytest.mark.parametrize("name", THEMES)
def test_primary_link_and_border_stand_out_from_background(name: str) -> None:
    theme = _theme(name)
    background = theme["backgroundColor"]
    assert contrast(theme["primaryColor"], background) >= 3.0
    assert contrast(theme["borderColor"], background) >= 3.0
    assert contrast(theme["linkColor"], background) >= 4.5


@pytest.mark.parametrize("name", THEMES)
@pytest.mark.parametrize("color", STATUS_COLORS)
def test_status_badge_text_meets_4_5(name: str, color: str) -> None:
    theme = _theme(name)
    # Streamlit 1.54 has no grayTextColor for badges alone; gray badges reuse
    # the muted caption color.
    text_key = "grayTextColor" if color == "gray" else f"{color}TextColor"
    assert contrast(theme[text_key], theme[f"{color}BackgroundColor"]) >= 4.5


@pytest.mark.parametrize("name", THEMES)
def test_every_status_color_is_set(name: str) -> None:
    theme = _theme(name)
    for color in STATUS_COLORS:
        assert f"{color}Color" in theme
        assert f"{color}BackgroundColor" in theme
    assert theme["sidebar"]["backgroundColor"]


def _flatten(table: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in table.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def test_every_key_is_a_real_streamlit_option() -> None:
    # Streamlit ignores an unknown key without a warning, so a typo would ship
    # the default color unnoticed.
    known = set(st_config._config_options_template)
    for key in _flatten(_load()):
        option = key.replace("theme.light.", "theme.").replace("theme.dark.", "theme.")
        assert option in known, key


def test_fonts_are_self_hosted() -> None:
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    assert "fonts.googleapis.com" not in raw
    assert "fonts.gstatic.com" not in raw
    data = _load()
    assert data["server"]["enableStaticServing"] is True
    faces = data["theme"]["fontFaces"]
    assert {face["family"] for face in faces} == {"Nunito", "Nunito Sans"}
    for face in faces:
        assert face["url"].startswith("app/static/fonts/")
        assert (ROOT_DIR / face["url"]).is_file(), face["url"]
    assert (ROOT_DIR / "app" / "static" / "fonts" / "OFL.txt").is_file()
