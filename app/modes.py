"""Student, parent and operator views over one profile and one database.

The family shares a household machine, so the gate here is a selector, not an
account system.  Student mode is the daily surface: what is due and what to
write.  Parent mode adds curation, money, outcomes and settings, and reads the
student's essays without editing them.  Operator mode keeps the pipeline
controls this app was built around, and stays hidden unless the family turns
on the ``operator_enabled`` setting, so the tuning surface never greets a
student.

A PIN in ``.streamlit/secrets.toml`` turns the parent selector into a soft
lock.  With no secrets file there is no PIN and the selector is open, which is
what a trusting household wants by default.

Every decision function here is pure so it can be tested without Streamlit;
only the two ``render_*`` helpers touch the page.
"""
from __future__ import annotations

import hmac
from collections.abc import Mapping
from typing import Any, Literal, cast

import streamlit as st

Mode = Literal["student", "parent", "operator"]

DEFAULT_MODE: Mode = "student"

MODE_LABELS: dict[Mode, str] = {
    "student": "Student",
    "parent": "Parent",
    "operator": "Operator",
}

MODE_CAPTIONS: dict[Mode, str] = {
    "student": "What is due and what to write.",
    "parent": "Curation, money, outcomes and settings. Essays are read-only.",
    "operator": "Everything, plus the ranking pipeline and ingest controls.",
}

OPERATOR_ENABLED_SETTING = "operator_enabled"
PARENT_PIN_SECRET = "parent_pin"

MODE_STATE_KEY = "mode"
MODE_REQUEST_STATE_KEY = "mode_request"
UNLOCK_STATE_KEY = "mode_unlocked"
SECTION_STATE_KEY = "section"
PIN_STATE_KEY = "mode_pin"

SECTION_LABELS: dict[str, str] = {
    "this_week": "This Week",
    "find": "Find Scholarships",
    "applications": "My Applications",
    "essays": "Essays",
    "recommenders": "Recommenders",
    "catalog_inbox": "Catalog & Inbox",
    "timeline": "Timeline",
    "colleges_money": "Colleges & Money",
    "outcomes": "Outcomes",
    "settings": "Settings",
}

# "find" is today's ranked-results surface. It sits in the student list because
# saving an award (Task 3.3) starts from a ranked card.
STUDENT_SECTIONS: tuple[str, ...] = (
    "this_week",
    "find",
    "applications",
    "essays",
    "recommenders",
)

PARENT_ONLY_SECTIONS: tuple[str, ...] = (
    "catalog_inbox",
    "timeline",
    "colleges_money",
    "outcomes",
    "settings",
)

PARENT_SECTIONS: tuple[str, ...] = STUDENT_SECTIONS + PARENT_ONLY_SECTIONS


def normalize_mode(value: object) -> Mode:
    """Coerce anything to a known mode, falling back to student."""
    text = str(value or "").strip().casefold()
    if text in MODE_LABELS:
        return cast(Mode, text)
    return DEFAULT_MODE


def available_modes(operator_enabled: bool) -> tuple[Mode, ...]:
    """Return the modes the selector may offer."""
    if operator_enabled:
        return ("student", "parent", "operator")
    return ("student", "parent")


def sections_for_mode(mode: Mode) -> tuple[str, ...]:
    """Return the sections visible in ``mode``, in nav order."""
    if normalize_mode(mode) == "student":
        return STUDENT_SECTIONS
    return PARENT_SECTIONS


def can_view(mode: Mode, section: str) -> bool:
    return section in sections_for_mode(mode)


def essays_read_only(mode: Mode) -> bool:
    """Parent mode reads the student's essays; it does not write them."""
    return normalize_mode(mode) == "parent"


def can_edit_essays(mode: Mode) -> bool:
    return not essays_read_only(mode)


def show_operator_tools(mode: Mode, operator_enabled: bool) -> bool:
    return operator_enabled and normalize_mode(mode) == "operator"


def parent_pin(secrets: Mapping[str, Any] | None) -> str | None:
    """Read the optional parent PIN out of a secrets mapping."""
    if not secrets:
        return None
    raw = secrets.get(PARENT_PIN_SECRET)
    text = str(raw if raw is not None else "").strip()
    return text or None


def parent_pin_from_secrets() -> str | None:
    """Read the PIN from ``st.secrets``, treating an absent file as no PIN."""
    try:
        return parent_pin(cast(Mapping[str, Any], st.secrets))
    except Exception:
        # Streamlit raises rather than returning empty when there is no
        # secrets.toml at all, and that is the common case for this family.
        return None


def pin_required(mode: Mode, pin: str | None) -> bool:
    return normalize_mode(mode) != "student" and bool(pin)


def check_pin(expected: str | None, supplied: str | None) -> bool:
    """Compare a supplied PIN against the configured one; no PIN means open."""
    if not expected:
        return True
    return hmac.compare_digest(str(expected).strip(), str(supplied or "").strip())


def resolve_mode(
    requested: object,
    *,
    operator_enabled: bool,
    pin: str | None = None,
    unlocked: bool = False,
) -> Mode:
    """Return the mode actually granted for a request."""
    mode = normalize_mode(requested)
    if mode not in available_modes(operator_enabled):
        return DEFAULT_MODE
    if pin_required(mode, pin) and not unlocked:
        return DEFAULT_MODE
    return mode


def resolve_section(requested: object, mode: Mode) -> str:
    """Return a section visible in ``mode``, defaulting to its first."""
    sections = sections_for_mode(mode)
    text = str(requested or "").strip().casefold()
    return text if text in sections else sections[0]


def render_mode_selector(*, operator_enabled: bool, pin: str | None = None) -> Mode:
    """Draw the sidebar mode selector (with PIN prompt) and return the mode."""
    if pin is None:
        pin = parent_pin_from_secrets()

    choices = available_modes(operator_enabled)
    # The widget owns this key, so a stale request (operator selected, then
    # operator tools switched off) has to be cleared before the widget draws.
    if normalize_mode(st.session_state.get(MODE_REQUEST_STATE_KEY)) not in choices:
        st.session_state[MODE_REQUEST_STATE_KEY] = DEFAULT_MODE

    with st.sidebar:
        requested = cast(
            Mode,
            st.radio(
                "View",
                options=choices,
                format_func=lambda name: MODE_LABELS[cast(Mode, name)],
                horizontal=True,
                key=MODE_REQUEST_STATE_KEY,
            ),
        )
        unlocked = bool(st.session_state.get(UNLOCK_STATE_KEY, False))
        if pin_required(requested, pin) and not unlocked:
            supplied = st.text_input("Family PIN", type="password", key=PIN_STATE_KEY)
            if not supplied:
                st.caption(f"{MODE_LABELS[requested]} view needs the family PIN.")
            elif check_pin(pin, supplied):
                st.session_state[UNLOCK_STATE_KEY] = True
                unlocked = True
            else:
                st.error("Incorrect PIN.")

        mode = resolve_mode(
            requested, operator_enabled=operator_enabled, pin=pin, unlocked=unlocked
        )
        st.caption(MODE_CAPTIONS[mode])

    st.session_state[MODE_STATE_KEY] = mode
    return mode


def render_section_selector(mode: Mode) -> str:
    """Draw the sidebar section nav for ``mode`` and return the chosen section."""
    sections = sections_for_mode(mode)
    # Leaving parent mode while a parent-only section is selected would leave
    # the widget holding a value its own options no longer contain.
    st.session_state[SECTION_STATE_KEY] = resolve_section(
        st.session_state.get(SECTION_STATE_KEY), mode
    )
    with st.sidebar:
        section = str(
            st.radio(
                "Section",
                options=sections,
                format_func=lambda name: SECTION_LABELS[str(name)],
                key=SECTION_STATE_KEY,
            )
        )
    return section
