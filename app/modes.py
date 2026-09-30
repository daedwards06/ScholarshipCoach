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
only the ``render_*`` helpers touch the page.
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
PIN_STATE_KEY = "mode_pin"

SECTION_LABELS: dict[str, str] = {
    "this_week": "This Week",
    "find": "Find Scholarships",
    "applications": "My Applications",
    "essays": "Essays",
    "recommenders": "Recommenders",
    "what_if": "What If",
    "catalog_inbox": "Catalog & Inbox",
    "timeline": "Timeline",
    "colleges_money": "Colleges & Money",
    "outcomes": "Outcomes",
    "settings": "Settings",
    "profile": "My Profile",
}

SECTION_ICONS: dict[str, str] = {
    "this_week": ":material/today:",
    "find": ":material/search:",
    "applications": ":material/assignment:",
    "essays": ":material/edit_note:",
    "recommenders": ":material/group:",
    "what_if": ":material/tune:",
    "catalog_inbox": ":material/inbox:",
    "timeline": ":material/calendar_month:",
    "colleges_money": ":material/payments:",
    "outcomes": ":material/emoji_events:",
    "settings": ":material/settings:",
    "profile": ":material/person:",
}

SECTION_CAPTIONS: dict[str, str] = {
    "find": "Awards that match your profile, best fit first.",
    "applications": "Every award you have saved and how far along each one is.",
    "essays": "Your essay drafts and the prompts they can answer.",
    "recommenders": "The people writing your letters and which awards need them.",
    "what_if": (
        "Change one thing about the profile and see which awards open up. "
        "Nothing here is saved — the stored profile is untouched."
    ),
    "catalog_inbox": "Add awards to the catalog and review new leads.",
    "timeline": (
        "Deadlines, tasks, letters and milestones by month. Milestone dates are typical, "
        "not guaranteed — confirm each one with the college or agency."
    ),
    "colleges_money": "Each college's net price and how much of it the awards won so far cover.",
    "outcomes": "What each decided application brought in.",
    "settings": "Operator tools and the family milestone dates.",
    "profile": "What the app uses to match you with awards.",
}

PROFILE_SECTION = "profile"

# "find" is today's ranked-results surface. It sits in the student list because
# saving an award (Task 3.3) starts from a ranked card.  "what_if" is a student
# surface too: the record it asks about is the student's to build.
STUDENT_SECTIONS: tuple[str, ...] = (
    "this_week",
    "find",
    "applications",
    "essays",
    "recommenders",
    "what_if",
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


def pages_for_mode(mode: Mode) -> dict[str, tuple[str, ...]]:
    """Return the top-nav groups for ``mode``: group header -> sections, in nav order.

    The ``""`` header holds ungrouped pages.  The grouping is provisional; the
    page list and grouping belong to the Roles & Product Plan, and changing
    them should only touch this function and ``SECTION_LABELS``.
    """
    return {"": sections_for_mode(mode) + (PROFILE_SECTION,)}


def nav_sections(mode: Mode) -> tuple[str, ...]:
    """Every page in ``mode``'s navigation, flattened in nav order."""
    return tuple(section for group in pages_for_mode(mode).values() for section in group)


def section_url_path(section: str) -> str:
    return section.replace("_", "-")


def can_view(mode: Mode, section: str) -> bool:
    return section in nav_sections(mode)


def essays_read_only(mode: Mode) -> bool:
    """Parent mode reads the student's essays; it does not write them."""
    return normalize_mode(mode) == "parent"


def can_edit_essays(mode: Mode) -> bool:
    return not essays_read_only(mode)


def layout_for_mode(mode: Mode) -> Literal["centered", "wide"]:
    """Student reads in a centered column; parent and operator get dense wide pages."""
    return "centered" if normalize_mode(mode) == "student" else "wide"


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
    sections = nav_sections(mode)
    text = str(requested or "").strip().casefold()
    return text if text in sections else sections[0]


def render_page_header(section: str, caption: str | None = None) -> None:
    """Open a page with its own h1 and one line saying what the page is for."""
    st.title(SECTION_LABELS[section])
    line = caption if caption is not None else SECTION_CAPTIONS.get(section)
    if line:
        st.caption(line)


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
