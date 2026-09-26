from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from app import modes, sidebar, state
from app.helpers import phone_width_css
from app.sections import (
    applications,
    catalog_inbox,
    colleges_money,
    essays,
    find,
    outcomes,
    recommenders,
    settings,
    this_week,
    timeline,
    what_if,
)

SECTION_RENDERERS: dict[str, Callable[[], None]] = {
    "find": find.render,
    "this_week": this_week.render,
    "applications": applications.render,
    "essays": essays.render,
    "recommenders": recommenders.render,
    "timeline": timeline.render,
    "colleges_money": colleges_money.render,
    "catalog_inbox": catalog_inbox.render,
    "what_if": what_if.render,
    "outcomes": outcomes.render,
    "settings": settings.render,
}


def main() -> None:
    st.set_page_config(page_title="Scholarship Coach", layout="wide")
    st.markdown(phone_width_css(), unsafe_allow_html=True)
    st.title("Scholarship Coach")
    st.caption("Find scholarships matched to your profile")

    state.ensure_session_state()

    operator_enabled = state.operator_enabled()
    mode = modes.render_mode_selector(operator_enabled=operator_enabled)
    section = modes.render_section_selector(mode)

    with st.sidebar:
        sidebar.render_profile_sidebar()
        if modes.show_operator_tools(mode, operator_enabled):
            sidebar.render_operator_sidebar()

    st.session_state.profile = state.profile_from_widgets()

    renderer = SECTION_RENDERERS.get(section)
    if renderer is not None:
        renderer()


if __name__ == "__main__":
    main()
