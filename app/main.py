from __future__ import annotations

from collections.abc import Callable
from datetime import date

import streamlit as st
from streamlit.navigation.page import StreamlitPage
from streamlit.runtime.scriptrunner import get_script_run_ctx

from app import identity, modes, sidebar, state
from app.helpers import phone_width_css
from app.sections import (
    applications,
    catalog_inbox,
    colleges_money,
    essays,
    find,
    outcomes,
    profile,
    recommenders,
    settings,
    this_week,
    timeline,
    what_if,
)
from src.store import repo
from src.store.db import open_db

ACTIVITY_STATE_KEY = "activity_last_counted"

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
    "profile": profile.render,
}


def _page(section: str) -> StreamlitPage:
    return st.Page(
        SECTION_RENDERERS[section],
        title=modes.SECTION_LABELS[section],
        icon=modes.SECTION_ICONS[section],
        url_path=modes.section_url_path(section),
    )


def _requests_hidden_page(visible: list[StreamlitPage], every: list[StreamlitPage]) -> bool:
    """Whether this run asks for a page that exists but ``mode`` may not see.

    Streamlit answers an unknown page with a "Page not found" error dialog, which
    is what a student deep link to a parent URL (or switching to Student while
    on Settings) would otherwise hit.  The requested page is only exposed on the
    run context, so this reads it there.
    """
    ctx = get_script_run_ctx()
    manager = getattr(ctx, "pages_manager", None)
    if manager is None:
        return False
    page_hash = manager.intended_page_script_hash
    name = manager.intended_page_name

    def matches(page: StreamlitPage) -> bool:
        if page_hash:
            return bool(page._script_hash == page_hash)
        return bool(name) and page.url_path == name

    return any(matches(page) for page in every) and not any(matches(page) for page in visible)


def _record_activity(person: str, page: StreamlitPage) -> None:
    """Count a page opened once per person, day and page change (D6), not per rerun."""
    marker = (person, date.today().isoformat(), page.url_path)
    if st.session_state.get(ACTIVITY_STATE_KEY) == marker:
        return
    st.session_state[ACTIVITY_STATE_KEY] = marker
    try:
        with open_db() as conn:
            repo.record_activity(conn, person, marker[1])
    except Exception:
        # A missed count must never stop her page from opening.
        pass


def _render_identity_check() -> None:
    """Operator-only check that tailscale serve's headers reach Streamlit (Task A.2)."""
    presence = identity.header_presence(identity.headers_from_context())
    with st.expander("Identity headers", expanded=False):
        for name, present in presence.items():
            st.caption(f"{name}: {'present' if present else 'absent'}")


def main() -> None:
    state.ensure_session_state()

    operator_enabled = state.operator_enabled()
    who = identity.current_identity()
    mode = modes.render_mode_selector(operator_enabled=operator_enabled, identity=who)
    st.set_page_config(page_title="Scholarship Coach", layout=modes.layout_for_mode(mode))
    st.markdown(phone_width_css(), unsafe_allow_html=True)

    if modes.show_operator_tools(mode, operator_enabled):
        with st.sidebar:
            sidebar.render_operator_sidebar()
            _render_identity_check()

    st.session_state.profile = state.profile_from_widgets()

    pages = {section: _page(section) for section in SECTION_RENDERERS}
    groups = {
        header: [pages[section] for section in sections]
        for header, sections in modes.pages_for_mode(mode).items()
    }
    visible = [page for group in groups.values() for page in group]
    every = visible + [page for page in pages.values() if page not in visible]
    if _requests_hidden_page(visible, every):
        st.navigation(every, position="hidden")
        st.switch_page(visible[0])

    page = st.navigation(groups, position="top")
    _record_activity(identity.activity_key(who, mode), page)
    page.run()


if __name__ == "__main__":
    main()
