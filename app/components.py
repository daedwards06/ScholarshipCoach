"""Small page pieces the design system specifies once and every page reuses."""
from __future__ import annotations

import streamlit as st
from streamlit.navigation.page import StreamlitPage


def render_empty_state(
    message: str,
    *,
    next_up: str | None = None,
    link_page: StreamlitPage | str | None = None,
    link_label: str | None = None,
) -> None:
    """MASTER "Empty state": what would be here, the next-best fact, one way on.

    A neutral bordered container, never ``st.success``: an empty list is not
    an achievement, and the green box is how "nothing due" hid a deadline
    33 days out.
    """
    with st.container(border=True):
        st.markdown(message)
        if next_up:
            st.markdown(f"**Next up:** {next_up}")
        if link_page is not None:
            st.page_link(link_page, label=link_label)
