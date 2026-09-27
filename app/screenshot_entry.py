"""Streamlit entry point used only by ``scripts/screenshot_app.py``.

It repoints the private data paths before any ``app`` module is imported
(``app.state`` resolves the profile path at import time), then lets the
harness open a section directly with ``?shot_mode=...&shot_section=...``
so a capture does not have to drive the navigation widgets.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.profile import store as profile_store
from src.store import db

PRIVATE_DIR_ENV = "COACH_SHOT_PRIVATE_DIR"
_APPLIED_STATE_KEY = "_shot_params_applied"

_private = os.environ.get(PRIVATE_DIR_ENV)
if _private:
    _private_dir = Path(_private)
    db.PRIVATE_DIR = _private_dir
    db.DEFAULT_DB_PATH = _private_dir / "coach.db"
    profile_store.PRIVATE_DIR = _private_dir
    profile_store.STUDENTS_DIR = _private_dir / "students"

from app import modes  # noqa: E402
from app.main import main  # noqa: E402

if not st.session_state.get(_APPLIED_STATE_KEY):
    st.session_state[_APPLIED_STATE_KEY] = True
    requested_mode = st.query_params.get("shot_mode")
    requested_section = st.query_params.get("shot_section")
    if requested_mode:
        st.session_state[modes.MODE_REQUEST_STATE_KEY] = modes.normalize_mode(requested_mode)
        # A configured family PIN would otherwise stop every parent capture.
        st.session_state[modes.UNLOCK_STATE_KEY] = True
    if requested_section:
        st.session_state[modes.SECTION_STATE_KEY] = requested_section

main()
