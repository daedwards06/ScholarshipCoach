"""Streamlit entry point used only by ``scripts/screenshot_app.py``.

It repoints the private data paths before any ``app`` module is imported
(``app.state`` resolves the profile path at import time), then lets the
harness pick the mode with ``?shot_mode=...`` so a capture does not have to
drive the mode switch or type the PIN.  The page itself comes from the URL
path, as it would for anyone opening a bookmark.

The journey walk also repoints the catalog (records and inbox) and the
snapshot folder at scratch copies, so its catalog form and rebuild never
write the real ``data/catalog/records/`` or ``data/processed/``.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from scripts import run_ingest
from src.catalog import inbox
from src.ingest.sources import curated_catalog
from src.normalize import catalog_schema
from src.profile import store as profile_store
from src.store import db

PRIVATE_DIR_ENV = "COACH_SHOT_PRIVATE_DIR"
DATA_ROOT_ENV = "COACH_SHOT_DATA_ROOT"
_APPLIED_STATE_KEY = "_shot_params_applied"

_private = os.environ.get(PRIVATE_DIR_ENV)
if _private:
    _private_dir = Path(_private)
    db.PRIVATE_DIR = _private_dir
    db.DEFAULT_DB_PATH = _private_dir / "coach.db"
    profile_store.PRIVATE_DIR = _private_dir
    profile_store.STUDENTS_DIR = _private_dir / "students"

_data_root = os.environ.get(DATA_ROOT_ENV)
_records_dir: Path | None = None
if _data_root:
    _catalog_dir = Path(_data_root) / "data" / "catalog"
    _records_dir = _catalog_dir / "records"
    catalog_schema.CATALOG_DIR = _catalog_dir
    catalog_schema.RECORDS_DIR = _records_dir
    inbox.CATALOG_DIR = _catalog_dir
    inbox.RECORDS_DIR = _records_dir
    inbox.INBOX_DIR = _catalog_dir / "inbox"
    inbox.REJECTED_DIR = inbox.INBOX_DIR / "rejected"
    curated_catalog.RECORDS_DIR = _records_dir
    # Snapshots, change files and run reports all hang off this root.
    run_ingest.ROOT_DIR = Path(_data_root)

from app import modes  # noqa: E402
from app.main import main  # noqa: E402
from app.sections import catalog_inbox  # noqa: E402

if _records_dir is not None:
    catalog_inbox.RECORDS_DIR = _records_dir

if not st.session_state.get(_APPLIED_STATE_KEY):
    st.session_state[_APPLIED_STATE_KEY] = True
    requested_mode = st.query_params.get("shot_mode")
    if requested_mode:
        st.session_state[modes.MODE_REQUEST_STATE_KEY] = modes.normalize_mode(requested_mode)
        # A configured family PIN would otherwise stop every parent capture.
        st.session_state[modes.UNLOCK_STATE_KEY] = True

main()
