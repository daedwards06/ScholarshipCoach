"""Source registry: returns all active ingest connectors for a pipeline run.

Which connectors run is configuration, not code: ``data/catalog/sources.json``
carries an ``enabled`` flag and a note per connector so a dead scraper can be
switched off without deleting its module.  A connector missing from the file is
enabled, so adding a new source needs no config change.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .base import BaseSource
from .sources.bold_org import BoldOrgSource
from .sources.curated_catalog import CuratedCatalogSource
from .sources.scholarship_america_live import ScholarshipAmericaLiveSource

logger = logging.getLogger(__name__)

SOURCES_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "catalog" / "sources.json"


def _all_sources() -> list[BaseSource]:
    return [
        CuratedCatalogSource(),
        ScholarshipAmericaLiveSource(),
        BoldOrgSource(),
    ]


def load_source_settings(config_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return ``{source_name: {"enabled": bool, "note": str | None}}``.

    An unreadable or malformed config leaves every connector enabled; a broken
    file must not silently stop the whole ingest.
    """
    path = config_path or SOURCES_CONFIG_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        logger.error("Source config %s is unreadable (%s); enabling every connector.", path, exc)
        return {}

    entries = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(entries, dict):
        logger.error("Source config %s has no 'sources' object; enabling every connector.", path)
        return {}

    settings: dict[str, dict[str, Any]] = {}
    for name, entry in entries.items():
        enabled = entry.get("enabled") if isinstance(entry, dict) else None
        note = entry.get("note") if isinstance(entry, dict) else None
        settings[str(name)] = {
            "enabled": enabled if isinstance(enabled, bool) else True,
            "note": str(note) if note else None,
        }
    return settings


def disabled_sources(config_path: Path | None = None) -> list[dict[str, Any]]:
    """Return ``[{"source": name, "note": note}]`` for every connector switched off."""
    settings = load_source_settings(config_path)
    return [
        {"source": source.name, "note": settings[source.name].get("note")}
        for source in _all_sources()
        if not settings.get(source.name, {"enabled": True})["enabled"]
    ]


def register_sources(config_path: Path | None = None) -> list[BaseSource]:
    """Return every enabled :class:`~src.ingest.base.BaseSource` instance for ingest."""
    settings = load_source_settings(config_path)
    active: list[BaseSource] = []
    for source in _all_sources():
        entry = settings.get(source.name)
        if entry is not None and not entry["enabled"]:
            logger.info("Source %s is disabled in config: %s", source.name, entry.get("note") or "no note")
            continue
        active.append(source)
    return active
