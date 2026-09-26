from __future__ import annotations

import json
from pathlib import Path

from src.ingest.registry import (
    disabled_sources,
    load_source_settings,
    register_sources,
)


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_disabled_source_is_skipped(tmp_path: Path) -> None:
    config = _write_config(tmp_path, {"sources": {"bold_org": {"enabled": False, "note": "dead"}}})

    names = [source.name for source in register_sources(config)]

    assert "bold_org" not in names
    assert "curated_catalog" in names
    assert "open_scholarships" in names
    assert "scholarship_america" in names


def test_source_absent_from_config_stays_enabled(tmp_path: Path) -> None:
    config = _write_config(tmp_path, {"sources": {}})

    names = [source.name for source in register_sources(config)]

    assert "bold_org" in names


def test_malformed_config_enables_every_source(tmp_path: Path) -> None:
    path = tmp_path / "sources.json"
    path.write_text("{not json", encoding="utf-8")

    assert load_source_settings(path) == {}
    assert len(register_sources(path)) == 4


def test_disabled_sources_reports_the_note(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path, {"sources": {"bold_org": {"enabled": False, "note": "rewrite pending"}}}
    )

    assert disabled_sources(config) == [{"source": "bold_org", "note": "rewrite pending"}]


def test_shipped_config_disables_the_scrapers() -> None:
    assert [entry["source"] for entry in disabled_sources()] == ["scholarship_america", "bold_org"]
    assert [source.name for source in register_sources()] == ["curated_catalog", "open_scholarships"]
