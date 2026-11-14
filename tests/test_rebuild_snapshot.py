from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts import rebuild_snapshot


def _touch(path: Path, when: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        path.write_text("{}", encoding="utf-8")
    os.utime(path, (when, when))
    return path


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    records = tmp_path / "records"
    processed = tmp_path / "processed"
    records.mkdir()
    processed.mkdir()
    return records, processed


def test_stale_without_a_snapshot(dirs: tuple[Path, Path]) -> None:
    records, processed = dirs
    assert rebuild_snapshot.snapshot_is_stale(records, processed)


def test_stale_only_when_a_record_is_newer_than_the_latest_snapshot(
    dirs: tuple[Path, Path],
) -> None:
    records, processed = dirs
    record = _touch(records / "award.json", 1_000)
    _touch(records, 1_000)
    _touch(processed / "scholarships_snapshot_20261101.parquet", 500)
    latest = _touch(processed / "scholarships_snapshot_20261102.parquet", 2_000)
    assert not rebuild_snapshot.snapshot_is_stale(records, processed)

    _touch(record, 3_000)
    _touch(records, 1_000)
    assert rebuild_snapshot.snapshot_is_stale(records, processed)

    _touch(latest, 4_000)
    assert not rebuild_snapshot.snapshot_is_stale(records, processed)


def test_stale_when_a_record_was_removed(dirs: tuple[Path, Path]) -> None:
    records, processed = dirs
    _touch(records / "award.json", 1_000)
    _touch(processed / "scholarships_snapshot_20261102.parquet", 2_000)
    _touch(records, 3_000)
    assert rebuild_snapshot.snapshot_is_stale(records, processed)


def _report(snapshot: str | None, *, blocked: bool = False) -> dict:
    return {
        "artifact_paths": {"snapshot": snapshot, "delta": "changes.json", "report": "report.json"},
        "artifact_notes": {
            "snapshot_skip_reason": "Guardrail blocked the write." if blocked else None,
            "snapshot_blocked": blocked,
        },
        "delta_counts": {"added": 1, "removed": 0, "changed": 0},
    }


def test_if_stale_does_nothing_when_the_snapshot_is_current(monkeypatch, capsys) -> None:
    monkeypatch.setattr(rebuild_snapshot, "snapshot_is_stale", lambda: False)

    def fail() -> dict:
        raise AssertionError("rebuilt a current snapshot")

    monkeypatch.setattr(rebuild_snapshot, "rebuild_curated_snapshot", fail)

    assert rebuild_snapshot.main(["--if-stale"]) == 0
    assert "nothing to do" in capsys.readouterr().out


def test_if_stale_rebuilds_a_stale_snapshot(monkeypatch) -> None:
    monkeypatch.setattr(rebuild_snapshot, "snapshot_is_stale", lambda: True)
    monkeypatch.setattr(
        rebuild_snapshot, "rebuild_curated_snapshot", lambda: _report("snapshot.parquet")
    )

    assert rebuild_snapshot.main(["--if-stale"]) == 0


def test_a_blocked_guardrail_exits_non_zero(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        rebuild_snapshot, "rebuild_curated_snapshot", lambda: _report(None, blocked=True)
    )

    assert rebuild_snapshot.main([]) == 1
    assert "Snapshot NOT written: Guardrail blocked the write." in capsys.readouterr().out


def test_rebuild_runs_the_curated_source_alone(monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        rebuild_snapshot.run_ingest, "run_ingest", lambda **kwargs: calls.append(kwargs) or {}
    )

    rebuild_snapshot.rebuild_curated_snapshot()

    assert calls == [{"date": None, "only_sources": ["curated_catalog"]}]
