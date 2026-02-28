from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.run_ingest import _build_guardrail_warnings, _normalize_records, run_ingest


def test_normalize_records_stores_timestamps_as_utc_iso() -> None:
    records = [
        {
            "scholarship_id": "abc",
            "source": "fixture",
            "source_id": "1",
            "source_url": "https://example.com",
            "title": "Scholarship",
            "first_seen_at": "2026-02-01T07:00:00-05:00",
            "last_seen_at": "2026-02-01T12:00:00Z",
        }
    ]

    normalized = _normalize_records(records)

    assert normalized.loc[0, "first_seen_at"] == "2026-02-01T12:00:00Z"
    assert normalized.loc[0, "last_seen_at"] == "2026-02-01T12:00:00Z"


def test_build_guardrail_warnings_for_drop_and_missing_fields() -> None:
    warnings = _build_guardrail_warnings(
        prior_count=100,
        current_count=40,
        missing_title_or_source_count=3,
    )

    assert len(warnings) == 2
    assert "dropped by more than 50%" in warnings[0]
    assert "More than 5% of records" in warnings[1]


def test_build_guardrail_warnings_ignores_non_triggering_values() -> None:
    warnings = _build_guardrail_warnings(
        prior_count=100,
        current_count=60,
        missing_title_or_source_count=3,
    )

    assert warnings == []


def test_run_ingest_writes_report_on_snapshot_failure(monkeypatch, tmp_path: Path) -> None:
    class _FakeSource:
        name = "fake_source"

        def fetch_records(self, http_client, *, raw_root, **kwargs):  # noqa: ANN001
            fetched_at = datetime(2026, 2, 28, 15, 0, tzinfo=UTC)
            record = {
                "scholarship_id": "abc123",
                "source": self.name,
                "source_id": "fake-1",
                "source_url": "https://example.com/scholarships/fake-1",
                "title": "Fake Scholarship",
                "first_seen_at": fetched_at,
                "last_seen_at": fetched_at,
            }
            return [record], [], {"listing_urls_processed": 1, "detail_urls_attempted": 1}

    def _raise_timeout(*args, **kwargs):  # noqa: ANN002, ANN003
        raise TimeoutError("forced timeout")

    monkeypatch.setattr("scripts.run_ingest.register_sources", lambda: [_FakeSource()])
    monkeypatch.setattr("scripts.run_ingest.build_and_write_snapshot", _raise_timeout)

    report = run_ingest(
        date=datetime(2026, 2, 28, tzinfo=UTC).date(),
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        report_dir=tmp_path / "reports",
    )

    report_path = Path(report["artifact_paths"]["report"])
    assert report_path.exists()

    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "partial"
    assert persisted["exception_summary"]["type"] == "TimeoutError"
    assert persisted["artifact_paths"]["snapshot"] is None
    assert "failed after records were normalized" in persisted["artifact_notes"]["snapshot_skip_reason"]
