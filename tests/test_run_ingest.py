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


class _CountingSource:
    """Minimal connector returning ``count`` synthetic records."""

    def __init__(self, name: str, count: int) -> None:
        self.name = name
        self._count = count

    def fetch_records(self, http_client, *, raw_root, **kwargs):  # noqa: ANN001
        fetched_at = datetime(2026, 2, 28, 15, 0, tzinfo=UTC)
        records = [
            {
                "scholarship_id": f"{self.name}-{index}",
                "source": self.name,
                "source_id": f"{self.name}-{index}",
                "source_url": f"https://example.com/scholarships/{self.name}-{index}",
                "title": f"{self.name} Scholarship {index}",
                "first_seen_at": fetched_at,
                "last_seen_at": fetched_at,
            }
            for index in range(self._count)
        ]
        return records, [], {}


def _run(monkeypatch, tmp_path: Path, sources: list[_CountingSource]) -> dict:
    monkeypatch.setattr("scripts.run_ingest.register_sources", lambda: list(sources))
    return run_ingest(
        date=datetime(2026, 2, 28, tzinfo=UTC).date(),
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        report_dir=tmp_path / "reports",
    )


def _source_detail(report: dict, name: str) -> dict:
    return next(item for item in report["sources"]["details"] if item["source"] == name)


def test_first_run_health_reports_no_prior_and_no_regression(monkeypatch, tmp_path: Path) -> None:
    report = _run(monkeypatch, tmp_path, [_CountingSource("alpha", 2)])

    health = _source_detail(report, "alpha")["health"]
    assert health == {
        "records_this_run": 2,
        "records_prior_run": None,
        "zero_record_regression": False,
    }
    assert report["sources"]["zero_record_regressions"] == []


def test_zero_record_regression_fails_the_source_and_the_run(monkeypatch, tmp_path: Path) -> None:
    _run(monkeypatch, tmp_path, [_CountingSource("alpha", 2), _CountingSource("beta", 3)])
    report = _run(monkeypatch, tmp_path, [_CountingSource("alpha", 2), _CountingSource("beta", 0)])

    beta = _source_detail(report, "beta")
    assert beta["health"] == {
        "records_this_run": 0,
        "records_prior_run": 3,
        "zero_record_regression": True,
    }
    assert beta["status"] == "failed"
    assert beta["error"] == "zero_records"
    assert report["sources"]["zero_record_regressions"] == ["beta"]
    assert report["sources"]["failed"] == ["beta"]
    assert report["status"] == "partial"
    assert any("returned 0 records" in warning for warning in report["guardrail_warnings"])


def test_zero_records_without_prior_records_is_not_a_regression(monkeypatch, tmp_path: Path) -> None:
    _run(monkeypatch, tmp_path, [_CountingSource("alpha", 2), _CountingSource("beta", 0)])
    report = _run(monkeypatch, tmp_path, [_CountingSource("alpha", 2), _CountingSource("beta", 0)])

    beta = _source_detail(report, "beta")
    assert beta["health"]["zero_record_regression"] is False
    assert beta["status"] == "succeeded"
    assert report["status"] == "success"


def test_report_lists_disabled_sources(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "scripts.run_ingest.disabled_sources",
        lambda: [{"source": "bold_org", "note": "returns 0 records"}],
    )
    report = _run(monkeypatch, tmp_path, [_CountingSource("alpha", 1)])

    assert report["sources"]["disabled"] == [{"source": "bold_org", "note": "returns 0 records"}]
    assert report["sources"]["disabled_count"] == 1
    assert "bold_org" not in report["sources"]["attempted"]
