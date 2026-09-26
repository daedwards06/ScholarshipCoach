from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from scripts.run_ingest import (
    _build_guardrail_warnings,
    _dedupe_records,
    _normalize_records,
    run_ingest,
)

SOURCE_ORDER = ["curated_catalog", "open_scholarships", "scholarship_america_live", "bold_org"]


def _dedupe_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "scholarship_id": "id",
        "source": "open_scholarships",
        "source_id": "1",
        "source_url": "https://example.org/awards/one",
        "title": "Example Award",
        "sponsor": "Example Foundation",
        "trust": "structured_feed",
        "aliases": [],
    }
    row.update(overrides)
    return row


def _dedupe(rows: list[dict[str, object]]):
    return _dedupe_records(pd.DataFrame(rows), source_order=SOURCE_ORDER)


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
    guardrail = _build_guardrail_warnings(
        prior_count=100,
        current_count=40,
        missing_title_or_source_count=3,
    )

    assert len(guardrail.warnings) == 2
    assert "dropped by more than 50%" in guardrail.warnings[0]
    assert "More than 5% of records" in guardrail.warnings[1]
    assert guardrail.blocking is True


def test_build_guardrail_warnings_ignores_non_triggering_values() -> None:
    guardrail = _build_guardrail_warnings(
        prior_count=100,
        current_count=60,
        missing_title_or_source_count=3,
    )

    assert guardrail.warnings == []
    assert guardrail.blocking is False


def test_build_guardrail_warnings_force_clears_the_block_not_the_warning() -> None:
    guardrail = _build_guardrail_warnings(
        prior_count=100,
        current_count=40,
        missing_title_or_source_count=0,
        force=True,
    )

    assert "dropped by more than 50%" in guardrail.warnings[0]
    assert guardrail.blocking is False


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


def test_only_sources_runs_the_named_connector_alone(monkeypatch, tmp_path: Path) -> None:
    sources = [_CountingSource("curated_catalog", 2), _CountingSource("scraper", 3)]
    monkeypatch.setattr("scripts.run_ingest.register_sources", lambda: list(sources))

    report = run_ingest(
        date=datetime(2026, 2, 28, tzinfo=UTC).date(),
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        report_dir=tmp_path / "reports",
        only_sources=["curated_catalog"],
    )

    assert report["sources"]["attempted"] == ["curated_catalog"]
    assert report["records"]["snapshot_total"] == 2
    assert report["config"]["only_sources"] == ["curated_catalog"]
    assert report["artifact_paths"]["snapshot"] is not None


def _run_on(
    monkeypatch,
    tmp_path: Path,
    sources: list[_CountingSource],
    *,
    day: int,
    only_sources: list[str] | None = None,
    force: bool = False,
) -> dict:
    monkeypatch.setattr("scripts.run_ingest.register_sources", lambda: list(sources))
    return run_ingest(
        date=datetime(2026, 2, day, tzinfo=UTC).date(),
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        report_dir=tmp_path / "reports",
        only_sources=only_sources,
        force=force,
    )


def test_only_sources_carries_forward_the_sources_that_did_not_run(
    monkeypatch, tmp_path: Path
) -> None:
    sources = [_CountingSource("curated_catalog", 2), _CountingSource("scraper", 3)]
    _run_on(monkeypatch, tmp_path, sources, day=27)

    report = _run_on(
        monkeypatch, tmp_path, sources, day=28, only_sources=["curated_catalog"]
    )

    assert report["sources"]["attempted"] == ["curated_catalog"]
    assert report["records"]["carried_forward"] == {"scraper": 3}
    assert report["records"]["carried_forward_total"] == 3
    assert report["records"]["snapshot_total"] == 5
    assert report["delta_counts"]["removed"] == 0

    snapshot = pd.read_parquet(report["artifact_paths"]["snapshot"])
    assert sorted(snapshot["source"].value_counts().to_dict().items()) == [
        ("curated_catalog", 2),
        ("scraper", 3),
    ]


def test_carry_forward_skips_a_source_that_is_no_longer_enabled(
    monkeypatch, tmp_path: Path
) -> None:
    _run_on(
        monkeypatch,
        tmp_path,
        [_CountingSource("curated_catalog", 4), _CountingSource("retired", 3)],
        day=27,
    )

    report = _run_on(
        monkeypatch,
        tmp_path,
        [_CountingSource("curated_catalog", 4)],
        day=28,
        only_sources=["curated_catalog"],
    )

    assert report["records"]["carried_forward"] == {}
    assert report["records"]["snapshot_total"] == 4
    assert [entry["scholarship_id"] for entry in _delta(report)["removed"]] == [
        "retired-0",
        "retired-1",
        "retired-2",
    ]


def _delta(report: dict) -> dict:
    return json.loads(Path(report["artifact_paths"]["delta"]).read_text(encoding="utf-8"))


def test_record_collapse_blocks_the_snapshot_write(monkeypatch, tmp_path: Path) -> None:
    first = _run_on(monkeypatch, tmp_path, [_CountingSource("alpha", 10)], day=27)
    prior_snapshot = Path(first["artifact_paths"]["snapshot"])

    report = _run_on(monkeypatch, tmp_path, [_CountingSource("alpha", 2)], day=28)

    assert report["artifact_paths"]["snapshot"] is None
    assert report["artifact_paths"]["delta"] is None
    assert report["artifact_notes"]["snapshot_blocked"] is True
    assert "Guardrail blocked the write" in report["artifact_notes"]["snapshot_skip_reason"]
    assert "--force" in report["artifact_notes"]["snapshot_skip_reason"]
    assert report["status"] == "partial"
    assert Path(report["artifact_paths"]["report"]).exists()
    assert len(pd.read_parquet(prior_snapshot)) == 10


def test_force_writes_the_snapshot_through_a_collapse(monkeypatch, tmp_path: Path) -> None:
    _run_on(monkeypatch, tmp_path, [_CountingSource("alpha", 10)], day=27)

    report = _run_on(
        monkeypatch, tmp_path, [_CountingSource("alpha", 2)], day=28, force=True
    )

    assert report["artifact_paths"]["snapshot"] is not None
    assert report["artifact_notes"]["snapshot_blocked"] is False
    assert report["records"]["snapshot_total"] == 2
    assert any("dropped by more than 50%" in w for w in report["guardrail_warnings"])


def test_dedupe_collapses_rows_sharing_a_url_and_keeps_the_curated_row() -> None:
    deduped, superseded = _dedupe(
        [
            _dedupe_row(
                scholarship_id="feed-dell",
                source="open_scholarships",
                source_url="https://www.dellscholars.org/scholarship/?utm_source=feed",
                title="Dell Scholars Program",
                sponsor="Michael & Susan Dell Foundation",
                trust="structured_feed",
            ),
            _dedupe_row(
                scholarship_id="catalog-dell",
                source="curated_catalog",
                source_url="https://dellscholars.org/scholarship",
                title="Dell Technologies Scholars Program",
                sponsor="Dell Technologies",
                trust="verified_local",
            ),
        ]
    )

    assert deduped["scholarship_id"].tolist() == ["catalog-dell"]
    assert deduped.loc[0, "superseded_ids"] == ["feed-dell"]
    assert superseded == {"open_scholarships": 1}


def test_dedupe_matches_on_normalized_title_and_sponsor() -> None:
    deduped, superseded = _dedupe(
        [
            _dedupe_row(
                scholarship_id="feed-row",
                source="open_scholarships",
                source_url="https://feed.example.org/listing/99",
                title="The Tar Heel STEM Scholarship!",
                sponsor="Tar Heel Foundation",
            ),
            _dedupe_row(
                scholarship_id="catalog-row",
                source="curated_catalog",
                source_url="https://sponsor.example.org/apply",
                title="Tar Heel STEM Program",
                sponsor="tar heel foundation",
                trust="verified_local",
            ),
        ]
    )

    assert deduped["scholarship_id"].tolist() == ["catalog-row"]
    assert deduped.loc[0, "superseded_ids"] == ["feed-row"]
    assert superseded == {"open_scholarships": 1}


def test_dedupe_precedence_prefers_trust_then_registry_order() -> None:
    deduped, superseded = _dedupe(
        [
            _dedupe_row(
                scholarship_id="aggregator-row",
                source="scholarship_america_live",
                trust="aggregator",
            ),
            _dedupe_row(
                scholarship_id="unverified-row",
                source="bold_org",
                trust="unverified",
            ),
            _dedupe_row(
                scholarship_id="feed-row",
                source="open_scholarships",
                trust="structured_feed",
            ),
        ]
    )

    assert deduped["scholarship_id"].tolist() == ["feed-row"]
    assert deduped.loc[0, "superseded_ids"] == ["aggregator-row", "unverified-row"]
    assert superseded == {"bold_org": 1, "scholarship_america_live": 1}


def test_dedupe_tie_on_trust_keeps_the_earlier_source_in_registry_order() -> None:
    deduped, _ = _dedupe(
        [
            _dedupe_row(
                scholarship_id="later-source",
                source="bold_org",
                trust="aggregator",
            ),
            _dedupe_row(
                scholarship_id="earlier-source",
                source="scholarship_america_live",
                trust="aggregator",
            ),
        ]
    )

    assert deduped["scholarship_id"].tolist() == ["earlier-source"]


def test_dedupe_matches_a_feed_title_named_in_curated_aliases() -> None:
    deduped, superseded = _dedupe(
        [
            _dedupe_row(
                scholarship_id="feed-row",
                source="open_scholarships",
                source_url="https://feed.example.org/listing/7",
                title="Coastal Carolina Engineering Grant",
                sponsor="Coastal Trust",
            ),
            _dedupe_row(
                scholarship_id="catalog-row",
                source="curated_catalog",
                source_url="https://sponsor.example.org/engineering",
                title="Coastal Carolina Award for Engineers",
                sponsor="Coastal Carolina Trust",
                trust="verified_local",
                aliases=["Coastal Carolina Engineering Grant"],
            ),
        ]
    )

    assert deduped["scholarship_id"].tolist() == ["catalog-row"]
    assert deduped.loc[0, "superseded_ids"] == ["feed-row"]
    assert superseded == {"open_scholarships": 1}


def test_dedupe_keeps_two_distinct_awards_from_one_sponsor() -> None:
    deduped, superseded = _dedupe(
        [
            _dedupe_row(
                scholarship_id="award-stem",
                source_url="https://sponsor.example.org/awards/stem",
                title="Acme STEM Scholarship",
                sponsor="Acme Foundation",
            ),
            _dedupe_row(
                scholarship_id="award-arts",
                source_url="https://sponsor.example.org/awards/arts",
                title="Acme Arts Scholarship",
                sponsor="Acme Foundation",
            ),
        ]
    )

    assert sorted(deduped["scholarship_id"].tolist()) == ["award-arts", "award-stem"]
    assert deduped["superseded_ids"].tolist() == [[], []]
    assert superseded == {}


def test_dedupe_does_not_merge_awards_sharing_only_a_sponsor_home_page() -> None:
    deduped, superseded = _dedupe(
        [
            _dedupe_row(
                scholarship_id="award-one",
                source_url="https://sponsor.example.org/",
                title="Acme STEM Scholarship",
                sponsor="Acme Foundation",
            ),
            _dedupe_row(
                scholarship_id="award-two",
                source_url="https://sponsor.example.org",
                title="Acme Arts Scholarship",
                sponsor="Acme Foundation",
            ),
        ]
    )

    assert len(deduped) == 2
    assert superseded == {}


def test_dedupe_keeps_sibling_awards_published_on_one_page() -> None:
    deduped, superseded = _dedupe(
        [
            _dedupe_row(
                scholarship_id="afcea-stem-major",
                source="curated_catalog",
                source_url="https://www.afcea.org/stem-majors-scholarships",
                title="AFCEA STEM Major Scholarship",
                sponsor="AFCEA Educational Foundation",
                trust="verified_local",
            ),
            _dedupe_row(
                scholarship_id="afcea-cyber-security",
                source="curated_catalog",
                source_url="https://www.afcea.org/stem-majors-scholarships",
                title="AFCEA Cyber Security Scholarship",
                sponsor="AFCEA Educational Foundation",
                trust="verified_local",
            ),
            _dedupe_row(
                scholarship_id="afcea-student-member",
                source="curated_catalog",
                source_url="https://www.afcea.org/stem-majors-scholarships",
                title="AFCEA Student Member Scholarship",
                sponsor="AFCEA Educational Foundation",
                trust="verified_local",
            ),
        ]
    )

    assert sorted(deduped["scholarship_id"].tolist()) == [
        "afcea-cyber-security",
        "afcea-stem-major",
        "afcea-student-member",
    ]
    assert deduped["superseded_ids"].tolist() == [[], [], []]
    assert superseded == {}


def test_dedupe_still_collapses_a_feed_row_onto_a_page_with_sibling_awards() -> None:
    deduped, superseded = _dedupe(
        [
            _dedupe_row(
                scholarship_id="feed-afcea",
                source="open_scholarships",
                source_url="https://afcea.org/stem-majors-scholarships/",
                title="AFCEA STEM Major Scholarship",
                sponsor="AFCEA Educational Foundation",
                trust="structured_feed",
            ),
            _dedupe_row(
                scholarship_id="afcea-stem-major",
                source="curated_catalog",
                source_url="https://www.afcea.org/stem-majors-scholarships",
                title="AFCEA STEM Major Scholarship",
                sponsor="AFCEA Educational Foundation",
                trust="verified_local",
            ),
            _dedupe_row(
                scholarship_id="afcea-cyber-security",
                source="curated_catalog",
                source_url="https://www.afcea.org/stem-majors-scholarships",
                title="AFCEA Cyber Security Scholarship",
                sponsor="AFCEA Educational Foundation",
                trust="verified_local",
            ),
        ]
    )

    assert sorted(deduped["scholarship_id"].tolist()) == [
        "afcea-cyber-security",
        "afcea-stem-major",
    ]
    assert superseded == {"open_scholarships": 1}
