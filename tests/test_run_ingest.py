from __future__ import annotations

from scripts.run_ingest import _build_guardrail_warnings, _normalize_records


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
