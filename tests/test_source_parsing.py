from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.io.snapshotting import REQUIRED_COLUMNS
from src.ingest.sources.scholarship_america import ScholarshipAmericaSource


def test_scholarship_america_fixture_parses_to_schema_and_stable_ids() -> None:
    fixture_path = Path(__file__).resolve().parent / "resources" / "scholarship_america_sample.json"
    raw_content = fixture_path.read_bytes()

    source = ScholarshipAmericaSource()
    fetched_at = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)

    first = source.parse(raw_content, fetched_at=fetched_at)
    second = source.parse(raw_content, fetched_at=fetched_at)

    assert len(first) == 2
    assert [record["scholarship_id"] for record in first] == [
        record["scholarship_id"] for record in second
    ]

    for record in first:
        assert set(REQUIRED_COLUMNS).issubset(record.keys())
        assert len(record["scholarship_id"]) == 40
        assert int(record["scholarship_id"], 16) >= 0


def test_scholarship_america_fixture_parsing_is_deterministic_for_id_order() -> None:
    fixture_path = Path(__file__).resolve().parent / "resources" / "scholarship_america_sample.json"
    raw_content = fixture_path.read_bytes()

    source = ScholarshipAmericaSource()
    fetched_at = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)

    first_ids = [record["scholarship_id"] for record in source.parse(raw_content, fetched_at=fetched_at)]
    second_ids = [record["scholarship_id"] for record in source.parse(raw_content, fetched_at=fetched_at)]

    assert first_ids == second_ids
