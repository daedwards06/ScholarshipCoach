from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.io.snapshotting import REQUIRED_COLUMNS
from src.ingest.sources.static_feed import StaticFeedSource

_FETCHED_AT = datetime(2026, 6, 26, 12, 0, tzinfo=UTC)
_FEED_PATH = Path(__file__).resolve().parents[1] / "data" / "static_feed" / "scholarships.json"


def _parse_feed() -> list[dict]:
    return StaticFeedSource(feed_path=_FEED_PATH).parse(
        _FEED_PATH.read_bytes(), fetched_at=_FETCHED_AT
    )


def test_static_feed_parses_all_records() -> None:
    records = _parse_feed()
    assert len(records) == 5


def test_static_feed_records_have_required_columns() -> None:
    for record in _parse_feed():
        assert set(REQUIRED_COLUMNS).issubset(record.keys()), f"Missing columns in: {record}"


def test_static_feed_ids_are_stable_sha1_hex() -> None:
    for record in _parse_feed():
        sid = record["scholarship_id"]
        assert len(sid) == 40
        assert int(sid, 16) >= 0


def test_static_feed_ids_are_deterministic() -> None:
    raw = _FEED_PATH.read_bytes()
    source = StaticFeedSource(feed_path=_FEED_PATH)
    first = [r["scholarship_id"] for r in source.parse(raw, fetched_at=_FETCHED_AT)]
    second = [r["scholarship_id"] for r in source.parse(raw, fetched_at=_FETCHED_AT)]
    assert first == second


def test_static_feed_field_population() -> None:
    records = _parse_feed()
    titles = [r["title"] for r in records]
    assert "Google Generation Scholarship" in titles

    google = next(r for r in records if "Google" in r["title"])
    assert google["sponsor"] == "Google LLC"
    assert google["amount_min"] == 10000.0
    assert google["amount_max"] == 10000.0
    assert google["deadline"] == "2027-12-01"
    assert google["source"] == "static_feed"
    assert google["eligibility_text"] is not None
    assert google["keywords"] is not None


def test_static_feed_all_have_nonzero_amount() -> None:
    for record in _parse_feed():
        assert record["amount_max"] is not None and record["amount_max"] > 0, (
            f"Expected positive amount for '{record['title']}', got {record['amount_max']}"
        )


def test_static_feed_fetch_returns_raw_response() -> None:
    source = StaticFeedSource(feed_path=_FEED_PATH)

    class _NoOpClient:
        pass

    raw = source.fetch(_NoOpClient())
    assert raw.extension == "json"
    assert len(raw.content) > 0
    assert raw.fetched_at is not None
