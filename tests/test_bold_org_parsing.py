from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.io.snapshotting import REQUIRED_COLUMNS
from src.ingest.sources.bold_org import BoldOrgSource


_FETCHED_AT = datetime(2026, 6, 26, 12, 0, tzinfo=UTC)
_FIXTURE = Path(__file__).resolve().parent / "resources" / "bold_org_listing_sample.html"


def _parse_fixture() -> list[dict]:
    raw_content = _FIXTURE.read_bytes()
    return BoldOrgSource().parse(raw_content, fetched_at=_FETCHED_AT)


def test_bold_org_fixture_parses_expected_count() -> None:
    records = _parse_fixture()
    assert len(records) == 3


def test_bold_org_records_have_required_columns() -> None:
    for record in _parse_fixture():
        assert set(REQUIRED_COLUMNS).issubset(record.keys()), f"Missing columns in: {record}"


def test_bold_org_ids_are_stable_sha1_hex() -> None:
    records = _parse_fixture()
    for record in records:
        sid = record["scholarship_id"]
        assert len(sid) == 40
        assert int(sid, 16) >= 0


def test_bold_org_ids_are_deterministic() -> None:
    raw_content = _FIXTURE.read_bytes()
    source = BoldOrgSource()
    first = [r["scholarship_id"] for r in source.parse(raw_content, fetched_at=_FETCHED_AT)]
    second = [r["scholarship_id"] for r in source.parse(raw_content, fetched_at=_FETCHED_AT)]
    assert first == second


def test_bold_org_field_population() -> None:
    records = _parse_fixture()
    titles = [r["title"] for r in records]
    assert "National STEM Leadership Scholarship" in titles
    assert "First Generation College Student Scholarship" in titles
    assert "Computer Science Innovation Award" in titles

    stem = next(r for r in records if "STEM" in r["title"])
    assert stem["sponsor"] == "STEM Leaders Foundation"
    assert stem["amount_min"] == 5000.0
    assert stem["amount_max"] == 5000.0
    assert stem["deadline"] == "2027-03-15"
    assert stem["source"] == "bold_org"
    assert stem["eligibility_text"] is not None
    assert stem["keywords"] is not None


def test_bold_org_source_url_populated() -> None:
    for record in _parse_fixture():
        assert record["source_url"].startswith("https://bold.org/"), (
            f"Expected bold.org URL, got: {record['source_url']}"
        )
