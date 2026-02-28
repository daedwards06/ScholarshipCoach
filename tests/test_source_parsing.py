from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

from src.io.snapshotting import REQUIRED_COLUMNS
from src.ingest.sources.scholarship_america import ScholarshipAmericaSource
from src.ingest.sources.scholarship_america_live import ScholarshipAmericaLiveSource


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


def test_scholarship_america_live_listing_and_detail_parse_to_schema() -> None:
    browse_path = Path(__file__).resolve().parent / "resources" / "scholarship_america_browse_sample.html"
    detail_path = Path(__file__).resolve().parent / "resources" / "scholarship_america_detail_sample.html"

    browse_html = browse_path.read_text(encoding="utf-8")
    detail_html = detail_path.read_text(encoding="utf-8")

    source = ScholarshipAmericaLiveSource()
    detail_urls, category_urls, json_urls = source.parse_listing_html(
        browse_html,
        base_url="https://scholarshipamerica.org/students/browse-scholarships/",
    )

    assert "https://scholarshipamerica.org/scholarships/future-leaders-scholarship" in detail_urls
    assert "https://scholarshipamerica.org/scholarships/stem-excellence-scholarship" in detail_urls
    assert "https://scholarshipamerica.org/students/scholarships-for-high-school-seniors" in category_urls
    assert json_urls == set()

    fetched_at = datetime(2026, 2, 22, 12, 0, tzinfo=UTC)
    record = source.parse_detail_html(
        detail_html,
        detail_url="https://scholarshipamerica.org/scholarships/future-leaders-scholarship/",
        fetched_at=fetched_at,
    )

    assert record is not None
    assert set(REQUIRED_COLUMNS).issubset(record.keys())
    assert record["title"] == "Future Leaders Scholarship"
    assert record["sponsor"] == "Acme Foundation"
    assert record["deadline"] == "2026-03-15"
    assert record["amount_min"] == 1000.0
    assert record["amount_max"] == 5000.0
    assert record["education_level"]
    assert record["essay_required"] is True
    assert record["essay_prompt"] is not None
    assert record["states_allowed"] is not None


def test_scholarship_america_live_detail_ids_are_deterministic() -> None:
    detail_path = Path(__file__).resolve().parent / "resources" / "scholarship_america_detail_sample.html"
    detail_html = detail_path.read_text(encoding="utf-8")

    source = ScholarshipAmericaLiveSource()
    fetched_at = datetime(2026, 2, 22, 12, 0, tzinfo=UTC)

    first = source.parse_detail_html(
        detail_html,
        detail_url="https://scholarshipamerica.org/scholarships/future-leaders-scholarship/",
        fetched_at=fetched_at,
    )
    second = source.parse_detail_html(
        detail_html,
        detail_url="https://scholarshipamerica.org/scholarships/future-leaders-scholarship/",
        fetched_at=fetched_at,
    )

    assert first is not None
    assert second is not None
    assert first["scholarship_id"] == second["scholarship_id"]


def test_scholarship_america_live_fetch_records_preserves_sorted_order_with_concurrency(
    monkeypatch, tmp_path: Path
) -> None:
    source = ScholarshipAmericaLiveSource()
    detail_urls = [
        "https://scholarshipamerica.org/scholarships/c-scholarship",
        "https://scholarshipamerica.org/scholarships/a-scholarship",
        "https://scholarshipamerica.org/scholarships/b-scholarship",
    ]
    delays = {
        detail_urls[0]: 0.03,
        detail_urls[1]: 0.01,
        detail_urls[2]: 0.02,
    }

    class _FakeHttpClient:
        def get_text(self, url: str) -> str:
            if "browse-scholarships" in url:
                return "<html></html>"
            return f"<html><title>{url}</title></html>"

        def get_json(self, url: str):  # pragma: no cover - unused in this test
            raise AssertionError(f"Unexpected JSON fetch for {url}")

    def _fake_parse_listing_html(html: str, *, base_url: str) -> tuple[set[str], set[str], set[str]]:
        return set(detail_urls), set(), set()

    def _fake_parse_detail_html(html: str, *, detail_url: str, fetched_at: datetime):
        time.sleep(delays[detail_url])
        slug = detail_url.rstrip("/").split("/")[-1]
        return {
            "scholarship_id": slug,
            "source": source.name,
            "source_id": slug,
            "source_url": detail_url,
            "title": slug.replace("-", " ").title(),
            "sponsor": "Test Sponsor",
            "description": None,
            "eligibility_text": None,
            "deadline": None,
            "amount_min": None,
            "amount_max": None,
            "is_recurring": None,
            "states_allowed": None,
            "majors_allowed": None,
            "min_gpa": None,
            "citizenship": None,
            "education_level": None,
            "essay_required": None,
            "essay_prompt": None,
            "keywords": None,
            "first_seen_at": fetched_at,
            "last_seen_at": fetched_at,
        }

    monkeypatch.setattr(source, "parse_listing_html", _fake_parse_listing_html)
    monkeypatch.setattr(source, "parse_detail_html", _fake_parse_detail_html)

    first_records, _, _ = source.fetch_records(
        _FakeHttpClient(),
        raw_root=tmp_path / "raw",
        max_listing_pages=1,
        max_detail_pages=10,
        max_runtime_seconds=60,
        concurrency=3,
    )
    second_records, _, _ = source.fetch_records(
        _FakeHttpClient(),
        raw_root=tmp_path / "raw",
        max_listing_pages=1,
        max_detail_pages=10,
        max_runtime_seconds=60,
        concurrency=3,
    )

    expected_urls = sorted(detail_urls)
    assert [record["source_url"] for record in first_records] == expected_urls
    assert [record["source_url"] for record in second_records] == expected_urls
