from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.io.snapshotting import REQUIRED_COLUMNS
from src.ingest.sources.open_scholarships import (
    ATTRIBUTION,
    LICENSE,
    OpenScholarshipsSource,
)

FIXTURE = Path(__file__).resolve().parent / "resources" / "open_scholarships_sample.json"
FETCHED_AT = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)


class _StubClient:
    """Returns canned pages and records the params it was called with."""

    def __init__(self, pages: list[Any]) -> None:
        self._pages = pages
        self.calls: list[dict[str, Any]] = []

    def get_json(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        self.calls.append({"url": url, "params": params or {}})
        index = len(self.calls) - 1
        return self._pages[index] if index < len(self._pages) else {"results": []}


def _parse_fixture() -> list[dict[str, Any]]:
    return OpenScholarshipsSource().parse(FIXTURE.read_bytes(), fetched_at=FETCHED_AT)


def _by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["source_id"]: record for record in records}


def test_fixture_maps_to_the_normalized_schema_with_stable_ids() -> None:
    first = _parse_fixture()
    second = _parse_fixture()

    assert len(first) == 4
    assert [record["scholarship_id"] for record in first] == [
        record["scholarship_id"] for record in second
    ]
    for record in first:
        assert set(REQUIRED_COLUMNS).issubset(record.keys())
        assert len(record["scholarship_id"]) == 40
        assert record["source"] == "open_scholarships"
        assert record["trust"] == "structured_feed"


def test_award_fields_map_from_the_verified_feed_shape() -> None:
    record = _by_id(_parse_fixture())["us-american-welding-society-scholarships"]

    assert record["title"] == "American Welding Society Scholarships"
    assert record["sponsor"] == "American Welding Society Foundation"
    assert record["amount_min"] is None
    assert record["amount_max"] == 10000.0
    assert record["deadline"] == "2026-11-30"
    assert record["is_recurring"] is True
    assert record["cycle"] == {"recurring": True, "opens_month": None, "deadline_month": 11}
    assert record["eligibility_text"] == (
        "Students pursuing welding and welding-engineering education"
    )
    assert record["source_url"] == "https://www.aws.org/foundation/page/scholarships"


def test_availability_maps_to_status() -> None:
    records = _by_id(_parse_fixture())

    assert records["us-american-welding-society-scholarships"]["status"] == "open"
    assert records["us-courage-to-grow-scholarship"]["status"] == "closed"
    assert records["nv-nevada-womens-fund-scholarships"]["status"] == "upcoming"
    assert records["nv-nevada-promise"]["status"] == "unknown"


def test_provenance_carries_source_url_and_cc_by_attribution() -> None:
    record = _by_id(_parse_fixture())["nv-nevada-womens-fund-scholarships"]

    assert record["provenance"] == {
        "added_on": "2026-05-27",
        "verified_on": "2026-05-27",
        "verified_by": "Nevada Women's Fund",
        "source_kind": "structured_feed",
        "source_url": "https://www.nevadawomensfund.org/scholarships",
        "license": LICENSE,
        "attribution": ATTRIBUTION,
    }


def test_national_residency_is_not_a_state_restriction() -> None:
    records = _by_id(_parse_fixture())

    assert records["us-american-welding-society-scholarships"]["states_allowed"] == []
    assert records["nv-nevada-womens-fund-scholarships"]["states_allowed"] == ["NV"]


def test_education_level_only_set_when_the_feed_lists_one_level() -> None:
    records = _by_id(_parse_fixture())

    # ["undergraduate", "graduate", "community-college"] spans levels: stay empty.
    assert records["nv-nevada-womens-fund-scholarships"]["education_level"] is None
    assert records["nv-nevada-promise"]["education_level"] == "high school"


def test_need_basis_sets_need_based_and_other_bases_do_not() -> None:
    records = _by_id(_parse_fixture())

    assert records["nv-nevada-womens-fund-scholarships"]["need_based"] is True
    assert records["us-american-welding-society-scholarships"]["need_based"] is None
    assert records["nv-nevada-promise"]["min_gpa"] is None
    assert records["nv-nevada-womens-fund-scholarships"]["min_gpa"] == 3.0


def test_empty_response_yields_no_records() -> None:
    source = OpenScholarshipsSource()
    payload = json.dumps({"total": 0, "license": LICENSE, "results": []}).encode("utf-8")

    assert source.parse(payload, fetched_at=FETCHED_AT) == []


def test_malformed_payloads_are_logged_and_skipped() -> None:
    source = OpenScholarshipsSource()

    assert source.parse(b"{not json", fetched_at=FETCHED_AT) == []
    assert source.parse(b'"a string"', fetched_at=FETCHED_AT) == []
    assert source.parse(b'{"results": {"id": "x"}}', fetched_at=FETCHED_AT) == []
    assert source.parse(b'{"results": ["not an object"]}', fetched_at=FETCHED_AT) == []
    assert source.parse(b'{"results": [{"id": "x"}]}', fetched_at=FETCHED_AT) == []


def test_partial_records_leave_unknown_fields_empty() -> None:
    source = OpenScholarshipsSource()
    payload = json.dumps(
        {"results": [{"id": "x-1", "name": "Bare Award", "award": None, "deadline": []}]}
    ).encode("utf-8")

    record = source.parse(payload, fetched_at=FETCHED_AT)[0]

    assert record["deadline"] is None
    assert record["amount_min"] is None
    assert record["is_recurring"] is None
    assert record["status"] == "unknown"
    assert record["education_level"] is None
    assert record["provenance"]["attribution"] == ATTRIBUTION


def test_fetch_pages_with_offset_until_total_is_reached() -> None:
    client = _StubClient(
        [
            {
                "total": 3,
                "license": LICENSE,
                "attribution": ATTRIBUTION,
                "results": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
            },
            {"total": 3, "results": [{"id": "c", "name": "C"}]},
        ]
    )
    source = OpenScholarshipsSource(state="NC", page_limit=2)

    raw = source.fetch(client)
    envelope = json.loads(raw.content)

    assert [call["params"]["offset"] for call in client.calls] == [0, 2]
    assert {call["params"]["state"] for call in client.calls} == {"NC"}
    assert envelope["attribution"] == ATTRIBUTION
    assert [item["id"] for item in envelope["results"]] == ["a", "b", "c"]
    assert raw.extension == "json"


def test_fetch_stops_on_an_empty_page() -> None:
    client = _StubClient([{"total": 99, "results": []}])
    source = OpenScholarshipsSource()

    envelope = json.loads(source.fetch(client).content)

    assert len(client.calls) == 1
    assert envelope["results"] == []
