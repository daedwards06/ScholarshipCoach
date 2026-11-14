from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from scripts import discover, run_ingest
from src.catalog import discovery, inbox
from src.ingest.base import RawResponse
from src.ingest.sources.open_scholarships import OpenScholarshipsSource
from src.normalize.catalog_schema import RECORDS_DIR
from src.profile.store import load_demo_profile

SAMPLE = json.loads(
    (Path(__file__).parent / "resources" / "open_scholarships_sample.json").read_text(
        encoding="utf-8"
    )
)
WELDING = "us-american-welding-society-scholarships"


class _FakeFeed(OpenScholarshipsSource):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__()
        self.payload = payload

    def fetch(self, http_client: Any) -> RawResponse:
        return RawResponse(
            content=json.dumps(self.payload).encode("utf-8"),
            extension="json",
            fetched_at=datetime.now(tz=UTC),
        )


def _payload(*, welding_deadline: str = "2099-11-30", keep: int | None = None) -> dict[str, Any]:
    payload = copy.deepcopy(SAMPLE)
    for item in payload["results"]:
        if item["id"] == WELDING:
            item["deadline"]["date"] = welding_deadline
    payload["results"] = payload["results"][:keep]
    return payload


@pytest.fixture
def feed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _FakeFeed:
    source = _FakeFeed(_payload())
    monkeypatch.setattr("scripts.run_ingest.register_sources", lambda: [source])
    monkeypatch.setattr("scripts.discover.register_sources", lambda: [source])
    monkeypatch.setattr(run_ingest, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(discover, "REPORT_DIR", tmp_path / "reports" / "discovery")
    monkeypatch.setattr(discover, "load_profile", load_demo_profile)
    monkeypatch.setattr(discovery, "STATE_PATH", tmp_path / "reports" / "discovery" / "state.json")
    monkeypatch.setattr(inbox, "INBOX_DIR", tmp_path / "inbox")
    monkeypatch.setattr(inbox, "REJECTED_DIR", tmp_path / "inbox" / "rejected")
    return source


def _records_digest() -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(RECORDS_DIR.glob("*.json"))
    }


def _last_report(tmp_path: Path) -> dict[str, Any]:
    reports = sorted((tmp_path / "reports" / "discovery").glob("discovery_*.json"))
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def test_first_run_baselines_and_a_changed_deadline_is_proposed(
    feed: _FakeFeed, tmp_path: Path
) -> None:
    before = _records_digest()

    assert discover.main([]) == 0
    first = _last_report(tmp_path)
    source = first["sources"][0]
    assert first["status"] == "completed"
    assert source["status"] == "succeeded"
    assert source["records"] == 4
    assert source["discovery"]["baseline"] is True
    assert source["discovery"]["proposals_written"] == 0
    assert first["snapshot"] is not None
    assert not list((tmp_path / "inbox").glob("*.json"))

    feed.payload = _payload(welding_deadline="2099-12-15")
    assert discover.main([]) == 0
    second = _last_report(tmp_path)["sources"][0]["discovery"]
    assert second["proposed"] == [f"feed-open-scholarships-{WELDING}"]
    assert second["left_out"]["unchanged"] == 3
    assert (tmp_path / "inbox" / f"feed-open-scholarships-{WELDING}.json").is_file()

    assert _records_digest() == before


def test_dry_run_writes_nothing(feed: _FakeFeed, tmp_path: Path, capsys) -> None:
    assert discover.main(["--dry-run"]) == 0
    assert "Dry run: nothing written." in capsys.readouterr().out
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


def test_a_source_that_returns_nothing_after_records_fails_and_keeps_its_state(
    feed: _FakeFeed, tmp_path: Path
) -> None:
    assert discover.main([]) == 0
    state_before = discovery.STATE_PATH.read_text(encoding="utf-8")

    feed.payload = _payload(keep=0)
    assert discover.main([]) == 1
    report = _last_report(tmp_path)
    source = report["sources"][0]
    assert report["status"] == "failed"
    assert source["status"] == "failed"
    assert source["zero_record_regression"] is True
    assert source["discovery"] is None
    assert discovery.STATE_PATH.read_text(encoding="utf-8") == state_before


def test_a_blocked_guardrail_exits_non_zero_and_plans_nothing(
    feed: _FakeFeed, tmp_path: Path
) -> None:
    # The guardrail compares against an earlier day's snapshot, never today's.
    run_ingest.run_ingest(date=date(2026, 1, 4), only_sources=["open_scholarships"])

    feed.payload = _payload(keep=1)
    assert discover.main([]) == 1
    report = _last_report(tmp_path)
    assert report["guardrail"]["blocked"] is True
    assert report["sources"][0]["discovery_skipped"] == "guardrail blocked the snapshot write"
    assert report["snapshot"] is None


def test_only_feed_sources_are_run(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Scraper:
        name = "some_scraper"

    monkeypatch.setattr(
        "scripts.discover.register_sources", lambda: [_Scraper(), OpenScholarshipsSource()]
    )
    assert discover.enabled_feed_sources() == ["open_scholarships"]
    monkeypatch.setattr("scripts.discover.register_sources", lambda: [_Scraper()])
    assert discover.enabled_feed_sources() == []
