from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from scripts.verify_catalog import main as verify_main
from src.catalog.inbox import list_proposals
from src.catalog.verify import (
    OUTCOME_BLOCKED,
    OUTCOME_CHANGED,
    OUTCOME_DEAD_LINK,
    OUTCOME_ERROR,
    OUTCOME_UNCHANGED,
    content_hash,
    load_prior_hashes,
    parse_page_status,
    verify_catalog,
)
from src.ingest.extract_common import html_to_text

_TODAY = date(2026, 9, 12)


def _page(
    *,
    deadline: str = "March 1, 2027",
    amount: str = "$2,500",
    status: str = "Open",
) -> str:
    return f"""
    <html><head><meta property="og:title" content="Test Foundation Award" /></head>
    <body>
      <h1>Test Foundation Award</h1>
      <p>Status: {status}</p>
      <p>Deadline: {deadline}</p>
      <p>Amount: {amount}</p>
      <p>Applicants submit one essay and an official transcript.</p>
    </body></html>
    """


class _StubClient:
    """Serves canned HTML per URL; a mapped exception stands in for a bad fetch."""

    def __init__(self, pages: dict[str, str | Exception]) -> None:
        self.pages = pages
        self.requested: list[str] = []

    def get_text(self, url: str) -> str:
        self.requested.append(url)
        page = self.pages[url]
        if isinstance(page, Exception):
            raise page
        return page


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _HttpError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"{status_code} Client Error")
        self.response = _Response(status_code)


def _record(**overrides: object) -> dict:
    record = {
        "catalog_id": "test-foundation-award",
        "title": "Test Foundation Award",
        "sponsor": "Test Foundation",
        "source_url": "https://example.invalid/award",
        "deadline": "2027-03-01",
        "amount_min": 2500.0,
        "amount_max": 2500.0,
        "status": "open",
        "requirements": {"essay": True, "transcript": True},
        "trust": "verified_local",
        "provenance": {
            "added_on": "2024-01-01",
            "verified_on": None,
            "verified_by": None,
            "source_kind": "sponsor_site",
        },
    }
    record.update(overrides)
    return record


@pytest.fixture
def catalog_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    records = tmp_path / "records"
    inbox = tmp_path / "inbox"
    reports = tmp_path / "reports"
    records.mkdir()
    inbox.mkdir()
    reports.mkdir()
    return records, inbox, reports


def _write_record(records_dir: Path, record: dict) -> Path:
    path = records_dir / f"{record['catalog_id']}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def _run(
    catalog_dirs: tuple[Path, Path, Path],
    client: _StubClient,
    **kwargs: object,
) -> object:
    records, inbox, reports = catalog_dirs
    return verify_catalog(
        records_dir=records,
        inbox_dir=inbox,
        reports_dir=reports,
        client=client,
        today=_TODAY,
        since_days=0,
        **kwargs,
    )


def test_unchanged_page_stamps_verified_on_and_queues_nothing(
    catalog_dirs: tuple[Path, Path, Path],
) -> None:
    records, inbox, _ = catalog_dirs
    path = _write_record(records, _record())
    client = _StubClient({"https://example.invalid/award": _page()})

    report = _run(catalog_dirs, client)

    assert [result.outcome for result in report.results] == [OUTCOME_UNCHANGED]
    assert list_proposals(inbox) == []
    provenance = json.loads(path.read_text(encoding="utf-8"))["provenance"]
    assert provenance["verified_on"] == "2026-09-12"
    assert provenance["verified_by"] == "verify_catalog"


def test_changed_deadline_queues_a_reverify_proposal(
    catalog_dirs: tuple[Path, Path, Path],
) -> None:
    records, inbox, _ = catalog_dirs
    path = _write_record(records, _record())
    client = _StubClient({"https://example.invalid/award": _page(deadline="April 15, 2027")})

    report = _run(catalog_dirs, client)

    result = report.results[0]
    assert result.outcome == OUTCOME_CHANGED
    assert result.changes["deadline"] == {"old": "2027-03-01", "new": "2027-04-15"}
    assert result.proposal_id == "reverify-test-foundation-award"

    proposals = list_proposals(inbox)
    assert len(proposals) == 1
    assert proposals[0].kind == "reverify"
    assert proposals[0].record["deadline"] == "2027-04-15"
    assert proposals[0].diff["deadline"] == {"old": "2027-03-01", "new": "2027-04-15"}

    unchanged = json.loads(path.read_text(encoding="utf-8"))
    assert unchanged["deadline"] == "2027-03-01"
    assert unchanged["provenance"]["verified_on"] is None


def test_verified_on_stamp_leaves_the_hand_formatted_file_otherwise_intact(
    catalog_dirs: tuple[Path, Path, Path],
) -> None:
    records, _, _ = catalog_dirs
    path = records / "test-foundation-award.json"
    original = """{
  "catalog_id": "test-foundation-award",
  "title": "Test Foundation Award",
  "source_url": "https://example.invalid/award",
  "deadline": "2027-03-01",
  "amount_min": 2500.0,
  "amount_max": 2500.0,
  "status": "open",
  "min_test_scores": { "sat": null, "act": null },
  "trust": "verified_local",
  "provenance": {
    "added_on": "2024-01-01",
    "verified_on": null,
    "verified_by": null,
    "source_kind": "sponsor_site"
  }
}
"""
    path.write_text(original, encoding="utf-8")
    client = _StubClient({"https://example.invalid/award": _page()})

    _run(catalog_dirs, client)

    patched = path.read_text(encoding="utf-8")
    assert patched == original.replace('"verified_on": null', '"verified_on": "2026-09-12"').replace(
        '"verified_by": null', '"verified_by": "verify_catalog"'
    )
    assert '"min_test_scores": { "sat": null, "act": null },' in patched


def test_unlabeled_dollar_figures_are_not_an_amount_change(
    catalog_dirs: tuple[Path, Path, Path],
) -> None:
    records, inbox, _ = catalog_dirs
    _write_record(records, _record())
    page = """
    <html><body>
      <h1>Test Foundation Award</h1>
      <p>Deadline: March 1, 2027</p>
      <p>The foundation has given away more than $450,000 since 1998.</p>
    </body></html>
    """
    client = _StubClient({"https://example.invalid/award": page})

    report = _run(catalog_dirs, client)

    assert report.results[0].outcome == OUTCOME_UNCHANGED
    assert list_proposals(inbox) == []


def test_a_newly_announced_deadline_fills_a_null(
    catalog_dirs: tuple[Path, Path, Path],
) -> None:
    records, inbox, _ = catalog_dirs
    _write_record(records, _record(deadline=None))
    client = _StubClient({"https://example.invalid/award": _page()})

    report = _run(catalog_dirs, client)

    assert report.results[0].changes["deadline"] == {"old": None, "new": "2027-03-01"}
    assert list_proposals(inbox)[0].record["deadline"] == "2027-03-01"


@pytest.mark.parametrize("status", [404, 410])
def test_dead_link_proposes_status_unknown(
    catalog_dirs: tuple[Path, Path, Path], status: int
) -> None:
    records, inbox, _ = catalog_dirs
    _write_record(records, _record())
    client = _StubClient({"https://example.invalid/award": _HttpError(status)})

    report = _run(catalog_dirs, client)

    result = report.results[0]
    assert result.outcome == OUTCOME_DEAD_LINK
    assert result.http_status == status
    assert result.check_by_hand is False

    proposals = list_proposals(inbox)
    assert len(proposals) == 1
    assert proposals[0].record["status"] == "unknown"
    assert proposals[0].record["deadline"] == "2027-03-01"


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_a_refused_fetch_is_blocked_and_proposes_nothing(
    catalog_dirs: tuple[Path, Path, Path], status: int
) -> None:
    records, inbox, _ = catalog_dirs
    path = _write_record(records, _record())
    client = _StubClient({"https://example.invalid/award": _HttpError(status)})

    report = _run(catalog_dirs, client)

    result = report.results[0]
    assert result.outcome == OUTCOME_BLOCKED
    assert result.http_status == status
    assert result.check_by_hand is True
    assert list_proposals(inbox) == []
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "open"
    assert report.counts[OUTCOME_BLOCKED] == 1
    assert [blocked.source_url for blocked in report.blocked] == [
        "https://example.invalid/award"
    ]


def test_other_client_errors_are_an_error_not_a_dead_link(
    catalog_dirs: tuple[Path, Path, Path],
) -> None:
    records, inbox, _ = catalog_dirs
    _write_record(records, _record())
    client = _StubClient({"https://example.invalid/award": _HttpError(400)})

    report = _run(catalog_dirs, client)

    assert report.results[0].outcome == OUTCOME_ERROR
    assert report.results[0].check_by_hand is False
    assert list_proposals(inbox) == []


def test_a_gone_page_after_a_blocked_run_still_proposes(
    catalog_dirs: tuple[Path, Path, Path],
) -> None:
    records, inbox, _ = catalog_dirs
    _write_record(records, _record())

    blocked = _run(catalog_dirs, _StubClient({"https://example.invalid/award": _HttpError(403)}))
    assert blocked.results[0].outcome == OUTCOME_BLOCKED
    assert list_proposals(inbox) == []

    gone = _run(catalog_dirs, _StubClient({"https://example.invalid/award": _HttpError(404)}))

    assert gone.results[0].outcome == OUTCOME_DEAD_LINK
    proposals = list_proposals(inbox)
    assert len(proposals) == 1
    assert proposals[0].record["status"] == "unknown"


def test_unreachable_host_is_an_error_not_a_dead_link(
    catalog_dirs: tuple[Path, Path, Path],
) -> None:
    records, inbox, _ = catalog_dirs
    _write_record(records, _record())
    client = _StubClient({"https://example.invalid/award": RuntimeError("connection reset")})

    report = _run(catalog_dirs, client)

    assert report.results[0].outcome == OUTCOME_ERROR
    assert list_proposals(inbox) == []


def test_closed_page_changes_status(catalog_dirs: tuple[Path, Path, Path]) -> None:
    records, inbox, _ = catalog_dirs
    _write_record(records, _record())
    client = _StubClient({"https://example.invalid/award": _page(status="Closed")})

    report = _run(catalog_dirs, client)

    assert report.results[0].changes["status"] == {"old": "open", "new": "closed"}
    assert list_proposals(inbox)[0].record["status"] == "closed"


def test_contradicted_requirement_flag_is_a_change(
    catalog_dirs: tuple[Path, Path, Path],
) -> None:
    records, _, _ = catalog_dirs
    _write_record(records, _record(requirements={"essay": False, "transcript": True}))
    client = _StubClient({"https://example.invalid/award": _page()})

    report = _run(catalog_dirs, client)

    requirements = report.results[0].changes["requirements"]
    assert requirements["old"]["essay"] is False
    assert requirements["new"]["essay"] is True


def test_requirements_the_record_leaves_null_are_left_alone(
    catalog_dirs: tuple[Path, Path, Path],
) -> None:
    records, _, _ = catalog_dirs
    _write_record(records, _record(requirements={"essay": None, "transcript": None}))
    client = _StubClient({"https://example.invalid/award": _page()})

    report = _run(catalog_dirs, client)

    assert report.results[0].outcome == OUTCOME_UNCHANGED


def test_since_days_skips_recently_verified_records(
    catalog_dirs: tuple[Path, Path, Path],
) -> None:
    records, _, reports = catalog_dirs
    _write_record(
        records,
        _record(
            provenance={
                "added_on": "2024-01-01",
                "verified_on": "2026-08-01",
                "verified_by": "parent",
                "source_kind": "sponsor_site",
            }
        ),
    )
    client = _StubClient({"https://example.invalid/award": _page()})

    report = verify_catalog(
        records_dir=records,
        reports_dir=reports,
        client=client,
        today=_TODAY,
        since_days=365,
    )

    assert report.records_due == 0
    assert client.requested == []


def test_max_records_caps_the_pass_oldest_first(
    catalog_dirs: tuple[Path, Path, Path],
) -> None:
    records, _, _ = catalog_dirs
    _write_record(records, _record())
    _write_record(
        records,
        _record(
            catalog_id="second-award",
            source_url="https://example.invalid/second",
            provenance={
                "added_on": "2024-01-01",
                "verified_on": "2025-01-01",
                "verified_by": "parent",
                "source_kind": "sponsor_site",
            },
        ),
    )
    client = _StubClient(
        {
            "https://example.invalid/award": _page(),
            "https://example.invalid/second": _page(),
        }
    )

    report = _run(catalog_dirs, client, max_records=1)

    assert report.records_due == 2
    assert client.requested == ["https://example.invalid/award"]
    assert [result.catalog_id for result in report.results] == ["test-foundation-award"]


def test_report_records_content_hashes_and_next_run_reads_them(
    catalog_dirs: tuple[Path, Path, Path],
) -> None:
    records, _, reports = catalog_dirs
    _write_record(records, _record())
    client = _StubClient({"https://example.invalid/award": _page()})

    first = _run(catalog_dirs, client)
    assert first.report_path is not None and first.report_path.is_file()
    digest = first.results[0].content_hash
    assert digest == content_hash(html_to_text(_page()))
    assert first.results[0].content_changed is None
    assert load_prior_hashes(reports) == {"test-foundation-award": digest}

    second = _run(catalog_dirs, _StubClient({"https://example.invalid/award": _page()}))
    assert second.results[0].content_changed is False


def test_parse_page_status_ignores_a_bare_open_in_body_copy() -> None:
    assert parse_page_status("This award is open to US citizens nationwide.") is None
    assert parse_page_status("Status: Closed") == "closed"
    assert parse_page_status("Applications are now closed for this cycle.") == "closed"
    assert parse_page_status("We are now accepting applications.") == "open"


def test_cli_reports_and_exits_zero(
    catalog_dirs: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records, inbox, reports = catalog_dirs
    _write_record(records, _record())
    client = _StubClient({"https://example.invalid/award": _page(deadline="April 15, 2027")})
    monkeypatch.setattr(
        "scripts.verify_catalog.verify_catalog",
        lambda **kwargs: verify_catalog(**{**kwargs, "client": client, "today": _TODAY}),
    )

    exit_code = verify_main(
        [
            "--records-dir",
            str(records),
            "--inbox-dir",
            str(inbox),
            "--reports-dir",
            str(reports),
            "--since-days",
            "0",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "changed=1" in output
    assert "reverify-test-foundation-award" in output


def test_cli_lists_blocked_urls_to_open_by_hand(
    catalog_dirs: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records, inbox, reports = catalog_dirs
    _write_record(records, _record())
    client = _StubClient({"https://example.invalid/award": _HttpError(403)})
    monkeypatch.setattr(
        "scripts.verify_catalog.verify_catalog",
        lambda **kwargs: verify_catalog(**{**kwargs, "client": client, "today": _TODAY}),
    )

    exit_code = verify_main(
        [
            "--records-dir",
            str(records),
            "--inbox-dir",
            str(inbox),
            "--reports-dir",
            str(reports),
            "--since-days",
            "0",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "blocked=1" in output
    assert "dead_link=0" in output
    assert "check by hand" in output
    assert "HTTP 403  https://example.invalid/award" in output
    assert list_proposals(inbox) == []


def test_cli_errors_on_an_empty_catalog(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "records"
    empty.mkdir()

    exit_code = verify_main(
        ["--records-dir", str(empty), "--reports-dir", str(tmp_path / "reports")]
    )

    assert exit_code == 1
    assert "no catalog records found" in capsys.readouterr().out
