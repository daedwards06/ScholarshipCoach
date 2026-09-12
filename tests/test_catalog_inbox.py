from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from scripts.catalog_inbox import main as inbox_main
from src.catalog.inbox import (
    INBOX_DIR,
    ProposalError,
    confirm,
    list_proposals,
    load_proposal,
    propose,
    reject,
)
from src.ingest.sources.curated_catalog import CuratedCatalogSource
from src.normalize.catalog_schema import RECORDS_DIR

_FETCHED_AT = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)


def _valid_record(**overrides: object) -> dict:
    record = {
        "catalog_id": "test-foundation-award",
        "title": "Test Foundation Award",
        "sponsor": "Test Foundation",
        "source_url": "https://example.invalid/scholarship",
        "description": "An award used by the inbox tests.",
        "deadline": "2027-03-01",
        "status": "open",
        "trust": "verified_local",
        "provenance": {
            "added_on": "2026-09-12",
            "verified_on": "2026-09-12",
            "verified_by": "parent",
            "source_kind": "community_foundation",
        },
    }
    record.update(overrides)
    return record


@pytest.fixture
def catalog_dirs(tmp_path: Path) -> tuple[Path, Path]:
    inbox = tmp_path / "inbox"
    records = tmp_path / "records"
    inbox.mkdir()
    records.mkdir()
    return inbox, records


def test_propose_writes_a_proposal_file(catalog_dirs: tuple[Path, Path]) -> None:
    inbox, records = catalog_dirs
    proposal = propose(
        _valid_record(),
        kind="prefill",
        notes="pasted from the foundation page",
        created_on=date(2026, 9, 12),
        inbox_dir=inbox,
        records_dir=records,
    )

    assert proposal.proposal_id == "prefill-test-foundation-award"
    assert proposal.path == inbox / "prefill-test-foundation-award.json"

    payload = json.loads(proposal.path.read_text(encoding="utf-8"))
    assert payload["proposal"]["kind"] == "prefill"
    assert payload["proposal"]["created_on"] == "2026-09-12"
    assert payload["proposal"]["notes"] == "pasted from the foundation page"
    assert payload["proposal"]["diff"] == {}
    assert payload["record"]["title"] == "Test Foundation Award"


def test_propose_confirm_round_trip(catalog_dirs: tuple[Path, Path]) -> None:
    inbox, records = catalog_dirs
    proposal = propose(_valid_record(), kind="manual", inbox_dir=inbox, records_dir=records)

    written = confirm(proposal.proposal_id, inbox_dir=inbox, records_dir=records)

    assert written == records / "test-foundation-award.json"
    assert json.loads(written.read_text(encoding="utf-8")) == _valid_record()
    assert not proposal.path.exists()
    assert list_proposals(inbox) == []


def test_confirm_applies_edits(catalog_dirs: tuple[Path, Path]) -> None:
    inbox, records = catalog_dirs
    # Automation could not read a deadline; the person supplies it at confirm time.
    propose(
        _valid_record(deadline=None),
        kind="prefill",
        proposal_id="prefill-needs-a-deadline",
        inbox_dir=inbox,
        records_dir=records,
    )

    written = confirm(
        "prefill-needs-a-deadline",
        {"deadline": "2027-04-15", "min_gpa": 3.0},
        inbox_dir=inbox,
        records_dir=records,
    )

    record = json.loads(written.read_text(encoding="utf-8"))
    assert record["deadline"] == "2027-04-15"
    assert record["min_gpa"] == 3.0


def test_confirm_rejects_a_schema_invalid_record(catalog_dirs: tuple[Path, Path]) -> None:
    inbox, records = catalog_dirs
    proposal = propose(
        _valid_record(status="maybe-open"),
        kind="feed",
        inbox_dir=inbox,
        records_dir=records,
    )

    with pytest.raises(ProposalError, match="not a valid catalog record"):
        confirm(proposal.proposal_id, inbox_dir=inbox, records_dir=records)

    assert proposal.path.exists(), "an invalid proposal must stay in the queue"
    assert list(records.glob("*.json")) == []


def test_confirm_requires_a_catalog_id(catalog_dirs: tuple[Path, Path]) -> None:
    inbox, records = catalog_dirs
    record = _valid_record()
    record.pop("catalog_id")
    propose(record, kind="prefill", inbox_dir=inbox, records_dir=records)

    with pytest.raises(ProposalError, match="no catalog_id"):
        confirm("prefill-test-foundation-award", inbox_dir=inbox, records_dir=records)


def test_reject_archives_the_proposal_with_its_reason(catalog_dirs: tuple[Path, Path]) -> None:
    inbox, records = catalog_dirs
    proposal = propose(_valid_record(), kind="feed", inbox_dir=inbox, records_dir=records)

    archived = reject(proposal.proposal_id, "  duplicate of an existing record  ", inbox_dir=inbox)

    assert archived == inbox / "rejected" / "feed-test-foundation-award.json"
    payload = json.loads(archived.read_text(encoding="utf-8"))
    assert payload["proposal"]["reason"] == "duplicate of an existing record"
    assert payload["proposal"]["rejected_on"] == date.today().isoformat()
    assert not proposal.path.exists()
    assert list_proposals(inbox) == []


def test_reject_needs_a_reason(catalog_dirs: tuple[Path, Path]) -> None:
    inbox, records = catalog_dirs
    proposal = propose(_valid_record(), kind="feed", inbox_dir=inbox, records_dir=records)

    with pytest.raises(ProposalError, match="needs a reason"):
        reject(proposal.proposal_id, "   ", inbox_dir=inbox)
    assert proposal.path.exists()


def test_proposal_diffs_against_an_existing_record(catalog_dirs: tuple[Path, Path]) -> None:
    inbox, records = catalog_dirs
    (records / "test-foundation-award.json").write_text(
        json.dumps(_valid_record()), encoding="utf-8"
    )

    proposal = propose(
        {"catalog_id": "test-foundation-award", "deadline": "2028-03-01", "status": "open"},
        kind="reverify",
        inbox_dir=inbox,
        records_dir=records,
    )

    assert proposal.diff == {"deadline": {"old": "2027-03-01", "new": "2028-03-01"}}


def test_confirming_a_reverify_proposal_needs_the_full_record(
    catalog_dirs: tuple[Path, Path],
) -> None:
    inbox, records = catalog_dirs
    (records / "test-foundation-award.json").write_text(
        json.dumps(_valid_record()), encoding="utf-8"
    )
    proposal = propose(
        {"catalog_id": "test-foundation-award", "deadline": "2028-03-01"},
        kind="reverify",
        inbox_dir=inbox,
        records_dir=records,
    )

    with pytest.raises(ProposalError, match="not a valid catalog record"):
        confirm(proposal.proposal_id, inbox_dir=inbox, records_dir=records)

    written = confirm(
        proposal.proposal_id,
        _valid_record(deadline="2028-03-01"),
        inbox_dir=inbox,
        records_dir=records,
    )
    assert json.loads(written.read_text(encoding="utf-8"))["deadline"] == "2028-03-01"


def test_propose_rejects_an_unknown_kind(catalog_dirs: tuple[Path, Path]) -> None:
    inbox, records = catalog_dirs
    with pytest.raises(ProposalError, match="unknown proposal kind"):
        propose(_valid_record(), kind="scraped", inbox_dir=inbox, records_dir=records)


def test_proposal_ids_are_confined_to_the_inbox(catalog_dirs: tuple[Path, Path]) -> None:
    inbox, records = catalog_dirs
    with pytest.raises(ProposalError, match="not a lowercase slug"):
        propose(
            _valid_record(),
            kind="manual",
            proposal_id="../records/test-foundation-award",
            inbox_dir=inbox,
            records_dir=records,
        )
    with pytest.raises(ProposalError, match="not a lowercase slug"):
        load_proposal("../../etc/passwd", inbox)


def test_reproposing_an_award_replaces_its_pending_proposal(
    catalog_dirs: tuple[Path, Path],
) -> None:
    inbox, records = catalog_dirs
    propose(_valid_record(deadline="2027-03-01"), kind="feed", inbox_dir=inbox, records_dir=records)
    propose(_valid_record(deadline="2027-05-01"), kind="feed", inbox_dir=inbox, records_dir=records)

    proposals = list_proposals(inbox)
    assert len(proposals) == 1
    assert proposals[0].record["deadline"] == "2027-05-01"


def test_list_proposals_skips_unreadable_files(catalog_dirs: tuple[Path, Path]) -> None:
    inbox, records = catalog_dirs
    propose(_valid_record(), kind="manual", inbox_dir=inbox, records_dir=records)
    (inbox / "broken.json").write_text("{not json", encoding="utf-8")
    (inbox / "wrong-shape.json").write_text('{"record": {}}', encoding="utf-8")

    assert [p.proposal_id for p in list_proposals(inbox)] == ["manual-test-foundation-award"]


def test_inbox_is_outside_the_records_directory() -> None:
    assert INBOX_DIR.parent == RECORDS_DIR.parent
    assert RECORDS_DIR not in INBOX_DIR.parents


def test_catalog_source_ignores_the_inbox(catalog_dirs: tuple[Path, Path]) -> None:
    inbox, records = catalog_dirs
    (records / "test-foundation-award.json").write_text(
        json.dumps(_valid_record()), encoding="utf-8"
    )
    propose(
        _valid_record(catalog_id="proposed-award", title="Proposed Award"),
        kind="feed",
        inbox_dir=inbox,
        records_dir=records,
    )

    source = CuratedCatalogSource(records_dir=records)
    parsed = source.parse(source.fetch(None).content, fetched_at=_FETCHED_AT)

    assert [record["catalog_id"] for record in parsed] == ["test-foundation-award"]


def test_cli_list_show_confirm_and_reject(
    catalog_dirs: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    inbox, records = catalog_dirs
    dirs = ["--inbox-dir", str(inbox), "--records-dir", str(records)]
    propose(
        _valid_record(status="unknown"),
        kind="prefill",
        inbox_dir=inbox,
        records_dir=records,
    )
    propose(
        _valid_record(catalog_id="unwanted-award", title="Unwanted Award"),
        kind="feed",
        inbox_dir=inbox,
        records_dir=records,
    )

    assert inbox_main([*dirs, "list"]) == 0
    listing = capsys.readouterr().out
    assert "2 proposal(s)" in listing
    assert "prefill-test-foundation-award" in listing

    assert inbox_main([*dirs, "show", "prefill-test-foundation-award"]) == 0
    assert json.loads(capsys.readouterr().out)["record"]["status"] == "unknown"

    assert (
        inbox_main(
            [
                *dirs,
                "confirm",
                "prefill-test-foundation-award",
                "--set",
                "status=open",
                "--set",
                "min_gpa=3.25",
            ]
        )
        == 0
    )
    capsys.readouterr()
    confirmed = json.loads((records / "test-foundation-award.json").read_text(encoding="utf-8"))
    assert confirmed["status"] == "open"
    assert confirmed["min_gpa"] == 3.25

    assert inbox_main([*dirs, "reject", "feed-unwanted-award", "--reason", "not in NC"]) == 0
    capsys.readouterr()
    assert (inbox / "rejected" / "feed-unwanted-award.json").is_file()

    assert inbox_main([*dirs, "list"]) == 0
    assert "No proposals" in capsys.readouterr().out


def test_cli_reports_a_schema_failure_without_writing(
    catalog_dirs: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    inbox, records = catalog_dirs
    propose(_valid_record(trust="probably"), kind="feed", inbox_dir=inbox, records_dir=records)

    exit_code = inbox_main(
        ["--inbox-dir", str(inbox), "--records-dir", str(records), "confirm", "feed-test-foundation-award"]
    )

    assert exit_code == 1
    assert "not a valid catalog record" in capsys.readouterr().out
    assert list(records.glob("*.json")) == []
