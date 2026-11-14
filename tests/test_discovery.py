from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from src.catalog.discovery import (
    DiscoveryResult,
    feed_proposal_id,
    load_state,
    run_feed_discovery,
)
from src.catalog.inbox import confirm, list_proposals, load_proposal, reject
from src.profile.store import load_demo_profile

_TODAY = date(2026, 11, 14)
_SOURCE = "open_scholarships"


def _row(source_id: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source": _SOURCE,
        "source_id": source_id,
        "source_url": f"https://example.invalid/awards/{source_id}",
        "title": f"Example Award {source_id.upper()}",
        "sponsor": "Example Foundation",
        "description": "An award used by the discovery tests.",
        "deadline": "2027-03-01",
        "status": "open",
        "amount_min": None,
        "amount_max": 1000.0,
        "states_allowed": [],
        "grade_levels": [],
        "trust": "structured_feed",
        "cycle": {"recurring": True, "opens_month": None, "deadline_month": 3},
        "provenance": {
            "added_on": "2026-05-01",
            "source_kind": "structured_feed",
            "license": "CC-BY-4.0",
            "attribution": "Example feed (CC BY 4.0)",
        },
        "notes": None,
    }
    row.update(overrides)
    return row


@pytest.fixture
def dirs(tmp_path: Path) -> dict[str, Path]:
    records = tmp_path / "records"
    records.mkdir()
    return {
        "state_path": tmp_path / "reports" / "feed_state.json",
        "inbox_dir": tmp_path / "inbox",
        "records_dir": records,
    }


def _run(rows: list[dict[str, Any]], dirs: dict[str, Path], **kwargs: Any) -> DiscoveryResult:
    kwargs.setdefault("profile", load_demo_profile())
    return run_feed_discovery(rows, source=_SOURCE, today=_TODAY, **dirs, **kwargs)


def _baseline(rows: list[dict[str, Any]], dirs: dict[str, Path]) -> None:
    result = _run(rows, dirs)
    assert result.baseline


def test_first_run_writes_the_baseline_and_proposes_nothing(dirs: dict[str, Path]) -> None:
    result = _run([_row("a"), _row("b")], dirs)

    assert result.baseline
    assert result.proposed == []
    assert result.left_out["baseline"] == 2
    assert list_proposals(dirs["inbox_dir"]) == []
    state = load_state(dirs["state_path"])
    assert set(state["sources"][_SOURCE]["rows"]) == {"a", "b"}


def test_state_holds_feed_values_only(dirs: dict[str, Path]) -> None:
    _run([_row("a")], dirs)

    state = load_state(dirs["state_path"])
    assert state["sources"][_SOURCE]["rows"]["a"] == {
        "deadline": "2027-03-01",
        "status": "open",
        "amount_min": None,
        "amount_max": 1000.0,
    }


def test_a_new_row_is_proposed_with_a_deterministic_id(dirs: dict[str, Path]) -> None:
    _baseline([_row("a")], dirs)

    first = _run([_row("a"), _row("b")], dirs)
    assert first.proposed == ["feed-open-scholarships-b"]
    assert first.left_out["unchanged"] == 1

    proposal = load_proposal("feed-open-scholarships-b", dirs["inbox_dir"])
    assert proposal.kind == "feed"
    assert proposal.source == _SOURCE
    assert proposal.why == "New in the feed since the last run."
    assert proposal.record["trust"] == "unverified"
    assert proposal.record["catalog_id"] == "example-award-b"
    assert proposal.record["provenance"]["source_kind"] == "structured_feed"
    assert "Example feed (CC BY 4.0)" in proposal.record["notes"]

    # The same row with a new deadline replaces its own proposal.
    second = _run([_row("a"), _row("b", deadline="2027-04-01")], dirs)
    assert second.proposed == first.proposed
    assert [p.proposal_id for p in list_proposals(dirs["inbox_dir"])] == first.proposed


def test_proposal_ids_never_collide_across_sources() -> None:
    assert feed_proposal_id("open_scholarships", _row("x")) != feed_proposal_id(
        "other_feed", _row("x")
    )


def test_a_changed_deadline_re_proposes_and_an_unchanged_row_does_not(
    dirs: dict[str, Path],
) -> None:
    _baseline([_row("a"), _row("b")], dirs)

    unchanged = _run([_row("a"), _row("b")], dirs)
    assert unchanged.proposed == []
    assert unchanged.left_out["unchanged"] == 2

    changed = _run([_row("a"), _row("b", deadline="2027-04-15")], dirs)
    assert changed.proposed == ["feed-open-scholarships-b"]
    proposal = load_proposal("feed-open-scholarships-b", dirs["inbox_dir"])
    assert proposal.why == "The feed changed deadline 2027-03-01 → 2027-04-15."


def test_rejected_then_unchanged_stays_out_and_rejected_then_changed_returns(
    dirs: dict[str, Path],
) -> None:
    _baseline([], dirs)
    _run([_row("a")], dirs)
    reject("feed-open-scholarships-a", "not for her", inbox_dir=dirs["inbox_dir"])

    again = _run([_row("a")], dirs)
    assert again.proposed == []
    assert list_proposals(dirs["inbox_dir"]) == []

    # The state is lost, so the row reads as new: the rejection still keeps it out.
    dirs["state_path"].write_text(
        json.dumps({"sources": {_SOURCE: {"rows": {}}}}), encoding="utf-8"
    )
    lost_state = _run([_row("a")], dirs)
    assert lost_state.proposed == []
    assert lost_state.left_out["previously_rejected"] == 1

    changed = _run([_row("a", amount_max=2500.0)], dirs)
    assert changed.proposed == ["feed-open-scholarships-a"]


def test_the_cap_keeps_the_soonest_deadlines_and_counts_the_rest(dirs: dict[str, Path]) -> None:
    _baseline([], dirs)
    rows = [_row(f"r{index:02d}", deadline=f"2027-{index + 1:02d}-01") for index in range(5)]

    result = _run(rows, dirs, cap=3)

    assert result.proposed == [
        "feed-open-scholarships-r00",
        "feed-open-scholarships-r01",
        "feed-open-scholarships-r02",
    ]
    assert result.left_out["over_cap"] == 2


def test_a_row_covered_by_a_curated_record_is_left_out(dirs: dict[str, Path]) -> None:
    curated = {
        "catalog_id": "example-award-a",
        "title": "Example Award A",
        "sponsor": "Example Foundation",
        "source_url": "https://example.invalid/awards/a",
        "deadline": "2027-03-01",
        "status": "open",
        "amount_max": 1000,
        "grade_levels": ["11", "12"],
        "trust": "verified_local",
        "provenance": {"added_on": "2026-09-12", "source_kind": "sponsor_site"},
    }
    (dirs["records_dir"] / "example-award-a.json").write_text(json.dumps(curated), encoding="utf-8")
    _baseline([], dirs)

    covered = _run([_row("a")], dirs)
    assert covered.proposed == []
    assert covered.left_out["covered_by_curated"] == 1

    # The feed posts a new date: proposed against the curated record, keeping its trust.
    moved = _run([_row("a", deadline="2027-02-15")], dirs)
    assert moved.proposed == ["feed-open-scholarships-a"]
    proposal = load_proposal("feed-open-scholarships-a", dirs["inbox_dir"])
    assert proposal.record["catalog_id"] == "example-award-a"
    assert proposal.record["trust"] == "verified_local"
    assert proposal.diff == {"deadline": {"old": "2027-03-01", "new": "2027-02-15"}}


def test_the_profile_screen_ignores_grade_and_drops_what_she_cannot_apply_to(
    dirs: dict[str, Path],
) -> None:
    _baseline([], dirs)
    rows = [
        _row("in-state", states_allowed=["NC"]),
        _row("out-of-state", states_allowed=["NV"]),
        _row("other-grade", grade_levels=["9"]),
    ]

    result = _run(rows, dirs)

    assert sorted(result.proposed) == [
        "feed-open-scholarships-in-state",
        "feed-open-scholarships-other-grade",
    ]
    assert result.left_out["failed_profile_screen"] == 1


def test_no_saved_profile_screens_nothing(dirs: dict[str, Path]) -> None:
    _baseline([], dirs)

    result = _run([_row("out-of-state", states_allowed=["NV"])], dirs, profile=None)

    assert result.proposed == ["feed-open-scholarships-out-of-state"]


def test_dry_run_writes_nothing(dirs: dict[str, Path]) -> None:
    _baseline([], dirs)
    before = dirs["state_path"].read_text(encoding="utf-8")

    result = _run([_row("a")], dirs, dry_run=True)

    assert result.proposed == ["feed-open-scholarships-a"]
    assert list_proposals(dirs["inbox_dir"]) == []
    assert dirs["state_path"].read_text(encoding="utf-8") == before


def test_a_feed_proposal_confirms_once_a_person_adds_grades(dirs: dict[str, Path]) -> None:
    _baseline([], dirs)
    _run([_row("a")], dirs)

    written = confirm(
        "feed-open-scholarships-a",
        {"grade_levels": ["11", "12"]},
        inbox_dir=dirs["inbox_dir"],
        records_dir=dirs["records_dir"],
    )

    record = json.loads(written.read_text(encoding="utf-8"))
    assert record["cycle"]["deadline_month"] == 3
    assert record["trust"] == "unverified"
