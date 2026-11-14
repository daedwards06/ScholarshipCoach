from __future__ import annotations

import json
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from scripts.check_watchlist import main as check_main
from scripts.validate_catalog import validate_watchlist_file
from src.catalog.inbox import list_proposals, reject
from src.catalog.watch import (
    FINDING_AGREES,
    FINDING_BLOCKED,
    FINDING_NOTHING_EXTRACTABLE,
    FINDING_PDF,
    FINDING_POSTBACK,
    OUTCOME_BASELINE,
    OUTCOME_CHANGED,
    OUTCOME_FINDING,
    OUTCOME_SKIPPED,
    OUTCOME_UNCHANGED,
    PRIVATE_WATCHLIST_PATH,
    WATCHLIST_PATH,
    check_watchlist,
    extract_links,
    load_watchlist,
)
from src.normalize.catalog_schema import ROOT_DIR

_TODAY = date(2026, 11, 14)
_SITE = "https://foundation.example.org"


def _award_page(title: str, *, deadline: str = "March 1, 2027", amount: str = "$2,500") -> str:
    return f"""
    <html><head><meta property="og:title" content="{title}" /></head>
    <body><main>
      <h1>{title}</h1>
      <p>Deadline: {deadline}</p>
      <p>Amount: {amount}</p>
    </main></body></html>
    """


def _program_page(*, cost: str = "Cost of Attendance $1,550") -> str:
    return f"""
    <html><head><meta property="og:title" content="Engineering Camp" /></head>
    <body>
      <nav>Summer programs for rising 5th- through 12th-graders</nav>
      <main>
        <h1>Engineering Camp</h1>
        <p>For rising 11th and 12th grade high school students.</p>
        <p>Program Dates Week 1: June 14–19, 2027 Week 2: June 21–26, 2027</p>
        <p>{cost}</p>
      </main>
    </body></html>
    """


def _listing_page(*paths: str) -> str:
    links = "".join(f'<li><a href="{path}">Award</a></li>' for path in paths)
    return f"""
    <html><body><main><h1>Explore Scholarships</h1><ul>{links}</ul>
    <a href="https://elsewhere.example.com/other">Partner</a>
    <a href="#top">Top</a><a href="mailto:x@example.org">Mail</a>
    </main></body></html>
    """


class _StubClient:
    """Serves canned pages per URL; a mapped exception stands in for a bad fetch."""

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


def _entry(entry_id: str, url: str, **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": entry_id,
        "url": url,
        "label": f"Label for {entry_id}",
        "page": "single",
        "kind": "scholarship",
        "months": [11],
        "terms_note": "2026-11-14: terms read; robots.txt allows the page.",
        "enabled": True,
        "notes": None,
    }
    entry.update(overrides)
    return entry


@pytest.fixture
def env(tmp_path: Path) -> dict[str, Path]:
    records = tmp_path / "records"
    records.mkdir()
    return {
        "watchlist": tmp_path / "watchlist.json",
        "private": tmp_path / "private" / "watchlist.json",
        "records": records,
        "inbox": tmp_path / "inbox",
        "reports": tmp_path / "reports",
    }


def _write_watchlist(path: Path, *entries: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"entries": list(entries)}), encoding="utf-8")


def _run(env: dict[str, Path], client: _StubClient, **kwargs: Any) -> Any:
    return check_watchlist(
        watchlist_path=env["watchlist"],
        private_watchlist_path=env["private"],
        records_dir=env["records"],
        inbox_dir=env["inbox"],
        reports_dir=env["reports"],
        client=client,
        today=kwargs.pop("today", _TODAY),
        **kwargs,
    )


# --- the list ----------------------------------------------------------------


def test_committed_watchlist_is_valid() -> None:
    assert validate_watchlist_file(WATCHLIST_PATH) == []


def test_validator_reports_a_bad_entry(tmp_path: Path) -> None:
    path = tmp_path / "watchlist.json"
    bad = _entry("x1", "http://insecure.example.org", page="both")
    _write_watchlist(path, bad, _entry("dup", f"{_SITE}/a"), _entry("dup", f"{_SITE}/b"))
    errors = validate_watchlist_file(path)
    assert any("entries.0.page" in error for error in errors)
    assert any("entries.0.url" in error for error in errors)


def test_validator_holds_back_a_committed_entry_naming_a_privacy_term(tmp_path: Path) -> None:
    private_dir = tmp_path / "private"
    private_dir.mkdir()
    (private_dir / "privacy_names.txt").write_text("Example High School\n", encoding="utf-8")
    (private_dir / "privacy_abbrevs.txt").write_text("EHS\n", encoding="utf-8")
    path = tmp_path / "watchlist.json"
    _write_watchlist(
        path,
        _entry("counselor", f"{_SITE}/a", label="example high school counselor page"),
        _entry("abbrev", f"{_SITE}/ehs/list"),
        _entry("fine", f"{_SITE}/scholarships", notes="mentions behs only inside a word"),
    )

    errors = validate_watchlist_file(path, private_dir)

    assert len(errors) == 2
    assert "'counselor'" in errors[0] and "'abbrev'" in errors[1]
    assert not any("Example" in error or "EHS" in error for error in errors)


def test_loader_reads_both_lists_and_marks_private(env: dict[str, Path]) -> None:
    _write_watchlist(env["watchlist"], _entry("public-one", f"{_SITE}/a"))
    _write_watchlist(
        env["private"],
        _entry("family-one", f"{_SITE}/b"),
        _entry("public-one", f"{_SITE}/c"),
    )
    entries, problems = load_watchlist(env["watchlist"], env["private"])
    assert [(e.id, e.private) for e in entries] == [("public-one", False), ("family-one", True)]
    assert problems == ["private watch list: id 'public-one' is already committed"]


def test_loader_without_a_private_list(env: dict[str, Path]) -> None:
    _write_watchlist(env["watchlist"], _entry("public-one", f"{_SITE}/a"))
    entries, problems = load_watchlist(env["watchlist"], env["private"])
    assert [e.id for e in entries] == ["public-one"] and problems == []


def test_only_the_committed_watchlist_is_in_git() -> None:
    if shutil.which("git") is None or not (ROOT_DIR / ".git").exists():
        pytest.skip("not a git checkout")

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=ROOT_DIR, capture_output=True, text=True, check=False
        )

    tracked = git("ls-files", "--", "*watchlist*.json").stdout.split()
    assert tracked == ["data/catalog/watchlist.json", "data/catalog/watchlist.schema.json"] or (
        tracked == []  # a fresh checkout before the first commit of this file
    )
    private = PRIVATE_WATCHLIST_PATH.relative_to(ROOT_DIR).as_posix()
    assert git("check-ignore", "-q", private).returncode == 0
    assert git("check-ignore", "-q", "reports/watchlist/watch_state.json").returncode == 0


def test_an_entry_without_terms_or_out_of_season_does_not_run(env: dict[str, Path]) -> None:
    _write_watchlist(
        env["watchlist"],
        _entry("no-terms", f"{_SITE}/a", terms_note=None),
        _entry("off-season", f"{_SITE}/b", months=[1, 2]),
        _entry("disabled", f"{_SITE}/c", enabled=False),
    )
    client = _StubClient({})
    report = _run(env, client)
    assert [r.outcome for r in report.results] == [OUTCOME_SKIPPED] * 3
    assert client.requested == []


# --- single pages ------------------------------------------------------------


def test_first_sight_is_a_baseline_and_an_unchanged_page_writes_nothing(
    env: dict[str, Path],
) -> None:
    url = f"{_SITE}/awards/myers"
    _write_watchlist(env["watchlist"], _entry("myers", url))
    client = _StubClient({url: _award_page("Myers Scholarship")})

    first = _run(env, client)
    assert first.results[0].outcome == OUTCOME_BASELINE
    state_path = env["reports"] / "watch_state.json"
    state_before = state_path.read_bytes()

    second = _run(env, client)
    assert second.results[0].outcome == OUTCOME_UNCHANGED
    assert client.requested == [url, url]
    assert state_path.read_bytes() == state_before
    assert list_proposals(env["inbox"]) == []


def test_a_changed_single_page_proposes_once_across_two_runs(env: dict[str, Path]) -> None:
    url = f"{_SITE}/awards/myers"
    _write_watchlist(env["watchlist"], _entry("myers", url))
    client = _StubClient({url: _award_page("Myers Scholarship", deadline="March 2, 2026")})
    _run(env, client)

    client.pages[url] = _award_page("Myers Scholarship", deadline="March 1, 2027")
    changed = _run(env, client)
    third = _run(env, client)

    assert changed.results[0].outcome == OUTCOME_CHANGED
    assert changed.proposal_ids == ["prefill-myers-myers"]
    assert third.results[0].outcome == OUTCOME_UNCHANGED and third.proposal_ids == []
    proposals = list_proposals(env["inbox"])
    assert [p.proposal_id for p in proposals] == ["prefill-myers-myers"]
    proposal = proposals[0]
    assert proposal.kind == "prefill"
    assert proposal.source == "watchlist:myers"
    assert proposal.record["deadline"] == "2027-03-01"
    assert proposal.record["trust"] == "unverified"
    assert "Label for myers" in (proposal.why or "")


def test_a_changed_page_with_a_record_is_diffed_against_it(env: dict[str, Path]) -> None:
    url = f"{_SITE}/camp"
    record = {
        "catalog_id": "engineering-camp",
        "title": "Engineering Camp",
        "kind": "program",
        "source_url": url,
        "cost": 1400.0,
        "grade_levels": ["10", "11"],
        "status": "upcoming",
        "trust": "verified_local",
        "provenance": {"added_on": "2026-10-11", "source_kind": "sponsor_site"},
    }
    (env["records"] / "engineering-camp.json").write_text(json.dumps(record), encoding="utf-8")
    _write_watchlist(env["watchlist"], _entry("ncsu-camps", url, kind="program"))
    client = _StubClient({url: _program_page(cost="Cost of Attendance $1,400")})
    _run(env, client)

    client.pages[url] = _program_page(cost="Cost of Attendance $1,550")
    report = _run(env, client)

    assert report.proposal_ids == ["prefill-ncsu-camps-engineering-camp"]
    proposal = list_proposals(env["inbox"])[0]
    assert proposal.diff == {"cost": {"old": 1400.0, "new": 1550.0}}
    assert "June 14–19, 2027" in (proposal.notes or "")


def test_a_change_that_still_matches_the_record_is_a_finding(env: dict[str, Path]) -> None:
    url = f"{_SITE}/camp"
    record = {
        "catalog_id": "engineering-camp",
        "title": "Engineering Camp",
        "kind": "program",
        "source_url": url,
        "cost": 1550.0,
        "grade_levels": ["10", "11"],
        "status": "upcoming",
        "trust": "verified_local",
        "provenance": {"added_on": "2026-10-11", "source_kind": "sponsor_site"},
    }
    (env["records"] / "engineering-camp.json").write_text(json.dumps(record), encoding="utf-8")
    _write_watchlist(env["watchlist"], _entry("ncsu-camps", url, kind="program"))
    client = _StubClient({url: _program_page()})
    _run(env, client)
    client.pages[url] = _program_page().replace("Week 2", "Second week")

    report = _run(env, client)

    assert [f.reason for f in report.findings] == [FINDING_AGREES]
    assert list_proposals(env["inbox"]) == []


def test_a_new_award_leaves_last_cycles_deadline_out(env: dict[str, Path]) -> None:
    listing = f"{_SITE}/explore"
    _write_watchlist(env["watchlist"], _entry("gaston", listing, page="listing"))
    client = _StubClient({listing: _listing_page()})
    _run(env, client)
    client.pages[listing] = _listing_page("/awards/bohanan")
    client.pages[f"{_SITE}/awards/bohanan"] = _award_page("Bohanan Scholarship", deadline="March 2, 2026")

    _run(env, client)

    proposal = list_proposals(env["inbox"])[0]
    assert proposal.record["deadline"] is None
    assert "last cycle's deadline, 2026-03-02" in (proposal.record["notes"] or "")


def test_a_change_with_nothing_readable_is_a_finding(env: dict[str, Path]) -> None:
    url = f"{_SITE}/news"
    _write_watchlist(env["watchlist"], _entry("news", url))
    client = _StubClient({url: "<html><body><p>Welcome.</p></body></html>"})
    _run(env, client)
    client.pages[url] = "<html><body><p>Welcome back.</p></body></html>"

    report = _run(env, client)

    assert report.results[0].outcome == OUTCOME_FINDING
    assert [f.reason for f in report.findings] == [FINDING_NOTHING_EXTRACTABLE]
    assert list_proposals(env["inbox"]) == []


def test_a_pdf_becomes_a_finding(env: dict[str, Path]) -> None:
    url = f"{_SITE}/forms/application.pdf"
    _write_watchlist(env["watchlist"], _entry("pdf-form", url))
    client = _StubClient({url: "%PDF-1.7 first version"})
    _run(env, client)
    client.pages[url] = "%PDF-1.7 second version"

    report = _run(env, client)

    assert [f.reason for f in report.findings] == [FINDING_PDF]
    assert list_proposals(env["inbox"]) == []


def test_a_refused_fetch_is_a_finding(env: dict[str, Path]) -> None:
    url = f"{_SITE}/guarded"
    _write_watchlist(env["watchlist"], _entry("guarded", url))
    report = _run(env, _StubClient({url: _HttpError(403)}))
    assert report.results[0].outcome == OUTCOME_FINDING
    assert [f.reason for f in report.findings] == [FINDING_BLOCKED]


# --- listings ----------------------------------------------------------------


def test_extract_links_keeps_same_site_pages_only() -> None:
    links = extract_links(_listing_page("/awards/a", "awards/b", "/style.css"), f"{_SITE}/list")
    assert links == [f"{_SITE}/awards/a", f"{_SITE}/awards/b"]


def test_a_listing_that_gains_a_link_proposes_that_award_only(env: dict[str, Path]) -> None:
    listing = f"{_SITE}/explore"
    _write_watchlist(env["watchlist"], _entry("gaston", listing, page="listing"))
    client = _StubClient({listing: _listing_page("/awards/myers")})
    _run(env, client)

    client.pages[listing] = _listing_page("/awards/myers", "/awards/barnard")
    client.pages[f"{_SITE}/awards/barnard"] = _award_page("Barnard Scholarship")
    report = _run(env, client)

    assert report.proposal_ids == ["prefill-gaston-barnard"]
    assert f"{_SITE}/awards/myers" not in client.requested
    proposal = list_proposals(env["inbox"])[0]
    assert proposal.record["title"] == "Barnard Scholarship"
    assert proposal.record["source_url"] == f"{_SITE}/awards/barnard"


def test_a_rejected_link_stays_out(env: dict[str, Path]) -> None:
    listing = f"{_SITE}/explore"
    _write_watchlist(env["watchlist"], _entry("gaston", listing, page="listing"))
    client = _StubClient({listing: _listing_page("/awards/myers")})
    _run(env, client)
    client.pages[listing] = _listing_page("/awards/myers", "/awards/barnard")
    client.pages[f"{_SITE}/awards/barnard"] = _award_page("Barnard Scholarship")
    _run(env, client)
    reject("prefill-gaston-barnard", "Mecklenburg residency only", inbox_dir=env["inbox"])

    # The listing changes again; the rejected award's link is still on it.
    client.pages[listing] = _listing_page("/awards/myers", "/awards/barnard", "/awards/beam")
    client.pages[f"{_SITE}/awards/beam"] = _award_page("Beam Scholarship")
    client.requested.clear()
    report = _run(env, client)

    assert report.proposal_ids == ["prefill-gaston-beam"]
    assert f"{_SITE}/awards/barnard" not in client.requested


def test_a_listing_follows_at_most_ten_links_and_keeps_the_rest_for_later(
    env: dict[str, Path],
) -> None:
    listing = f"{_SITE}/explore"
    _write_watchlist(env["watchlist"], _entry("gaston", listing, page="listing"))
    client = _StubClient({listing: _listing_page()})
    _run(env, client)
    paths = [f"/awards/a{index:02d}" for index in range(12)]
    client.pages[listing] = _listing_page(*paths)
    for index, path in enumerate(paths):
        client.pages[f"{_SITE}{path}"] = _award_page(f"Award {index}")

    report = _run(env, client)

    assert len(report.proposal_ids) == 10
    assert report.results[0].left_out["over_link_cap"] == 2
    state = json.loads((env["reports"] / "watch_state.json").read_text(encoding="utf-8"))
    assert "foundation.example.org/awards/a11" not in " ".join(state["entries"]["gaston"]["links"])


def test_the_run_cap_counts_the_rest(env: dict[str, Path]) -> None:
    listing = f"{_SITE}/explore"
    _write_watchlist(env["watchlist"], _entry("gaston", listing, page="listing"))
    client = _StubClient({listing: _listing_page()})
    _run(env, client)
    paths = [f"/awards/a{index}" for index in range(4)]
    client.pages[listing] = _listing_page(*paths)
    for index, path in enumerate(paths):
        client.pages[f"{_SITE}{path}"] = _award_page(f"Award {index}")

    report = _run(env, client, cap=3)

    assert len(report.proposal_ids) == 3
    assert report.left_out["over_cap"] == 1


def test_a_listing_change_without_new_links_is_a_finding(env: dict[str, Path]) -> None:
    listing = f"{_SITE}/explore"
    _write_watchlist(env["watchlist"], _entry("gaston", listing, page="listing"))
    client = _StubClient({listing: _listing_page("/awards/myers")})
    _run(env, client)
    client.pages[listing] = _listing_page("/awards/myers").replace("Explore", "Find")

    report = _run(env, client)

    assert [f.reason for f in report.findings] == [FINDING_NOTHING_EXTRACTABLE]


# --- private entries ---------------------------------------------------------


def test_a_private_entry_is_proposed_only_to_the_inbox(env: dict[str, Path]) -> None:
    url = f"{_SITE}/counselor/scholarships"
    _write_watchlist(env["watchlist"])
    _write_watchlist(
        env["private"], _entry("her-counselor", url, label="Counselor's list, private")
    )
    client = _StubClient({url: _award_page("Local Civic Award", deadline="March 2, 2026")})
    _run(env, client)
    committed_before = env["watchlist"].read_bytes()
    client.pages[url] = _award_page("Local Civic Award")

    report = _run(env, client)

    assert report.results[0].private is True
    proposals = list_proposals(env["inbox"])
    assert [p.proposal_id for p in proposals] == ["prefill-her-counselor-scholarships"]
    record_text = json.dumps(proposals[0].record)
    assert "her-counselor" not in record_text and "private" not in record_text
    assert list(env["records"].iterdir()) == []
    assert env["watchlist"].read_bytes() == committed_before


# --- the script --------------------------------------------------------------


def test_dry_run_writes_nothing(env: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    _write_watchlist(env["watchlist"])
    code = check_main(
        [
            "--watchlist", str(env["watchlist"]),
            "--private-watchlist", str(env["private"]),
            "--records-dir", str(env["records"]),
            "--inbox-dir", str(env["inbox"]),
            "--reports-dir", str(env["reports"]),
            "--dry-run",
        ]
    )
    assert code == 0
    assert not env["reports"].exists() and not env["inbox"].exists()
    assert "Dry run: nothing written." in capsys.readouterr().out


def test_an_invalid_watchlist_fails_the_run(env: dict[str, Path]) -> None:
    env["watchlist"].write_text('{"entries": [{"id": "x"}]}', encoding="utf-8")
    code = check_main(
        [
            "--watchlist", str(env["watchlist"]),
            "--private-watchlist", str(env["private"]),
            "--reports-dir", str(env["reports"]),
            "--inbox-dir", str(env["inbox"]),
            "--dry-run",
        ]
    )
    assert code == 1


def test_a_postback_portal_listing_is_a_finding(env: dict[str, Path]) -> None:
    listing = f"{_SITE}/Funds/Search.aspx"
    _write_watchlist(env["watchlist"], _entry("portal", listing, page="listing"))
    portal = (
        '<html><body><form><input type="hidden" name="__VIEWSTATE" value="abc" />'
        "<a href=\"javascript:__doPostBack('fund1','')\">{name}</a></form></body></html>"
    )
    client = _StubClient({listing: portal.format(name="Trawick Fund")})
    _run(env, client)
    client.pages[listing] = portal.format(name="Annable Fund")

    report = _run(env, client)

    assert [f.reason for f in report.findings] == [FINDING_POSTBACK]
    assert list_proposals(env["inbox"]) == []
