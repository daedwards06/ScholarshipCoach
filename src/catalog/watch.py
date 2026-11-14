"""The watch list: re-read the pages local awards and programs live on.

Local awards and summer programs are in no feed.  They are on a few dozen
pages that change a few times a year.  This module fetches each watched page
politely, hashes its cleaned text, and compares with the last run:

* an unchanged page costs one request and writes nothing;
* the first sight of a page records a baseline and proposes nothing;
* a changed ``single`` page is prefilled into a ``prefill`` proposal, diffed
  against the curated record that shares its URL, if any;
* a changed ``listing`` page is diffed by its links: same-site links that are
  new and match no record or rejected proposal are fetched and prefilled, at
  most :data:`LINKS_PER_LISTING` a page, one proposal per award;
* what cannot be read -- a PDF, a postback portal, a refused fetch, a change
  with nothing extractable, or a change that reads the same as its record --
  becomes a *finding* ("this page changed; open it") for the digest, with no
  proposal.

The committed list (``data/catalog/watchlist.json``) holds only pages that say
nothing about the student.  Family pages go in ``data/private/watchlist.json``
on the server, same shape, merged at run time.  A proposal never names its
watch entry inside the record, so a confirmed record carries nothing from a
private entry; the entry is named only in the proposal's metadata, in the
git-ignored inbox.  State and reports live under ``reports/watchlist/``,
git-ignored.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from functools import lru_cache
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from jsonschema import Draft202012Validator

from src.catalog import entry as catalog_entry
from src.catalog import inbox
from src.catalog.verify import (
    DEFAULT_REQUESTS_PER_SECOND,
    MIN_CHANGE_CONFIDENCE,
    OUTCOME_BLOCKED,
    OUTCOME_DEAD_LINK,
    classify_http_status,
    compare_record,
    content_hash,
)
from src.ingest.dedupe import normalize_url_for_dedupe
from src.ingest.prefill import PrefillResult, prefill_from_html
from src.io.snapshotting import write_json_atomic
from src.normalize.catalog_schema import CATALOG_DIR, RECORDS_DIR, ROOT_DIR, iter_catalog_files

logger = logging.getLogger(__name__)

WATCHLIST_PATH = CATALOG_DIR / "watchlist.json"
WATCHLIST_SCHEMA_PATH = CATALOG_DIR / "watchlist.schema.json"
PRIVATE_WATCHLIST_PATH = ROOT_DIR / "data" / "private" / "watchlist.json"
REPORTS_DIR = ROOT_DIR / "reports" / "watchlist"
STATE_PATH = REPORTS_DIR / "watch_state.json"

DEFAULT_CAP = 15
LINKS_PER_LISTING = 10
SOURCE_PREFIX = "watchlist:"

OUTCOME_SKIPPED = "skipped"
OUTCOME_BASELINE = "baseline"
OUTCOME_UNCHANGED = "unchanged"
OUTCOME_CHANGED = "changed"
OUTCOME_FINDING = "finding"
OUTCOME_ERROR = "error"
OUTCOMES = (
    OUTCOME_SKIPPED,
    OUTCOME_BASELINE,
    OUTCOME_UNCHANGED,
    OUTCOME_CHANGED,
    OUTCOME_FINDING,
    OUTCOME_ERROR,
)

FINDING_PDF = "pdf"
FINDING_POSTBACK = "postback_portal"
FINDING_BLOCKED = "blocked"
FINDING_DEAD_LINK = "dead_link"
FINDING_NOTHING_EXTRACTABLE = "nothing_extractable"
FINDING_AGREES = "changed_agrees_with_record"

LEFT_OUT_REASONS = ("known_record", "previously_rejected", "over_link_cap", "over_cap")

_FINDING_MESSAGES = {
    FINDING_PDF: "This is a PDF, which the watch list cannot read. It changed; open it.",
    FINDING_POSTBACK: "This page changed, but its links are a portal's buttons, not pages. Open it.",
    FINDING_BLOCKED: "The site refused the fetch. Open it in a browser.",
    FINDING_DEAD_LINK: "The page is gone. Find where it moved and update the watch list.",
    FINDING_NOTHING_EXTRACTABLE: "This page changed, and nothing on it could be read. Open it.",
    FINDING_AGREES: "This page changed, but what could be read still matches the record. Open it.",
}

_HREF_PATTERN = re.compile(r"""<a\b[^>]*?\bhref\s*=\s*["']([^"'#][^"']*)["']""", re.IGNORECASE)
_POSTBACK_MARKERS = ("__doPostBack", "__VIEWSTATE")
_SKIPPED_LINK_SUFFIXES = (
    ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".zip", ".mp4",
)
_SLUG_UNSAFE = re.compile(r"[^a-z0-9]+")
_MAX_ID_CHARS = 120


@dataclass(frozen=True, slots=True)
class WatchEntry:
    """One watched page, from the committed list or the private overlay."""

    id: str
    url: str
    label: str
    page: str
    kind: str
    months: tuple[int, ...]
    terms_note: str | None
    enabled: bool
    notes: str | None = None
    private: bool = False

    @property
    def source(self) -> str:
        return f"{SOURCE_PREFIX}{self.id}"


@dataclass(slots=True)
class Finding:
    """Something a person should open, because the watch list could not read it."""

    entry_id: str
    url: str
    reason: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "url": self.url,
            "reason": self.reason,
            "message": self.message,
        }


@dataclass(slots=True)
class EntryResult:
    """What one run found for one watch entry."""

    entry_id: str
    url: str
    page: str
    kind: str
    private: bool
    outcome: str
    message: str | None = None
    http_status: int | None = None
    content_hash: str | None = None
    proposals: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    left_out: dict[str, int] = field(default_factory=lambda: dict.fromkeys(LEFT_OUT_REASONS, 0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "url": self.url,
            "page": self.page,
            "kind": self.kind,
            "private": self.private,
            "outcome": self.outcome,
            "message": self.message,
            "http_status": self.http_status,
            "content_hash": self.content_hash,
            "proposals": self.proposals,
            "findings": [finding.to_dict() for finding in self.findings],
            "left_out": self.left_out,
        }


@dataclass(slots=True)
class WatchReport:
    """The summary one run writes under ``reports/watchlist/``, shaped like verify's."""

    started_at: str
    finished_at: str
    entries_total: int
    dry_run: bool
    problems: list[str] = field(default_factory=list)
    results: list[EntryResult] = field(default_factory=list)
    report_path: Path | None = None

    @property
    def counts(self) -> dict[str, int]:
        tally = dict.fromkeys(OUTCOMES, 0)
        for result in self.results:
            tally[result.outcome] = tally.get(result.outcome, 0) + 1
        return tally

    @property
    def proposal_ids(self) -> list[str]:
        return [proposal for result in self.results for proposal in result.proposals]

    @property
    def findings(self) -> list[Finding]:
        return [finding for result in self.results for finding in result.findings]

    @property
    def left_out(self) -> dict[str, int]:
        tally = dict.fromkeys(LEFT_OUT_REASONS, 0)
        for result in self.results:
            for reason, count in result.left_out.items():
                tally[reason] = tally.get(reason, 0) + count
        return tally

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "dry_run": self.dry_run,
            "entries_total": self.entries_total,
            "entries_checked": sum(1 for r in self.results if r.outcome != OUTCOME_SKIPPED),
            "counts": self.counts,
            "left_out": self.left_out,
            "problems": self.problems,
            "proposals": self.proposal_ids,
            "findings": [finding.to_dict() for finding in self.findings],
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(slots=True)
class _Candidate:
    entry: WatchEntry
    proposal_id: str
    record: dict[str, Any]
    why: str
    notes: str
    link: str | None = None


@lru_cache(maxsize=2)
def load_watchlist_schema(schema_path: Path | None = None) -> dict[str, Any]:
    path = schema_path or WATCHLIST_SCHEMA_PATH
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_watchlist(payload: Any, *, schema_path: Path | None = None) -> list[str]:
    """Schema violations and duplicate ids in one watch list file (empty when valid)."""
    validator = Draft202012Validator(load_watchlist_schema(schema_path))
    errors = []
    for error in sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path)):
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        errors.append(f"{location}: {error.message}")
    if errors:
        return errors
    seen: set[str] = set()
    for item in payload["entries"]:
        if item["id"] in seen:
            errors.append(f"entries: duplicate id {item['id']!r}")
        seen.add(item["id"])
    return errors


def load_watchlist(
    path: Path | None = None,
    private_path: Path | None = None,
    *,
    schema_path: Path | None = None,
) -> tuple[list[WatchEntry], list[str]]:
    """Return the committed entries, then the private overlay's, and any problems.

    A file that fails validation contributes no entries: a half-valid list is
    a mistake, not a partial instruction.  The private file is optional; a
    private entry reusing a committed id is dropped.
    """
    entries: list[WatchEntry] = []
    problems: list[str] = []
    sources = ((path or WATCHLIST_PATH, False), (private_path or PRIVATE_WATCHLIST_PATH, True))
    for file_path, private in sources:
        if private and not file_path.is_file():
            continue
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            problems.append(f"{file_path.name}: unreadable ({exc})")
            continue
        errors = validate_watchlist(payload, schema_path=schema_path)
        if errors:
            label = "private watch list" if private else file_path.name
            problems.extend(f"{label}: {error}" for error in errors)
            continue
        taken = {entry.id for entry in entries}
        for item in payload["entries"]:
            if item["id"] in taken:
                problems.append(f"private watch list: id {item['id']!r} is already committed")
                continue
            entries.append(
                WatchEntry(
                    id=item["id"],
                    url=item["url"],
                    label=item["label"],
                    page=item["page"],
                    kind=item["kind"],
                    months=tuple(item["months"]),
                    terms_note=item["terms_note"],
                    enabled=item["enabled"],
                    notes=item.get("notes"),
                    private=private,
                )
            )
    return entries, problems


def skip_reason(entry: WatchEntry, today: date, *, ignore_months: bool = False) -> str | None:
    """Why this entry does not run today, or ``None`` when it does."""
    if not entry.enabled:
        return "disabled"
    if not entry.terms_note:
        return "no terms_note: nobody has read the site's terms yet"
    if not ignore_months and today.month not in entry.months:
        return "not a month to check it"
    return None


def extract_links(html: str, base_url: str) -> list[str]:
    """Same-site page links on ``html``, absolute, without fragments, in page order."""
    site = _site(base_url)
    own = normalize_url_for_dedupe(base_url)
    links: list[str] = []
    for raw in _HREF_PATTERN.findall(html):
        href = unescape(raw).strip()
        if href.lower().startswith(("mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(base_url, href).split("#", 1)[0]
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https") or _site(absolute) != site:
            continue
        if parsed.path.lower().endswith(_SKIPPED_LINK_SUFFIXES):
            continue
        if normalize_url_for_dedupe(absolute) == own:
            continue
        links.append(absolute)
    return list(dict.fromkeys(links))


def is_pdf(url: str, body: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf") or body.lstrip().startswith("%PDF")


def check_watchlist(
    *,
    watchlist_path: Path | None = None,
    private_watchlist_path: Path | None = None,
    records_dir: Path | None = None,
    inbox_dir: Path | None = None,
    rejected_dir: Path | None = None,
    reports_dir: Path | None = None,
    state_path: Path | None = None,
    client: Any | None = None,
    today: date | None = None,
    cap: int = DEFAULT_CAP,
    ignore_months: bool = False,
    dry_run: bool = False,
    requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND,
) -> WatchReport:
    """Check every due watch entry, write proposals and findings, and return the report.

    ``dry_run`` fetches and reports, and writes no proposal, state or report.
    """
    run_date = today or date.today()
    started_at = datetime.now(UTC)
    resolved_records = records_dir or RECORDS_DIR
    resolved_state = state_path or (reports_dir / "watch_state.json" if reports_dir else STATE_PATH)
    resolved_rejected = rejected_dir or ((inbox_dir / "rejected") if inbox_dir else inbox.REJECTED_DIR)

    entries, problems = load_watchlist(watchlist_path, private_watchlist_path)
    state = load_state(resolved_state)
    records = _load_records(resolved_records)
    record_urls = {normalize_url_for_dedupe(r.get("source_url")): r for r in records}
    rejected_urls = _rejected_urls(resolved_rejected)

    results: list[EntryResult] = []
    candidates: list[tuple[EntryResult, _Candidate]] = []
    new_state: dict[str, dict[str, Any]] = {}

    due = [e for e in entries if skip_reason(e, run_date, ignore_months=ignore_months) is None]
    http_client = client
    owns_client = False
    if http_client is None and due:
        from src.ingest.http import PoliteHttpClient

        http_client = PoliteHttpClient(requests_per_second=requests_per_second)
        owns_client = True

    try:
        for watch in entries:
            reason = skip_reason(watch, run_date, ignore_months=ignore_months)
            if reason is not None:
                results.append(_result(watch, OUTCOME_SKIPPED, message=reason))
                continue
            result, entry_candidates, entry_state = _check_entry(
                watch,
                client=http_client,
                prior=state["entries"].get(watch.id),
                record_urls=record_urls,
                rejected_urls=rejected_urls,
                run_date=run_date,
            )
            results.append(result)
            candidates.extend((result, candidate) for candidate in entry_candidates)
            if entry_state is not None:
                new_state[watch.id] = entry_state
    finally:
        if owns_client:
            close = getattr(http_client, "close", None)
            if callable(close):
                close()

    written = 0
    for result, candidate in candidates:
        if inbox.is_previously_rejected(
            candidate.record,
            proposal_id=candidate.proposal_id,
            kind="prefill",
            inbox_dir=inbox_dir,
            records_dir=resolved_records,
            rejected_dir=resolved_rejected,
        ):
            result.left_out["previously_rejected"] += 1
            continue
        if written >= cap:
            result.left_out["over_cap"] += 1
            _forget(new_state, state, candidate)
            continue
        if not dry_run:
            try:
                inbox.propose(
                    candidate.record,
                    kind="prefill",
                    proposal_id=candidate.proposal_id,
                    notes=candidate.notes,
                    source=candidate.entry.source,
                    why=candidate.why,
                    created_on=run_date,
                    inbox_dir=inbox_dir,
                    records_dir=resolved_records,
                    rejected_dir=resolved_rejected,
                )
            except inbox.PreviouslyRejectedError:
                result.left_out["previously_rejected"] += 1
                continue
            except (inbox.ProposalError, OSError) as exc:
                result.message = f"could not queue {candidate.proposal_id}: {exc}"
                logger.error("watch: %s", result.message)
                _forget(new_state, state, candidate)
                continue
        result.proposals.append(candidate.proposal_id)
        written += 1

    report = WatchReport(
        started_at=_stamp(started_at),
        finished_at=_stamp(datetime.now(UTC)),
        entries_total=len(entries),
        dry_run=dry_run,
        problems=problems,
        results=results,
    )
    if not dry_run:
        if new_state:
            state["entries"].update(new_state)
            write_json_atomic(state, resolved_state)
        stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
        report.report_path = (reports_dir or REPORTS_DIR) / f"watchlist_{stamp}.json"
        write_json_atomic(report.to_dict(), report.report_path)
    return report


def load_state(path: Path | None = None) -> dict[str, Any]:
    """Return ``{"entries": {id: {"content_hash", "checked_on", "links"}}}``; empty when unreadable."""
    resolved = path or STATE_PATH
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {"entries": {}}
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        logger.error("watch: unreadable state %s (%s); treating as a first run", resolved, exc)
        return {"entries": {}}
    entries = payload.get("entries") if isinstance(payload, dict) else None
    return {"entries": entries if isinstance(entries, dict) else {}}


def _check_entry(
    watch: WatchEntry,
    *,
    client: Any,
    prior: Mapping[str, Any] | None,
    record_urls: Mapping[str, Mapping[str, Any]],
    rejected_urls: set[str],
    run_date: date,
) -> tuple[EntryResult, list[_Candidate], dict[str, Any] | None]:
    """Fetch one entry's page; return its result, its candidates and its next state.

    The next state is ``None`` when nothing about the entry should be written.
    """
    body, status, error = _fetch(client, watch.url)
    if body is None:
        return _fetch_failure(watch, watch.url, status, error), [], None

    pdf = is_pdf(watch.url, body)
    prefill = None if pdf else prefill_from_html(body, url=watch.url)
    digest = content_hash(body if prefill is None else prefill.text)
    links = [] if pdf or watch.page != "listing" else extract_links(body, watch.url)
    entry_state: dict[str, Any] = {"content_hash": digest, "checked_on": run_date.isoformat()}
    if watch.page == "listing":
        entry_state["links"] = sorted(normalize_url_for_dedupe(link) for link in links)

    result = _result(watch, OUTCOME_CHANGED, http_status=status or 200, content_hash=digest)
    if not isinstance(prior, Mapping) or not prior.get("content_hash"):
        result.outcome = OUTCOME_BASELINE
        return result, [], entry_state
    if prior.get("content_hash") == digest:
        result.outcome = OUTCOME_UNCHANGED
        return result, [], None

    since = str(prior.get("checked_on") or "the last check")
    if pdf:
        result.findings.append(_finding(watch, watch.url, FINDING_PDF))
        result.outcome = OUTCOME_FINDING
        return result, [], entry_state

    assert prefill is not None
    if watch.page == "single":
        candidate, finding = _single_page_candidate(watch, prefill, record_urls, run_date, since)
        if finding is not None:
            result.findings.append(finding)
            result.outcome = OUTCOME_FINDING
        return result, [candidate] if candidate else [], entry_state

    candidates: list[_Candidate] = []
    prior_links = set(prior.get("links") or [])
    new_links = [link for link in links if normalize_url_for_dedupe(link) not in prior_links]
    if not new_links:
        reason = FINDING_POSTBACK if any(m in body for m in _POSTBACK_MARKERS) else (
            FINDING_NOTHING_EXTRACTABLE
        )
        result.findings.append(_finding(watch, watch.url, reason))
        result.outcome = OUTCOME_FINDING
        return result, [], entry_state

    unhandled: set[str] = set()
    followed = 0
    for link in new_links:
        key = normalize_url_for_dedupe(link)
        if key in record_urls:
            result.left_out["known_record"] += 1
            continue
        if key in rejected_urls:
            result.left_out["previously_rejected"] += 1
            continue
        if followed >= LINKS_PER_LISTING:
            result.left_out["over_link_cap"] += 1
            unhandled.add(key)
            continue
        followed += 1
        candidate, finding = _listing_link_candidate(watch, link, client, run_date, since)
        if candidate is not None:
            candidates.append(candidate)
        if finding is not None:
            result.findings.append(finding)
    # A link over the cap stays out of the state, so the next change follows it.
    entry_state["links"] = [link for link in entry_state["links"] if link not in unhandled]
    if result.findings and not candidates:
        result.outcome = OUTCOME_FINDING
    return result, candidates, entry_state


def _single_page_candidate(
    watch: WatchEntry,
    prefill: PrefillResult,
    record_urls: Mapping[str, Mapping[str, Any]],
    run_date: date,
    since: str,
) -> tuple[_Candidate | None, Finding | None]:
    why = f'The watched page "{watch.label}" changed since {since}.'
    record = record_urls.get(normalize_url_for_dedupe(watch.url))
    if record is None:
        return _new_award_candidate(watch, watch.url, prefill, run_date, why)

    changes = compare_record(record, prefill, today=run_date)
    if str(record.get("kind") or "scholarship") == "program":
        changes.update(_program_changes(record, prefill))
    if not changes:
        return None, _finding(watch, watch.url, FINDING_AGREES)

    proposed = dict(record)
    for key, change in changes.items():
        proposed[key] = change["new"]
    fields = ", ".join(sorted(changes))
    notes = f"The page disagrees with the record on {fields}."
    deadline_note = changes.get("deadline", {}).get("note")
    if deadline_note:
        notes = f"{notes} {deadline_note}"
    notes = _with_sessions(notes, prefill)
    catalog_id = str(record["catalog_id"])
    return (
        _Candidate(
            entry=watch,
            proposal_id=watch_proposal_id(watch.id, catalog_id),
            record=proposed,
            why=why,
            notes=notes,
        ),
        None,
    )


def _listing_link_candidate(
    watch: WatchEntry, link: str, client: Any, run_date: date, since: str
) -> tuple[_Candidate | None, Finding | None]:
    body, status, error = _fetch(client, link)
    if body is None:
        failure = _fetch_failure(watch, link, status, error)
        return None, failure.findings[0] if failure.findings else None
    if is_pdf(link, body):
        return None, _finding(watch, link, FINDING_PDF)
    why = f'A new link on the watched page "{watch.label}" since {since}.'
    candidate, finding = _new_award_candidate(
        watch, link, prefill_from_html(body, url=link), run_date, why
    )
    if candidate is not None:
        candidate.link = normalize_url_for_dedupe(link)
    return candidate, finding


def _new_award_candidate(
    watch: WatchEntry, url: str, prefill: PrefillResult, run_date: date, why: str
) -> tuple[_Candidate | None, Finding | None]:
    """A new record from a page, or a finding when the page gave nothing to go on."""
    extraction = prefill.extraction
    program = watch.kind == "program"
    stated = bool(
        extraction.deadline_candidates
        or extraction.amount_candidates
        or extraction.grade_levels
        or (program and (extraction.cost is not None or extraction.session_dates))
    )
    if not prefill.title or not stated:
        return None, _finding(watch, url, FINDING_NOTHING_EXTRACTABLE)

    values = catalog_entry.form_from_prefill(prefill.to_form_dict())
    values["kind"] = watch.kind
    values["trust"] = "unverified"
    values["source_kind"] = "sponsor_site"
    if program:
        values["cost"] = extraction.cost
    notes = "Read by the watch list; open the page before confirming."
    # Last cycle's date still on the page is not this cycle's deadline (Task 1.2's rule).
    if values["deadline"] and values["deadline"] <= run_date.isoformat():
        notes = f"{notes} The page still shows last cycle's deadline, {values['deadline']}."
        values["deadline"] = ""
    values["notes"] = _with_sessions(notes, prefill)
    record = catalog_entry.record_from_form(values, today=run_date)
    return (
        _Candidate(
            entry=watch,
            proposal_id=watch_proposal_id(watch.id, _url_slug(url)),
            record=record,
            why=why,
            notes="Not in the catalog yet. Check the grades and the date before confirming.",
        ),
        None,
    )


def _program_changes(record: Mapping[str, Any], prefill: PrefillResult) -> dict[str, dict[str, Any]]:
    """Cost and grades the page states and the program record holds differently."""
    extraction = prefill.extraction
    confidence = prefill.confidence
    changes: dict[str, dict[str, Any]] = {}
    if extraction.cost is not None and confidence.get("cost", 0.0) >= MIN_CHANGE_CONFIDENCE:
        old_cost = record.get("cost")
        if old_cost is None or float(old_cost) != extraction.cost:
            changes["cost"] = {"old": old_cost, "new": extraction.cost}
    grades = extraction.grade_levels
    if grades and confidence.get("grade_levels", 0.0) >= MIN_CHANGE_CONFIDENCE:
        old_grades = list(record.get("grade_levels") or [])
        if sorted(old_grades) != sorted(grades):
            changes["grade_levels"] = {"old": old_grades, "new": list(grades)}
    return changes


def watch_proposal_id(watch_id: str, slug: str) -> str:
    """``prefill-<watch id>-<slug>``: a re-run replaces its own proposal."""
    return f"prefill-{watch_id}-{slug}"[:_MAX_ID_CHARS].strip("-")


def _url_slug(url: str) -> str:
    """The page's last path segment as a slug: ``/portfolio-item/myers-scholarship`` -> ``myers-scholarship``."""
    parsed = urlparse(url)
    segments = [segment for segment in parsed.path.lower().split("/") if segment]
    last = re.sub(r"\.(?:html?|aspx?|php)$", "", segments[-1]) if segments else parsed.netloc.lower()
    return _SLUG_UNSAFE.sub("-", last).strip("-")[:60].strip("-") or "page"


def _with_sessions(notes: str, prefill: PrefillResult) -> str:
    sessions = prefill.extraction.session_dates
    if not sessions:
        return notes
    return f"{notes} Session dates on the page: {'; '.join(sessions)}."


def _fetch(client: Any, url: str) -> tuple[str | None, int | None, str | None]:
    try:
        return str(client.get_text(url)), None, None
    except Exception as exc:  # a scheduled pass must survive one bad page
        status = getattr(getattr(exc, "response", None), "status_code", None)
        message = f"{type(exc).__name__}: {exc}"
        logger.warning("watch: %s failed: %s", url, message)
        return None, int(status) if isinstance(status, int) else None, message


def _fetch_failure(
    watch: WatchEntry, url: str, status: int | None, error: str | None
) -> EntryResult:
    """A refused or dead page is a finding; a network error is only an error."""
    result = _result(watch, OUTCOME_ERROR, message=error, http_status=status)
    if status is None:
        return result
    outcome = classify_http_status(status)
    if outcome == OUTCOME_BLOCKED:
        result.findings.append(_finding(watch, url, FINDING_BLOCKED))
        result.outcome = OUTCOME_FINDING
    elif outcome == OUTCOME_DEAD_LINK:
        result.findings.append(_finding(watch, url, FINDING_DEAD_LINK))
        result.outcome = OUTCOME_FINDING
    return result


def _forget(
    new_state: dict[str, dict[str, Any]], state: Mapping[str, Any], candidate: _Candidate
) -> None:
    """Keep a candidate that was not written askable on the next change.

    A link is dropped from the listing's saved links; a single page keeps its
    old hash, so the next run sees the change again.
    """
    entry_state = new_state.get(candidate.entry.id)
    if entry_state is None:
        return
    if candidate.link is not None:
        entry_state["links"] = [link for link in entry_state.get("links", []) if link != candidate.link]
        return
    prior = state["entries"].get(candidate.entry.id)
    if isinstance(prior, Mapping):
        new_state[candidate.entry.id] = dict(prior)
    else:
        new_state.pop(candidate.entry.id, None)


def _result(watch: WatchEntry, outcome: str, **kwargs: Any) -> EntryResult:
    return EntryResult(
        entry_id=watch.id,
        url=watch.url,
        page=watch.page,
        kind=watch.kind,
        private=watch.private,
        outcome=outcome,
        **kwargs,
    )


def _finding(watch: WatchEntry, url: str, reason: str) -> Finding:
    return Finding(entry_id=watch.id, url=url, reason=reason, message=_FINDING_MESSAGES[reason])


def _site(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _rejected_urls(rejected_dir: Path) -> set[str]:
    urls: set[str] = set()
    if not rejected_dir.is_dir():
        return urls
    for path in rejected_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        record = payload.get("record") if isinstance(payload, dict) else None
        if isinstance(record, dict) and record.get("source_url"):
            urls.add(normalize_url_for_dedupe(record["source_url"]))
    return urls


def _load_records(records_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_catalog_files(records_dir):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            logger.error("watch: skipping unreadable record %s: %s", path, exc)
            continue
        if isinstance(payload, dict) and payload.get("catalog_id"):
            records.append(payload)
    return records


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")

