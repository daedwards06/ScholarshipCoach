"""Annual re-verification: re-read every catalog record's page and diff it.

Deadlines move, awards close, and links die, so a curated catalog is only
honest for as long as someone re-opens each ``source_url``.  This module is
that pass, run on a schedule (see ``docs/operations.md``).

It never rewrites a record's content.  A page that still agrees with the
record only gets its ``provenance.verified_on`` stamped; a page that
disagrees, or one that is gone, becomes a ``reverify`` proposal in the confirm
queue (:mod:`src.catalog.inbox`) for a person to accept or reject.

Comparison is deliberately conservative, because the regex extractor is
recall-oriented and the queue is only useful while it stays short:

* ``deadline`` and the amounts count only when the extractor read them from a
  labeled field (``Deadline:``, ``Amount:``) and found exactly one value, i.e.
  at :data:`MIN_CHANGE_CONFIDENCE` or better.  Dollar figures loose in body
  copy are a sponsor's endowment as often as this award's value.  A labeled
  value *does* count against a field the record leaves ``null`` -- a next
  cycle finally being announced is the whole point of the pass.
* ``status`` is read from an explicit ``Status:`` label or from unambiguous
  closed/open wording -- never from a bare "open" loose in the body.
* requirement flags count only when the record states a value and the page
  *contradicts* it.  A flag the record leaves ``null`` is left alone rather
  than filed as a change, which would propose something on nearly every page.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.catalog.inbox import INBOX_DIR, ProposalError, propose
from src.ingest.extract_common import extract_field_value
from src.ingest.prefill import Extractor, PrefillResult, prefill_from_html
from src.io.snapshotting import write_json_atomic
from src.normalize.catalog_schema import RECORDS_DIR, ROOT_DIR, iter_catalog_files

logger = logging.getLogger(__name__)

REPORTS_DIR = ROOT_DIR / "reports" / "catalog_verify"
VERIFIED_BY = "verify_catalog"

DEFAULT_SINCE_DAYS = 365
DEFAULT_REQUESTS_PER_SECOND = 0.5

# Only a labeled, unambiguous value is allowed to contradict a curated record;
# RegexExtractor scores those at 0.85 and everything looser at 0.6 or below.
MIN_CHANGE_CONFIDENCE = 0.7

OUTCOME_UNCHANGED = "unchanged"
OUTCOME_CHANGED = "changed"
OUTCOME_DEAD_LINK = "dead_link"
OUTCOME_ERROR = "error"

_REQUIREMENT_KEYS = (
    "essay",
    "recommendation_letters",
    "transcript",
    "fafsa",
    "video_or_portfolio",
    "interview",
)

_STATUS_LABELS = ["status", "application status"]
_STATUS_LABEL_KEYWORDS = (
    ("closed", "closed"),
    ("opening soon", "upcoming"),
    ("not yet open", "upcoming"),
    ("upcoming", "upcoming"),
    ("open", "open"),
)
_CLOSED_PHRASES = (
    "applications are closed",
    "applications are now closed",
    "applications have closed",
    "application is closed",
    "no longer accepting applications",
    "this scholarship is closed",
    "the deadline has passed",
)
_OPEN_PHRASES = (
    "applications are open",
    "applications are now open",
    "now accepting applications",
    "application is open",
)

_WHITESPACE_PATTERN = re.compile(r"\s+")
_VALUE_PATTERNS = {
    key: re.compile(rf'("{key}"\s*:\s*)(?:null|"[^"]*")')
    for key in ("verified_on", "verified_by")
}


@dataclass(slots=True)
class RecordVerification:
    """What one re-verification pass found for one catalog record."""

    catalog_id: str
    source_url: str
    outcome: str
    http_status: int | None = None
    content_hash: str | None = None
    content_changed: bool | None = None
    page_status: str | None = None
    changes: dict[str, dict[str, Any]] = field(default_factory=dict)
    proposal_id: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "source_url": self.source_url,
            "outcome": self.outcome,
            "http_status": self.http_status,
            "content_hash": self.content_hash,
            "content_changed": self.content_changed,
            "page_status": self.page_status,
            "changes": self.changes,
            "proposal_id": self.proposal_id,
            "message": self.message,
        }


@dataclass(slots=True)
class VerificationReport:
    """The summary one scheduled run writes under ``reports/catalog_verify/``."""

    started_at: str
    finished_at: str
    records_total: int
    records_due: int
    since_days: int
    results: list[RecordVerification] = field(default_factory=list)
    report_path: Path | None = None

    @property
    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {
            OUTCOME_UNCHANGED: 0,
            OUTCOME_CHANGED: 0,
            OUTCOME_DEAD_LINK: 0,
            OUTCOME_ERROR: 0,
        }
        for result in self.results:
            tally[result.outcome] = tally.get(result.outcome, 0) + 1
        return tally

    @property
    def proposal_ids(self) -> list[str]:
        return [result.proposal_id for result in self.results if result.proposal_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "records_total": self.records_total,
            "records_due": self.records_due,
            "records_checked": len(self.results),
            "since_days": self.since_days,
            "counts": self.counts,
            "proposals": self.proposal_ids,
            "results": [result.to_dict() for result in self.results],
        }


def content_hash(text: str | None) -> str:
    """Hash the cleaned page text, whitespace- and case-normalised."""
    cleaned = _WHITESPACE_PATTERN.sub(" ", str(text or "")).strip().lower()
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def parse_page_status(text: str | None) -> str | None:
    """Return ``open|upcoming|closed`` when the page says so outright, else ``None``."""
    if not text:
        return None

    haystack = str(text)
    labeled = extract_field_value(haystack, _STATUS_LABELS)
    if labeled:
        lowered_label = labeled.lower()
        for keyword, value in _STATUS_LABEL_KEYWORDS:
            if keyword in lowered_label:
                return value

    lowered = haystack.lower()
    if any(phrase in lowered for phrase in _CLOSED_PHRASES):
        return "closed"
    if any(phrase in lowered for phrase in _OPEN_PHRASES):
        return "open"
    return None


def compare_record(
    record: Mapping[str, Any], prefill: PrefillResult
) -> dict[str, dict[str, Any]]:
    """Return the fields where the page contradicts the record."""
    changes: dict[str, dict[str, Any]] = {}
    extraction = prefill.extraction
    confidence = prefill.confidence

    deadline = extraction.deadline
    if deadline and deadline != record.get("deadline") and _is_stated(confidence, "deadline"):
        changes["deadline"] = {"old": record.get("deadline"), "new": deadline}

    amounts = (("amount_min", extraction.amount_min), ("amount_max", extraction.amount_max))
    for key, value in amounts:
        if value is None or not _is_stated(confidence, key):
            continue
        if _as_float(record.get(key)) != float(value):
            changes[key] = {"old": record.get(key), "new": float(value)}

    page_status = parse_page_status(prefill.text)
    if page_status and page_status != record.get("status"):
        changes["status"] = {"old": record.get("status"), "new": page_status}

    requirements = _requirement_changes(record, extraction.requirements)
    if requirements:
        changes["requirements"] = requirements

    return changes


def verify_record(
    record: Mapping[str, Any],
    *,
    client: Any,
    extractor: Extractor | None = None,
    prior_hash: str | None = None,
) -> tuple[RecordVerification, dict[str, Any] | None]:
    """Fetch one record's page and classify it.

    Returns:
        The verification result and, when the page disagrees with the record
        or the link is dead, the record a ``reverify`` proposal should carry.
    """
    catalog_id = str(record.get("catalog_id") or "")
    source_url = str(record.get("source_url") or "")

    html, http_status, error = _fetch(client, source_url)
    if html is None:
        if http_status is None:
            return (
                RecordVerification(
                    catalog_id=catalog_id,
                    source_url=source_url,
                    outcome=OUTCOME_ERROR,
                    message=error,
                ),
                None,
            )
        result = RecordVerification(
            catalog_id=catalog_id,
            source_url=source_url,
            outcome=OUTCOME_DEAD_LINK,
            http_status=http_status,
            message=error,
        )
        return result, _dead_link_record(record)

    prefill = prefill_from_html(html, url=source_url, extractor=extractor)
    digest = content_hash(prefill.text)
    changes = compare_record(record, prefill)
    result = RecordVerification(
        catalog_id=catalog_id,
        source_url=source_url,
        outcome=OUTCOME_CHANGED if changes else OUTCOME_UNCHANGED,
        http_status=http_status or 200,
        content_hash=digest,
        content_changed=None if prior_hash is None else prior_hash != digest,
        page_status=parse_page_status(prefill.text),
        changes=changes,
    )
    return result, (_changed_record(record, changes) if changes else None)


def verify_catalog(
    *,
    records_dir: Path | None = None,
    inbox_dir: Path | None = None,
    reports_dir: Path | None = None,
    client: Any | None = None,
    extractor: Extractor | None = None,
    since_days: int = DEFAULT_SINCE_DAYS,
    max_records: int | None = None,
    today: date | None = None,
    requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND,
    write_report: bool = True,
) -> VerificationReport:
    """Re-verify every record due for a check and return the run's report.

    Args:
        records_dir: Catalog to verify.  Defaults to ``data/catalog/records/``.
        inbox_dir: Where proposals are written.  Defaults to ``data/catalog/inbox/``.
        reports_dir: Where the JSON summary lands, and where the previous run's
            content hashes are read from.
        client: Anything with ``get_text(url) -> str``; a rate-limited
            :class:`~src.ingest.http.PoliteHttpClient` is built when omitted.
        extractor: The :class:`Extractor` to run over each page.
        since_days: Skip records verified within this many days.
        max_records: Stop after this many fetches.
        today: Run date, used for the ``verified_on`` stamp.
        requests_per_second: Rate limit for the client built when ``client`` is None.
        write_report: Set False to run without leaving a report behind.
    """
    run_date = today or date.today()
    started_at = datetime.now(UTC)
    resolved_records = records_dir or RECORDS_DIR
    resolved_reports = reports_dir or REPORTS_DIR

    records = _load_records(resolved_records)
    due = _due_records(records, today=run_date, since_days=since_days)
    selected = due[:max_records] if max_records is not None else due
    prior_hashes = load_prior_hashes(resolved_reports)

    http_client = client
    owns_client = False
    if http_client is None and selected:
        from src.ingest.http import PoliteHttpClient

        http_client = PoliteHttpClient(requests_per_second=requests_per_second)
        owns_client = True

    results: list[RecordVerification] = []
    try:
        for record in selected:
            result, proposed = verify_record(
                record,
                client=http_client,
                extractor=extractor,
                prior_hash=prior_hashes.get(str(record.get("catalog_id") or "")),
            )
            if proposed is not None:
                _record_proposal(
                    result,
                    proposed,
                    run_date=run_date,
                    inbox_dir=inbox_dir or INBOX_DIR,
                    records_dir=resolved_records,
                )
            elif result.outcome == OUTCOME_UNCHANGED:
                _stamp_verified_on(record, resolved_records, run_date)
            results.append(result)
    finally:
        if owns_client:
            close = getattr(http_client, "close", None)
            if callable(close):
                close()

    report = VerificationReport(
        started_at=_stamp(started_at),
        finished_at=_stamp(datetime.now(UTC)),
        records_total=len(records),
        records_due=len(due),
        since_days=since_days,
        results=results,
    )
    if write_report:
        report.report_path = _write_report(report, resolved_reports, started_at)
    return report


def load_prior_hashes(reports_dir: Path | None = None) -> dict[str, str]:
    """Return ``catalog_id -> content_hash`` from the most recent readable report."""
    directory = reports_dir or REPORTS_DIR
    if not directory.is_dir():
        return {}

    for path in sorted(directory.glob("catalog_verify_*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            logger.error("verify: skipping unreadable report %s: %s", path, exc)
            continue
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            continue
        return {
            str(entry.get("catalog_id")): str(entry.get("content_hash"))
            for entry in results
            if isinstance(entry, dict) and entry.get("catalog_id") and entry.get("content_hash")
        }
    return {}


def _fetch(client: Any, url: str) -> tuple[str | None, int | None, str | None]:
    """Return ``(html, http_status, error)``; ``html`` is None when the fetch failed.

    An HTTP status on the exception is what separates a dead link from a
    laptop that was offline: only the former is worth queueing a proposal for.
    """
    if not url:
        return None, None, "record has no source_url"
    try:
        return str(client.get_text(url)), None, None
    except Exception as exc:  # a scheduled pass must survive one bad page
        status = getattr(getattr(exc, "response", None), "status_code", None)
        message = f"{type(exc).__name__}: {exc}"
        logger.warning("verify: %s failed: %s", url, message)
        return None, int(status) if isinstance(status, int) else None, message


def _load_records(records_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_catalog_files(records_dir):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            logger.error("verify: skipping unreadable record %s: %s", path, exc)
            continue
        if isinstance(payload, dict) and payload.get("catalog_id"):
            records.append(payload)
    return records


def _due_records(
    records: Sequence[Mapping[str, Any]], *, today: date, since_days: int
) -> list[dict[str, Any]]:
    """Records never verified, or verified at least ``since_days`` ago, oldest first."""
    due: list[tuple[str, str, dict[str, Any]]] = []
    for record in records:
        verified_on = _verified_on(record)
        if verified_on is not None and (today - verified_on).days < since_days:
            continue
        sort_key = verified_on.isoformat() if verified_on else ""
        due.append((sort_key, str(record.get("catalog_id")), dict(record)))
    due.sort(key=lambda item: (item[0], item[1]))
    return [record for _, _, record in due]


def _verified_on(record: Mapping[str, Any]) -> date | None:
    provenance = record.get("provenance")
    raw = provenance.get("verified_on") if isinstance(provenance, Mapping) else None
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw))
    except ValueError:
        return None


def _stamp_verified_on(record: Mapping[str, Any], records_dir: Path, run_date: date) -> None:
    """Write today's ``verified_on`` back into a record the page still agrees with.

    Edited as text rather than re-serialised, because the catalog is hand-kept
    and a two-field stamp must not reflow the whole file into a diff nobody
    can read.  A file this cannot patch is rewritten wholesale instead.
    """
    path = records_dir / f"{record['catalog_id']}.json"
    provenance = _verified_provenance(record, run_date)
    try:
        original = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        logger.error("verify: cannot read %s to stamp verified_on: %s", path, exc)
        return

    patched = original
    for key in ("verified_on", "verified_by"):
        candidate = _patch_json_value(patched, key, provenance[key])
        if candidate is None:
            updated = dict(record)
            updated["provenance"] = provenance
            write_json_atomic(updated, path)
            return
        patched = candidate

    _write_text_atomic(patched, path)


def _patch_json_value(text: str, key: str, value: Any) -> str | None:
    """Replace one scalar value in JSON source text, or ``None`` if it is not there once."""
    encoded = json.dumps(value)
    patched, substitutions = _VALUE_PATTERNS[key].subn(
        lambda match: match.group(1) + encoded, text, count=1
    )
    return patched if substitutions == 1 else None


def _write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f"{path.name}.{uuid4().hex}.tmp"
    try:
        temp_path.write_text(text, encoding="utf-8")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _verified_provenance(record: Mapping[str, Any], run_date: date) -> dict[str, Any]:
    provenance = record.get("provenance")
    updated = dict(provenance) if isinstance(provenance, Mapping) else {}
    updated["verified_on"] = run_date.isoformat()
    updated["verified_by"] = VERIFIED_BY
    return updated


def _changed_record(
    record: Mapping[str, Any], changes: Mapping[str, dict[str, Any]]
) -> dict[str, Any]:
    proposed = dict(record)
    for key, change in changes.items():
        proposed[key] = change["new"]
    return proposed


def _dead_link_record(record: Mapping[str, Any]) -> dict[str, Any]:
    proposed = dict(record)
    proposed["status"] = "unknown"
    return proposed


def _record_proposal(
    result: RecordVerification,
    proposed: dict[str, Any],
    *,
    run_date: date,
    inbox_dir: Path,
    records_dir: Path,
) -> None:
    """Queue the proposal and note its id on ``result``, or why it could not be queued."""
    if result.outcome == OUTCOME_DEAD_LINK:
        notes = (
            f"Annual re-verification: {result.source_url} returned HTTP {result.http_status}. "
            "Status set to unknown pending a working link."
        )
    else:
        proposed["provenance"] = _verified_provenance(proposed, run_date)
        fields = ", ".join(sorted(result.changes))
        notes = f"Annual re-verification: the page disagrees with the record on {fields}."

    try:
        proposal = propose(
            proposed,
            kind="reverify",
            notes=notes,
            created_on=run_date,
            inbox_dir=inbox_dir,
            records_dir=records_dir,
        )
    except ProposalError as exc:
        result.message = f"could not queue a proposal: {exc}"
        logger.error("verify: %s: %s", result.catalog_id, result.message)
        return
    result.proposal_id = proposal.proposal_id


def _requirement_changes(
    record: Mapping[str, Any], extracted: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return an old/new pair when the page contradicts a requirement the record states."""
    stated = record.get("requirements")
    if not isinstance(stated, Mapping) or not extracted:
        return None

    merged = dict(stated)
    contradicted = False
    for key in _REQUIREMENT_KEYS:
        new_value = extracted.get(key)
        old_value = stated.get(key)
        if new_value is None or old_value is None or old_value == new_value:
            continue
        merged[key] = new_value
        contradicted = True

    if not contradicted:
        return None
    return {"old": dict(stated), "new": merged}


def _write_report(report: VerificationReport, reports_dir: Path, started_at: datetime) -> Path:
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    path = reports_dir / f"catalog_verify_{stamp}.json"
    write_json_atomic(report.to_dict(), path)
    return path


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_stated(confidence: Mapping[str, float], field_name: str) -> bool:
    """True when the extractor read this field from a label rather than body copy."""
    return confidence.get(field_name, 0.0) >= MIN_CHANGE_CONFIDENCE
