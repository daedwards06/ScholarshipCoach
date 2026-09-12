"""The confirm queue that stands between automation and the curated catalog.

Feeds, URL prefill and annual re-verification never write to
``data/catalog/records/``.  They write *proposals* into
``data/catalog/inbox/``, and a person confirms or rejects each one.  That is
the rule that keeps automated extraction -- which is wrong often enough to
matter -- out of the catalog the family actually applies from.

A proposal file is a partial catalog record plus ``proposal`` metadata: its
kind, the day it was created, and a field-level diff against the record it
would replace.  :func:`confirm` is the only path from the inbox into
``records/``, and it validates against ``data/catalog/schema.json`` first, so
a proposal that would write an invalid record stays in the queue instead.

Proposal ids are deterministic, so re-running a feed or a re-verification pass
replaces the pending proposal for an award rather than stacking duplicates.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from src.io.snapshotting import write_json_atomic
from src.normalize.catalog_schema import (
    CATALOG_DIR,
    RECORDS_DIR,
    load_catalog_schema,
    validate_catalog_record,
)

logger = logging.getLogger(__name__)

INBOX_DIR = CATALOG_DIR / "inbox"
REJECTED_DIR = INBOX_DIR / "rejected"

PROPOSAL_KINDS = ("prefill", "feed", "reverify", "manual")

_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_SLUG_UNSAFE_PATTERN = re.compile(r"[^a-z0-9]+")
_MAX_PROPOSAL_ID_CHARS = 120


class ProposalError(RuntimeError):
    """A proposal could not be written, found, or confirmed."""


@dataclass(slots=True)
class Proposal:
    """One pending change to the curated catalog, awaiting a person."""

    proposal_id: str
    kind: str
    record: dict[str, Any]
    created_on: str
    diff: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: str | None = None
    path: Path | None = None

    @property
    def catalog_id(self) -> str | None:
        value = self.record.get("catalog_id")
        return str(value) if value else None

    @property
    def title(self) -> str | None:
        value = self.record.get("title")
        return str(value) if value else None

    def to_dict(self) -> dict[str, Any]:
        """Return the on-disk payload: ``proposal`` metadata plus the record."""
        return {
            "proposal": {
                "proposal_id": self.proposal_id,
                "kind": self.kind,
                "created_on": self.created_on,
                "diff": self.diff,
                "notes": self.notes,
            },
            "record": self.record,
        }

    @classmethod
    def from_dict(cls, payload: Any, *, path: Path | None = None) -> Proposal:
        """Rebuild a proposal from its on-disk payload."""
        label = str(path) if path is not None else "<payload>"
        if not isinstance(payload, dict):
            raise ProposalError(f"{label}: expected a JSON object")

        meta = payload.get("proposal")
        record = payload.get("record")
        if not isinstance(meta, dict) or not isinstance(record, dict):
            raise ProposalError(f"{label}: expected 'proposal' and 'record' objects")

        proposal_id = str(meta.get("proposal_id") or (path.stem if path else "")).strip()
        if not proposal_id:
            raise ProposalError(f"{label}: missing proposal_id")

        diff = meta.get("diff")
        notes = meta.get("notes")
        return cls(
            proposal_id=proposal_id,
            kind=str(meta.get("kind") or "manual"),
            record=record,
            created_on=str(meta.get("created_on") or ""),
            diff=diff if isinstance(diff, dict) else {},
            notes=str(notes) if notes else None,
            path=path,
        )


def propose(
    record: Mapping[str, Any],
    *,
    kind: str,
    notes: str | None = None,
    proposal_id: str | None = None,
    created_on: date | str | None = None,
    inbox_dir: Path | None = None,
    records_dir: Path | None = None,
) -> Proposal:
    """Write ``record`` to the inbox as a proposal and return it.

    The record is deliberately not schema-validated here: automation is allowed
    to propose something incomplete, and a person fills the gaps at
    :func:`confirm` time.

    Args:
        record: A partial catalog record.
        kind: One of :data:`PROPOSAL_KINDS`.
        notes: Free text for whoever works the queue.
        proposal_id: Overrides the deterministic id derived from the record.
        created_on: Defaults to today.
        inbox_dir: Defaults to ``data/catalog/inbox/``.
        records_dir: Catalog searched for the record this proposal would
            replace, to compute ``diff``.  Defaults to ``data/catalog/records/``.

    Raises:
        ProposalError: On an unknown ``kind``, an empty record, or an id that is
            not a safe slug.
    """
    if kind not in PROPOSAL_KINDS:
        raise ProposalError(f"unknown proposal kind {kind!r}; expected one of {PROPOSAL_KINDS}")
    if not isinstance(record, Mapping) or not record:
        raise ProposalError("a proposal needs a non-empty record")

    payload_record = dict(record)
    resolved_id = proposal_id or _default_proposal_id(payload_record, kind)
    path = _proposal_path(resolved_id, inbox_dir)

    proposal = Proposal(
        proposal_id=resolved_id,
        kind=kind,
        record=payload_record,
        created_on=_as_iso_date(created_on),
        diff=_diff_against_catalog(payload_record, records_dir),
        notes=notes,
        path=path,
    )
    write_json_atomic(proposal.to_dict(), path)
    return proposal


def list_proposals(inbox_dir: Path | None = None) -> list[Proposal]:
    """Return every pending proposal, sorted by id.

    An unreadable or malformed file is logged and skipped so one bad write
    never hides the rest of the queue.
    """
    directory = inbox_dir or INBOX_DIR
    if not directory.is_dir():
        return []

    proposals: list[Proposal] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            proposals.append(Proposal.from_dict(payload, path=path))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError, ProposalError) as exc:
            logger.error("inbox: skipping unreadable proposal %s: %s", path, exc)
    return proposals


def load_proposal(proposal_id: str, inbox_dir: Path | None = None) -> Proposal:
    """Return one pending proposal by id."""
    path = _proposal_path(proposal_id, inbox_dir)
    if not path.is_file():
        raise ProposalError(f"no proposal {proposal_id!r} in {path.parent}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise ProposalError(f"{path}: unreadable proposal ({exc})") from exc
    return Proposal.from_dict(payload, path=path)


def confirm(
    proposal_id: str,
    edits: Mapping[str, Any] | None = None,
    *,
    inbox_dir: Path | None = None,
    records_dir: Path | None = None,
    schema_path: Path | None = None,
) -> Path:
    """Promote a proposal into the curated catalog and clear it from the inbox.

    ``edits`` is merged over the proposed record at the top level, so a person
    can supply the fields automation could not read.  The merged record must
    pass the catalog schema; if it does not, nothing is written and the
    proposal stays in the queue.

    Returns:
        The path of the record written under ``records/``.
    """
    proposal = load_proposal(proposal_id, inbox_dir)
    merged = {**proposal.record, **dict(edits or {})}

    catalog_id = str(merged.get("catalog_id") or "").strip()
    if not catalog_id:
        raise ProposalError(f"proposal {proposal_id!r} has no catalog_id; supply one as an edit")

    errors = validate_catalog_record(merged, schema=load_catalog_schema(schema_path))
    if errors:
        raise ProposalError(
            f"proposal {proposal_id!r} is not a valid catalog record: " + "; ".join(errors)
        )

    record_path = (records_dir or RECORDS_DIR) / f"{catalog_id}.json"
    write_json_atomic(merged, record_path)
    if proposal.path is not None:
        proposal.path.unlink(missing_ok=True)
    return record_path


def reject(
    proposal_id: str,
    reason: str,
    *,
    inbox_dir: Path | None = None,
    rejected_dir: Path | None = None,
) -> Path:
    """Archive a proposal under ``inbox/rejected/`` with the reason it was refused.

    Rejections are kept rather than deleted so a later re-verification pass can
    tell that a person already looked at this award and said no.
    """
    cleaned = reason.strip()
    if not cleaned:
        raise ProposalError("a rejection needs a reason")

    proposal = load_proposal(proposal_id, inbox_dir)
    payload = proposal.to_dict()
    payload["proposal"]["rejected_on"] = _as_iso_date(None)
    payload["proposal"]["reason"] = cleaned

    target_dir = rejected_dir or ((inbox_dir / "rejected") if inbox_dir else REJECTED_DIR)
    target = target_dir / f"{proposal.proposal_id}.json"
    write_json_atomic(payload, target)
    if proposal.path is not None:
        proposal.path.unlink(missing_ok=True)
    return target


def _proposal_path(proposal_id: str, inbox_dir: Path | None) -> Path:
    """Resolve a proposal id to its file, refusing anything but a bare slug."""
    cleaned = proposal_id.strip()
    if not _SLUG_PATTERN.match(cleaned):
        raise ProposalError(f"proposal_id {proposal_id!r} is not a lowercase slug")
    return (inbox_dir or INBOX_DIR) / f"{cleaned}.json"


def _default_proposal_id(record: Mapping[str, Any], kind: str) -> str:
    """Derive a stable id: the same award proposed twice replaces its own proposal."""
    for key in ("catalog_id", "title", "source_url"):
        slug = _slugify(str(record.get(key) or ""))
        if slug:
            return f"{kind}-{slug}"[:_MAX_PROPOSAL_ID_CHARS].strip("-")
    return f"{kind}-untitled"


def _slugify(value: str) -> str:
    return _SLUG_UNSAFE_PATTERN.sub("-", value.strip().lower()).strip("-")


def _as_iso_date(value: date | str | None) -> str:
    if value is None:
        return date.today().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _diff_against_catalog(
    record: Mapping[str, Any], records_dir: Path | None
) -> dict[str, dict[str, Any]]:
    """Field-level diff of ``record`` against the catalog record it would replace.

    Only fields the proposal actually carries are compared -- a partial record
    must not read as "deleted everything it did not mention".  An empty diff
    means either a brand-new award or a proposal that changes nothing.
    """
    catalog_id = str(record.get("catalog_id") or "").strip()
    if not catalog_id:
        return {}

    existing_path = (records_dir or RECORDS_DIR) / f"{catalog_id}.json"
    if not existing_path.is_file():
        return {}

    try:
        existing = json.loads(existing_path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        logger.error("inbox: cannot diff against %s: %s", existing_path, exc)
        return {}
    if not isinstance(existing, dict):
        return {}

    diff: dict[str, dict[str, Any]] = {}
    for key in sorted(record):
        old_value = existing.get(key)
        new_value = record[key]
        if old_value != new_value:
            diff[key] = {"old": old_value, "new": new_value}
    return diff
