"""Turn feed rows into ``feed`` proposals, quietly.

A feed row can rank for a parent, but it reaches the student only after a
person confirms it into a curated record with grades.  This module decides
which rows are worth asking about:

* new, or its deadline, status or amount changed since the last run;
* not an award the curated catalog already carries with the same values;
* not rejected before with the same content;
* passing Stage 1 for the saved profile, with the grade axis ignored, so an
  award she can apply to in two years still counts;
* at most :data:`DEFAULT_CAP` a run.  The rest are counted, not queued.

A source with no saved state writes its baseline and proposes nothing, so the
first run does not drop the whole backlog into the inbox.  State holds feed
values only (``reports/discovery/feed_state.json``, git-ignored); nothing about
the profile is written anywhere but the proposals themselves.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.catalog import inbox
from src.ingest.dedupe import collapse_cross_source_duplicates
from src.io.snapshotting import SUPERSEDED_IDS_COLUMN, write_json_atomic
from src.normalize.catalog_schema import RECORDS_DIR, ROOT_DIR, iter_catalog_files
from src.rank.stage1_eligibility import StudentProfile, apply_eligibility_filter

logger = logging.getLogger(__name__)

STATE_PATH = ROOT_DIR / "reports" / "discovery" / "feed_state.json"
DEFAULT_CAP = 15

WATCHED_FIELDS = ("deadline", "status", "amount_min", "amount_max")
_FIELD_LABELS = {
    "deadline": "deadline",
    "status": "status",
    "amount_min": "minimum amount",
    "amount_max": "amount",
}

# Catalog fields a feed row may carry; identity, trust and provenance are set here.
_COPIED_FIELDS = (
    "sponsor",
    "description",
    "eligibility_text",
    "amount_min",
    "amount_max",
    "deadline",
    "cycle",
    "status",
    "education_level",
    "grade_levels",
    "majors_allowed",
    "states_allowed",
    "counties_allowed",
    "min_gpa",
    "citizenship",
    "need_based",
    "first_gen_only",
    "gender",
    "heritage",
    "military_family",
    "disability",
    "religion",
    "employer_restricted",
    "membership_required",
    "min_test_scores",
    "requirements",
    "renewal_terms",
    "keywords",
)

# The screen ignores the grade axis: an award for a later grade is a future target.
_SCREEN_IGNORED_REASONS = frozenset({"GRADE_LEVEL_MISMATCH"})

LEFT_OUT_REASONS = (
    "baseline",
    "unchanged",
    "covered_by_curated",
    "previously_rejected",
    "failed_profile_screen",
    "over_cap",
)

_SLUG_UNSAFE = re.compile(r"[^a-z0-9]+")
_MAX_KEY_CHARS = 80


@dataclass(slots=True)
class FeedProposal:
    """One row this run would put in the inbox."""

    proposal_id: str
    record: dict[str, Any]
    why: str
    notes: str


@dataclass(slots=True)
class FeedPlan:
    """What one source's rows come to: proposals, left-out counts, the next state."""

    source: str
    proposals: list[FeedProposal] = field(default_factory=list)
    left_out: dict[str, int] = field(default_factory=lambda: dict.fromkeys(LEFT_OUT_REASONS, 0))
    rows_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    baseline: bool = False


@dataclass(slots=True)
class DiscoveryResult:
    """What a run wrote: the proposal ids and the counts of everything left out."""

    source: str
    rows: int
    baseline: bool
    proposed: list[str]
    left_out: dict[str, int]


def row_key(row: Mapping[str, Any]) -> str:
    """The row's identity within its source: the feed's id, else its URL or title."""
    for key in ("source_id", "source_url", "title"):
        slug = _slugify(row.get(key))
        if slug:
            return slug[:_MAX_KEY_CHARS].strip("-")
    return "untitled"


def feed_proposal_id(source: str, row: Mapping[str, Any]) -> str:
    """``feed-<source>-<source id>``: a re-run replaces its own proposal, sources never collide."""
    return f"feed-{_slugify(source)}-{row_key(row)}"


def row_fingerprint(row: Mapping[str, Any]) -> dict[str, Any]:
    return {name: _json_value(row.get(name)) for name in WATCHED_FIELDS}


def plan_feed_proposals(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str,
    prior_rows: Mapping[str, Mapping[str, Any]] | None,
    curated_records: Sequence[Mapping[str, Any]],
    profile: StudentProfile | None,
    today: date,
    cap: int = DEFAULT_CAP,
    is_rejected: Callable[[str, Mapping[str, Any]], bool] | None = None,
) -> FeedPlan:
    """Decide which of one source's rows to propose.  Pure apart from ``is_rejected``.

    Args:
        rows: This run's parsed rows from ``source``.
        prior_rows: ``{row key: fingerprint}`` from the last run, or ``None``
            when the source has no state yet (a baseline run).
        curated_records: The curated catalog, for the cross-source match.
        profile: The saved profile for the Stage 1 screen; ``None`` screens nothing.
        is_rejected: ``(proposal_id, record) -> bool`` for Task 1.2's rule.
    """
    plan = FeedPlan(source=source)
    keyed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        keyed.setdefault(row_key(row), row)
    plan.rows_state = {key: row_fingerprint(row) for key, row in sorted(keyed.items())}

    if prior_rows is None:
        plan.baseline = True
        plan.left_out["baseline"] = len(keyed)
        return plan

    changed: dict[str, Mapping[str, Any]] = {}
    for key, row in keyed.items():
        if key in prior_rows and dict(prior_rows[key]) == plan.rows_state[key]:
            plan.left_out["unchanged"] += 1
        else:
            changed[key] = row
    if not changed:
        return plan

    matches = _curated_matches(changed, curated_records, source=source)
    candidates: list[FeedProposal] = []
    unmatched: dict[str, Mapping[str, Any]] = {}
    for key, row in changed.items():
        curated = matches.get(key)
        if curated is None:
            unmatched[key] = row
            continue
        disagreements = _disagreements(row, curated)
        if not disagreements:
            plan.left_out["covered_by_curated"] += 1
            continue
        record = {**dict(curated), **{name: _json_value(row.get(name)) for name in disagreements}}
        fields = ", ".join(_FIELD_LABELS[name] for name in disagreements)
        candidates.append(
            FeedProposal(
                proposal_id=feed_proposal_id(source, row),
                record=record,
                why=_why(prior_rows.get(key), plan.rows_state[key]),
                notes=f"The feed disagrees with the catalog record on {fields}.",
            )
        )

    passing = _profile_screen(unmatched, profile=profile, today=today)
    plan.left_out["failed_profile_screen"] += len(unmatched) - len(passing)
    taken_ids = {str(record.get("catalog_id") or "") for record in curated_records}
    for key in passing:
        row = unmatched[key]
        record = _new_award_record(row, source=source, taken_ids=taken_ids, today=today)
        taken_ids.add(record["catalog_id"])
        candidates.append(
            FeedProposal(
                proposal_id=feed_proposal_id(source, row),
                record=record,
                why=_why(prior_rows.get(key), plan.rows_state[key]),
                notes="Not in the catalog yet. Add the grades it is open to before confirming.",
            )
        )

    candidates.sort(key=_cap_order)
    for candidate in candidates:
        if is_rejected is not None and is_rejected(candidate.proposal_id, candidate.record):
            plan.left_out["previously_rejected"] += 1
        elif len(plan.proposals) >= cap:
            plan.left_out["over_cap"] += 1
        else:
            plan.proposals.append(candidate)
    return plan


def run_feed_discovery(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str,
    profile: Mapping[str, Any] | None,
    today: date | None = None,
    cap: int = DEFAULT_CAP,
    state_path: Path | None = None,
    inbox_dir: Path | None = None,
    records_dir: Path | None = None,
    rejected_dir: Path | None = None,
    dry_run: bool = False,
) -> DiscoveryResult:
    """Plan one source's rows, write the proposals and the new state.

    ``profile`` is a stored profile dict (``src.profile.store.load_profile``),
    or ``None`` when no profile is saved.  ``dry_run`` writes nothing.
    """
    from src.profile.store import to_stage1_profile

    run_date = today or date.today()
    resolved_state = state_path or STATE_PATH
    resolved_records = records_dir or RECORDS_DIR
    state = load_state(resolved_state)
    prior = state["sources"].get(source)
    prior_rows = prior.get("rows") if isinstance(prior, dict) else None

    def is_rejected(proposal_id: str, record: Mapping[str, Any]) -> bool:
        return inbox.is_previously_rejected(
            record,
            proposal_id=proposal_id,
            kind="feed",
            inbox_dir=inbox_dir,
            records_dir=resolved_records,
            rejected_dir=rejected_dir,
        )

    plan = plan_feed_proposals(
        rows,
        source=source,
        prior_rows=prior_rows if isinstance(prior_rows, dict) else None,
        curated_records=load_curated_records(resolved_records),
        profile=to_stage1_profile(dict(profile), today=run_date) if profile else None,
        today=run_date,
        cap=cap,
        is_rejected=is_rejected,
    )

    proposed: list[str] = []
    for candidate in plan.proposals:
        if dry_run:
            proposed.append(candidate.proposal_id)
            continue
        try:
            inbox.propose(
                candidate.record,
                kind="feed",
                proposal_id=candidate.proposal_id,
                notes=candidate.notes,
                source=source,
                why=candidate.why,
                created_on=run_date,
                inbox_dir=inbox_dir,
                records_dir=resolved_records,
                rejected_dir=rejected_dir,
            )
        except inbox.PreviouslyRejectedError:
            plan.left_out["previously_rejected"] += 1
            continue
        except inbox.ProposalError as exc:
            logger.error("discovery: could not queue %s: %s", candidate.proposal_id, exc)
            continue
        proposed.append(candidate.proposal_id)

    if not dry_run:
        state["sources"][source] = {"run_date": run_date.isoformat(), "rows": plan.rows_state}
        write_json_atomic(state, resolved_state)

    return DiscoveryResult(
        source=source,
        rows=len(plan.rows_state),
        baseline=plan.baseline,
        proposed=proposed,
        left_out=dict(plan.left_out),
    )


def load_state(path: Path | None = None) -> dict[str, Any]:
    """Return ``{"sources": {name: {"run_date", "rows"}}}``; empty when unreadable."""
    resolved = path or STATE_PATH
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {"sources": {}}
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        logger.error("discovery: unreadable state %s (%s); treating as a first run", resolved, exc)
        return {"sources": {}}
    sources = payload.get("sources") if isinstance(payload, dict) else None
    return {"sources": sources if isinstance(sources, dict) else {}}


def load_curated_records(records_dir: Path | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in iter_catalog_files(records_dir):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            logger.error("discovery: skipping unreadable record %s: %s", path, exc)
            continue
        if isinstance(payload, dict) and payload.get("catalog_id"):
            records.append(payload)
    return records


def _curated_matches(
    rows: Mapping[str, Mapping[str, Any]],
    curated_records: Sequence[Mapping[str, Any]],
    *,
    source: str,
) -> dict[str, Mapping[str, Any]]:
    """``{row key: curated record}`` for rows the ingest cross-source match folds into one."""
    if not rows or not curated_records:
        return {}
    curated_source = "curated_catalog"
    frame_rows: list[dict[str, Any]] = []
    for record in curated_records:
        frame_rows.append(
            {
                "scholarship_id": f"curated:{record['catalog_id']}",
                "source": curated_source,
                "source_url": record.get("source_url"),
                "title": record.get("title"),
                "sponsor": record.get("sponsor"),
                "aliases": list(record.get("aliases") or []),
                "trust": record.get("trust"),
            }
        )
    for key, row in rows.items():
        frame_rows.append(
            {
                "scholarship_id": f"feed:{key}",
                "source": source,
                "source_url": row.get("source_url"),
                "title": row.get("title"),
                "sponsor": row.get("sponsor"),
                "aliases": [],
                "trust": row.get("trust"),
            }
        )
    collapsed, _ = collapse_cross_source_duplicates(
        pd.DataFrame(frame_rows), source_order=[curated_source, source]
    )

    by_id = {f"curated:{record['catalog_id']}": record for record in curated_records}
    matches: dict[str, Mapping[str, Any]] = {}
    for _, winner in collapsed.iterrows():
        curated = by_id.get(str(winner["scholarship_id"]))
        if curated is None:
            continue
        for loser in winner[SUPERSEDED_IDS_COLUMN] or []:
            if str(loser).startswith("feed:"):
                matches[str(loser)[len("feed:"):]] = curated
    return matches


def _disagreements(row: Mapping[str, Any], curated: Mapping[str, Any]) -> list[str]:
    """Watched fields the feed states and the curated record holds differently."""
    names = []
    for name in WATCHED_FIELDS:
        value = _json_value(row.get(name))
        if value is None or (name == "status" and value == "unknown"):
            continue
        if value != _json_value(curated.get(name)):
            names.append(name)
    return names


def _profile_screen(
    rows: Mapping[str, Mapping[str, Any]], *, profile: StudentProfile | None, today: date
) -> list[str]:
    """Keys of the rows that pass Stage 1 for ``profile``, grade ignored, in input order."""
    if profile is None or not rows:
        return list(rows)
    screened = replace(profile, today=today)
    frame = pd.DataFrame([dict(row) for row in rows.values()], index=list(rows))
    eligible, ineligible = apply_eligibility_filter(frame, screened, include_unconfirmed=True)
    passing = set(eligible.index)
    for key, reasons in ineligible["reasons"].items():
        if set(reasons) <= _SCREEN_IGNORED_REASONS:
            passing.add(key)
    return [key for key in rows if key in passing]


def _new_award_record(
    row: Mapping[str, Any], *, source: str, taken_ids: set[str], today: date
) -> dict[str, Any]:
    title = str(row.get("title") or "").strip()
    catalog_id = _slugify(title)[:_MAX_KEY_CHARS].strip("-") or row_key(row)
    if catalog_id in taken_ids:
        catalog_id = f"{catalog_id}-{row_key(row)}"[: _MAX_KEY_CHARS * 2].strip("-")

    record: dict[str, Any] = {"catalog_id": catalog_id, "title": title, "kind": "scholarship"}
    record["source_url"] = row.get("source_url") or ""
    for name in _COPIED_FIELDS:
        value = _json_value(row.get(name))
        if _given(value):
            record[name] = value
    record["trust"] = "unverified"

    raw_provenance = row.get("provenance")
    feed_provenance: Mapping[str, Any] = raw_provenance if isinstance(raw_provenance, Mapping) else {}
    record["provenance"] = {
        "added_on": feed_provenance.get("added_on") or today.isoformat(),
        "verified_on": None,
        "verified_by": None,
        "source_kind": "structured_feed",
    }
    license_line = _license_line(feed_provenance, source)
    feed_notes = str(row.get("notes") or "").strip()
    record["notes"] = " ".join(part for part in (feed_notes, license_line) if part)
    return record


def _license_line(provenance: Mapping[str, Any], source: str) -> str:
    attribution = str(provenance.get("attribution") or source).strip()
    license_name = str(provenance.get("license") or "").strip()
    if not license_name or license_name.replace("-", " ") in attribution:
        return f"Source data: {attribution}."
    return f"Source data: {attribution} ({license_name})."


def _why(prior: Mapping[str, Any] | None, current: Mapping[str, Any]) -> str:
    if prior is None:
        return "New in the feed since the last run."
    changes = [
        f"{_FIELD_LABELS[name]} {_shown(prior.get(name))} → {_shown(current.get(name))}"
        for name in WATCHED_FIELDS
        if prior.get(name) != current.get(name)
    ]
    return "The feed changed " + "; ".join(changes) + "."


def _shown(value: Any) -> str:
    if value is None:
        return "not stated"
    if isinstance(value, float):
        return f"${value:,.0f}"
    return str(value)


def _cap_order(candidate: FeedProposal) -> tuple[bool, str, str]:
    """Soonest dated deadline first, then undated, then by id: the cap keeps what is urgent."""
    deadline = candidate.record.get("deadline")
    return (deadline is None, str(deadline or ""), candidate.proposal_id)


def _given(value: Any) -> bool:
    if value is None or value == [] or value == "":
        return False
    if isinstance(value, dict):
        return any(_given(item) for item in value.values())
    return True


def _json_value(value: Any) -> Any:
    """A JSON-safe, comparable value: dates as ISO text, NaN as None."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return None if pd.isna(value) else value
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    text = str(value).strip()
    return text or None


def _slugify(value: Any) -> str:
    return _SLUG_UNSAFE.sub("-", str(value or "").strip().lower()).strip("-")
