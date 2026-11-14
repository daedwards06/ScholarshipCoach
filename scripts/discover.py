"""Weekly discovery: refresh the feed sources, then propose what changed.

Runs ingest for the enabled feed sources alone, with the usual guardrails (a
record drop of more than half still blocks the snapshot write; the curated
rows are carried forward from the prior snapshot), then hands each source's
rows to :mod:`src.catalog.discovery`, which writes at most a few ``feed``
proposals to the inbox.  Nothing here confirms a proposal or edits a catalog
record.  The server's Sunday 05:00 timer runs it.

    python scripts/discover.py
    python scripts/discover.py --dry-run

Writes ``reports/discovery/discovery_<UTC stamp>.json``.  ``--dry-run``
fetches and reports, and writes nothing.

Exit 0 for a completed pass, including one that found nothing new; 1 when a
feed source failed (a source that returned records before and none now
counts as failed) or a guardrail blocked the write.  A source that did not
succeed is not planned, so its state is kept for the next run.
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts import run_ingest
from src.catalog.discovery import run_feed_discovery
from src.ingest.registry import register_sources
from src.ingest.sources.open_scholarships import OpenScholarshipsSource
from src.io.snapshotting import write_json_atomic
from src.profile.store import load_profile

REPORT_DIR = run_ingest.ROOT_DIR / "reports" / "discovery"

# Connectors whose rows are proposed.  Scrapers are not feeds: their rows stay
# ineligible aggregator rows until a person adds them through the inbox.
FEED_SOURCES = (OpenScholarshipsSource.name,)


def enabled_feed_sources() -> list[str]:
    enabled = {source.name for source in register_sources()}
    return [name for name in FEED_SOURCES if name in enabled]


def run_discovery(*, dry_run: bool = False, report_dir: Path | None = None) -> dict[str, Any]:
    """Refresh the feeds, plan proposals per source, and return the discovery report."""
    started_at = datetime.now(tz=UTC)
    run_date = started_at.date()
    names = enabled_feed_sources()
    rows: dict[str, list[dict[str, Any]]] = {}
    ingest: dict[str, Any] | None = None
    if names:
        ingest = run_ingest.run_ingest(
            date=run_date, only_sources=names, dry_run=dry_run, parsed_rows=rows
        )
    blocked = bool(ingest and ingest["artifact_notes"]["snapshot_blocked"])
    profile = load_profile()

    sources: list[dict[str, Any]] = []
    for detail in ingest["sources"]["details"] if ingest else []:
        name = detail["source"]
        entry: dict[str, Any] = {
            "source": name,
            "status": detail["status"],
            "records": int(detail.get("records", 0)),
            "zero_record_regression": bool(detail["health"]["zero_record_regression"]),
            "error": detail.get("error"),
            "discovery": None,
            "discovery_skipped": None,
        }
        if blocked:
            entry["discovery_skipped"] = "guardrail blocked the snapshot write"
        elif detail["status"] != "succeeded":
            entry["discovery_skipped"] = f"source {detail['status']}"
        else:
            result = run_feed_discovery(
                rows.get(name, []), source=name, profile=profile, today=run_date, dry_run=dry_run
            )
            entry["discovery"] = {
                "baseline": result.baseline,
                "rows": result.rows,
                "proposals_written": len(result.proposed),
                "proposed": result.proposed,
                "left_out": result.left_out,
            }
        sources.append(entry)

    failed = blocked or any(entry["status"] == "failed" for entry in sources)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    report_path = (report_dir or REPORT_DIR) / f"discovery_{stamp}.json"
    report: dict[str, Any] = {
        "status": "failed" if failed else "completed",
        "run_started_at": started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_date": run_date.isoformat(),
        "dry_run": dry_run,
        "feed_sources": names,
        "guardrail": {
            "blocked": blocked,
            "warnings": list(ingest["guardrail_warnings"]) if ingest else [],
        },
        "snapshot": ingest["artifact_paths"]["snapshot"] if ingest else None,
        "ingest_report": ingest["artifact_paths"]["report"] if ingest else None,
        "sources": sources,
        "report": None if dry_run else str(report_path.resolve()),
    }
    if not dry_run:
        write_json_atomic(report, report_path)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the feed sources and propose new or changed awards to the inbox."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and report only: no snapshot, proposals, state or report are written.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = run_discovery(dry_run=args.dry_run)

    if not report["feed_sources"]:
        print("No feed source is enabled in data/catalog/sources.json; nothing to do.")
    for entry in report["sources"]:
        line = f"{entry['source']}: {entry['status']}, {entry['records']} records"
        if entry["zero_record_regression"]:
            line += " (returned records on the prior run, none now)"
        discovery = entry["discovery"]
        if discovery is None:
            line += f"; not planned: {entry['discovery_skipped']}"
        elif discovery["baseline"]:
            line += "; first run for this source: a baseline, nothing proposed"
        else:
            left_out = ", ".join(f"{k}={v}" for k, v in discovery["left_out"].items() if v)
            line += f"; {discovery['proposals_written']} proposed"
            line += f" (left out: {left_out})" if left_out else ""
        print(line)
    for warning in report["guardrail"]["warnings"]:
        print(f"Guardrail: {warning}")
    if report["guardrail"]["blocked"]:
        print("Snapshot NOT written: a guardrail blocked it.")
    elif report["snapshot"]:
        print(f"Wrote snapshot: {report['snapshot']}")
    if report["dry_run"]:
        print("Dry run: nothing written.")
    else:
        print(f"Wrote discovery report: {report['report']}")
    print(f"Discovery {report['status']}.")
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
