"""Re-read the watch list's pages and propose what changed.

Each due entry in ``data/catalog/watchlist.json`` (and, on the server, the
private overlay ``data/private/watchlist.json``) is fetched politely and
compared with the last run.  A changed page becomes ``prefill`` proposals in
the inbox, or a finding to open by hand; nothing here confirms a proposal or
edits a catalog record (see ``docs/operations.md``, "Watch list").

    python scripts/check_watchlist.py
    python scripts/check_watchlist.py --dry-run --ignore-months

Writes ``reports/watchlist/watchlist_<UTC stamp>.json`` and the state beside
it.  ``--dry-run`` fetches and reports, and writes nothing.

Exit 0 when the pass ran, whatever it found; 1 when a watch list file is
unreadable or invalid, since its entries were then not checked.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from src.catalog.inbox import INBOX_DIR
from src.catalog.verify import DEFAULT_REQUESTS_PER_SECOND
from src.catalog.watch import (
    PRIVATE_WATCHLIST_PATH,
    REPORTS_DIR,
    WATCHLIST_PATH,
    EntryResult,
    check_watchlist,
)
from src.normalize.catalog_schema import RECORDS_DIR


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-read the watch list's pages and propose new or changed awards."
    )
    parser.add_argument("--watchlist", type=Path, default=WATCHLIST_PATH)
    parser.add_argument("--private-watchlist", type=Path, default=PRIVATE_WATCHLIST_PATH)
    parser.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    parser.add_argument("--inbox-dir", type=Path, default=INBOX_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and report only: no proposals, state or report are written.",
    )
    parser.add_argument(
        "--ignore-months",
        action="store_true",
        help="Check every enabled entry, not only those whose months include this one.",
    )
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=DEFAULT_REQUESTS_PER_SECOND,
        help="Polite rate limit for page fetches (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _summarize(result: EntryResult) -> str:
    detail = result.message or ""
    if result.proposals:
        detail = f"{len(result.proposals)} proposed: {', '.join(result.proposals)}"
    elif result.findings:
        detail = "; ".join(f"{f.reason} ({f.url})" for f in result.findings)
    left_out = ", ".join(f"{k}={v}" for k, v in result.left_out.items() if v)
    if left_out:
        detail = f"{detail} (left out: {left_out})".strip()
    private = " [private]" if result.private else ""
    return f"  {result.entry_id:<40} {result.outcome:<10} {detail}{private}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    report = check_watchlist(
        watchlist_path=args.watchlist,
        private_watchlist_path=args.private_watchlist,
        records_dir=args.records_dir,
        inbox_dir=args.inbox_dir,
        reports_dir=args.reports_dir,
        ignore_months=args.ignore_months,
        dry_run=args.dry_run,
        requests_per_second=args.requests_per_second,
    )

    for problem in report.problems:
        print(f"Watch list problem: {problem}")
    print(f"{report.entries_total} watch entr{'y' if report.entries_total == 1 else 'ies'}.")
    for result in report.results:
        print(_summarize(result))
    counts = " ".join(f"{name}={count}" for name, count in report.counts.items())
    print(counts)
    if report.findings:
        print()
        print(f"{len(report.findings)} finding(s) to open by hand:")
        for finding in report.findings:
            print(f"  {finding.entry_id}: {finding.message} {finding.url}")
    if report.proposal_ids and not report.dry_run:
        print(f"{len(report.proposal_ids)} proposal(s) queued in {args.inbox_dir}.")
    if report.dry_run:
        print("Dry run: nothing written.")
    elif report.report_path is not None:
        print(f"Report: {report.report_path}")
    return 1 if report.problems else 0


if __name__ == "__main__":
    sys.exit(main())
