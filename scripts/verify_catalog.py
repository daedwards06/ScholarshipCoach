"""Re-verify curated catalog records against their source pages.

Run monthly (see ``docs/operations.md``).  Each due record's ``source_url`` is
re-fetched, hashed, and re-extracted; a page that still agrees only gets its
``provenance.verified_on`` stamped, and anything else lands in the confirm
queue as a ``reverify`` proposal for a person to work.

    python scripts/verify_catalog.py
    python scripts/verify_catalog.py --since-days 30 --max-records 5

The exit status reports whether the pass ran, not what it found: dead links
and changed pages are the output of a healthy run, and they are reported as
proposals.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from src.catalog.inbox import INBOX_DIR
from src.catalog.verify import (
    DEFAULT_REQUESTS_PER_SECOND,
    DEFAULT_SINCE_DAYS,
    REPORTS_DIR,
    RecordVerification,
    verify_catalog,
)
from src.normalize.catalog_schema import RECORDS_DIR


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-verify curated catalog records against their source pages."
    )
    parser.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    parser.add_argument("--inbox-dir", type=Path, default=INBOX_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument(
        "--since-days",
        type=int,
        default=DEFAULT_SINCE_DAYS,
        help="Skip records verified within this many days (default: %(default)s).",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Stop after this many pages; useful for a quick smoke run.",
    )
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=DEFAULT_REQUESTS_PER_SECOND,
        help="Polite rate limit for source fetches (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _summarize(result: RecordVerification) -> str:
    detail = ""
    if result.changes:
        detail = f"changed: {', '.join(sorted(result.changes))}"
    elif result.message:
        detail = result.message
    proposal = f" -> {result.proposal_id}" if result.proposal_id else ""
    return f"  {result.catalog_id:<44} {result.outcome:<10} {detail}{proposal}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = verify_catalog(
        records_dir=args.records_dir,
        inbox_dir=args.inbox_dir,
        reports_dir=args.reports_dir,
        since_days=args.since_days,
        max_records=args.max_records,
        requests_per_second=args.requests_per_second,
    )

    if report.records_total == 0:
        print(f"ERROR: no catalog records found in {args.records_dir}")
        return 1

    print(
        f"{report.records_total} record(s) in {args.records_dir}; "
        f"{report.records_due} due after {args.since_days} day(s); "
        f"{len(report.results)} checked."
    )
    for result in report.results:
        print(_summarize(result))

    counts = report.counts
    print(
        "unchanged={unchanged} changed={changed} dead_link={dead_link} error={error}".format(
            **counts
        )
    )
    if report.proposal_ids:
        print(
            f"{len(report.proposal_ids)} proposal(s) queued in {args.inbox_dir}; "
            "review them with: python scripts/catalog_inbox.py list"
        )
    if report.report_path is not None:
        print(f"Report: {report.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
