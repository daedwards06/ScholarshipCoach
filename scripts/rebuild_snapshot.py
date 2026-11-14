"""Rebuild the snapshot from the curated catalog alone.

No scrapers and no network: the curated records are re-read and every other
enabled source's rows are carried forward from the prior snapshot.  The app
calls :func:`rebuild_curated_snapshot` right after a confirm; the server's
03:50 timer runs this script with ``--if-stale``, so a record changed outside
the app (a merged pull, the verify pass's stamp) is picked up the same night.

    python scripts/rebuild_snapshot.py
    python scripts/rebuild_snapshot.py --if-stale

Exit 0 when the snapshot was written or there was nothing to do, 1 when a
guardrail blocked the write or no snapshot came out of the run.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

from scripts import run_ingest
from src.ingest.sources import curated_catalog
from src.io.snapshotting import get_latest_snapshot_path
from src.normalize.catalog_schema import iter_catalog_files


def rebuild_curated_snapshot() -> dict[str, Any]:
    """Run the curated source only and return the ingest report.

    ``report["artifact_paths"]["snapshot"]`` is ``None`` when nothing was
    written; the prior snapshot is then untouched.
    """
    return run_ingest.run_ingest(
        date=None, only_sources=[curated_catalog.CuratedCatalogSource.name]
    )


def snapshot_is_stale(
    records_dir: Path | None = None, processed_dir: Path | None = None
) -> bool:
    """True when there is no snapshot, or the records changed after the newest one."""
    records = records_dir or curated_catalog.RECORDS_DIR
    latest = get_latest_snapshot_path(
        processed_dir or run_ingest.ROOT_DIR / "data" / "processed"
    )
    if latest is None:
        return True
    built = latest.stat().st_mtime_ns
    # The folder's own time moves when a record is removed, which no file shows.
    candidates = [*iter_catalog_files(records), *([records] if records.is_dir() else [])]
    return any(path.stat().st_mtime_ns > built for path in candidates)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the snapshot from the curated catalog alone."
    )
    parser.add_argument(
        "--if-stale",
        action="store_true",
        help="Rebuild only when a catalog record is newer than the latest snapshot.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.if_stale and not snapshot_is_stale():
        print("Snapshot is newer than every catalog record; nothing to do.")
        return 0

    report = rebuild_curated_snapshot()
    snapshot = report["artifact_paths"]["snapshot"]
    if snapshot is None:
        print(f"Snapshot NOT written: {report['artifact_notes']['snapshot_skip_reason']}")
        print(f"Wrote ingest report: {report['artifact_paths']['report']}")
        return 1
    print(f"Wrote snapshot: {snapshot}")
    print(f"Wrote changes: {report['artifact_paths']['delta']}")
    print(
        "Delta counts: "
        f"added={report['delta_counts']['added']}, "
        f"removed={report['delta_counts']['removed']}, "
        f"changed={report['delta_counts']['changed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
