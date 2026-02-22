from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ingest.cache import write_raw_payload
from src.ingest.http import PoliteHttpClient
from src.ingest.registry import register_sources
from src.io.snapshotting import REQUIRED_COLUMNS, build_and_write_snapshot

logger = logging.getLogger("run_ingest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scholarship ingestion and snapshot generation.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Run date in YYYYMMDD format. Defaults to current UTC date.",
    )
    parser.add_argument("--requests-per-second", type=float, default=1.0)
    return parser.parse_args()


def _coerce_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned or None
    return None


def _normalize_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = pd.DataFrame(records)

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = None

    for numeric_column in ("amount_min", "amount_max", "min_gpa"):
        df[numeric_column] = pd.to_numeric(df[numeric_column], errors="coerce")
        df[numeric_column] = df[numeric_column].where(pd.notna(df[numeric_column]), None)

    for list_column in ("states_allowed", "majors_allowed", "keywords"):
        df[list_column] = df[list_column].apply(_coerce_list)

    for bool_column in ("is_recurring", "essay_required"):
        df[bool_column] = df[bool_column].apply(
            lambda value: value if isinstance(value, bool) or value is None else None
        )

    df["deadline"] = pd.to_datetime(df["deadline"], errors="coerce").dt.date

    now_utc = datetime.now(tz=UTC)
    for ts_column in ("first_seen_at", "last_seen_at"):
        df[ts_column] = pd.to_datetime(df[ts_column], utc=True, errors="coerce")
        df[ts_column] = df[ts_column].fillna(pd.Timestamp(now_utc))

    return df[REQUIRED_COLUMNS]


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    source_records: list[dict[str, Any]] = []
    sources = register_sources()
    client = PoliteHttpClient(requests_per_second=args.requests_per_second)

    try:
        for source in sources:
            try:
                raw_response = source.fetch(client)
                raw_path = write_raw_payload(
                    source_name=source.name,
                    payload=raw_response.content,
                    extension=raw_response.extension,
                    raw_root=args.raw_dir,
                    timestamp=raw_response.fetched_at,
                )
                parsed = source.parse(raw_response.content, fetched_at=raw_response.fetched_at)
                source_records.extend(parsed)
                logger.info("Source=%s cached=%s records=%d", source.name, raw_path, len(parsed))
            except Exception:
                logger.exception("Source %s failed. Continuing with remaining sources.", source.name)
    finally:
        client.close()

    normalized_df = _normalize_records(source_records)
    snapshot_path, changes_path, delta = build_and_write_snapshot(
        normalized_df,
        processed_dir=args.processed_dir,
        run_date=args.date,
    )

    print(f"Wrote snapshot: {snapshot_path}")
    print(f"Wrote changes: {changes_path}")
    print(
        "Delta counts: "
        f"added={len(delta['added'])}, "
        f"removed={len(delta['removed'])}, "
        f"changed={len(delta['changed'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
