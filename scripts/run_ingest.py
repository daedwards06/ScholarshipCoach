from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ingest.cache import write_raw_payload
from src.ingest.http import PoliteHttpClient
from src.ingest.registry import register_sources
from src.io.snapshotting import (
    REQUIRED_COLUMNS,
    build_and_write_snapshot,
    find_prior_snapshot,
    write_json_atomic,
)

logger = logging.getLogger("run_ingest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scholarship ingestion and snapshot generation.")
    parser.add_argument("--raw-dir", type=Path, default=ROOT_DIR / "data" / "raw")
    parser.add_argument("--processed-dir", type=Path, default=ROOT_DIR / "data" / "processed")
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


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def _format_utc_iso_z(timestamp: pd.Timestamp) -> str:
    utc_ts = timestamp.tz_convert("UTC")
    return utc_ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _coerce_run_date(run_date: str | None) -> date:
    if run_date is None:
        return datetime.now(tz=UTC).date()
    return datetime.strptime(run_date, "%Y%m%d").date()


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
        df[ts_column] = df[ts_column].apply(_format_utc_iso_z)

    return df[REQUIRED_COLUMNS]


def _missing_text(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def _build_guardrail_warnings(
    *,
    prior_count: int | None,
    current_count: int,
    missing_title_or_source_count: int,
) -> list[str]:
    warnings: list[str] = []
    if prior_count and prior_count > 0 and current_count < (prior_count * 0.5):
        warnings.append(
            f"Record count dropped by more than 50% vs prior snapshot "
            f"({current_count} vs {prior_count})."
        )
    if current_count > 0:
        missing_ratio = missing_title_or_source_count / current_count
        if missing_ratio > 0.05:
            warnings.append(
                f"More than 5% of records are missing title or source_url "
                f"({missing_title_or_source_count}/{current_count}, {missing_ratio:.1%})."
            )
    return warnings


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    started_at = datetime.now(tz=UTC)
    raw_dir = _resolve_repo_path(args.raw_dir)
    processed_dir = _resolve_repo_path(args.processed_dir)
    report_dir = ROOT_DIR / "reports" / "ingest_runs"
    run_date = _coerce_run_date(args.date)

    source_records: list[dict[str, Any]] = []
    source_attempts: list[dict[str, Any]] = []
    sources = register_sources()
    client = PoliteHttpClient(requests_per_second=args.requests_per_second)

    try:
        for source in sources:
            source_report: dict[str, Any] = {"source": source.name, "status": "failed"}
            try:
                raw_response = source.fetch(client)
                raw_path = write_raw_payload(
                    source_name=source.name,
                    payload=raw_response.content,
                    extension=raw_response.extension,
                    raw_root=raw_dir,
                    timestamp=raw_response.fetched_at,
                )
                parsed = source.parse(raw_response.content, fetched_at=raw_response.fetched_at)
                source_records.extend(parsed)
                source_report["status"] = "succeeded"
                source_report["records"] = len(parsed)
                source_report["cache_path"] = str(raw_path.resolve())
                logger.info("Source=%s cached=%s records=%d", source.name, raw_path, len(parsed))
            except Exception:
                source_report["error"] = "fetch_or_parse_failed"
                logger.exception("Source %s failed. Continuing with remaining sources.", source.name)
            source_attempts.append(source_report)
    finally:
        client.close()

    normalized_df = _normalize_records(source_records)
    prior_count: int | None = None
    prior_snapshot_path = find_prior_snapshot(processed_dir, run_date)
    if prior_snapshot_path is not None:
        try:
            prior_count = len(pd.read_parquet(prior_snapshot_path))
        except Exception:
            logger.exception("Failed to read prior snapshot at %s", prior_snapshot_path)

    missing_title_mask = _missing_text(normalized_df["title"])
    missing_source_url_mask = _missing_text(normalized_df["source_url"])
    missing_title_or_source_count = int((missing_title_mask | missing_source_url_mask).sum())
    guardrail_warnings = _build_guardrail_warnings(
        prior_count=prior_count,
        current_count=len(normalized_df),
        missing_title_or_source_count=missing_title_or_source_count,
    )
    for warning in guardrail_warnings:
        logger.warning("Guardrail: %s", warning)

    snapshot_path, changes_path, delta = build_and_write_snapshot(
        normalized_df,
        processed_dir=processed_dir,
        run_date=run_date,
    )
    finished_at = datetime.now(tz=UTC)

    attempted_sources = [entry["source"] for entry in source_attempts]
    succeeded_sources = [entry["source"] for entry in source_attempts if entry["status"] == "succeeded"]
    failed_sources = [entry["source"] for entry in source_attempts if entry["status"] != "succeeded"]
    cache_paths = [entry["cache_path"] for entry in source_attempts if "cache_path" in entry]

    report_stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"ingest_{report_stamp}.json"
    report_payload = {
        "run_started_at": started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_finished_at": finished_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "run_date": run_date.isoformat(),
        "sources": {
            "attempted": attempted_sources,
            "succeeded": succeeded_sources,
            "failed": failed_sources,
            "attempted_count": len(attempted_sources),
            "succeeded_count": len(succeeded_sources),
            "failed_count": len(failed_sources),
            "details": source_attempts,
        },
        "records": {
            "parsed_total": len(source_records),
            "snapshot_total": len(normalized_df),
            "prior_snapshot_total": prior_count,
            "missing_title_or_source_url_count": missing_title_or_source_count,
            "missing_title_or_source_url_pct": (
                round((missing_title_or_source_count / len(normalized_df)) * 100, 3)
                if len(normalized_df) > 0
                else 0.0
            ),
        },
        "cache_paths": cache_paths,
        "artifact_paths": {
            "snapshot": str(snapshot_path.resolve()),
            "delta": str(changes_path.resolve()),
            "report": str(report_path.resolve()),
            "prior_snapshot": str(prior_snapshot_path.resolve()) if prior_snapshot_path else None,
        },
        "guardrail_warnings": guardrail_warnings,
        "delta_counts": {
            "added": len(delta["added"]),
            "removed": len(delta["removed"]),
            "changed": len(delta["changed"]),
        },
    }
    write_json_atomic(report_payload, report_path)

    print(f"Wrote snapshot: {snapshot_path}")
    print(f"Wrote changes: {changes_path}")
    print(f"Wrote ingest report: {report_path}")
    print(
        "Delta counts: "
        f"added={len(delta['added'])}, "
        f"removed={len(delta['removed'])}, "
        f"changed={len(delta['changed'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
