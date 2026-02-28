from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

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
    get_latest_snapshot_path as _get_latest_snapshot_path,
    load_latest_snapshot_df as _load_latest_snapshot_df,
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
    parser.add_argument("--max-listing-pages", type=int, default=3)
    parser.add_argument("--max-detail-pages", type=int, default=200)
    parser.add_argument("--request-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--max-runtime-seconds", type=int, default=600)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
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


def _normalize_url_for_dedupe(value: Any) -> str:
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""

    parsed = urlparse(raw)
    scheme = parsed.scheme.lower() if parsed.scheme else "https"
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query_items = sorted((key, val) for key, val in parse_qsl(parsed.query, keep_blank_values=False))
    query = urlencode(query_items)
    return urlunparse((scheme, netloc, path, "", query, ""))


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


def _dedupe_records(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    dedupe_df = df.copy()
    dedupe_df["_source_url_normalized"] = dedupe_df["source_url"].apply(_normalize_url_for_dedupe)
    dedupe_df = dedupe_df.sort_values(
        by=["scholarship_id", "_source_url_normalized", "title"], kind="mergesort"
    )
    dedupe_df = dedupe_df.drop_duplicates(
        subset=["scholarship_id", "_source_url_normalized"], keep="first"
    )
    dedupe_df = dedupe_df.drop(columns=["_source_url_normalized"])
    dedupe_df = dedupe_df.drop_duplicates(subset=["scholarship_id"], keep="first")
    return dedupe_df.reset_index(drop=True)


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


def _exception_summary(exc: Exception) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def run_ingest(
    *,
    date: date | None = None,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    requests_per_second: float = 1.0,
    max_listing_pages: int = 3,
    max_detail_pages: int = 200,
    request_timeout_seconds: float = 20.0,
    max_runtime_seconds: int = 600,
    concurrency: int = 4,
    resume: bool = False,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(tz=UTC)
    resolved_raw_dir = _resolve_repo_path(raw_dir or (ROOT_DIR / "data" / "raw"))
    resolved_processed_dir = _resolve_repo_path(processed_dir or (ROOT_DIR / "data" / "processed"))
    resolved_report_dir = _resolve_repo_path(report_dir or (ROOT_DIR / "reports" / "ingest_runs"))
    effective_run_date = date or datetime.now(tz=UTC).date()
    report_stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    report_path = resolved_report_dir / f"ingest_{report_stamp}.json"

    source_records: list[dict[str, Any]] = []
    source_attempts: list[dict[str, Any]] = []
    normalized_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    guardrail_warnings: list[str] = []
    prior_count: int | None = None
    prior_snapshot_path: Path | None = None
    snapshot_path: Path | None = None
    changes_path: Path | None = None
    delta: dict[str, Any] = {"added": [], "removed": [], "changed": []}
    missing_title_or_source_count = 0
    snapshot_skip_reason: str | None = None
    run_exception: dict[str, str] | None = None

    try:
        sources = register_sources()
        client = PoliteHttpClient(
            requests_per_second=requests_per_second,
            timeout_seconds=request_timeout_seconds,
        )
        try:
            for source in sources:
                source_report: dict[str, Any] = {
                    "source": source.name,
                    "status": "failed",
                    "records": 0,
                    "cache_paths": [],
                }
                try:
                    source_fetch_records = getattr(source, "fetch_records", None)
                    if callable(source_fetch_records):
                        fetch_result = source_fetch_records(
                            client,
                            raw_root=resolved_raw_dir,
                            max_listing_pages=max_listing_pages,
                            max_detail_pages=max_detail_pages,
                            max_runtime_seconds=max_runtime_seconds,
                            concurrency=concurrency,
                            resume=resume,
                        )
                        if len(fetch_result) == 3:
                            parsed, cache_paths, source_meta = fetch_result
                        else:
                            parsed, cache_paths = fetch_result
                            source_meta = {}
                        source_records.extend(parsed)
                        source_report["records"] = len(parsed)
                        source_report["cache_paths"] = [str(path.resolve()) for path in cache_paths]
                        source_report.update(source_meta)
                        source_report["status"] = (
                            "partial"
                            if source_meta.get("caps_hit") or source_meta.get("parse_failures")
                            else "succeeded"
                        )
                        logger.info(
                            "Source=%s cached=%d files records=%d",
                            source.name,
                            len(cache_paths),
                            len(parsed),
                        )
                    else:
                        raw_response = source.fetch(client)
                        raw_path = write_raw_payload(
                            source_name=source.name,
                            payload=raw_response.content,
                            extension=raw_response.extension,
                            raw_root=resolved_raw_dir,
                            timestamp=raw_response.fetched_at,
                        )
                        parsed = source.parse(raw_response.content, fetched_at=raw_response.fetched_at)
                        source_records.extend(parsed)
                        source_report["status"] = "succeeded"
                        source_report["records"] = len(parsed)
                        source_report["cache_paths"] = [str(raw_path.resolve())]
                        source_report["cached_files_written"] = 1
                        logger.info("Source=%s cached=%s records=%d", source.name, raw_path, len(parsed))
                except Exception as exc:
                    source_report["error"] = "fetch_or_parse_failed"
                    source_report["exception_summary"] = _exception_summary(exc)
                    logger.exception("Source %s failed. Continuing with remaining sources.", source.name)
                source_attempts.append(source_report)
        finally:
            client.close()

        normalized_df = _normalize_records(source_records)
        normalized_df = _dedupe_records(normalized_df)

        prior_snapshot_path = find_prior_snapshot(resolved_processed_dir, effective_run_date)
        if prior_snapshot_path is not None:
            try:
                prior_count = len(pd.read_parquet(prior_snapshot_path))
            except Exception:
                logger.exception("Failed to read prior snapshot at %s", prior_snapshot_path)

        if not normalized_df.empty:
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
                processed_dir=resolved_processed_dir,
                run_date=effective_run_date,
            )
        else:
            snapshot_skip_reason = "No parsed records available; snapshot and delta were skipped."
            logger.warning(snapshot_skip_reason)
    except Exception as exc:
        run_exception = _exception_summary(exc)
        logger.exception("Ingest run failed after partial progress.")
        if normalized_df.empty:
            snapshot_skip_reason = snapshot_skip_reason or "Ingest failed before any records were normalized."
        else:
            snapshot_skip_reason = snapshot_skip_reason or "Snapshot generation failed after records were normalized."
    finally:
        finished_at = datetime.now(tz=UTC)
        attempted_sources = [entry["source"] for entry in source_attempts]
        succeeded_sources = [entry["source"] for entry in source_attempts if entry["status"] == "succeeded"]
        partial_sources = [entry["source"] for entry in source_attempts if entry["status"] == "partial"]
        failed_sources = [entry["source"] for entry in source_attempts if entry["status"] == "failed"]
        cache_paths = [path for entry in source_attempts for path in entry.get("cache_paths", [])]
        detail_attempted = sum(int(entry.get("detail_urls_attempted", 0)) for entry in source_attempts)
        detail_succeeded = sum(int(entry.get("detail_urls_succeeded", 0)) for entry in source_attempts)
        detail_failed = sum(int(entry.get("detail_urls_failed", 0)) for entry in source_attempts)
        listing_processed = sum(int(entry.get("listing_urls_processed", 0)) for entry in source_attempts)
        cached_files_written = sum(int(entry.get("cached_files_written", 0)) for entry in source_attempts)

        if run_exception is not None:
            status = "partial" if not normalized_df.empty or bool(succeeded_sources or partial_sources) else "failed"
        elif failed_sources or partial_sources:
            status = "partial" if not normalized_df.empty or bool(succeeded_sources) else "failed"
        else:
            status = "success"

        report_payload = {
            "status": status,
            "run_started_at": started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run_finished_at": finished_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
            "run_date": effective_run_date.isoformat(),
            "config": {
                "requests_per_second": requests_per_second,
                "max_listing_pages": max_listing_pages,
                "max_detail_pages": max_detail_pages,
                "request_timeout_seconds": request_timeout_seconds,
                "max_runtime_seconds": max_runtime_seconds,
                "concurrency": concurrency,
                "resume": resume,
            },
            "sources": {
                "attempted": attempted_sources,
                "succeeded": succeeded_sources,
                "partial": partial_sources,
                "failed": failed_sources,
                "attempted_count": len(attempted_sources),
                "succeeded_count": len(succeeded_sources),
                "partial_count": len(partial_sources),
                "failed_count": len(failed_sources),
                "details": source_attempts,
            },
            "progress": {
                "listing_urls_processed": listing_processed,
                "detail_urls_attempted": detail_attempted,
                "detail_urls_succeeded": detail_succeeded,
                "detail_urls_failed": detail_failed,
                "cached_files_written": cached_files_written,
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
                "snapshot": str(snapshot_path.resolve()) if snapshot_path else None,
                "delta": str(changes_path.resolve()) if changes_path else None,
                "report": str(report_path.resolve()),
                "prior_snapshot": str(prior_snapshot_path.resolve()) if prior_snapshot_path else None,
            },
            "artifact_notes": {"snapshot_skip_reason": snapshot_skip_reason},
            "guardrail_warnings": guardrail_warnings,
            "delta_counts": {
                "added": len(delta["added"]),
                "removed": len(delta["removed"]),
                "changed": len(delta["changed"]),
            },
            "exception_summary": run_exception,
        }
        write_json_atomic(report_payload, report_path)
    return report_payload


def get_latest_snapshot_path() -> Path:
    processed_dir = ROOT_DIR / "data" / "processed"
    latest = _get_latest_snapshot_path(processed_dir)
    if latest is None:
        raise FileNotFoundError(f"No snapshot parquet found in '{processed_dir}'.")
    return latest


def load_latest_snapshot_df() -> pd.DataFrame:
    processed_dir = ROOT_DIR / "data" / "processed"
    return _load_latest_snapshot_df(processed_dir)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = run_ingest(
        date=_coerce_run_date(args.date),
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        requests_per_second=args.requests_per_second,
        max_listing_pages=args.max_listing_pages,
        max_detail_pages=args.max_detail_pages,
        request_timeout_seconds=args.request_timeout_seconds,
        max_runtime_seconds=args.max_runtime_seconds,
        concurrency=args.concurrency,
        resume=args.resume,
    )

    print(f"Run status: {report['status']}")
    print(f"Wrote snapshot: {report['artifact_paths']['snapshot']}")
    print(f"Wrote changes: {report['artifact_paths']['delta']}")
    print(f"Wrote ingest report: {report['artifact_paths']['report']}")
    print(
        "Delta counts: "
        f"added={report['delta_counts']['added']}, "
        f"removed={report['delta_counts']['removed']}, "
        f"changed={report['delta_counts']['changed']}"
    )
    return 0 if report["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
