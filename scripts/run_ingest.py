from __future__ import annotations

import argparse
import logging
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pandas as pd

from src.ingest.cache import write_raw_payload
from src.ingest.http import PoliteHttpClient
from src.ingest.registry import register_sources
from src.io.snapshotting import (
    LLM_PROVENANCE_COLUMN,
    REQUIRED_COLUMNS,
    build_and_write_snapshot,
    find_prior_snapshot,
    get_latest_snapshot_path as _get_latest_snapshot_path,
    load_latest_snapshot_df as _load_latest_snapshot_df,
    write_json_atomic,
)
from src.llm.cache import (
    compute_extraction_key,
    extraction_path,
    get_or_extract,
    load_extraction,
)
from src.llm.client import DEFAULT_MODEL, MODEL_ENV, LlmClient, client_from_env
from src.llm.extraction import EXTRACTION_FIELDS, EXTRACTION_PROMPT_VERSION

ROOT_DIR = Path(__file__).resolve().parents[1]

logger = logging.getLogger("run_ingest")

LLM_SOURCE_TEXT_FIELDS = ("title", "description", "eligibility_text")
LLM_LIST_FIELDS = ("states_allowed", "majors_allowed", "keywords")
LLM_NUMERIC_FIELDS = ("amount_min", "amount_max", "min_gpa")


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
    parser.add_argument(
        "--llm-enrich",
        action="store_true",
        help="Fill empty structured fields from listing text using the LLM extraction layer.",
    )
    parser.add_argument(
        "--llm-max-calls",
        type=int,
        default=100,
        help="Maximum live LLM API calls per run (cache hits are free and do not count).",
    )
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


def _resolve_llm_model_name(client: LlmClient | None) -> str:
    if client is not None:
        return client.model
    return os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL


def _empty_llm_summary(*, requested: bool = False, model_name: str | None = None) -> dict[str, Any]:
    return {
        "requested": requested,
        "enabled": False,
        "model": model_name,
        "prompt_version": EXTRACTION_PROMPT_VERSION,
        "api_call_limit": 0,
        "records_scanned": 0,
        "records_eligible": 0,
        "cache_hits": 0,
        "api_calls": 0,
        "max_calls_reached": False,
        "records_enriched": 0,
        "fields_filled": 0,
        "fields_filled_by_field": {},
    }


def _is_empty_field(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, bool):
        return False
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    if hasattr(value, "tolist") and hasattr(value, "__len__"):
        return len(value) == 0
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _coerce_llm_value(field_name: str, value: Any) -> Any:
    """Convert a validated extraction value into the column's in-frame type."""
    if value is None:
        return None
    if field_name == "deadline":
        try:
            return date.fromisoformat(str(value))
        except ValueError:
            return None
    if field_name in ("amount_min", "amount_max", "min_gpa"):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if field_name in LLM_LIST_FIELDS:
        return _coerce_list(value)
    if field_name == "essay_required":
        return value if isinstance(value, bool) else None
    text = str(value).strip()
    return text or None


def _enrich_records_with_llm(
    df: pd.DataFrame,
    *,
    client: LlmClient | None,
    model_name: str,
    processed_dir: Path,
    max_calls: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill empty structured fields from listing text, cache-first and fill-only.

    A record is a candidate only when it has at least one empty extraction
    target field and some description/eligibility text to extract from.  Values
    the deterministic parsers produced are never overwritten; every field the
    LLM does fill is recorded in the :data:`LLM_PROVENANCE_COLUMN` column.

    Cached extractions resolve without a client, so a keyless run still applies
    prior work.  ``max_calls`` caps live API calls only.

    Returns:
        Tuple of (enriched DataFrame, summary counts for the ingest report).
    """
    summary = _empty_llm_summary(requested=True, model_name=model_name)
    summary["enabled"] = client is not None
    summary["api_call_limit"] = max_calls

    enriched = df.copy()
    if enriched.empty:
        enriched[LLM_PROVENANCE_COLUMN] = pd.Series(dtype="object")
        return enriched, summary

    # Non-numeric target columns must hold Python objects (dates, lists, bools)
    # before a filled value can be written into them.
    for column in EXTRACTION_FIELDS:
        if column in enriched.columns and column not in LLM_NUMERIC_FIELDS:
            as_object = enriched[column].astype(object)
            enriched[column] = as_object.where(as_object.notna(), None)

    provenance: list[list[str]] = [[] for _ in range(len(enriched))]
    target_fields = [name for name in EXTRACTION_FIELDS if name in enriched.columns]
    by_field: dict[str, int] = {}
    api_calls = 0

    for position, (index, row) in enumerate(list(enriched.iterrows())):
        summary["records_scanned"] += 1
        empty_fields = [name for name in target_fields if _is_empty_field(row.get(name))]
        if not empty_fields:
            continue

        record = {name: row.get(name) for name in LLM_SOURCE_TEXT_FIELDS}
        has_source_text = any(
            str(record.get(name) or "").strip()
            for name in ("description", "eligibility_text")
        )
        if not has_source_text:
            continue
        summary["records_eligible"] += 1

        key = compute_extraction_key(record, model_name=model_name)
        cached = load_extraction(extraction_path(key, model_name, processed_dir=processed_dir))
        if cached is not None:
            summary["cache_hits"] += 1
            cached_fields = cached.get("fields")
            fields = dict(cached_fields) if isinstance(cached_fields, dict) else {}
        elif client is None:
            continue
        elif api_calls >= max_calls:
            summary["max_calls_reached"] = True
            continue
        else:
            api_calls += 1
            fields = get_or_extract(
                client,
                record,
                processed_dir=processed_dir,
                model_name=model_name,
            )
        if not fields:
            continue

        filled: list[str] = []
        for name in empty_fields:
            value = _coerce_llm_value(name, fields.get(name))
            if value is None:
                continue
            enriched.at[index, name] = value
            filled.append(name)
            by_field[name] = by_field.get(name, 0) + 1

        if filled:
            provenance[position] = filled
            summary["records_enriched"] += 1
            summary["fields_filled"] += len(filled)

    enriched[LLM_PROVENANCE_COLUMN] = provenance
    summary["api_calls"] = api_calls
    summary["fields_filled_by_field"] = dict(sorted(by_field.items()))
    return enriched, summary


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
    llm_enrich: bool = False,
    llm_max_calls: int = 100,
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
    llm_summary: dict[str, Any] = _empty_llm_summary(requested=llm_enrich)

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

        if llm_enrich and not normalized_df.empty:
            llm_client = client_from_env()
            llm_model_name = _resolve_llm_model_name(llm_client)
            if llm_client is None:
                logger.info(
                    "LLM enrichment disabled (no key); applying cached extractions only."
                )
            normalized_df, llm_summary = _enrich_records_with_llm(
                normalized_df,
                client=llm_client,
                model_name=llm_model_name,
                processed_dir=resolved_processed_dir,
                max_calls=llm_max_calls,
            )
            logger.info(
                "LLM enrichment: scanned=%d eligible=%d cache_hits=%d api_calls=%d "
                "records_enriched=%d fields_filled=%d",
                llm_summary["records_scanned"],
                llm_summary["records_eligible"],
                llm_summary["cache_hits"],
                llm_summary["api_calls"],
                llm_summary["records_enriched"],
                llm_summary["fields_filled"],
            )
            if llm_summary["max_calls_reached"]:
                logger.warning(
                    "LLM enrichment hit the --llm-max-calls cap of %d; "
                    "remaining records were left unenriched.",
                    llm_max_calls,
                )

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
                "llm_enrich": llm_enrich,
                "llm_max_calls": llm_max_calls,
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
            "llm_enrichment": llm_summary,
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
        llm_enrich=args.llm_enrich,
        llm_max_calls=args.llm_max_calls,
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
    if args.llm_enrich:
        llm_report = report["llm_enrichment"]
        print(
            "LLM enrichment: "
            f"enabled={llm_report['enabled']}, "
            f"cache_hits={llm_report['cache_hits']}, "
            f"api_calls={llm_report['api_calls']}, "
            f"fields_filled={llm_report['fields_filled']}"
        )
    return 0 if report["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
