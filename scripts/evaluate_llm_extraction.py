"""Measure LLM extraction quality against the deterministic parsers.

The regex/HTML parsers already populated structured fields for the catalog, so
the latest snapshot ships free gold labels.  This script samples records whose
parsers produced at least one target field, re-extracts those fields from the
record's raw text with the LLM (cache-aware), and scores per-field agreement.

Gold is parser output, not human judgement.  Two consequences are reported
honestly rather than hidden:

* A gold value can itself be wrong, so "accuracy" here means *agreement with
  the parser*, an upper bound on nothing and a lower bound on nothing.
* Where gold is empty the comparison is undecidable — the parser may have
  missed a stated value, or the value may genuinely not be stated.  Those cases
  are reported separately as an **unverified** value-added rate, never folded
  into an accuracy number.

Fields the Task 4 enrichment pass filled (per the ``llm_enriched_fields``
provenance column) are excluded from gold so the LLM is never scored against
its own earlier output.

Requires ``SCHOLARSHIPCOACH_LLM_API_KEY`` for a live run; without a key the run
still proceeds against cached extractions only and says so in the report.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.io.snapshotting import (
    LLM_PROVENANCE_COLUMN,
    coerce_provenance_list,
    get_latest_snapshot_path,
)
from src.llm.cache import (
    compute_extraction_key,
    extraction_path,
    get_or_extract,
    load_extraction,
)
from src.llm.client import DEFAULT_MODEL, MODEL_ENV, LlmClient, client_from_env
from src.llm.extraction import EXTRACTION_FIELDS, EXTRACTION_PROMPT_VERSION
from src.text_utils import coerce_text, normalize_text

ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_SAMPLE_SIZE = 60
SOURCE_TEXT_FIELDS = ("title", "description", "eligibility_text")
LIST_FIELDS: tuple[str, ...] = ("states_allowed", "majors_allowed", "keywords")
NUMERIC_FIELDS: tuple[str, ...] = ("amount_min", "amount_max", "min_gpa")
SCALAR_FIELDS: tuple[str, ...] = tuple(name for name in EXTRACTION_FIELDS if name not in LIST_FIELDS)
AMOUNT_TOLERANCE = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score LLM structured extraction against deterministic parser output."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of snapshot records to score. Defaults to {DEFAULT_SAMPLE_SIZE}.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Snapshot parquet path. If omitted, the latest in --processed-dir is used.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT_DIR / "data" / "processed",
        help="Processed directory used to resolve the latest snapshot and the extraction cache.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT_DIR / "reports",
        help="Output directory for the markdown report and its JSON twin.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override the cache namespace model name. Defaults to the configured LLM model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sampling seed, so a re-run scores the same records. Defaults to 0.",
    )
    parser.add_argument(
        "--max-api-calls",
        type=int,
        default=None,
        help="Cap live API calls (cache hits are free). Defaults to the sample size.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Never call the API; score whatever the extraction cache already holds.",
    )
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def _resolve_snapshot_path(snapshot: Path | None, processed_dir: Path) -> Path:
    if snapshot is not None:
        return _resolve_path(snapshot)
    latest = get_latest_snapshot_path(_resolve_path(processed_dir))
    if latest is None:
        raise FileNotFoundError(f"No snapshot parquet found in '{processed_dir}'.")
    return latest


def _resolve_model_name(client: LlmClient | None, override: str | None) -> str:
    if override:
        return override
    if client is not None:
        return client.model
    return os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL


def is_empty(value: Any) -> bool:
    """Return True when a snapshot cell carries no usable value."""
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


def _as_items(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [value]
    if hasattr(value, "tolist") and not isinstance(value, (bytes, bytearray)):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def canonical_value(field: str, value: Any) -> Any:
    """Reduce a gold or predicted value to a comparable canonical form.

    Both sides go through this function, so comparison is insensitive to case,
    whitespace, and the container type parquet happens to return.  Returns
    ``None`` for anything that carries no value.
    """
    if is_empty(value):
        return None

    if field in LIST_FIELDS:
        items: set[str] = set()
        for item in _as_items(value):
            if is_empty(item):
                continue
            text = normalize_text(item)
            if not text:
                continue
            items.add(text.upper() if field == "states_allowed" else text)
        return tuple(sorted(items)) or None

    if field == "deadline":
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        try:
            return pd.Timestamp(str(value)).date().isoformat()
        except (ValueError, TypeError):
            return None

    if field in NUMERIC_FIELDS:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    if field == "essay_required":
        return bool(value) if isinstance(value, bool) else None

    return normalize_text(value) or None


def scalar_matches(field: str, gold: Any, predicted: Any) -> bool:
    """Compare two canonical scalars, allowing a $1 tolerance on amounts."""
    if gold is None or predicted is None:
        return False
    if field in ("amount_min", "amount_max"):
        return abs(float(gold) - float(predicted)) <= AMOUNT_TOLERANCE
    return bool(gold == predicted)


def gold_fields_for_row(row: Any) -> dict[str, Any]:
    """Return the canonical parser-derived values for one snapshot record.

    Fields the LLM itself filled during ingest (``llm_enriched_fields``) are
    excluded, so the LLM is never scored against its own output.  The
    provenance column is optional: snapshots written before Task 4 do not have
    it, and its absence simply means nothing was LLM-filled.
    """
    enriched = set(coerce_provenance_list(_row_get(row, LLM_PROVENANCE_COLUMN)))
    gold: dict[str, Any] = {}
    for field in EXTRACTION_FIELDS:
        if field in enriched:
            continue
        canonical = canonical_value(field, _row_get(row, field))
        if canonical is not None:
            gold[field] = canonical
    return gold


def _row_get(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def has_source_text(row: Any) -> bool:
    """Return True when a record has text for the LLM to extract from."""
    return any(
        coerce_text(_row_get(row, field))
        for field in ("description", "eligibility_text")
    )


def select_sample(
    snapshot_df: pd.DataFrame,
    *,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    """Sample records that carry both gold labels and text to extract from."""
    if snapshot_df.empty:
        return snapshot_df

    keep = [
        index
        for index, row in snapshot_df.iterrows()
        if has_source_text(row) and gold_fields_for_row(row)
    ]
    eligible = snapshot_df.loc[keep]
    if eligible.empty or sample_size <= 0:
        return eligible.head(0)
    if len(eligible) > sample_size:
        eligible = eligible.sample(n=sample_size, random_state=seed)
    return eligible.sort_values(by=["scholarship_id"], kind="mergesort").reset_index(drop=True)


def compare_record(gold: dict[str, Any], predicted: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Pair gold and predicted canonical values for every extraction field."""
    comparison: dict[str, dict[str, Any]] = {}
    for field in EXTRACTION_FIELDS:
        comparison[field] = {
            "gold": gold.get(field),
            "predicted": canonical_value(field, predicted.get(field)),
        }
    return comparison


def _scalar_field_metrics(pairs: list[dict[str, Any]], field: str) -> dict[str, Any]:
    gold_present = [pair for pair in pairs if pair["gold"] is not None]
    gold_absent = [pair for pair in pairs if pair["gold"] is None]
    abstained = [pair for pair in gold_present if pair["predicted"] is None]
    answered = [pair for pair in gold_present if pair["predicted"] is not None]
    correct = [pair for pair in answered if scalar_matches(field, pair["gold"], pair["predicted"])]
    volunteered = [pair for pair in gold_absent if pair["predicted"] is not None]

    return {
        "kind": "scalar",
        "gold_present": len(gold_present),
        "abstained": len(abstained),
        "answered": len(answered),
        "correct": len(correct),
        "exact_match_accuracy": _ratio(len(correct), len(gold_present)),
        "accuracy_when_answered": _ratio(len(correct), len(answered)),
        "abstention_rate": _ratio(len(abstained), len(gold_present)),
        "gold_absent": len(gold_absent),
        "value_added_when_gold_absent": len(volunteered),
        "unverified_value_added_rate": _ratio(len(volunteered), len(gold_absent)),
    }


def _list_field_metrics(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    gold_present = [pair for pair in pairs if pair["gold"] is not None]
    gold_absent = [pair for pair in pairs if pair["gold"] is None]
    abstained = [pair for pair in gold_present if pair["predicted"] is None]
    volunteered = [pair for pair in gold_absent if pair["predicted"] is not None]

    true_positives = 0
    predicted_total = 0
    gold_total = 0
    exact_set_matches = 0
    for pair in gold_present:
        gold_set = set(pair["gold"])
        predicted_set = set(pair["predicted"] or ())
        true_positives += len(gold_set & predicted_set)
        predicted_total += len(predicted_set)
        gold_total += len(gold_set)
        if gold_set == predicted_set:
            exact_set_matches += 1

    precision = _ratio(true_positives, predicted_total)
    recall = _ratio(true_positives, gold_total)
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "kind": "list",
        "gold_present": len(gold_present),
        "abstained": len(abstained),
        "answered": len(gold_present) - len(abstained),
        "true_positives": true_positives,
        "predicted_items": predicted_total,
        "gold_items": gold_total,
        "set_precision": precision,
        "set_recall": recall,
        "set_f1": f1,
        "exact_set_match": exact_set_matches,
        "exact_set_match_rate": _ratio(exact_set_matches, len(gold_present)),
        "abstention_rate": _ratio(len(abstained), len(gold_present)),
        "gold_absent": len(gold_absent),
        "value_added_when_gold_absent": len(volunteered),
        "unverified_value_added_rate": _ratio(len(volunteered), len(gold_absent)),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def aggregate_metrics(comparisons: list[dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Aggregate per-record comparisons into per-field metrics."""
    metrics: dict[str, dict[str, Any]] = {}
    for field in EXTRACTION_FIELDS:
        pairs = [comparison[field] for comparison in comparisons if field in comparison]
        if field in LIST_FIELDS:
            metrics[field] = _list_field_metrics(pairs)
        else:
            metrics[field] = _scalar_field_metrics(pairs, field)
    return metrics


def overall_summary(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Roll per-field counts up into headline totals."""
    scalar_gold = sum(m["gold_present"] for m in metrics.values() if m["kind"] == "scalar")
    scalar_correct = sum(m["correct"] for m in metrics.values() if m["kind"] == "scalar")
    gold_present = sum(m["gold_present"] for m in metrics.values())
    abstained = sum(m["abstained"] for m in metrics.values())
    gold_absent = sum(m["gold_absent"] for m in metrics.values())
    volunteered = sum(m["value_added_when_gold_absent"] for m in metrics.values())
    return {
        "scalar_gold_present": scalar_gold,
        "scalar_correct": scalar_correct,
        "scalar_exact_match_accuracy": _ratio(scalar_correct, scalar_gold),
        "gold_present": gold_present,
        "abstained": abstained,
        "abstention_rate": _ratio(abstained, gold_present),
        "gold_absent": gold_absent,
        "value_added_when_gold_absent": volunteered,
        "unverified_value_added_rate": _ratio(volunteered, gold_absent),
    }


def _format_ratio(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def build_markdown_report(
    *,
    generated_at: str,
    snapshot_path: Path,
    snapshot_date: str | None,
    snapshot_count: int,
    sample_size: int,
    unanswered: int = 0,
    model_name: str,
    prompt_version: str,
    provenance_column_present: bool,
    extraction_stats: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    """Render the per-field evaluation report."""
    lines: list[str] = []
    lines.append("# LLM Extraction Quality Evaluation")
    lines.append("")
    lines.append(f"- Generated at (UTC): {generated_at}")
    lines.append(f"- Snapshot: `{snapshot_path.name}`")
    lines.append(f"- Snapshot date: {snapshot_date or 'unknown'}")
    lines.append(f"- Snapshot records: {snapshot_count}")
    lines.append(f"- Scored sample: {sample_size}")
    lines.append(f"- Model: `{model_name}`")
    lines.append(f"- Prompt version: `{prompt_version}`")
    lines.append(f"- Provenance column present: {provenance_column_present}")
    lines.append(f"- Cache hits: {extraction_stats.get('cache_hits', 0)}")
    lines.append(f"- Live API calls: {extraction_stats.get('api_calls', 0)}")
    lines.append(f"- Failed calls (excluded, not cached): {extraction_stats.get('api_failures', 0)}")
    lines.append(f"- Unanswered records (excluded from scoring): {unanswered}")
    lines.append(
        f"- Answered with no extractable field: {extraction_stats.get('empty_extractions', 0)}"
    )
    lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append(
        "Gold labels are the values the deterministic regex/HTML parsers extracted for the "
        "same records — **not** human judgements. Agreement therefore measures whether the "
        "LLM reproduces the parser, and a parser value that is itself wrong counts against "
        "the LLM. Fields the ingest enrichment pass filled with the LLM are excluded from "
        "gold via the `llm_enriched_fields` provenance column."
    )
    lines.append("")
    lines.append(
        "Records where no response was obtained — a failed call, a rate limit, an "
        "API-call cap — are excluded from every number below rather than counted as "
        "abstentions. A call that never happened is not evidence about the model."
    )
    lines.append("")
    lines.append(
        "Where gold is empty the comparison is undecidable: the parser may have missed a "
        "value the listing states, or the listing may not state one. Those cases are "
        "reported below as *unverified value-added*, never as accuracy and never as a "
        "hallucination count — distinguishing the two requires reading the source text."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"- Scalar exact-match agreement: {_format_ratio(summary['scalar_exact_match_accuracy'])} "
        f"({summary['scalar_correct']}/{summary['scalar_gold_present']})"
    )
    lines.append(
        f"- Abstention rate (gold present, LLM said null): "
        f"{_format_ratio(summary['abstention_rate'])} "
        f"({summary['abstained']}/{summary['gold_present']})"
    )
    lines.append(
        f"- Unverified value-added rate (gold empty, LLM gave a value): "
        f"{_format_ratio(summary['unverified_value_added_rate'])} "
        f"({summary['value_added_when_gold_absent']}/{summary['gold_absent']})"
    )
    lines.append("")
    lines.append("## Scalar Fields")
    lines.append("")
    lines.append(
        "Exact match, except amounts which allow a $1 tolerance. `Acc (answered)` excludes "
        "abstentions, so it separates *being wrong* from *declining to answer*."
    )
    lines.append("")
    lines.append("| Field | Gold present | Correct | Exact-match acc | Acc (answered) | Abstention |")
    lines.append("|:---|---:|---:|---:|---:|---:|")
    for field in SCALAR_FIELDS:
        row = metrics[field]
        lines.append(
            f"| `{field}` | {row['gold_present']} | {row['correct']} | "
            f"{_format_ratio(row['exact_match_accuracy'])} | "
            f"{_format_ratio(row['accuracy_when_answered'])} | "
            f"{_format_ratio(row['abstention_rate'])} |"
        )
    lines.append("")
    lines.append("## List Fields")
    lines.append("")
    lines.append(
        "Micro-averaged set precision/recall over records where the parser produced a "
        "non-empty list. `Exact set` is the share of those records where the two sets match "
        "element for element."
    )
    lines.append("")
    lines.append("| Field | Gold present | Gold items | Predicted items | Precision | Recall | F1 | Exact set | Abstention |")
    lines.append("|:---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for field in LIST_FIELDS:
        row = metrics[field]
        lines.append(
            f"| `{field}` | {row['gold_present']} | {row['gold_items']} | "
            f"{row['predicted_items']} | {_format_ratio(row['set_precision'])} | "
            f"{_format_ratio(row['set_recall'])} | {_format_ratio(row['set_f1'])} | "
            f"{_format_ratio(row['exact_set_match_rate'])} | "
            f"{_format_ratio(row['abstention_rate'])} |"
        )
    lines.append("")
    lines.append("## Unverified Value-Added (gold empty)")
    lines.append("")
    lines.append(
        "Records where the parser produced nothing for the field and the LLM produced a "
        "value. This is the enrichment upside *and* the hallucination risk — the same "
        "number, unlabelled, because gold cannot adjudicate it. Treat it as a count to "
        "spot-check, not a score."
    )
    lines.append("")
    lines.append("| Field | Gold empty | LLM supplied a value | Rate |")
    lines.append("|:---|---:|---:|---:|")
    for field in EXTRACTION_FIELDS:
        row = metrics[field]
        lines.append(
            f"| `{field}` | {row['gold_absent']} | {row['value_added_when_gold_absent']} | "
            f"{_format_ratio(row['unverified_value_added_rate'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _snapshot_date_from_path(snapshot_path: Path) -> str | None:
    stem = snapshot_path.stem.replace("scholarships_snapshot_", "")
    try:
        return datetime.strptime(stem, "%Y%m%d").date().isoformat()
    except ValueError:
        return None


def run_extractions(
    sample_df: pd.DataFrame,
    *,
    client: LlmClient | None,
    model_name: str,
    processed_dir: Path,
    max_api_calls: int,
) -> tuple[list[dict[str, Any] | None], dict[str, Any]]:
    """Extract fields for every sampled record, reusing the cache where possible.

    An entry is ``None`` when no response was obtained at all — the call failed,
    the API-call cap was reached, or there is no client and no cached entry.
    Those records carry no evidence about the model and are dropped before
    scoring rather than counted as abstentions.
    """
    stats = {
        "cache_hits": 0,
        "api_calls": 0,
        "api_failures": 0,
        "empty_extractions": 0,
        "skipped_at_cap": 0,
    }
    extractions: list[dict[str, Any]] = []

    for _, row in sample_df.iterrows():
        record = {field: _row_get(row, field) for field in SOURCE_TEXT_FIELDS}
        key = compute_extraction_key(record, model_name=model_name)
        cached = load_extraction(extraction_path(key, model_name, processed_dir=processed_dir))
        if cached is not None:
            stats["cache_hits"] += 1
            fields = cached.get("fields")
            extracted = dict(fields) if isinstance(fields, dict) else {}
        elif client is None:
            extracted = None
        elif stats["api_calls"] >= max_api_calls:
            stats["skipped_at_cap"] += 1
            extracted = None
        else:
            stats["api_calls"] += 1
            extracted = get_or_extract(
                client,
                record,
                processed_dir=processed_dir,
                model_name=model_name,
            )
            if extracted is None:
                stats["api_failures"] += 1
        if extracted == {}:
            stats["empty_extractions"] += 1
        extractions.append(extracted)

    return extractions, stats


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    processed_dir = _resolve_path(args.processed_dir)
    snapshot_path = _resolve_snapshot_path(args.snapshot, args.processed_dir)
    snapshot_df = pd.read_parquet(snapshot_path)

    client = None if args.cache_only else client_from_env()
    model_name = _resolve_model_name(client, args.model_name)
    max_api_calls = args.max_api_calls if args.max_api_calls is not None else args.sample_size

    sample_df = select_sample(snapshot_df, sample_size=args.sample_size, seed=args.seed)
    if sample_df.empty:
        raise SystemExit(
            f"No records in '{snapshot_path.name}' have both parser-extracted gold fields "
            "and source text to extract from."
        )

    extractions, extraction_stats = run_extractions(
        sample_df,
        client=client,
        model_name=model_name,
        processed_dir=processed_dir,
        max_api_calls=max_api_calls,
    )
    if client is None and extraction_stats["cache_hits"] == 0:
        raise SystemExit(
            "No API key and no cached extractions, so there is nothing to score. Set "
            "SCHOLARSHIPCOACH_LLM_API_KEY (see docs/llm_extraction.md) and re-run."
        )

    comparisons: list[dict[str, dict[str, Any]]] = []
    per_record: list[dict[str, Any]] = []
    for (_, row), extracted in zip(sample_df.iterrows(), extractions, strict=True):
        if extracted is None:
            continue
        gold = gold_fields_for_row(row)
        comparison = compare_record(gold, extracted)
        comparisons.append(comparison)
        per_record.append(
            {
                "scholarship_id": str(_row_get(row, "scholarship_id")),
                "title": coerce_text(_row_get(row, "title")),
                "gold_fields": sorted(gold),
                "predicted_fields": sorted(
                    field
                    for field, pair in comparison.items()
                    if pair["predicted"] is not None
                ),
            }
        )

    unanswered = int(len(sample_df) - len(comparisons))
    if not comparisons:
        raise SystemExit(
            f"No record in the sample produced a response ({unanswered} unanswered). "
            "Check the API key, the model name, and the provider's rate limits."
        )

    metrics = aggregate_metrics(comparisons)
    summary = overall_summary(metrics)

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    generated_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    reports_dir = _resolve_path(args.reports_dir)
    markdown_path = reports_dir / f"llm_extraction_eval_{timestamp}.md"
    json_path = reports_dir / "artifacts" / f"llm_extraction_eval_{timestamp}.json"

    markdown_text = build_markdown_report(
        generated_at=generated_at,
        snapshot_path=snapshot_path,
        snapshot_date=_snapshot_date_from_path(snapshot_path),
        snapshot_count=int(len(snapshot_df)),
        sample_size=len(comparisons),
        unanswered=unanswered,
        model_name=model_name,
        prompt_version=EXTRACTION_PROMPT_VERSION,
        provenance_column_present=LLM_PROVENANCE_COLUMN in snapshot_df.columns,
        extraction_stats=extraction_stats,
        metrics=metrics,
        summary=summary,
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_text, encoding="utf-8")

    _write_json(
        json_path,
        {
            "generated_at": generated_at,
            "snapshot_path": str(snapshot_path),
            "snapshot_date": _snapshot_date_from_path(snapshot_path),
            "snapshot_count": int(len(snapshot_df)),
            "sample_size": len(comparisons),
            "sampled_records": int(len(sample_df)),
            "unanswered_excluded": unanswered,
            "seed": args.seed,
            "model": model_name,
            "prompt_version": EXTRACTION_PROMPT_VERSION,
            "provenance_column_present": LLM_PROVENANCE_COLUMN in snapshot_df.columns,
            "extraction_stats": extraction_stats,
            "gold_source": "deterministic parser output in the snapshot",
            "metrics": metrics,
            "summary": summary,
            "per_record": per_record,
        },
    )

    print(f"Wrote markdown report: {markdown_path}")
    print(f"Wrote JSON artifact: {json_path}")
    print(
        f"Scored {len(comparisons)}/{len(sample_df)} records "
        f"({unanswered} unanswered, excluded) | "
        f"cache_hits={extraction_stats['cache_hits']} "
        f"api_calls={extraction_stats['api_calls']} "
        f"api_failures={extraction_stats['api_failures']} | scalar exact-match "
        f"{_format_ratio(summary['scalar_exact_match_accuracy'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
