from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4
from typing import Any

import pandas as pd

from src.embeddings.cache import ensure_embedding_store_for_df
from src.embeddings.model import DEFAULT_MODEL_NAME

SNAPSHOT_PREFIX = "scholarships_snapshot_"
CHANGES_PREFIX = "changes_"
SNAPSHOT_PATTERN = re.compile(r"^scholarships_snapshot_(\d{8})\.parquet$")

REQUIRED_COLUMNS = [
    "scholarship_id",
    "source",
    "source_id",
    "source_url",
    "title",
    "sponsor",
    "description",
    "eligibility_text",
    "deadline",
    "amount_min",
    "amount_max",
    "is_recurring",
    "states_allowed",
    "majors_allowed",
    "min_gpa",
    "citizenship",
    "education_level",
    "essay_required",
    "essay_prompt",
    "keywords",
    "first_seen_at",
    "last_seen_at",
]

TRACKED_DIFF_FIELDS = ("deadline", "amount", "title", "eligibility_text")
SNAPSHOT_COLUMNS = [*REQUIRED_COLUMNS, "embedding_key"]


def _coerce_output_date(run_date: date | str | None) -> date:
    if run_date is None:
        return datetime.now(tz=UTC).date()
    if isinstance(run_date, date):
        return run_date
    return datetime.strptime(run_date, "%Y%m%d").date()


def _snapshot_filename(run_date: date) -> str:
    return f"{SNAPSHOT_PREFIX}{run_date.strftime('%Y%m%d')}.parquet"


def _changes_filename(run_date: date) -> str:
    return f"{CHANGES_PREFIX}{run_date.strftime('%Y%m%d')}.json"


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, float) and pd.isna(value):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            return value.tz_convert("UTC").isoformat()
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _normalize_amount(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "amount_min": _jsonable(record.get("amount_min")),
        "amount_max": _jsonable(record.get("amount_max")),
    }


def _tracked_value(record: dict[str, Any], field: str) -> Any:
    if field == "amount":
        return _normalize_amount(record)
    return _jsonable(record.get(field))


def list_snapshot_files(processed_dir: Path) -> list[Path]:
    snapshots: list[tuple[datetime, Path]] = []
    for candidate in processed_dir.glob("scholarships_snapshot_*.parquet"):
        match = SNAPSHOT_PATTERN.match(candidate.name)
        if not match:
            continue
        snapshot_date = datetime.strptime(match.group(1), "%Y%m%d")
        snapshots.append((snapshot_date, candidate))

    snapshots.sort(key=lambda item: item[0])
    return [item[1] for item in snapshots]


def get_latest_snapshot_path(processed_dir: Path) -> Path | None:
    snapshots = list_snapshot_files(processed_dir)
    if not snapshots:
        return None
    return snapshots[-1]


def load_latest_snapshot_df(processed_dir: Path) -> pd.DataFrame:
    latest_path = get_latest_snapshot_path(processed_dir)
    if latest_path is None:
        raise FileNotFoundError(
            f"No snapshot parquet found in '{processed_dir}'. "
            "Run ingest to generate data/processed/scholarships_snapshot_YYYYMMDD.parquet."
        )
    return pd.read_parquet(latest_path)


def find_prior_snapshot(processed_dir: Path, target_date: date) -> Path | None:
    target_name = _snapshot_filename(target_date)
    candidates = [path for path in list_snapshot_files(processed_dir) if path.name != target_name]
    return candidates[-1] if candidates else None


def prepare_snapshot_df(records: pd.DataFrame) -> pd.DataFrame:
    snapshot_df = records.copy()
    for column in REQUIRED_COLUMNS:
        if column not in snapshot_df.columns:
            snapshot_df[column] = None

    ordered = snapshot_df[REQUIRED_COLUMNS]
    return ordered.sort_values(by=["scholarship_id"], kind="mergesort").reset_index(drop=True)


def write_parquet_atomic(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.parent / f"{output_path.name}.{uuid4().hex}.tmp"
    try:
        df.to_parquet(temp_path, index=False, engine="pyarrow")
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _records_by_id(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if df.empty:
        return {}
    keyed = df.set_index("scholarship_id", drop=False).to_dict(orient="index")
    return {str(key): value for key, value in keyed.items()}


def build_delta(current_df: pd.DataFrame, prior_df: pd.DataFrame | None) -> dict[str, Any]:
    prior = prior_df if prior_df is not None else pd.DataFrame(columns=current_df.columns)
    current_records = _records_by_id(current_df)
    prior_records = _records_by_id(prior)

    current_ids = set(current_records)
    prior_ids = set(prior_records)

    added_ids = sorted(current_ids - prior_ids)
    removed_ids = sorted(prior_ids - current_ids)
    shared_ids = sorted(current_ids & prior_ids)

    added = [_jsonable(current_records[scholarship_id]) for scholarship_id in added_ids]
    removed = [_jsonable(prior_records[scholarship_id]) for scholarship_id in removed_ids]

    changed: list[dict[str, Any]] = []
    for scholarship_id in shared_ids:
        old_record = prior_records[scholarship_id]
        new_record = current_records[scholarship_id]
        fields_changed: dict[str, Any] = {}
        for field in TRACKED_DIFF_FIELDS:
            old_value = _tracked_value(old_record, field)
            new_value = _tracked_value(new_record, field)
            if old_value != new_value:
                fields_changed[field] = {"old": old_value, "new": new_value}

        if fields_changed:
            changed.append(
                {"scholarship_id": scholarship_id, "fields_changed": fields_changed}
            )

    return {"added": added, "removed": removed, "changed": changed}


def write_json_atomic(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.parent / f"{output_path.name}.{uuid4().hex}.tmp"
    try:
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def build_and_write_snapshot(
    records: pd.DataFrame,
    *,
    processed_dir: Path,
    run_date: date | str | None = None,
    embedding_model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[Path, Path, dict[str, Any]]:
    snapshot_date = _coerce_output_date(run_date)
    snapshot_df = prepare_snapshot_df(records)
    snapshot_df = ensure_embedding_store_for_df(
        snapshot_df,
        embedding_model_name,
        processed_dir=processed_dir,
    )
    snapshot_df = snapshot_df[SNAPSHOT_COLUMNS]

    processed_dir.mkdir(parents=True, exist_ok=True)
    prior_snapshot_path = find_prior_snapshot(processed_dir, snapshot_date)
    prior_df = pd.read_parquet(prior_snapshot_path) if prior_snapshot_path else None

    snapshot_path = processed_dir / _snapshot_filename(snapshot_date)
    changes_path = processed_dir / _changes_filename(snapshot_date)

    delta = build_delta(snapshot_df, prior_df)
    write_parquet_atomic(snapshot_df, snapshot_path)
    write_json_atomic(delta, changes_path)
    return snapshot_path, changes_path, delta
