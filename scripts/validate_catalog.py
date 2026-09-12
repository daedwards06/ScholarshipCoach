"""Validate every curated catalog record against the catalog JSON Schema.

Checks each file in ``data/catalog/records/`` for readable JSON, schema
conformance, a filename matching its ``catalog_id``, slug uniqueness across the
catalog, and a usable ``source_url``.  Exits non-zero on any error so CI blocks
a malformed record before it reaches a snapshot.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

from src.normalize.catalog_schema import (
    RECORDS_DIR,
    SCHEMA_PATH,
    iter_catalog_files,
    load_catalog_schema,
    validate_catalog_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the curated scholarship catalog.")
    parser.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    return parser.parse_args()


def _url_errors(source_url: object) -> list[str]:
    parsed = urlparse(str(source_url or ""))
    if parsed.scheme != "https" or not parsed.netloc:
        return [f"source_url: expected an https URL with a host, got {source_url!r}"]
    return []


def validate_catalog(records_dir: Path, schema_path: Path) -> list[str]:
    """Return every validation error found in ``records_dir`` (empty when clean)."""
    schema = load_catalog_schema(schema_path)
    files = iter_catalog_files(records_dir)
    if not files:
        return [f"{records_dir}: no catalog records found"]

    errors: list[str] = []
    seen_ids: dict[str, Path] = {}
    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            errors.append(f"{path.name}: not valid JSON ({exc})")
            continue

        if not isinstance(record, dict):
            errors.append(f"{path.name}: expected a JSON object, got {type(record).__name__}")
            continue

        schema_errors = validate_catalog_record(record, schema=schema)
        errors.extend(f"{path.name}: {message}" for message in schema_errors)
        if schema_errors:
            continue

        catalog_id = str(record["catalog_id"])
        if path.stem != catalog_id:
            errors.append(f"{path.name}: filename must match catalog_id {catalog_id!r}")

        duplicate = seen_ids.get(catalog_id)
        if duplicate is not None:
            errors.append(f"{path.name}: duplicate catalog_id {catalog_id!r} (also {duplicate.name})")
        else:
            seen_ids[catalog_id] = path

        errors.extend(f"{path.name}: {message}" for message in _url_errors(record.get("source_url")))

    return errors


def main() -> int:
    args = parse_args()
    errors = validate_catalog(args.records_dir, args.schema)
    if errors:
        print(f"Catalog validation FAILED with {len(errors)} error(s):")
        for message in errors:
            print(f"  - {message}")
        return 1

    count = len(iter_catalog_files(args.records_dir))
    print(f"Catalog validation passed: {count} record(s) in {args.records_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
