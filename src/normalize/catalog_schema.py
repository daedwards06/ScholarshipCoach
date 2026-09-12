"""JSON Schema loading and validation for curated catalog records.

The curated catalog is the project's primary data asset: one JSON file per
award under ``data/catalog/records/``, validated against
``data/catalog/schema.json``.  Both the ingest source and
``scripts/validate_catalog.py`` validate through here so a record that reaches
a snapshot is a record that passed the same checks CI enforces.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT_DIR = Path(__file__).resolve().parents[2]
CATALOG_DIR = ROOT_DIR / "data" / "catalog"
SCHEMA_PATH = CATALOG_DIR / "schema.json"
RECORDS_DIR = CATALOG_DIR / "records"


@lru_cache(maxsize=4)
def load_catalog_schema(schema_path: Path | None = None) -> dict[str, Any]:
    """Load and cache the curated catalog JSON Schema."""
    path = schema_path or SCHEMA_PATH
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_catalog_record(
    record: Any, *, schema: dict[str, Any] | None = None
) -> list[str]:
    """Return human-readable schema violations for ``record`` (empty when valid)."""
    resolved_schema = schema if schema is not None else load_catalog_schema()
    validator = Draft202012Validator(resolved_schema)
    errors = []
    for error in sorted(validator.iter_errors(record), key=lambda err: list(err.absolute_path)):
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        errors.append(f"{location}: {error.message}")
    return errors


def iter_catalog_files(records_dir: Path | None = None) -> list[Path]:
    """Return every catalog record file, sorted by name for deterministic runs."""
    directory = records_dir or RECORDS_DIR
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))
