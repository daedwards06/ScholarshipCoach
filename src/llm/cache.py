"""Content-addressable cache for LLM structured extractions.

This is what reconciles a nondeterministic API with the project's determinism
principle.  An extraction is keyed by the listing's source text, the model
name, and :data:`~src.llm.extraction.EXTRACTION_PROMPT_VERSION`, so an
unchanged listing is extracted exactly once: re-runs are free, offline, and
reproducible, and the snapshot build never depends on live API behavior.

Mirrors the layout of :mod:`src.embeddings.cache` — a per-model sidecar store
under ``data/processed/`` holding one JSON document per key:

    data/processed/llm_extractions/<model_sanitized>/<key>.json
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.llm.client import LlmClient
from src.llm.extraction import EXTRACTION_PROMPT_VERSION, extract_fields
from src.text_utils import normalize_text

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
EXTRACTION_TEXT_FIELDS = ("title", "description", "eligibility_text")


def resolve_processed_dir(processed_dir: Path | None = None) -> Path:
    """Return the resolved processed-data directory, defaulting to ``data/processed/``."""
    if processed_dir is None:
        return DEFAULT_PROCESSED_DIR
    return processed_dir if processed_dir.is_absolute() else ROOT_DIR / processed_dir


def sanitize_model_name(model_name: str) -> str:
    """Return a filesystem-safe version of the model name for use as a directory name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_name)).strip("_") or "unknown"


def build_extraction_text_fingerprint(record: Mapping[str, Any]) -> str:
    """Build a pipe-separated fingerprint from the listing's source text fields."""
    return " | ".join(normalize_text(record.get(field)) for field in EXTRACTION_TEXT_FIELDS)


def compute_extraction_key(
    record: Mapping[str, Any],
    *,
    model_name: str,
    prompt_version: str = EXTRACTION_PROMPT_VERSION,
) -> str:
    """Compute the SHA-1 cache key for one listing's extraction.

    The key covers the source text, the model, and the prompt version, so a
    prompt change or a model switch invalidates previously cached extractions
    rather than silently reusing them.
    """
    payload = "|".join(
        (
            build_extraction_text_fingerprint(record),
            normalize_text(model_name),
            normalize_text(prompt_version),
        )
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def extraction_cache_dir(
    model_name: str,
    *,
    processed_dir: Path | None = None,
) -> Path:
    """Return the directory holding cached extractions for the given model."""
    return resolve_processed_dir(processed_dir) / "llm_extractions" / sanitize_model_name(
        model_name
    )


def extraction_path(
    key: str,
    model_name: str,
    *,
    processed_dir: Path | None = None,
) -> Path:
    """Return the JSON path for a single cached extraction."""
    return extraction_cache_dir(model_name, processed_dir=processed_dir) / f"{key}.json"


def load_extraction(path: Path) -> dict[str, Any] | None:
    """Load a cached extraction document, or ``None`` if missing or unreadable."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable extraction cache entry %s: %s", path, exc)
        return None
    if not isinstance(document, dict):
        return None
    return document


def write_extraction(
    path: Path,
    fields: Mapping[str, Any],
    *,
    model_name: str,
    prompt_version: str = EXTRACTION_PROMPT_VERSION,
) -> dict[str, Any]:
    """Write one extraction document atomically and return it."""
    document: dict[str, Any] = {
        "fields": dict(fields),
        "model": model_name,
        "prompt_version": prompt_version,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f"{path.name}.tmp"
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
    temp_path.replace(path)
    return document


def get_or_extract(
    client: LlmClient | None,
    record: Mapping[str, Any],
    *,
    processed_dir: Path | None = None,
    model_name: str | None = None,
    prompt_version: str = EXTRACTION_PROMPT_VERSION,
) -> dict[str, Any]:
    """Return validated extracted fields for one listing, using the cache first.

    A cache hit returns the stored fields with **no API call**.  On a miss the
    client is called, the validated result is written, and the fields are
    returned.  With ``client=None`` (feature disabled) a miss returns ``{}`` —
    cached extractions still resolve, so a keyless run can reuse prior work.

    Args:
        client: The LLM client, or ``None`` when the feature is disabled.
        record: Mapping supplying ``title``, ``description``, ``eligibility_text``.
        processed_dir: Override for the data/processed directory.
        model_name: Model identifier for the cache namespace.  Defaults to the
            client's model; required when ``client`` is ``None``.
        prompt_version: Prompt version participating in the cache key.

    Returns:
        A dict of validated fields, possibly empty.
    """
    resolved_model = model_name or (client.model if client is not None else None)
    if not resolved_model:
        return {}

    key = compute_extraction_key(
        record,
        model_name=resolved_model,
        prompt_version=prompt_version,
    )
    path = extraction_path(key, resolved_model, processed_dir=processed_dir)

    cached = load_extraction(path)
    if cached is not None:
        fields = cached.get("fields")
        return dict(fields) if isinstance(fields, dict) else {}

    if client is None:
        return {}

    fields = extract_fields(
        client,
        title=record.get("title"),
        description=record.get("description"),
        eligibility_text=record.get("eligibility_text"),
    )
    write_extraction(
        path,
        fields,
        model_name=resolved_model,
        prompt_version=prompt_version,
    )
    return fields
