"""Curated scholarship catalog source.

Reads the hand-curated catalog committed to the repository: one JSON file per
award under ``data/catalog/records/``, each validated against
``data/catalog/schema.json``.  This is the primary data asset of the project and
the only source that populates the eligibility axes (majors, GPA, education
level, status, requirements) that no scraper has ever filled.

No HTTP client is used, so the source is fully deterministic and safe in CI.
An unreadable or schema-invalid file is logged and skipped; it never fails the
run or the other records.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.ingest.base import BaseSource, RawResponse
from src.normalize.canonical_id import generate_scholarship_id
from src.normalize.catalog_schema import (
    RECORDS_DIR,
    iter_catalog_files,
    load_catalog_schema,
    validate_catalog_record,
)

logger = logging.getLogger(__name__)

_SITE_NAME = "curated_catalog"

_CYCLE_KEYS = ("recurring", "opens_month", "deadline_month")
_TEST_SCORE_KEYS = ("sat", "act")
_REQUIREMENT_KEYS = (
    "essay",
    "essay_prompts",
    "recommendation_letters",
    "transcript",
    "fafsa",
    "video_or_portfolio",
    "interview",
)
_PROVENANCE_KEYS = ("added_on", "verified_on", "verified_by", "source_kind")


class CuratedCatalogSource(BaseSource):
    """Loads every curated award record from ``data/catalog/records/``.

    ``fetch`` collects the readable record files into a single JSON array so the
    raw payload is cached like any other source; ``parse`` validates each record
    against the catalog schema before mapping it to a normalized record.
    """

    name = _SITE_NAME

    def __init__(
        self, records_dir: Path | None = None, schema_path: Path | None = None
    ) -> None:
        self._records_dir = records_dir or RECORDS_DIR
        self._schema_path = schema_path

    def fetch(self, http_client: Any) -> RawResponse:
        items: list[dict[str, Any]] = []
        for path in iter_catalog_files(self._records_dir):
            try:
                item = json.loads(path.read_text(encoding="utf-8-sig"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
                logger.error("CuratedCatalogSource: unreadable catalog file %s: %s", path, exc)
                continue
            if not isinstance(item, dict):
                logger.error(
                    "CuratedCatalogSource: %s must contain a JSON object, got %s",
                    path,
                    type(item).__name__,
                )
                continue
            items.append(item)

        payload = json.dumps(items, indent=2, sort_keys=True).encode("utf-8")
        return RawResponse(content=payload, extension="json", fetched_at=self.utcnow())

    def parse(self, raw_content: bytes, *, fetched_at: datetime) -> list[dict[str, Any]]:
        try:
            items = json.loads(raw_content.decode("utf-8-sig"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error("CuratedCatalogSource: failed to parse catalog payload: %s", exc)
            return []

        if not isinstance(items, list):
            logger.error(
                "CuratedCatalogSource: expected a JSON array, got %s", type(items).__name__
            )
            return []

        schema = load_catalog_schema(self._schema_path)
        records: list[dict[str, Any]] = []
        for item in items:
            errors = validate_catalog_record(item, schema=schema)
            if errors:
                label = item.get("catalog_id") if isinstance(item, dict) else "<not an object>"
                logger.error(
                    "CuratedCatalogSource: skipping invalid record %s: %s",
                    label,
                    "; ".join(errors),
                )
                continue
            records.append(self._map_item(item, fetched_at))
        return records

    def _map_item(self, item: dict[str, Any], fetched_at: datetime) -> dict[str, Any]:
        catalog_id = str(item["catalog_id"]).strip()
        requirements = _fixed_shape(item.get("requirements"), _REQUIREMENT_KEYS)
        essay_prompts = _string_list(requirements.get("essay_prompts"))
        requirements["essay_prompts"] = essay_prompts

        amount_min = _float_or_none(item.get("amount_min"))
        amount_max = _float_or_none(item.get("amount_max"))
        cycle = _fixed_shape(item.get("cycle"), _CYCLE_KEYS)

        return {
            "scholarship_id": generate_scholarship_id(
                title=str(item.get("title") or ""),
                sponsor=_text_or_none(item.get("sponsor")),
                amount_min=amount_min,
                amount_max=amount_max,
                deadline=_text_or_none(item.get("deadline")),
                source_url=str(item.get("source_url") or ""),
                catalog_id=catalog_id,
            ),
            "source": self.name,
            "source_id": catalog_id,
            "source_url": str(item.get("source_url") or "").strip(),
            "title": str(item.get("title") or "").strip(),
            "sponsor": _text_or_none(item.get("sponsor")),
            "description": _text_or_none(item.get("description")),
            "eligibility_text": _text_or_none(item.get("eligibility_text")),
            "deadline": _text_or_none(item.get("deadline")),
            "amount_min": amount_min,
            "amount_max": amount_max,
            "is_recurring": cycle.get("recurring"),
            "states_allowed": _string_list(item.get("states_allowed")),
            "majors_allowed": _string_list(item.get("majors_allowed")),
            "min_gpa": _float_or_none(item.get("min_gpa")),
            "citizenship": _text_or_none(item.get("citizenship")),
            "education_level": _text_or_none(item.get("education_level")),
            "essay_required": requirements.get("essay"),
            "essay_prompt": essay_prompts[0] if essay_prompts else None,
            "keywords": _string_list(item.get("keywords")),
            "first_seen_at": fetched_at,
            "last_seen_at": fetched_at,
            "catalog_id": catalog_id,
            "status": _text_or_none(item.get("status")),
            "cycle": cycle,
            "grade_levels": _string_list(item.get("grade_levels")),
            "counties_allowed": _string_list(item.get("counties_allowed")),
            "need_based": _bool_or_none(item.get("need_based")),
            "first_gen_only": _bool_or_none(item.get("first_gen_only")),
            "gender": _text_or_none(item.get("gender")),
            "heritage": _string_list(item.get("heritage")),
            "military_family": _bool_or_none(item.get("military_family")),
            "disability": _bool_or_none(item.get("disability")),
            "religion": _text_or_none(item.get("religion")),
            "employer_restricted": _string_list(item.get("employer_restricted")),
            "membership_required": _string_list(item.get("membership_required")),
            "min_test_scores": _fixed_shape(item.get("min_test_scores"), _TEST_SCORE_KEYS),
            "requirements": requirements,
            "renewal_terms": _text_or_none(item.get("renewal_terms")),
            "trust": _text_or_none(item.get("trust")),
            "provenance": _fixed_shape(item.get("provenance"), _PROVENANCE_KEYS),
            "notes": _text_or_none(item.get("notes")),
            "aliases": _string_list(item.get("aliases")),
        }


def _fixed_shape(value: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    """Return a dict carrying every key in ``keys``, so parquet infers one struct type."""
    source = value if isinstance(value, dict) else {}
    return {key: source.get(key) for key in keys}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip() or None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
        return result if result > 0 else None
    except (TypeError, ValueError):
        return None
