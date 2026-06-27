"""Static curated JSON scholarship feed.

Reads a hand-curated JSON file committed to the repository.  This source
requires no live HTTP requests, making it fully deterministic for CI and
proving that the ``BaseSource`` abstraction generalizes beyond web scrapers.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.ingest.base import BaseSource, RawResponse
from src.normalize.canonical_id import generate_scholarship_id

logger = logging.getLogger(__name__)

_DEFAULT_FEED_PATH = Path(__file__).resolve().parents[3] / "data" / "static_feed" / "scholarships.json"
_SITE_NAME = "static_feed"


class StaticFeedSource(BaseSource):
    """Loads scholarships from a curated JSON file bundled with the repository.

    The ``fetch`` method reads the file and returns its bytes as a
    ``RawResponse`` — no HTTP client is used.  This makes the source
    fully deterministic and safe to run in CI without network access.
    """

    name = _SITE_NAME

    def __init__(self, feed_path: Path | None = None) -> None:
        self._feed_path = feed_path or _DEFAULT_FEED_PATH

    def fetch(self, http_client: Any) -> RawResponse:
        raw_bytes = self._feed_path.read_bytes()
        return RawResponse(content=raw_bytes, extension="json", fetched_at=self.utcnow())

    def parse(self, raw_content: bytes, *, fetched_at: datetime) -> list[dict[str, Any]]:
        try:
            items = json.loads(raw_content.decode("utf-8-sig"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error("StaticFeedSource: failed to parse JSON feed: %s", exc)
            return []

        if not isinstance(items, list):
            logger.error("StaticFeedSource: expected a JSON array, got %s", type(items).__name__)
            return []

        records: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            rec = self._map_item(item, fetched_at)
            if rec:
                records.append(rec)
        return records

    def _map_item(self, item: dict[str, Any], fetched_at: datetime) -> dict[str, Any] | None:
        title = str(item.get("title") or "").strip()
        if not title:
            return None

        source_url = str(item.get("source_url") or "").strip() or _DEFAULT_FEED_PATH.as_uri()
        sponsor = str(item.get("sponsor") or "").strip() or None
        description = str(item.get("description") or "").strip() or None
        eligibility_text = str(item.get("eligibility_text") or "").strip() or None

        amount_min = _float_or_none(item.get("amount_min"))
        amount_max = _float_or_none(item.get("amount_max"))

        raw_deadline = item.get("deadline")
        deadline: str | None = str(raw_deadline).strip() if raw_deadline else None

        keywords_raw = item.get("keywords")
        if isinstance(keywords_raw, list):
            keywords: list[str] | None = [str(k).strip() for k in keywords_raw if str(k).strip()] or None
        else:
            keywords = None

        scholarship_id = generate_scholarship_id(
            title=title,
            sponsor=sponsor,
            amount_min=amount_min,
            amount_max=amount_max,
            deadline=deadline,
            source_url=source_url,
        )

        return {
            "scholarship_id": scholarship_id,
            "source": self.name,
            "source_id": None,
            "source_url": source_url,
            "title": title,
            "sponsor": sponsor,
            "description": description,
            "eligibility_text": eligibility_text,
            "deadline": deadline,
            "amount_min": amount_min,
            "amount_max": amount_max,
            "is_recurring": None,
            "states_allowed": None,
            "majors_allowed": None,
            "min_gpa": None,
            "citizenship": None,
            "education_level": None,
            "essay_required": None,
            "essay_prompt": None,
            "keywords": keywords,
            "first_seen_at": fetched_at,
            "last_seen_at": fetched_at,
        }


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
        return result if result > 0 else None
    except (TypeError, ValueError):
        return None
