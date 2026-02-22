from __future__ import annotations

import json
import re
from datetime import datetime
from html import unescape
from typing import Any

from src.ingest.base import BaseSource, RawResponse
from src.normalize.canonical_id import generate_scholarship_id

_TAG_PATTERN = re.compile(r"<[^>]+>")
_MONEY_PATTERN = re.compile(r"\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")
_ISO_DATE_PATTERN = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")


class ScholarshipAmericaSource(BaseSource):
    name = "scholarship_america"
    endpoint = "https://www.scholarshipamerica.org/wp-json/wp/v2/posts"

    def fetch(self, http_client: Any) -> RawResponse:
        payload = http_client.get_json(
            self.endpoint,
            params={"search": "scholarship", "per_page": 50, "orderby": "date"},
        )
        content = json.dumps(payload, sort_keys=True).encode("utf-8")
        return RawResponse(content=content, extension="json", fetched_at=self.utcnow())

    def parse(self, raw_content: bytes, *, fetched_at: datetime) -> list[dict[str, Any]]:
        loaded = json.loads(raw_content.decode("utf-8-sig"))
        if isinstance(loaded, dict):
            candidates = [loaded]
        elif isinstance(loaded, list):
            candidates = [item for item in loaded if isinstance(item, dict)]
        else:
            return []

        normalized: list[dict[str, Any]] = []
        for item in candidates:
            title = _clean_text(_nested(item, "title", "rendered"))
            source_url = str(item.get("link") or self.endpoint)
            description = _clean_text(_nested(item, "excerpt", "rendered"))
            full_text = _clean_text(_nested(item, "content", "rendered"))
            text_for_signals = " ".join(part for part in [description, full_text] if part)

            amount_min, amount_max = _extract_amount_range(text_for_signals)
            deadline = _extract_deadline(text_for_signals)

            scholarship_id = generate_scholarship_id(
                title=title or source_url,
                sponsor="Scholarship America",
                amount_min=amount_min,
                amount_max=amount_max,
                deadline=deadline,
                source_url=source_url,
            )

            normalized.append(
                {
                    "scholarship_id": scholarship_id,
                    "source": self.name,
                    "source_id": _string_or_none(item.get("id")),
                    "source_url": source_url,
                    "title": title or source_url,
                    "sponsor": "Scholarship America",
                    "description": description or None,
                    "eligibility_text": text_for_signals or None,
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
                    "keywords": _extract_keywords(item),
                    "first_seen_at": fetched_at,
                    "last_seen_at": fetched_at,
                }
            )

        normalized.sort(key=lambda row: (row["source_id"] or "", row["title"]))
        return normalized


def _nested(obj: dict[str, Any], *keys: str) -> Any:
    current: Any = obj
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    as_str = str(value).strip()
    return as_str or None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = unescape(str(value))
    text = _TAG_PATTERN.sub(" ", text)
    return " ".join(text.split())


def _extract_amount_range(text: str) -> tuple[float | None, float | None]:
    matches = [float(chunk.replace(",", "")) for chunk in _MONEY_PATTERN.findall(text)]
    if not matches:
        return None, None
    return min(matches), max(matches)


def _extract_deadline(text: str) -> str | None:
    match = _ISO_DATE_PATTERN.search(text)
    if not match:
        return None
    return match.group(1)


def _extract_keywords(item: dict[str, Any]) -> list[str] | None:
    keywords = []
    slug = item.get("slug")
    if isinstance(slug, str) and slug.strip():
        keywords.append(slug.strip().lower())

    if not keywords:
        return None
    return sorted(set(keywords))
