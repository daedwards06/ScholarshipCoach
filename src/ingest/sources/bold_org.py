"""Bold.org scholarship listing scraper.

Fetches the public scholarship listing from bold.org/scholarships/ and extracts
records from the Next.js ``__NEXT_DATA__`` JSON blob embedded in each page.
Falls back to HTML card parsing when the JSON blob is absent.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from html import unescape
from typing import Any
from urllib.parse import urljoin

from src.ingest.base import BaseSource, RawResponse
from src.normalize.canonical_id import generate_scholarship_id

logger = logging.getLogger(__name__)

_LISTING_URL = "https://bold.org/scholarships/"
_SITE_NAME = "bold_org"

_NEXT_DATA_PATTERN = re.compile(
    r'<script[^>]*id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
    flags=re.IGNORECASE | re.DOTALL,
)
_TAG_PATTERN = re.compile(r"<[^>]+>")
_WS_PATTERN = re.compile(r"\s+")
_MONEY_PATTERN = re.compile(r"\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")
_ISO_DATE_PATTERN = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_LONG_DATE_PATTERN = re.compile(
    r"\b((?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2},?\s+20\d{2})\b",
    flags=re.IGNORECASE,
)


def _strip_html(html: str) -> str:
    return _WS_PATTERN.sub(" ", unescape(_TAG_PATTERN.sub(" ", html))).strip()


def _extract_amount(text: str | None) -> tuple[float | None, float | None]:
    if not text:
        return None, None
    matches = [float(chunk.replace(",", "")) for chunk in _MONEY_PATTERN.findall(str(text))]
    if not matches:
        try:
            val = float(str(text).replace(",", "").replace("$", "").strip())
            if val > 0:
                return val, val
        except ValueError:
            pass
        return None, None
    return min(matches), max(matches)


def _parse_deadline(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = str(raw).strip()
    iso = _ISO_DATE_PATTERN.search(cleaned)
    if iso:
        return iso.group(1)
    long_match = _LONG_DATE_PATTERN.search(cleaned)
    if long_match:
        candidate = long_match.group(1).replace(",", "")
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                return datetime.strptime(candidate, fmt).date().isoformat()
            except ValueError:
                continue
    return None


def _extract_keywords(text: str | None) -> list[str] | None:
    if not text:
        return None
    words = re.findall(r"\b[a-z]{4,}\b", (text or "").lower())
    _STOP = {
        "that", "this", "with", "from", "they", "have", "will", "been", "their",
        "into", "when", "also", "some", "more", "than", "your", "open", "must",
        "each", "such", "only", "most", "very", "were", "what", "about",
    }
    unique = [w for w in dict.fromkeys(words) if w not in _STOP]
    return unique[:20] or None


def _records_from_next_data(next_data: dict[str, Any], fetched_at: datetime) -> list[dict[str, Any]]:
    """Walk the Next.js pageProps tree looking for scholarship arrays."""
    page_props = next_data.get("props", {}).get("pageProps", {})

    candidates: list[dict[str, Any]] = []
    for key in ("scholarships", "items", "results", "data"):
        value = page_props.get(key)
        if isinstance(value, list):
            candidates = value
            break

    if not candidates:
        # Try one level deeper (e.g. pageProps.initialData.scholarships)
        for sub in page_props.values():
            if isinstance(sub, dict):
                for key in ("scholarships", "items", "results"):
                    value = sub.get(key)
                    if isinstance(value, list):
                        candidates = value
                        break
            if candidates:
                break

    records: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        rec = _map_bold_item(item, fetched_at)
        if rec:
            records.append(rec)
    return records


def _map_bold_item(item: dict[str, Any], fetched_at: datetime) -> dict[str, Any] | None:
    title = str(item.get("title") or item.get("name") or "").strip()
    if not title:
        return None

    slug = str(item.get("slug") or item.get("id") or "").strip()
    source_url = item.get("url") or item.get("link") or (
        urljoin(_LISTING_URL, slug) if slug else _LISTING_URL
    )
    source_url = str(source_url).strip() or _LISTING_URL

    org = item.get("organization") or item.get("sponsor") or {}
    sponsor = (
        str(org.get("name") or org).strip()
        if isinstance(org, dict)
        else str(org).strip()
    ) or None

    description_raw = item.get("description") or item.get("summary") or ""
    description = _strip_html(str(description_raw)).strip() or None

    eligibility_raw = item.get("eligibility") or item.get("eligibility_text") or ""
    eligibility_text = _strip_html(str(eligibility_raw)).strip() or None

    amount_raw = item.get("amount") or item.get("award_amount") or ""
    if isinstance(amount_raw, (int, float)):
        amount_min = amount_max = float(amount_raw) if amount_raw > 0 else None
    else:
        amount_min, amount_max = _extract_amount(str(amount_raw))

    deadline = _parse_deadline(
        str(item.get("deadline") or item.get("close_date") or "")
    )

    keywords_src = " ".join(filter(None, [description_raw, eligibility_raw]))
    keywords = _extract_keywords(keywords_src)

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
        "source": _SITE_NAME,
        "source_id": slug or None,
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


def _records_from_html_cards(html: str, fetched_at: datetime) -> list[dict[str, Any]]:
    """Fallback: extract scholarship cards from rendered HTML."""
    card_pattern = re.compile(
        r'<(?:div|article)[^>]*class="[^"]*scholarship[^"]*"[^>]*>(.*?)</(?:div|article)>',
        flags=re.IGNORECASE | re.DOTALL,
    )
    title_pattern = re.compile(r"<h[1-4][^>]*>(.*?)</h[1-4]>", flags=re.IGNORECASE | re.DOTALL)
    link_pattern = re.compile(r'href="(/scholarships/[^"]+)"', flags=re.IGNORECASE)

    records: list[dict[str, Any]] = []
    for card_match in card_pattern.finditer(html):
        card_html = card_match.group(1)
        card_text = _strip_html(card_html)

        title_match = title_pattern.search(card_html)
        title = _strip_html(title_match.group(1)) if title_match else ""
        if not title:
            continue

        link_match = link_pattern.search(card_html)
        path = link_match.group(1) if link_match else f"/scholarships/{re.sub(r'[^a-z0-9]+', '-', title.lower())}"
        source_url = urljoin(_LISTING_URL, path)

        amount_min, amount_max = _extract_amount(card_text)
        deadline = _parse_deadline(card_text)
        keywords = _extract_keywords(card_text)

        scholarship_id = generate_scholarship_id(
            title=title,
            sponsor=None,
            amount_min=amount_min,
            amount_max=amount_max,
            deadline=deadline,
            source_url=source_url,
        )

        records.append({
            "scholarship_id": scholarship_id,
            "source": _SITE_NAME,
            "source_id": path.rstrip("/").split("/")[-1] or None,
            "source_url": source_url,
            "title": title,
            "sponsor": None,
            "description": card_text or None,
            "eligibility_text": None,
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
        })
    return records


class BoldOrgSource(BaseSource):
    """Scrapes the public Bold.org scholarship listing at bold.org/scholarships/.

    Extracts scholarship data from the Next.js ``__NEXT_DATA__`` JSON blob
    embedded in the listing page.  Falls back to HTML card parsing when the
    JSON blob is absent (e.g. in test fixtures without Next.js hydration).
    """

    name = _SITE_NAME
    listing_url = _LISTING_URL

    def fetch(self, http_client: Any) -> RawResponse:
        html_bytes = http_client.get_bytes(self.listing_url)
        return RawResponse(content=html_bytes, extension="html", fetched_at=self.utcnow())

    def parse(self, raw_content: bytes, *, fetched_at: datetime) -> list[dict[str, Any]]:
        html = raw_content.decode("utf-8", errors="replace")

        next_data_match = _NEXT_DATA_PATTERN.search(html)
        if next_data_match:
            try:
                next_data = json.loads(next_data_match.group(1))
                records = _records_from_next_data(next_data, fetched_at)
                if records:
                    logger.info("BoldOrgSource: extracted %d records from __NEXT_DATA__", len(records))
                    return records
            except json.JSONDecodeError:
                logger.warning("BoldOrgSource: failed to parse __NEXT_DATA__ JSON, falling back to HTML")

        records = _records_from_html_cards(html, fetched_at)
        logger.info("BoldOrgSource: extracted %d records from HTML cards", len(records))
        return records
