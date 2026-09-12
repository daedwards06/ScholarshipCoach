"""Open Scholarships structured-feed connector.

Reads the openly licensed, machine-readable scholarship directory published by
Grudged LLC at ``https://scholarships.grudged.io``.  Unlike the scrapers, this
source already ships structured eligibility fields, so its records carry
``trust = structured_feed`` rather than ``aggregator``.

Data is licensed CC BY 4.0; the required attribution travels with every record
in ``provenance`` and is repeated in the README "Data sources" section.

Verified live on 2026-09-12 against ``/api/scholarships?state=NC``: the response
is an envelope ``{total, limit, offset, license, attribution, results[]}``,
``limit`` is capped at 500, ``offset`` pages, and ``?state=XX`` returns that
state's records plus the national ones.  ``tests/resources/
open_scholarships_sample.json`` pins that shape.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from src.ingest.base import BaseSource, RawResponse
from src.normalize.canonical_id import generate_scholarship_id

logger = logging.getLogger(__name__)

_SITE_NAME = "open_scholarships"
_API_URL = "https://scholarships.grudged.io/api/scholarships"
_DEFAULT_STATE = "NC"
_PAGE_LIMIT = 500
_MAX_PAGES = 20

LICENSE = "CC-BY-4.0"
ATTRIBUTION = (
    "Open Scholarships by Grudged LLC - "
    "https://github.com/Grudged/open-scholarships (CC BY 4.0)"
)

_TRUST = "structured_feed"
_SOURCE_KIND = "structured_feed"

# Feed ``availability`` -> catalog ``status`` vocabulary. Rolling intake is an
# award you can apply to today, so it reads as open.
_STATUS_BY_AVAILABILITY = {
    "open": "open",
    "closed": "closed",
    "upcoming": "upcoming",
    "rolling": "open",
    "unknown": "unknown",
}

# Feed ``deadline.type`` -> ``is_recurring``. "varies" carries no signal.
_RECURRING_BY_DEADLINE_TYPE = {
    "annual": True,
    "rolling": True,
    "one-time": False,
}

# Feed education levels -> the Stage 1 level vocabulary.
_EDUCATION_LEVELS = {
    "high-school": "high school",
    "high-school-senior": "high school",
    "undergraduate": "undergraduate",
    "community-college": "undergraduate",
    "graduate": "graduate",
    "vocational": "vocational",
}

_NON_STATE_RESIDENCY = {"US", "USA"}

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


class OpenScholarshipsSource(BaseSource):
    """Pulls the Open Scholarships feed for one state plus national awards.

    ``fetch`` pages the API with ``limit``/``offset`` and caches every page in a
    single JSON envelope so the raw payload is replayable; ``parse`` maps each
    record onto the normalized schema without guessing: a field the feed leaves
    empty stays ``None``.
    """

    name = _SITE_NAME

    def __init__(
        self,
        *,
        state: str = _DEFAULT_STATE,
        api_url: str = _API_URL,
        page_limit: int = _PAGE_LIMIT,
        max_pages: int = _MAX_PAGES,
    ) -> None:
        self._state = state
        self._api_url = api_url
        self._page_limit = page_limit
        self._max_pages = max_pages

    def fetch(self, http_client: Any) -> RawResponse:
        results: list[Any] = []
        license_name: str | None = None
        attribution: str | None = None
        total: int | None = None

        for page in range(self._max_pages):
            payload = http_client.get_json(
                self._api_url,
                params={
                    "state": self._state,
                    "limit": self._page_limit,
                    "offset": page * self._page_limit,
                },
            )
            if not isinstance(payload, dict):
                logger.error(
                    "OpenScholarshipsSource: expected a JSON object, got %s",
                    type(payload).__name__,
                )
                break

            license_name = license_name or _text_or_none(payload.get("license"))
            attribution = attribution or _text_or_none(payload.get("attribution"))
            total = _int_or_none(payload.get("total")) if total is None else total

            page_results = payload.get("results")
            if not isinstance(page_results, list) or not page_results:
                break
            results.extend(page_results)

            if total is not None and len(results) >= total:
                break
        else:
            logger.warning(
                "OpenScholarshipsSource: stopped after %d pages; feed may have more records.",
                self._max_pages,
            )

        envelope = {
            "license": license_name or LICENSE,
            "attribution": attribution or ATTRIBUTION,
            "total": total if total is not None else len(results),
            "results": results,
        }
        content = json.dumps(envelope, indent=2, sort_keys=True).encode("utf-8")
        return RawResponse(content=content, extension="json", fetched_at=self.utcnow())

    def parse(self, raw_content: bytes, *, fetched_at: datetime) -> list[dict[str, Any]]:
        try:
            payload = json.loads(raw_content.decode("utf-8-sig"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error("OpenScholarshipsSource: failed to parse feed payload: %s", exc)
            return []

        if isinstance(payload, list):
            items: Any = payload
            attribution = ATTRIBUTION
            license_name = LICENSE
        elif isinstance(payload, dict):
            items = payload.get("results")
            attribution = _text_or_none(payload.get("attribution")) or ATTRIBUTION
            license_name = _text_or_none(payload.get("license")) or LICENSE
        else:
            logger.error(
                "OpenScholarshipsSource: expected a JSON object or array, got %s",
                type(payload).__name__,
            )
            return []

        if not isinstance(items, list):
            logger.error(
                "OpenScholarshipsSource: 'results' must be an array, got %s",
                type(items).__name__,
            )
            return []

        records: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                logger.error(
                    "OpenScholarshipsSource: skipping non-object record (%s)",
                    type(item).__name__,
                )
                continue
            record = self._map_item(
                item,
                fetched_at=fetched_at,
                attribution=attribution,
                license_name=license_name,
            )
            if record is not None:
                records.append(record)
        return records

    def _map_item(
        self,
        item: dict[str, Any],
        *,
        fetched_at: datetime,
        attribution: str,
        license_name: str,
    ) -> dict[str, Any] | None:
        title = _text_or_none(item.get("name"))
        if not title:
            logger.error("OpenScholarshipsSource: skipping record without a name: %r", item.get("id"))
            return None

        award = _mapping(item.get("award"))
        deadline_block = _mapping(item.get("deadline"))
        eligibility = _mapping(item.get("eligibility"))
        geo = _mapping(item.get("geo"))
        links = _mapping(item.get("links"))
        provenance = _mapping(item.get("provenance"))

        source_url = (
            _text_or_none(provenance.get("source_url"))
            or _text_or_none(links.get("info_url"))
            or ""
        )
        amount_min = _positive_float(award.get("amount_min"))
        amount_max = _positive_float(award.get("amount_max"))
        deadline = _iso_date_or_none(deadline_block.get("date"))
        opens = _iso_date_or_none(deadline_block.get("opens"))
        is_recurring = _RECURRING_BY_DEADLINE_TYPE.get(
            _normalized_token(deadline_block.get("type"))
        )
        source_id = _text_or_none(item.get("id")) or ""

        return {
            "scholarship_id": generate_scholarship_id(
                title=title,
                sponsor=_text_or_none(item.get("sponsor")),
                amount_min=amount_min,
                amount_max=amount_max,
                deadline=deadline,
                source_url=source_url,
            ),
            "source": self.name,
            "source_id": source_id,
            "source_url": source_url,
            "title": title,
            "sponsor": _text_or_none(item.get("sponsor")),
            "description": _text_or_none(item.get("summary")),
            "eligibility_text": _joined(eligibility.get("other")),
            "deadline": deadline,
            "amount_min": amount_min,
            "amount_max": amount_max,
            "is_recurring": is_recurring,
            "states_allowed": _states_allowed(eligibility.get("residency"), geo.get("state")),
            "majors_allowed": _string_list(eligibility.get("fields_of_study")),
            "min_gpa": _positive_float(eligibility.get("gpa_min")),
            "citizenship": _single(eligibility.get("citizenship")),
            "education_level": _education_level(eligibility.get("education_level")),
            "essay_required": None,
            "essay_prompt": None,
            "keywords": _string_list(eligibility.get("tags")),
            "first_seen_at": fetched_at,
            "last_seen_at": fetched_at,
            "catalog_id": None,
            "status": _STATUS_BY_AVAILABILITY.get(
                _normalized_token(item.get("availability")), "unknown"
            ),
            "cycle": {
                "recurring": is_recurring,
                "opens_month": _month(opens),
                "deadline_month": _month(deadline),
            },
            "grade_levels": [],
            "counties_allowed": _string_list(geo.get("counties")),
            "need_based": True if _normalized_token(award.get("basis")) == "need" else None,
            "first_gen_only": None,
            "gender": None,
            "heritage": [],
            "military_family": None,
            "disability": None,
            "religion": None,
            "employer_restricted": [],
            "membership_required": [],
            "min_test_scores": dict.fromkeys(_TEST_SCORE_KEYS),
            "requirements": {
                **dict.fromkeys(_REQUIREMENT_KEYS),
                "essay_prompts": [],
            },
            "renewal_terms": _text_or_none(award.get("notes")),
            "trust": _TRUST,
            "provenance": {
                "added_on": _iso_date_or_none(provenance.get("added")),
                "verified_on": _iso_date_or_none(provenance.get("last_verified")),
                "verified_by": _text_or_none(provenance.get("source_name")),
                "source_kind": _SOURCE_KIND,
                "source_url": source_url,
                "license": license_name,
                "attribution": attribution,
            },
            "notes": _text_or_none(deadline_block.get("notes")),
        }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text_or_none(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list, bool)):
        return None
    return str(value).strip() or None


def _normalized_token(value: Any) -> str:
    text = _text_or_none(value)
    return text.lower() if text else ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _joined(value: Any) -> str | None:
    return "; ".join(_string_list(value)) or None


def _single(value: Any) -> str | None:
    """Return the lone entry of a list field, or ``None`` — never pick a winner."""
    entries = _string_list(value)
    return entries[0] if len(entries) == 1 else None


def _positive_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iso_date_or_none(value: Any) -> str | None:
    text = _text_or_none(value)
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


def _month(iso_date: str | None) -> int | None:
    return int(iso_date[5:7]) if iso_date else None


def _states_allowed(residency: Any, state: Any) -> list[str]:
    """Return the state codes that actually restrict the award.

    ``residency = ["US"]`` on a national award is not a restriction, so it is
    dropped; leaving it in would make every national record fail Stage 1's
    state check.
    """
    codes = [
        code.upper()
        for code in _string_list(residency)
        if code.upper() not in _NON_STATE_RESIDENCY
    ]
    geo_state = _text_or_none(state)
    if geo_state:
        codes.append(geo_state.upper())
    return list(dict.fromkeys(codes))


def _education_level(value: Any) -> str | None:
    """Collapse the feed's level list to one Stage 1 level, or ``None``.

    The normalized schema carries a single level and Stage 1 filters on it, so a
    record spanning several levels must stay empty rather than be narrowed to a
    level the award does not actually require.
    """
    levels = {
        _EDUCATION_LEVELS[token]
        for token in (_normalized_token(entry) for entry in _string_list(value))
        if token in _EDUCATION_LEVELS
    }
    return levels.pop() if len(levels) == 1 else None
