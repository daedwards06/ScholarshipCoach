"""Deterministic scholarship ID generation via content-based hashing.

``generate_scholarship_id`` produces a stable SHA-1 hex digest from the
canonical identity fields (title, sponsor, amounts, deadline, source domain)
so the same scholarship is assigned the same ID across ingest runs.

Curated catalog records instead carry a ``catalog_id`` slug, which is hashed
alone.  Because the deadline is excluded, a recurring award keeps one
``scholarship_id`` across application cycles and can be tracked year to year.
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Optional
from urllib.parse import urlparse

from src.text_utils import normalize_text as _normalize_text_base


def _normalize_text(value: Optional[str]) -> str:
    """Normalize for hash stability: lowercase, strip, and collapse internal whitespace."""
    return " ".join(_normalize_text_base(value).split())


def _normalize_amount(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}"


def _normalize_deadline(value: Optional[date | datetime | str]) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    cleaned = value.strip()
    if not cleaned:
        return ""

    # Prefer normalized ISO date strings when possible.
    candidate = cleaned.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate).date().isoformat()
    except ValueError:
        pass

    try:
        return date.fromisoformat(cleaned).isoformat()
    except ValueError:
        return cleaned.lower()


def _normalize_source_domain(source_url: Optional[str]) -> str:
    if not source_url:
        return ""

    parsed = urlparse(source_url.strip())
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def generate_scholarship_id(
    *,
    title: str,
    sponsor: Optional[str],
    amount_min: Optional[float],
    amount_max: Optional[float],
    deadline: Optional[date | datetime | str],
    source_url: Optional[str],
    catalog_id: Optional[str] = None,
) -> str:
    """Build a deterministic scholarship_id from canonical scholarship identity fields.

    When ``catalog_id`` is given it is the only input to the hash, so the ID
    survives deadline, amount, and title revisions between cycles.  Scraped
    sources omit it and keep the content-hash behavior.
    """

    slug = _normalize_text(catalog_id)
    if slug:
        return hashlib.sha1(f"catalog|{slug}".encode("utf-8")).hexdigest()

    payload = "|".join(
        [
            _normalize_text(title),
            _normalize_text(sponsor),
            _normalize_amount(amount_min),
            _normalize_amount(amount_max),
            _normalize_deadline(deadline),
            _normalize_source_domain(source_url),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()
