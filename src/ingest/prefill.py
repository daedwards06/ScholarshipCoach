"""Turn a pasted scholarship URL into a mostly-filled hand-entry form.

Hand entry is the backbone of the catalog, so it has to be fast: paste a URL,
get title, sponsor, description and every field the page states outright,
then confirm.  Nothing here ever guesses -- a field with no evidence on the
page comes back ``None``, and when a page offers several dates they are all
returned so the person chooses.

:class:`Extractor` is the seam.  :class:`RegexExtractor` is the deterministic
implementation used today; an LLM extractor plugs in later by satisfying the
same protocol (see :func:`extraction_result_from_fields`, which converts the
validated field mapping from :mod:`src.llm.extraction` into an
:class:`ExtractionResult`), without the form or the confirm queue changing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from src.ingest.extract_common import (
    extract_amounts,
    extract_field_value,
    find_min_gpa,
    find_nc_counties,
    find_requirement_flags,
    find_states,
    html_to_text,
    meta_content,
    parse_date_candidates,
    tag_text,
)

logger = logging.getLogger(__name__)

_MAX_TEXT_CHARS = 20_000
_LABELED_DEADLINE_FIELDS = ["deadline", "due", "application deadline", "closes"]
_LABELED_AMOUNT_FIELDS = ["amount", "award", "award amount", "value"]
_LABELED_STATE_FIELDS = ["state/territory", "state", "territory", "residency"]


@dataclass(slots=True)
class ExtractionResult:
    """Structured fields recovered from one listing's text.

    Every field is ``None`` or empty unless the text stated it.  ``confidence``
    maps a field name to a 0-1 score for the fields that were populated; a
    field absent from ``confidence`` was not extracted.
    """

    deadline_candidates: list[str] = field(default_factory=list)
    amount_candidates: list[float] = field(default_factory=list)
    amount_min: float | None = None
    amount_max: float | None = None
    min_gpa: float | None = None
    states_allowed: list[str] | None = None
    counties_allowed: list[str] | None = None
    majors_allowed: list[str] | None = None
    education_level: str | None = None
    citizenship: str | None = None
    keywords: list[str] | None = None
    requirements: dict[str, Any] = field(default_factory=dict)
    requirement_evidence: dict[str, str] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)

    @property
    def deadline(self) -> str | None:
        """The single deadline, or ``None`` when the page offered zero or several."""
        if len(self.deadline_candidates) == 1:
            return self.deadline_candidates[0]
        return None


@runtime_checkable
class Extractor(Protocol):
    """Anything that can read structured fields out of a listing's text.

    Implementations must never raise: an extractor that cannot answer returns
    an empty :class:`ExtractionResult` so a prefill degrades to a blank form
    rather than failing.
    """

    name: str

    def extract(self, title: str | None, text: str | None) -> ExtractionResult:
        """Return the fields stated in ``text`` (``title`` is extra context)."""
        ...


class RegexExtractor:
    """Deterministic extractor built on :mod:`src.ingest.extract_common`.

    Confidence is a function of how the evidence was found, not of how likely
    the value looks: a value read out of an explicit ``Deadline:`` label scores
    higher than the same value found loose in the page body, and a field with
    several competing candidates scores lower than one with exactly one.
    """

    name = "regex"

    def extract(self, title: str | None, text: str | None) -> ExtractionResult:
        result = ExtractionResult()
        if not text:
            return result

        haystack = str(text)
        confidence: dict[str, float] = {}

        labeled_deadline = extract_field_value(haystack, _LABELED_DEADLINE_FIELDS)
        labeled_candidates = parse_date_candidates(labeled_deadline)
        page_candidates = parse_date_candidates(haystack)
        result.deadline_candidates = labeled_candidates or page_candidates
        if result.deadline_candidates:
            confidence["deadline"] = _candidate_confidence(
                count=len(result.deadline_candidates), labeled=bool(labeled_candidates)
            )

        labeled_amount = extract_field_value(haystack, _LABELED_AMOUNT_FIELDS)
        labeled_amounts = extract_amounts(labeled_amount)
        result.amount_candidates = labeled_amounts or extract_amounts(haystack)
        if result.amount_candidates:
            result.amount_min = min(result.amount_candidates)
            result.amount_max = max(result.amount_candidates)
            amount_confidence = _candidate_confidence(
                count=len(set(result.amount_candidates)), labeled=bool(labeled_amounts)
            )
            confidence["amount_min"] = amount_confidence
            confidence["amount_max"] = amount_confidence

        labeled_states = extract_field_value(haystack, _LABELED_STATE_FIELDS)
        result.states_allowed = find_states(labeled_states) if labeled_states else None
        if result.states_allowed:
            confidence["states_allowed"] = 0.7
        else:
            # A bare state name in body copy is as often the sponsor's address
            # as an eligibility rule, so it is offered at low confidence only.
            body_states = find_states(haystack)
            if body_states:
                result.states_allowed = body_states
                confidence["states_allowed"] = 0.35

        result.counties_allowed = find_nc_counties(haystack)
        if result.counties_allowed:
            confidence["counties_allowed"] = 0.6

        result.min_gpa = find_min_gpa(haystack)
        if result.min_gpa is not None:
            confidence["min_gpa"] = 0.6

        result.requirements, result.requirement_evidence = find_requirement_flags(haystack)
        for key in result.requirements:
            confidence[f"requirements.{key}"] = 0.6

        result.confidence = confidence
        return result


def extraction_result_from_fields(
    fields: dict[str, Any], *, confidence: float = 0.5
) -> ExtractionResult:
    """Build an :class:`ExtractionResult` from a validated LLM field mapping.

    This is what makes :mod:`src.llm.extraction` satisfy :class:`Extractor`:
    its ``parse_extraction`` output keys are exactly the fields below, and
    every key present there has already passed validation, so each one gets
    the same flat ``confidence``.
    """
    result = ExtractionResult()
    deadline = fields.get("deadline")
    if deadline:
        result.deadline_candidates = [str(deadline)]

    amounts = [
        float(fields[key])
        for key in ("amount_min", "amount_max")
        if fields.get(key) is not None
    ]
    if amounts:
        result.amount_candidates = sorted(set(amounts))
        result.amount_min = min(amounts)
        result.amount_max = max(amounts)

    result.min_gpa = fields.get("min_gpa")
    result.states_allowed = fields.get("states_allowed")
    result.majors_allowed = fields.get("majors_allowed")
    result.education_level = fields.get("education_level")
    result.citizenship = fields.get("citizenship")
    result.keywords = fields.get("keywords")
    essay_required = fields.get("essay_required")
    if essay_required is not None:
        result.requirements["essay"] = bool(essay_required)

    result.confidence = {key: confidence for key in fields if fields.get(key) is not None}
    return result


@dataclass(slots=True)
class PrefillResult:
    """Everything the hand-entry form needs from one URL.

    ``error`` is set when the page could not be fetched; the rest of the
    result is then empty and the form opens blank rather than the paste
    failing.
    """

    url: str
    title: str | None = None
    sponsor: str | None = None
    description: str | None = None
    text: str = ""
    extraction: ExtractionResult = field(default_factory=ExtractionResult)
    extractor: str = ""
    error: str | None = None

    @property
    def confidence(self) -> dict[str, float]:
        """Per-field confidence, including the page-level title/sponsor fields."""
        merged = dict(self.extraction.confidence)
        merged.update(self._page_confidence)
        return merged

    _page_confidence: dict[str, float] = field(default_factory=dict, repr=False)

    def to_form_dict(self) -> dict[str, Any]:
        """Flatten to the shape the entry form and the confirm queue consume."""
        return {
            "source_url": self.url,
            "title": self.title,
            "sponsor": self.sponsor,
            "description": self.description,
            "deadline_candidates": list(self.extraction.deadline_candidates),
            "deadline": self.extraction.deadline,
            "amount_candidates": list(self.extraction.amount_candidates),
            "amount_min": self.extraction.amount_min,
            "amount_max": self.extraction.amount_max,
            "min_gpa": self.extraction.min_gpa,
            "states_allowed": self.extraction.states_allowed,
            "counties_allowed": self.extraction.counties_allowed,
            "majors_allowed": self.extraction.majors_allowed,
            "education_level": self.extraction.education_level,
            "citizenship": self.extraction.citizenship,
            "keywords": self.extraction.keywords,
            "requirements": dict(self.extraction.requirements),
            "requirement_evidence": dict(self.extraction.requirement_evidence),
            "confidence": self.confidence,
            "extractor": self.extractor,
            "error": self.error,
        }


def prefill_from_html(
    html: str, *, url: str, extractor: Extractor | None = None
) -> PrefillResult:
    """Run the prefill pipeline over already-fetched HTML."""
    active = extractor or RegexExtractor()
    text = html_to_text(html)[:_MAX_TEXT_CHARS]

    page_confidence: dict[str, float] = {}
    title, title_confidence = _page_title(html)
    if title:
        page_confidence["title"] = title_confidence

    sponsor = meta_content(html, prop="og:site_name") or None
    if sponsor:
        page_confidence["sponsor"] = 0.7

    description = (
        meta_content(html, name="description")
        or meta_content(html, prop="og:description")
        or None
    )
    if description:
        page_confidence["description"] = 0.8

    return PrefillResult(
        url=url,
        title=title or None,
        sponsor=sponsor,
        description=description,
        text=text,
        extraction=active.extract(title or None, text),
        extractor=active.name,
        _page_confidence=page_confidence,
    )


def prefill_from_url(
    url: str, client: Any | None = None, *, extractor: Extractor | None = None
) -> PrefillResult:
    """Fetch ``url`` and return a prefilled form payload.

    A fetch failure is reported in ``PrefillResult.error`` rather than raised:
    a bad paste should open an empty form, never crash the app.

    Args:
        url: The scholarship page to read.
        client: Anything with ``get_text(url) -> str``; a
            :class:`~src.ingest.http.PoliteHttpClient` is built when omitted.
        extractor: The :class:`Extractor` to run; :class:`RegexExtractor` by default.
    """
    http_client = client
    owns_client = False
    if http_client is None:
        from src.ingest.http import PoliteHttpClient

        http_client = PoliteHttpClient()
        owns_client = True

    try:
        html = http_client.get_text(url)
    except Exception as exc:  # a bad paste must not propagate into the form
        logger.warning("prefill_from_url failed for %s: %s", url, exc)
        return PrefillResult(url=url, error=f"{type(exc).__name__}: {exc}")
    finally:
        if owns_client:
            close = getattr(http_client, "close", None)
            if callable(close):
                close()

    return prefill_from_html(html, url=url, extractor=extractor)


def _page_title(html: str) -> tuple[str, float]:
    og_title = meta_content(html, prop="og:title")
    if og_title:
        return og_title, 0.9
    h1 = tag_text(html, "h1")
    if h1:
        return h1, 0.75
    title = tag_text(html, "title")
    if title:
        return title, 0.6
    return "", 0.0


def _candidate_confidence(*, count: int, labeled: bool) -> float:
    base = 0.85 if labeled else 0.6
    if count == 1:
        return base
    return round(max(base - 0.25, 0.2), 2)
