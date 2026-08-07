"""Structured-field extraction from messy scholarship listing text.

The extraction contract has three parts: a versioned prompt, a strict-JSON
response, and a validator that coerces or rejects every field before it can
touch a record.  A field that fails validation is dropped, never guessed —
an invalid extraction degrades to "field stays empty".

``parse_extraction`` returns only the fields that produced a usable value, so
the result is directly consumable by the fill-only enrichment pass: every key
present is a value safe to write into an empty record field.

``EXTRACTION_PROMPT_VERSION`` is an input to the extraction cache key (Task 3);
bump it on any change to the prompt or to the validation semantics so stale
extractions invalidate.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Any

from src.llm.client import LlmClient
from src.text_utils import normalize_text

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT_VERSION = "v1"

EXTRACTION_FIELDS: tuple[str, ...] = (
    "deadline",
    "amount_min",
    "amount_max",
    "min_gpa",
    "states_allowed",
    "majors_allowed",
    "education_level",
    "citizenship",
    "essay_required",
    "keywords",
)

_MIN_PLAUSIBLE_YEAR = 2020
_MAX_PLAUSIBLE_YEAR = 2040
_MAX_GPA = 5.0
_MAX_KEYWORDS = 20
_MAX_TEXT_CHARS = 4000

EXTRACTION_SYSTEM_PROMPT = (
    "You extract structured eligibility data from scholarship listings. "
    "Respond with a single JSON object and nothing else — no prose, no code fences. "
    "The object must contain exactly these keys: "
    "deadline, amount_min, amount_max, min_gpa, states_allowed, majors_allowed, "
    "education_level, citizenship, essay_required, keywords.\n"
    "Field rules:\n"
    "- deadline: application deadline as an ISO date string (YYYY-MM-DD).\n"
    "- amount_min, amount_max: award amounts in US dollars as plain numbers "
    "(no currency symbols or commas). For a single fixed award, set both to the "
    "same number.\n"
    "- min_gpa: minimum required GPA as a number on a 4.0-style scale.\n"
    "- states_allowed: list of two-letter US state codes the applicant may reside "
    "in. Omit entirely (null) when the award is open nationwide.\n"
    "- majors_allowed: list of required fields of study.\n"
    "- education_level: exactly one of \"high school\", \"undergraduate\", "
    "\"graduate\".\n"
    "- citizenship: exactly one of \"us\", \"permanent resident\", "
    "\"international\".\n"
    "- essay_required: true or false.\n"
    "- keywords: list of up to 20 lowercase topical keywords describing the award.\n"
    "Use null when a field is not stated in the listing. Never guess, never infer "
    "from typical scholarship conventions, and never carry over an example value. "
    "Only report what the text states."
)


def build_user_prompt(
    *,
    title: str | None,
    description: str | None,
    eligibility_text: str | None,
) -> str:
    """Compose the user message for one scholarship listing."""
    return "\n".join(
        [
            "TITLE:",
            _clip(title),
            "",
            "DESCRIPTION:",
            _clip(description),
            "",
            "ELIGIBILITY:",
            _clip(eligibility_text),
        ]
    )


def parse_extraction(raw: str) -> dict[str, Any]:
    """Recover a JSON object from an LLM response and validate every field.

    Tolerates code fences and surrounding prose.  Unknown keys are discarded and
    any field failing validation is dropped, so the returned mapping contains
    only fields with a usable, pipeline-compatible value.

    Args:
        raw: The raw assistant message text.

    Returns:
        A dict keyed by a subset of :data:`EXTRACTION_FIELDS`.  Empty when the
        response held no recoverable JSON object or no field validated.
    """
    payload = _recover_json_object(raw)
    if payload is None:
        return {}

    validated: dict[str, Any] = {}
    for field_name, validator in _VALIDATORS.items():
        if field_name not in payload:
            continue
        value = validator(payload[field_name])
        if value is not None:
            validated[field_name] = value

    amount_min = validated.get("amount_min")
    amount_max = validated.get("amount_max")
    if amount_min is not None and amount_max is not None and amount_min > amount_max:
        validated.pop("amount_min")
        validated.pop("amount_max")

    return validated


def extract_fields(
    client: LlmClient,
    *,
    title: str | None,
    description: str | None,
    eligibility_text: str | None,
) -> dict[str, Any]:
    """Extract validated structured fields from one listing's text.

    Extraction is always best-effort: any client failure yields ``{}`` rather
    than propagating, so a flaky provider can never fail an ingest run.
    """
    try:
        raw = client.complete(
            EXTRACTION_SYSTEM_PROMPT,
            build_user_prompt(
                title=title,
                description=description,
                eligibility_text=eligibility_text,
            ),
        )
    except Exception as exc:  # best-effort: never fail ingest on a provider error
        logger.warning("LLM extraction failed for %r: %s", title, exc)
        return {}
    return parse_extraction(raw)


def _clip(text: str | None) -> str:
    if not text:
        return "(not provided)"
    cleaned = str(text).strip()
    if len(cleaned) <= _MAX_TEXT_CHARS:
        return cleaned
    return cleaned[:_MAX_TEXT_CHARS] + " ..."


_FENCE_PATTERN = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def _recover_json_object(raw: str) -> dict[str, Any] | None:
    if not isinstance(raw, str):
        return None
    cleaned = _FENCE_PATTERN.sub("", raw.strip()).strip()

    for candidate in (cleaned, _first_balanced_object(cleaned)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _first_balanced_object(text: str) -> str | None:
    """Return the first brace-balanced ``{...}`` span, ignoring braces in strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace("$", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _validate_deadline(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = date.fromisoformat(value.strip())
    except ValueError:
        return None
    if not _MIN_PLAUSIBLE_YEAR <= parsed.year <= _MAX_PLAUSIBLE_YEAR:
        return None
    return parsed.isoformat()


def _validate_amount(value: Any) -> float | None:
    amount = _coerce_float(value)
    if amount is None or amount < 0:
        return None
    return amount


def _validate_gpa(value: Any) -> float | None:
    gpa = _coerce_float(value)
    if gpa is None or not 0.0 <= gpa <= _MAX_GPA:
        return None
    return gpa


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str)]
    return []


def _validate_states(value: Any) -> list[str] | None:
    codes: list[str] = []
    for item in _as_string_list(value):
        code = _STATE_NAME_TO_CODE.get(normalize_text(item))
        if code and code not in codes:
            codes.append(code)
    return sorted(codes) or None


def _validate_majors(value: Any) -> list[str] | None:
    majors: list[str] = []
    for item in _as_string_list(value):
        major = normalize_text(item)
        if major and major not in majors:
            majors.append(major)
    return majors or None


def _validate_education_level(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return _EDUCATION_LEVEL_SYNONYMS.get(normalize_text(value))


def _validate_citizenship(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return _CITIZENSHIP_SYNONYMS.get(normalize_text(value))


def _validate_essay_required(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _BOOL_WORDS.get(normalize_text(value))
    return None


def _validate_keywords(value: Any) -> list[str] | None:
    keywords: list[str] = []
    for item in _as_string_list(value):
        keyword = normalize_text(item)
        if keyword and keyword not in keywords:
            keywords.append(keyword)
    return keywords[:_MAX_KEYWORDS] or None


_VALIDATORS: dict[str, Any] = {
    "deadline": _validate_deadline,
    "amount_min": _validate_amount,
    "amount_max": _validate_amount,
    "min_gpa": _validate_gpa,
    "states_allowed": _validate_states,
    "majors_allowed": _validate_majors,
    "education_level": _validate_education_level,
    "citizenship": _validate_citizenship,
    "essay_required": _validate_essay_required,
    "keywords": _validate_keywords,
}

_STATE_CODES: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "district of columbia": "DC", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI",
    "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "puerto rico": "PR", "guam": "GU", "american samoa": "AS",
    "u.s. virgin islands": "VI", "us virgin islands": "VI",
    "northern mariana islands": "MP",
}

# Accepts either a full state name or an already-correct code.
_STATE_NAME_TO_CODE: dict[str, str] = {
    **_STATE_CODES,
    **{code.lower(): code for code in _STATE_CODES.values()},
}

# Maps onto the vocabulary src.rank.taxonomy already reasons about.
_EDUCATION_LEVEL_SYNONYMS: dict[str, str] = {
    "high school": "high school",
    "highschool": "high school",
    "high-school": "high school",
    "secondary": "high school",
    "undergraduate": "undergraduate",
    "undergrad": "undergraduate",
    "college": "undergraduate",
    "bachelor": "undergraduate",
    "bachelors": "undergraduate",
    "bachelor's": "undergraduate",
    "graduate": "graduate",
    "postgraduate": "graduate",
    "master": "graduate",
    "masters": "graduate",
    "master's": "graduate",
    "doctoral": "graduate",
    "phd": "graduate",
}

# Stage 1 compares citizenship by exact normalized equality, so anything outside
# this vocabulary would over-filter and is dropped instead.
_CITIZENSHIP_SYNONYMS: dict[str, str] = {
    "us": "us",
    "u.s.": "us",
    "usa": "us",
    "citizen": "us",
    "us citizen": "us",
    "u.s. citizen": "us",
    "united states citizen": "us",
    "american": "us",
    "permanent resident": "permanent resident",
    "us permanent resident": "permanent resident",
    "green card": "permanent resident",
    "green card holder": "permanent resident",
    "international": "international",
    "international student": "international",
    "non-us": "international",
}

_BOOL_WORDS: dict[str, bool] = {
    "true": True, "yes": True, "y": True, "required": True,
    "false": False, "no": False, "n": False, "not required": False,
}
