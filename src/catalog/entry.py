"""Turn hand-entry form values into a curated catalog record.

The Add-Award page is the front door for data, and every field it offers is a
choice rather than free text wherever the catalog has a vocabulary: schema
enums (status, trust, gender, source kind), the grade sequence, the major
families Stage 1 matches on, the state and county lists the extractors use.
Typed spellings are how a catalog drifts out of reach of its own matcher.

Everything here is pure so the whole path -- prefill payload to form values to
validated record -- is testable without Streamlit.  The page draws widgets and
calls :func:`validate_form`; nothing else in the app builds a record.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.ingest.extract_common import NC_COUNTIES, US_STATES_AND_TERRITORIES
from src.normalize.catalog_schema import load_catalog_schema, validate_catalog_record
from src.profile.grade_levels import GRADE_SEQUENCE
from src.rank.taxonomy import known_majors

_SLUG_UNSAFE_PATTERN = re.compile(r"[^a-z0-9]+")
_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MAX_CATALOG_ID_CHARS = 120
_MIN_CATALOG_ID_CHARS = 3

MONTH_OPTIONS: tuple[int, ...] = tuple(range(1, 13))
GRADE_LEVEL_OPTIONS: tuple[str, ...] = GRADE_SEQUENCE
MAJOR_OPTIONS: tuple[str, ...] = known_majors()
STATE_OPTIONS: tuple[str, ...] = tuple(US_STATES_AND_TERRITORIES)
COUNTY_OPTIONS: tuple[str, ...] = tuple(NC_COUNTIES)

# Stage 1's level vocabulary (src.rank.taxonomy), not a schema enum: the
# catalog stores education_level as free text so a new level needs no migration.
EDUCATION_LEVEL_OPTIONS: tuple[str, ...] = ("high school", "undergraduate", "graduate")

# The requirement flags that are a plain yes/no. ``recommendation_letters`` is a
# count and ``essay_prompts`` a list, so both are asked for separately.
REQUIREMENT_FLAGS: tuple[str, ...] = (
    "essay",
    "transcript",
    "fafsa",
    "video_or_portfolio",
    "interview",
)

_TEXT_FIELDS: tuple[str, ...] = (
    "sponsor",
    "description",
    "eligibility_text",
    "citizenship",
    "religion",
    "renewal_terms",
    "notes",
)

_LIST_FIELDS: tuple[str, ...] = (
    "majors_allowed",
    "counties_allowed",
    "heritage",
    "employer_restricted",
    "membership_required",
    "keywords",
)

_TRISTATE_FIELDS: tuple[str, ...] = (
    "need_based",
    "first_gen_only",
    "military_family",
    "disability",
)

_TRUE_WORDS = frozenset({"yes", "true", "1"})
_FALSE_WORDS = frozenset({"no", "false", "0"})

# Full state name (and the code itself) -> the two-letter code the schema wants.
_STATE_NAME_TO_CODE: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "district of columbia": "DC", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "puerto rico": "PR",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI",
    "wyoming": "WY", "guam": "GU", "american samoa": "AS",
    "u.s. virgin islands": "VI", "northern mariana islands": "MP",
}
_STATE_CODE_TO_NAME: dict[str, str] = {
    code: name for name, code in _STATE_NAME_TO_CODE.items()
}


@lru_cache(maxsize=8)
def enum_options(field_path: str, schema_path: Path | None = None) -> tuple[str, ...]:
    """Return a schema enum as dropdown options, dropping the null member.

    ``field_path`` is dotted (``"provenance.source_kind"``).  Reading the
    options out of ``schema.json`` rather than restating them is what keeps the
    form from offering a value the catalog would reject.
    """
    node: Any = load_catalog_schema(schema_path)
    for part in field_path.split("."):
        node = (node.get("properties") or {}).get(part) or {}
    values = node.get("enum")
    if not isinstance(values, list):
        return ()
    return tuple(str(value) for value in values if value is not None)


def status_options() -> tuple[str, ...]:
    return enum_options("status")


def trust_options() -> tuple[str, ...]:
    return enum_options("trust")


def gender_options() -> tuple[str, ...]:
    return enum_options("gender")


def source_kind_options() -> tuple[str, ...]:
    return enum_options("provenance.source_kind")


def state_codes(names: Iterable[str] | None) -> list[str]:
    """Map full state names (or codes already) to sorted two-letter codes."""
    codes: list[str] = []
    for item in _as_str_list(names):
        code = _STATE_NAME_TO_CODE.get(item.casefold())
        if code is None and item.upper() in _STATE_CODE_TO_NAME:
            code = item.upper()
        if code and code not in codes:
            codes.append(code)
    return sorted(codes)


def state_names(codes: Iterable[str] | None) -> list[str]:
    """Inverse of :func:`state_codes`, for loading a record back into the form."""
    lookup = {option.casefold(): option for option in STATE_OPTIONS}
    names: list[str] = []
    for item in _as_str_list(codes):
        name = _STATE_CODE_TO_NAME.get(item.upper()) or item
        option = lookup.get(name.casefold())
        if option and option not in names:
            names.append(option)
    return sorted(names)


def normalize_majors(value: Any) -> list[str]:
    """Canonicalize majors to the taxonomy's spelling, keeping ones it does not know.

    A record written before a major entered the taxonomy still matches through
    :func:`~src.rank.taxonomy.majors_match`, so loading it into the form must
    not drop the spelling it has.
    """
    lookup = {option.casefold(): option for option in MAJOR_OPTIONS}
    majors: list[str] = []
    for item in _as_str_list(value):
        major = lookup.get(item.casefold(), item)
        if major not in majors:
            majors.append(major)
    return majors


def catalog_id_from_title(title: str | None, sponsor: str | None = None) -> str:
    """Suggest a stable slug for a new award; empty when there is nothing to slug.

    The sponsor is only folded in when the title alone is too short to be a
    valid ``catalog_id`` -- two foundations can both run a "STEM Award", so the
    form always lets a person edit the suggestion.
    """
    slug = _slugify(title)
    if len(slug) < _MIN_CATALOG_ID_CHARS:
        slug = _slugify(f"{sponsor or ''} {title or ''}")
    return slug[:_MAX_CATALOG_ID_CHARS].strip("-")


def blank_form() -> dict[str, Any]:
    """Every form key with an empty default, so manual entry needs no URL."""
    values: dict[str, Any] = {
        "catalog_id": "",
        "title": "",
        "source_url": "",
        "amount_min": None,
        "amount_max": None,
        "amount_candidates": [],
        "deadline": "",
        "deadline_candidates": [],
        "cycle_recurring": None,
        "cycle_opens_month": None,
        "cycle_deadline_month": None,
        "status": "unknown",
        "education_level": "",
        "grade_levels": [],
        "states_allowed": [],
        "min_gpa": None,
        "gender": "",
        "sat": None,
        "act": None,
        "recommendation_letters": None,
        "essay_prompts": [],
        "trust": "unverified",
        "added_on": "",
        "verified_on": "",
        "verified_by": "",
        "checked_on": "",
        "checked_by": "",
        "source_kind": "sponsor_site",
        "confidence": {},
        "requirement_evidence": {},
    }
    values.update({key: "" for key in _TEXT_FIELDS})
    values.update({key: [] for key in _LIST_FIELDS})
    values.update({key: None for key in _TRISTATE_FIELDS})
    values.update({f"req_{flag}": None for flag in REQUIREMENT_FLAGS})
    return values


def form_from_prefill(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seed form values from ``PrefillResult.to_form_dict()``.

    Only what the page stated survives: an ambiguous deadline or amount stays
    in the candidate lists for the person to choose from rather than being
    filled in, and a vocabulary hit the dropdown does not offer is dropped
    instead of being typed into the catalog.
    """
    values = blank_form()
    values["source_url"] = _text(payload.get("source_url"))
    values["title"] = _text(payload.get("title"))
    values["sponsor"] = _text(payload.get("sponsor"))
    values["description"] = _text(payload.get("description"))
    values["catalog_id"] = catalog_id_from_title(values["title"], values["sponsor"])

    values["deadline_candidates"] = [
        item for item in _as_str_list(payload.get("deadline_candidates")) if _is_iso_date(item)
    ]
    deadline = _text(payload.get("deadline"))
    values["deadline"] = deadline if _is_iso_date(deadline) else ""

    values["amount_candidates"] = _as_float_list(payload.get("amount_candidates"))
    values["amount_min"] = _float_or_none(payload.get("amount_min"))
    values["amount_max"] = _float_or_none(payload.get("amount_max"))

    values["min_gpa"] = _float_or_none(payload.get("min_gpa"))
    values["states_allowed"] = _only_known(payload.get("states_allowed"), STATE_OPTIONS)
    values["counties_allowed"] = _only_known(payload.get("counties_allowed"), COUNTY_OPTIONS)
    values["majors_allowed"] = _only_known(payload.get("majors_allowed"), MAJOR_OPTIONS)
    level = _text(payload.get("education_level")).casefold()
    values["education_level"] = level if level in EDUCATION_LEVEL_OPTIONS else ""
    values["citizenship"] = _text(payload.get("citizenship"))
    values["keywords"] = _as_str_list(payload.get("keywords"))

    requirements = payload.get("requirements")
    if isinstance(requirements, Mapping):
        for flag in REQUIREMENT_FLAGS:
            values[f"req_{flag}"] = _tristate(requirements.get(flag))
        values["recommendation_letters"] = _int_or_none(
            requirements.get("recommendation_letters")
        )
        values["essay_prompts"] = _as_str_list(requirements.get("essay_prompts"))

    confidence = payload.get("confidence")
    values["confidence"] = dict(confidence) if isinstance(confidence, Mapping) else {}
    evidence = payload.get("requirement_evidence")
    values["requirement_evidence"] = dict(evidence) if isinstance(evidence, Mapping) else {}
    return values


def form_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Load an existing (or proposed) record back into form values for editing."""
    values = blank_form()
    for key in ("catalog_id", "title", "source_url", *_TEXT_FIELDS):
        values[key] = _text(record.get(key))
    values["amount_min"] = _float_or_none(record.get("amount_min"))
    values["amount_max"] = _float_or_none(record.get("amount_max"))
    deadline = _text(record.get("deadline"))
    values["deadline"] = deadline if _is_iso_date(deadline) else ""
    values["min_gpa"] = _float_or_none(record.get("min_gpa"))
    values["status"] = _one_of(record.get("status"), status_options(), "unknown")
    values["trust"] = _one_of(record.get("trust"), trust_options(), "unverified")
    values["gender"] = _one_of(record.get("gender"), gender_options(), "")
    level = _text(record.get("education_level")).casefold()
    values["education_level"] = level if level in EDUCATION_LEVEL_OPTIONS else ""
    values["grade_levels"] = _only_known(record.get("grade_levels"), GRADE_LEVEL_OPTIONS)
    values["states_allowed"] = state_names(record.get("states_allowed"))
    for key in _LIST_FIELDS:
        values[key] = _as_str_list(record.get(key))
    values["majors_allowed"] = normalize_majors(record.get("majors_allowed"))
    for key in _TRISTATE_FIELDS:
        values[key] = _tristate(record.get(key))

    cycle = record.get("cycle")
    if isinstance(cycle, Mapping):
        values["cycle_recurring"] = _tristate(cycle.get("recurring"))
        values["cycle_opens_month"] = _month_or_none(cycle.get("opens_month"))
        values["cycle_deadline_month"] = _month_or_none(cycle.get("deadline_month"))

    scores = record.get("min_test_scores")
    if isinstance(scores, Mapping):
        values["sat"] = _int_or_none(scores.get("sat"))
        values["act"] = _int_or_none(scores.get("act"))

    requirements = record.get("requirements")
    if isinstance(requirements, Mapping):
        for flag in REQUIREMENT_FLAGS:
            values[f"req_{flag}"] = _tristate(requirements.get(flag))
        values["recommendation_letters"] = _int_or_none(
            requirements.get("recommendation_letters")
        )
        values["essay_prompts"] = _as_str_list(requirements.get("essay_prompts"))

    provenance = record.get("provenance")
    if isinstance(provenance, Mapping):
        values["added_on"] = _text(provenance.get("added_on"))
        values["verified_on"] = _text(provenance.get("verified_on"))
        values["verified_by"] = _text(provenance.get("verified_by"))
        values["checked_on"] = _text(provenance.get("checked_on"))
        values["checked_by"] = _text(provenance.get("checked_by"))
        values["source_kind"] = _one_of(
            provenance.get("source_kind"), source_kind_options(), "other"
        )
    return values


def record_from_form(values: Mapping[str, Any], *, today: date | None = None) -> dict[str, Any]:
    """Build a catalog record from form values.

    Nothing is invented: a field left empty becomes ``null`` or an empty list,
    never a guess.  The result is deliberately not validated here --
    :func:`validate_form` does that -- so a half-filled form still produces a
    record the page can show errors against.
    """
    stamp = (today or date.today()).isoformat()
    title = _text(values.get("title"))
    catalog_id = _slugify(values.get("catalog_id")) or catalog_id_from_title(
        title, _text(values.get("sponsor"))
    )

    record: dict[str, Any] = {
        "catalog_id": catalog_id[:_MAX_CATALOG_ID_CHARS].strip("-"),
        "title": title,
        "sponsor": _text_or_none(values.get("sponsor")),
        "source_url": _text(values.get("source_url")),
        "description": _text_or_none(values.get("description")),
        "eligibility_text": _text_or_none(values.get("eligibility_text")),
        "amount_min": _float_or_none(values.get("amount_min")),
        "amount_max": _float_or_none(values.get("amount_max")),
        "deadline": _iso_or_none(values.get("deadline")),
        "cycle": {
            "recurring": _tristate(values.get("cycle_recurring")),
            "opens_month": _month_or_none(values.get("cycle_opens_month")),
            "deadline_month": _month_or_none(values.get("cycle_deadline_month")),
        },
        "status": _one_of(values.get("status"), status_options(), "unknown"),
        "education_level": _text_or_none(values.get("education_level")),
        "grade_levels": _only_known(values.get("grade_levels"), GRADE_LEVEL_OPTIONS),
        "majors_allowed": _as_str_list(values.get("majors_allowed")),
        "states_allowed": state_codes(values.get("states_allowed")),
        "counties_allowed": _as_str_list(values.get("counties_allowed")),
        "min_gpa": _float_or_none(values.get("min_gpa")),
        "citizenship": _text_or_none(values.get("citizenship")),
        "need_based": _tristate(values.get("need_based")),
        "first_gen_only": _tristate(values.get("first_gen_only")),
        "gender": _one_of(values.get("gender"), gender_options(), "") or None,
        "heritage": _as_str_list(values.get("heritage")),
        "military_family": _tristate(values.get("military_family")),
        "disability": _tristate(values.get("disability")),
        "religion": _text_or_none(values.get("religion")),
        "employer_restricted": _as_str_list(values.get("employer_restricted")),
        "membership_required": _as_str_list(values.get("membership_required")),
        "min_test_scores": {
            "sat": _int_or_none(values.get("sat")),
            "act": _int_or_none(values.get("act")),
        },
        "requirements": {
            "essay_prompts": _as_str_list(values.get("essay_prompts")),
            "recommendation_letters": _int_or_none(values.get("recommendation_letters")),
        },
        "renewal_terms": _text_or_none(values.get("renewal_terms")),
        "keywords": _as_str_list(values.get("keywords")),
        "trust": _one_of(values.get("trust"), trust_options(), "unverified"),
        "provenance": {
            "added_on": _iso_or_none(values.get("added_on")) or stamp,
            "verified_on": _iso_or_none(values.get("verified_on")),
            "verified_by": _text_or_none(values.get("verified_by")),
            "checked_on": _iso_or_none(values.get("checked_on")),
            "checked_by": _text_or_none(values.get("checked_by")),
            "source_kind": _one_of(values.get("source_kind"), source_kind_options(), "other"),
        },
        "notes": _text_or_none(values.get("notes")),
    }
    for flag in REQUIREMENT_FLAGS:
        record["requirements"][flag] = _tristate(values.get(f"req_{flag}"))
    return record


def validate_form(
    values: Mapping[str, Any],
    *,
    today: date | None = None,
    schema_path: Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Return ``(record, errors)`` for form values; no errors means valid."""
    record = record_from_form(values, today=today)
    errors = validate_catalog_record(record, schema=load_catalog_schema(schema_path))
    return record, errors


@dataclass(frozen=True, slots=True)
class DiffRow:
    """One field of a proposal's diff, rendered for display."""

    field: str
    old: str
    new: str


def diff_rows(diff: Mapping[str, Any] | None) -> list[DiffRow]:
    """Render a proposal ``diff`` as sorted display rows.

    An entry that is not the ``{"old": ..., "new": ...}`` shape is shown as a
    new value with no old one rather than dropped: a malformed proposal still
    has to be reviewable by the person deciding about it.
    """
    if not isinstance(diff, Mapping):
        return []
    rows: list[DiffRow] = []
    for field_name in sorted(diff):
        change = diff[field_name]
        if isinstance(change, Mapping) and ("old" in change or "new" in change):
            rows.append(
                DiffRow(
                    field=str(field_name),
                    old=render_value(change.get("old")),
                    new=render_value(change.get("new")),
                )
            )
        else:
            rows.append(DiffRow(field=str(field_name), old="—", new=render_value(change)))
    return rows


def render_value(value: Any) -> str:
    """Render a record field for a diff cell: readable, never a ``repr``."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, Mapping):
        return json.dumps(dict(value), sort_keys=True, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        items = [render_value(item) for item in value]
        return ", ".join(items) if items else "—"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return _text(value) or "—"


def _slugify(value: Any) -> str:
    return _SLUG_UNSAFE_PATTERN.sub("-", str(value or "").strip().casefold()).strip("-")


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _text_or_none(value: Any) -> str | None:
    return _text(value) or None


def _is_iso_date(value: Any) -> bool:
    return bool(_ISO_DATE_PATTERN.match(_text(value)))


def _iso_or_none(value: Any) -> str | None:
    if isinstance(value, date):
        return value.isoformat()
    text = _text(value)
    return text if _is_iso_date(text) else None


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _month_or_none(value: Any) -> int | None:
    month = _int_or_none(value)
    return month if month in MONTH_OPTIONS else None


def _tristate(value: Any) -> bool | None:
    """Coerce a yes/no/unknown widget value; anything unrecognized is unknown."""
    if isinstance(value, bool):
        return value
    text = _text(value).casefold()
    if text in _TRUE_WORDS:
        return True
    if text in _FALSE_WORDS:
        return False
    return None


def _one_of(value: Any, options: Sequence[str], fallback: str) -> str:
    text = _text(value).casefold()
    for option in options:
        if option.casefold() == text:
            return option
    return fallback


def _as_str_list(value: Any) -> list[str]:
    """Accept a list of values or a comma-separated string; drop the blanks."""
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return [item.strip() for item in _text(value).split(",") if item.strip()]
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _as_float_list(value: Any) -> list[float]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return []
    amounts = [_float_or_none(item) for item in value]
    return [amount for amount in amounts if amount is not None]


def _only_known(value: Any, options: Sequence[str]) -> list[str]:
    """Keep only values the dropdown offers, matched case-insensitively."""
    lookup = {option.casefold(): option for option in options}
    kept: list[str] = []
    for item in _as_str_list(value):
        option = lookup.get(item.casefold())
        if option and option not in kept:
            kept.append(option)
    return kept
