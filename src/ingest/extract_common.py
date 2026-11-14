"""Shared deterministic extraction helpers for scholarship listing text.

Every connector that reads a human-facing page needs the same primitives: turn
HTML into text, find dollar amounts, find dates in the three formats listings
actually use, and recognise US states, NC counties, and application
requirements.  Those helpers used to live privately in two scrapers; they live
here now so the URL-prefill path (:mod:`src.ingest.prefill`) and any future
connector share one implementation.

Contract for every ``find_*`` / ``extract_*`` function: no evidence means
``None`` (or an empty list).  Nothing here infers a value from what a typical
scholarship would say.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from html import unescape
from typing import Any

TAG_PATTERN = re.compile(r"<[^>]+>")
WS_PATTERN = re.compile(r"\s+")
SCRIPT_STYLE_PATTERN = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1>", flags=re.IGNORECASE | re.DOTALL
)
MONEY_PATTERN = re.compile(r"\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")
ISO_DATE_PATTERN = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
US_DATE_PATTERN = re.compile(r"\b(\d{1,2}/\d{1,2}/20\d{2})\b")
_MONTH_NAMES = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)"
)
LONG_DATE_PATTERN = re.compile(
    rf"\b({_MONTH_NAMES}\s+\d{{1,2}},?\s+20\d{{2}})\b",
    flags=re.IGNORECASE,
)
YEARLESS_DATE_PATTERN = re.compile(
    rf"\b({_MONTH_NAMES})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?!,?\s*20\d{{2}})(?![:\d])",
    flags=re.IGNORECASE,
)
NTH_WEEKDAY_PATTERN = re.compile(
    r"\b(first|second|third|fourth|last)\s+"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(?:in|of)\s+"
    rf"({_MONTH_NAMES})\b",
    flags=re.IGNORECASE,
)
# A deadline label counts only when it sits right before the date, give or
# take a colon, "Due", a time of day and a weekday ("Deadline: 5 p.m. ET,
# Friday, March 13").  Read loosely -- the date anywhere after the label --
# it picks up the next date on the page, which is often a different event.
_DATE_LEAD_IN = (
    r"\s*(?:date\s*)?(?:is\s*)?[:\-–—]?\s*(?:(?:due|on|by|the)\s+)*"
    r"(?:\d{1,2}(?::\d{2})?\s*[ap]\.?m\.?\s*(?:[ECMP][SD]?T\s*)?,?\s*)?"
    r"(?:(?:mon|tues|wednes|thurs|fri|satur|sun)day,?\s*)?$"
)
DEADLINE_LEAD_PATTERN = re.compile(rf"\bdeadlines?{_DATE_LEAD_IN}", flags=re.IGNORECASE)
# "Due" and "no later than" date the transcript mailing, the immunization
# form and the payment as often as the application, so they count only in a
# sentence about the deadline or the application, or as a "Due:" label.
WEAK_DEADLINE_LEAD_PATTERN = re.compile(
    rf"\b(?:due|no later than|closes? on){_DATE_LEAD_IN}", flags=re.IGNORECASE
)
_DEADLINE_SENTENCE_PATTERN = re.compile(r"\b(?:deadlines?|applications?)\b", re.IGNORECASE)
OPENING_LEAD_PATTERN = re.compile(
    rf"\b(?:opens?|opening|will open|available|begins?|starts?|starting){_DATE_LEAD_IN}",
    flags=re.IGNORECASE,
)
_LEAD_WINDOW_CHARS = 80
_SENTENCE_CAP_CHARS = 100
LABEL_BLOCK_PATTERN = re.compile(
    r"(Sponsor|Provider|Organization|Eligibility|Deadline|Amount|Award|Education|"
    r"Institution|Status|State|Territory|County|Essay(?: Prompt)?|Essay Required)\s*:?\s*",
    re.IGNORECASE,
)
GPA_PATTERN = re.compile(
    r"\b(?:gpa|grade point average)\b[^.\n]{0,40}?(\d(?:\.\d{1,2})?)"
    r"|(\d\.\d{1,2})\s*(?:cumulative\s+)?(?:gpa|grade point average)",
    flags=re.IGNORECASE,
)

US_STATES_AND_TERRITORIES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois",
    "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
    "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
    "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming", "Guam", "U.S. Virgin Islands", "Northern Mariana Islands",
    "American Samoa",
]

NC_COUNTIES = [
    "Alamance", "Alexander", "Alleghany", "Anson", "Ashe", "Avery", "Beaufort", "Bertie",
    "Bladen", "Brunswick", "Buncombe", "Burke", "Cabarrus", "Caldwell", "Camden", "Carteret",
    "Caswell", "Catawba", "Chatham", "Cherokee", "Chowan", "Clay", "Cleveland", "Columbus",
    "Craven", "Cumberland", "Currituck", "Dare", "Davidson", "Davie", "Duplin", "Durham",
    "Edgecombe", "Forsyth", "Franklin", "Gaston", "Gates", "Graham", "Granville", "Greene",
    "Guilford", "Halifax", "Harnett", "Haywood", "Henderson", "Hertford", "Hoke", "Hyde",
    "Iredell", "Jackson", "Johnston", "Jones", "Lee", "Lenoir", "Lincoln", "Macon", "Madison",
    "Martin", "McDowell", "Mecklenburg", "Mitchell", "Montgomery", "Moore", "Nash",
    "New Hanover", "Northampton", "Onslow", "Orange", "Pamlico", "Pasquotank", "Pender",
    "Perquimans", "Person", "Pitt", "Polk", "Randolph", "Richmond", "Robeson", "Rockingham",
    "Rowan", "Rutherford", "Sampson", "Scotland", "Stanly", "Stokes", "Surry", "Swain",
    "Transylvania", "Tyrrell", "Union", "Vance", "Wake", "Warren", "Washington", "Watauga",
    "Wayne", "Wilkes", "Wilson", "Yadkin", "Yancey",
]

# Requirement keys match ``requirements`` in data/catalog/schema.json so a
# confirmed prefill can be written straight into a catalog record.
_REQUIREMENT_PATTERNS: dict[str, re.Pattern[str]] = {
    "essay": re.compile(
        r"\b(essays?|personal statement|writing sample|written response)\b", re.IGNORECASE
    ),
    "recommendation_letters": re.compile(
        r"\b(letters? of recommendation|recommendation letters?|letters? of reference)\b",
        re.IGNORECASE,
    ),
    "transcript": re.compile(r"\btranscripts?\b", re.IGNORECASE),
    "fafsa": re.compile(
        r"\b(fafsa|free application for federal student aid)\b", re.IGNORECASE
    ),
    "video_or_portfolio": re.compile(
        r"\b(video (?:submission|essay|entry)|portfolio|demo reel)\b", re.IGNORECASE
    ),
    "interview": re.compile(r"\binterviews?\b", re.IGNORECASE),
}

_LETTER_COUNT_PATTERN = re.compile(
    r"\b(one|two|three|four|five|six|\d{1,2})\s+(?:letters?|references?)\b", re.IGNORECASE
)
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}

_NEGATION_PATTERN = re.compile(r"\b(no|not|without)\b", re.IGNORECASE)
_MAX_REQUIREMENT_EVIDENCE_CHARS = 160

_MONTH_PREFIXES = (
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
)
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "last": -1}


@dataclass(frozen=True, slots=True)
class YearlessDate:
    """A date a page states without a year: "March 1", "first Monday in February".

    Which year it means depends on when the page is read and on the award's
    usual cycle, so the caller resolves it with :meth:`on`.
    """

    month: int
    day: int | None = None
    weekday: int | None = None
    nth: int | None = None
    text: str = field(default="", compare=False)

    def on(self, year: int) -> date:
        """The date this falls on in ``year`` (Feb 29 clamps to Feb 28)."""
        last_day = calendar.monthrange(year, self.month)[1]
        if self.weekday is None or self.nth is None:
            return date(year, self.month, min(self.day or 1, last_day))
        if self.nth < 0:
            last = date(year, self.month, last_day)
            return date(year, self.month, last_day - (last.weekday() - self.weekday) % 7)
        first = date(year, self.month, 1)
        offset = (self.weekday - first.weekday()) % 7
        return date(year, self.month, 1 + offset + 7 * (self.nth - 1))


@dataclass(slots=True)
class DeadlineDates:
    """The dates on a page, sorted by what sits in front of them.

    ``labeled`` dates follow a deadline label; ``yearless`` are labeled dates
    with no year; ``other`` dates have no label at all.  A date that follows
    an opening phrase ("will open on", "Open date:") is in none of them.
    """

    labeled: list[str] = field(default_factory=list)
    yearless: list[YearlessDate] = field(default_factory=list)
    other: list[str] = field(default_factory=list)


def strip_html(html: str) -> str:
    """Collapse HTML to a single line of text, leaving script/style content in place."""
    return WS_PATTERN.sub(" ", unescape(TAG_PATTERN.sub(" ", html))).strip()


def html_to_text(html: str) -> str:
    """Collapse a full page to text, dropping ``<script>`` and ``<style>`` bodies first."""
    without_script = SCRIPT_STYLE_PATTERN.sub(" ", html)
    return WS_PATTERN.sub(" ", unescape(TAG_PATTERN.sub(" ", without_script))).strip()


def meta_content(html: str, *, prop: str | None = None, name: str | None = None) -> str:
    """Return the ``content`` of the first matching ``<meta>`` tag, or ``""``."""
    attr, value = ("property", prop) if prop is not None else ("name", name)
    if not value:
        return ""
    forward = re.search(
        rf"<meta[^>]*{attr}=['\"]{re.escape(value)}['\"][^>]*content=['\"](.*?)['\"][^>]*>",
        html,
        flags=re.IGNORECASE,
    )
    if forward:
        return WS_PATTERN.sub(" ", unescape(forward.group(1))).strip()

    # Some templates emit content= before the property/name attribute.
    reversed_match = re.search(
        rf"<meta[^>]*content=['\"](.*?)['\"][^>]*{attr}=['\"]{re.escape(value)}['\"][^>]*>",
        html,
        flags=re.IGNORECASE,
    )
    if reversed_match:
        return WS_PATTERN.sub(" ", unescape(reversed_match.group(1))).strip()
    return ""


def tag_text(html: str, tag: str) -> str:
    """Return the text of the first ``<tag>...</tag>`` block, or ``""``."""
    match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return WS_PATTERN.sub(" ", html_to_text(match.group(1))).strip()


def extract_amounts(text: str | None) -> list[float]:
    """Return every ``$``-prefixed amount in ``text``, in order of appearance."""
    if not text:
        return []
    return [float(chunk.replace(",", "")) for chunk in MONEY_PATTERN.findall(str(text))]


def extract_amount_range(text: str | None) -> tuple[float | None, float | None]:
    """Return ``(min, max)`` of the amounts in ``text``, or ``(None, None)``."""
    matches = extract_amounts(text)
    if not matches:
        return None, None
    return min(matches), max(matches)


def parse_date_candidates(text: str | None) -> list[str]:
    """Return every parseable date in ``text`` as ISO strings, deduped, in page order.

    All three formats listings use are scanned -- ``2026-03-01``, ``3/1/2026``,
    and ``March 1, 2026`` (the comma is optional) -- and merged by position so
    the caller can offer a choice rather than a guess.
    """
    if not text:
        return []
    ordered = [value for _, value in _dated_mentions(str(text))]
    return list(dict.fromkeys(ordered))


def find_deadline_dates(text: str | None) -> DeadlineDates:
    """Sort the dates in ``text`` by the words right in front of them.

    A labeled date outranks every other date on the page, so a sidebar post
    date or an info-session date cannot crowd out ``Deadline March 2``.  A
    date announcing when applications open is dropped outright: it is never
    the deadline.  Year-less dates are kept only under a deadline label;
    loose in body copy they are as often an event as a cutoff.  Each list is
    deduped, in page order.
    """
    found = DeadlineDates()
    if not text:
        return found
    haystack = str(text)

    for start, value in _dated_mentions(haystack):
        lead = _lead_kind(haystack, start)
        if lead == "deadline":
            found.labeled.append(value)
        elif lead is None:
            found.other.append(value)

    yearless: list[tuple[int, YearlessDate]] = []
    for day_match in YEARLESS_DATE_PATTERN.finditer(haystack):
        month = _month_number(day_match.group(1))
        day = int(day_match.group(2))
        if month and _is_valid_month_day(month, day):
            yearless.append(
                (day_match.start(), YearlessDate(month=month, day=day, text=day_match.group(0)))
            )
    for nth_match in NTH_WEEKDAY_PATTERN.finditer(haystack):
        month = _month_number(nth_match.group(3))
        if month:
            yearless.append(
                (
                    nth_match.start(),
                    YearlessDate(
                        month=month,
                        weekday=_WEEKDAYS.index(nth_match.group(2).lower()),
                        nth=_ORDINALS[nth_match.group(1).lower()],
                        text=nth_match.group(0),
                    ),
                )
            )
    for start, mention in sorted(yearless, key=lambda pair: pair[0]):
        if _lead_kind(haystack, start) == "deadline":
            found.yearless.append(mention)

    found.labeled = list(dict.fromkeys(found.labeled))
    found.yearless = list(dict.fromkeys(found.yearless))
    found.other = list(dict.fromkeys(found.other))
    return found


def parse_first_date(text: str | None) -> str | None:
    """Return the first parseable date in ``text``, preferring the least ambiguous format.

    ISO beats ``m/d/Y`` beats long form, regardless of position: an ISO date on
    a page is almost always machine-written, while a long-form date is often
    prose ("applications opened January 3, 2026").
    """
    if not text:
        return None
    haystack = str(text)

    iso = ISO_DATE_PATTERN.search(haystack)
    if iso and _is_valid_iso(iso.group(1)):
        return iso.group(1)

    us_date = US_DATE_PATTERN.search(haystack)
    if us_date:
        parsed = _parse_us_date(us_date.group(1))
        if parsed:
            return parsed

    long_date = LONG_DATE_PATTERN.search(haystack)
    if long_date:
        return _parse_long_date(long_date.group(1))
    return None


def extract_field_value(text: str, field_names: list[str]) -> str:
    """Return the value following the first ``<label>:`` found, or ``""``.

    The value ends where the next known label begins, so a run-together block
    like ``Deadline: March 1, 2026 Amount: $500`` yields just the date.
    """
    lowered = text.lower()
    for field_name in field_names:
        key = f"{field_name.lower()}:"
        start = lowered.find(key)
        if start == -1:
            continue
        raw = text[start + len(key) :]
        split = LABEL_BLOCK_PATTERN.split(raw, maxsplit=1)
        candidate = split[0].strip(" .;:\n\t")
        if candidate:
            return candidate
    return ""


def find_states(text: str | None) -> list[str] | None:
    """Return the US states and territories named in ``text``, sorted, or ``None``.

    Full names only.  Two-letter codes are not scanned because ``IN``, ``OR``,
    ``ME`` and ``OK`` are ordinary English words and would fire on every page.
    """
    if not text:
        return None
    lowered = str(text).lower()
    found = [
        name
        for name in US_STATES_AND_TERRITORIES
        if re.search(r"\b" + re.escape(name.lower()) + r"\b", lowered)
    ]
    return sorted(set(found)) if found else None


def find_nc_counties(text: str | None) -> list[str] | None:
    """Return the NC counties named in ``text``, sorted, or ``None``.

    The literal word "County" is required ("Wake County", not a bare "Wake"):
    a third of NC county names are also common nouns, surnames, or other
    states, so a bare-name scan is mostly false positives.  Names come back
    without the suffix, matching how ``counties_allowed`` is compared in
    Stage 1.
    """
    if not text:
        return None
    lowered = str(text).lower()
    found = [
        name
        for name in NC_COUNTIES
        if re.search(r"\b" + re.escape(name.lower()) + r"\s+count(?:y|ies)\b", lowered)
    ]
    return sorted(set(found)) if found else None


def find_min_gpa(text: str | None) -> float | None:
    """Return a stated minimum GPA on a 0-5 scale, or ``None`` when unstated."""
    if not text:
        return None
    for match in GPA_PATTERN.finditer(str(text)):
        raw = match.group(1) or match.group(2)
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if 0.0 < value <= 5.0:
            return value
    return None


def find_requirement_flags(text: str | None) -> tuple[dict[str, Any], dict[str, str]]:
    """Return ``(requirements, evidence)`` for application requirements named in ``text``.

    ``requirements`` is shaped like the catalog's ``requirements`` object and
    only carries keys with positive evidence -- an absent key means "the page
    did not say", never ``False``.  ``recommendation_letters`` is an integer
    only when the page states a count; when letters are mentioned without one
    the key is omitted and the mention is reported in ``evidence`` instead, so
    a person fills in the number.

    ``evidence`` maps every detected key to the sentence it matched, for the
    form to show beside the field.
    """
    flags: dict[str, Any] = {}
    evidence: dict[str, str] = {}
    if not text:
        return flags, evidence

    haystack = str(text)
    for key, pattern in _REQUIREMENT_PATTERNS.items():
        match = pattern.search(haystack)
        if not match:
            continue
        start = _sentence_start(haystack, match.start())
        if _NEGATION_PATTERN.search(haystack[start : match.start()]):
            continue
        evidence[key] = _sentence_around(haystack, match.start())
        if key == "recommendation_letters":
            count = _letter_count(evidence[key]) or _letter_count(haystack)
            if count is not None:
                flags[key] = count
            continue
        flags[key] = True
    return flags, evidence


def _dated_mentions(haystack: str) -> list[tuple[int, str]]:
    """Every full date in ``haystack`` as ``(position, iso)``, in page order."""
    found: list[tuple[int, str]] = []
    for iso_match in ISO_DATE_PATTERN.finditer(haystack):
        if _is_valid_iso(iso_match.group(1)):
            found.append((iso_match.start(), iso_match.group(1)))
    for us_match in US_DATE_PATTERN.finditer(haystack):
        parsed_us = _parse_us_date(us_match.group(1))
        if parsed_us:
            found.append((us_match.start(), parsed_us))
    for long_match in LONG_DATE_PATTERN.finditer(haystack):
        parsed_long = _parse_long_date(long_match.group(1))
        if parsed_long:
            found.append((long_match.start(), parsed_long))
    return sorted(found, key=lambda pair: pair[0])


def _lead_kind(haystack: str, start: int) -> str | None:
    """``"deadline"`` or ``"opening"`` when that phrase sits right before ``start``."""
    lead = haystack[max(0, start - _LEAD_WINDOW_CHARS) : start]
    if DEADLINE_LEAD_PATTERN.search(lead):
        return "deadline"
    weak = WEAK_DEADLINE_LEAD_PATTERN.search(lead)
    if weak:
        cue_start = start - len(lead) + weak.start()
        # Bulleted pages have no full stops, so a "sentence" is capped too.
        sentence_start = max(_sentence_start(haystack, cue_start), cue_start - _SENTENCE_CAP_CHARS)
        sentence = haystack[sentence_start:cue_start]
        if ":" in weak.group(0) or _DEADLINE_SENTENCE_PATTERN.search(sentence):
            return "deadline"
    if OPENING_LEAD_PATTERN.search(lead):
        return "opening"
    return None


def _month_number(name: str) -> int | None:
    prefix = name.strip(".").lower()[:3]
    return _MONTH_PREFIXES.index(prefix) + 1 if prefix in _MONTH_PREFIXES else None


def _is_valid_month_day(month: int, day: int) -> bool:
    # A leap year, so "February 29" on a page is accepted and clamped later.
    return 1 <= day <= calendar.monthrange(2000, month)[1]


def _is_valid_iso(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _parse_us_date(value: str) -> str | None:
    try:
        return datetime.strptime(value, "%m/%d/%Y").date().isoformat()
    except ValueError:
        return None


def _parse_long_date(value: str) -> str | None:
    candidate = WS_PATTERN.sub(" ", value.replace(",", " ")).strip()
    for fmt in ("%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(candidate, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _sentence_start(text: str, index: int) -> int:
    boundary = max(text.rfind(". ", 0, index), text.rfind("\n", 0, index))
    return 0 if boundary < 0 else boundary + 1


def _sentence_around(text: str, index: int) -> str:
    start = _sentence_start(text, index)
    end = text.find(". ", index)
    end = len(text) if end < 0 else end + 1
    snippet = text[start:end].strip()
    if len(snippet) > _MAX_REQUIREMENT_EVIDENCE_CHARS:
        snippet = snippet[:_MAX_REQUIREMENT_EVIDENCE_CHARS].rstrip() + "..."
    return snippet


def _letter_count(text: str) -> int | None:
    match = _LETTER_COUNT_PATTERN.search(text)
    if not match:
        return None
    token = match.group(1).lower()
    if token in _NUMBER_WORDS:
        return _NUMBER_WORDS[token]
    try:
        value = int(token)
    except ValueError:
        return None
    return value if 0 <= value <= 10 else None
