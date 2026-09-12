from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.ingest.extract_common import (
    extract_amount_range,
    find_nc_counties,
    find_requirement_flags,
    find_states,
    html_to_text,
    parse_date_candidates,
    parse_first_date,
)
from src.ingest.prefill import (
    ExtractionResult,
    Extractor,
    PrefillResult,
    RegexExtractor,
    extraction_result_from_fields,
    prefill_from_html,
    prefill_from_url,
)

_RESOURCES = Path(__file__).resolve().parent / "resources"
_FOUNDATION = _RESOURCES / "prefill_foundation_sample.html"
_AGGREGATOR = _RESOURCES / "prefill_aggregator_sample.html"
_NO_SIGNALS = _RESOURCES / "prefill_no_signals_sample.html"


class _StubClient:
    def __init__(self, html: str) -> None:
        self.html = html
        self.requested: list[str] = []

    def get_text(self, url: str) -> str:
        self.requested.append(url)
        return self.html


class _FailingClient:
    def get_text(self, url: str) -> str:
        raise RuntimeError("connection reset")


def _prefill(fixture: Path) -> PrefillResult:
    client = _StubClient(fixture.read_text(encoding="utf-8"))
    return prefill_from_url("https://example.org/award", client)


# --- extract_common -------------------------------------------------------


def test_html_to_text_drops_script_and_style_bodies() -> None:
    text = html_to_text(_FOUNDATION.read_text(encoding="utf-8"))
    assert "2001-01-01" not in text
    assert "Deadline 1/1/2000" not in text
    assert "Tar Heel Engineering Scholarship" in text


def test_parse_date_candidates_merges_all_three_formats_in_page_order() -> None:
    text = "Opens 2027-01-10, closes 4/30/2027, final round June 1, 2027."
    assert parse_date_candidates(text) == ["2027-01-10", "2027-04-30", "2027-06-01"]


def test_parse_date_candidates_dedupes_and_rejects_impossible_dates() -> None:
    assert parse_date_candidates("2027-02-30 and 2027-03-01 and 2027-03-01") == ["2027-03-01"]


def test_parse_first_date_prefers_iso_over_long_form_regardless_of_position() -> None:
    assert parse_first_date("Opened January 3, 2026; machine deadline 2026-05-01") == "2026-05-01"


def test_parse_first_date_accepts_long_form_without_a_comma() -> None:
    assert parse_first_date("Due March 15 2027") == "2027-03-15"


def test_extract_amount_range_returns_none_without_evidence() -> None:
    assert extract_amount_range("no dollar figures here") == (None, None)


def test_find_states_ignores_two_letter_codes() -> None:
    assert find_states("Applicants in OR IN ME may apply") is None
    assert find_states("Open to North Carolina and Virginia") == ["North Carolina", "Virginia"]


def test_find_nc_counties_requires_the_word_county() -> None:
    assert find_nc_counties("Serving Wake County and Durham County") == ["Durham", "Wake"]
    assert find_nc_counties("Named for Justice Union and Mr. Moore") is None


def test_find_requirement_flags_omits_negated_requirements() -> None:
    flags, evidence = find_requirement_flags("No essay is required for this award.")
    assert "essay" not in flags
    assert "essay" not in evidence


def test_find_requirement_flags_reads_a_stated_letter_count() -> None:
    flags, evidence = find_requirement_flags("Two letters of recommendation are required.")
    assert flags["recommendation_letters"] == 2
    assert "letters of recommendation" in evidence["recommendation_letters"]


def test_find_requirement_flags_omits_letter_count_when_unstated() -> None:
    flags, evidence = find_requirement_flags("Letters of recommendation must be submitted.")
    assert "recommendation_letters" not in flags
    assert "recommendation_letters" in evidence


# --- foundation page ------------------------------------------------------


def test_foundation_page_prefills_title_sponsor_and_description() -> None:
    result = _prefill(_FOUNDATION)
    assert result.error is None
    assert result.title == "Tar Heel Engineering Scholarship"
    assert result.sponsor == "Piedmont Education Foundation"
    assert result.description is not None
    assert result.description.startswith("A renewable award")
    assert result.confidence["title"] == pytest.approx(0.9)


def test_foundation_page_prefers_the_labeled_deadline_over_a_page_date() -> None:
    result = _prefill(_FOUNDATION)
    assert result.extraction.deadline_candidates == ["2027-03-15"]
    assert result.extraction.deadline == "2027-03-15"
    assert result.confidence["deadline"] == pytest.approx(0.85)


def test_foundation_page_prefills_amount_state_county_and_gpa() -> None:
    result = _prefill(_FOUNDATION)
    extraction = result.extraction
    assert extraction.amount_candidates == [2500.0]
    assert extraction.amount_min == 2500.0
    assert extraction.amount_max == 2500.0
    assert extraction.states_allowed == ["North Carolina"]
    assert extraction.counties_allowed == ["Durham", "Wake"]
    assert extraction.min_gpa == pytest.approx(3.0)
    assert result.confidence["states_allowed"] == pytest.approx(0.7)


def test_foundation_page_prefills_requirements_with_evidence() -> None:
    extraction = _prefill(_FOUNDATION).extraction
    assert extraction.requirements["essay"] is True
    assert extraction.requirements["recommendation_letters"] == 2
    assert extraction.requirements["transcript"] is True
    assert extraction.requirements["interview"] is True
    assert "video_or_portfolio" not in extraction.requirements
    assert "fafsa" not in extraction.requirements
    assert set(extraction.requirement_evidence) >= set(extraction.requirements)


def test_form_dict_carries_every_field_the_entry_form_needs() -> None:
    form = _prefill(_FOUNDATION).to_form_dict()
    assert form["source_url"] == "https://example.org/award"
    assert form["title"] == "Tar Heel Engineering Scholarship"
    assert form["deadline_candidates"] == ["2027-03-15"]
    assert form["extractor"] == "regex"
    assert form["error"] is None


# --- aggregator page ------------------------------------------------------


def test_aggregator_page_returns_every_date_candidate_without_choosing() -> None:
    extraction = _prefill(_AGGREGATOR).extraction
    assert extraction.deadline_candidates == ["2027-01-10", "2027-04-30", "2027-06-01"]
    assert extraction.deadline is None


def test_aggregator_page_scores_unlabeled_multi_candidate_fields_lower() -> None:
    result = _prefill(_AGGREGATOR)
    assert result.extraction.amount_candidates == [1000.0, 5000.0]
    assert result.extraction.amount_min == 1000.0
    assert result.extraction.amount_max == 5000.0
    assert result.confidence["deadline"] < 0.6
    assert result.confidence["states_allowed"] == pytest.approx(0.35)


def test_aggregator_page_falls_back_to_the_h1_for_a_title() -> None:
    result = _prefill(_AGGREGATOR)
    assert result.title == "STEM Futures Award"
    assert result.sponsor is None
    assert result.confidence["title"] == pytest.approx(0.75)


def test_aggregator_page_respects_a_negated_essay_requirement() -> None:
    extraction = _prefill(_AGGREGATOR).extraction
    assert "essay" not in extraction.requirements
    assert extraction.requirements["video_or_portfolio"] is True


# --- page with no signals -------------------------------------------------


def test_page_without_signals_leaves_every_field_empty() -> None:
    result = _prefill(_NO_SIGNALS)
    extraction = result.extraction
    assert result.title == "About our foundation"
    assert result.sponsor is None
    assert result.description is None
    assert extraction.deadline_candidates == []
    assert extraction.amount_candidates == []
    assert extraction.amount_min is None
    assert extraction.states_allowed is None
    assert extraction.counties_allowed is None
    assert extraction.min_gpa is None
    assert extraction.requirements == {}
    assert "deadline" not in result.confidence
    assert "amount_min" not in result.confidence


# --- fetch behaviour and the extractor seam -------------------------------


def test_prefill_from_url_reports_a_fetch_failure_instead_of_raising() -> None:
    result = prefill_from_url("https://example.org/gone", _FailingClient())
    assert result.error is not None
    assert "connection reset" in result.error
    assert result.title is None
    assert result.extraction.deadline_candidates == []


def test_prefill_from_url_fetches_the_requested_url_once() -> None:
    client = _StubClient(_NO_SIGNALS.read_text(encoding="utf-8"))
    prefill_from_url("https://example.org/award", client)
    assert client.requested == ["https://example.org/award"]


def test_regex_extractor_satisfies_the_extractor_protocol() -> None:
    assert isinstance(RegexExtractor(), Extractor)


def test_a_custom_extractor_replaces_the_regex_one_without_touching_the_form() -> None:
    class _StubExtractor:
        name = "stub"

        def extract(self, title: str | None, text: str | None) -> ExtractionResult:
            return ExtractionResult(
                deadline_candidates=["2030-01-01"], confidence={"deadline": 1.0}
            )

    result = prefill_from_html(
        _NO_SIGNALS.read_text(encoding="utf-8"),
        url="https://example.org/award",
        extractor=_StubExtractor(),
    )
    assert result.extractor == "stub"
    assert result.to_form_dict()["deadline"] == "2030-01-01"


def test_llm_extraction_fields_convert_into_an_extraction_result() -> None:
    fields: dict[str, Any] = {
        "deadline": "2027-03-15",
        "amount_min": 1000.0,
        "amount_max": 5000.0,
        "min_gpa": 3.0,
        "states_allowed": ["nc"],
        "majors_allowed": ["computer science"],
        "education_level": "undergraduate",
        "citizenship": "us",
        "essay_required": True,
        "keywords": ["stem"],
    }
    result = extraction_result_from_fields(fields)
    assert result.deadline == "2027-03-15"
    assert result.amount_candidates == [1000.0, 5000.0]
    assert result.majors_allowed == ["computer science"]
    assert result.requirements == {"essay": True}
    assert set(result.confidence) == set(fields)


def test_llm_extraction_result_is_empty_when_nothing_validated() -> None:
    result = extraction_result_from_fields({})
    assert result.deadline is None
    assert result.amount_candidates == []
    assert result.confidence == {}
