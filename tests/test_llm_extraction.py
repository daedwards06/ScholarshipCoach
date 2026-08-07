from __future__ import annotations

import json
from typing import Any

import pytest

from src.llm.client import LlmError
from src.llm.extraction import (
    EXTRACTION_FIELDS,
    EXTRACTION_PROMPT_VERSION,
    EXTRACTION_SYSTEM_PROMPT,
    build_user_prompt,
    extract_fields,
    parse_extraction,
)


class _CannedClient:
    """Returns a scripted response (or raises) and records the prompts it saw."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


_FULL_EXTRACTION = {
    "deadline": "2026-11-30",
    "amount_min": 1000,
    "amount_max": 5000,
    "min_gpa": 3.0,
    "states_allowed": ["NC", "South Carolina"],
    "majors_allowed": ["Computer Science"],
    "education_level": "Undergraduate",
    "citizenship": "US Citizen",
    "essay_required": True,
    "keywords": ["stem", "diversity"],
}


def test_prompt_version_is_declared() -> None:
    assert isinstance(EXTRACTION_PROMPT_VERSION, str)
    assert EXTRACTION_PROMPT_VERSION


def test_system_prompt_names_every_field_and_forbids_guessing() -> None:
    for field_name in EXTRACTION_FIELDS:
        assert field_name in EXTRACTION_SYSTEM_PROMPT
    assert "null" in EXTRACTION_SYSTEM_PROMPT
    assert "Never guess" in EXTRACTION_SYSTEM_PROMPT


def test_build_user_prompt_includes_all_source_text() -> None:
    prompt = build_user_prompt(
        title="Rural STEM Award",
        description="For students in rural counties.",
        eligibility_text="Must reside in North Carolina.",
    )

    assert "Rural STEM Award" in prompt
    assert "rural counties" in prompt
    assert "North Carolina" in prompt


def test_build_user_prompt_marks_missing_sections() -> None:
    prompt = build_user_prompt(title="Award", description=None, eligibility_text="")

    assert prompt.count("(not provided)") == 2


def test_parse_valid_json_validates_and_normalizes_every_field() -> None:
    result = parse_extraction(json.dumps(_FULL_EXTRACTION))

    assert result == {
        "deadline": "2026-11-30",
        "amount_min": 1000.0,
        "amount_max": 5000.0,
        "min_gpa": 3.0,
        "states_allowed": ["NC", "SC"],
        "majors_allowed": ["computer science"],
        "education_level": "undergraduate",
        "citizenship": "us",
        "essay_required": True,
        "keywords": ["stem", "diversity"],
    }


def test_parse_strips_code_fences() -> None:
    fenced = "```json\n" + json.dumps({"min_gpa": 2.5}) + "\n```"

    assert parse_extraction(fenced) == {"min_gpa": 2.5}


def test_parse_recovers_object_embedded_in_prose() -> None:
    raw = 'Sure! Here is the data:\n{"education_level": "college"}\nHope that helps.'

    assert parse_extraction(raw) == {"education_level": "undergraduate"}


def test_parse_ignores_braces_inside_strings() -> None:
    raw = '{"majors_allowed": ["computer {science}"], "min_gpa": 3.5}'

    result = parse_extraction(raw)

    assert result["majors_allowed"] == ["computer {science}"]
    assert result["min_gpa"] == 3.5


def test_parse_ignores_braces_inside_escaped_strings() -> None:
    raw = 'Here: {"majors_allowed": ["say \\"{hi}\\" now"], "min_gpa": 2.0} done'

    result = parse_extraction(raw)

    assert result["majors_allowed"] == ['say "{hi}" now']
    assert result["min_gpa"] == 2.0


@pytest.mark.parametrize(
    "raw",
    ["", "not json at all", "{ broken json", "[1, 2, 3]", '"a string"'],
)
def test_parse_malformed_response_returns_empty(raw: str) -> None:
    assert parse_extraction(raw) == {}


def test_parse_drops_null_fields() -> None:
    payload = {field_name: None for field_name in EXTRACTION_FIELDS}

    assert parse_extraction(json.dumps(payload)) == {}


def test_parse_discards_hallucinated_extra_fields() -> None:
    raw = json.dumps(
        {
            "min_gpa": 3.0,
            "sponsor_phone_number": "555-0100",
            "renewable": True,
            "scholarship_id": "made-up-id",
        }
    )

    assert parse_extraction(raw) == {"min_gpa": 3.0}


@pytest.mark.parametrize("gpa", [-0.5, 5.1, 100, "n/a", None, True])
def test_parse_drops_out_of_range_gpa(gpa: Any) -> None:
    assert "min_gpa" not in parse_extraction(json.dumps({"min_gpa": gpa}))


def test_parse_accepts_gpa_at_range_bounds() -> None:
    assert parse_extraction(json.dumps({"min_gpa": 0}))["min_gpa"] == 0.0
    assert parse_extraction(json.dumps({"min_gpa": 5}))["min_gpa"] == 5.0


@pytest.mark.parametrize("deadline", ["2019-12-31", "2041-01-01", "11/30/2026", "soon", 20261130])
def test_parse_drops_implausible_deadline(deadline: Any) -> None:
    assert "deadline" not in parse_extraction(json.dumps({"deadline": deadline}))


def test_parse_drops_negative_amount() -> None:
    result = parse_extraction(json.dumps({"amount_min": -100, "amount_max": 500}))

    assert result == {"amount_max": 500.0}


def test_parse_drops_both_amounts_when_min_exceeds_max() -> None:
    result = parse_extraction(json.dumps({"amount_min": 9000, "amount_max": 500}))

    assert "amount_min" not in result
    assert "amount_max" not in result


def test_parse_coerces_currency_formatted_amounts() -> None:
    result = parse_extraction(json.dumps({"amount_min": "$1,500", "amount_max": "$1,500"}))

    assert result == {"amount_min": 1500.0, "amount_max": 1500.0}


def test_parse_drops_unknown_states_and_sorts_the_rest() -> None:
    raw = json.dumps({"states_allowed": ["texas", "Freedonia", "ZZ", "CA", "CA"]})

    assert parse_extraction(raw) == {"states_allowed": ["CA", "TX"]}


def test_parse_drops_states_field_when_nothing_maps() -> None:
    assert parse_extraction(json.dumps({"states_allowed": ["Freedonia"]})) == {}


@pytest.mark.parametrize(
    ("raw_level", "expected"),
    [
        ("High School", "high school"),
        ("undergrad", "undergraduate"),
        ("PhD", "graduate"),
        ("Masters", "graduate"),
    ],
)
def test_parse_maps_education_level_to_pipeline_vocabulary(raw_level: str, expected: str) -> None:
    result = parse_extraction(json.dumps({"education_level": raw_level}))

    assert result["education_level"] == expected


def test_parse_drops_unknown_education_level() -> None:
    assert parse_extraction(json.dumps({"education_level": "post-doc fellow"})) == {}


@pytest.mark.parametrize(
    ("raw_citizenship", "expected"),
    [
        ("U.S. Citizen", "us"),
        ("Green Card holder", "permanent resident"),
        ("International Student", "international"),
    ],
)
def test_parse_maps_citizenship_to_pipeline_vocabulary(
    raw_citizenship: str, expected: str
) -> None:
    result = parse_extraction(json.dumps({"citizenship": raw_citizenship}))

    assert result["citizenship"] == expected


def test_parse_drops_unknown_citizenship() -> None:
    assert parse_extraction(json.dumps({"citizenship": "dual national"})) == {}


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [(True, True), (False, False), ("Yes", True), ("no", False), ("Required", True)],
)
def test_parse_coerces_essay_required(raw_value: Any, expected: bool) -> None:
    result = parse_extraction(json.dumps({"essay_required": raw_value}))

    assert result["essay_required"] is expected


def test_parse_drops_uninterpretable_essay_required() -> None:
    assert parse_extraction(json.dumps({"essay_required": "maybe"})) == {}


def test_parse_dedupes_and_caps_keywords() -> None:
    raw = json.dumps({"keywords": ["STEM", "stem", ""] + [f"kw{i}" for i in range(30)]})

    keywords = parse_extraction(raw)["keywords"]

    assert len(keywords) == 20
    assert keywords[0] == "stem"
    assert keywords.count("stem") == 1


def test_parse_ignores_non_string_list_items() -> None:
    raw = json.dumps({"majors_allowed": ["Biology", 42, None, {"x": 1}]})

    assert parse_extraction(raw) == {"majors_allowed": ["biology"]}


def test_parse_drops_list_field_given_a_scalar_of_wrong_type() -> None:
    assert parse_extraction(json.dumps({"keywords": 7})) == {}


def test_parse_accepts_bare_string_for_list_field() -> None:
    assert parse_extraction(json.dumps({"majors_allowed": "Nursing"})) == {
        "majors_allowed": ["nursing"]
    }


def test_extract_fields_sends_prompts_and_returns_validated_fields() -> None:
    client = _CannedClient(json.dumps(_FULL_EXTRACTION))

    result = extract_fields(
        client,
        title="Rural STEM Award",
        description="For students in rural counties.",
        eligibility_text="North Carolina residents, 3.0 GPA.",
    )

    assert result["min_gpa"] == 3.0
    assert result["states_allowed"] == ["NC", "SC"]
    assert len(client.calls) == 1
    system, user = client.calls[0]
    assert system == EXTRACTION_SYSTEM_PROMPT
    assert "Rural STEM Award" in user


def test_extract_fields_returns_empty_on_client_error() -> None:
    client = _CannedClient(LlmError("provider down"))

    result = extract_fields(client, title="T", description="D", eligibility_text="E")

    assert result == {}


def test_extract_fields_returns_empty_on_unusable_response() -> None:
    client = _CannedClient("I'm sorry, I can't help with that.")

    assert extract_fields(client, title="T", description="D", eligibility_text="E") == {}


def test_extract_fields_truncates_very_long_text() -> None:
    client = _CannedClient("{}")

    extract_fields(
        client,
        title="T",
        description="x" * 10_000,
        eligibility_text="E",
    )

    _, user = client.calls[0]
    assert len(user) < 6_000
    assert "..." in user
