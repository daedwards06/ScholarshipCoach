from __future__ import annotations

from datetime import date

from src.catalog import entry
from src.ingest.prefill import prefill_from_html


def _minimal_form() -> dict:
    values = entry.blank_form()
    values["title"] = "Wake County STEM Award"
    values["source_url"] = "https://example.org/stem-award"
    return values


def test_minimal_form_produces_a_valid_record() -> None:
    record, errors = entry.validate_form(_minimal_form(), today=date(2026, 9, 13))

    assert errors == []
    assert record["catalog_id"] == "wake-county-stem-award"
    assert record["provenance"]["added_on"] == "2026-09-13"
    assert record["status"] == "unknown"
    assert record["trust"] == "unverified"


def test_form_fills_every_axis_the_dropdowns_offer() -> None:
    values = _minimal_form()
    values.update(
        {
            "catalog_id": "Stem Award 2027",
            "sponsor": "Example Foundation",
            "amount_min": 1000,
            "amount_max": 2500.0,
            "deadline": date(2027, 3, 1),
            "cycle_recurring": "Yes",
            "cycle_opens_month": 11,
            "cycle_deadline_month": 3,
            "status": "open",
            "education_level": "undergraduate",
            "grade_levels": ["college_1", "college_2"],
            "majors_allowed": ["computer science", "engineering"],
            "states_allowed": ["North Carolina", "Virginia"],
            "counties_allowed": ["Wake", "Durham"],
            "min_gpa": 3.0,
            "sat": 1200,
            "act": 26,
            "gender": "female",
            "need_based": "No",
            "heritage": "Cherokee, Lumbee",
            "membership_required": ["Society of Women Engineers"],
            "req_essay": True,
            "req_interview": "No",
            "recommendation_letters": 2,
            "essay_prompts": ["Why engineering?"],
            "keywords": "stem, local",
            "trust": "verified_local",
            "source_kind": "community_foundation",
            "verified_on": "2026-09-13",
            "verified_by": "parent",
        }
    )

    record, errors = entry.validate_form(values, today=date(2026, 9, 13))

    assert errors == []
    assert record["catalog_id"] == "stem-award-2027"
    assert record["deadline"] == "2027-03-01"
    assert record["cycle"] == {"recurring": True, "opens_month": 11, "deadline_month": 3}
    assert record["states_allowed"] == ["NC", "VA"]
    assert record["min_test_scores"] == {"sat": 1200, "act": 26}
    assert record["requirements"]["essay"] is True
    assert record["requirements"]["interview"] is False
    assert record["requirements"]["fafsa"] is None
    assert record["requirements"]["recommendation_letters"] == 2
    assert record["heritage"] == ["Cherokee", "Lumbee"]
    assert record["keywords"] == ["stem", "local"]
    assert record["provenance"]["verified_on"] == "2026-09-13"


def test_missing_title_and_bad_url_are_reported_not_written() -> None:
    values = entry.blank_form()
    values["source_url"] = "example.org/award"

    record, errors = entry.validate_form(values)

    assert record["catalog_id"] == ""
    assert any("title" in message for message in errors)
    assert any("source_url" in message for message in errors)


def test_out_of_vocabulary_choices_are_dropped_rather_than_stored() -> None:
    values = _minimal_form()
    values["grade_levels"] = ["college_1", "kindergarten"]
    values["states_allowed"] = ["North Carolina", "Atlantis"]
    values["status"] = "maybe"
    values["source_kind"] = "a friend"

    record, errors = entry.validate_form(values)

    assert errors == []
    assert record["grade_levels"] == ["college_1"]
    assert record["states_allowed"] == ["NC"]
    assert record["status"] == "unknown"
    assert record["provenance"]["source_kind"] == "other"


def test_record_round_trips_back_into_form_values() -> None:
    values = _minimal_form()
    values.update(
        {
            "states_allowed": ["North Carolina"],
            "deadline": "2027-03-01",
            "req_essay": True,
            "recommendation_letters": 1,
            "cycle_deadline_month": 3,
        }
    )
    record, _ = entry.validate_form(values)

    reloaded = entry.form_from_record(record)

    assert reloaded["states_allowed"] == ["North Carolina"]
    assert reloaded["deadline"] == "2027-03-01"
    assert reloaded["req_essay"] is True
    assert reloaded["recommendation_letters"] == 1
    assert reloaded["cycle_deadline_month"] == 3
    assert entry.record_from_form(reloaded) == record


def test_prefill_payload_seeds_the_form_without_guessing() -> None:
    html = """
    <html><head><meta property="og:title" content="Example STEM Scholarship">
    <meta name="description" content="An award for STEM students."></head>
    <body><p>Deadline: March 1, 2027. Amount: $2,500.</p>
    <p>Open to residents of Wake County, North Carolina.</p>
    <p>An essay and a transcript are required.</p></body></html>
    """
    payload = prefill_from_html(html, url="https://example.org/award").to_form_dict()

    values = entry.form_from_prefill(payload)

    assert values["title"] == "Example STEM Scholarship"
    assert values["catalog_id"] == "example-stem-scholarship"
    assert values["source_url"] == "https://example.org/award"
    assert values["deadline"] == "2027-03-01"
    assert values["amount_candidates"] == [2500.0]
    assert values["states_allowed"] == ["North Carolina"]
    assert values["counties_allowed"] == ["Wake"]
    assert values["req_essay"] is True
    assert values["req_transcript"] is True
    assert values["req_interview"] is None

    record, errors = entry.validate_form(values)
    assert errors == []
    assert record["states_allowed"] == ["NC"]


def test_ambiguous_dates_stay_candidates_instead_of_being_picked() -> None:
    html = "<html><body><p>Applications open 2026-11-01 and close 2027-03-01.</p></body></html>"
    payload = prefill_from_html(html, url="https://example.org/award").to_form_dict()

    values = entry.form_from_prefill(payload)

    assert values["deadline"] == ""
    assert values["deadline_candidates"] == ["2026-11-01", "2027-03-01"]


def test_diff_rows_render_every_value_kind() -> None:
    rows = entry.diff_rows(
        {
            "status": {"old": "open", "new": "closed"},
            "amount_max": {"old": None, "new": 5000.0},
            "states_allowed": {"old": [], "new": ["NC", "VA"]},
            "requirements": {"old": {"essay": True}, "new": {"essay": False}},
        }
    )

    assert [row.field for row in rows] == [
        "amount_max",
        "requirements",
        "states_allowed",
        "status",
    ]
    assert (rows[0].old, rows[0].new) == ("—", "5000")
    assert rows[1].new == '{"essay": false}'
    assert (rows[2].old, rows[2].new) == ("—", "NC, VA")
    assert (rows[3].old, rows[3].new) == ("open", "closed")


def test_diff_rows_tolerate_a_malformed_entry_and_no_diff() -> None:
    assert entry.diff_rows(None) == []
    assert entry.diff_rows({}) == []

    rows = entry.diff_rows({"notes": "replaced wholesale"})

    assert rows == [entry.DiffRow(field="notes", old="—", new="replaced wholesale")]


def test_render_value_spells_out_booleans_and_blanks() -> None:
    assert entry.render_value(True) == "yes"
    assert entry.render_value(False) == "no"
    assert entry.render_value(None) == "—"
    assert entry.render_value("") == "—"
    assert entry.render_value([]) == "—"


def test_dropdown_options_come_from_the_schema_and_the_matcher() -> None:
    assert entry.status_options() == ("open", "upcoming", "closed", "unknown")
    assert "verified_local" in entry.trust_options()
    assert None not in entry.gender_options()
    assert "community_foundation" in entry.source_kind_options()
    assert "computer science" in entry.MAJOR_OPTIONS
    assert "Wake" in entry.COUNTY_OPTIONS


def test_existing_record_majors_survive_a_load_into_the_form() -> None:
    reloaded = entry.form_from_record(
        {"majors_allowed": ["Computer Science", "Cybersecurity"]}
    )

    assert reloaded["majors_allowed"] == ["computer science", "Cybersecurity"]
