from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from src.ingest.sources.curated_catalog import CuratedCatalogSource
from src.io.snapshotting import (
    CATALOG_COLUMNS,
    REQUIRED_COLUMNS,
    prepare_snapshot_df,
    write_parquet_atomic,
)
from src.normalize.canonical_id import generate_scholarship_id
from src.normalize.catalog_schema import (
    RECORDS_DIR,
    iter_catalog_files,
    load_catalog_schema,
    validate_catalog_record,
)
from scripts.validate_catalog import validate_catalog

_FETCHED_AT = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)
_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "data" / "catalog" / "schema.json"


# Inlined rather than read from RECORDS_DIR: the curated catalog is owner-maintained
# data, so pinning a fixture to a live record makes confirming or deleting an award
# break the test suite.
_FIXTURE_ID = "fixture-cs-scholarship"


def _valid_record() -> dict:
    return {
        "catalog_id": _FIXTURE_ID,
        "title": "Fixture CS Scholarship",
        "sponsor": "Fixture Sponsor",
        "source_url": "https://example.org/fixture-cs-scholarship",
        "description": "A representative undergraduate computing award used as a test fixture.",
        "eligibility_text": "Full-time undergraduates studying computer science or computer engineering.",
        "amount_min": 10000.0,
        "amount_max": 10000.0,
        "deadline": "2027-12-01",
        "cycle": {"recurring": True, "opens_month": None, "deadline_month": 12},
        "status": "unknown",
        "education_level": "undergraduate",
        "grade_levels": ["college_1", "college_2", "college_3", "college_4"],
        "majors_allowed": ["Computer Science", "Computer Engineering"],
        "states_allowed": [],
        "counties_allowed": [],
        "min_gpa": None,
        "citizenship": None,
        "need_based": None,
        "first_gen_only": None,
        "gender": None,
        "heritage": [],
        "military_family": None,
        "disability": None,
        "religion": None,
        "employer_restricted": [],
        "membership_required": [],
        "min_test_scores": {"sat": None, "act": None},
        "requirements": {
            "essay": True,
            "essay_prompts": [],
            "recommendation_letters": None,
            "transcript": True,
            "fafsa": None,
            "video_or_portfolio": None,
            "interview": None,
        },
        "renewal_terms": None,
        "keywords": ["computer science", "computer engineering", "undergraduate"],
        "trust": "unverified",
        "provenance": {
            "added_on": "2026-09-12",
            "verified_on": None,
            "verified_by": None,
            "source_kind": "sponsor_site",
        },
        "notes": None,
    }


def _parse_fixture() -> list[dict]:
    payload = json.dumps([_valid_record()]).encode("utf-8")
    return CuratedCatalogSource().parse(payload, fetched_at=_FETCHED_AT)


def _parse_catalog() -> list[dict]:
    source = CuratedCatalogSource()

    class _NoOpClient:
        pass

    raw = source.fetch(_NoOpClient())
    return source.parse(raw.content, fetched_at=_FETCHED_AT)


def test_catalog_parses_every_record() -> None:
    assert len(_parse_catalog()) == len(iter_catalog_files())


def test_catalog_records_have_required_and_catalog_columns() -> None:
    for record in _parse_catalog():
        assert set(REQUIRED_COLUMNS).issubset(record.keys()), f"Missing columns in: {record}"
        assert set(CATALOG_COLUMNS).issubset(record.keys()), f"Missing columns in: {record}"


def test_catalog_populates_eligibility_axes() -> None:
    parsed = _parse_fixture()[0]
    assert parsed["source"] == "curated_catalog"
    assert parsed["majors_allowed"] == ["Computer Science", "Computer Engineering"]
    assert parsed["education_level"] == "undergraduate"
    assert parsed["status"] == "unknown"
    assert parsed["trust"] == "unverified"
    assert parsed["essay_required"] is True
    assert parsed["requirements"]["transcript"] is True
    assert parsed["provenance"]["added_on"] == "2026-09-12"


def test_catalog_ids_are_deterministic() -> None:
    first = sorted(r["scholarship_id"] for r in _parse_catalog())
    second = sorted(r["scholarship_id"] for r in _parse_catalog())
    assert first == second


def test_catalog_id_is_stable_across_a_deadline_change() -> None:
    record = _valid_record()
    source = CuratedCatalogSource()

    original = source.parse(json.dumps([record]).encode("utf-8"), fetched_at=_FETCHED_AT)
    next_cycle_record = deepcopy(record)
    next_cycle_record["deadline"] = "2028-12-01"
    next_cycle_record["amount_max"] = 12000.0
    next_cycle = source.parse(
        json.dumps([next_cycle_record]).encode("utf-8"), fetched_at=_FETCHED_AT
    )

    assert original[0]["deadline"] != next_cycle[0]["deadline"]
    assert original[0]["scholarship_id"] == next_cycle[0]["scholarship_id"]


def test_generate_scholarship_id_without_catalog_id_still_tracks_deadline() -> None:
    kwargs = dict(
        title="Scraped Award",
        sponsor="Some Sponsor",
        amount_min=1000.0,
        amount_max=2000.0,
        source_url="https://example.org/award",
    )
    assert generate_scholarship_id(deadline="2027-01-01", **kwargs) != generate_scholarship_id(
        deadline="2028-01-01", **kwargs
    )


def test_invalid_record_is_skipped_without_failing_the_source() -> None:
    bad_enum = _valid_record() | {"catalog_id": "bad-status-award", "status": "sort-of-open"}
    missing_required = {"title": "No Identity Award"}
    payload = json.dumps([_valid_record(), bad_enum, missing_required]).encode("utf-8")

    records = CuratedCatalogSource().parse(payload, fetched_at=_FETCHED_AT)
    assert [r["catalog_id"] for r in records] == [_FIXTURE_ID]


def test_unreadable_file_is_skipped_at_fetch(tmp_path: Path) -> None:
    (tmp_path / "good-award.json").write_text(json.dumps(_valid_record()), encoding="utf-8")
    (tmp_path / "broken-award.json").write_text("{not json", encoding="utf-8")

    source = CuratedCatalogSource(records_dir=tmp_path)

    class _NoOpClient:
        pass

    raw = source.fetch(_NoOpClient())
    assert len(json.loads(raw.content)) == 1
    assert len(source.parse(raw.content, fetched_at=_FETCHED_AT)) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        {"catalog_id": "Not A Slug"},
        {"status": "expired"},
        {"trust": "trust-me"},
        {"source_url": "ftp://example.org/award"},
        {"min_gpa": 9.5},
        {"grade_levels": ["13"]},
        {"unexpected_field": True},
    ],
    ids=["slug", "status", "trust", "url", "gpa", "grade", "extra"],
)
def test_schema_rejects_bad_values(mutation: dict) -> None:
    record = _valid_record() | mutation
    assert validate_catalog_record(record, schema=load_catalog_schema(_SCHEMA_PATH))


def test_schema_accepts_the_committed_records() -> None:
    schema = load_catalog_schema(_SCHEMA_PATH)
    for path in iter_catalog_files():
        record = json.loads(path.read_text(encoding="utf-8"))
        assert validate_catalog_record(record, schema=schema) == []


def test_validate_catalog_script_passes_on_the_committed_catalog() -> None:
    assert validate_catalog(RECORDS_DIR, _SCHEMA_PATH) == []


def test_validate_catalog_script_flags_filename_and_duplicate_slugs(tmp_path: Path) -> None:
    record = _valid_record()
    (tmp_path / "wrong-name.json").write_text(json.dumps(record), encoding="utf-8")
    (tmp_path / f"{_FIXTURE_ID}.json").write_text(json.dumps(record), encoding="utf-8")

    errors = validate_catalog(tmp_path, _SCHEMA_PATH)
    assert any("filename must match catalog_id" in message for message in errors)
    assert any("duplicate catalog_id" in message for message in errors)


def test_catalog_columns_survive_a_parquet_round_trip(tmp_path: Path) -> None:
    snapshot_df = prepare_snapshot_df(pd.DataFrame(_parse_fixture()))
    assert set(CATALOG_COLUMNS).issubset(snapshot_df.columns)

    output_path = tmp_path / "snapshot.parquet"
    write_parquet_atomic(snapshot_df, output_path)
    loaded = pd.read_parquet(output_path)

    row = loaded.set_index("catalog_id").loc[_FIXTURE_ID]
    assert row["requirements"]["essay"] is True
    assert row["cycle"]["deadline_month"] == 12
    assert row["provenance"]["source_kind"] == "sponsor_site"
    assert list(row["grade_levels"]) == ["college_1", "college_2", "college_3", "college_4"]


def test_prepare_snapshot_df_fills_catalog_columns_for_scraped_rows() -> None:
    scraped = pd.DataFrame([{column: None for column in REQUIRED_COLUMNS}])
    scraped["scholarship_id"] = "scraped-1"

    prepared = prepare_snapshot_df(scraped)
    for column in CATALOG_COLUMNS:
        assert column in prepared.columns
        assert prepared.loc[0, column] is None
