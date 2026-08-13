from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.run_ingest import (
    _enrich_records_with_llm,
    _normalize_records,
    run_ingest,
)
from src.io.snapshotting import LLM_PROVENANCE_COLUMN


@pytest.fixture(autouse=True)
def stub_embeddings(monkeypatch) -> None:
    """Keep the snapshot path real while skipping sentence-transformer loading."""

    def _fake_store(df, model_name, *, processed_dir=None, batch_size=32):  # noqa: ANN001
        keyed = df.copy()
        keyed["embedding_key"] = [f"key-{position}" for position in range(len(keyed))]
        return keyed

    monkeypatch.setattr("src.io.snapshotting.ensure_embedding_store_for_df", _fake_store)


class _CountingClient:
    """Fake LLM client returning a fixed extraction and recording every call."""

    model = "fake-model/v1"

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._response


_EXTRACTION_JSON = json.dumps(
    {
        "deadline": "2026-11-30",
        "amount_min": 1000,
        "amount_max": 5000,
        "min_gpa": 3.0,
        "states_allowed": ["North Carolina"],
        "majors_allowed": ["Computer Science"],
        "education_level": "undergraduate",
        "citizenship": "us",
        "essay_required": True,
        "keywords": ["stem"],
    }
)


def _raw_record(scholarship_id: str, **overrides: Any) -> dict[str, Any]:
    fetched_at = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    record: dict[str, Any] = {
        "scholarship_id": scholarship_id,
        "source": "fixture",
        "source_id": scholarship_id,
        "source_url": f"https://example.com/{scholarship_id}",
        "title": "Tar Heel STEM Award",
        "description": "For North Carolina undergraduates studying computer science.",
        "eligibility_text": "Minimum 3.0 GPA. Essay required. Award: $1,000-$5,000.",
        "first_seen_at": fetched_at,
        "last_seen_at": fetched_at,
    }
    record.update(overrides)
    return record


def _frame(*records: dict[str, Any]) -> pd.DataFrame:
    return _normalize_records(list(records))


def test_enrichment_fills_only_empty_fields(tmp_path: Path) -> None:
    client = _CountingClient(_EXTRACTION_JSON)
    df = _frame(_raw_record("a1", min_gpa=2.5, states_allowed=["VA"]))

    enriched, summary = _enrich_records_with_llm(
        df,
        client=client,
        model_name=client.model,
        processed_dir=tmp_path,
        max_calls=10,
    )

    row = enriched.iloc[0]
    assert row["min_gpa"] == 2.5, "parser value must never be overwritten"
    assert row["states_allowed"] == ["VA"]
    assert row["deadline"] == date(2026, 11, 30)
    assert row["amount_min"] == 1000.0
    assert row["education_level"] == "undergraduate"
    assert row["essay_required"] is True

    assert "min_gpa" not in row[LLM_PROVENANCE_COLUMN]
    assert "states_allowed" not in row[LLM_PROVENANCE_COLUMN]
    assert set(row[LLM_PROVENANCE_COLUMN]) == {
        "deadline",
        "amount_min",
        "amount_max",
        "majors_allowed",
        "education_level",
        "citizenship",
        "essay_required",
        "keywords",
    }
    assert summary["records_enriched"] == 1
    assert summary["fields_filled"] == 8
    assert summary["fields_filled_by_field"]["deadline"] == 1


def test_records_without_empty_fields_or_text_are_skipped(tmp_path: Path) -> None:
    client = _CountingClient(_EXTRACTION_JSON)
    complete = _raw_record(
        "complete",
        deadline="2026-05-01",
        amount_min=500,
        amount_max=500,
        min_gpa=3.0,
        states_allowed=["NC"],
        majors_allowed=["Computer Science"],
        education_level="undergraduate",
        citizenship="us",
        essay_required=False,
        keywords=["stem"],
    )
    textless = _raw_record("textless", description="", eligibility_text=None)

    enriched, summary = _enrich_records_with_llm(
        _frame(complete, textless),
        client=client,
        model_name=client.model,
        processed_dir=tmp_path,
        max_calls=10,
    )

    assert client.calls == []
    assert summary["records_scanned"] == 2
    assert summary["records_eligible"] == 0
    assert all(len(value) == 0 for value in enriched[LLM_PROVENANCE_COLUMN])


def test_cache_hit_avoids_a_second_api_call(tmp_path: Path) -> None:
    client = _CountingClient(_EXTRACTION_JSON)
    df = _frame(_raw_record("a1"))

    _, first = _enrich_records_with_llm(
        df, client=client, model_name=client.model, processed_dir=tmp_path, max_calls=10
    )
    _, second = _enrich_records_with_llm(
        df, client=client, model_name=client.model, processed_dir=tmp_path, max_calls=10
    )

    assert len(client.calls) == 1
    assert (first["api_calls"], first["cache_hits"]) == (1, 0)
    assert (second["api_calls"], second["cache_hits"]) == (0, 1)


def test_max_calls_caps_live_api_calls(tmp_path: Path) -> None:
    client = _CountingClient(_EXTRACTION_JSON)
    df = _frame(
        _raw_record("a1", title="Award One"),
        _raw_record("a2", title="Award Two"),
        _raw_record("a3", title="Award Three"),
    )

    enriched, summary = _enrich_records_with_llm(
        df, client=client, model_name=client.model, processed_dir=tmp_path, max_calls=2
    )

    assert len(client.calls) == 2
    assert summary["api_calls"] == 2
    assert summary["records_eligible"] == 3
    assert summary["max_calls_reached"] is True
    assert sum(1 for value in enriched[LLM_PROVENANCE_COLUMN] if value) == 2


def test_disabled_client_applies_cached_extractions_only(tmp_path: Path) -> None:
    client = _CountingClient(_EXTRACTION_JSON)
    cached_df = _frame(_raw_record("a1"))
    _enrich_records_with_llm(
        cached_df,
        client=client,
        model_name=client.model,
        processed_dir=tmp_path,
        max_calls=10,
    )

    df = _frame(_raw_record("a1"), _raw_record("a2", title="Uncached Award"))
    enriched, summary = _enrich_records_with_llm(
        df, client=None, model_name=client.model, processed_dir=tmp_path, max_calls=10
    )

    assert summary["enabled"] is False
    assert summary["api_calls"] == 0
    assert summary["cache_hits"] == 1
    assert enriched.iloc[0][LLM_PROVENANCE_COLUMN]
    assert list(enriched.iloc[1][LLM_PROVENANCE_COLUMN]) == []
    assert len(client.calls) == 1


def test_invalid_extraction_leaves_fields_empty(tmp_path: Path) -> None:
    client = _CountingClient(json.dumps({"min_gpa": 99, "deadline": "not-a-date"}))
    df = _frame(_raw_record("a1"))

    enriched, summary = _enrich_records_with_llm(
        df, client=client, model_name=client.model, processed_dir=tmp_path, max_calls=10
    )

    row = enriched.iloc[0]
    assert pd.isna(row["min_gpa"])
    assert row["deadline"] is None or pd.isna(row["deadline"])
    assert list(row[LLM_PROVENANCE_COLUMN]) == []
    assert summary["fields_filled"] == 0


def test_run_ingest_round_trips_provenance_through_the_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    class _FakeSource:
        name = "fake_source"

        def fetch_records(self, http_client, *, raw_root, **kwargs):  # noqa: ANN001
            return [_raw_record("a1")], [], {"listing_urls_processed": 1}

    client = _CountingClient(_EXTRACTION_JSON)
    monkeypatch.setattr("scripts.run_ingest.register_sources", lambda: [_FakeSource()])
    monkeypatch.setattr("scripts.run_ingest.client_from_env", lambda: client)

    report = run_ingest(
        date=date(2026, 3, 1),
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        report_dir=tmp_path / "reports",
        llm_enrich=True,
        llm_max_calls=5,
    )

    assert report["status"] == "success"
    assert report["config"]["llm_enrich"] is True
    assert report["llm_enrichment"]["enabled"] is True
    assert report["llm_enrichment"]["api_calls"] == 1
    assert report["llm_enrichment"]["fields_filled"] > 0

    snapshot_df = pd.read_parquet(report["artifact_paths"]["snapshot"])
    assert LLM_PROVENANCE_COLUMN in snapshot_df.columns
    assert "deadline" in list(snapshot_df.iloc[0][LLM_PROVENANCE_COLUMN])
    assert snapshot_df.iloc[0]["education_level"] == "undergraduate"


def test_run_ingest_without_a_key_completes_and_reports_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    class _FakeSource:
        name = "fake_source"

        def fetch_records(self, http_client, *, raw_root, **kwargs):  # noqa: ANN001
            return [_raw_record("a1")], [], {"listing_urls_processed": 1}

    monkeypatch.setattr("scripts.run_ingest.register_sources", lambda: [_FakeSource()])
    monkeypatch.setattr("scripts.run_ingest.client_from_env", lambda: None)

    report = run_ingest(
        date=date(2026, 3, 1),
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        report_dir=tmp_path / "reports",
        llm_enrich=True,
    )

    assert report["status"] == "success"
    assert report["llm_enrichment"]["enabled"] is False
    assert report["llm_enrichment"]["api_calls"] == 0

    snapshot_df = pd.read_parquet(report["artifact_paths"]["snapshot"])
    assert list(snapshot_df.iloc[0][LLM_PROVENANCE_COLUMN]) == []


def test_run_ingest_without_enrichment_still_writes_the_column(
    monkeypatch, tmp_path: Path
) -> None:
    class _FakeSource:
        name = "fake_source"

        def fetch_records(self, http_client, *, raw_root, **kwargs):  # noqa: ANN001
            return [_raw_record("a1")], [], {"listing_urls_processed": 1}

    monkeypatch.setattr("scripts.run_ingest.register_sources", lambda: [_FakeSource()])

    report = run_ingest(
        date=date(2026, 3, 1),
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        report_dir=tmp_path / "reports",
    )

    assert report["llm_enrichment"]["requested"] is False
    snapshot_df = pd.read_parquet(report["artifact_paths"]["snapshot"])
    assert list(snapshot_df.iloc[0][LLM_PROVENANCE_COLUMN]) == []
