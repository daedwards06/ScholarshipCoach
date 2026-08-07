from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.llm import cache as extraction_cache
from src.llm.cache import (
    compute_extraction_key,
    extraction_path,
    get_or_extract,
    sanitize_model_name,
)


class _CountingClient:
    """Fake client that records every call and returns a scripted JSON body."""

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
        "states_allowed": ["NC"],
        "majors_allowed": ["Computer Science"],
        "education_level": "undergraduate",
        "citizenship": "us",
        "essay_required": True,
        "keywords": ["stem"],
    }
)

_RECORD: dict[str, Any] = {
    "title": "Tar Heel STEM Award",
    "description": "For NC undergraduates studying computer science.",
    "eligibility_text": "Minimum 3.0 GPA. Essay required.",
}


def _client() -> _CountingClient:
    return _CountingClient(_EXTRACTION_JSON)


def test_miss_calls_client_then_hit_avoids_it(tmp_path: Path) -> None:
    client = _client()

    first = get_or_extract(client, _RECORD, processed_dir=tmp_path)
    assert len(client.calls) == 1
    assert first["min_gpa"] == 3.0
    assert first["states_allowed"] == ["NC"]

    second = get_or_extract(client, _RECORD, processed_dir=tmp_path)
    assert len(client.calls) == 1, "cache hit must not call the client"
    assert second == first


def test_written_document_carries_metadata(tmp_path: Path) -> None:
    client = _client()
    fields = get_or_extract(client, _RECORD, processed_dir=tmp_path)

    key = compute_extraction_key(_RECORD, model_name=client.model)
    path = extraction_path(key, client.model, processed_dir=tmp_path)
    assert path.exists()

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["fields"] == fields
    assert document["model"] == client.model
    assert document["prompt_version"] == extraction_cache.EXTRACTION_PROMPT_VERSION
    assert document["extracted_at"]


def test_cached_path_lives_under_sanitized_model_dir(tmp_path: Path) -> None:
    client = _client()
    get_or_extract(client, _RECORD, processed_dir=tmp_path)

    model_dir = tmp_path / "llm_extractions" / sanitize_model_name(client.model)
    assert model_dir.is_dir()
    assert sanitize_model_name("gemini-2.0/flash") == "gemini-2.0_flash"
    assert list(model_dir.glob("*.json"))


def test_prompt_version_bump_changes_key_and_recalls_client(tmp_path: Path) -> None:
    client = _client()
    get_or_extract(client, _RECORD, processed_dir=tmp_path, prompt_version="v1")
    get_or_extract(client, _RECORD, processed_dir=tmp_path, prompt_version="v2")

    assert len(client.calls) == 2
    assert compute_extraction_key(
        _RECORD, model_name=client.model, prompt_version="v1"
    ) != compute_extraction_key(_RECORD, model_name=client.model, prompt_version="v2")


def test_model_and_text_changes_change_the_key() -> None:
    baseline = compute_extraction_key(_RECORD, model_name="model-a")
    assert compute_extraction_key(_RECORD, model_name="model-b") != baseline

    changed_text = {**_RECORD, "eligibility_text": "Minimum 3.5 GPA."}
    assert compute_extraction_key(changed_text, model_name="model-a") != baseline


def test_disabled_client_returns_empty_on_miss(tmp_path: Path) -> None:
    assert get_or_extract(None, _RECORD, processed_dir=tmp_path, model_name="m") == {}
    assert not (tmp_path / "llm_extractions").exists()


def test_disabled_client_still_reads_the_cache(tmp_path: Path) -> None:
    client = _client()
    expected = get_or_extract(client, _RECORD, processed_dir=tmp_path)

    assert get_or_extract(
        None, _RECORD, processed_dir=tmp_path, model_name=client.model
    ) == expected


def test_disabled_client_without_model_name_returns_empty(tmp_path: Path) -> None:
    assert get_or_extract(None, _RECORD, processed_dir=tmp_path) == {}


def test_empty_extraction_is_cached_so_it_is_not_retried(tmp_path: Path) -> None:
    client = _CountingClient("not json at all")

    assert get_or_extract(client, _RECORD, processed_dir=tmp_path) == {}
    assert get_or_extract(client, _RECORD, processed_dir=tmp_path) == {}
    assert len(client.calls) == 1


def test_corrupt_cache_entry_is_ignored_and_rewritten(tmp_path: Path) -> None:
    client = _client()
    key = compute_extraction_key(_RECORD, model_name=client.model)
    path = extraction_path(key, client.model, processed_dir=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ truncated", encoding="utf-8")

    fields = get_or_extract(client, _RECORD, processed_dir=tmp_path)

    assert len(client.calls) == 1
    assert fields["min_gpa"] == 3.0
    assert json.loads(path.read_text(encoding="utf-8"))["fields"] == fields
