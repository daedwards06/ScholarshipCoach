from __future__ import annotations

from typing import Any

import pytest
import requests

from src.llm.client import (
    API_KEY_ENV,
    BASE_URL_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MODEL_ENV,
    LlmClient,
    LlmError,
    client_from_env,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self) -> dict[str, Any]:
        return self._payload


def _ok_payload(content: str = "hello") -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


class _FakeSession:
    """Records posts and replays a scripted sequence of responses/exceptions."""

    def __init__(self, results: list[Any]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: tuple[float, float],
    ) -> _FakeResponse:
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _client(session: _FakeSession, **kwargs: Any) -> LlmClient:
    kwargs.setdefault("backoff_factor", 0.0)  # keep retry tests instant
    return LlmClient(api_key="secret-key", session=session, **kwargs)


def test_client_from_env_returns_none_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    assert client_from_env() is None


def test_client_from_env_returns_none_with_blank_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "   ")
    assert client_from_env() is None


def test_client_from_env_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "abc123")
    monkeypatch.delenv(BASE_URL_ENV, raising=False)
    monkeypatch.delenv(MODEL_ENV, raising=False)

    client = client_from_env()

    assert client is not None
    assert client.api_key == "abc123"
    assert client.base_url == DEFAULT_BASE_URL
    assert client.model == DEFAULT_MODEL


def test_client_from_env_honors_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "abc123")
    monkeypatch.setenv(BASE_URL_ENV, "https://api.groq.com/openai/v1/")
    monkeypatch.setenv(MODEL_ENV, "llama-3.1-8b-instant")

    client = client_from_env()

    assert client is not None
    assert client.base_url == "https://api.groq.com/openai/v1"  # trailing slash stripped
    assert client.model == "llama-3.1-8b-instant"


def test_complete_returns_content_and_sends_expected_payload() -> None:
    session = _FakeSession([_FakeResponse(200, _ok_payload("the answer"))])
    client = _client(session, model="test-model", base_url="https://example.com/v1")

    result = client.complete("be terse", "hi")

    assert result == "the answer"
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == "https://example.com/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer secret-key"
    assert call["headers"]["Content-Type"] == "application/json"
    body = call["json"]
    assert body["model"] == "test-model"
    assert body["temperature"] == 0.0
    assert body["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]


def test_complete_retries_on_transient_status_then_succeeds() -> None:
    session = _FakeSession(
        [_FakeResponse(429), _FakeResponse(503), _FakeResponse(200, _ok_payload("ok"))]
    )
    client = _client(session, max_retries=3)

    assert client.complete("s", "u") == "ok"
    assert len(session.calls) == 3


def test_complete_retries_on_network_error_then_succeeds() -> None:
    session = _FakeSession(
        [requests.ConnectionError("boom"), _FakeResponse(200, _ok_payload("recovered"))]
    )
    client = _client(session, max_retries=3)

    assert client.complete("s", "u") == "recovered"
    assert len(session.calls) == 2


def test_complete_raises_after_exhausting_retries_on_timeout() -> None:
    session = _FakeSession([requests.Timeout("slow")] * 4)
    client = _client(session, max_retries=3)

    with pytest.raises(LlmError):
        client.complete("s", "u")

    assert len(session.calls) == 4  # initial + 3 retries


def test_complete_raises_immediately_on_client_error() -> None:
    session = _FakeSession([_FakeResponse(400)])
    client = _client(session, max_retries=3)

    with pytest.raises(LlmError):
        client.complete("s", "u")

    assert len(session.calls) == 1  # 4xx (non-429) is not retried


def test_complete_raises_on_malformed_payload() -> None:
    session = _FakeSession([_FakeResponse(200, {"unexpected": True})])
    client = _client(session)

    with pytest.raises(LlmError):
        client.complete("s", "u")
