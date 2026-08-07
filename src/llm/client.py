"""Provider-agnostic chat client for the ingest-boundary LLM layer.

A thin wrapper over ``requests`` targeting any OpenAI-compatible
``/chat/completions`` endpoint (Gemini's OpenAI-compat surface, Groq,
OpenRouter, ...).  It mirrors the retry/backoff/timeout conventions of
:class:`src.ingest.http.PoliteHttpClient` so all outbound HTTP in the project
behaves consistently.

Configuration is entirely environment-driven.  Without an API key the feature
is cleanly disabled: :func:`client_from_env` returns ``None`` and no network
call is ever made.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import requests

logger = logging.getLogger(__name__)

API_KEY_ENV = "SCHOLARSHIPCOACH_LLM_API_KEY"
BASE_URL_ENV = "SCHOLARSHIPCOACH_LLM_BASE_URL"
MODEL_ENV = "SCHOLARSHIPCOACH_LLM_MODEL"

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_MODEL = "gemini-2.0-flash"

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class LlmError(RuntimeError):
    """Raised when the LLM endpoint cannot produce a usable completion."""


class _Transport(Protocol):
    """Minimal subset of ``requests.Session`` the client depends on.

    Declared so tests can inject a fake transport with zero network access.
    """

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: tuple[float, float],
    ) -> requests.Response: ...


@dataclass(slots=True)
class LlmClient:
    """Chat client for an OpenAI-compatible ``/chat/completions`` endpoint.

    Attributes:
        api_key: Bearer token for the provider.  Never logged.
        base_url: Endpoint root, e.g. ``https://api.groq.com/openai/v1``.
            The ``/chat/completions`` suffix is appended automatically.
        model: Model identifier passed through to the provider.
        timeout_seconds: Total request timeout; connect timeout is capped at 5 s.
        max_retries: Retries on transient errors (429, 5xx, network faults).
        backoff_factor: Exponential back-off multiplier between attempts.
        temperature: Sampling temperature; ``0`` for repeatability.
        session: Injectable transport (defaults to a ``requests.Session``).
    """

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 30.0
    max_retries: int = 3
    backoff_factor: float = 0.5
    temperature: float = 0.0
    session: _Transport = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    @property
    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def timeout_tuple(self) -> tuple[float, float]:
        """Return ``(connect_timeout, read_timeout)`` derived from ``timeout_seconds``."""
        connect_timeout = max(1.0, min(self.timeout_seconds, 5.0))
        read_timeout = max(connect_timeout, self.timeout_seconds)
        return connect_timeout, read_timeout

    def complete(self, system: str, user: str) -> str:
        """Send a two-message chat prompt and return the assistant's text.

        Retries transient failures (429, 5xx, network errors) up to
        ``max_retries`` times with exponential back-off before raising
        :class:`LlmError`.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    self._endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_tuple,
                )
            except requests.RequestException as exc:
                last_error = exc
                logger.warning(
                    "LLM request error (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                self._sleep_before_retry(attempt)
                continue

            if response.status_code in _RETRYABLE_STATUS:
                last_error = LlmError(f"transient status {response.status_code}")
                logger.warning(
                    "LLM transient status %d (attempt %d/%d)",
                    response.status_code,
                    attempt + 1,
                    self.max_retries + 1,
                )
                self._sleep_before_retry(attempt)
                continue

            if response.status_code >= 400:
                raise LlmError(f"LLM request failed with status {response.status_code}")

            return _extract_content(response)

        raise LlmError(
            f"LLM request failed after {self.max_retries + 1} attempts"
        ) from last_error

    def _sleep_before_retry(self, attempt: int) -> None:
        if self.backoff_factor <= 0:
            return
        time.sleep(self.backoff_factor * (2**attempt))


def _extract_content(response: requests.Response) -> str:
    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LlmError("malformed LLM response payload") from exc
    if not isinstance(content, str):
        raise LlmError("LLM response content was not text")
    return content


def client_from_env() -> LlmClient | None:
    """Build an :class:`LlmClient` from environment variables.

    Returns ``None`` (feature disabled — no exception, no network call) when
    ``SCHOLARSHIPCOACH_LLM_API_KEY`` is absent or blank.  ``BASE_URL`` and
    ``MODEL`` fall back to sensible defaults.
    """
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        return None
    base_url = os.environ.get(BASE_URL_ENV, "").strip() or DEFAULT_BASE_URL
    model = os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL
    return LlmClient(api_key=api_key, base_url=base_url, model=model)
