from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_USER_AGENT = "ScholarshipCoachBot/0.1 (+https://localhost; contact=local)"
logger = logging.getLogger(__name__)
_SLOW_REQUEST_SECONDS = 5.0


@dataclass(slots=True)
class PoliteHttpClient:
    requests_per_second: float = 1.0
    timeout_seconds: float = 20.0
    user_agent: str = DEFAULT_USER_AGENT
    max_retries: int = 3
    backoff_factor: float = 0.5
    _session: requests.Session = field(init=False, repr=False)
    _last_request_monotonic: float = field(init=False, default=0.0)
    _rate_limit_lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.user_agent})

        retry = Retry(
            total=self.max_retries,
            connect=self.max_retries,
            read=self.max_retries,
            status=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._last_request_monotonic = 0.0
        self._rate_limit_lock = threading.Lock()

    def close(self) -> None:
        self._session.close()

    def get_bytes(self, url: str, *, params: dict[str, Any] | None = None) -> bytes:
        return self._request("GET", url, params=params).content

    def get_text(self, url: str, *, params: dict[str, Any] | None = None) -> str:
        return self._request("GET", url, params=params).text

    def get_json(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", url, params=params).json()

    @property
    def timeout_tuple(self) -> tuple[float, float]:
        connect_timeout = max(1.0, min(self.timeout_seconds, 5.0))
        read_timeout = max(connect_timeout, self.timeout_seconds)
        return connect_timeout, read_timeout

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Response:
        self._sleep_for_rate_limit()
        started_at = time.monotonic()
        response = self._session.request(
            method=method,
            url=url,
            params=params,
            timeout=self.timeout_tuple,
        )
        elapsed = time.monotonic() - started_at
        if elapsed > _SLOW_REQUEST_SECONDS:
            logger.warning("Slow HTTP %s %.3fs %s", method, elapsed, url)
        response.raise_for_status()
        return response

    def _sleep_for_rate_limit(self) -> None:
        if self.requests_per_second <= 0:
            return
        min_interval = 1.0 / self.requests_per_second
        with self._rate_limit_lock:
            elapsed = time.monotonic() - self._last_request_monotonic
            sleep_seconds = min_interval - elapsed
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            self._last_request_monotonic = time.monotonic()
