from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(slots=True)
class RawResponse:
    content: bytes
    extension: str
    fetched_at: datetime


class BaseSource(ABC):
    name: str

    @abstractmethod
    def fetch(self, http_client: Any) -> RawResponse:
        """Fetch raw source payload."""

    @abstractmethod
    def parse(self, raw_content: bytes, *, fetched_at: datetime) -> list[dict[str, Any]]:
        """Parse raw source payload into normalized scholarship records."""

    @staticmethod
    def utcnow() -> datetime:
        return datetime.now(tz=UTC)
