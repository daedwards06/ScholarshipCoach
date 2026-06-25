"""Abstract base classes for scholarship data source connectors.

Each ingest source extends :class:`BaseSource` and implements
:meth:`~BaseSource.fetch` (HTTP retrieval) and :meth:`~BaseSource.parse`
(HTML/JSON → normalized records).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(slots=True)
class RawResponse:
    """Unparsed HTTP response payload returned by a source's ``fetch`` method.

    Attributes:
        content: Raw response bytes (HTML or JSON).
        extension: File extension hint for caching (e.g. ``"html"``, ``"json"``).
        fetched_at: UTC timestamp of the HTTP request.
    """

    content: bytes
    extension: str
    fetched_at: datetime


class BaseSource(ABC):
    """Abstract base class for scholarship data source connectors.

    Subclasses implement ``fetch`` (HTTP retrieval) and ``parse`` (extraction
    of normalized scholarship records from raw response bytes).  The ``name``
    class attribute uniquely identifies the source in the snapshot.
    """

    name: str

    @abstractmethod
    def fetch(self, http_client: Any) -> RawResponse:
        """Fetch raw source payload."""

    @abstractmethod
    def parse(self, raw_content: bytes, *, fetched_at: datetime) -> list[dict[str, Any]]:
        """Parse raw source payload into normalized scholarship records."""

    @staticmethod
    def utcnow() -> datetime:
        """Return the current UTC datetime."""
        return datetime.now(tz=UTC)
