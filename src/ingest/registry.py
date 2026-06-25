"""Source registry: returns all active ingest connectors for a pipeline run."""
from __future__ import annotations

from .base import BaseSource
from .sources.scholarship_america_live import ScholarshipAmericaLiveSource


def register_sources() -> list[BaseSource]:
    """Return all active :class:`~src.ingest.base.BaseSource` instances for ingest."""
    return [ScholarshipAmericaLiveSource()]
