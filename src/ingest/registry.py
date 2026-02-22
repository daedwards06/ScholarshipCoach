from __future__ import annotations

from .base import BaseSource
from .sources.scholarship_america_live import ScholarshipAmericaLiveSource


def register_sources() -> list[BaseSource]:
    return [ScholarshipAmericaLiveSource()]
