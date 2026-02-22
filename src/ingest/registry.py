from __future__ import annotations

from .base import BaseSource
from .sources.scholarship_america import ScholarshipAmericaSource


def register_sources() -> list[BaseSource]:
    return [ScholarshipAmericaSource()]
