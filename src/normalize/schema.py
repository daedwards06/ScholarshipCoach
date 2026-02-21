from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


@dataclass(slots=True)
class NormalizedScholarshipRecord:
    """Canonical normalized scholarship record used across the pipeline."""

    scholarship_id: str
    source: str
    source_id: Optional[str]
    source_url: str
    title: str
    sponsor: Optional[str]
    description: Optional[str]
    eligibility_text: Optional[str]
    deadline: Optional[date]
    amount_min: Optional[float]
    amount_max: Optional[float]
    is_recurring: Optional[bool]
    states_allowed: Optional[list[str]]
    majors_allowed: Optional[list[str]]
    min_gpa: Optional[float]
    citizenship: Optional[str]
    education_level: Optional[str]
    essay_required: Optional[bool]
    essay_prompt: Optional[str]
    keywords: Optional[list[str]]
    first_seen_at: datetime
    last_seen_at: datetime
