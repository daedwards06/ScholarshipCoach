"""Canonical normalized scholarship record schema used across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional


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

    # Curated-catalog fields. Optional and default-None so scraped sources,
    # existing snapshots, and golden fixtures keep working unchanged.
    catalog_id: Optional[str] = None
    status: Optional[str] = None
    cycle: Optional[dict[str, Any]] = None
    grade_levels: Optional[list[str]] = None
    counties_allowed: Optional[list[str]] = None
    need_based: Optional[bool] = None
    first_gen_only: Optional[bool] = None
    gender: Optional[str] = None
    heritage: Optional[list[str]] = None
    military_family: Optional[bool] = None
    disability: Optional[bool] = None
    religion: Optional[str] = None
    employer_restricted: Optional[list[str]] = None
    membership_required: Optional[list[str]] = None
    min_test_scores: Optional[dict[str, Any]] = None
    requirements: Optional[dict[str, Any]] = None
    renewal_terms: Optional[str] = None
    trust: Optional[str] = None
    provenance: Optional[dict[str, Any]] = None
    notes: Optional[str] = None
