"""Shared type definitions for the ScholarshipCoach pipeline."""
from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable


@runtime_checkable
class ProfileLike(Protocol):
    """Protocol for student profile objects accessible via attribute access.

    Satisfied by ``StudentProfile`` (dataclass).  Dict-based profiles from
    ``GoldenStudent.as_stage2_profile()`` are handled via the ``dict[str, Any]``
    branch of ``_get_profile_value`` / ``_profile_value`` helpers in stage2 and
    win_model respectively.  Function signatures use ``ProfileLike | dict[str, Any]``
    to document both concrete types that the pipeline accepts.
    """

    gpa: float | None
    state: str | None
    major: str | None
    education_level: str | None
    citizenship: str | None
    today: date | None
