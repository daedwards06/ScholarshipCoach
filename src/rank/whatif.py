"""What-if analysis: which awards a changed profile would unlock.

Sophomore year is for building the record, so the useful question is not only
"what am I eligible for today" but "what would a 3.5 GPA or a 1200 SAT open
up".  This module re-runs Stage 1 and the timeline against a hypothetical
profile and diffs the two results, reporting the awards that open, the awards
that close, and the dollars unlocked in each timeline bucket.

Nothing here writes to the stored profile: an override is a question, not an
edit.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any

import pandas as pd

from src.rank.stage1_eligibility import (
    TRUST_UNCONFIRMED_CODE,
    StudentProfile,
    apply_eligibility_filter,
)
from src.rank.timeline import TIMELINE_BUCKETS, classify_timeline

# The profile attributes the what-if view offers, and the wording it uses.
WHATIF_FIELD_LABELS: dict[str, str] = {
    "gpa": "GPA",
    "sat": "SAT",
    "act": "ACT",
    "service_hours": "Service hours",
    "first_gen": "First-generation status",
    "financial_need": "Financial need",
}

# Stage 1 reason codes an override can clear, phrased for a family rather than
# for the pipeline.  An unlisted code falls back to its own spelling.
REASON_LABELS: dict[str, str] = {
    "GPA_BELOW_MIN": "GPA below the minimum",
    "TEST_SCORE_BELOW_MIN": "Test score below the minimum",
    "NEED_BASED_NOT_MET": "Requires demonstrated financial need",
    "FIRST_GEN_ONLY": "First-generation students only",
}


def reason_label(code: str) -> str:
    """Return the family-facing phrasing for a Stage 1 reason code."""
    text = str(code).strip()
    if not text:
        return ""
    return REASON_LABELS.get(text, text.replace("_", " ").capitalize())


@dataclass(frozen=True, slots=True)
class WhatIfAward:
    """One award that changed side when the overrides were applied."""

    scholarship_id: str
    title: str
    amount: float
    timeline_bucket: str
    reasons: list[str] = field(default_factory=list)

    @property
    def reason_text(self) -> str:
        return ", ".join(reason_label(code) for code in self.reasons if str(code).strip())


@dataclass(frozen=True, slots=True)
class WhatIfSummary:
    """The delta between the real profile and an overridden one."""

    overrides: dict[str, Any] = field(default_factory=dict)
    newly_eligible: list[WhatIfAward] = field(default_factory=list)
    newly_ineligible: list[WhatIfAward] = field(default_factory=list)
    dollars_by_bucket: dict[str, float] = field(default_factory=dict)
    needs_confirmation: int = 0

    @property
    def dollars_unlocked(self) -> float:
        return sum(self.dollars_by_bucket.values())

    @property
    def is_noop(self) -> bool:
        return not self.newly_eligible and not self.newly_ineligible


_PROFILE_FIELD_NAMES = frozenset(profile_field.name for profile_field in fields(StudentProfile))


def applied_overrides(profile: StudentProfile, overrides: dict[str, Any]) -> dict[str, Any]:
    """Return only the overrides that name a real field and change its value.

    A value equal to what the profile already holds is dropped, so a summary's
    ``overrides`` says what the question actually asked.
    """
    applied: dict[str, Any] = {}
    for name, value in (overrides or {}).items():
        if name not in _PROFILE_FIELD_NAMES or value is None:
            continue
        if getattr(profile, name) == value:
            continue
        applied[name] = value
    return applied


def apply_overrides(profile: StudentProfile, overrides: dict[str, Any]) -> StudentProfile:
    """Return a copy of ``profile`` with ``overrides`` applied; never mutates it."""
    return replace(profile, **applied_overrides(profile, overrides))


def _award_amount(row: pd.Series) -> float:
    """Resolve an award's dollar value the way Stage 1 does: max, else min, else 0."""
    for column in ("amount_max", "amount_min"):
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        return max(float(value), 0.0)
    return 0.0


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _reason_codes(row: pd.Series | None) -> list[str]:
    if row is None:
        return []
    reasons = row.get("reasons")
    if reasons is None:
        return []
    if hasattr(reasons, "tolist") and not isinstance(reasons, (str, bytes)):
        reasons = reasons.tolist()
    if isinstance(reasons, (list, tuple)):
        return [str(code) for code in reasons if str(code).strip()]
    return [str(reasons)] if str(reasons).strip() else []


def _needs_confirmation(ineligible_df: pd.DataFrame) -> int:
    """Count awards the profile otherwise clears and only trust holds back."""
    return sum(
        1
        for _, row in ineligible_df.iterrows()
        if _reason_codes(row) == [TRUST_UNCONFIRMED_CODE]
    )


def _award(row: pd.Series, reasons: list[str]) -> WhatIfAward:
    return WhatIfAward(
        scholarship_id=_text(row.get("scholarship_id")) or _text(row.get("canonical_id")),
        title=_text(row.get("title")),
        amount=_award_amount(row),
        timeline_bucket=_text(row.get("timeline_bucket")) or "now",
        reasons=reasons,
    )


def _rows_by_index(df: pd.DataFrame) -> dict[Any, pd.Series]:
    return {index: row for index, row in df.iterrows()}


def whatif_eligibility(
    df: pd.DataFrame, profile: StudentProfile, overrides: dict[str, Any]
) -> WhatIfSummary:
    """Diff Stage 1 eligibility between ``profile`` and ``profile`` plus ``overrides``.

    Args:
        df: Scholarship DataFrame with normalized columns.
        profile: The family's real Stage 1 profile.
        overrides: Field name to hypothetical value (see
            :data:`WHATIF_FIELD_LABELS`).  Values equal to the profile's own,
            ``None`` values, and unknown field names are ignored.

    Returns:
        A :class:`WhatIfSummary` whose ``newly_eligible`` awards carry the
        reason codes the overrides cleared, ``newly_ineligible`` awards carry
        the codes the overrides introduced, ``dollars_by_bucket`` totals the
        newly eligible dollars per timeline bucket, and ``needs_confirmation``
        counts the awards the overridden profile clears on every axis except
        ``TRUST_UNCONFIRMED``.  A trust rejection is a gap in the catalog, not
        an axis the student can move, so it is reported as its own count and
        never as a reason an override could clear.
    """
    applied = applied_overrides(profile, overrides)
    if not applied:
        return WhatIfSummary(overrides={})

    what_if_profile = replace(profile, **applied)

    base_eligible, base_ineligible = apply_eligibility_filter(df, profile)
    what_if_eligible, what_if_ineligible = apply_eligibility_filter(df, what_if_profile)
    base_eligible = classify_timeline(base_eligible, profile)
    what_if_eligible = classify_timeline(what_if_eligible, what_if_profile)

    base_eligible_ids = set(base_eligible.index)
    what_if_eligible_ids = set(what_if_eligible.index)
    base_ineligible_rows = _rows_by_index(base_ineligible)
    what_if_ineligible_rows = _rows_by_index(what_if_ineligible)

    newly_eligible = [
        _award(row, _reason_codes(base_ineligible_rows.get(index)))
        for index, row in what_if_eligible.iterrows()
        if index not in base_eligible_ids
    ]
    newly_ineligible = [
        _award(row, _reason_codes(what_if_ineligible_rows.get(index)))
        for index, row in base_eligible.iterrows()
        if index not in what_if_eligible_ids
    ]

    dollars_by_bucket: dict[str, float] = {}
    for award in newly_eligible:
        dollars_by_bucket[award.timeline_bucket] = (
            dollars_by_bucket.get(award.timeline_bucket, 0.0) + award.amount
        )

    return WhatIfSummary(
        overrides=applied,
        newly_eligible=sorted(newly_eligible, key=lambda award: -award.amount),
        newly_ineligible=sorted(newly_ineligible, key=lambda award: -award.amount),
        dollars_by_bucket={
            bucket: dollars_by_bucket[bucket]
            for bucket in TIMELINE_BUCKETS
            if bucket in dollars_by_bucket
        },
        needs_confirmation=_needs_confirmation(what_if_ineligible),
    )
