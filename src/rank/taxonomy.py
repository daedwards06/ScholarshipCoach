"""Static major-family and education-level adjacency maps for Stage 1 matching.

Stage 1 originally required exact-string equality on the major and education
checks, which dropped near-adjacent matches: a "Computer Science" student
missed "Computer Engineering"/"STEM" awards, and a college-bound "high school"
student was filtered out of every "Undergraduate" award. The helpers here relax
both checks to shared-family / adjacent-level matching while preserving exact
behavior for unknown values and under a strict escape hatch.
"""
from __future__ import annotations

from collections.abc import Iterable

from src.text_utils import normalize_list, normalize_text

# Normalized major (or family token) -> set of families it belongs to.
# Family tokens ("stem", "engineering", "health") map to themselves so that a
# scholarship listing a family as its allowed "major" still matches a specific
# member major. Unknown majors are intentionally absent: they fall back to
# exact-string matching only.
_MAJOR_FAMILIES: dict[str, frozenset[str]] = {
    "stem": frozenset({"stem"}),
    "engineering": frozenset({"engineering", "stem"}),
    "health": frozenset({"health"}),
    "computer science": frozenset({"engineering", "stem"}),
    "computer engineering": frozenset({"engineering", "stem"}),
    "software engineering": frozenset({"engineering", "stem"}),
    "electrical engineering": frozenset({"engineering", "stem"}),
    "mechanical engineering": frozenset({"engineering", "stem"}),
    "civil engineering": frozenset({"engineering", "stem"}),
    "biomedical engineering": frozenset({"engineering", "stem", "health"}),
    "information technology": frozenset({"stem"}),
    "data science": frozenset({"stem"}),
    "mathematics": frozenset({"stem"}),
    "statistics": frozenset({"stem"}),
    "physics": frozenset({"stem"}),
    "chemistry": frozenset({"stem"}),
    "biology": frozenset({"stem"}),
    "environmental science": frozenset({"stem"}),
    "nursing": frozenset({"health"}),
    "public health": frozenset({"health"}),
    "kinesiology": frozenset({"health"}),
}

# Normalized profile education level -> scholarship levels it is allowed to match
# beyond an exact hit. A college-bound high-school student matches undergraduate
# awards; every level always matches itself (handled in the helper).
_EDUCATION_ADJACENCY: dict[str, frozenset[str]] = {
    "high school": frozenset({"undergraduate"}),
}


def _families(major: str) -> frozenset[str]:
    return _MAJOR_FAMILIES.get(major, frozenset())


def majors_match(profile_major: str | None, majors_allowed: Iterable[str] | str | None) -> bool:
    """Return whether a profile major satisfies a scholarship's allowed majors.

    Passes when the scholarship lists no majors (open to all), on an exact
    normalized match, or when the profile major and some allowed major share a
    family (e.g. computer science ↔ engineering ↔ STEM). A profile major absent
    from the taxonomy falls back to exact matching only.

    Args:
        profile_major: The student's major (any casing; ``None`` treated empty).
        majors_allowed: The scholarship's allowed majors — a list, a bare
            string, or ``None``.

    Returns:
        ``True`` if the major is allowed, ``False`` otherwise.
    """
    allowed = normalize_list(majors_allowed)
    if not allowed:
        return True
    profile = normalize_text(profile_major)
    if profile in allowed:
        return True
    profile_families = _families(profile)
    if not profile_families:
        return False
    return any(profile_families & _families(allowed_major) for allowed_major in allowed)


def education_level_matches(
    profile_level: str | None,
    scholarship_level: str | None,
    *,
    strict: bool = False,
) -> bool:
    """Return whether a profile education level satisfies a scholarship's level.

    Passes when the scholarship states no level, on an exact normalized match,
    or (unless ``strict``) when the scholarship level is adjacent to the profile
    level — a high-school profile matches undergraduate awards.

    Args:
        profile_level: The student's education level (any casing).
        scholarship_level: The scholarship's required level (any casing).
        strict: When ``True``, only an exact match passes; adjacency is ignored.

    Returns:
        ``True`` if the level is allowed, ``False`` otherwise.
    """
    scholarship = normalize_text(scholarship_level)
    if not scholarship:
        return True
    profile = normalize_text(profile_level)
    if scholarship == profile:
        return True
    if strict:
        return False
    return scholarship in _EDUCATION_ADJACENCY.get(profile, frozenset())
