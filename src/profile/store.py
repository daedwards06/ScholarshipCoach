"""Private, per-student profile storage.

A profile is personal data, so it lives under ``data/private/students/``
(git-ignored), one JSON file per student.  Multi-student is modeled from the
start because the file layout is the expensive part to change later.

A fictional demo profile is committed at ``data/demo/student_demo.json`` and
loaded when no private profile exists, so a fresh clone runs with realistic
data and the UI can say plainly that it is showing a demo.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from src.profile.grade_levels import grade_label_to_levels, levels_to_grade_label
from src.rank.stage1_eligibility import StudentProfile

ROOT_DIR = Path(__file__).resolve().parents[2]
PRIVATE_DIR = ROOT_DIR / "data" / "private"
STUDENTS_DIR = PRIVATE_DIR / "students"
DEMO_PROFILE_PATH = ROOT_DIR / "data" / "demo" / "student_demo.json"

DEFAULT_STUDENT_ID = "student_1"

_STUDENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_STAGE1_EDUCATION_LEVELS = frozenset({"high school", "undergraduate", "graduate"})

_TEXT_FIELDS = (
    "name",
    "state",
    "county",
    "high_school",
    "major",
    "education_level",
    "grade_level",
    "citizenship",
    "goals",
)
_OPTIONAL_TEXT_FIELDS = ("gender", "religion")
_LIST_FIELDS = (
    "heritage",
    "parent_employers",
    "memberships",
    "intended_colleges",
    "extracurriculars",
    "profile_keywords",
)
_TRISTATE_FIELDS = ("financial_need", "first_gen", "military_family", "disability")
_INT_FIELDS = ("graduation_year", "service_hours", "sat", "act")


def default_profile() -> dict[str, Any]:
    """Return an empty profile with every field present at its unset value."""
    return {
        "student_id": DEFAULT_STUDENT_ID,
        "name": "",
        "gpa": 0.0,
        "state": "",
        "county": "",
        "high_school": "",
        "major": "",
        "education_level": "",
        "grade_level": "",
        "graduation_year": None,
        "citizenship": "",
        "financial_need": None,
        "first_gen": None,
        "gender": None,
        "heritage": [],
        "military_family": None,
        "disability": None,
        "religion": None,
        "parent_employers": [],
        "memberships": [],
        "service_hours": None,
        "sat": None,
        "act": None,
        "intended_colleges": [],
        "essay_ready": False,
        "extracurriculars": [],
        "profile_keywords": [],
        "goals": "",
        "today_override": date.today().isoformat(),
        "use_today_override": False,
    }


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _coerce_tristate(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"yes", "true", "1"}:
            return True
        if lowered in {"no", "false", "0"}:
            return False
        return None
    return bool(value)


def _coerce_optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def normalize_profile(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Return ``payload`` merged onto the defaults with field types coerced.

    Unknown keys are preserved so a profile written by a later task survives a
    round-trip through an older build.
    """
    profile = default_profile()
    profile.update(payload or {})

    profile["student_id"] = str(profile.get("student_id") or DEFAULT_STUDENT_ID)
    profile["gpa"] = float(profile.get("gpa") or 0.0)
    for key in _TEXT_FIELDS:
        profile[key] = str(profile.get(key) or "")
    for key in _OPTIONAL_TEXT_FIELDS:
        profile[key] = _coerce_optional_text(profile.get(key))
    for key in _LIST_FIELDS:
        profile[key] = _coerce_list(profile.get(key))
    for key in _TRISTATE_FIELDS:
        profile[key] = _coerce_tristate(profile.get(key))
    for key in _INT_FIELDS:
        profile[key] = _coerce_int(profile.get(key))
    profile["essay_ready"] = bool(profile.get("essay_ready", False))
    profile["use_today_override"] = bool(profile.get("use_today_override", False))
    profile["today_override"] = str(profile.get("today_override") or date.today().isoformat())

    # A profile saved before Task 1.2 stored a UI label ("High School Senior")
    # in education_level; recover the Stage 1 vocabulary from it.
    if not profile["grade_level"] or profile["education_level"] not in _STAGE1_EDUCATION_LEVELS:
        label = levels_to_grade_label(profile["education_level"], profile["grade_level"])
        if label:
            education_level, grade_level = grade_label_to_levels(label)
            profile["education_level"] = education_level or profile["education_level"]
            profile["grade_level"] = grade_level or profile["grade_level"]

    return profile


def _validate_student_id(student_id: str) -> str:
    if not _STUDENT_ID_RE.match(student_id):
        raise ValueError(
            f"Invalid student_id {student_id!r}: use letters, digits, hyphen or "
            "underscore (max 64 characters)."
        )
    return student_id


def profile_path(student_id: str = DEFAULT_STUDENT_ID, students_dir: Path | None = None) -> Path:
    """Return the private JSON path for ``student_id``."""
    directory = students_dir or STUDENTS_DIR
    return directory / f"{_validate_student_id(student_id)}.json"


def list_students(students_dir: Path | None = None) -> list[str]:
    """Return every stored student id, sorted, or ``[]`` when none exist."""
    directory = students_dir or STUDENTS_DIR
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.json"))


def load_profile(
    student_id: str = DEFAULT_STUDENT_ID, students_dir: Path | None = None
) -> dict[str, Any] | None:
    """Load a private profile, or ``None`` when the student has no file yet."""
    path = profile_path(student_id, students_dir)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return normalize_profile(payload)


def save_profile(
    profile: dict[str, Any],
    student_id: str | None = None,
    students_dir: Path | None = None,
) -> Path:
    """Write ``profile`` to private storage and return the path written."""
    normalized = normalize_profile(profile)
    resolved_id = student_id or normalized["student_id"]
    normalized["student_id"] = _validate_student_id(str(resolved_id))
    path = profile_path(normalized["student_id"], students_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_demo_profile(demo_path: Path | None = None) -> dict[str, Any]:
    """Load the committed fictional demo profile."""
    path = demo_path or DEMO_PROFILE_PATH
    return normalize_profile(json.loads(path.read_text(encoding="utf-8-sig")))


def load_profile_or_demo(
    student_id: str = DEFAULT_STUDENT_ID,
    students_dir: Path | None = None,
    demo_path: Path | None = None,
) -> tuple[dict[str, Any], bool]:
    """Return ``(profile, is_demo)``, falling back to the committed demo profile."""
    profile = load_profile(student_id, students_dir)
    if profile is not None:
        return profile, False
    return load_demo_profile(demo_path), True


def to_stage1_profile(profile: dict[str, Any], today: date | None = None) -> StudentProfile:
    """Build the Stage 1 ``StudentProfile`` from a stored profile dict."""
    normalized = normalize_profile(profile)
    return StudentProfile(
        gpa=float(normalized["gpa"] or 0.0),
        state=normalized["state"] or None,
        major=normalized["major"] or None,
        education_level=normalized["education_level"] or None,
        citizenship=normalized["citizenship"] or None,
        today=today,
        student_id=normalized["student_id"],
        graduation_year=normalized["graduation_year"],
        grade_level=normalized["grade_level"] or None,
        county=normalized["county"] or None,
        high_school=normalized["high_school"] or None,
        financial_need=normalized["financial_need"],
        first_gen=normalized["first_gen"],
        gender=normalized["gender"],
        heritage=normalized["heritage"],
        military_family=normalized["military_family"],
        disability=normalized["disability"],
        religion=normalized["religion"],
        parent_employers=normalized["parent_employers"],
        memberships=normalized["memberships"],
        service_hours=normalized["service_hours"],
        sat=normalized["sat"],
        act=normalized["act"],
        intended_colleges=normalized["intended_colleges"],
        essay_ready=normalized["essay_ready"],
    )


def to_stage2_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Build the Stage 2 scoring dict from a stored profile dict."""
    normalized = normalize_profile(profile)
    keywords = normalized["profile_keywords"]
    return {
        "major": normalized["major"],
        "keywords": keywords,
        "interests": keywords,
        "goals": normalized["goals"],
        "extracurriculars": normalized["extracurriculars"],
    }
