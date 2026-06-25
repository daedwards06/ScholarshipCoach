from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.eval.golden_students import GoldenStudent
from src.rank.stage1_eligibility import StudentProfile
from src.rank.weights import Stage2Weights, Stage3Weights


@pytest.fixture
def sample_profile() -> StudentProfile:
    """Standard student profile for testing: CA CS major, 3.5 GPA."""
    return StudentProfile(
        gpa=3.5,
        state="CA",
        major="Computer Science",
        education_level="Undergraduate",
        citizenship="US",
        today=date(2026, 2, 22),
    )


@pytest.fixture
def sample_scholarship_df() -> pd.DataFrame:
    """5-row scholarship DataFrame covering common test scenarios."""
    return pd.DataFrame(
        [
            {
                "scholarship_id": "cs-near",
                "title": "AI and Machine Learning Award",
                "description": "Supports computer science students focused on python and AI.",
                "eligibility_text": "Open to CS undergraduates in CA.",
                "essay_prompt": "Describe your AI project.",
                "sponsor": "Tech Foundation",
                "deadline": date(2026, 3, 1),
                "amount_min": 2000.0,
                "amount_max": 6000.0,
                "essay_required": False,
                "states_allowed": ["CA", "WA"],
                "majors_allowed": ["Computer Science"],
                "min_gpa": 3.0,
                "education_level": "Undergraduate",
                "citizenship": "US",
                "keywords": ["ai", "machine learning", "python"],
            },
            {
                "scholarship_id": "stem-general",
                "title": "General STEM Scholarship",
                "description": "STEM support for undergraduates.",
                "eligibility_text": "Open to all STEM majors.",
                "essay_prompt": None,
                "sponsor": "STEM Foundation",
                "deadline": date(2026, 4, 15),
                "amount_min": 1000.0,
                "amount_max": 4000.0,
                "essay_required": True,
                "states_allowed": [],
                "majors_allowed": [],
                "min_gpa": None,
                "education_level": None,
                "citizenship": None,
                "keywords": ["stem", "undergraduate"],
            },
            {
                "scholarship_id": "cs-highgpa",
                "title": "Excellence in Computing Award",
                "description": "For top computer science students.",
                "eligibility_text": "High-achieving CS students.",
                "essay_prompt": None,
                "sponsor": "Computing Society",
                "deadline": date(2026, 5, 1),
                "amount_min": 5000.0,
                "amount_max": 10000.0,
                "essay_required": False,
                "states_allowed": ["CA"],
                "majors_allowed": ["Computer Science", "Engineering"],
                "min_gpa": 3.8,
                "education_level": "Undergraduate",
                "citizenship": "US",
                "keywords": ["computing", "excellence"],
            },
            {
                "scholarship_id": "arts-low",
                "title": "Fine Arts Performance Scholarship",
                "description": "Supports dance and theater students.",
                "eligibility_text": "Visual and performing arts focus.",
                "essay_prompt": None,
                "sponsor": "Arts Alliance",
                "deadline": date(2026, 6, 1),
                "amount_min": 500.0,
                "amount_max": 2500.0,
                "essay_required": False,
                "states_allowed": [],
                "majors_allowed": ["Fine Arts", "Theater"],
                "min_gpa": None,
                "education_level": None,
                "citizenship": None,
                "keywords": ["dance", "theater"],
            },
            {
                "scholarship_id": "expired",
                "title": "Old Scholarship Award",
                "description": "This deadline has already passed.",
                "eligibility_text": "Anyone.",
                "essay_prompt": None,
                "sponsor": "Past Foundation",
                "deadline": date(2026, 1, 1),
                "amount_min": 1000.0,
                "amount_max": 3000.0,
                "essay_required": False,
                "states_allowed": [],
                "majors_allowed": [],
                "min_gpa": None,
                "education_level": None,
                "citizenship": None,
                "keywords": [],
            },
        ]
    )


@pytest.fixture
def sample_golden_student(sample_profile: StudentProfile) -> GoldenStudent:
    """Standard GoldenStudent fixture wrapping sample_profile."""
    return GoldenStudent(
        student_id="synthetic",
        description="Synthetic profile for testing.",
        profile=sample_profile,
        interests=("machine learning", "ai"),
        keywords=("python", "data science"),
        extracurriculars=("robotics club",),
        goals="build ethical AI systems",
    )


@pytest.fixture
def scholarship_row_factory():
    """Factory fixture for building custom scholarship rows as pd.Series."""
    def _make(
        *,
        scholarship_id: str = "test-row",
        title: str = "Test Scholarship",
        description: str = "A test scholarship.",
        eligibility_text: str = "",
        essay_prompt: str | None = None,
        sponsor: str = "Test Sponsor",
        deadline: date = date(2026, 3, 15),
        amount_min: float | None = 1000.0,
        amount_max: float | None = 5000.0,
        essay_required: bool = False,
        states_allowed: list[str] | None = None,
        majors_allowed: list[str] | None = None,
        min_gpa: float | None = None,
        education_level: str | None = None,
        citizenship: str | None = None,
        keywords: list[str] | None = None,
        source: str = "manual",
        stage2_score: float = 0.70,
        text_sim: float = 0.50,
        keyword_overlap: float = 0.30,
    ) -> pd.Series:
        return pd.Series(
            {
                "scholarship_id": scholarship_id,
                "title": title,
                "description": description,
                "eligibility_text": eligibility_text,
                "essay_prompt": essay_prompt,
                "sponsor": sponsor,
                "deadline": deadline,
                "amount_min": amount_min,
                "amount_max": amount_max,
                "essay_required": essay_required,
                "states_allowed": states_allowed if states_allowed is not None else [],
                "majors_allowed": majors_allowed if majors_allowed is not None else [],
                "min_gpa": min_gpa,
                "education_level": education_level,
                "citizenship": citizenship,
                "keywords": keywords if keywords is not None else [],
                "source": source,
                "stage2_score": stage2_score,
                "text_sim": text_sim,
                "keyword_overlap": keyword_overlap,
            }
        )

    return _make


@pytest.fixture
def sample_stage2_profile(sample_golden_student: GoldenStudent) -> dict:
    """Stage 2 profile dict derived from sample_golden_student."""
    return sample_golden_student.as_stage2_profile()


@pytest.fixture
def baseline_stage2_weights() -> Stage2Weights:
    return Stage2Weights.baseline()


@pytest.fixture
def baseline_stage3_weights() -> Stage3Weights:
    return Stage3Weights.baseline()
