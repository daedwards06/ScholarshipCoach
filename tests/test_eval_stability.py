from __future__ import annotations

from datetime import date

import pandas as pd

from src.rank.stage1_eligibility import StudentProfile, apply_eligibility_filter
from src.rank.stage2_scoring import score_stage2
from src.rank.stage3_rerank import rerank_stage3


def _run_pipeline_ids(df: pd.DataFrame, profile: StudentProfile, k: int) -> list[str]:
    eligible_df, _ = apply_eligibility_filter(df=df, profile=profile)
    if eligible_df.empty:
        return []
    scored_df = score_stage2(
        eligible_df,
        {
            "major": profile.major,
            "interests": ["ai", "robotics"],
            "keywords": ["python", "machine learning"],
            "extracurriculars": ["club"],
            "goals": "Build practical systems.",
        },
    )
    reranked_df = rerank_stage3(scored_df, today=profile.today)
    top_k = min(k, len(reranked_df))
    return reranked_df.head(top_k)["scholarship_id"].tolist()


def test_pipeline_topk_ids_are_identical_across_repeated_runs() -> None:
    profile = StudentProfile(
        gpa=3.4,
        state="CA",
        major="Computer Science",
        education_level="Undergraduate",
        citizenship="US",
        today=date(2026, 2, 22),
    )
    df = pd.DataFrame(
        [
            {
                "scholarship_id": "id-a",
                "title": "AI Scholars Award",
                "description": "AI and python projects.",
                "eligibility_text": "Computer science students.",
                "essay_prompt": None,
                "sponsor": "Tech Org",
                "deadline": date(2026, 3, 15),
                "amount_min": 1000.0,
                "amount_max": 5000.0,
                "essay_required": False,
                "states_allowed": ["CA", "WA"],
                "majors_allowed": ["Computer Science"],
                "education_level": "Undergraduate",
                "citizenship": "US",
                "min_gpa": 3.0,
                "keywords": ["ai", "python"],
            },
            {
                "scholarship_id": "id-b",
                "title": "General STEM Award",
                "description": "STEM support for undergraduates.",
                "eligibility_text": "",
                "essay_prompt": None,
                "sponsor": "Foundation",
                "deadline": date(2026, 4, 1),
                "amount_min": 500.0,
                "amount_max": 3000.0,
                "essay_required": True,
                "states_allowed": [],
                "majors_allowed": [],
                "education_level": None,
                "citizenship": None,
                "min_gpa": None,
                "keywords": ["stem"],
            },
            {
                "scholarship_id": "id-c",
                "title": "Arts Scholarship",
                "description": "For performing arts.",
                "eligibility_text": "",
                "essay_prompt": None,
                "sponsor": "Arts Alliance",
                "deadline": date(2026, 3, 10),
                "amount_min": 1000.0,
                "amount_max": 2500.0,
                "essay_required": False,
                "states_allowed": ["CA"],
                "majors_allowed": ["Computer Science", "Art"],
                "education_level": "Undergraduate",
                "citizenship": "US",
                "min_gpa": 2.5,
                "keywords": ["arts"],
            },
        ]
    )

    run_one_ids = _run_pipeline_ids(df, profile, k=3)
    run_two_ids = _run_pipeline_ids(df, profile, k=3)

    assert run_one_ids == run_two_ids
