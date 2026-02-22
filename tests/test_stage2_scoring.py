from __future__ import annotations

import pandas as pd

from src.rank.stage2_scoring import score_stage2


def test_score_stage2_ranks_most_relevant_text_higher() -> None:
    profile = {
        "major": "Computer Science",
        "interests": ["machine learning", "ai"],
        "keywords": ["data science", "python"],
        "extracurriculars": ["robotics club"],
        "goals": "build ethical AI systems",
    }
    df = pd.DataFrame(
        [
            {
                "scholarship_id": "ml-top",
                "title": "Machine Learning and AI Scholarship",
                "description": "Supports computer science students using python for data science.",
                "eligibility_text": "Open to robotics and AI leaders.",
                "essay_prompt": "Describe your machine learning project.",
                "sponsor": "Tech Foundation",
                "amount_min": 1000.0,
                "amount_max": 6000.0,
                "essay_required": False,
                "keywords": ["ai", "machine learning", "python"],
            },
            {
                "scholarship_id": "history-mid",
                "title": "General Undergraduate Award",
                "description": "Supports students from all majors.",
                "eligibility_text": "Open to many disciplines.",
                "essay_prompt": None,
                "sponsor": "Community Fund",
                "amount_min": 1000.0,
                "amount_max": 5500.0,
                "essay_required": False,
                "keywords": ["undergraduate", "leadership"],
            },
            {
                "scholarship_id": "arts-low",
                "title": "Fine Arts Performance Scholarship",
                "description": "Supports dance and theater students.",
                "eligibility_text": "Visual and performing arts focus.",
                "essay_prompt": None,
                "sponsor": "Arts Alliance",
                "amount_min": 1000.0,
                "amount_max": 5000.0,
                "essay_required": False,
                "keywords": ["dance", "theater"],
            },
        ]
    )

    scored_df = score_stage2(df, profile).sort_values("stage2_score", ascending=False)

    assert scored_df["scholarship_id"].tolist()[0] == "ml-top"
    assert scored_df["stage2_score"].iloc[0] > scored_df["stage2_score"].iloc[1]
    assert scored_df["stage2_score"].iloc[1] > scored_df["stage2_score"].iloc[2]


def test_score_stage2_penalizes_essay_required_when_otherwise_equal() -> None:
    profile = {
        "major": "Computer Science",
        "keywords": ["python", "ai"],
        "interests": [],
        "extracurriculars": [],
        "goals": "",
    }
    base_row = {
        "title": "AI Scholars Award",
        "description": "Supports python and ai students.",
        "eligibility_text": "Computer science applicants preferred.",
        "essay_prompt": "Describe your leadership experience.",
        "sponsor": "STEM Org",
        "amount_min": 1000.0,
        "amount_max": 4000.0,
        "keywords": ["ai", "python"],
    }
    df = pd.DataFrame(
        [
            {
                "scholarship_id": "no-essay",
                **base_row,
                "essay_required": False,
            },
            {
                "scholarship_id": "with-essay",
                **base_row,
                "essay_required": True,
            },
        ]
    )

    scored_df = score_stage2(df, profile).set_index("scholarship_id")

    assert scored_df.loc["with-essay", "effort_penalty"] > 0.0
    assert scored_df.loc["no-essay", "stage2_score"] > scored_df.loc["with-essay", "stage2_score"]


def test_score_stage2_is_deterministic_across_runs() -> None:
    profile = {
        "major": "Computer Science",
        "interests": ["machine learning"],
        "keywords": ["python", "data"],
        "extracurriculars": [],
        "goals": "research",
    }
    df = pd.DataFrame(
        [
            {
                "scholarship_id": "a",
                "title": "AI and Data Scholarship",
                "description": "python projects",
                "eligibility_text": "",
                "essay_prompt": "",
                "sponsor": "X",
                "amount_min": None,
                "amount_max": None,
                "essay_required": False,
                "keywords": ["ai", "python"],
            },
            {
                "scholarship_id": "b",
                "title": "Chemistry Scholarship",
                "description": "lab research",
                "eligibility_text": "",
                "essay_prompt": "",
                "sponsor": "Y",
                "amount_min": 1000.0,
                "amount_max": 2000.0,
                "essay_required": True,
                "keywords": ["chemistry"],
            },
            {
                "scholarship_id": "c",
                "title": "General Award",
                "description": "all majors",
                "eligibility_text": "",
                "essay_prompt": "",
                "sponsor": "Z",
                "amount_min": 500.0,
                "amount_max": 1500.0,
                "essay_required": False,
                "keywords": [],
            },
        ]
    )

    run_one = score_stage2(df, profile).reset_index(drop=True)
    run_two = score_stage2(df, profile).reset_index(drop=True)

    pd.testing.assert_frame_equal(run_one, run_two)
    assert run_one.loc[0, "amount_utility"] == 0.0
