from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.embeddings import model as embedding_model
from src.rank.stage2_scoring import score_stage2


def test_score_stage2_embeddings_mode_is_deterministic(monkeypatch, tmp_path: Path) -> None:
    profile = {
        "major": "Computer Science",
        "keywords": ["ai"],
        "interests": ["ml"],
        "extracurriculars": [],
        "goals": "Build machine learning systems",
    }
    df = pd.DataFrame(
        [
            {
                "scholarship_id": "alpha",
                "title": "AI Research Scholarship",
                "description": "Machine learning students preferred.",
                "eligibility_text": "Computer science applicants welcome.",
                "essay_prompt": None,
                "sponsor": "Tech Fund",
                "amount_min": 1000.0,
                "amount_max": 5000.0,
                "essay_required": False,
                "keywords": ["ai"],
            },
            {
                "scholarship_id": "beta",
                "title": "General Academic Award",
                "description": "Open to many majors.",
                "eligibility_text": "Leadership required.",
                "essay_prompt": None,
                "sponsor": "Civic Group",
                "amount_min": 1000.0,
                "amount_max": 4000.0,
                "essay_required": False,
                "keywords": ["leadership"],
            },
            {
                "scholarship_id": "gamma",
                "title": "Fine Arts Scholarship",
                "description": "Supports performing arts students.",
                "eligibility_text": "Dance and music focus.",
                "essay_prompt": None,
                "sponsor": "Arts Org",
                "amount_min": 1000.0,
                "amount_max": 3000.0,
                "essay_required": False,
                "keywords": ["arts"],
            },
        ]
    )

    vectors = {
        "student": np.array([1.0, 0.0], dtype=np.float32),
        "alpha": np.array([1.0, 0.0], dtype=np.float32),
        "beta": np.array([0.6, 0.8], dtype=np.float32),
        "gamma": np.array([0.0, 1.0], dtype=np.float32),
    }

    def fake_embed_texts(texts: list[str], model_name: str, batch_size: int = 32) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            normalized = text.lower()
            if "build machine learning systems" in normalized:
                rows.append(vectors["student"])
            elif "ai research scholarship" in normalized:
                rows.append(vectors["alpha"])
            elif "general academic award" in normalized:
                rows.append(vectors["beta"])
            else:
                rows.append(vectors["gamma"])
        return np.vstack(rows)

    monkeypatch.setattr(embedding_model, "embed_texts", fake_embed_texts)

    run_one = score_stage2(
        df,
        profile,
        similarity_mode="embeddings",
        model_name="all-MiniLM-L6-v2",
        processed_dir=tmp_path,
    ).sort_values("stage2_score", ascending=False, kind="mergesort")
    run_two = score_stage2(
        df,
        profile,
        similarity_mode="embeddings",
        model_name="all-MiniLM-L6-v2",
        processed_dir=tmp_path,
    ).sort_values("stage2_score", ascending=False, kind="mergesort")

    assert run_one["scholarship_id"].tolist() == ["alpha", "beta", "gamma"]
    np.testing.assert_allclose(run_one["text_sim"].to_numpy(), np.array([1.0, 0.6, 0.0]))
    assert "embed_sim" in run_one.columns
    assert (tmp_path / "embeddings").exists()
    pd.testing.assert_frame_equal(run_one.reset_index(drop=True), run_two.reset_index(drop=True))
