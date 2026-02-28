from __future__ import annotations

from src.embeddings.cache import compute_embedding_key


def test_embedding_key_changes_when_text_fields_change() -> None:
    base_record = {
        "scholarship_id": "abc123",
        "title": "STEM Scholars Award",
        "sponsor": "Tech Foundation",
        "description": "Supports students in STEM fields.",
        "eligibility_text": "Open to U.S. undergraduates.",
        "essay_prompt": "Describe your leadership experience.",
    }

    baseline_key = compute_embedding_key(base_record)

    for field_name, updated_value in (
        ("title", "Updated STEM Scholars Award"),
        ("sponsor", "Another Sponsor"),
        ("description", "Supports graduate students."),
        ("eligibility_text", "Open to all majors."),
        ("essay_prompt", "Describe your research goals."),
    ):
        variant = dict(base_record)
        variant[field_name] = updated_value
        assert compute_embedding_key(variant) != baseline_key

