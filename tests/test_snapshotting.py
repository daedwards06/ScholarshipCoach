from __future__ import annotations

import pandas as pd

from src.io.snapshotting import build_delta


def test_build_delta_reports_added_removed_and_tracked_field_changes() -> None:
    prior_df = pd.DataFrame(
        [
            {
                "scholarship_id": "same",
                "title": "STEM Scholars Award",
                "deadline": "2026-04-15",
                "eligibility_text": "Open to STEM undergraduates.",
                "amount_min": 1000.0,
                "amount_max": 5000.0,
            },
            {
                "scholarship_id": "removed",
                "title": "Legacy Scholarship",
                "deadline": "2026-05-01",
                "eligibility_text": "Open to all majors.",
                "amount_min": 500.0,
                "amount_max": 2500.0,
            },
        ]
    )
    current_df = pd.DataFrame(
        [
            {
                "scholarship_id": "same",
                "title": "STEM Scholars Award - Updated",
                "deadline": "2026-04-30",
                "eligibility_text": "Open to STEM majors in the US.",
                "amount_min": 1000.0,
                "amount_max": 5500.0,
            },
            {
                "scholarship_id": "added",
                "title": "New Opportunity Scholarship",
                "deadline": "2026-06-10",
                "eligibility_text": "First-generation students preferred.",
                "amount_min": 750.0,
                "amount_max": 3000.0,
            },
        ]
    )

    delta = build_delta(current_df=current_df, prior_df=prior_df)

    assert [entry["scholarship_id"] for entry in delta["added"]] == ["added"]
    assert [entry["scholarship_id"] for entry in delta["removed"]] == ["removed"]
    assert len(delta["changed"]) == 1

    changed_entry = delta["changed"][0]
    assert changed_entry["scholarship_id"] == "same"
    assert changed_entry["fields_changed"] == {
        "deadline": {"old": "2026-04-15", "new": "2026-04-30"},
        "amount": {
            "old": {"amount_min": 1000.0, "amount_max": 5000.0},
            "new": {"amount_min": 1000.0, "amount_max": 5500.0},
        },
        "title": {"old": "STEM Scholars Award", "new": "STEM Scholars Award - Updated"},
        "eligibility_text": {
            "old": "Open to STEM undergraduates.",
            "new": "Open to STEM majors in the US.",
        },
    }


def test_build_delta_serializes_array_like_values_in_removed_records() -> None:
    prior_df = pd.DataFrame(
        [
            {
                "scholarship_id": "removed",
                "title": "Legacy Scholarship",
                "deadline": "2026-05-01",
                "eligibility_text": "Open to all majors.",
                "amount_min": 500.0,
                "amount_max": 2500.0,
                "states_allowed": ["CA", "NY"],
            },
        ]
    )
    current_df = pd.DataFrame(columns=prior_df.columns)

    delta = build_delta(current_df=current_df, prior_df=prior_df)

    assert delta["removed"] == [
        {
            "scholarship_id": "removed",
            "title": "Legacy Scholarship",
            "deadline": "2026-05-01",
            "eligibility_text": "Open to all majors.",
            "amount_min": 500.0,
            "amount_max": 2500.0,
            "states_allowed": ["CA", "NY"],
        }
    ]
