from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.rank.stage1_eligibility import StudentProfile
from src.rank.stage3_rerank import rerank_stage3
from src.rank.timeline import (
    TIMELINE_BUCKETS,
    classify_timeline,
    project_next_deadline,
)

_TODAY = date(2026, 9, 12)


def _profile(grade_level: str | None = "10") -> StudentProfile:
    return StudentProfile(grade_level=grade_level, today=_TODAY)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "scholarship_id": "award",
        "deadline": None,
        "cycle": {"recurring": None, "opens_month": None, "deadline_month": None},
        "status": "unknown",
        "grade_levels": [],
    }
    row.update(overrides)
    return row


def _classify(**overrides: object) -> pd.Series:
    df = classify_timeline(pd.DataFrame([_row(**overrides)]), _profile(), today=_TODAY)
    return df.iloc[0]


def test_open_future_deadline_is_now() -> None:
    result = _classify(deadline=date(2026, 11, 1), status="open")

    assert result["timeline_bucket"] == "now"
    assert pd.isna(result["projected_deadline"])


def test_recurring_past_deadline_is_next_cycle_with_projection() -> None:
    result = _classify(
        deadline=date(2026, 4, 1),
        cycle={"recurring": True, "opens_month": None, "deadline_month": 4},
        status="closed",
    )

    assert result["timeline_bucket"] == "next_cycle"
    assert result["projected_deadline"].date() == date(2027, 4, 1)


def test_seniors_only_award_for_sophomore_is_senior_year() -> None:
    result = _classify(deadline=date(2027, 3, 1), status="open", grade_levels=["12"])

    assert result["timeline_bucket"] == "senior_year"


def test_non_recurring_closed_award_is_expired() -> None:
    result = _classify(
        deadline=date(2026, 4, 1),
        cycle={"recurring": False, "opens_month": None, "deadline_month": 4},
        status="closed",
    )

    assert result["timeline_bucket"] == "expired"
    assert pd.isna(result["projected_deadline"])


def test_student_ageing_out_before_next_cycle_is_not_applicable() -> None:
    # A grade-12-only award whose next cycle lands after this senior graduates.
    df = classify_timeline(
        pd.DataFrame(
            [
                _row(
                    deadline=date(2026, 8, 1),
                    cycle={"recurring": True, "opens_month": None, "deadline_month": 8},
                    grade_levels=["12"],
                )
            ]
        ),
        _profile(grade_level="12"),
        today=_TODAY,
    )

    assert df.loc[0, "timeline_bucket"] == "not_applicable"


def test_undated_unknown_award_needs_a_date() -> None:
    result = _classify()

    assert result["timeline_bucket"] == "needs_date"
    assert pd.isna(result["projected_deadline"])


def test_open_award_with_no_date_stays_in_now() -> None:
    # The sponsor saying it is open is the date the catalog has; keep it actionable.
    result = _classify(status="open")

    assert result["timeline_bucket"] == "now"


def test_cycle_month_with_no_deadline_projects_and_stays_out_of_needs_date() -> None:
    result = _classify(cycle={"recurring": True, "opens_month": None, "deadline_month": 5})

    assert result["timeline_bucket"] == "now"
    assert result["projected_deadline"].date() == date(2027, 5, 1)


def test_missing_status_is_treated_as_unknown() -> None:
    result = _classify(status=None)

    assert result["timeline_bucket"] == "needs_date"


def test_a_dated_award_is_never_needs_date() -> None:
    result = _classify(deadline=date(2026, 11, 1))

    assert result["timeline_bucket"] == "now"


def test_needs_date_is_excluded_from_the_default_rerank() -> None:
    df = pd.DataFrame(
        [
            {
                "scholarship_id": "dated",
                "stage2_score": 0.5,
                "deadline": date(2026, 11, 1),
                "timeline_bucket": "now",
                "projected_deadline": pd.NaT,
            },
            {
                "scholarship_id": "undated",
                "stage2_score": 0.9,
                "deadline": None,
                "timeline_bucket": "needs_date",
                "projected_deadline": pd.NaT,
            },
        ]
    )

    assert rerank_stage3(df, today=_TODAY)["scholarship_id"].tolist() == ["dated"]
    assert rerank_stage3(df, today=_TODAY, timeline_bucket="needs_date")[
        "scholarship_id"
    ].tolist() == ["undated"]


def test_every_bucket_value_is_declared() -> None:
    df = classify_timeline(
        pd.DataFrame(
            [
                _row(scholarship_id="open", deadline=date(2026, 11, 1)),
                _row(
                    scholarship_id="recurring",
                    deadline=date(2026, 4, 1),
                    cycle={"recurring": True, "opens_month": None, "deadline_month": 4},
                ),
                _row(scholarship_id="later", deadline=date(2027, 3, 1), grade_levels=["12"]),
                _row(
                    scholarship_id="gone",
                    deadline=date(2026, 4, 1),
                    cycle={"recurring": False, "opens_month": None, "deadline_month": 4},
                ),
                _row(scholarship_id="undated"),
            ]
        ),
        _profile(),
        today=_TODAY,
    )

    assert set(df["timeline_bucket"]) <= set(TIMELINE_BUCKETS)
    assert df["timeline_bucket"].tolist() == [
        "now",
        "next_cycle",
        "senior_year",
        "expired",
        "needs_date",
    ]


def test_unknown_grade_level_keeps_the_deadline_bucket() -> None:
    df = classify_timeline(
        pd.DataFrame([_row(deadline=date(2027, 3, 1), grade_levels=["12"])]),
        _profile(grade_level=None),
        today=_TODAY,
    )

    assert df.loc[0, "timeline_bucket"] == "now"


def test_classify_timeline_on_empty_frame_adds_columns() -> None:
    df = classify_timeline(pd.DataFrame(columns=["scholarship_id"]), _profile(), today=_TODAY)

    assert "timeline_bucket" in df.columns
    assert "projected_deadline" in df.columns
    assert df.empty


@pytest.mark.parametrize(
    "today,deadline,deadline_month,expected",
    [
        (date(2026, 12, 1), date(2026, 3, 15), 3, date(2027, 3, 15)),
        (date(2026, 12, 1), date(2026, 12, 31), 12, date(2026, 12, 31)),
        (date(2026, 1, 5), date(2024, 2, 29), 2, date(2026, 2, 28)),
        (date(2026, 9, 12), None, 5, date(2027, 5, 1)),
        (date(2026, 9, 12), date(2026, 5, 20), None, date(2027, 5, 20)),
        (date(2026, 9, 12), None, None, None),
    ],
)
def test_project_next_deadline(
    today: date, deadline: date | None, deadline_month: int | None, expected: date | None
) -> None:
    assert project_next_deadline(deadline, deadline_month, today) == expected


def test_rerank_stage3_ranks_the_now_bucket_by_default() -> None:
    df = pd.DataFrame(
        [
            {
                "scholarship_id": "now-row",
                "stage2_score": 0.5,
                "deadline": date(2026, 11, 1),
                "timeline_bucket": "now",
                "projected_deadline": pd.NaT,
            },
            {
                "scholarship_id": "next-row",
                "stage2_score": 0.9,
                "deadline": date(2026, 4, 1),
                "timeline_bucket": "next_cycle",
                "projected_deadline": pd.Timestamp("2027-04-01"),
            },
        ]
    )

    ranked_df = rerank_stage3(df, today=_TODAY)
    assert ranked_df["scholarship_id"].tolist() == ["now-row"]

    all_df = rerank_stage3(df, today=_TODAY, timeline_bucket=None)
    assert sorted(all_df["scholarship_id"].tolist()) == ["next-row", "now-row"]

    next_only_df = rerank_stage3(df, today=_TODAY, timeline_bucket="next_cycle")
    assert next_only_df["scholarship_id"].tolist() == ["next-row"]


def test_urgency_uses_the_projected_deadline_when_the_listed_one_has_passed() -> None:
    df = pd.DataFrame(
        [
            {
                "scholarship_id": "recurring",
                "stage2_score": 0.5,
                "deadline": date(2026, 4, 1),
                "timeline_bucket": "next_cycle",
                "projected_deadline": pd.Timestamp("2026-10-01"),
            }
        ]
    )

    ranked_df = rerank_stage3(df, today=_TODAY, timeline_bucket=None)

    assert ranked_df.loc[0, "days_to_deadline"] == 19.0
    assert ranked_df.loc[0, "urgency_boost"] > 0.0
