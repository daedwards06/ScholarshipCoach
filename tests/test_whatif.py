from __future__ import annotations

from datetime import date

import pandas as pd

from src.rank.stage1_eligibility import StudentProfile
from src.rank.whatif import apply_overrides, whatif_eligibility

_TODAY = date(2026, 9, 12)


def _profile(**overrides: object) -> StudentProfile:
    values: dict[str, object] = {
        "gpa": 3.25,
        "state": "NC",
        "grade_level": "10",
        "today": _TODAY,
        "sat": None,
        "act": None,
        "service_hours": 40,
        "first_gen": False,
        "financial_need": False,
    }
    values.update(overrides)
    return StudentProfile(**values)  # type: ignore[arg-type]


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "scholarship_id": "award",
        "title": "Award",
        "deadline": date(2026, 11, 1),
        "status": "open",
        "cycle": {"recurring": None, "opens_month": None, "deadline_month": None},
        "grade_levels": [],
        "states_allowed": [],
        "majors_allowed": [],
        "amount_min": 1000.0,
        "amount_max": 5000.0,
        "min_gpa": None,
        "min_test_scores": {},
        "need_based": None,
        "first_gen_only": None,
    }
    row.update(overrides)
    return row


def _frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def test_gpa_override_unlocks_award_above_threshold() -> None:
    df = _frame(
        _row(scholarship_id="honors", title="Honors Award", min_gpa=3.5),
        _row(scholarship_id="open", title="Open Award"),
    )

    summary = whatif_eligibility(df, _profile(), {"gpa": 3.5})

    assert summary.overrides == {"gpa": 3.5}
    assert [award.scholarship_id for award in summary.newly_eligible] == ["honors"]
    assert summary.newly_eligible[0].reasons == ["GPA_BELOW_MIN"]
    assert summary.newly_eligible[0].reason_text == "GPA below the minimum"
    assert summary.newly_ineligible == []


def test_gpa_override_below_threshold_unlocks_nothing() -> None:
    df = _frame(_row(scholarship_id="honors", min_gpa=3.5))

    summary = whatif_eligibility(df, _profile(), {"gpa": 3.4})

    assert summary.newly_eligible == []
    assert summary.is_noop


def test_test_score_override_unlocks_award_and_totals_dollars_by_bucket() -> None:
    df = _frame(
        _row(
            scholarship_id="merit",
            title="Merit Award",
            min_test_scores={"sat": 1200, "act": None},
            amount_max=7500.0,
        ),
        _row(
            scholarship_id="senior_merit",
            title="Senior Merit Award",
            min_test_scores={"sat": 1200, "act": None},
            amount_max=2500.0,
            grade_levels=["12"],
        ),
    )

    summary = whatif_eligibility(df, _profile(sat=1100), {"sat": 1250})

    assert {award.scholarship_id for award in summary.newly_eligible} == {
        "merit",
        "senior_merit",
    }
    assert summary.dollars_by_bucket == {"now": 7500.0, "senior_year": 2500.0}
    assert summary.dollars_unlocked == 10000.0
    assert all(
        award.reasons == ["TEST_SCORE_BELOW_MIN"] for award in summary.newly_eligible
    )


def test_act_override_clears_an_award_stating_both_minimums() -> None:
    df = _frame(_row(scholarship_id="merit", min_test_scores={"sat": 1200, "act": 26}))

    summary = whatif_eligibility(df, _profile(sat=1100), {"act": 28})

    assert [award.scholarship_id for award in summary.newly_eligible] == ["merit"]


def test_no_op_override_reports_no_change() -> None:
    df = _frame(_row(scholarship_id="honors", min_gpa=3.5), _row(scholarship_id="open"))
    profile = _profile()

    summary = whatif_eligibility(df, profile, {"gpa": profile.gpa, "sat": None})

    assert summary.overrides == {}
    assert summary.newly_eligible == []
    assert summary.newly_ineligible == []
    assert summary.dollars_by_bucket == {}
    assert summary.dollars_unlocked == 0.0
    assert summary.is_noop


def test_flag_overrides_open_restricted_awards() -> None:
    df = _frame(
        _row(scholarship_id="need", need_based=True, amount_max=3000.0),
        _row(scholarship_id="first_gen", first_gen_only=True, amount_max=2000.0),
    )

    summary = whatif_eligibility(
        df, _profile(), {"financial_need": True, "first_gen": True}
    )

    assert {award.scholarship_id for award in summary.newly_eligible} == {
        "need",
        "first_gen",
    }
    assert summary.dollars_by_bucket == {"now": 5000.0}


def test_turning_a_flag_off_closes_an_award() -> None:
    df = _frame(_row(scholarship_id="need", need_based=True))

    summary = whatif_eligibility(df, _profile(financial_need=True), {"financial_need": False})

    assert summary.newly_eligible == []
    assert [award.scholarship_id for award in summary.newly_ineligible] == ["need"]
    assert summary.newly_ineligible[0].reasons == ["NEED_BASED_NOT_MET"]


def test_apply_overrides_leaves_the_original_profile_untouched() -> None:
    profile = _profile()

    updated = apply_overrides(profile, {"gpa": 3.9, "unknown_field": 1, "sat": None})

    assert updated.gpa == 3.9
    assert profile.gpa == 3.25
    assert updated.sat is None
