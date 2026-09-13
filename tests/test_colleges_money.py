from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.store import money, repo
from src.store.db import connect


@pytest.fixture
def conn(tmp_path: Path):
    connection = connect(tmp_path / "coach.db")
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def student(conn: sqlite3.Connection) -> repo.Student:
    return repo.upsert_student(conn, "student_1", "Test Student")


def _outcome(
    application_id: int,
    result: str = "won",
    amount: float | None = None,
    decided_on: str | None = None,
    renewal_terms: str = "",
) -> repo.Outcome:
    return repo.Outcome(
        id=application_id,
        application_id=application_id,
        result=result,
        amount_awarded=amount,
        renewal_terms=renewal_terms,
        decided_on=decided_on,
    )


def _college(college_id: int = 1, **kwargs: object) -> repo.College:
    fields: dict[str, object] = {
        "id": college_id,
        "student_id": "student_1",
        "name": "NC State University",
    }
    fields.update(kwargs)
    return repo.College(**fields)  # type: ignore[arg-type]


# -- college fields round-trip ----------------------------------------------


def test_college_money_fields_round_trip(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    college = repo.create_college(
        conn,
        student.student_id,
        "NC State University",
        cost_of_attendance=28000.0,
        in_state=True,
        net_price_estimate=15500.0,
        merit_aid_notes="Park Scholarship, separate application",
        outside_award_policy="Reduces loans first, then institutional grant",
        deadline_type="Early Action",
    )
    assert college.deadline_type == "early_action"

    stored = repo.get_college(conn, college.id)
    assert stored is not None
    assert stored.in_state is True
    assert stored.net_price_estimate == 15500.0
    assert stored.merit_aid_notes == "Park Scholarship, separate application"
    assert stored.outside_award_policy == "Reduces loans first, then institutional grant"
    assert stored.deadline_type == "early_action"

    assert repo.update_college(conn, college.id, in_state=False, deadline_type="nonsense")
    updated = repo.get_college(conn, college.id)
    assert updated is not None
    assert updated.in_state is False
    assert updated.deadline_type == ""


def test_net_price_prefers_the_hand_entered_estimate() -> None:
    both = _college(cost_of_attendance=28000.0, aid_offered=12000.0, net_price_estimate=9000.0)
    assert repo.net_price(both) == 9000.0

    fallback = _college(cost_of_attendance=28000.0, aid_offered=12000.0)
    assert repo.net_price(fallback) == 16000.0

    assert repo.net_price(_college(cost_of_attendance=28000.0)) is None


# -- summary math -----------------------------------------------------------


def test_total_won_counts_only_wins_with_amounts() -> None:
    outcomes = [
        _outcome(1, "won", 2500.0),
        _outcome(2, "lost", 5000.0),
        _outcome(3, "won", None),
        _outcome(4, "pending", 1000.0),
        _outcome(5, "won", 1500.0),
    ]
    assert money.total_won(outcomes) == 4000.0

    summary = money.money_summary([], outcomes)
    assert summary.total_won == 4000.0
    # The undecided-amount win still counts as a win.
    assert summary.award_count == 3


def test_won_by_year_uses_school_years_and_sorts_undated_last() -> None:
    rows = money.won_by_year(
        [
            _outcome(1, "won", 1000.0, decided_on="2026-08-15"),
            _outcome(2, "won", 500.0, decided_on="2026-03-01"),
            _outcome(3, "won", 250.0, decided_on="2026-06-30"),
            _outcome(4, "won", 750.0, decided_on=""),
            _outcome(5, "lost", 9999.0, decided_on="2026-03-01"),
        ]
    )
    assert [(row.label, row.total, row.count) for row in rows] == [
        ("2025-26", 750.0, 2),
        ("2026-27", 1000.0, 1),
        (money.UNDATED_YEAR_LABEL, 750.0, 1),
    ]


def test_renewal_conditions_list_only_wins_with_terms() -> None:
    rows = money.renewal_conditions(
        [
            _outcome(1, "won", 2500.0, renewal_terms="Renewable 4 years at 3.0 GPA"),
            _outcome(2, "won", 1000.0, renewal_terms="   "),
            _outcome(3, "lost", 500.0, renewal_terms="Renewable"),
        ],
        titles={1: "NC Engineering Award"},
    )
    assert [(row.award_title, row.terms) for row in rows] == [
        ("NC Engineering Award", "Renewable 4 years at 3.0 GPA")
    ]


def test_renewal_conditions_fall_back_to_the_application_id() -> None:
    rows = money.renewal_conditions([_outcome(7, "won", 100.0, renewal_terms="3.0 GPA")])
    assert rows[0].award_title == "Application 7"


def test_net_price_minus_won_per_college() -> None:
    summary = money.money_summary(
        [
            _college(1, name="NC State University", net_price_estimate=15500.0, in_state=True),
            _college(2, name="Duke University", cost_of_attendance=85000.0, aid_offered=40000.0),
            _college(3, name="Unknown Cost College"),
        ],
        [_outcome(1, "won", 4000.0), _outcome(2, "won", 1000.0)],
    )
    assert summary.total_won == 5000.0
    assert [(row.name, row.remaining) for row in summary.colleges] == [
        ("NC State University", 10500.0),
        ("Duke University", 40000.0),
        ("Unknown Cost College", None),
    ]
    assert summary.colleges[0].in_state is True
    assert summary.colleges[1].in_state is False
    # The hand-entered estimate stays separate from the price derived from aid,
    # so the form edits the column the family typed into.
    assert summary.colleges[0].net_price_estimate == 15500.0
    assert summary.colleges[1].net_price_estimate is None
    assert summary.colleges[1].net_price == 45000.0


def test_summary_of_an_empty_family_is_all_zeros() -> None:
    summary = money.money_summary([], [])
    assert summary.total_won == 0.0
    assert summary.award_count == 0
    assert summary.by_year == []
    assert summary.renewals == []
    assert summary.colleges == []


def test_summary_reads_a_real_database(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    application = repo.create_application(
        conn, student.student_id, "nc-cs-award-2026", title="NC CS Award"
    )
    repo.set_outcome(
        conn,
        application.id,
        result="won",
        amount_awarded=3000.0,
        renewal_terms="Renewable at 3.0 GPA",
        decided_on="2026-04-01",
    )
    repo.create_college(
        conn, student.student_id, "NC State University", net_price_estimate=15500.0
    )

    summary = money.money_summary(
        repo.list_colleges(conn, student.student_id),
        repo.list_outcomes(conn, student.student_id),
        {application.id: application.title},
    )
    assert summary.total_won == 3000.0
    assert [row.label for row in summary.by_year] == ["2025-26"]
    assert [row.award_title for row in summary.renewals] == ["NC CS Award"]
    assert summary.colleges[0].remaining == 12500.0


# -- the view is parent-only ------------------------------------------------


def test_colleges_money_is_a_parent_section() -> None:
    from app import modes

    assert "colleges_money" in modes.PARENT_ONLY_SECTIONS
    assert not modes.can_view("student", "colleges_money")
    assert modes.can_view("parent", "colleges_money")
    assert modes.resolve_section("colleges_money", "student") != "colleges_money"
