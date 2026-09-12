from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from src.store import repo, tracker
from src.store.db import connect

TODAY = date(2026, 9, 12)


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


FULL_REQUIREMENTS = {
    "essay": True,
    "essay_prompts": ["Describe a challenge you overcame.", "Why computer science?"],
    "recommendation_letters": 2,
    "transcript": True,
    "fafsa": True,
    "video_or_portfolio": True,
    "interview": True,
}


# -- checklist generation ---------------------------------------------------


def test_checklist_covers_every_stated_requirement() -> None:
    labels = tracker.checklist_labels(FULL_REQUIREMENTS)
    assert labels == [
        "Essay 1: Describe a challenge you overcame.",
        "Essay 2: Why computer science?",
        "Recommendation letter 1 of 2",
        "Recommendation letter 2 of 2",
        "Request transcript",
        "Complete the FAFSA",
        "Record video or assemble portfolio",
        "Prepare for the interview",
    ]


def test_checklist_ignores_unknown_and_false_requirements() -> None:
    labels = tracker.checklist_labels(
        {
            "essay": None,
            "essay_prompts": [],
            "recommendation_letters": None,
            "transcript": False,
            "fafsa": None,
            "video_or_portfolio": False,
            "interview": None,
        }
    )
    assert labels == []


def test_checklist_handles_essay_without_a_prompt() -> None:
    assert tracker.checklist_labels({"essay": True}) == ["Write the essay"]


def test_checklist_drops_the_of_n_suffix_for_a_single_letter() -> None:
    assert tracker.checklist_labels({"recommendation_letters": 1}) == ["Recommendation letter"]


def test_checklist_truncates_a_long_prompt() -> None:
    prompt = "Tell us about a time you led a team " * 10
    (label,) = tracker.checklist_labels({"essay_prompts": [prompt]})
    assert label.endswith("…")
    assert len(label) <= len("Essay: ") + 80


def test_checklist_of_a_missing_requirements_object_is_empty() -> None:
    assert tracker.checklist_labels(None) == []


# -- saving -----------------------------------------------------------------


def _save(conn: sqlite3.Connection, student_id: str, **overrides: object):
    payload: dict = {
        "title": "STEM Award",
        "source_url": "https://example.org/stem",
        "deadline": "2026-09-20",
        "requirements": FULL_REQUIREMENTS,
    }
    payload.update(overrides)
    return tracker.save_award(conn, student_id, "stem-award", **payload)


def test_save_creates_the_application_and_its_checklist(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    application, created = _save(conn, student.student_id)

    assert created is True
    assert application.status == "saved"
    assert application.title == "STEM Award"
    assert application.deadline == "2026-09-20"
    assert application.submitted_on is None

    items = repo.list_checklist_items(conn, application.id)
    assert [item.label for item in items] == tracker.checklist_labels(FULL_REQUIREMENTS)
    assert [item.position for item in items] == list(range(len(items)))
    assert all(item.done is False for item in items)


def test_saving_twice_returns_the_same_application_and_keeps_progress(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    application, _ = _save(conn, student.student_id)
    first_item = repo.list_checklist_items(conn, application.id)[0]
    repo.update_checklist_item(conn, first_item.id, done=True)

    again, created = _save(conn, student.student_id, title="Renamed")

    assert created is False
    assert again.id == application.id
    assert again.title == "STEM Award"
    items = repo.list_checklist_items(conn, application.id)
    assert len(items) == len(tracker.checklist_labels(FULL_REQUIREMENTS))
    assert items[0].done is True


def test_free_form_items_append_after_the_template(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    application, _ = _save(conn, student.student_id)
    added = tracker.add_checklist_item(conn, application.id, "Order a headshot", "2026-09-15")

    items = repo.list_checklist_items(conn, application.id)
    assert items[-1].id == added.id
    assert items[-1].label == "Order a headshot"
    assert items[-1].due_on == "2026-09-15"


def test_checklist_progress_counts_done_over_total(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    application, _ = _save(conn, student.student_id)
    items = repo.list_checklist_items(conn, application.id)
    repo.update_checklist_item(conn, items[0].id, done=True)

    assert tracker.checklist_progress(repo.list_checklist_items(conn, application.id)) == (
        1,
        len(items),
    )


# -- status transitions -----------------------------------------------------


@pytest.mark.parametrize(
    ("current", "target"),
    [
        ("saved", "planning"),
        ("saved", "skipped"),
        ("planning", "in_progress"),
        ("in_progress", "submitted"),
        ("submitted", "won"),
        ("submitted", "lost"),
        ("won", "submitted"),
        ("skipped", "saved"),
        ("saved", "saved"),
    ],
)
def test_allowed_transitions(current: str, target: str) -> None:
    assert tracker.can_transition(current, target) is True


@pytest.mark.parametrize(
    ("current", "target"),
    [
        ("saved", "won"),
        ("saved", "lost"),
        ("planning", "won"),
        ("in_progress", "lost"),
        ("won", "lost"),
        ("skipped", "submitted"),
    ],
)
def test_rejected_transitions(current: str, target: str) -> None:
    assert tracker.can_transition(current, target) is False


def test_next_statuses_offers_the_current_one_first() -> None:
    assert tracker.next_statuses("in_progress")[0] == "in_progress"
    assert set(tracker.next_statuses("in_progress")[1:]) == set(
        tracker.ALLOWED_TRANSITIONS["in_progress"]
    )


def test_normalize_status_falls_back_to_saved() -> None:
    assert tracker.normalize_status("In Progress") == "in_progress"
    assert tracker.normalize_status("nonsense") == "saved"
    assert tracker.normalize_status(None) == "saved"


def test_submitting_stamps_the_date_once(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    application, _ = _save(conn, student.student_id)
    submitted = tracker.set_status(conn, application.id, "submitted", today=TODAY)
    assert submitted.submitted_on == "2026-09-12"

    back = tracker.set_status(conn, application.id, "in_progress", today=TODAY)
    again = tracker.set_status(conn, back.id, "submitted", today=date(2026, 10, 1))
    assert again.submitted_on == "2026-09-12"


def test_an_illegal_transition_raises_and_changes_nothing(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    application, _ = _save(conn, student.student_id)
    with pytest.raises(tracker.TransitionError):
        tracker.set_status(conn, application.id, "won", today=TODAY)

    stored = repo.get_application(conn, application.id)
    assert stored is not None
    assert stored.status == "saved"
    assert repo.get_outcome(conn, application.id) is None


def test_winning_writes_an_outcome_row(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    application, _ = _save(conn, student.student_id)
    tracker.set_status(conn, application.id, "submitted", today=TODAY)
    tracker.set_status(conn, application.id, "won", today=TODAY)

    outcome = repo.get_outcome(conn, application.id)
    assert outcome is not None
    assert outcome.result == "won"
    assert outcome.decided_on == "2026-09-12"


def test_record_outcome_walks_an_unsubmitted_award_through_submitted(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    application, _ = _save(conn, student.student_id)
    outcome = tracker.record_outcome(
        conn,
        application.id,
        "won",
        amount_awarded=2500.0,
        paid_to="School",
        renewal_terms="Renewable at 3.0 GPA",
        today=TODAY,
    )

    assert outcome.amount_awarded == 2500.0
    assert outcome.paid_to == "School"
    assert outcome.renewal_terms == "Renewable at 3.0 GPA"

    stored = repo.get_application(conn, application.id)
    assert stored is not None
    assert stored.status == "won"
    assert stored.submitted_on == "2026-09-12"


def test_setting_the_status_it_already_has_is_a_no_op(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    application, _ = _save(conn, student.student_id)
    unchanged = tracker.set_status(conn, application.id, "saved", today=TODAY)

    assert unchanged == application


@pytest.mark.parametrize("call", ["set_status", "record_outcome"])
def test_an_unknown_application_raises(conn: sqlite3.Connection, call: str) -> None:
    with pytest.raises(tracker.TransitionError):
        if call == "set_status":
            tracker.set_status(conn, 999, "planning", today=TODAY)
        else:
            tracker.record_outcome(conn, 999, "won", today=TODAY)


def test_record_outcome_rejects_a_non_outcome_result(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    application, _ = _save(conn, student.student_id)
    with pytest.raises(tracker.TransitionError):
        tracker.record_outcome(conn, application.id, "submitted", today=TODAY)


# -- This Week --------------------------------------------------------------


def test_this_week_returns_deadlines_and_dated_items_in_the_window(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    near, _ = tracker.save_award(
        conn,
        student.student_id,
        "near-award",
        title="Near Award",
        deadline="2026-09-20",
        requirements={"essay": True},
    )
    tracker.save_award(
        conn,
        student.student_id,
        "far-award",
        title="Far Award",
        deadline="2026-12-01",
        requirements={"essay": True},
    )
    essay_item = repo.list_checklist_items(conn, near.id)[0]
    repo.update_checklist_item(conn, essay_item.id, due_on="2026-09-15")

    due = tracker.this_week(conn, student.student_id, today=TODAY)

    assert [(item.kind, item.due_on) for item in due] == [
        ("checklist", "2026-09-15"),
        ("deadline", "2026-09-20"),
    ]
    assert due[0].award_title == "Near Award"
    assert due[0].label == "Write the essay"
    assert due[1].days_until == 8


def test_this_week_keeps_overdue_work_and_drops_done_items(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    application, _ = tracker.save_award(
        conn,
        student.student_id,
        "overdue-award",
        title="Overdue Award",
        deadline="2026-09-05",
        requirements={"recommendation_letters": 2},
    )
    first, second = repo.list_checklist_items(conn, application.id)
    repo.update_checklist_item(conn, first.id, due_on="2026-09-01")
    repo.update_checklist_item(conn, second.id, due_on="2026-09-02", done=True)

    due = tracker.this_week(conn, student.student_id, today=TODAY)

    assert [(item.kind, item.due_on) for item in due] == [
        ("checklist", "2026-09-01"),
        ("deadline", "2026-09-05"),
    ]
    assert due[0].is_overdue is True
    assert due[0].days_until == -11


def test_this_week_ignores_undated_items_and_closed_applications(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    application, _ = _save(conn, student.student_id)
    assert tracker.this_week(conn, student.student_id, today=TODAY)

    tracker.set_status(conn, application.id, "submitted", today=TODAY)
    assert tracker.this_week(conn, student.student_id, today=TODAY) == []


def test_this_week_window_is_configurable(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    _save(conn, student.student_id, deadline="2026-09-20")

    assert tracker.this_week(conn, student.student_id, today=TODAY, within_days=3) == []
    assert len(tracker.this_week(conn, student.student_id, today=TODAY, within_days=14)) == 1


def test_this_week_survives_an_unparseable_deadline(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    _save(conn, student.student_id, deadline="rolling")
    assert tracker.this_week(conn, student.student_id, today=TODAY) == []
