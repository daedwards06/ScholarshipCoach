from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from src.store import essays, repo, tracker
from src.store.db import connect

TODAY = date(2026, 9, 12)

PROMPT_ONE = "Describe a challenge you overcame."
PROMPT_TWO = "Why computer science?"

TWO_PROMPT_REQUIREMENTS = {"essay_prompts": [PROMPT_ONE, PROMPT_TWO]}


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


@pytest.fixture
def application(conn: sqlite3.Connection, student: repo.Student) -> repo.Application:
    saved, _ = tracker.save_award(
        conn,
        student.student_id,
        "nc-cs-award-2026",
        title="NC CS Award",
        requirements=TWO_PROMPT_REQUIREMENTS,
    )
    return saved


# -- essays: themes, word count, versions -----------------------------------


def test_word_count_counts_whitespace_tokens() -> None:
    assert repo.word_count("") == 0
    assert repo.word_count("   ") == 0
    assert repo.word_count("one two\nthree\tfour  five") == 5


def test_create_essay_stores_theme_and_word_count(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    essay = repo.create_essay(
        conn, student.student_id, "The robot summer", "one two three", "challenge"
    )
    assert (essay.theme, essay.word_count) == ("challenge", 3)

    stored = repo.get_essay(conn, essay.id)
    assert stored is not None
    assert (stored.theme, stored.word_count) == ("challenge", 3)


def test_unknown_theme_falls_back_to_other(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    assert repo.normalize_theme("Why Major") == "why_major"
    assert repo.normalize_theme("sports") == "other"
    assert repo.normalize_theme(None) == "other"

    essay = repo.create_essay(conn, student.student_id, "Untagged", "", "sports")
    assert essay.theme == "other"


def test_editing_the_text_keeps_the_earlier_draft_as_a_row(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    essay = repo.create_essay(conn, student.student_id, "Draft", "one two three")
    assert [version.body for version in repo.list_essay_versions(conn, essay.id)] == [
        "one two three"
    ]

    repo.update_essay(conn, essay.id, body="one two three four five")
    versions = repo.list_essay_versions(conn, essay.id)
    assert [version.word_count for version in versions] == [5, 3]
    assert versions[0].body == "one two three four five"

    current = repo.get_essay(conn, essay.id)
    assert current is not None
    assert current.word_count == 5


def test_retagging_an_essay_does_not_add_a_version(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    essay = repo.create_essay(conn, student.student_id, "Draft", "one two", "other")
    repo.update_essay(conn, essay.id, theme="leadership")

    assert len(repo.list_essay_versions(conn, essay.id)) == 1
    stored = repo.get_essay(conn, essay.id)
    assert stored is not None
    assert stored.theme == "leadership"


def test_deleting_an_essay_takes_its_versions(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    essay = repo.create_essay(conn, student.student_id, "Draft", "one two")
    repo.delete_essay(conn, essay.id)
    assert repo.list_essay_versions(conn, essay.id) == []


# -- prompts and reuse ------------------------------------------------------


def test_award_prompts_start_open(
    conn: sqlite3.Connection, application: repo.Application
) -> None:
    slots = essays.prompt_slots(conn, application.id)
    assert [slot.prompt for slot in slots] == [PROMPT_ONE, PROMPT_TWO]
    assert all(slot.is_open for slot in slots)
    assert essays.open_prompt_slots(conn, application.id) == slots


def test_prompt_slots_ignore_non_essay_checklist_items(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    saved, _ = tracker.save_award(
        conn,
        student.student_id,
        "letters-only",
        requirements={"recommendation_letters": 2, "transcript": True},
    )
    assert essays.prompt_slots(conn, saved.id) == []


def test_a_prompt_with_no_essay_is_an_open_checklist_item(
    conn: sqlite3.Connection, student: repo.Student, application: repo.Application
) -> None:
    essay = repo.create_essay(conn, student.student_id, "The robot summer", "a b c")
    first, second = essays.prompt_slots(conn, application.id)

    essays.use_essay_for_prompt(conn, essay.id, first)

    slots = essays.prompt_slots(conn, application.id)
    assert (slots[0].essay_id, slots[0].essay_title) == (essay.id, "The robot summer")
    assert slots[0].done is True
    assert not slots[0].is_open
    assert [slot.prompt for slot in essays.open_prompt_slots(conn, application.id)] == [
        second.prompt
    ]


def test_unlinking_puts_the_task_back_on_the_checklist(
    conn: sqlite3.Connection, student: repo.Student, application: repo.Application
) -> None:
    essay = repo.create_essay(conn, student.student_id, "The robot summer")
    slot = essays.prompt_slots(conn, application.id)[0]
    essays.use_essay_for_prompt(conn, essay.id, slot)

    linked = essays.prompt_slots(conn, application.id)[0]
    essays.clear_prompt(conn, linked)

    cleared = essays.prompt_slots(conn, application.id)[0]
    assert cleared.is_open
    assert cleared.done is False
    assert repo.list_essay_links(conn, application_id=application.id) == []


def test_reuse_count_rises_with_every_award_the_essay_answers(
    conn: sqlite3.Connection, student: repo.Student, application: repo.Application
) -> None:
    essay = repo.create_essay(conn, student.student_id, "The robot summer", "a b c")
    other, _ = tracker.save_award(
        conn,
        student.student_id,
        "second-award",
        title="Second Award",
        requirements={"essay_prompts": [PROMPT_ONE]},
    )

    assert repo.essay_reuse_counts(conn, student.student_id) == {essay.id: 0}

    essays.use_essay_for_prompt(conn, essay.id, essays.prompt_slots(conn, application.id)[0])
    essays.use_essay_for_prompt(conn, essay.id, essays.prompt_slots(conn, other.id)[0])

    entry = essays.essay_bank(conn, student.student_id)[0]
    assert entry.reuse_count == 2
    assert set(entry.used_by) == {"NC CS Award", "Second Award"}


def test_essay_bank_summarises_themes(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    repo.create_essay(conn, student.student_id, "One", "a", "challenge")
    repo.create_essay(conn, student.student_id, "Two", "b", "challenge")
    repo.create_essay(conn, student.student_id, "Three", "c", "leadership")

    entries = essays.essay_bank(conn, student.student_id)
    assert {entry.theme_label for entry in entries} == {"Challenge", "Leadership"}
    counts = essays.theme_counts(entries)
    assert counts["challenge"] == 2
    assert counts["leadership"] == 1
    assert counts["identity"] == 0


# -- recommendation requests ------------------------------------------------


@pytest.fixture
def recommender(conn: sqlite3.Connection, student: repo.Student) -> repo.Recommender:
    return repo.create_recommender(
        conn, student.student_id, "Ms. Rivera", "AP Physics teacher"
    )


def test_request_lifecycle_stamps_each_date_once(
    conn: sqlite3.Connection,
    recommender: repo.Recommender,
    application: repo.Application,
) -> None:
    request = repo.create_recommendation_request(
        conn, recommender.id, application.id, due_on="2026-09-20"
    )
    assert request.status == "planned"

    asked = tracker.set_request_status(conn, request.id, "asked", today=TODAY)
    assert (asked.status, asked.asked_on, asked.received_on) == ("asked", "2026-09-12", None)

    received = tracker.set_request_status(
        conn, request.id, "received", today=date(2026, 9, 18)
    )
    assert (received.status, received.received_on) == ("received", "2026-09-18")

    # A misclick back to asked and forward again keeps the first dates.
    tracker.set_request_status(conn, request.id, "asked", today=date(2026, 9, 30))
    final = tracker.set_request_status(conn, request.id, "received", today=date(2026, 9, 30))
    assert (final.asked_on, final.received_on) == ("2026-09-12", "2026-09-18")


def test_request_transitions_are_bounded(
    conn: sqlite3.Connection,
    recommender: repo.Recommender,
    application: repo.Application,
) -> None:
    assert tracker.next_request_statuses("planned") == ("planned", "asked", "declined")
    assert tracker.can_transition_request("planned", "planned")
    assert not tracker.can_transition_request("planned", "received")
    assert tracker.normalize_request_status("Asked") == "asked"
    assert tracker.normalize_request_status("posted") == "planned"

    request = repo.create_recommendation_request(conn, recommender.id, application.id)
    with pytest.raises(tracker.TransitionError):
        tracker.set_request_status(conn, request.id, "received", today=TODAY)


def test_set_request_status_rejects_an_unknown_request(conn: sqlite3.Connection) -> None:
    with pytest.raises(tracker.TransitionError):
        tracker.set_request_status(conn, 999, "asked", today=TODAY)


def test_declined_letter_can_be_planned_again(
    conn: sqlite3.Connection,
    recommender: repo.Recommender,
    application: repo.Application,
) -> None:
    request = repo.create_recommendation_request(conn, recommender.id, application.id)
    tracker.set_request_status(conn, request.id, "declined", today=TODAY)
    replanned = tracker.set_request_status(conn, request.id, "planned", today=TODAY)
    assert replanned.status == "planned"


# -- This Week --------------------------------------------------------------


def test_this_week_includes_outstanding_letter_due_dates(
    conn: sqlite3.Connection,
    student: repo.Student,
    recommender: repo.Recommender,
    application: repo.Application,
) -> None:
    request = repo.create_recommendation_request(
        conn, recommender.id, application.id, due_on="2026-09-18"
    )
    tracker.set_request_status(conn, request.id, "asked", today=TODAY)

    letters = [
        item for item in tracker.this_week(conn, student.student_id, today=TODAY)
        if item.kind == "letter"
    ]
    assert len(letters) == 1
    assert letters[0].label == "Letter from Ms. Rivera (asked)"
    assert (letters[0].due_on, letters[0].days_until) == ("2026-09-18", 6)
    assert letters[0].award_title == "NC CS Award"


def test_this_week_drops_letters_once_they_land(
    conn: sqlite3.Connection,
    student: repo.Student,
    recommender: repo.Recommender,
    application: repo.Application,
) -> None:
    request = repo.create_recommendation_request(
        conn, recommender.id, application.id, due_on="2026-09-18"
    )
    tracker.set_request_status(conn, request.id, "asked", today=TODAY)
    tracker.set_request_status(conn, request.id, "received", today=TODAY)

    assert not [
        item for item in tracker.this_week(conn, student.student_id, today=TODAY)
        if item.kind == "letter"
    ]


def test_this_week_keeps_an_overdue_letter(
    conn: sqlite3.Connection,
    student: repo.Student,
    recommender: repo.Recommender,
    application: repo.Application,
) -> None:
    repo.create_recommendation_request(
        conn, recommender.id, application.id, due_on="2026-09-01"
    )
    letters = [
        item for item in tracker.this_week(conn, student.student_id, today=TODAY)
        if item.kind == "letter"
    ]
    assert len(letters) == 1
    assert letters[0].is_overdue
    assert letters[0].label == "Letter from Ms. Rivera (planned)"


def test_this_week_ignores_a_letter_beyond_the_horizon(
    conn: sqlite3.Connection,
    student: repo.Student,
    recommender: repo.Recommender,
    application: repo.Application,
) -> None:
    repo.create_recommendation_request(
        conn, recommender.id, application.id, due_on="2026-12-01"
    )
    assert not [
        item for item in tracker.this_week(conn, student.student_id, today=TODAY)
        if item.kind == "letter"
    ]
