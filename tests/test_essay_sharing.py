from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from app.sections import recommenders as recommenders_page
from app.sections import todos as todos_page
from src.store import essays, repo, todos, tracker
from src.store.db import connect

TODAY = date(2026, 10, 10)


@pytest.fixture
def conn(tmp_path: Path):
    connection = connect(tmp_path / "coach.db")
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def sid(conn: sqlite3.Connection) -> str:
    return repo.upsert_student(conn, "student_1", "Test Student").student_id


@pytest.fixture
def drafts(conn: sqlite3.Connection, sid: str) -> tuple[repo.Essay, repo.Essay]:
    shared = repo.create_essay(conn, sid, "Robot summer", "one two", "challenge")
    private = repo.create_essay(conn, sid, "Why CS", "three four", "why_major")
    repo.set_essay_shared(conn, shared.id, True)
    return shared, private


# -- essays: what a parent may read -----------------------------------------


def test_parent_lists_only_shared_essays(
    conn: sqlite3.Connection, sid: str, drafts: tuple[repo.Essay, repo.Essay]
) -> None:
    shared, private = drafts
    parent_view = essays.essay_bank(conn, sid, shared_only=True)
    assert [entry.essay.id for entry in parent_view] == [shared.id]
    assert {entry.essay.id for entry in essays.essay_bank(conn, sid)} == {shared.id, private.id}
    assert essays.unshared_count(conn, sid) == 1


def test_parent_cannot_open_an_unshared_essay_by_id(
    conn: sqlite3.Connection, drafts: tuple[repo.Essay, repo.Essay]
) -> None:
    shared, private = drafts
    assert essays.readable_essay(conn, private.id, shared_only=True) is None
    opened = essays.readable_essay(conn, shared.id, shared_only=True)
    assert opened is not None and opened.id == shared.id
    assert essays.readable_essay(conn, private.id, shared_only=False) is not None
    assert essays.readable_essay(conn, 9999, shared_only=False) is None


def test_sharing_toggles_both_ways(
    conn: sqlite3.Connection, sid: str, drafts: tuple[repo.Essay, repo.Essay]
) -> None:
    shared, private = drafts
    repo.set_essay_shared(conn, private.id, True)
    repo.set_essay_shared(conn, shared.id, False)
    assert [entry.essay.id for entry in essays.essay_bank(conn, sid, shared_only=True)] == [
        private.id
    ]
    assert essays.readable_essay(conn, shared.id, shared_only=True) is None


def test_suggest_a_change_is_her_to_do_linking_the_essay(
    conn: sqlite3.Connection, sid: str, drafts: tuple[repo.Essay, repo.Essay]
) -> None:
    shared, _ = drafts
    todo = todos.suggest_essay_change(
        conn, shared, "  Say what you'd build next. ", created_by="Dana Lee"
    )
    assert (todo.assignee, todo.created_role, todo.notes) == (
        "student",
        "parent",
        "Say what you'd build next.",
    )
    assert shared.title in todo.title
    row = todos.row_from_todo(todo)
    assert todos_page.source_page(row) == "essays"
    assert todos.visible_to(todo, "student")


def test_deleting_an_essay_drops_its_open_suggestions(
    conn: sqlite3.Connection, sid: str, drafts: tuple[repo.Essay, repo.Essay]
) -> None:
    shared, private = drafts
    done = todos.suggest_essay_change(conn, shared, "Tighten the opening.")
    todos.set_todo_done(conn, done.id, True, today=TODAY)
    todos.suggest_essay_change(conn, shared, "Name the robot.")
    other = todos.suggest_essay_change(conn, private, "Keep this one.")
    unrelated = todos.create_todo(conn, sid, "Call the counselor")

    assert essays.delete_essay(conn, shared.id)
    assert repo.get_essay(conn, shared.id) is None
    assert {todo.id for todo in todos.list_todos(conn, sid)} == {done.id, other.id, unrelated.id}


# -- recommenders: letter counts and "Ask ... by ..." -----------------------


@pytest.fixture
def request_row(conn: sqlite3.Connection, sid: str) -> repo.RecommendationRequest:
    recommender = repo.create_recommender(conn, sid, "Ms. Rivera", "AP Physics teacher")
    application = repo.create_application(
        conn, sid, "award_1", status="planning", title="Robotics Award"
    )
    return repo.create_recommendation_request(
        conn, recommender.id, application.id, due_on="2026-10-20"
    )


def test_letter_badge_counts_in_and_waiting() -> None:
    def request(status: str) -> repo.RecommendationRequest:
        return repo.RecommendationRequest(id=1, recommender_id=1, application_id=1, status=status)

    assert recommenders_page.letter_badge([]) == ("No letters yet", "gray")
    assert recommenders_page.letter_badge(
        [request("received"), request("received"), request("asked"), request("declined")]
    ) == ("2 in / 1 waiting", "blue")
    assert recommenders_page.letter_badge([request("received")]) == ("1 in / 0 waiting", "green")


def test_dated_request_puts_an_ask_on_her_to_dos(
    conn: sqlite3.Connection, sid: str, request_row: repo.RecommendationRequest
) -> None:
    todo = todos.sync_letter_todo(conn, sid, request_row.id, created_role="parent", today=TODAY)
    assert todo is not None
    assert (todo.title, todo.due_on, todo.assignee) == (
        "Ask Ms. Rivera by Oct 20",
        "2026-10-20",
        "student",
    )
    assert todo.notes == "Letter for Robotics Award"
    # Re-syncing updates in place rather than adding a second row.
    repo.update_recommendation_request(conn, request_row.id, due_on="2026-10-25")
    again = todos.sync_letter_todo(conn, sid, request_row.id, today=TODAY)
    assert again is not None and again.id == todo.id and again.title.endswith("Oct 25")
    assert len(todos.list_todos(conn, sid)) == 1


def test_the_ask_replaces_the_live_letter_row_until_ticked(
    conn: sqlite3.Connection, sid: str, request_row: repo.RecommendationRequest
) -> None:
    todos.sync_letter_todo(conn, sid, request_row.id, today=TODAY)
    rows = todos.assemble_rows(conn, sid, "student", today=TODAY)
    assert [(row.kind, row.title) for row in rows] == [("todo", "Ask Ms. Rivera by Oct 20")]

    todos.set_row_done(conn, rows[0], True, done_by="Her", today=TODAY)
    request = repo.get_recommendation_request(conn, request_row.id)
    assert request is not None and request.status == "asked"
    kinds = [row.kind for row in todos.assemble_rows(conn, sid, "student", today=TODAY)]
    assert sorted(kinds) == ["letter", "todo"]


def test_marking_asked_elsewhere_ticks_the_ask(
    conn: sqlite3.Connection, sid: str, request_row: repo.RecommendationRequest
) -> None:
    todo = todos.sync_letter_todo(conn, sid, request_row.id, today=TODAY)
    assert todo is not None
    tracker.set_request_status(conn, request_row.id, "asked", today=TODAY)
    todos.sync_letter_todo(conn, sid, request_row.id, today=TODAY)
    ticked = todos.get_todo(conn, todo.id)
    assert ticked is not None and ticked.done


def test_undated_or_removed_request_drops_its_open_ask(
    conn: sqlite3.Connection, sid: str, request_row: repo.RecommendationRequest
) -> None:
    todos.sync_letter_todo(conn, sid, request_row.id, today=TODAY)
    repo.update_recommendation_request(conn, request_row.id, due_on=None)
    assert todos.sync_letter_todo(conn, sid, request_row.id, today=TODAY) is None
    assert todos.list_todos(conn, sid) == []

    repo.update_recommendation_request(conn, request_row.id, due_on="2026-10-20")
    todos.sync_letter_todo(conn, sid, request_row.id, today=TODAY)
    repo.delete_recommendation_request(conn, request_row.id)
    todos.sync_letter_todo(conn, sid, request_row.id, today=TODAY)
    assert todos.list_todos(conn, sid) == []
