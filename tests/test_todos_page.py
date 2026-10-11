from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from app.identity import Identity
from app.sections import todos as page
from src.store import milestones, repo, todos
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


def _application(conn: sqlite3.Connection, sid: str, deadline: str | None = None) -> repo.Application:
    return repo.create_application(
        conn, sid, "award_1", status="planning", title="Robotics Award", deadline=deadline
    )


# -- bucketing ----------------------------------------------------------------


def test_rows_are_grouped_by_date(conn: sqlite3.Connection, sid: str) -> None:
    todos.create_todo(conn, sid, "Late", due_on="2026-10-01")
    todos.create_todo(conn, sid, "Soon", due_on="2026-10-14")
    todos.create_todo(conn, sid, "Next month", due_on="2026-11-20")
    todos.create_todo(conn, sid, "Spring", due_on="2027-03-01")
    todos.create_todo(conn, sid, "Whenever")

    rows = todos.assemble_rows(conn, sid, "student", today=TODAY)
    groups = page.group_rows(rows, TODAY)

    titles = {bucket: [row.title for row in found] for bucket, found in groups.items()}
    assert titles["overdue"] == ["Late"]
    assert titles["this_week"] == ["Soon"]
    assert titles["coming_up"] == ["Next month"]
    assert titles["later"] == ["Spring"]
    assert titles["someday"] == ["Whenever"]


def test_next_up_is_the_nearest_dated_item_past_this_week(
    conn: sqlite3.Connection, sid: str
) -> None:
    todos.create_todo(conn, sid, "Whenever")
    todos.create_todo(conn, sid, "Spring", due_on="2027-03-01")
    todos.create_todo(conn, sid, "Course registration", due_on="2026-11-20")

    groups = page.group_rows(todos.assemble_rows(conn, sid, "student", today=TODAY), TODAY)

    assert page.next_up_text(groups, TODAY) == "Course registration, Fri, Nov 20"
    assert page.next_up_text(page.group_rows([], TODAY), TODAY) is None


# -- authorship ---------------------------------------------------------------


def test_author_comes_from_identity_when_signed_in() -> None:
    who = Identity(login="dana@example.com", name="Dana Smith", role="parent")
    assert page.author_for(who, "parent") == ("Dana Smith", "parent")

    row = todos.TodoRow(
        key="todo_1", kind="todo", ref_id=1, title="x", created_by="Dana Smith",
        created_role="parent",
    )
    assert page.author_label(row, "student") == "from Dana"
    assert page.author_label(row, "parent", viewer_name="Dana Smith") == "from you"


def test_author_falls_back_to_the_mode_under_the_pin() -> None:
    assert page.author_for(None, "student") == ("", "student")

    by_her = todos.TodoRow(key="t", kind="todo", ref_id=1, title="x", created_role="student")
    by_parents = todos.TodoRow(key="t", kind="todo", ref_id=2, title="x", created_role="parent")
    assert page.author_label(by_her, "student") == "from you"
    assert page.author_label(by_her, "parent") == "from her"
    assert page.author_label(by_parents, "student") == "from your parents"


def test_milestone_and_tracker_rows_name_their_source_not_a_person() -> None:
    milestone = todos.TodoRow(key="t", kind="todo", ref_id=1, title="x", source_kind="milestone")
    deadline = todos.TodoRow(key="d", kind="deadline", ref_id=1, title="x")
    assert page.author_label(milestone, "student") == "planning date"
    assert page.author_label(deadline, "student") == ""


def test_assignee_labels_speak_to_the_reader() -> None:
    assert page.assignee_label("student", "student") == "For you"
    assert page.assignee_label("student", "parent") == "For her"
    assert page.assignee_label("parent", "student") == "For your parents"
    assert page.assignee_label("family", "parent") == "For the family"


# -- assignee filtering -------------------------------------------------------


def test_student_sees_hers_the_familys_and_what_she_asked_for(
    conn: sqlite3.Connection, sid: str
) -> None:
    todos.create_todo(conn, sid, "Hers", assignee="student", created_role="parent")
    todos.create_todo(conn, sid, "Family", assignee="family", created_role="parent")
    todos.create_todo(conn, sid, "Parents' own", assignee="parent", created_role="parent")
    todos.create_todo(conn, sid, "Pay the PSAT fee", assignee="parent", created_role="student")

    student_titles = {row.title for row in todos.assemble_rows(conn, sid, "student", today=TODAY)}
    parent_titles = {row.title for row in todos.assemble_rows(conn, sid, "parent", today=TODAY)}

    assert student_titles == {"Hers", "Family", "Pay the PSAT fee"}
    assert parent_titles == student_titles | {"Parents' own"}


# -- tracker rows -------------------------------------------------------------


def test_tracker_work_appears_as_linked_rows(conn: sqlite3.Connection, sid: str) -> None:
    application = _application(conn, sid, deadline="2026-11-01")
    repo.create_checklist_item(conn, application.id, "Request transcript", due_on="2026-10-12")
    recommender = repo.create_recommender(conn, sid, "Ms. Rivera")
    repo.create_recommendation_request(
        conn, recommender.id, application.id, status="asked", due_on="2026-10-20"
    )

    rows = {row.kind: row for row in todos.assemble_rows(conn, sid, "student", today=TODAY)}

    assert rows["deadline"].title == "Submit Robotics Award"
    assert rows["deadline"].assignee == "family"
    assert rows["checklist"].detail == "Robotics Award"
    assert rows["letter"].title == "Letter from Ms. Rivera (asked)"
    assert {row.application_id for row in rows.values()} == {application.id}
    assert len({row.key for row in rows.values()}) == 3


def test_checklist_step_tick_is_two_way(conn: sqlite3.Connection, sid: str) -> None:
    application = _application(conn, sid)
    item = repo.create_checklist_item(conn, application.id, "Request transcript", due_on="2026-10-12")

    [row] = todos.assemble_rows(conn, sid, "student", today=TODAY)
    assert todos.set_row_done(conn, row, True, today=TODAY)
    ticked = repo.get_checklist_item(conn, item.id)
    assert ticked is not None and ticked.done
    assert todos.assemble_rows(conn, sid, "student", today=TODAY) == []

    repo.update_checklist_item(conn, item.id, done=False)
    [back] = todos.assemble_rows(conn, sid, "student", today=TODAY)
    assert back.kind == "checklist" and back.ref_id == item.id


def test_ticking_a_deadline_submits_and_a_letter_moves_one_step(
    conn: sqlite3.Connection, sid: str
) -> None:
    application = _application(conn, sid, deadline="2026-10-15")
    recommender = repo.create_recommender(conn, sid, "Ms. Rivera")
    request = repo.create_recommendation_request(
        conn, recommender.id, application.id, due_on="2026-10-12"
    )
    rows = {row.kind: row for row in todos.assemble_rows(conn, sid, "student", today=TODAY)}

    assert todos.set_row_done(conn, rows["letter"], True, today=TODAY)
    asked = repo.get_recommendation_request(conn, request.id)
    assert asked is not None and asked.status == "asked"

    assert todos.set_row_done(conn, rows["deadline"], True, today=TODAY)
    submitted = repo.get_application(conn, application.id)
    assert submitted is not None and submitted.status == "submitted"
    assert not todos.set_row_done(conn, rows["deadline"], False, today=TODAY)


def test_stored_row_tick_records_who(conn: sqlite3.Connection, sid: str) -> None:
    todo = todos.create_todo(conn, sid, "Pick next year's classes")
    row = todos.row_from_todo(todo)

    assert todos.set_row_done(conn, row, True, done_by="Dana Smith", today=TODAY)
    done = todos.get_todo(conn, todo.id)
    assert done is not None and (done.done_on, done.done_by) == ("2026-10-10", "Dana Smith")


# -- milestones ---------------------------------------------------------------


MILESTONES = [
    milestones.Milestone(
        id="psat_window", title="PSAT window", month=10, day=1, end_month=10, end_day=31,
        grade_levels=("10", "11"),
    ),
    milestones.Milestone(
        id="family_course_registration", title="Course registration", month=11, day=20,
        grade_levels=("10",), source=milestones.FAMILY_SOURCE,
    ),
    milestones.Milestone(id="fafsa_opens", title="FAFSA opens", month=10, day=1,
                         grade_levels=("12",)),
    milestones.Milestone(id="far_off", title="Spring thing", month=4, day=1),
]


def test_her_grades_milestones_land_once_per_school_year(
    conn: sqlite3.Connection, sid: str
) -> None:
    created = todos.sync_milestone_todos(conn, sid, "10", today=TODAY, milestones=MILESTONES)

    assert [(todo.title, todo.due_on) for todo in created] == [
        ("PSAT window", "2026-10-31"),
        ("Course registration", "2026-11-20"),
    ]
    assert {todo.source_kind for todo in created} == {"milestone"}

    todos.set_todo_done(conn, created[0].id, True, today=TODAY)
    assert todos.sync_milestone_todos(conn, sid, "10", today=TODAY, milestones=MILESTONES) == []
    # A year on she is in 11th grade: the PSAT comes round again, registration does not.
    later = todos.sync_milestone_todos(
        conn, sid, "11", today=date(2027, 10, 5), milestones=MILESTONES
    )
    assert [todo.source_ref for todo in later] == ["psat_window:2028"]


def test_no_grade_writes_no_milestones(conn: sqlite3.Connection, sid: str) -> None:
    assert todos.sync_milestone_todos(conn, sid, "", today=TODAY, milestones=MILESTONES) == []
