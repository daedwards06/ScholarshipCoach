from __future__ import annotations

import shutil
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from src.store import repo, todos
from src.store.db import MIGRATIONS_DIR, applied_migrations, connect, migration_files, open_db

PHASE_A_MIGRATIONS = (
    "0005_shared_todos.sql",
    "0006_college_requirements.sql",
    "0007_college_priority.sql",
    "0008_essay_sharing.sql",
    "0009_activity.sql",
)

# Each Phase A file creates only IF NOT EXISTS objects or holds a single ALTER,
# so an interrupted CREATE-only file can be run again from the top.
RERUNNABLE = ("0005_shared_todos.sql", "0006_college_requirements.sql", "0009_activity.sql")

TABLES_AT_0004 = (
    "students",
    "applications",
    "checklist_items",
    "essays",
    "essay_versions",
    "essay_links",
    "recommenders",
    "recommendation_requests",
    "outcomes",
    "colleges",
    "settings",
)


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


def _migrations_up_to(tmp_path: Path, last: str) -> Path:
    directory = tmp_path / f"migrations_{last[:4]}"
    directory.mkdir()
    for path in migration_files():
        if path.name > last:
            break
        shutil.copy(path, directory / path.name)
    return directory


def _seed_0004(conn: sqlite3.Connection) -> None:
    """One row in every table, written with the 0004 columns only."""
    now = "2026-09-01T00:00:00+00:00"
    conn.executescript(
        f"""
        INSERT INTO students VALUES ('student_1', 'Test Student', '{now}', '{now}');
        INSERT INTO applications (id, student_id, catalog_id, status, notes, title,
            source_url, deadline, created_at, updated_at)
            VALUES (1, 'student_1', 'award-1', 'planning', '', 'Award', '', '2027-01-15',
            '{now}', '{now}');
        INSERT INTO checklist_items (application_id, label, created_at, updated_at)
            VALUES (1, 'Essay', '{now}', '{now}');
        INSERT INTO essays (id, student_id, title, body, word_count, theme, created_at,
            updated_at) VALUES (1, 'student_1', 'Draft', 'one two', 2, 'other', '{now}', '{now}');
        INSERT INTO essay_versions (essay_id, title, body, word_count, created_at)
            VALUES (1, 'Draft', 'one two', 2, '{now}');
        INSERT INTO essay_links (essay_id, application_id, prompt, created_at)
            VALUES (1, 1, 'Why CS?', '{now}');
        INSERT INTO recommenders (id, student_id, name, created_at, updated_at)
            VALUES (1, 'student_1', 'Teacher', '{now}', '{now}');
        INSERT INTO recommendation_requests (recommender_id, application_id, created_at,
            updated_at) VALUES (1, 1, '{now}', '{now}');
        INSERT INTO outcomes (application_id, created_at, updated_at) VALUES (1, '{now}', '{now}');
        INSERT INTO colleges (student_id, name, created_at, updated_at)
            VALUES ('student_1', 'Example State University', '{now}', '{now}');
        INSERT INTO settings VALUES ('parent_mode', 'true', '{now}');
        """
    )
    conn.commit()


def _counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in TABLES_AT_0004
    }


# -- migrations -------------------------------------------------------------


def test_phase_a_migrations_ship_in_order() -> None:
    names = [path.name for path in migration_files()]
    assert names[4:9] == list(PHASE_A_MIGRATIONS)


@pytest.mark.parametrize("last", PHASE_A_MIGRATIONS)
def test_each_migration_upgrades_a_populated_0004_database(tmp_path: Path, last: str) -> None:
    db_path = tmp_path / "coach.db"
    with open_db(db_path, _migrations_up_to(tmp_path, "0004_colleges_money.sql")) as conn:
        _seed_0004(conn)
        before = _counts(conn)
        assert all(before.values())

    with open_db(db_path, _migrations_up_to(tmp_path, last)) as conn:
        assert applied_migrations(conn)[-1] == last
        assert _counts(conn) == before


def test_full_upgrade_keeps_rows_and_reopening_is_a_no_op(tmp_path: Path) -> None:
    db_path = tmp_path / "coach.db"
    with open_db(db_path, _migrations_up_to(tmp_path, "0004_colleges_money.sql")) as conn:
        _seed_0004(conn)
        before = _counts(conn)

    with open_db(db_path) as conn:
        assert _counts(conn) == before
        (essay,) = repo.list_essays(conn, "student_1")
        assert essay.shared_with_parents is False
        (college,) = repo.list_colleges(conn, "student_1")
        assert college.priority is None
        applied = applied_migrations(conn)
        schema = conn.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()

    with open_db(db_path) as conn:
        assert applied_migrations(conn) == applied
        assert conn.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall() == schema
        assert _counts(conn) == before


@pytest.mark.parametrize("name", RERUNNABLE)
def test_create_only_migrations_are_safe_to_run_twice(
    conn: sqlite3.Connection, name: str
) -> None:
    conn.executescript((MIGRATIONS_DIR / name).read_text(encoding="utf-8-sig"))


# -- essay sharing ----------------------------------------------------------


def test_essays_start_private_and_parents_see_only_shared(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    private = repo.create_essay(conn, student.student_id, "First draft", "words")
    shared = repo.create_essay(conn, student.student_id, "Ready", "more words")
    assert private.shared_with_parents is False

    assert repo.set_essay_shared(conn, shared.id, True)
    assert [e.id for e in repo.list_essays(conn, student.student_id, shared_only=True)] == [
        shared.id
    ]
    assert len(repo.list_essays(conn, student.student_id)) == 2


def test_sharing_adds_no_version_and_keeps_order(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    essay = repo.create_essay(conn, student.student_id, "Draft", "words")
    repo.set_essay_shared(conn, essay.id, True)
    stored = repo.get_essay(conn, essay.id)
    assert stored is not None
    assert stored.shared_with_parents is True
    assert stored.updated_at == essay.updated_at
    assert len(repo.list_essay_versions(conn, essay.id)) == 1

    repo.set_essay_shared(conn, essay.id, False)
    assert repo.list_essays(conn, student.student_id, shared_only=True) == []
    assert not repo.set_essay_shared(conn, 999, True)


# -- colleges and requirements ----------------------------------------------


def test_first_target_college(conn: sqlite3.Connection, student: repo.Student) -> None:
    assert repo.first_target_college(conn, student.student_id) is None
    repo.create_college(conn, student.student_id, "Example Tech")
    target = repo.create_college(conn, student.student_id, "Example State", priority=1)
    assert target.priority == 1

    found = repo.first_target_college(conn, student.student_id)
    assert found is not None and found.id == target.id

    repo.update_college(conn, target.id, priority=None)
    assert repo.first_target_college(conn, student.student_id) is None


def test_requirement_crud_normalizes_and_sorts(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    college = repo.create_college(conn, student.student_id, "Example State")
    gpa = repo.create_requirement(
        conn, college.id, "Unweighted GPA", "GPA", target="3.5", due_by="12", position=2
    )
    course = repo.create_requirement(
        conn, college.id, "Precalculus", "course", due_by="11", status="In progress", position=1
    )
    odd = repo.create_requirement(conn, college.id, "Something", "nonsense", status="??")
    assert gpa.category == "gpa"
    assert course.status == "in_progress"
    assert (odd.category, odd.status) == ("other", "not_started")

    listed = repo.list_requirements(conn, college.id)
    assert [item.id for item in listed] == [odd.id, course.id, gpa.id]
    assert listed[2] == gpa

    assert repo.update_requirement(conn, course.id, status="met", verified_on="2026-10-03")
    stored = repo.get_requirement(conn, course.id)
    assert stored is not None
    assert (stored.status, stored.verified_on) == ("met", "2026-10-03")
    assert not repo.update_requirement(conn, course.id)

    assert repo.delete_requirement(conn, odd.id)
    assert repo.get_requirement(conn, odd.id) is None


def test_deleting_a_college_removes_its_requirements(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    college = repo.create_college(conn, student.student_id, "Example State")
    repo.create_requirement(conn, college.id, "Essay")
    repo.delete_college(conn, college.id)
    assert repo.list_requirements(conn, college.id) == []


def test_requirement_progress_leaves_out_not_needed() -> None:
    def req(status: str) -> repo.CollegeRequirement:
        return repo.CollegeRequirement(id=0, college_id=1, label="x", status=status)

    rows = [req("met"), req("met"), req("in_progress"), req("not_started"), req("not_needed")]
    assert repo.requirement_progress(rows) == (2, 4)
    assert repo.requirement_progress([]) == (0, 0)


# -- activity ---------------------------------------------------------------


def test_activity_is_one_row_per_person_per_day(conn: sqlite3.Connection) -> None:
    repo.record_activity(conn, "student", "2026-10-03")
    repo.record_activity(conn, "student", "2026-10-03")
    repo.record_activity(conn, "student", "2026-10-04")
    repo.record_activity(conn, "parent", "2026-10-03")

    assert repo.list_activity(conn, "student") == [
        repo.ActivityDay("student", "2026-10-03", 2),
        repo.ActivityDay("student", "2026-10-04", 1),
    ]
    assert len(repo.list_activity(conn)) == 3


# -- to-dos -----------------------------------------------------------------


def test_todo_crud_and_authorship(conn: sqlite3.Connection, student: repo.Student) -> None:
    todo = todos.create_todo(
        conn,
        student.student_id,
        "Register for PSAT",
        due_on="2026-10-10",
        assignee="Family",
        created_by="parent@example.com",
        created_role="parent",
    )
    stored = todos.get_todo(conn, todo.id)
    assert stored == todo
    assert stored.assignee == "family"
    assert not stored.done

    assert todos.set_todo_done(
        conn, todo.id, True, done_by="student@example.com", today=date(2026, 10, 5)
    )
    done = todos.get_todo(conn, todo.id)
    assert done is not None
    assert (done.done_on, done.done_by, done.done) == ("2026-10-05", "student@example.com", True)

    todos.set_todo_done(conn, todo.id, False)
    undone = todos.get_todo(conn, todo.id)
    assert undone is not None
    assert (undone.done_on, undone.done_by) == (None, "")

    assert todos.update_todo(conn, todo.id, title="Take PSAT", due_on=None, assignee="bogus")
    edited = todos.get_todo(conn, todo.id)
    assert edited is not None
    assert (edited.title, edited.due_on, edited.assignee) == ("Take PSAT", None, "student")

    assert todos.delete_todo(conn, todo.id)
    assert todos.get_todo(conn, todo.id) is None


def test_list_todos_filters_and_orders(conn: sqlite3.Connection, student: repo.Student) -> None:
    sid = student.student_id
    someday = todos.create_todo(conn, sid, "Someday", assignee="student")
    late = todos.create_todo(conn, sid, "Later", due_on="2026-12-01", assignee="parent")
    soon = todos.create_todo(conn, sid, "Soon", due_on="2026-10-05", assignee="family")
    todos.set_todo_done(conn, late.id, True)

    assert [t.id for t in todos.list_todos(conn, sid)] == [soon.id, late.id, someday.id]
    assert [t.id for t in todos.list_todos(conn, sid, include_done=False)] == [
        soon.id,
        someday.id,
    ]
    assert [t.id for t in todos.list_todos(conn, sid, assignees=("student", "family"))] == [
        soon.id,
        someday.id,
    ]


def test_find_todo_by_source(conn: sqlite3.Connection, student: repo.Student) -> None:
    sid = student.student_id
    assert todos.find_todo(conn, sid, "requirement", "7") is None
    todo = todos.create_todo(conn, sid, "Precalculus", source_kind="requirement", source_ref="7")
    found = todos.find_todo(conn, sid, "requirement", "7")
    assert found is not None and found.id == todo.id
    assert todos.create_todo(conn, sid, "x", source_kind="weird").source_kind == "manual"


def test_check_constraint_rejects_raw_bad_assignee(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO todos (student_id, title, assignee, created_at, updated_at) "
            "VALUES (?, 'x', 'teacher', '', '')",
            (student.student_id,),
        )


def test_deleting_a_student_removes_their_todos(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    todos.create_todo(conn, student.student_id, "Anything")
    repo.delete_student(conn, student.student_id)
    assert todos.list_todos(conn, student.student_id) == []


@pytest.mark.parametrize(
    ("due_on", "done_on", "bucket"),
    [
        (None, None, "someday"),
        ("not a date", None, "someday"),
        ("2026-10-02", None, "overdue"),
        ("2026-10-03", None, "this_week"),
        ("2026-10-10", None, "this_week"),
        ("2026-10-11", None, "coming_up"),
        ("2026-12-02", None, "coming_up"),
        ("2026-12-03", None, "later"),
        ("2026-10-02", "2026-10-01", "done"),
    ],
)
def test_todo_bucket(due_on: str | None, done_on: str | None, bucket: str) -> None:
    todo = todos.Todo(id=1, student_id="s", title="t", due_on=due_on, done_on=done_on)
    assert todos.todo_bucket(todo, date(2026, 10, 3)) == bucket
    assert bucket in todos.TODO_BUCKETS
