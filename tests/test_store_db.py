from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.store import repo
from src.store.db import (
    MIGRATIONS_DIR,
    MigrationError,
    apply_migrations,
    applied_migrations,
    connect,
    migration_files,
    open_db,
)

TABLES = (
    "students",
    "applications",
    "checklist_items",
    "essays",
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


@pytest.fixture
def application(conn: sqlite3.Connection, student: repo.Student) -> repo.Application:
    return repo.create_application(conn, student.student_id, "nc-cs-award-2026")


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {str(row["name"]) for row in rows}


# -- migrations -------------------------------------------------------------


def test_connect_creates_file_and_every_table(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "coach.db"
    with open_db(db_path) as conn:
        assert db_path.exists()
        assert set(TABLES) <= _table_names(conn)


def test_migrations_are_recorded_once_and_reopening_applies_nothing(tmp_path: Path) -> None:
    db_path = tmp_path / "coach.db"
    with open_db(db_path) as conn:
        first = applied_migrations(conn)
    assert first == [path.name for path in migration_files()]

    with open_db(db_path) as conn:
        assert apply_migrations(conn) == []
        assert applied_migrations(conn) == first


def test_applying_migrations_twice_on_one_connection_is_a_no_op(
    conn: sqlite3.Connection,
) -> None:
    before = applied_migrations(conn)
    assert apply_migrations(conn) == []
    assert applied_migrations(conn) == before


def test_repeat_migration_preserves_existing_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "coach.db"
    with open_db(db_path) as conn:
        repo.upsert_student(conn, "student_1", "Test Student")

    with open_db(db_path) as conn:
        stored = repo.get_student(conn, "student_1")
    assert stored is not None
    assert stored.name == "Test Student"


def test_migration_names_are_numbered(tmp_path: Path) -> None:
    assert migration_files(), "expected at least the initial migration"
    (tmp_path / "initial.sql").write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(MigrationError):
        migration_files(tmp_path)


def test_migrations_apply_in_number_order(tmp_path: Path) -> None:
    (tmp_path / "0002_second.sql").write_text("CREATE TABLE b (id INTEGER);", encoding="utf-8")
    (tmp_path / "0001_first.sql").write_text("CREATE TABLE a (id INTEGER);", encoding="utf-8")
    assert [path.name for path in migration_files(tmp_path)] == [
        "0001_first.sql",
        "0002_second.sql",
    ]


def test_failed_migration_is_not_recorded(tmp_path: Path) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0001_broken.sql").write_text("CREATE TABLE;", encoding="utf-8")
    with pytest.raises(MigrationError):
        connect(tmp_path / "coach.db", migrations)

    conn = sqlite3.connect(tmp_path / "coach.db")
    conn.row_factory = sqlite3.Row
    try:
        assert applied_migrations(conn) == []
    finally:
        conn.close()


def test_missing_migrations_directory_yields_no_files(tmp_path: Path) -> None:
    assert migration_files(tmp_path / "absent") == []


def test_initial_migration_is_the_shipped_one() -> None:
    assert (MIGRATIONS_DIR / "0001_initial.sql").exists()


def test_foreign_keys_are_enforced(conn: sqlite3.Connection, student: repo.Student) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_application(conn, "no_such_student", "nc-cs-award-2026")


# -- students ---------------------------------------------------------------


def test_upsert_student_creates_then_updates(conn: sqlite3.Connection) -> None:
    created = repo.upsert_student(conn, "student_1", "First Name")
    assert created.name == "First Name"

    updated = repo.upsert_student(conn, "student_1", "Second Name")
    assert updated.name == "Second Name"
    assert [s.student_id for s in repo.list_students(conn)] == ["student_1"]


def test_get_student_missing_is_none(conn: sqlite3.Connection) -> None:
    assert repo.get_student(conn, "nobody") is None


def test_delete_student_reports_whether_a_row_went(conn: sqlite3.Connection) -> None:
    repo.upsert_student(conn, "student_1")
    assert repo.delete_student(conn, "student_1") is True
    assert repo.delete_student(conn, "student_1") is False


# -- applications -----------------------------------------------------------


def test_application_crud(conn: sqlite3.Connection, student: repo.Student) -> None:
    created = repo.create_application(conn, student.student_id, "nc-cs-award-2026")
    assert created.status == "saved"

    assert repo.update_application(conn, created.id, status="submitted", notes="mailed") is True
    stored = repo.get_application(conn, created.id)
    assert stored is not None
    assert (stored.status, stored.notes) == ("submitted", "mailed")

    assert repo.delete_application(conn, created.id) is True
    assert repo.get_application(conn, created.id) is None


def test_update_with_no_arguments_changes_nothing(
    conn: sqlite3.Connection, application: repo.Application
) -> None:
    assert repo.update_application(conn, application.id) is False
    stored = repo.get_application(conn, application.id)
    assert stored is not None
    assert stored.status == application.status


def test_find_and_list_applications(conn: sqlite3.Connection, student: repo.Student) -> None:
    saved = repo.create_application(conn, student.student_id, "award-a")
    repo.create_application(conn, student.student_id, "award-b", status="submitted")

    found = repo.find_application(conn, student.student_id, "award-a")
    assert found is not None and found.id == saved.id
    assert repo.find_application(conn, student.student_id, "award-z") is None

    assert len(repo.list_applications(conn, student.student_id)) == 2
    submitted = repo.list_applications(conn, student.student_id, status="submitted")
    assert [a.catalog_id for a in submitted] == ["award-b"]


def test_one_application_per_award(
    conn: sqlite3.Connection, application: repo.Application
) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_application(conn, application.student_id, application.catalog_id)


# -- checklist items --------------------------------------------------------


def test_checklist_item_crud(
    conn: sqlite3.Connection, application: repo.Application
) -> None:
    item = repo.create_checklist_item(
        conn, application.id, "Request transcript", due_on="2026-10-01", position=1
    )
    assert item.done is False

    assert repo.update_checklist_item(conn, item.id, done=True, due_on=None) is True
    stored = repo.get_checklist_item(conn, item.id)
    assert stored is not None
    assert stored.done is True
    assert stored.due_on is None

    assert [i.id for i in repo.list_checklist_items(conn, application.id)] == [item.id]
    assert repo.delete_checklist_item(conn, item.id) is True
    assert repo.list_checklist_items(conn, application.id) == []


def test_checklist_items_sort_by_position(
    conn: sqlite3.Connection, application: repo.Application
) -> None:
    repo.create_checklist_item(conn, application.id, "Second", position=2)
    repo.create_checklist_item(conn, application.id, "First", position=1)
    assert [i.label for i in repo.list_checklist_items(conn, application.id)] == [
        "First",
        "Second",
    ]


# -- essays and essay links -------------------------------------------------


def test_essay_crud_tracks_word_count(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    essay = repo.create_essay(conn, student.student_id, "Why engineering", "one two three")
    assert essay.word_count == 3

    assert repo.update_essay(conn, essay.id, body="one two three four") is True
    stored = repo.get_essay(conn, essay.id)
    assert stored is not None
    assert stored.word_count == 4

    assert repo.update_essay(conn, essay.id, title="Why CS") is True
    retitled = repo.get_essay(conn, essay.id)
    assert retitled is not None
    assert retitled.title == "Why CS"
    assert retitled.word_count == 4

    assert [e.id for e in repo.list_essays(conn, student.student_id)] == [essay.id]
    assert repo.delete_essay(conn, essay.id) is True
    assert repo.get_essay(conn, essay.id) is None


def test_essay_links_pair_a_prompt_with_an_essay(
    conn: sqlite3.Connection, student: repo.Student, application: repo.Application
) -> None:
    essay = repo.create_essay(conn, student.student_id, "Leadership")
    link = repo.link_essay(conn, essay.id, application.id, "Describe a time you led.")
    assert link.prompt == "Describe a time you led."

    relinked = repo.link_essay(conn, essay.id, application.id, "Describe your leadership.")
    assert relinked.id == link.id
    assert relinked.prompt == "Describe your leadership."

    assert len(repo.list_essay_links(conn, essay_id=essay.id)) == 1
    assert len(repo.list_essay_links(conn, application_id=application.id)) == 1
    assert len(repo.list_essay_links(conn)) == 1

    assert repo.unlink_essay(conn, essay.id, application.id) is True
    assert repo.list_essay_links(conn) == []


def test_deleting_an_essay_removes_its_links(
    conn: sqlite3.Connection, student: repo.Student, application: repo.Application
) -> None:
    essay = repo.create_essay(conn, student.student_id, "Leadership")
    repo.link_essay(conn, essay.id, application.id)
    repo.delete_essay(conn, essay.id)
    assert repo.list_essay_links(conn) == []


# -- recommenders and requests ----------------------------------------------


def test_recommender_crud(conn: sqlite3.Connection, student: repo.Student) -> None:
    rec = repo.create_recommender(
        conn, student.student_id, "Ms. Rivera", role="Physics teacher"
    )
    assert repo.update_recommender(conn, rec.id, email="rivera@example.edu") is True
    stored = repo.get_recommender(conn, rec.id)
    assert stored is not None
    assert stored.email == "rivera@example.edu"
    assert stored.role == "Physics teacher"

    assert [r.id for r in repo.list_recommenders(conn, student.student_id)] == [rec.id]
    assert repo.delete_recommender(conn, rec.id) is True
    assert repo.get_recommender(conn, rec.id) is None


def test_recommendation_request_crud(
    conn: sqlite3.Connection, student: repo.Student, application: repo.Application
) -> None:
    rec = repo.create_recommender(conn, student.student_id, "Ms. Rivera")
    request = repo.create_recommendation_request(
        conn, rec.id, application.id, requested_on="2026-09-20", due_on="2026-10-15"
    )
    assert request.status == "planned"
    assert request.submitted_on is None

    assert (
        repo.update_recommendation_request(
            conn, request.id, status="submitted", submitted_on="2026-10-02"
        )
        is True
    )
    stored = repo.get_recommendation_request(conn, request.id)
    assert stored is not None
    assert (stored.status, stored.submitted_on) == ("submitted", "2026-10-02")

    assert len(repo.list_recommendation_requests(conn, recommender_id=rec.id)) == 1
    assert len(repo.list_recommendation_requests(conn, application_id=application.id)) == 1
    assert len(repo.list_recommendation_requests(conn)) == 1

    assert repo.delete_recommendation_request(conn, request.id) is True
    assert repo.get_recommendation_request(conn, request.id) is None


def test_deleting_a_recommender_removes_their_requests(
    conn: sqlite3.Connection, student: repo.Student, application: repo.Application
) -> None:
    rec = repo.create_recommender(conn, student.student_id, "Ms. Rivera")
    repo.create_recommendation_request(conn, rec.id, application.id)
    repo.delete_recommender(conn, rec.id)
    assert repo.list_recommendation_requests(conn) == []


# -- outcomes ---------------------------------------------------------------


def test_set_outcome_inserts_then_replaces(
    conn: sqlite3.Connection, student: repo.Student, application: repo.Application
) -> None:
    outcome = repo.set_outcome(conn, application.id, result="pending")
    assert outcome.amount_awarded is None

    updated = repo.set_outcome(
        conn,
        application.id,
        result="won",
        amount_awarded=2500.0,
        paid_to="NC State University",
        renewal_terms="Renewable with a 3.0 GPA",
        decided_on="2026-11-01",
    )
    assert updated.id == outcome.id
    assert updated.amount_awarded == 2500.0
    assert updated.paid_to == "NC State University"
    assert updated.renewal_terms == "Renewable with a 3.0 GPA"

    assert [o.id for o in repo.list_outcomes(conn, student.student_id)] == [outcome.id]
    assert repo.delete_outcome(conn, application.id) is True
    assert repo.get_outcome(conn, application.id) is None


def test_deleting_an_application_removes_its_outcome_and_checklist(
    conn: sqlite3.Connection, application: repo.Application
) -> None:
    repo.create_checklist_item(conn, application.id, "Request transcript")
    repo.set_outcome(conn, application.id, result="won", amount_awarded=1000.0)
    repo.delete_application(conn, application.id)
    assert repo.get_outcome(conn, application.id) is None
    assert repo.list_checklist_items(conn, application.id) == []


# -- colleges ---------------------------------------------------------------


def test_college_crud_and_net_price(conn: sqlite3.Connection, student: repo.Student) -> None:
    college = repo.create_college(
        conn, student.student_id, "NC State University", cost_of_attendance=28000.0
    )
    assert repo.net_price(college) is None

    assert repo.update_college(conn, college.id, aid_offered=12000.0, status="applied") is True
    stored = repo.get_college(conn, college.id)
    assert stored is not None
    assert repo.net_price(stored) == 16000.0
    assert stored.status == "applied"

    assert [c.id for c in repo.list_colleges(conn, student.student_id)] == [college.id]
    assert repo.delete_college(conn, college.id) is True
    assert repo.get_college(conn, college.id) is None


# -- settings ---------------------------------------------------------------


def test_settings_round_trip(conn: sqlite3.Connection) -> None:
    assert repo.get_setting(conn, "mode") is None
    assert repo.get_setting(conn, "mode", default="student") == "student"

    repo.set_setting(conn, "mode", "parent")
    assert repo.get_setting(conn, "mode") == "parent"
    repo.set_setting(conn, "mode", "student")
    assert repo.get_setting(conn, "mode") == "student"

    assert repo.all_settings(conn) == {"mode": "student"}
    assert repo.delete_setting(conn, "mode") is True
    assert repo.delete_setting(conn, "mode") is False


def test_flags_round_trip(conn: sqlite3.Connection) -> None:
    assert repo.get_flag(conn, "operator_enabled") is False
    assert repo.get_flag(conn, "operator_enabled", default=True) is True

    repo.set_flag(conn, "operator_enabled", True)
    assert repo.get_flag(conn, "operator_enabled") is True
    repo.set_flag(conn, "operator_enabled", False)
    assert repo.get_flag(conn, "operator_enabled") is False


# -- cascade ----------------------------------------------------------------


def test_deleting_a_student_cascades_to_everything_they_own(
    conn: sqlite3.Connection,
) -> None:
    student = repo.upsert_student(conn, "student_1", "Test Student")
    application = repo.create_application(conn, student.student_id, "nc-cs-award-2026")
    repo.create_checklist_item(conn, application.id, "Request transcript")
    essay = repo.create_essay(conn, student.student_id, "Leadership", "a b c")
    repo.link_essay(conn, essay.id, application.id, "Describe a time you led.")
    recommender = repo.create_recommender(conn, student.student_id, "Ms. Rivera")
    repo.create_recommendation_request(conn, recommender.id, application.id)
    repo.set_outcome(conn, application.id, result="won", amount_awarded=2500.0)
    repo.create_college(conn, student.student_id, "NC State University")

    other = repo.upsert_student(conn, "student_2", "Other Student")
    other_application = repo.create_application(conn, other.student_id, "nc-cs-award-2026")

    assert repo.delete_student(conn, student.student_id) is True

    assert repo.list_applications(conn, student.student_id) == []
    assert repo.list_checklist_items(conn, application.id) == []
    assert repo.list_essays(conn, student.student_id) == []
    assert repo.list_essay_links(conn) == []
    assert repo.list_recommenders(conn, student.student_id) == []
    assert repo.list_recommendation_requests(conn) == []
    assert repo.list_outcomes(conn, student.student_id) == []
    assert repo.list_colleges(conn, student.student_id) == []

    assert [a.id for a in repo.list_applications(conn, other.student_id)] == [
        other_application.id
    ]
