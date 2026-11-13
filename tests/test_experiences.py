from __future__ import annotations

import shutil
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from app.sections import opportunities as opp_page
from app.sections import todos as todos_page
from scripts.export_outcomes import build_outcomes_frame
from src.store import experiences, repo, todos
from src.store.db import applied_migrations, connect, migration_files, open_db
from src.store.tracker import TransitionError

TODAY = date(2026, 11, 13)
PROGRAM = "example-camp"
PROGRAMS = frozenset({PROGRAM})


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


def _apply_todo(conn: sqlite3.Connection, sid: str, ref: str = PROGRAM) -> todos.Todo:
    return todos.create_todo(
        conn, sid, "Apply to Example Camp", source_kind="opportunity", source_ref=ref
    )


def _tick(conn: sqlite3.Connection, todo: todos.Todo, done: bool, today: date = TODAY) -> None:
    todos.set_row_done(
        conn, todos.row_from_todo(todo), done, today=today, program_ids=PROGRAMS
    )


# -- migration ----------------------------------------------------------------


def test_0010_upgrades_a_populated_0009_database_and_reruns(tmp_path: Path) -> None:
    directory = tmp_path / "migrations_0009"
    directory.mkdir()
    for path in migration_files():
        if path.name <= "0009_activity.sql":
            shutil.copy(path, directory / path.name)
    db_path = tmp_path / "coach.db"
    with open_db(db_path, directory) as conn:
        repo.upsert_student(conn, "student_1")
        todos.create_todo(conn, "student_1", "Pay the PSAT fee")
        repo.record_activity(conn, "student", TODAY)
        tables = [
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            if row[0] != "schema_migrations"
        ]
        before = {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }

    with open_db(db_path) as conn:
        assert applied_migrations(conn)[-1] == "0010_experiences.sql"
        for table, count in before.items():
            assert int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) == count
        assert experiences.list_experiences(conn, "student_1") == []
        script = (migration_files()[-1]).read_text(encoding="utf-8-sig")
        conn.executescript(script)


def test_one_row_per_program_and_term(conn: sqlite3.Connection, sid: str) -> None:
    experiences.create_experience(conn, sid, "Camp", catalog_id=PROGRAM, term="2026–27")
    with pytest.raises(sqlite3.IntegrityError):
        experiences.create_experience(conn, sid, "Camp", catalog_id=PROGRAM, term="2026–27")
    experiences.create_experience(conn, sid, "Club", term="2026–27")
    experiences.create_experience(conn, sid, "Club", term="2026–27")


# -- status graph -------------------------------------------------------------


def test_program_graph_allows_results_refuses_jumps_and_steps_back(
    conn: sqlite3.Connection, sid: str
) -> None:
    row = experiences.create_experience(conn, sid, "Camp", catalog_id=PROGRAM)
    assert row.status == "applied"
    assert experiences.next_statuses(row) == ("applied", "accepted", "waitlisted", "not_accepted")

    with pytest.raises(TransitionError):
        experiences.set_status(conn, row.id, "completed")

    row = experiences.set_status(conn, row.id, "waitlisted")
    row = experiences.set_status(conn, row.id, "accepted")
    row = experiences.set_status(conn, row.id, "completed")
    # A misclick is not permanent: each status can step back one.
    row = experiences.set_status(conn, row.id, "accepted")
    row = experiences.set_status(conn, row.id, "applied")
    assert row.status == "applied"


def test_hand_entered_rows_are_active_or_completed(conn: sqlite3.Connection, sid: str) -> None:
    club = experiences.create_experience(conn, sid, "Robotics club", kind="club", status="accepted")
    assert club.status == "active"
    with pytest.raises(TransitionError):
        experiences.set_status(conn, club.id, "accepted")
    assert experiences.set_status(conn, club.id, "completed").status == "completed"
    assert experiences.set_status(conn, club.id, "active").status == "active"


# -- ticking the apply to-do --------------------------------------------------


def test_ticking_the_apply_todo_records_applied_once(conn: sqlite3.Connection, sid: str) -> None:
    todo = _apply_todo(conn, sid)
    _tick(conn, todo, True)
    _tick(conn, todo, True)
    (row,) = experiences.list_experiences(conn, sid)
    assert (row.catalog_id, row.status, row.title, row.term) == (
        PROGRAM, "applied", "Example Camp", "2026–27"
    )

    _tick(conn, todo, False)
    assert experiences.list_experiences(conn, sid) == []


def test_a_heard_back_result_survives_an_untick(conn: sqlite3.Connection, sid: str) -> None:
    todo = _apply_todo(conn, sid)
    _tick(conn, todo, True)
    (row,) = experiences.list_experiences(conn, sid)
    experiences.set_status(conn, row.id, "accepted")
    _tick(conn, todo, False)
    _tick(conn, todo, True)
    (kept,) = experiences.list_experiences(conn, sid)
    assert kept.status == "accepted"


def test_an_awards_apply_todo_records_nothing(conn: sqlite3.Connection, sid: str) -> None:
    todo = _apply_todo(conn, sid, ref="example-award")
    _tick(conn, todo, True)
    assert experiences.list_experiences(conn, sid) == []
    assert todos.get_todo(conn, todo.id).done  # type: ignore[union-attr]


def test_reminders_and_follow_ups_are_not_applications(
    conn: sqlite3.Connection, sid: str
) -> None:
    watch = todos.create_todo(
        conn, sid, "Watch for Example Camp to open", source_kind="opportunity", source_ref=PROGRAM
    )
    _tick(conn, watch, True)
    assert experiences.list_experiences(conn, sid) == []


def test_not_accepted_last_year_applies_again_as_a_new_cycle(
    conn: sqlite3.Connection, sid: str
) -> None:
    todo = _apply_todo(conn, sid)
    _tick(conn, todo, True, today=date(2025, 11, 1))
    (first,) = experiences.list_experiences(conn, sid)
    conn.execute(
        "UPDATE experiences SET created_at = ?, term = ? WHERE id = ?",
        ("2025-11-01T00:00:00+00:00", "2025–26", first.id),
    )
    conn.commit()
    first = experiences.set_status(conn, first.id, "not_accepted")
    assert experiences.apply_again_due(first, TODAY)

    reopened = todos.reopen_apply_todo(conn, sid, PROGRAM, "Example Camp", "2027-02-01")
    assert reopened is not None and not reopened.done and reopened.due_on == "2027-02-01"
    _tick(conn, reopened, True)
    rows = experiences.list_experiences(conn, sid)
    assert [(row.status, row.term) for row in rows] == [
        ("not_accepted", "2025–26"),
        ("applied", "2026–27"),
    ]


# -- Opportunities ------------------------------------------------------------


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"catalog_id": "accepted-camp", "is_program": True, "opportunity_group": "open_now"},
            {"catalog_id": "done-camp", "is_program": True, "opportunity_group": "passed"},
            {"catalog_id": "rejected-camp", "is_program": True, "opportunity_group": "opens_later"},
            {"catalog_id": "award", "is_program": False, "opportunity_group": "open_now"},
        ]
    )


def test_accepted_programs_move_to_yours_and_not_accepted_stays(
    conn: sqlite3.Connection, sid: str
) -> None:
    for catalog_id, status in (
        ("accepted-camp", "accepted"),
        ("done-camp", "completed"),
        ("rejected-camp", "not_accepted"),
    ):
        row = experiences.create_experience(conn, sid, catalog_id, catalog_id=catalog_id)
        if status == "completed":
            experiences.set_status(conn, row.id, "accepted")
        experiences.set_status(conn, row.id, status)
    results = experiences.latest_by_catalog(experiences.list_experiences(conn, sid))

    yours, rest = opp_page.split_yours(_frame(), results)
    assert list(yours["catalog_id"]) == ["accepted-camp", "done-camp"]
    assert list(rest["catalog_id"]) == ["rejected-camp", "award"]


def test_result_headline_and_membership_offer() -> None:
    accepted = experiences.Experience(
        id=1, student_id="s", title="Camp", catalog_id=PROGRAM, status="accepted", term="Spring 2027"
    )
    assert opp_page.result_headline(accepted) == "Accepted · Spring 2027"
    assert opp_page.membership_offer("Example University", ["4-H"]) == "Example University"
    assert opp_page.membership_offer("Example University", ["example university"]) is None
    assert opp_page.membership_offer("", []) is None


def test_follow_up_refs_never_collide_with_the_apply_todo(
    conn: sqlite3.Connection, sid: str
) -> None:
    apply_todo = _apply_todo(conn, sid)
    row = experiences.create_experience(
        conn, sid, "Example Camp", catalog_id=PROGRAM, starts_on="2027-06-14", ends_on="2027-07-09"
    )
    experiences.set_status(conn, row.id, "accepted")
    accepted = experiences.get_experience(conn, row.id)
    assert accepted is not None
    todos.sync_program_todos(conn, accepted)
    todos.sync_program_todos(conn, accepted)

    found = todos.find_todo(conn, sid, "opportunity", PROGRAM)
    assert found is not None and found.id == apply_todo.id
    refs = {t.source_ref: t for t in todos.list_todos(conn, sid)}
    assert set(refs) == {PROGRAM, f"{PROGRAM}:start", f"{PROGRAM}:finish"}
    assert refs[f"{PROGRAM}:start"].title == "Start Example Camp"
    assert refs[f"{PROGRAM}:start"].due_on == "2027-06-14"

    experiences.update_experience(conn, row.id, ends_on=None)
    cleared = experiences.get_experience(conn, row.id)
    assert cleared is not None
    todos.sync_program_todos(conn, cleared)
    assert todos.find_todo(conn, sid, "opportunity", f"{PROGRAM}:finish") is None


def test_programs_stay_out_of_the_money_views(conn: sqlite3.Connection, sid: str) -> None:
    todo = _apply_todo(conn, sid)
    _tick(conn, todo, True)
    (row,) = experiences.list_experiences(conn, sid)
    experiences.set_status(conn, row.id, "accepted")
    experiences.set_status(conn, row.id, "completed")

    assert repo.list_applications(conn, sid) == []
    assert repo.list_outcomes(conn, sid) == []
    snapshot = pd.DataFrame([{"catalog_id": PROGRAM, "kind": "program", "title": "Example Camp"}])
    assert build_outcomes_frame(conn, sid, {}, snapshot, today=TODAY).empty


# -- Activities & Honors ------------------------------------------------------


def _honors(conn: sqlite3.Connection, sid: str) -> list[experiences.Experience]:
    camp = experiences.create_experience(
        conn, sid, "Example Camp", catalog_id=PROGRAM, organization="Example University",
        starts_on="2027-06-14", ends_on="2027-07-09", term="Summer 2027",
    )
    experiences.set_status(conn, camp.id, "accepted")
    experiences.create_experience(conn, sid, "Other Camp", catalog_id="other-camp")
    experiences.create_experience(
        conn, sid, "Robotics club", kind="club", starts_on="2025-09-01",
        hours_per_week=3, notes="Built the drive train",
    )
    return experiences.list_experiences(conn, sid)


def test_honors_list_is_hers_newest_first(conn: sqlite3.Connection, sid: str) -> None:
    shown = experiences.honors_list(_honors(conn, sid))
    assert [e.title for e in shown] == ["Example Camp", "Robotics club"]


def test_brag_sheet_is_plain_text(conn: sqlite3.Connection, sid: str) -> None:
    text = experiences.honors_text(_honors(conn, sid), name="Test Student", today=TODAY)
    assert text.splitlines()[:2] == ["Activities & Honors: Test Student", "Updated Nov 13, 2026"]
    assert "Example University · Program · Accepted · Summer 2027" in text
    assert "Jun 14, 2027 – Jul 9, 2027" in text
    assert "Club · Active" in text and "3 hours/week" in text
    assert "Other Camp" not in text
    assert experiences.honors_text([], today=TODAY).endswith("Nothing listed yet.\n")


def test_parent_week_shows_the_acceptance_without_notes(
    conn: sqlite3.Connection, sid: str
) -> None:
    rows = _honors(conn, sid)
    experiences.update_experience(conn, rows[0].id, notes="private thoughts")
    recent = experiences.recently_accepted(
        experiences.list_experiences(conn, sid), date.today(), todos_page.DONE_VISIBLE_DAYS
    )
    assert [todos_page.acceptance_line(e) for e in recent] == ["Accepted to Example Camp"]
    assert experiences.recently_accepted(
        experiences.list_experiences(conn, sid), date(2030, 1, 1), 14
    ) == []
