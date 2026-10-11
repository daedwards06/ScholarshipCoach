from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from app.sections import colleges_money
from app.sections import my_path as page
from src.store import repo, requirements, todos
from src.store.db import connect
from src.store.repo import CollegeRequirement

# A fall day of her 10th-grade year (school year ending 2027).
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
def college(conn: sqlite3.Connection, sid: str) -> repo.College:
    made_up = repo.create_college(conn, sid, "Example State University")
    repo.set_first_target(conn, sid, made_up.id)
    return made_up


def _req(**values: object) -> CollegeRequirement:
    fields: dict = {"id": 1, "college_id": 1, "label": "x"}
    fields.update(values)
    return CollegeRequirement(**fields)


# -- grouping -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("due_by", "expected"),
    [
        ("10", "this_semester"),
        ("9", "this_semester"),
        ("11", "next_year"),
        ("12", "senior_year"),
        ("11th grade", "next_year"),
        ("2026-12-01", "this_semester"),
        ("2027-05-01", "this_semester"),
        ("2027-09-15", "next_year"),
        ("2028-11-01", "senior_year"),
        ("", "this_semester"),
    ],
)
def test_requirements_group_by_grade_and_date(due_by: str, expected: str) -> None:
    assert requirements.requirement_group(_req(due_by=due_by), "10", TODAY) == expected


@pytest.mark.parametrize("status", ["met", "not_needed"])
def test_met_and_not_needed_rows_are_done(status: str) -> None:
    assert requirements.requirement_group(_req(due_by="11", status=status), "10", TODAY) == "done"


def test_grade_rows_without_a_profile_grade_do_not_guess() -> None:
    assert requirements.requirement_group(_req(due_by="12"), None, TODAY) == "senior_year"
    assert requirements.requirement_group(_req(due_by="11"), None, TODAY) == "this_semester"


def test_an_eleventh_grader_sees_senior_rows_as_next_year() -> None:
    assert requirements.requirement_group(_req(due_by="12"), "11", TODAY) == "next_year"


@pytest.mark.parametrize(
    ("raw", "stored"),
    [("11", "11"), ("Grade 12", "12"), ("2027-03-01", "2027-03-01"), ("", ""), ("soon", None)],
)
def test_due_by_is_a_grade_or_a_date(raw: str, stored: str | None) -> None:
    assert requirements.normalize_due_by(raw) == stored


def test_due_by_text_reads_as_words() -> None:
    assert requirements.due_by_text("11", TODAY) == "By 11th grade"
    assert requirements.due_by_text("2027-03-01", TODAY) == "By Mar 1"
    assert requirements.due_by_text("2028-03-01", TODAY) == "By Mar 1, 2028"
    assert requirements.due_by_text("", TODAY) == ""


# -- GPA ----------------------------------------------------------------------


def test_gpa_gap_text() -> None:
    row = _req(category="gpa", target="GPA 3.5+")
    assert requirements.gpa_gap_text(row, 3.25) == "You're at 3.25 · 0.25 to go"
    assert requirements.gpa_gap_text(row, 3.6) == "You're at 3.6 · you meet this"
    assert requirements.gpa_gap_text(row, None) is None
    assert requirements.gpa_gap_text(_req(category="gpa", target="strong"), 3.25) is None
    assert requirements.gpa_gap_text(_req(category="course", target="3.5"), 3.25) is None


def test_gpa_rows_follow_her_gpa_in_the_progress_count() -> None:
    rows = [
        _req(id=1, category="gpa", target="3.0"),
        _req(id=2, category="gpa", target="3.5", status="met"),
        _req(id=3, category="test", status="not_needed"),
        _req(id=4, category="course", status="met"),
    ]
    shown = requirements.with_effective_status(rows, 3.25)
    assert [row.status for row in shown] == ["met", "in_progress", "not_needed", "met"]
    met, counted = repo.requirement_progress(shown)
    assert requirements.progress_text(met, counted) == "2 of 3 on track"


# -- who can change what ------------------------------------------------------


def test_she_sets_status_on_course_and_test_rows_only() -> None:
    assert page.can_set_status(_req(category="course"), "student")
    assert page.can_set_status(_req(category="test"), "student")
    assert not page.can_set_status(_req(category="gpa"), "student")
    assert not page.can_set_status(_req(category="application"), "student")
    assert not page.can_set_status(_req(category="test", status="not_needed"), "student")


@pytest.mark.parametrize("mode", ["parent", "operator"])
def test_parent_view_is_read_only(mode: str) -> None:
    assert not page.can_set_status(_req(category="course"), mode)  # type: ignore[arg-type]


def test_heading_speaks_to_the_reader() -> None:
    assert page.path_heading("Example State", "student") == "Your path to Example State"
    assert page.path_heading("Example State", "parent") == "Her path to Example State"


# -- first target -------------------------------------------------------------


def test_one_first_target_at_a_time(conn: sqlite3.Connection, sid: str) -> None:
    first = repo.create_college(conn, sid, "Example State University")
    second = repo.create_college(conn, sid, "Sample Tech")
    repo.set_first_target(conn, sid, first.id)
    repo.set_first_target(conn, sid, second.id)

    target = repo.first_target_college(conn, sid)
    assert target is not None and target.id == second.id
    assert repo.get_college(conn, first.id).priority is None  # type: ignore[union-attr]

    repo.set_first_target(conn, sid, None)
    assert repo.first_target_college(conn, sid) is None


# -- requirement -> to-do -----------------------------------------------------


def _requirement_todos(conn: sqlite3.Connection, sid: str) -> list[todos.Todo]:
    return [t for t in todos.list_todos(conn, sid) if t.source_kind == "requirement"]


def test_dated_requirements_land_on_todos_once(
    conn: sqlite3.Connection, sid: str, college: repo.College
) -> None:
    dated = repo.create_requirement(conn, college.id, "Send transcript", due_by="2026-11-01")
    repo.create_requirement(conn, college.id, "Precalculus", "course", due_by="11")

    requirements.sync_requirement_todos(conn, sid, today=TODAY)
    requirements.sync_requirement_todos(conn, sid, today=TODAY)

    found = _requirement_todos(conn, sid)
    assert [(t.title, t.due_on, t.source_ref) for t in found] == [
        ("Send transcript", "2026-11-01", str(dated.id))
    ]
    assert found[0].assignee == "student"


def test_requirement_edits_reach_the_todo(
    conn: sqlite3.Connection, sid: str, college: repo.College
) -> None:
    item = repo.create_requirement(conn, college.id, "Send transcript", due_by="2026-11-01")
    requirements.sync_requirement_todos(conn, sid, today=TODAY)

    repo.update_requirement(conn, item.id, label="Send final transcript", due_by="2026-11-15")
    requirements.sync_requirement_todos(conn, sid, today=TODAY)
    (row,) = _requirement_todos(conn, sid)
    assert (row.title, row.due_on, row.done) == ("Send final transcript", "2026-11-15", False)

    repo.update_requirement(conn, item.id, status="met")
    requirements.sync_requirement_todos(conn, sid, today=TODAY)
    assert _requirement_todos(conn, sid)[0].done

    repo.delete_requirement(conn, item.id)
    requirements.sync_requirement_todos(conn, sid, today=TODAY)
    assert _requirement_todos(conn, sid)[0].done, "a done row stays as history"


def test_open_rows_leave_when_the_school_is_no_longer_first(
    conn: sqlite3.Connection, sid: str, college: repo.College
) -> None:
    repo.create_requirement(conn, college.id, "Send transcript", due_by="2026-11-01")
    requirements.sync_requirement_todos(conn, sid, today=TODAY)
    repo.set_first_target(conn, sid, None)
    requirements.sync_requirement_todos(conn, sid, today=TODAY)
    assert _requirement_todos(conn, sid) == []


def test_ticking_the_todo_meets_the_requirement(
    conn: sqlite3.Connection, sid: str, college: repo.College
) -> None:
    item = repo.create_requirement(conn, college.id, "Register for the SAT", "test", due_by="2027-02-01")
    requirements.sync_requirement_todos(conn, sid, today=TODAY)
    row = next(
        r for r in todos.assemble_rows(conn, sid, "student", today=TODAY)
        if r.source_kind == "requirement"
    )

    todos.set_row_done(conn, row, True, today=TODAY)
    assert repo.get_requirement(conn, item.id).status == "met"  # type: ignore[union-attr]

    todos.set_row_done(conn, row, False, today=TODAY)
    assert repo.get_requirement(conn, item.id).status == "in_progress"  # type: ignore[union-attr]
    requirements.sync_requirement_todos(conn, sid, today=TODAY)
    assert not _requirement_todos(conn, sid)[0].done


def test_coming_up_is_the_next_three_open_dated_rows() -> None:
    def row(key: str, due: str | None, done: str | None = None) -> todos.TodoRow:
        return todos.TodoRow(key=key, kind="todo", ref_id=0, title=key, due_on=due, done_on=done)

    rows = [
        row("later", "2027-01-01"),
        row("someday", None),
        row("finished", "2026-10-11", done="2026-10-09"),
        row("overdue", "2026-10-01"),
        row("soon", "2026-10-20"),
        row("next", "2026-11-05"),
    ]
    assert [r.key for r in page.coming_up(rows)] == ["overdue", "soon", "next"]


# -- the Colleges & Money requirements table ----------------------------------


def test_requirement_table_round_trip_is_a_no_op(
    conn: sqlite3.Connection, college: repo.College
) -> None:
    repo.create_requirement(conn, college.id, "Precalculus", "course", due_by="11")
    repo.create_requirement(
        conn, college.id, "GPA", "gpa", target="3.0", verified_on="2026-10-01", position=1
    )
    items = repo.list_requirements(conn, college.id)

    edits = colleges_money.requirement_edits(items, colleges_money.requirement_records(items))
    assert (edits.creates, edits.updates, edits.deletes, edits.errors) == ([], [], [], [])


def test_requirement_table_adds_edits_reorders_and_removes(
    conn: sqlite3.Connection, college: repo.College
) -> None:
    first = repo.create_requirement(conn, college.id, "Precalculus", "course", due_by="11")
    second = repo.create_requirement(conn, college.id, "GPA", "gpa", target="3.0", position=1)
    third = repo.create_requirement(conn, college.id, "Old row", position=2)
    records = colleges_money.requirement_records(repo.list_requirements(conn, college.id))

    records[0]["order"] = 5
    records[1]["verified_on"] = date(2026, 10, 10)
    del records[2]
    records.append({"id": float("nan"), "order": float("nan"), "label": "Essay", "due_by": "12"})

    edits = colleges_money.requirement_edits(repo.list_requirements(conn, college.id), records)

    assert edits.deletes == [third.id]
    assert dict(edits.updates) == {
        second.id: {"verified_on": "2026-10-10", "position": 0},
        first.id: {"position": 1},
    }
    assert [(c["label"], c["due_by"], c["position"]) for c in edits.creates] == [("Essay", "12", 2)]


def test_requirement_table_refuses_an_unreadable_due_by(college: repo.College) -> None:
    edits = colleges_money.requirement_edits(
        [], [{"id": None, "order": 1, "label": "Essay", "due_by": "sometime"}]
    )
    assert edits.creates == [] and len(edits.errors) == 1
