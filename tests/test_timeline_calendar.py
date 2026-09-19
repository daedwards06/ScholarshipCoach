from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from src.profile.grade_levels import grade_level_in_school_year, school_year_end
from src.rank.timeline import TIMELINE_BUCKETS
from src.store import calendar_feed, milestones, repo, tracker
from src.store.db import connect

TODAY = date(2026, 9, 12)
NOW = datetime(2026, 9, 12, 15, 30, tzinfo=timezone.utc)

FAFSA = milestones.Milestone(
    id="fafsa_opens",
    title="FAFSA opens",
    month=10,
    day=1,
    grade_levels=("12", "college_1"),
    note="Check studentaid.gov",
)
PSAT = milestones.Milestone(
    id="psat_window",
    title="PSAT/NMSQT window",
    month=10,
    day=1,
    end_month=10,
    end_day=31,
    grade_levels=("10", "11"),
)
NEW_YEAR_WINDOW = milestones.Milestone(
    id="ed_two",
    title="Early decision II",
    month=12,
    day=1,
    end_month=1,
    end_day=15,
    grade_levels=("12",),
)
EVERYONE = milestones.Milestone(id="open_house", title="Campus open house", month=4, day=10)


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


# -- the shipped milestone file ---------------------------------------------


def test_shipped_milestones_parse_and_carry_grades() -> None:
    rows = milestones.load_default_milestones()
    assert rows, "data/milestones.json should ship a general milestone set"
    assert all(milestone.source == milestones.DEFAULT_SOURCE for milestone in rows)

    by_id = {milestone.id: milestone for milestone in rows}
    assert "fafsa_opens" in by_id
    assert "psat_window" in by_id
    assert by_id["psat_window"].is_window
    assert "12" in by_id["fafsa_opens"].grade_levels


def test_malformed_rows_are_dropped_not_raised(tmp_path: Path) -> None:
    path = tmp_path / "milestones.json"
    path.write_text(
        '{"milestones": [{"id": "ok", "title": "Fine", "month": 3, "day": 1}, '
        '{"id": "bad", "title": "No month"}, {"title": "No id", "month": 1, "day": 1}]}',
        encoding="utf-8",
    )
    rows = milestones.load_default_milestones(path)
    assert [milestone.id for milestone in rows] == ["ok"]


# -- applicability by grade --------------------------------------------------


def test_applies_to_grade_matches_listed_grades_only() -> None:
    assert milestones.applies_to_grade(FAFSA, "12")
    assert not milestones.applies_to_grade(FAFSA, "11")
    assert not milestones.applies_to_grade(FAFSA, None)


def test_milestone_with_no_grades_applies_to_everyone() -> None:
    assert milestones.applies_to_grade(EVERYONE, "9")
    assert milestones.applies_to_grade(EVERYONE, "college_4")
    assert milestones.applies_to_grade(EVERYONE, None)


def test_applicability_follows_the_student_forward(conn: sqlite3.Connection) -> None:
    rows = [FAFSA, PSAT]
    # A sophomore in the 2026-27 year is a senior in 2028-29.
    assert grade_level_in_school_year("10", 2029, today=TODAY) == "12"

    this_year = milestones.occurrences_for_school_year(rows, "10", 2027, today=TODAY)
    assert [item.milestone.id for item in this_year] == ["psat_window"]

    senior_year = milestones.occurrences_for_school_year(rows, "10", 2029, today=TODAY)
    assert [item.milestone.id for item in senior_year] == ["fafsa_opens"]


def test_graduated_school_year_has_no_milestones() -> None:
    # A college senior in 2026-27 has left the sequence by 2027-28.
    assert milestones.occurrences_for_school_year([EVERYONE], "college_4", 2028, today=TODAY) == []


def test_unknown_grade_keeps_every_milestone() -> None:
    found = milestones.occurrences_for_school_year([FAFSA, PSAT], "", 2027, today=TODAY)
    assert {item.milestone.id for item in found} == {"fafsa_opens", "psat_window"}


# -- landing a milestone on a school year ------------------------------------


def test_occurrence_splits_the_school_year_at_july() -> None:
    autumn = milestones.occurrence(FAFSA, 2027)
    assert autumn.starts_on == date(2026, 10, 1)

    spring = milestones.occurrence(EVERYONE, 2027)
    assert spring.starts_on == date(2027, 4, 10)


def test_window_crossing_the_new_year_ends_in_the_later_year() -> None:
    occurrence = milestones.occurrence(NEW_YEAR_WINDOW, 2027)
    assert occurrence.starts_on == date(2026, 12, 1)
    assert occurrence.ends_on == date(2027, 1, 15)
    assert occurrence.is_window


def test_occurrence_clamps_a_day_the_month_does_not_have() -> None:
    leap_day = milestones.Milestone(id="leap", title="Leap", month=2, day=30)
    assert milestones.occurrence(leap_day, 2027).starts_on == date(2027, 2, 28)


def test_school_year_label_reads_the_way_a_family_says_it() -> None:
    assert milestones.school_year_label(2027) == "2026-27"
    assert milestones.school_year_label(2030) == "2029-30"


# -- family overrides --------------------------------------------------------


def test_overrides_hide_edit_and_add(conn: sqlite3.Connection) -> None:
    defaults = [FAFSA, PSAT]
    resolved = milestones.apply_overrides(
        defaults,
        [
            {"id": "psat_window", "hidden": True},
            {"id": "fafsa_opens", "day": 15},
            {"id": "family_night", "title": "Scholarship night", "month": 3, "day": 4},
        ],
    )
    by_id = {milestone.id: milestone for milestone in resolved}
    assert "psat_window" not in by_id
    assert by_id["fafsa_opens"].day == 15
    assert by_id["fafsa_opens"].title == "FAFSA opens"
    assert by_id["fafsa_opens"].grade_levels == ("12", "college_1")
    assert by_id["family_night"].source == milestones.FAMILY_SOURCE


def test_overrides_round_trip_through_settings(conn: sqlite3.Connection) -> None:
    assert milestones.load_overrides(conn) == []
    milestones.save_overrides(conn, [{"id": "fafsa_opens", "hidden": True}])
    assert milestones.load_overrides(conn) == [{"id": "fafsa_opens", "hidden": True}]

    hidden = {milestone.id for milestone in milestones.load_milestones(conn)}
    assert "fafsa_opens" not in hidden
    assert "fafsa_opens" in {m.id for m in milestones.load_default_milestones()}


def test_unreadable_override_setting_falls_back_to_defaults(conn: sqlite3.Connection) -> None:
    repo.set_setting(conn, milestones.FAMILY_MILESTONES_SETTING, "{not json")
    assert milestones.load_overrides(conn) == []
    assert milestones.load_milestones(conn) == milestones.load_default_milestones()


def test_family_milestone_id_is_a_slug_and_avoids_collisions() -> None:
    assert milestones.family_milestone_id("District scholarship night") == (
        "family_district_scholarship_night"
    )
    taken = {"family_night"}
    assert milestones.family_milestone_id("Night", taken=taken) == "family_night_2"


# -- buckets and grouping ----------------------------------------------------


def test_bucket_for_date_counts_school_years() -> None:
    assert calendar_feed.bucket_for_date(date(2026, 11, 1), TODAY) == "now"
    assert calendar_feed.bucket_for_date(date(2027, 5, 1), TODAY) == "now"
    assert calendar_feed.bucket_for_date(date(2027, 9, 1), TODAY) == "next_cycle"
    assert calendar_feed.bucket_for_date(date(2029, 3, 1), TODAY) == "senior_year"


def test_an_overdue_date_stays_in_this_year() -> None:
    assert calendar_feed.bucket_for_date(date(2026, 8, 1), TODAY) == "now"
    assert calendar_feed.bucket_for_date(date(2025, 12, 1), TODAY) == "now"


def test_group_by_month_and_school_year_are_ordered() -> None:
    events = [
        calendar_feed.CalendarEvent("b", "milestone", "Later", date(2027, 10, 1)),
        calendar_feed.CalendarEvent("a", "milestone", "Sooner", date(2026, 10, 1)),
        calendar_feed.CalendarEvent("c", "milestone", "Middle", date(2027, 3, 1)),
    ]
    assert list(calendar_feed.group_by_school_year(events)) == [2027, 2028]
    assert list(calendar_feed.group_by_month(events)) == [(2026, 10), (2027, 3), (2027, 10)]


def test_needs_date_awards_never_reach_the_calendar() -> None:
    # The app walks CALENDAR_BUCKETS to turn awards into events; an award with
    # no date on record has nothing to put on a day.
    assert "needs_date" in TIMELINE_BUCKETS
    assert "needs_date" not in calendar_feed.CALENDAR_BUCKETS

    events = [
        calendar_feed.CalendarEvent("dated", "award", "Dated", date(2026, 11, 1)),
        calendar_feed.CalendarEvent(
            "undated", "award", "Undated", date(2026, 11, 1), bucket="needs_date"
        ),
    ]
    exported = [
        event
        for bucket in calendar_feed.CALENDAR_BUCKETS
        for event in calendar_feed.events_in_bucket(events, bucket)
    ]

    assert [event.uid for event in exported] == ["dated"]
    assert "Undated" not in calendar_feed.to_ics(exported, now=NOW)


# -- the family feed ---------------------------------------------------------


def _saved_award(conn: sqlite3.Connection, student_id: str) -> repo.Application:
    application, _ = tracker.save_award(
        conn,
        student_id,
        "nc-cs-award-2026",
        title="NC CS Award",
        deadline="2026-11-15",
        requirements={"essay_prompts": ["Why computer science?"], "recommendation_letters": 1},
    )
    return application


def test_family_events_cover_deadlines_tasks_letters_and_milestones(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    application = _saved_award(conn, student.student_id)
    item = repo.list_checklist_items(conn, application.id)[0]
    repo.update_checklist_item(conn, item.id, due_on="2026-10-20")
    recommender = repo.create_recommender(conn, student.student_id, "Ms. Reyes", role="Teacher")
    repo.create_recommendation_request(
        conn, application.id, recommender.id, due_on="2026-11-01"
    )

    events = calendar_feed.family_events(
        conn, student.student_id, grade_level="12", today=TODAY, milestones=[FAFSA, PSAT]
    )
    kinds = {event.kind for event in events}
    assert kinds == {"deadline", "checklist", "letter", "milestone"}
    assert [event.starts_on for event in events] == sorted(event.starts_on for event in events)

    milestone_titles = {event.title for event in events if event.kind == "milestone"}
    assert milestone_titles == {"FAFSA opens"}  # a senior gets no PSAT window


def test_a_submitted_application_leaves_the_calendar(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    application = _saved_award(conn, student.student_id)
    assert calendar_feed.application_events(conn, student.student_id, today=TODAY)

    tracker.set_status(conn, application.id, "submitted", today=TODAY)
    assert calendar_feed.application_events(conn, student.student_id, today=TODAY) == []


def test_a_done_task_leaves_the_calendar(
    conn: sqlite3.Connection, student: repo.Student
) -> None:
    application = _saved_award(conn, student.student_id)
    item = repo.list_checklist_items(conn, application.id)[0]
    repo.update_checklist_item(conn, item.id, due_on="2026-10-20")
    assert any(
        event.kind == "checklist"
        for event in calendar_feed.application_events(conn, student.student_id, today=TODAY)
    )

    repo.update_checklist_item(conn, item.id, done=True)
    assert not any(
        event.kind == "checklist"
        for event in calendar_feed.application_events(conn, student.student_id, today=TODAY)
    )


def test_milestone_events_follow_the_grade_across_the_horizon() -> None:
    # FAFSA applies to grade 12 and college_1, so a senior sees it twice.
    senior = calendar_feed.milestone_events([FAFSA], "12", today=TODAY, horizon_years=3)
    assert [event.starts_on for event in senior] == [date(2026, 10, 1), date(2027, 10, 1)]
    assert [event.bucket for event in senior] == ["now", "next_cycle"]

    # A sophomore only reaches those grades later in the horizon.
    sophomore = calendar_feed.milestone_events([FAFSA], "10", today=TODAY, horizon_years=3)
    assert [event.starts_on for event in sophomore] == [date(2028, 10, 1), date(2029, 10, 1)]
    assert {event.bucket for event in sophomore} == {"senior_year"}


def test_milestone_events_stop_once_the_student_graduates() -> None:
    graduating = calendar_feed.milestone_events(
        [EVERYONE], "college_4", today=TODAY, horizon_years=3
    )
    assert [event.starts_on for event in graduating] == [date(2027, 4, 10)]


# -- ICS ---------------------------------------------------------------------


def _ics_lines(text: str) -> list[str]:
    return text.split("\r\n")


def _unfolded(text: str) -> list[str]:
    """Join ICS continuation lines back into the content lines they came from."""
    lines: list[str] = []
    for line in _ics_lines(text):
        if line.startswith(" ") and lines:
            lines[-1] += line[1:]
        else:
            lines.append(line)
    return lines


def test_ics_wraps_events_in_a_calendar() -> None:
    event = calendar_feed.CalendarEvent(
        uid="application-1",
        kind="deadline",
        title="NC CS Award",
        starts_on=date(2026, 11, 15),
        detail="Application deadline",
    )
    lines = _ics_lines(calendar_feed.to_ics([event], now=NOW))

    assert lines[0] == "BEGIN:VCALENDAR"
    assert "VERSION:2.0" in lines
    assert f"PRODID:{calendar_feed.PRODUCT_ID}" in lines
    assert lines[-2] == "END:VCALENDAR"
    assert calendar_feed.to_ics([event], now=NOW).endswith("\r\n")

    assert f"UID:application-1@{calendar_feed.UID_DOMAIN}" in lines
    assert "DTSTAMP:20260912T153000Z" in lines
    assert "SUMMARY:Application deadline: NC CS Award" in lines
    assert "DESCRIPTION:Application deadline" in lines


def test_ics_all_day_end_is_exclusive() -> None:
    single = calendar_feed.CalendarEvent(
        uid="one", kind="milestone", title="FAFSA opens", starts_on=date(2026, 10, 1)
    )
    lines = _ics_lines(calendar_feed.to_ics([single], now=NOW))
    assert "DTSTART;VALUE=DATE:20261001" in lines
    assert "DTEND;VALUE=DATE:20261002" in lines

    window = calendar_feed.CalendarEvent(
        uid="two",
        kind="milestone",
        title="PSAT window",
        starts_on=date(2026, 10, 1),
        ends_on=date(2026, 10, 31),
    )
    window_lines = _ics_lines(calendar_feed.to_ics([window], now=NOW))
    assert "DTSTART;VALUE=DATE:20261001" in window_lines
    assert "DTEND;VALUE=DATE:20261101" in window_lines


def test_ics_escapes_text_values() -> None:
    event = calendar_feed.CalendarEvent(
        uid="three",
        kind="checklist",
        title="Essay: money, time; and a\\slash",
        starts_on=date(2026, 10, 1),
        detail="Line one\nline two",
    )
    text = calendar_feed.to_ics([event], now=NOW)
    assert "SUMMARY:Task: Essay: money\\, time\\; and a\\\\slash" in _ics_lines(text)
    assert "DESCRIPTION:Line one\\nline two" in _ics_lines(text)


def test_ics_folds_long_lines_at_75_octets() -> None:
    event = calendar_feed.CalendarEvent(
        uid="four", kind="milestone", title="A" * 200, starts_on=date(2026, 10, 1)
    )
    lines = _ics_lines(calendar_feed.to_ics([event], now=NOW))
    assert all(len(line.encode("utf-8")) <= 75 for line in lines)

    assert any(line.startswith(" A") for line in lines)
    assert f"SUMMARY:Milestone: {'A' * 200}" in _unfolded(calendar_feed.to_ics([event], now=NOW))


def test_ics_events_are_ordered_by_date() -> None:
    events = [
        calendar_feed.CalendarEvent("late", "milestone", "Later", date(2026, 12, 1)),
        calendar_feed.CalendarEvent("early", "milestone", "Sooner", date(2026, 10, 1)),
    ]
    lines = _ics_lines(calendar_feed.to_ics(events, now=NOW))
    assert lines.index("DTSTART;VALUE=DATE:20261001") < lines.index("DTSTART;VALUE=DATE:20261201")


def test_empty_calendar_is_still_valid() -> None:
    lines = _ics_lines(calendar_feed.to_ics([], now=NOW))
    assert lines[0] == "BEGIN:VCALENDAR"
    assert "BEGIN:VEVENT" not in lines


def test_export_script_writes_a_calendar(
    conn: sqlite3.Connection, student: repo.Student, tmp_path: Path, monkeypatch
) -> None:
    from scripts import export_calendar

    _saved_award(conn, student.student_id)
    conn.commit()
    monkeypatch.setattr(export_calendar, "load_profile", lambda student_id: {"grade_level": "12"})

    out_path = tmp_path / "out" / "family.ics"
    exit_code = export_calendar.main(
        [
            "--student-id",
            student.student_id,
            "--db",
            str(tmp_path / "coach.db"),
            "--out",
            str(out_path),
            "--today",
            TODAY.isoformat(),
        ]
    )

    assert exit_code == 0
    # Read the bytes: universal newlines would rewrite the CRLF the format needs.
    text = out_path.read_bytes().decode("utf-8")
    assert "\r\n" in text
    assert text.startswith("BEGIN:VCALENDAR")
    assert "SUMMARY:Application deadline: NC CS Award" in _ics_lines(text)
    assert "SUMMARY:Milestone: FAFSA opens" in _ics_lines(text)


def test_school_year_end_splits_at_july() -> None:
    assert school_year_end(date(2026, 6, 30)) == 2026
    assert school_year_end(date(2026, 7, 1)) == 2027
