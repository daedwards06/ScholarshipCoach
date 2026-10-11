from __future__ import annotations

import json
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.sections import opportunities as page
from src.catalog import entry
from src.ingest.sources.curated_catalog import CuratedCatalogSource
from src.normalize.catalog_schema import validate_catalog_record
from src.rank import opportunities
from src.rank.kinds import program_mask, record_kind, scholarships_only
from src.rank.stage1_eligibility import StudentProfile
from src.rank.stage2_scoring import score_stage2
from src.rank.stage3_rerank import rerank_stage3
from src.rank.whatif import whatif_eligibility
from src.store import repo, todos
from src.store.db import connect
from src.win_model.synthetic import generate_synthetic_training_data

TODAY = date(2026, 10, 11)


def _record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "catalog_id": "example-summer-program",
        "title": "Example Summer Program",
        "source_url": "https://example.org/summer",
        "status": "unknown",
        "trust": "verified_local",
        "provenance": {"added_on": "2026-10-11", "source_kind": "sponsor_site"},
    }
    record.update(overrides)
    return record


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scholarship_id": "row",
        "catalog_id": "row",
        "title": "Row",
        "sponsor": "Sponsor",
        "description": "computer science",
        "eligibility_text": "",
        "keywords": ["computer science"],
        "source_url": "https://example.org/row",
        "deadline": None,
        "amount_min": None,
        "amount_max": None,
        "cost": None,
        "status": "open",
        "cycle": {"recurring": None, "opens_month": None, "deadline_month": None},
        "grade_levels": ["10"],
        "states_allowed": ["NC"],
        "majors_allowed": [],
        "counties_allowed": [],
        "trust": "verified_local",
        "kind": "program",
        "essay_required": None,
    }
    row.update(overrides)
    return row


def _profile(grade: str | None = "10") -> StudentProfile:
    return StudentProfile(
        gpa=3.25, state="NC", major="Computer Science", grade_level=grade, today=TODAY
    )


# -- schema default -----------------------------------------------------------


def test_kind_is_optional_and_limited_to_scholarship_or_program() -> None:
    assert validate_catalog_record(_record()) == []
    assert validate_catalog_record(_record(kind="program", cost=0)) == []
    assert validate_catalog_record(_record(kind="scholarship")) == []
    assert validate_catalog_record(_record(kind="competition"))
    assert validate_catalog_record(_record(cost=-5))


def test_a_record_without_kind_loads_and_maps_as_a_scholarship() -> None:
    assert entry.form_from_record(_record())["kind"] == "scholarship"
    assert entry.kind_options() == ("scholarship", "program")

    payload = json.dumps([_record(), _record(catalog_id="free-camp", kind="program", cost=0)])
    parsed = CuratedCatalogSource().parse(
        payload.encode("utf-8"), fetched_at=datetime(2026, 10, 11, tzinfo=UTC)
    )
    by_id = {row["catalog_id"]: row for row in parsed}
    assert by_id["example-summer-program"]["kind"] == "scholarship"
    assert by_id["free-camp"]["kind"] == "program"
    assert by_id["free-camp"]["cost"] == 0.0


def test_form_round_trips_kind_and_cost() -> None:
    values = entry.form_from_record(_record(kind="program", cost=350))
    assert values["kind"] == "program"
    assert values["cost"] == 350.0

    record, errors = entry.validate_form(values, today=TODAY)
    assert errors == []
    assert record["kind"] == "program"
    assert record["cost"] == 350.0


def test_record_kind_treats_missing_and_unknown_as_scholarship() -> None:
    assert record_kind(None) == "scholarship"
    assert record_kind(float("nan")) == "scholarship"
    assert record_kind(" Program ") == "program"
    assert program_mask(pd.DataFrame([{"title": "no kind column"}])).tolist() == [False]


# -- programs never enter ranking ---------------------------------------------


def test_program_is_excluded_from_stage2_stage3_and_what_if() -> None:
    frame = pd.DataFrame(
        [
            _row(scholarship_id="award", catalog_id="award", kind="scholarship",
                 amount_max=2000.0, deadline=date(2026, 12, 1)),
            _row(scholarship_id="camp", catalog_id="camp", kind="program",
                 deadline=date(2026, 12, 1)),
            _row(scholarship_id="feed", catalog_id=None, kind=None, amount_max=500.0,
                 deadline=date(2026, 12, 1)),
        ]
    )
    assert scholarships_only(frame)["scholarship_id"].tolist() == ["award", "feed"]

    scored = score_stage2(frame, {"major": "Computer Science", "keywords": ["computer"]})
    assert set(scored["scholarship_id"]) == {"award", "feed"}

    reranked = rerank_stage3(frame.assign(stage2_score=0.5), today=TODAY, timeline_bucket=None)
    assert set(reranked["scholarship_id"]) == {"award", "feed"}

    summary = whatif_eligibility(frame.assign(min_gpa=3.5), _profile(), {"gpa": 3.6})
    assert {award.scholarship_id for award in summary.newly_eligible} == {"award", "feed"}


def test_win_model_training_never_samples_a_program() -> None:
    frame = pd.DataFrame(
        [
            _row(scholarship_id="award", kind="scholarship", amount_max=1000.0),
            _row(scholarship_id="camp", kind="program"),
        ]
    )

    class _Golden:
        profile = _profile()
        student_id = "golden_1"
        interests: list[str] = ["computer science"]

    _, _, meta = generate_synthetic_training_data(frame, [_Golden()], n_samples=40, seed=1)
    assert set(meta["scholarship_id"]) == {"award"}


# -- explicit grade and groups ------------------------------------------------


def _classify(rows: list[dict[str, Any]], grade: str | None = "10") -> pd.DataFrame:
    return opportunities.classify_opportunities(pd.DataFrame(rows), _profile(grade), TODAY)


def test_only_records_naming_a_grade_are_listed() -> None:
    result = _classify(
        [
            _row(catalog_id="named", deadline=date(2027, 2, 1)),
            _row(catalog_id="unnamed", grade_levels=[], deadline=date(2027, 2, 1)),
            _row(catalog_id="aged-out", grade_levels=["9"], deadline=date(2027, 2, 1)),
            _row(catalog_id="college", grade_levels=["college_2"], deadline=date(2027, 2, 1)),
        ]
    )
    assert result["catalog_id"].tolist() == ["named"]
    assert result.loc[0, "opportunity_group"] == opportunities.OPEN_NOW


def test_no_profile_grade_lists_nothing() -> None:
    result = _classify([_row(deadline=date(2027, 2, 1))], grade=None)
    assert result.empty


def test_groups_follow_grade_status_and_dates() -> None:
    result = _classify(
        [
            _row(catalog_id="open", deadline=date(2027, 3, 1), status="open"),
            _row(catalog_id="later-grade", grade_levels=["11", "12"], deadline=date(2027, 3, 1)),
            _row(
                catalog_id="opens-jan",
                deadline=date(2027, 3, 1),
                cycle={"recurring": True, "opens_month": 1, "deadline_month": 3},
            ),
            _row(catalog_id="upcoming", deadline=date(2027, 2, 1), status="upcoming"),
            _row(
                catalog_id="passed",
                grade_levels=["10", "11"],
                deadline=date(2026, 9, 1),
                cycle={"recurring": True, "opens_month": None, "deadline_month": 9},
            ),
            _row(
                catalog_id="missed",
                deadline=date(2026, 9, 15),
                cycle={"recurring": True, "opens_month": None, "deadline_month": 9},
            ),
            _row(catalog_id="no-date", status="unknown"),
            _row(catalog_id="unverified", trust="unverified", deadline=date(2027, 3, 1)),
        ]
    )
    groups = dict(zip(result["catalog_id"], result["opportunity_group"], strict=True))
    assert groups == {
        "open": opportunities.OPEN_NOW,
        "later-grade": opportunities.OPENS_LATER,
        "opens-jan": opportunities.OPENS_LATER,
        "upcoming": opportunities.OPENS_LATER,
        "passed": opportunities.PASSED,
        "missed": opportunities.PASSED,
        "no-date": opportunities.OPENS_LATER,
    }
    by_id = result.set_index("catalog_id")
    assert by_id.loc["later-grade", "opens_grade"] == "11"
    assert by_id.loc["opens-jan", "opens_on"] == date(2027, 1, 1)
    assert by_id.loc["passed", "apply_by"] == date(2027, 9, 1)
    assert bool(by_id.loc["passed", "apply_by_projected"]) is True
    # Too old for next year's round: passed, with no next round to offer.
    assert by_id.loc["missed", "apply_by"] is None
    # Dated "opens later" rows come before the ones waiting on a later grade.
    later = result[result["opportunity_group"] == opportunities.OPENS_LATER]["catalog_id"]
    assert later.tolist() == ["opens-jan", "upcoming", "no-date", "later-grade"]


def test_last_years_round_projects_into_this_year_instead_of_passed() -> None:
    result = _classify(
        [
            _row(
                catalog_id="recurring",
                deadline=date(2026, 2, 1),
                status="unknown",
                cycle={"recurring": True, "opens_month": 11, "deadline_month": 2},
            )
        ]
    )
    row = result.iloc[0]
    assert row["opportunity_group"] == opportunities.OPENS_LATER
    assert row["apply_by"] == date(2027, 2, 1)
    assert row["opens_on"] == date(2026, 11, 1)


def test_headline_is_the_next_step_and_its_date() -> None:
    def headline(**fields: Any) -> str:
        return page.headline(pd.Series(fields), TODAY)

    assert headline(opportunity_group="open_now", apply_by=date(2027, 3, 15)) == "Apply by Mar 15"
    assert headline(opportunity_group="open_now", apply_by=None) == "Open now · date not posted"
    assert headline(opportunity_group="opens_later", opens_grade="11") == "Opens in 11th grade"
    assert (
        headline(opportunity_group="opens_later", opens_on=date(2027, 1, 1), apply_by=date(2027, 3, 1))
        == "Opens Jan 1 · due Mar 1"
    )
    assert headline(opportunity_group="opens_later") == "Date not posted yet"
    assert (
        headline(opportunity_group="passed", apply_by=date(2027, 9, 1), apply_by_projected=True)
        == "Passed · next round ~Sep 1, 2027"
    )


def test_card_wording_for_programs_and_grades() -> None:
    assert page.money_badge_text(pd.Series({"is_program": True, "cost": 0.0})) == "Free"
    assert page.money_badge_text(pd.Series({"is_program": True, "cost": 450.0})) == "Costs $450"
    assert page.money_badge_text(pd.Series({"is_program": True, "cost": None})) == "Cost not listed"
    assert (
        page.money_badge_text(pd.Series({"is_program": False, "amount_min": None, "amount_max": 2000}))
        == "Up to $2,000"
    )
    assert page.grade_badge_text(["10", "11"], "10") == "Open to 10th grade"
    assert page.grade_badge_text(["11", "12"], "10") == "For 11th and 12th grade"
    assert page.grade_badge_text(["9", "10", "11", "12"], None) == "For 9th–12th grade"


def test_filters_by_kind_cost_and_place() -> None:
    frame = pd.DataFrame(
        [
            {"catalog_id": "camp", "is_program": True, "cost": 0.0, "counties_allowed": []},
            {"catalog_id": "paid", "is_program": True, "cost": 900.0, "counties_allowed": ["Gaston"]},
            {"catalog_id": "award", "is_program": False, "cost": None, "counties_allowed": []},
        ]
    )
    assert page.apply_filters(frame, kind="program")["catalog_id"].tolist() == ["camp", "paid"]
    assert page.apply_filters(frame, kind="scholarship")["catalog_id"].tolist() == ["award"]
    assert page.apply_filters(frame, free_only=True)["catalog_id"].tolist() == ["camp"]
    assert page.apply_filters(frame, local_only=True)["catalog_id"].tolist() == ["paid"]
    # Statewide is not local, even when the family confirmed the record.
    assert not page.is_local(pd.Series({"states_allowed": ["NC"], "trust": "verified_local"}))


# -- to-do creation -----------------------------------------------------------


@pytest.fixture
def conn(tmp_path: Path):
    connection = connect(tmp_path / "coach.db")
    try:
        yield connection
    finally:
        connection.close()


def test_add_to_my_todos_creates_one_dated_opportunity_row(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo.upsert_student(conn, "student_1")

    class _Db:
        def __enter__(self) -> sqlite3.Connection:
            return conn

        def __exit__(self, *_: object) -> None:
            return None

    class _Rerun(Exception):
        pass

    def _rerun() -> None:
        raise _Rerun

    monkeypatch.setattr(page, "open_db", _Db)
    monkeypatch.setattr(page.state, "ensure_student", lambda _conn: "student_1")
    monkeypatch.setattr(page.identity, "current_identity", lambda: None)
    monkeypatch.setattr(page.st, "toast", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(page.st, "rerun", _rerun)

    row = pd.Series(
        {
            "catalog_id": "example-camp",
            "title": "Example Camp",
            "source_url": "https://example.org/camp",
            "apply_by": date(2027, 2, 1),
            "apply_by_projected": False,
        }
    )
    for _ in range(2):
        with pytest.raises(_Rerun):
            page._add_to_todos(row, "student", TODAY)

    found = [todo for todo in todos.list_todos(conn, "student_1") if todo.source_kind == "opportunity"]
    assert len(found) == 1
    assert found[0].title == "Apply to Example Camp"
    assert found[0].source_ref == "example-camp"
    assert found[0].due_on == "2027-02-01"
    assert found[0].assignee == "student"
    assert found[0].created_role == "student"


def test_todo_plan_dates_a_posted_deadline_or_a_reminder() -> None:
    def plan(**fields: Any) -> tuple[str, str | None]:
        return page.todo_plan(pd.Series({"title": "Camp", **fields}), TODAY, "10")

    assert plan(apply_by=date(2027, 2, 1), apply_by_projected=False) == ("Apply to Camp", "2027-02-01")
    # Not posted yet: look again a month before the usual deadline...
    assert plan(apply_by=date(2027, 2, 1), apply_by_projected=True) == (
        "Check the dates for Camp", "2027-01-02"
    )
    # ...or when it usually opens.
    assert plan(apply_by=date(2027, 2, 1), apply_by_projected=True, opens_on=date(2026, 12, 1)) == (
        "Watch for Camp to open", "2026-12-01"
    )
    # A reminder never lands in the past.
    assert plan(apply_by=date(2026, 10, 20), apply_by_projected=True)[1] == "2026-10-11"
    assert plan(apply_by=None) == ("Apply to Camp", None)


def test_todo_plan_moves_a_later_grade_round_to_that_year() -> None:
    def plan(**fields: Any) -> tuple[str, str | None]:
        return page.todo_plan(pd.Series({"title": "Camp", **fields}), TODAY, "10")

    # This year's posted deadline is for 11th graders; hers is a year later.
    assert plan(opens_grade="11", apply_by=date(2027, 2, 1), apply_by_projected=False) == (
        "Check the dates for Camp (11th grade)", "2028-01-02"
    )
    assert plan(opens_grade="12", opens_on=date(2026, 11, 1)) == (
        "Watch for Camp to open (12th grade)", "2028-11-01"
    )
    assert plan(opens_grade="11") == ("Apply to Camp in 11th grade", None)
    unknown_grade = page.todo_plan(pd.Series({"title": "Camp", "opens_grade": "11"}), TODAY, None)
    assert unknown_grade == ("Apply to Camp in 11th grade", None)


def test_a_parent_suggestion_reads_as_from_your_family() -> None:
    suggested = todos.Todo(
        id=1, student_id="s", title="Apply to Camp", notes="", due_on=None, assignee="student",
        created_by="", created_role="parent", done_on=None, done_by="",
        source_kind="opportunity", source_ref="camp", created_at="", updated_at="",
    )
    row = pd.Series({"grade_levels": ["10"], "is_program": True, "cost": 0.0, "counties_allowed": []})

    student_badges = " ".join(page.card_badges(row, "10", suggested, "student"))
    parent_badges = " ".join(page.card_badges(row, "10", suggested, "parent"))
    assert "From your family" in student_badges
    assert "Suggested" in parent_badges
    assert "From your family" not in " ".join(page.card_badges(row, "10", None, "student"))


def test_money_badge_escapes_dollar_signs() -> None:
    row = pd.Series({"grade_levels": ["12"], "is_program": False, "amount_min": 2000, "amount_max": 4000,
                     "counties_allowed": []})
    assert r"\$2,000 - \$4,000" in " ".join(page.card_badges(row, "12", None, "student"))
