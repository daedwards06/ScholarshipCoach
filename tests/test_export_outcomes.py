from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from scripts.export_outcomes import OUTCOME_COLUMNS, build_outcomes_frame, main
from src.store import repo, tracker
from src.store.db import connect
from src.win_model.features import FEATURE_COLUMNS

TODAY = date(2026, 9, 13)
STUDENT_ID = "student_1"


@pytest.fixture
def conn(tmp_path: Path):
    connection = connect(tmp_path / "coach.db")
    repo.upsert_student(connection, STUDENT_ID, "Test Student")
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def profile() -> dict:
    return {
        "student_id": STUDENT_ID,
        "gpa": 3.25,
        "state": "NC",
        "major": "Computer Science",
        "education_level": "undergraduate",
        "citizenship": "US",
        "profile_keywords": ["python", "robotics"],
    }


@pytest.fixture
def snapshot_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scholarship_id": "sid-cs",
                "catalog_id": "nc-cs-award",
                "title": "NC Computing Award",
                "description": "Supports computer science students in python and robotics.",
                "eligibility_text": "Open to NC CS undergraduates.",
                "essay_prompt": "Describe a project.",
                "sponsor": "NC Foundation",
                "deadline": date(2026, 11, 1),
                "amount_min": 1000.0,
                "amount_max": 5000.0,
                "essay_required": True,
                "states_allowed": ["NC"],
                "majors_allowed": ["Computer Science"],
                "min_gpa": 3.0,
                "education_level": "Undergraduate",
                "citizenship": "US",
                "keywords": ["python", "robotics"],
                "source": "curated",
            },
            {
                "scholarship_id": "sid-stem",
                "catalog_id": None,
                "title": "General STEM Grant",
                "description": "STEM support for undergraduates.",
                "eligibility_text": "Open to all STEM majors.",
                "essay_prompt": None,
                "sponsor": "STEM Fund",
                "deadline": date(2026, 12, 15),
                "amount_min": 500.0,
                "amount_max": 2000.0,
                "essay_required": False,
                "states_allowed": [],
                "majors_allowed": [],
                "min_gpa": None,
                "education_level": None,
                "citizenship": None,
                "keywords": ["stem"],
                "source": "open_scholarships",
            },
        ]
    )


def _submit(
    conn: sqlite3.Connection, catalog_id: str, title: str, deadline: str, submitted_on: date
) -> repo.Application:
    application = repo.create_application(
        conn, STUDENT_ID, catalog_id, title=title, deadline=deadline
    )
    tracker.set_status(conn, application.id, "submitted", today=submitted_on)
    stored = repo.get_application(conn, application.id)
    assert stored is not None
    return stored


def test_export_is_empty_with_correct_columns_when_nothing_submitted(
    conn: sqlite3.Connection, profile: dict, snapshot_df: pd.DataFrame
) -> None:
    repo.create_application(conn, STUDENT_ID, "nc-cs-award", title="NC Computing Award")

    frame = build_outcomes_frame(conn, STUDENT_ID, profile, snapshot_df, today=TODAY)

    assert frame.empty
    assert list(frame.columns) == list(OUTCOME_COLUMNS)


def test_export_carries_result_amount_and_pair_features(
    conn: sqlite3.Connection, profile: dict, snapshot_df: pd.DataFrame
) -> None:
    application = _submit(
        conn, "nc-cs-award", "NC Computing Award", "2026-11-01", date(2026, 9, 1)
    )
    tracker.record_outcome(
        conn,
        application.id,
        "won",
        amount_awarded=4000.0,
        decided_on="2026-12-05",
        today=TODAY,
    )

    frame = build_outcomes_frame(conn, STUDENT_ID, profile, snapshot_df, today=TODAY)

    assert list(frame.columns) == list(OUTCOME_COLUMNS)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["student_id"] == STUDENT_ID
    assert row["catalog_id"] == "nc-cs-award"
    assert row["cycle_year"] == 2026
    assert bool(row["submitted"]) is True
    assert row["result"] == "won"
    assert row["amount_awarded"] == 4000.0
    assert row["decided_on"] == "2026-12-05"
    assert bool(row["features_available"]) is True
    assert row[list(FEATURE_COLUMNS)].notna().all()
    assert row["gpa"] == pytest.approx(3.25)
    assert row["amount_value"] == pytest.approx(5000.0)


def test_features_use_the_submission_date_not_today(
    conn: sqlite3.Connection, profile: dict, snapshot_df: pd.DataFrame
) -> None:
    application = _submit(
        conn, "nc-cs-award", "NC Computing Award", "2026-11-01", date(2026, 9, 1)
    )
    tracker.record_outcome(conn, application.id, "lost", today=TODAY)

    frame = build_outcomes_frame(
        conn, STUDENT_ID, profile, snapshot_df, today=date(2027, 3, 1)
    )

    # 2026-09-01 to the 2026-11-01 deadline, not zero days from a 2027 "today".
    assert frame.iloc[0]["days_to_deadline"] == pytest.approx(61.0)


def test_pending_and_unmatched_rows_stay_distinguishable(
    conn: sqlite3.Connection, profile: dict, snapshot_df: pd.DataFrame
) -> None:
    _submit(conn, "sid-stem", "General STEM Grant", "2026-12-15", date(2026, 9, 2))
    _submit(conn, "retired-award", "Retired Award", "2026-10-01", date(2026, 9, 3))

    frame = build_outcomes_frame(conn, STUDENT_ID, profile, snapshot_df, today=TODAY)
    by_id = frame.set_index("catalog_id")

    assert by_id.loc["sid-stem", "result"] == "pending"
    assert by_id.loc["sid-stem", "amount_awarded"] is None or pd.isna(
        by_id.loc["sid-stem", "amount_awarded"]
    )
    assert bool(by_id.loc["sid-stem", "features_available"]) is True

    # No snapshot match: features are blank, never zero-filled.
    assert bool(by_id.loc["retired-award", "features_available"]) is False
    assert by_id.loc["retired-award", list(FEATURE_COLUMNS)].isna().all()
    assert by_id.loc["retired-award", "cycle_year"] == 2026


def test_empty_snapshot_still_exports_rows_without_features(
    conn: sqlite3.Connection, profile: dict
) -> None:
    _submit(conn, "nc-cs-award", "NC Computing Award", "2026-11-01", date(2026, 9, 1))

    frame = build_outcomes_frame(conn, STUDENT_ID, profile, pd.DataFrame(), today=TODAY)

    assert len(frame) == 1
    assert bool(frame.iloc[0]["features_available"]) is False


def test_main_writes_csv_to_the_requested_path(
    tmp_path: Path, snapshot_df: pd.DataFrame
) -> None:
    db_path = tmp_path / "coach.db"
    connection = connect(db_path)
    repo.upsert_student(connection, STUDENT_ID, "Test Student")
    application = repo.create_application(
        connection, STUDENT_ID, "nc-cs-award", title="NC Computing Award", deadline="2026-11-01"
    )
    tracker.set_status(connection, application.id, "submitted", today=date(2026, 9, 1))
    tracker.record_outcome(connection, application.id, "won", amount_awarded=4000.0, today=TODAY)
    connection.close()

    snapshot_path = tmp_path / "scholarships_snapshot_20260913.parquet"
    snapshot_df.to_parquet(snapshot_path)
    out_path = tmp_path / "eval" / "outcomes.csv"

    exit_code = main(
        [
            "--student-id",
            STUDENT_ID,
            "--db",
            str(db_path),
            "--snapshot",
            str(snapshot_path),
            "--out",
            str(out_path),
            "--today",
            TODAY.isoformat(),
        ]
    )

    assert exit_code == 0
    written = pd.read_csv(out_path)
    assert list(written.columns) == list(OUTCOME_COLUMNS)
    assert len(written) == 1
    assert written.iloc[0]["result"] == "won"
