"""Export real application outcomes as training and evaluation data.

The win model has only ever seen synthetic labels.  Every application the
student actually submits produces the label the project has never had -- won or
lost, on a pair the ranker scored.  This script writes those pairs in the
feature contract the win model already uses (``FEATURE_COLUMNS``), so a later
model can be trained or evaluated on them without re-deriving anything.

Features are rebuilt *as of the submission date*, not today: a deadline that
has since passed was weeks away when the student applied, and the present-day
value would leak the outcome into the features.

The output lands under ``data/private/`` because a real student's application
history is personal data.  Rows whose award is no longer in the snapshot keep
empty feature columns and ``features_available = False`` rather than zeros --
a zero-filled row is a wrong training example, not a missing one.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.io.snapshotting import get_latest_snapshot_path
from src.profile.store import (
    DEFAULT_STUDENT_ID,
    load_profile_or_demo,
    to_stage1_profile,
    to_stage2_profile,
)
from src.rank.stage2_scoring import score_stage2
from src.store import repo
from src.store.db import DEFAULT_DB_PATH, open_db
from src.win_model.features import FEATURE_COLUMNS, build_pair_features

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "private" / "eval" / "outcomes.csv"

IDENTITY_COLUMNS: tuple[str, ...] = (
    "student_id",
    "catalog_id",
    "title",
    "cycle_year",
    "submitted",
    "submitted_on",
    "status",
    "result",
    "amount_awarded",
    "decided_on",
    "features_available",
)

OUTCOME_COLUMNS: tuple[str, ...] = (*IDENTITY_COLUMNS, *FEATURE_COLUMNS)

# Statuses that mean the student actually applied.  A saved or skipped award
# never became a trial, so it is not a label either way.
SUBMITTED_STATUSES: frozenset[str] = frozenset({"submitted", "won", "lost"})

DECIDED_RESULTS: frozenset[str] = frozenset({"won", "lost"})


def _parse_date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _row_deadline(row: pd.Series) -> date | None:
    deadline = pd.to_datetime(row.get("deadline"), errors="coerce")
    return None if pd.isna(deadline) else deadline.date()


def _snapshot_index(snapshot_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Map both ``catalog_id`` and ``scholarship_id`` to their snapshot row.

    The app saves whichever id the ranked row carried, so an application joins
    back on either one.
    """
    index: dict[str, pd.Series] = {}
    for _, row in snapshot_df.iterrows():
        for column in ("scholarship_id", "catalog_id"):
            key = str(row.get(column) or "").strip()
            if key:
                index.setdefault(key, row)
    return index


def _is_submitted(application: repo.Application) -> bool:
    return bool(application.submitted_on) or application.status in SUBMITTED_STATUSES


def _cycle_year(
    application: repo.Application, row: pd.Series | None, submitted_on: date | None
) -> int | None:
    """The award cycle this attempt belongs to, keyed on its deadline."""
    deadline = _parse_date(application.deadline)
    if deadline is None and row is not None:
        deadline = _row_deadline(row)
    if deadline is not None:
        return deadline.year
    if submitted_on is not None:
        return submitted_on.year
    created = _parse_date(application.created_at)
    return None if created is None else created.year


def build_outcomes_frame(
    conn: sqlite3.Connection,
    student_id: str,
    profile: dict[str, Any],
    snapshot_df: pd.DataFrame,
    *,
    today: date | None = None,
) -> pd.DataFrame:
    """Return one row per submitted application, with its features at submission.

    Args:
        conn: Open family database connection.
        student_id: Student whose applications are exported.
        profile: Stored profile dict for the same student.
        snapshot_df: Scholarship snapshot the features are rebuilt from; may be
            empty, in which case every row's feature columns are left blank.
        today: Fallback reference date for applications with no submission date.

    Returns:
        DataFrame with :data:`OUTCOME_COLUMNS`, one row per submitted application.
    """
    applications = [
        application
        for application in repo.list_applications(conn, student_id)
        if _is_submitted(application)
    ]
    if not applications:
        return pd.DataFrame(columns=list(OUTCOME_COLUMNS))

    scored_df = snapshot_df if snapshot_df.empty else score_stage2(
        snapshot_df, to_stage2_profile(profile)
    )
    index = _snapshot_index(scored_df)

    stage1_profile = to_stage1_profile(profile)
    reference_today = today or date.today()

    rows: list[dict[str, Any]] = []
    for application in applications:
        outcome = repo.get_outcome(conn, application.id)
        submitted_on = _parse_date(application.submitted_on)
        row = index.get(application.catalog_id)

        record: dict[str, Any] = {
            "student_id": student_id,
            "catalog_id": application.catalog_id,
            "title": application.title,
            "cycle_year": _cycle_year(application, row, submitted_on),
            "submitted": True,
            "submitted_on": application.submitted_on or "",
            "status": application.status,
            "result": outcome.result if outcome is not None else "pending",
            "amount_awarded": None if outcome is None else outcome.amount_awarded,
            "decided_on": (outcome.decided_on or "") if outcome is not None else "",
            "features_available": row is not None,
        }
        if row is None:
            record.update({name: None for name in FEATURE_COLUMNS})
        else:
            record.update(
                build_pair_features(
                    stage1_profile,
                    row,
                    row,
                    today=submitted_on or reference_today,
                )
            )
        rows.append(record)

    return pd.DataFrame(rows, columns=list(OUTCOME_COLUMNS))


def resolve_snapshot(snapshot: Path | None, processed_dir: Path) -> Path | None:
    if snapshot is not None:
        return snapshot if snapshot.is_absolute() else ROOT_DIR / snapshot
    resolved_dir = processed_dir if processed_dir.is_absolute() else ROOT_DIR / processed_dir
    return get_latest_snapshot_path(resolved_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export submitted applications and their outcomes as labeled pairs."
    )
    parser.add_argument("--student-id", default=DEFAULT_STUDENT_ID)
    parser.add_argument(
        "--db", type=Path, default=None, help=f"Family database (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Snapshot parquet the features are rebuilt from; defaults to the latest.",
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--today", default=None, help="Reference date as YYYY-MM-DD (default: today)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.today:
        today = _parse_date(args.today)
        if today is None:
            raise SystemExit(f"--today must be YYYY-MM-DD, got {args.today!r}")
    else:
        today = date.today()

    profile, is_demo = load_profile_or_demo(args.student_id)
    if is_demo:
        print(
            f"No private profile for '{args.student_id}'; using the committed demo profile. "
            "Exported features describe the demo student."
        )

    snapshot_path = resolve_snapshot(args.snapshot, args.processed_dir)
    if snapshot_path is None:
        print("No snapshot parquet found; exporting outcomes without feature columns.")
        snapshot_df = pd.DataFrame()
    else:
        snapshot_df = pd.read_parquet(snapshot_path)

    with open_db(args.db) as conn:
        frame = build_outcomes_frame(conn, args.student_id, profile, snapshot_df, today=today)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")

    decided = int(frame["result"].isin(DECIDED_RESULTS).sum()) if not frame.empty else 0
    unmatched = int((~frame["features_available"]).sum()) if not frame.empty else 0
    print(f"Wrote {len(frame)} submitted application(s), {decided} decided, to {output_path}")
    if unmatched:
        print(f"{unmatched} row(s) had no snapshot match; their feature columns are empty.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
