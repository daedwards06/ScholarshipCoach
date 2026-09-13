"""Export the family calendar -- saved applications and milestones -- as an ICS file.

The app has a download button; this is the same feed for a family that would
rather run it on a schedule and drop the file where a phone picks it up.
Everything is all-day, and the output is deterministic apart from ``DTSTAMP``,
so re-exporting over the same file produces the same calendar.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from src.profile.store import DEFAULT_STUDENT_ID, load_profile
from src.store.calendar_feed import DEFAULT_HORIZON_YEARS, family_events, to_ics
from src.store.db import DEFAULT_DB_PATH, open_db

DEFAULT_OUTPUT_PATH = DEFAULT_DB_PATH.parent / "scholarship_coach.ics"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export saved applications and milestones to an .ics calendar."
    )
    parser.add_argument("--student-id", default=DEFAULT_STUDENT_ID)
    parser.add_argument(
        "--db", type=Path, default=None, help="Family database (default: data/private/coach.db)"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--today", default=None, help="Reference date as YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--horizon-years",
        type=int,
        default=DEFAULT_HORIZON_YEARS,
        help="School years of milestones to include beyond the current one.",
    )
    return parser.parse_args(argv)


def _reference_date(value: str | None) -> date:
    if not value:
        return date.today()
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"--today must be YYYY-MM-DD, got {value!r}") from exc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    today = _reference_date(args.today)
    # A family can track applications before they ever save a profile; without
    # a grade, every milestone applies rather than none.
    profile = load_profile(args.student_id) or {}
    grade_level = str(profile.get("grade_level") or "")

    with open_db(args.db) as conn:
        events = family_events(
            conn,
            args.student_id,
            grade_level=grade_level or None,
            today=today,
            horizon_years=max(int(args.horizon_years), 0),
        )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(to_ics(events), encoding="utf-8", newline="")
    print(f"Wrote {len(events)} event(s) to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
