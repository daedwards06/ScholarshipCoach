"""Emit a human-labeling worksheet for a golden profile.

Samples across a profile's *eligible* scholarship set (not just the top-K, to
avoid ranking bias in the label set) and writes a CSV with an empty ``label``
column for a human to fill with 0/1/2 relevance judgements. The completed CSV is
consumed by ``evaluate_golden_students.py --human-labels`` to report a
human-judged NDCG@k alongside the proxy metric.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eval.golden_students import GoldenStudent, get_golden_student
from src.io.snapshotting import get_latest_snapshot_path
from src.rank.stage1_eligibility import apply_eligibility_filter

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "eval"
DESCRIPTION_SNIPPET_CHARS = 200

WORKSHEET_COLUMNS = [
    "profile_id",
    "scholarship_id",
    "title",
    "sponsor",
    "amount",
    "deadline",
    "source_url",
    "description_snippet",
    "label",
]


def get_student_by_id(profile_id: str) -> GoldenStudent:
    """Return the golden student with ``profile_id`` or exit with a CLI-friendly message."""
    try:
        return get_golden_student(profile_id)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _format_amount(row: pd.Series) -> str:
    amount_min = row.get("amount_min")
    amount_max = row.get("amount_max")
    low = None if amount_min is None or pd.isna(amount_min) else float(amount_min)
    high = None if amount_max is None or pd.isna(amount_max) else float(amount_max)
    if low is None and high is None:
        return "Unknown"
    if low is not None and high is not None and low != high:
        return f"${low:,.0f}-${high:,.0f}"
    value = high if high is not None else low
    return f"${value:,.0f}"


def _format_deadline(row: pd.Series) -> str:
    deadline = row.get("deadline")
    if deadline is None or pd.isna(deadline):
        return ""
    return str(pd.Timestamp(deadline).date())


def _str_or_empty(value: object) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


def _description_snippet(text: object) -> str:
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return ""
    snippet = " ".join(str(text).split())
    return snippet[:DESCRIPTION_SNIPPET_CHARS]


def build_worksheet(
    snapshot_df: pd.DataFrame,
    student: GoldenStudent,
    *,
    n: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a labeling worksheet sampled across ``student``'s eligible set.

    Args:
        snapshot_df: Full scholarship snapshot DataFrame.
        student: Golden student whose eligibility set is sampled.
        n: Target number of rows; capped at the eligible-set size.
        seed: Deterministic sampling seed.

    Returns:
        DataFrame with :data:`WORKSHEET_COLUMNS`; the ``label`` column is empty.
    """
    eligible_df, _ = apply_eligibility_filter(snapshot_df, student.profile)
    if eligible_df.empty:
        return pd.DataFrame(columns=WORKSHEET_COLUMNS)

    sample_n = min(int(n), len(eligible_df))
    sampled = eligible_df.sample(n=sample_n, random_state=seed).sort_values("scholarship_id")

    rows: list[dict[str, object]] = []
    for _, row in sampled.iterrows():
        rows.append(
            {
                "profile_id": student.student_id,
                "scholarship_id": row.get("scholarship_id"),
                "title": row.get("title"),
                "sponsor": row.get("sponsor"),
                "amount": _format_amount(row),
                "deadline": _format_deadline(row),
                "source_url": _str_or_empty(row.get("source_url")),
                "description_snippet": _description_snippet(row.get("description")),
                "label": "",
            }
        )
    return pd.DataFrame(rows, columns=WORKSHEET_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit a human-labeling worksheet CSV for a golden profile."
    )
    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        help="Golden profile id to sample (e.g. nc_cs_rising_sophomore).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=60,
        help="Number of rows to sample from the eligible set. Defaults to 60.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic sampling seed. Defaults to 0.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Snapshot parquet path. If omitted, latest in --processed-dir is used.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT_DIR / "data" / "processed",
        help="Processed directory used to resolve latest snapshot when --snapshot is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the worksheet CSV. Defaults to data/eval.",
    )
    return parser.parse_args()


def _resolve_snapshot_path(snapshot: Path | None, processed_dir: Path) -> Path:
    if snapshot is not None:
        return snapshot if snapshot.is_absolute() else ROOT_DIR / snapshot
    resolved_dir = processed_dir if processed_dir.is_absolute() else ROOT_DIR / processed_dir
    latest = get_latest_snapshot_path(resolved_dir)
    if latest is None:
        raise FileNotFoundError(f"No snapshot parquet found in '{processed_dir}'.")
    return latest


def main() -> int:
    args = parse_args()
    if args.n <= 0:
        raise SystemExit("--n must be greater than 0.")

    student = get_student_by_id(args.profile)
    snapshot_path = _resolve_snapshot_path(args.snapshot, args.processed_dir)
    snapshot_df = pd.read_parquet(snapshot_path)

    worksheet = build_worksheet(snapshot_df, student, n=args.n, seed=args.seed)
    if worksheet.empty:
        raise SystemExit(f"No eligible scholarships for profile '{args.profile}' in '{snapshot_path}'.")

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"labeling_worksheet_{args.profile}.csv"
    worksheet.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Wrote labeling worksheet ({len(worksheet)} rows): {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
