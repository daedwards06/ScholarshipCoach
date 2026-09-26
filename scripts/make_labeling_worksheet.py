"""Emit a human-labeling worksheet for a golden profile.

Samples across a profile's *eligible* scholarship set (not just the top-K, to
avoid ranking bias in the label set) and writes a CSV with an empty ``label``
column for a human to fill with 0/1/2 relevance judgements. The completed CSV is
consumed by ``evaluate_golden_students.py --human-labels`` to report a
human-judged NDCG@k alongside the proxy metric.

``--profile`` targets a golden evaluation persona.  ``--student`` targets the
real student stored under ``data/private/students/`` (falling back to the
committed demo profile), so the family's own labels can enter the evaluation
set.  ``--top-ranked`` samples the ranked prefix the student actually sees
instead of the eligible set; it is the faster ask of a real person, at the cost
of a label set that cannot reveal awards the ranker missed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from src.eval.golden_students import GoldenStudent, get_golden_student
from src.io.snapshotting import get_latest_snapshot_path
from src.profile.store import (
    DEFAULT_STUDENT_ID,
    load_demo_profile,
    load_profile,
    load_profile_or_demo,
    to_stage1_profile,
    to_stage2_profile,
)
from src.rank.stage1_eligibility import StudentProfile, apply_eligibility_filter
from src.rank.stage2_scoring import score_stage2
from src.rank.stage3_rerank import rerank_stage3

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "eval"
# A real student's profile and labels are personal data, so their worksheet
# lands beside the private profile rather than in the committed eval set.
PRIVATE_OUTPUT_DIR = ROOT_DIR / "data" / "private" / "eval"
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


class WorksheetSubject(Protocol):
    """The slice of a student a worksheet needs: an id and both stage profiles."""

    @property
    def student_id(self) -> str: ...

    @property
    def profile(self) -> StudentProfile: ...

    def as_stage2_profile(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class StoredStudentSubject:
    """Adapts a stored private/demo profile to the :class:`WorksheetSubject` shape."""

    student_id: str
    profile: StudentProfile
    stage2_profile: dict[str, Any]

    def as_stage2_profile(self) -> dict[str, Any]:
        return dict(self.stage2_profile)


def get_student_by_id(profile_id: str) -> GoldenStudent:
    """Return the golden student with ``profile_id`` or exit with a CLI-friendly message."""
    try:
        return get_golden_student(profile_id)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def get_stored_student(student_id: str) -> tuple[StoredStudentSubject, bool]:
    """Return the stored student subject and whether it fell back to the demo profile."""
    stored, is_demo = load_profile_or_demo(student_id)
    subject = StoredStudentSubject(
        student_id=str(stored.get("student_id") or student_id),
        profile=to_stage1_profile(stored),
        stage2_profile=to_stage2_profile(stored),
    )
    return subject, is_demo


def find_stored_student(student_id: str) -> StoredStudentSubject | None:
    """Return the stored student with exactly this id, or ``None``.

    Unlike :func:`get_stored_student` this never substitutes the demo profile
    for a different id: a labels file names whose judgements they are, and
    scoring them against someone else's ranking would be a wrong number.
    """
    try:
        stored = load_profile(student_id)
    except ValueError:
        return None
    if stored is None:
        demo = load_demo_profile()
        if str(demo.get("student_id") or "") != student_id:
            return None
        stored = demo
    return StoredStudentSubject(
        student_id=student_id,
        profile=to_stage1_profile(stored),
        stage2_profile=to_stage2_profile(stored),
    )


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
    student: WorksheetSubject,
    *,
    n: int,
    seed: int = 0,
    top_ranked: bool = False,
    similarity_mode: str = "tfidf",
) -> pd.DataFrame:
    """Return a labeling worksheet for ``student``.

    Args:
        snapshot_df: Full scholarship snapshot DataFrame.
        student: Subject whose eligibility set is sampled.
        n: Target number of rows; capped at the eligible-set size.
        seed: Deterministic sampling seed.
        top_ranked: Take the ranked top-``n`` instead of a random eligible sample.
        similarity_mode: Stage 2 similarity mode used when ``top_ranked`` is set.

    Returns:
        DataFrame with :data:`WORKSHEET_COLUMNS`; the ``label`` column is empty.
    """
    eligible_df, _ = apply_eligibility_filter(snapshot_df, student.profile)
    if eligible_df.empty:
        return pd.DataFrame(columns=WORKSHEET_COLUMNS)

    sample_n = min(int(n), len(eligible_df))
    if top_ranked:
        scored_df = score_stage2(
            eligible_df,
            student.as_stage2_profile(),
            similarity_mode=similarity_mode,
        )
        reranked_df = rerank_stage3(
            scored_df,
            today=student.profile.today,
            profile=student.profile,
        )
        sampled = reranked_df.head(sample_n)
    else:
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
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--profile",
        type=str,
        help="Golden profile id to sample (e.g. nc_cs_rising_sophomore).",
    )
    target.add_argument(
        "--student",
        type=str,
        nargs="?",
        const=DEFAULT_STUDENT_ID,
        help="Stored student id under data/private/students/ (default: "
        f"{DEFAULT_STUDENT_ID}). Falls back to the committed demo profile.",
    )
    parser.add_argument(
        "--top-ranked",
        action="store_true",
        help="Sample the ranked top-N the student actually sees instead of a random "
        "draw across the eligible set. Faster to label; cannot reveal missed awards.",
    )
    parser.add_argument(
        "--similarity-mode",
        choices=("tfidf", "embeddings"),
        default="tfidf",
        help="Stage 2 similarity mode used by --top-ranked. Defaults to tfidf.",
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
        default=None,
        help="Output directory for the worksheet CSV. Defaults to data/private/eval for "
        "--student and data/eval for --profile.",
    )
    return parser.parse_args()


def resolve_output_dir(output_dir: Path | None, *, stored_student: bool) -> Path:
    """Return where the worksheet is written; a stored student's stays private."""
    if output_dir is None:
        return PRIVATE_OUTPUT_DIR if stored_student else DEFAULT_OUTPUT_DIR
    return output_dir if output_dir.is_absolute() else ROOT_DIR / output_dir


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

    student: WorksheetSubject
    if args.profile is not None:
        student = get_student_by_id(args.profile)
    else:
        student, is_demo = get_stored_student(args.student)
        if is_demo:
            print(
                f"No private profile for '{args.student}'; using the committed demo profile. "
                "Labels from this worksheet describe the demo student, not a real one."
            )

    snapshot_path = _resolve_snapshot_path(args.snapshot, args.processed_dir)
    snapshot_df = pd.read_parquet(snapshot_path)

    worksheet = build_worksheet(
        snapshot_df,
        student,
        n=args.n,
        seed=args.seed,
        top_ranked=args.top_ranked,
        similarity_mode=args.similarity_mode,
    )
    if worksheet.empty:
        raise SystemExit(
            f"No eligible scholarships for profile '{student.student_id}' in '{snapshot_path}'."
        )

    output_dir = resolve_output_dir(args.output_dir, stored_student=args.student is not None)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"labeling_worksheet_{student.student_id}.csv"
    worksheet.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Wrote labeling worksheet ({len(worksheet)} rows): {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
