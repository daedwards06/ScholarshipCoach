"""Flag rows where a human relevance label disagrees with the proxy label.

Run this while hand-labeling to catch judgement rows worth a second look: the
proxy heuristic and a human should broadly agree, so a large gap (e.g. human 0
vs proxy 2) is either a proxy blind spot worth documenting or a labeling slip
worth fixing. It is a diagnostic, not a gate — the proxy is not ground truth.

The proxy label for each scholarship is recomputed by scoring the profile's
eligible set (Stage 2) and applying the same 0/1/2 heuristic the evaluator uses.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eval.golden_students import GoldenStudent, get_golden_student
from src.eval.human_labels import load_human_labels
from src.eval.relevance import RelevanceConfig, proxy_relevance_label
from src.io.snapshotting import get_latest_snapshot_path
from src.rank.stage1_eligibility import apply_eligibility_filter
from src.rank.stage2_scoring import score_stage2

ROOT_DIR = Path(__file__).resolve().parents[1]
SHARP_DELTA = 2


def compute_proxy_labels(
    snapshot_df: pd.DataFrame,
    student: GoldenStudent,
    *,
    similarity_mode: str,
    cfg: RelevanceConfig,
    processed_dir: Path | None = None,
) -> dict[str, int]:
    """Return ``{scholarship_id: proxy_label}`` for ``student``'s eligible set."""
    eligible_df, _ = apply_eligibility_filter(snapshot_df, student.profile)
    if eligible_df.empty:
        return {}
    scored_df = score_stage2(
        eligible_df,
        student.as_stage2_profile(),
        similarity_mode=similarity_mode,  # type: ignore[arg-type]
        processed_dir=processed_dir,
    )
    labels: dict[str, int] = {}
    for _, row in scored_df.iterrows():
        scholarship_id = row.get("scholarship_id")
        if scholarship_id is None or pd.isna(scholarship_id):
            continue
        labels[str(scholarship_id)] = proxy_relevance_label(
            row, student, similarity_mode=similarity_mode, cfg=cfg
        )
    return labels


def label_agreement(
    human_labels: dict[str, dict[str, int]],
    proxy_by_profile: dict[str, dict[str, int]],
    *,
    sharp_delta: int = SHARP_DELTA,
) -> list[dict[str, object]]:
    """Compare human vs proxy labels, one row per comparable (profile, scholarship).

    Scholarships a human labelled that are absent from the proxy map (not in the
    eligible/scored set) are skipped — there is nothing to compare against. Rows
    are returned worst-disagreement first.
    """
    rows: list[dict[str, object]] = []
    for profile_id, scholarship_labels in human_labels.items():
        proxy = proxy_by_profile.get(profile_id, {})
        for scholarship_id, human in scholarship_labels.items():
            if scholarship_id not in proxy:
                continue
            proxy_label = proxy[scholarship_id]
            delta = abs(int(human) - int(proxy_label))
            rows.append(
                {
                    "profile_id": profile_id,
                    "scholarship_id": scholarship_id,
                    "human": int(human),
                    "proxy": int(proxy_label),
                    "delta": delta,
                    "sharp": delta >= sharp_delta,
                }
            )
    rows.sort(key=lambda r: (-int(r["delta"]), str(r["profile_id"]), str(r["scholarship_id"])))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flag human-vs-proxy relevance-label disagreements for a labeling pass."
    )
    parser.add_argument(
        "--human-labels",
        type=Path,
        required=True,
        help="CSV of human labels (profile_id, scholarship_id, label).",
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
        "--similarity-mode",
        choices=("tfidf", "embeddings"),
        default="tfidf",
        help="Stage 2 text similarity mode used to recompute proxy labels. Defaults to tfidf.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("hybrid", "no_similarity"),
        default="hybrid",
        help="Proxy relevance labeling mode. Defaults to hybrid.",
    )
    parser.add_argument(
        "--tfidf-threshold",
        type=float,
        default=0.12,
        help="Proxy relevance threshold for tfidf mode. Defaults to 0.12.",
    )
    parser.add_argument(
        "--embed-threshold",
        type=float,
        default=0.30,
        help="Proxy relevance threshold for embeddings mode. Defaults to 0.30.",
    )
    parser.add_argument(
        "--fail-on-sharp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit non-zero if any sharp (|human - proxy| >= 2) disagreement is found.",
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
    human_path = args.human_labels if args.human_labels.is_absolute() else ROOT_DIR / args.human_labels
    human_labels = load_human_labels(human_path)
    if not human_labels:
        raise SystemExit(
            f"No usable human labels found in '{human_path}'. "
            "Fill the 'label' column (0/1/2) for at least one row before running this check."
        )

    snapshot_path = _resolve_snapshot_path(args.snapshot, args.processed_dir)
    snapshot_df = pd.read_parquet(snapshot_path)
    processed_dir = args.processed_dir if args.processed_dir.is_absolute() else ROOT_DIR / args.processed_dir
    cfg = RelevanceConfig(
        label_mode=args.label_mode,
        tfidf_threshold=args.tfidf_threshold,
        embed_threshold=args.embed_threshold,
    )
    title_by_id = {
        str(row.scholarship_id): str(row.title)
        for row in snapshot_df.itertuples(index=False)
    }

    proxy_by_profile: dict[str, dict[str, int]] = {}
    for profile_id in human_labels:
        try:
            student = get_golden_student(profile_id)
        except ValueError:
            print(f"! Skipping unknown profile '{profile_id}' (not a golden student).")
            continue
        proxy_by_profile[profile_id] = compute_proxy_labels(
            snapshot_df,
            student,
            similarity_mode=args.similarity_mode,
            cfg=cfg,
            processed_dir=processed_dir,
        )

    rows = label_agreement(human_labels, proxy_by_profile)
    total_human = sum(len(v) for v in human_labels.values())
    compared = len(rows)
    mismatches = [r for r in rows if int(r["delta"]) > 0]
    sharp = [r for r in rows if bool(r["sharp"])]

    print(f"Snapshot: {snapshot_path}")
    print(f"Similarity mode: {args.similarity_mode} | Label mode: {args.label_mode}")
    print(
        f"Human-labeled pairs: {total_human} | comparable: {compared} | "
        f"mismatches: {len(mismatches)} | sharp (delta>=2): {len(sharp)}"
    )
    if compared < total_human:
        print(f"  ({total_human - compared} labeled row(s) not in any eligible/scored set — skipped)")
    print("")

    if not mismatches:
        print("No human-vs-proxy disagreements. 🎉")
    else:
        print("delta  human  proxy  profile / scholarship")
        for row in mismatches:
            flag = "‼" if row["sharp"] else " "
            title = title_by_id.get(str(row["scholarship_id"]), "")
            print(
                f"{flag} {row['delta']:>3}   {row['human']:>4}  {row['proxy']:>5}  "
                f"{row['profile_id']} / {str(row['scholarship_id'])[:12]}  {title[:50]}"
            )

    if args.fail_on_sharp and sharp:
        raise SystemExit(f"{len(sharp)} sharp disagreement(s) found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
