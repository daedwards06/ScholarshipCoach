from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.eval.golden_students import GoldenStudent, get_golden_students
from src.eval.metrics import (
    amount_distribution_stats,
    compute_ndcg_at_k,
    coverage_at_k,
    eligibility_precision,
    ranking_stability,
)
from src.eval.relevance import (
    RelevanceConfig,
    calibrate_similarity_threshold,
    get_similarity_threshold,
    proxy_relevance_label,
    proxy_relevance_labels,
)
from src.io.snapshotting import get_latest_snapshot_path
from src.rank.stage1_eligibility import apply_eligibility_filter
from src.rank.stage2_scoring import score_stage2
from src.rank.stage3_rerank import rerank_stage3
from src.rank.weights import Stage2Weights, Stage3Weights

MAX_K = 10
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
BEST_WEIGHTS_PATH = ROOT_DIR / "data" / "processed" / "best_weights.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline evaluation against golden student profiles.")
    parser.add_argument(
        "--k",
        type=int,
        default=MAX_K,
        help=f"Top-K cutoff per profile. Defaults to {MAX_K}.",
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
        "--reports-dir",
        type=Path,
        default=ROOT_DIR / "reports",
        help="Output directory for markdown and JSON artifacts.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional JSON file with stage2_weights, stage3_weights, and amount_utility_mode overrides.",
    )
    parser.add_argument(
        "--similarity-mode",
        choices=("tfidf", "embeddings"),
        default="tfidf",
        help="Stage 2 text similarity mode. Defaults to tfidf.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Embedding model alias/name. Defaults to {DEFAULT_MODEL_NAME}.",
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
        "--calibrate-thresholds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Report a deterministic suggested similarity threshold from eligible-item distributions.",
    )
    parser.add_argument(
        "--use-best-weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Load data/processed/best_weights.json when available. Defaults to true if the file exists.",
    )
    return parser.parse_args()


def _resolve_snapshot_path(snapshot: Path | None, processed_dir: Path) -> Path:
    if snapshot is not None:
        return snapshot if snapshot.is_absolute() else ROOT_DIR / snapshot
    latest = get_latest_snapshot_path(processed_dir if processed_dir.is_absolute() else ROOT_DIR / processed_dir)
    if latest is None:
        raise FileNotFoundError(f"No snapshot parquet found in '{processed_dir}'.")
    return latest


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def _load_weight_overrides(
    weights_path: Path | None,
) -> tuple[Stage2Weights | None, Stage3Weights | None, str, Path | None]:
    if weights_path is None:
        return None, None, "log", None

    resolved_path = _resolve_path(weights_path)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    amount_utility_mode = str(payload.get("amount_utility_mode", "log")).strip().lower()
    if amount_utility_mode not in {"linear", "log"}:
        raise ValueError(
            f"Unsupported amount_utility_mode '{amount_utility_mode}' in '{resolved_path}'. "
            "Expected 'linear' or 'log'."
        )

    return (
        Stage2Weights.from_mapping(payload.get("stage2_weights")),
        Stage3Weights.from_mapping(payload.get("stage3_weights")),
        amount_utility_mode,
        resolved_path,
    )


def _resolve_weight_path(
    weights_path: Path | None,
    use_best_weights: bool | None,
) -> tuple[Path | None, bool]:
    if weights_path is not None:
        resolved_path = _resolve_path(weights_path)
        return resolved_path, resolved_path.resolve() == BEST_WEIGHTS_PATH.resolve()

    best_exists = BEST_WEIGHTS_PATH.exists()
    should_use_best = best_exists if use_best_weights is None else bool(use_best_weights)
    if not should_use_best:
        return None, False
    if not best_exists:
        raise FileNotFoundError(f"Requested best weights, but no file exists at '{BEST_WEIGHTS_PATH}'.")
    return BEST_WEIGHTS_PATH, True


def _safe_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _proxy_relevance_label(
    row: pd.Series,
    student: GoldenStudent,
    *,
    similarity_mode: str = "tfidf",
    relevance_config: RelevanceConfig = RelevanceConfig(),
) -> int:
    return proxy_relevance_label(
        row,
        student,
        similarity_mode=similarity_mode,
        cfg=relevance_config,
    )


def _profile_topk_records(topk_df: pd.DataFrame, k: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in topk_df.head(k).iterrows():
        records.append(
            {
                "scholarship_id": row.get("scholarship_id"),
                "title": row.get("title"),
                "final_score": _safe_number(row.get("final_score")),
                "stage2_score": _safe_number(row.get("stage2_score")),
                "text_sim": _safe_number(row.get("text_sim")),
                "tfidf_sim": _safe_number(row.get("tfidf_sim")),
                "embed_sim": _safe_number(row.get("embed_sim")),
                "amount_utility": _safe_number(row.get("amount_utility")),
                "keyword_overlap": _safe_number(row.get("keyword_overlap")),
                "effort_penalty": _safe_number(row.get("effort_penalty")),
                "urgency_boost": _safe_number(row.get("urgency_boost")),
                "ev_proxy_norm": _safe_number(row.get("ev_proxy_norm")),
                "amount_max": _safe_number(row.get("amount_max")),
            }
        )
    return records


def _run_per_profile(
    snapshot_df: pd.DataFrame,
    students: list[GoldenStudent],
    *,
    max_k: int,
    stage2_weights: Stage2Weights | None = None,
    stage3_weights: Stage3Weights | None = None,
    amount_utility_mode: str = "log",
    similarity_mode: str = "tfidf",
    model_name: str = DEFAULT_MODEL_NAME,
    relevance_config: RelevanceConfig = RelevanceConfig(),
    processed_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[str]], dict[str, list[int]]]:
    per_profile_results: list[dict[str, Any]] = []
    per_profile_topk: dict[str, list[dict[str, Any]]] = {}
    ordered_ids: dict[str, list[str]] = {}
    relevance_labels: dict[str, list[int]] = {}

    for student in students:
        eligible_df, ineligible_df = apply_eligibility_filter(snapshot_df, student.profile)
        k = min(max_k, int(len(eligible_df)))

        if k > 0:
            scored_df = score_stage2(
                eligible_df,
                student.as_stage2_profile(),
                weights=stage2_weights,
                amount_utility_mode=amount_utility_mode,
                similarity_mode=similarity_mode,
                model_name=model_name,
                processed_dir=processed_dir,
            )
            reranked_df = rerank_stage3(
                scored_df,
                today=student.profile.today,
                weights=stage3_weights,
            )
            topk_df = reranked_df.head(k).copy()
        else:
            scored_df = pd.DataFrame()
            reranked_df = pd.DataFrame()
            topk_df = pd.DataFrame()

        top_records = _profile_topk_records(topk_df, k)
        labels = proxy_relevance_labels(
            topk_df,
            student,
            similarity_mode=similarity_mode,
            cfg=relevance_config,
        )
        profile_id = student.student_id

        per_profile_results.append(
            {
                "student_id": profile_id,
                "description": student.description,
                "k": k,
                "eligible_df": eligible_df,
                "ineligible_df": ineligible_df,
                "scored_df": scored_df,
                "reranked_df": reranked_df,
            }
        )
        per_profile_topk[profile_id] = top_records
        ordered_ids[profile_id] = [str(rec["scholarship_id"]) for rec in top_records]
        relevance_labels[profile_id] = labels

    return per_profile_results, per_profile_topk, ordered_ids, relevance_labels


def _calibration_payload(
    per_profile_results: list[dict[str, Any]],
    *,
    enabled: bool,
    similarity_mode: str,
) -> dict[str, Any] | None:
    if not enabled:
        return None

    eligible_frames = [
        result["scored_df"]
        for result in per_profile_results
        if isinstance(result.get("scored_df"), pd.DataFrame)
    ]
    suggested_threshold = calibrate_similarity_threshold(eligible_frames)
    return {
        "enabled": True,
        "similarity_mode": similarity_mode,
        "target_similarity_only_share": 0.25,
        "suggested_threshold": suggested_threshold,
    }


def _metrics_payload(
    per_profile_results: list[dict[str, Any]],
    per_profile_topk: dict[str, list[dict[str, Any]]],
    relevance_labels: dict[str, list[int]],
    run_one_ids: dict[str, list[str]],
    run_two_ids: dict[str, list[str]],
    similarity_mode: str,
    relevance_config: RelevanceConfig,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    max_k_observed = max((len(records) for records in per_profile_topk.values()), default=0)
    coverage = coverage_at_k(per_profile_topk, k=max_k_observed)
    amounts = amount_distribution_stats(per_profile_topk, k=max_k_observed)
    stability = ranking_stability(run_one_ids, run_two_ids)
    ndcg = compute_ndcg_at_k(relevance_labels, k=max_k_observed)

    return {
        "eligibility": eligibility_precision(per_profile_results),
        "coverage_at_k": coverage,
        "amount_distribution_topk": amounts,
        "ranking_stability": stability,
        "ndcg_at_k": {"k": max_k_observed, "value": ndcg},
        "proxy_relevance": {
            "labels": {
                "2": "major/state/education match and keyword overlap > 0",
                "1": (
                    "keyword overlap > 0 OR text similarity >= mode threshold"
                    if relevance_config.label_mode == "hybrid"
                    else "keyword overlap > 0"
                ),
                "0": "otherwise",
            },
            "label_mode": relevance_config.label_mode,
            "strict_requires_all_of": list(relevance_config.strict_requires_all_of),
            "text_similarity_thresholds": {
                "tfidf": relevance_config.tfidf_threshold,
                "embeddings": relevance_config.embed_threshold,
            },
            "active_text_similarity_threshold": get_similarity_threshold(
                similarity_mode,
                relevance_config,
            ),
            "similarity_mode": similarity_mode,
            "calibration": calibration,
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _format_score(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{float(value):.4f}"


def _markdown_report(
    *,
    snapshot_path: Path,
    snapshot_count: int,
    generated_at: str,
    weights_path: Path | None,
    metrics: dict[str, Any],
    students: list[GoldenStudent],
    per_profile_topk: dict[str, list[dict[str, Any]]],
    used_best_weights: bool = False,
    similarity_mode: str = "tfidf",
    model_name: str | None = None,
) -> str:
    lines: list[str] = []
    proxy_relevance = metrics.get("proxy_relevance", {})
    thresholds = proxy_relevance.get("text_similarity_thresholds", {})
    calibration = proxy_relevance.get("calibration")
    lines.append("# Golden Student Offline Evaluation")
    lines.append("")
    lines.append(f"- Generated at (UTC): {generated_at}")
    lines.append(f"- Snapshot: `{snapshot_path}`")
    lines.append(f"- Snapshot records: {snapshot_count}")
    lines.append(f"- Golden profiles: {len(students)}")
    lines.append(f"- Similarity mode: `{similarity_mode}`")
    lines.append(f"- Label mode: `{proxy_relevance.get('label_mode', 'hybrid')}`")
    lines.append(f"- TF-IDF relevance threshold: {thresholds.get('tfidf', 0.12):.4f}")
    lines.append(f"- Embeddings relevance threshold: {thresholds.get('embeddings', 0.30):.4f}")
    lines.append(
        f"- Active relevance threshold: {proxy_relevance.get('active_text_similarity_threshold', 0.0):.4f}"
    )
    if calibration and calibration.get("enabled"):
        suggested = calibration.get("suggested_threshold")
        if suggested is None:
            lines.append(
                f"- Suggested calibrated {calibration.get('similarity_mode', similarity_mode)} threshold: n/a"
            )
        else:
            lines.append(
                f"- Suggested calibrated {calibration.get('similarity_mode', similarity_mode)} threshold: "
                f"{float(suggested):.4f}"
            )
    if similarity_mode == "embeddings" and model_name:
        lines.append(f"- Embedding model: `{model_name}`")
    if weights_path is None:
        lines.append("- Weights: baseline defaults")
    else:
        lines.append(f"- Weights file: `{weights_path}`")
    lines.append(f"- Used best_weights.json: {used_best_weights}")
    lines.append("")
    lines.append("## Metrics Summary")
    lines.append("")

    eligibility = metrics["eligibility"]
    lines.append(f"- Eligibility precision: {eligibility['eligibility_precision']:.4f}")
    lines.append(f"- Eligible count: {eligibility['eligible_count']}")
    lines.append(f"- Total evaluated rows: {eligibility['total_count']}")
    lines.append(
        f"- Coverage@K (K={metrics['coverage_at_k']['k']}): {metrics['coverage_at_k']['coverage_at_k']:.4f}"
    )
    lines.append(
        f"- Unique recommended scholarships: {metrics['coverage_at_k']['unique_recommended_count']}"
    )
    lines.append(
        f"- Amount stats (mean/median/max): "
        f"{metrics['amount_distribution_topk']['mean']:.2f} / "
        f"{metrics['amount_distribution_topk']['median']:.2f} / "
        f"{metrics['amount_distribution_topk']['max']:.2f}"
    )
    lines.append(f"- Ranking stability: {metrics['ranking_stability']['is_stable']}")
    lines.append(
        f"- NDCG@K (K={metrics['ndcg_at_k']['k']}): {metrics['ndcg_at_k']['value']}"
    )
    lines.append("")
    lines.append("### Ineligible Reason Breakdown")
    lines.append("")
    reason_breakdown = eligibility["ineligible_reason_breakdown"]
    if reason_breakdown:
        for reason, count in reason_breakdown.items():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Per Profile Top-K")
    lines.append("")

    for student in students:
        profile_id = student.student_id
        top_recs = per_profile_topk.get(profile_id, [])
        lines.append(f"### {profile_id}")
        lines.append("")
        lines.append(f"- Description: {student.description}")
        lines.append(f"- Top-K count: {len(top_recs)}")
        lines.append("")
        if not top_recs:
            lines.append("No eligible scholarships for this profile in the selected snapshot.")
            lines.append("")
            continue
        lines.append(
            "| scholarship_id | final_score | stage2_score | text_sim | amount_utility | keyword_overlap | urgency_boost | ev_proxy_norm | title |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for rec in top_recs:
            lines.append(
                f"| {rec['scholarship_id']} | "
                f"{_format_score(rec['final_score'])} | "
                f"{_format_score(rec['stage2_score'])} | "
                f"{_format_score(rec['text_sim'])} | "
                f"{_format_score(rec['amount_utility'])} | "
                f"{_format_score(rec['keyword_overlap'])} | "
                f"{_format_score(rec['urgency_boost'])} | "
                f"{_format_score(rec['ev_proxy_norm'])} | "
                f"{str(rec.get('title') or '').replace('|', '/')} |"
            )
        lines.append("")

    return "\n".join(lines)


def _to_serializable_profile_results(
    per_profile_results: list[dict[str, Any]],
    per_profile_topk: dict[str, list[dict[str, Any]]],
    relevance_labels: dict[str, list[int]],
) -> list[dict[str, Any]]:
    serializable: list[dict[str, Any]] = []
    for result in per_profile_results:
        student_id = result["student_id"]
        serializable.append(
            {
                "student_id": student_id,
                "description": result["description"],
                "k": result["k"],
                "eligible_count": int(len(result["eligible_df"])),
                "ineligible_count": int(len(result["ineligible_df"])),
                "top_k": per_profile_topk.get(student_id, []),
                "proxy_labels_top_k": relevance_labels.get(student_id, []),
            }
        )
    return serializable


def main() -> int:
    args = parse_args()
    if args.k <= 0:
        raise SystemExit("--k must be greater than 0.")

    relevance_config = RelevanceConfig(
        label_mode=args.label_mode,
        tfidf_threshold=args.tfidf_threshold,
        embed_threshold=args.embed_threshold,
    )
    snapshot_path = _resolve_snapshot_path(args.snapshot, args.processed_dir)
    weights_path, used_best_weights = _resolve_weight_path(args.weights, args.use_best_weights)
    stage2_weights, stage3_weights, amount_utility_mode, weights_path = _load_weight_overrides(weights_path)
    processed_dir = args.processed_dir if args.processed_dir.is_absolute() else ROOT_DIR / args.processed_dir
    snapshot_df = pd.read_parquet(snapshot_path)
    students = get_golden_students()

    run_one_results, run_one_topk, run_one_ids, relevance_labels = _run_per_profile(
        snapshot_df,
        students,
        max_k=args.k,
        stage2_weights=stage2_weights,
        stage3_weights=stage3_weights,
        amount_utility_mode=amount_utility_mode,
        similarity_mode=args.similarity_mode,
        model_name=args.model_name,
        relevance_config=relevance_config,
        processed_dir=processed_dir,
    )
    _, _, run_two_ids, _ = _run_per_profile(
        snapshot_df,
        students,
        max_k=args.k,
        stage2_weights=stage2_weights,
        stage3_weights=stage3_weights,
        amount_utility_mode=amount_utility_mode,
        similarity_mode=args.similarity_mode,
        model_name=args.model_name,
        relevance_config=relevance_config,
        processed_dir=processed_dir,
    )
    calibration = _calibration_payload(
        run_one_results,
        enabled=args.calibrate_thresholds,
        similarity_mode=args.similarity_mode,
    )

    metrics = _metrics_payload(
        run_one_results,
        run_one_topk,
        relevance_labels,
        run_one_ids,
        run_two_ids,
        args.similarity_mode,
        relevance_config,
        calibration,
    )

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    generated_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    reports_dir = args.reports_dir if args.reports_dir.is_absolute() else ROOT_DIR / args.reports_dir
    markdown_path = reports_dir / f"golden_eval_{timestamp}.md"
    json_path = reports_dir / "artifacts" / f"golden_eval_{timestamp}.json"

    markdown_text = _markdown_report(
        snapshot_path=snapshot_path,
        snapshot_count=len(snapshot_df),
        generated_at=generated_at,
        weights_path=weights_path,
        used_best_weights=used_best_weights,
        similarity_mode=args.similarity_mode,
        model_name=args.model_name if args.similarity_mode == "embeddings" else None,
        metrics=metrics,
        students=students,
        per_profile_topk=run_one_topk,
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_text, encoding="utf-8")

    payload = {
        "generated_at": generated_at,
        "snapshot_path": str(snapshot_path),
        "snapshot_count": int(len(snapshot_df)),
        "golden_profiles_count": len(students),
        "similarity_mode": args.similarity_mode,
        "model_name": args.model_name if args.similarity_mode == "embeddings" else None,
        "label_mode": relevance_config.label_mode,
        "tfidf_threshold": relevance_config.tfidf_threshold,
        "embed_threshold": relevance_config.embed_threshold,
        "active_text_similarity_threshold": get_similarity_threshold(
            args.similarity_mode,
            relevance_config,
        ),
        "calibration": calibration,
        "weights_path": str(weights_path) if weights_path is not None else None,
        "used_baseline_weights": weights_path is None,
        "used_best_weights": used_best_weights,
        "metrics": metrics,
        "per_profile": _to_serializable_profile_results(run_one_results, run_one_topk, relevance_labels),
    }
    _write_json(json_path, payload)

    print(f"Wrote markdown report: {markdown_path}")
    print(f"Wrote JSON artifact: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
