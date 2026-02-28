from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_golden_students import _proxy_relevance_label
from src.eval.golden_students import GoldenStudent, get_golden_students
from src.eval.metrics import amount_distribution_stats, compute_ndcg_at_k, coverage_at_k, eligibility_precision
from src.io.snapshotting import get_latest_snapshot_path, write_json_atomic
from src.rank.stage1_eligibility import apply_eligibility_filter
from src.rank.stage2_scoring import score_stage2
from src.rank.stage3_rerank import rerank_stage3
from src.rank.weights import Stage2Weights, Stage3Weights

DEFAULT_K = 10
DEFAULT_MAX_CONFIGS = 200
DEFAULT_OUTDIR = ROOT_DIR / "reports" / "weight_tuning"
BEST_WEIGHTS_PATH = ROOT_DIR / "data" / "processed" / "best_weights.json"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass(frozen=True, slots=True)
class WeightConfig:
    config_id: str
    stage2_weights: Stage2Weights
    stage3_weights: Stage3Weights
    amount_utility_mode: str = "log"

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "stage2_weights": self.stage2_weights.to_dict(),
            "stage3_weights": self.stage3_weights.to_dict(),
            "amount_utility_mode": self.amount_utility_mode,
        }


@dataclass(frozen=True, slots=True)
class ProfileCache:
    student: GoldenStudent
    eligible_df: pd.DataFrame
    ineligible_df: pd.DataFrame
    component_df: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic Stage 2/Stage 3 weight tuning.")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Snapshot parquet path. If omitted, the latest saved snapshot is used.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Top-K cutoff per golden profile. Defaults to {DEFAULT_K}.",
    )
    parser.add_argument(
        "--max_configs",
        type=int,
        default=DEFAULT_MAX_CONFIGS,
        help=f"Deterministic safety cap for evaluated configs. Defaults to {DEFAULT_MAX_CONFIGS}.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Output directory for tuning reports and artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Accepted for CLI stability. No randomness is used by this script.",
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
    return parser.parse_args()


def _resolve_snapshot_path(snapshot: Path | None) -> Path:
    if snapshot is not None:
        return snapshot if snapshot.is_absolute() else ROOT_DIR / snapshot
    latest = get_latest_snapshot_path(ROOT_DIR / "data" / "processed")
    if latest is None:
        raise FileNotFoundError("No snapshot parquet found in data/processed.")
    return latest


def _resolve_outdir(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def _format_config_id(
    stage2_weights: Stage2Weights,
    stage3_weights: Stage3Weights,
    amount_utility_mode: str,
) -> str:
    return (
        f"s2_t{stage2_weights.tfidf:.2f}_a{stage2_weights.amount:.2f}"
        f"_k{stage2_weights.keyword:.2f}_e{stage2_weights.effort:.2f}"
        f"__s3_s{stage3_weights.stage2:.2f}_u{stage3_weights.urgency:.2f}_v{stage3_weights.ev:.2f}"
        f"__{amount_utility_mode}"
    )


def _build_config(
    stage2_weights: Stage2Weights,
    stage3_weights: Stage3Weights,
    *,
    amount_utility_mode: str = "log",
) -> WeightConfig:
    return WeightConfig(
        config_id=_format_config_id(stage2_weights, stage3_weights, amount_utility_mode),
        stage2_weights=stage2_weights,
        stage3_weights=stage3_weights,
        amount_utility_mode=amount_utility_mode,
    )


def generate_candidate_configs(max_configs: int = DEFAULT_MAX_CONFIGS) -> list[WeightConfig]:
    if max_configs < 1:
        raise ValueError("max_configs must be at least 1.")

    stage2_tfidf_values = [0.70, 0.80, 0.85, 0.90]
    stage2_amount_values = [0.05, 0.10, 0.20]
    stage2_keyword_values = [0.05, 0.10, 0.15]
    stage2_effort_values = [0.05, 0.10, 0.15]

    stage3_stage2_values = [0.80, 0.85, 0.90, 0.95]
    stage3_urgency_values = [0.05, 0.10, 0.15]
    stage3_ev_values = [0.00, 0.05, 0.10]

    configs: list[WeightConfig] = []
    seen_ids: set[str] = set()

    baseline_config = _build_config(Stage2Weights.baseline(), Stage3Weights.baseline())
    configs.append(baseline_config)
    seen_ids.add(baseline_config.config_id)

    for tfidf in stage2_tfidf_values:
        for amount in stage2_amount_values:
            for keyword in stage2_keyword_values:
                for effort in stage2_effort_values:
                    stage2_weights = Stage2Weights(
                        tfidf=tfidf,
                        amount=amount,
                        keyword=keyword,
                        effort=effort,
                    )
                    for stage2_value in stage3_stage2_values:
                        for urgency in stage3_urgency_values:
                            for ev in stage3_ev_values:
                                total = stage2_value + urgency + ev
                                if not math.isclose(total, 1.0, abs_tol=1e-6):
                                    continue
                                stage3_weights = Stage3Weights(
                                    stage2=stage2_value,
                                    urgency=urgency,
                                    ev=ev,
                                )
                                config = _build_config(stage2_weights, stage3_weights)
                                if config.config_id in seen_ids:
                                    continue
                                configs.append(config)
                                seen_ids.add(config.config_id)
                                if len(configs) >= max_configs:
                                    return configs

    return configs[:max_configs]


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _normalized_label(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _topk_records(topk_df: pd.DataFrame, k: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in topk_df.head(k).iterrows():
        records.append(
            {
                "scholarship_id": str(row.get("scholarship_id") or ""),
                "amount_max": _safe_float(row.get("amount_max")),
                "sponsor": _normalized_label(row.get("sponsor")),
                "source": _normalized_label(row.get("source")),
            }
        )
    return records


def _field_diversity_at_k(
    per_profile_topk: dict[str, list[dict[str, Any]]],
    *,
    field: str,
    k: int,
) -> dict[str, Any]:
    unique_values: set[str] = set()
    total_labeled = 0

    for records in per_profile_topk.values():
        for record in records[:k]:
            value = _normalized_label(record.get(field))
            if value is None:
                continue
            unique_values.add(value)
            total_labeled += 1

    return {
        "k": k,
        "field": field,
        "unique_count": len(unique_values),
        "labeled_count": total_labeled,
        "diversity_at_k": (len(unique_values) / total_labeled) if total_labeled else 0.0,
    }


def _prepare_profile_caches(
    snapshot_df: pd.DataFrame,
    students: list[GoldenStudent],
    *,
    similarity_mode: str,
    model_name: str,
) -> list[ProfileCache]:
    caches: list[ProfileCache] = []

    for student in students:
        eligible_df, ineligible_df = apply_eligibility_filter(snapshot_df, student.profile)
        if eligible_df.empty:
            component_df = pd.DataFrame()
        else:
            scored_df = score_stage2(
                eligible_df,
                student.as_stage2_profile(),
                weights=Stage2Weights.baseline(),
                amount_utility_mode="log",
                similarity_mode=similarity_mode,
                model_name=model_name,
            )
            component_df = rerank_stage3(
                scored_df,
                today=student.profile.today,
                weights=Stage3Weights.baseline(),
            )
            component_df["_deadline_sort"] = pd.to_datetime(
                component_df.get("deadline"),
                errors="coerce",
            )
        caches.append(
            ProfileCache(
                student=student,
                eligible_df=eligible_df,
                ineligible_df=ineligible_df,
                component_df=component_df,
            )
        )

    return caches


def _rerank_profile_cache(cache: ProfileCache, config: WeightConfig) -> pd.DataFrame:
    if cache.component_df.empty:
        return cache.component_df.copy()

    reranked_df = cache.component_df.copy()
    similarity_series = pd.to_numeric(
        reranked_df.get("text_sim", reranked_df.get("tfidf_sim")),
        errors="coerce",
    ).fillna(0.0)
    stage2_score = (
        (config.stage2_weights.tfidf * similarity_series)
        + (
            config.stage2_weights.amount
            * pd.to_numeric(reranked_df["amount_utility"], errors="coerce").fillna(0.0)
        )
        + (
            config.stage2_weights.keyword
            * pd.to_numeric(reranked_df["keyword_overlap"], errors="coerce").fillna(0.0)
        )
        - (
            config.stage2_weights.effort
            * pd.to_numeric(reranked_df["effort_penalty"], errors="coerce").fillna(0.0)
        )
    )
    reranked_df["stage2_score"] = stage2_score.clip(lower=0.0, upper=1.0)
    reranked_df["final_score"] = (
        (config.stage3_weights.stage2 * reranked_df["stage2_score"])
        + (
            config.stage3_weights.urgency
            * pd.to_numeric(reranked_df["urgency_boost"], errors="coerce").fillna(0.0)
        )
        + (
            config.stage3_weights.ev
            * pd.to_numeric(reranked_df["ev_proxy_norm"], errors="coerce").fillna(0.0)
        )
    )
    reranked_df = reranked_df.sort_values(
        by=["final_score", "_deadline_sort", "scholarship_id"],
        ascending=[False, True, True],
        na_position="last",
        kind="mergesort",
    )
    return reranked_df.reset_index(drop=True)


def _run_config_once(
    profile_caches: list[ProfileCache],
    *,
    config: WeightConfig,
    k: int,
) -> dict[str, Any]:
    per_profile_results: list[dict[str, Any]] = []
    per_profile_topk: dict[str, list[dict[str, Any]]] = {}
    ordered_ids: dict[str, list[str]] = {}
    relevance_labels: dict[str, list[int]] = {}

    for cache in profile_caches:
        profile_k = min(k, len(cache.eligible_df))

        if profile_k > 0:
            reranked_df = _rerank_profile_cache(cache, config)
            topk_df = reranked_df.head(profile_k).copy()
        else:
            topk_df = pd.DataFrame()

        records = _topk_records(topk_df, profile_k)
        per_profile_topk[cache.student.student_id] = records
        ordered_ids[cache.student.student_id] = [record["scholarship_id"] for record in records]
        relevance_labels[cache.student.student_id] = [
            _proxy_relevance_label(row, cache.student) for _, row in topk_df.iterrows()
        ]
        per_profile_results.append(
            {
                "student_id": cache.student.student_id,
                "eligible_df": cache.eligible_df,
                "ineligible_df": cache.ineligible_df,
            }
        )

    return {
        "per_profile_results": per_profile_results,
        "per_profile_topk": per_profile_topk,
        "ordered_ids": ordered_ids,
        "relevance_labels": relevance_labels,
    }


def _metrics_summary(
    *,
    first_run: dict[str, Any],
    second_run: dict[str, Any],
    k: int,
) -> dict[str, Any]:
    coverage = coverage_at_k(first_run["per_profile_topk"], k=k)
    sponsor_diversity = _field_diversity_at_k(first_run["per_profile_topk"], field="sponsor", k=k)
    source_diversity = _field_diversity_at_k(first_run["per_profile_topk"], field="source", k=k)
    amount_stats = amount_distribution_stats(first_run["per_profile_topk"], k=k)
    eligibility = eligibility_precision(first_run["per_profile_results"])
    ndcg = compute_ndcg_at_k(first_run["relevance_labels"], k=k)

    return {
        "ndcg_at_k": float(ndcg) if isinstance(ndcg, float) else 0.0,
        "coverage_at_k": float(coverage["coverage_at_k"]),
        "sponsor_diversity_at_k": float(sponsor_diversity["diversity_at_k"]),
        "source_diversity_at_k": float(source_diversity["diversity_at_k"]),
        "median_amount_max_topk": float(amount_stats["median"]),
        "eligibility_rate": float(eligibility["eligibility_precision"]),
        "ranking_stability": first_run["ordered_ids"] == second_run["ordered_ids"],
        "details": {
            "k": k,
            "coverage": coverage,
            "sponsor_diversity": sponsor_diversity,
            "source_diversity": source_diversity,
            "amount_distribution": amount_stats,
            "eligibility": eligibility,
        },
    }


def _amount_penalty(metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> float:
    baseline_median = float(baseline_metrics["median_amount_max_topk"])
    if baseline_median <= 0.0:
        return 0.0
    threshold = baseline_median * 1.5
    return max(0.0, float(metrics["median_amount_max_topk"]) - threshold)


def _leaderboard_sort_key(result: dict[str, Any], baseline_metrics: dict[str, Any]) -> tuple[Any, ...]:
    metrics = result["metrics"]
    return (
        not bool(metrics["ranking_stability"]),
        -float(metrics["ndcg_at_k"]),
        -float(metrics["coverage_at_k"]),
        -float(metrics["sponsor_diversity_at_k"]),
        _amount_penalty(metrics, baseline_metrics),
        result["config_id"],
    )


def _format_delta(current: float, baseline: float) -> str:
    return f"{current - baseline:+.4f}"


def _write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _report_markdown(
    *,
    timestamp_label: str,
    snapshot_path: Path,
    snapshot_count: int,
    k: int,
    seed: int,
    config_count: int,
    baseline_result: dict[str, Any],
    best_result: dict[str, Any],
    leaderboard: list[dict[str, Any]],
    similarity_mode: str = "tfidf",
    model_name: str | None = None,
) -> str:
    baseline_metrics = baseline_result["metrics"]
    best_metrics = best_result["metrics"]

    lines: list[str] = []
    lines.append("# Weight Tuning Report")
    lines.append("")
    lines.append(f"- Generated at (UTC): {timestamp_label}")
    lines.append(f"- Snapshot: `{snapshot_path}`")
    lines.append(f"- Snapshot records: {snapshot_count}")
    lines.append(f"- Top-K: {k}")
    lines.append(f"- Configs evaluated: {config_count}")
    lines.append(f"- Seed argument: {seed} (accepted for CLI parity; no randomness is used)")
    lines.append(f"- Similarity mode: `{similarity_mode}`")
    if similarity_mode == "embeddings" and model_name:
        lines.append(f"- Embedding model: `{model_name}`")
    lines.append("")
    lines.append("## Baseline")
    lines.append("")
    lines.append(f"- Config: `{baseline_result['config_id']}`")
    lines.append(f"- NDCG@{k}: {baseline_metrics['ndcg_at_k']:.4f}")
    lines.append(f"- Coverage@{k}: {baseline_metrics['coverage_at_k']:.4f}")
    lines.append(f"- Sponsor diversity@{k}: {baseline_metrics['sponsor_diversity_at_k']:.4f}")
    lines.append(f"- Source diversity@{k}: {baseline_metrics['source_diversity_at_k']:.4f}")
    lines.append(f"- Median amount_max in top-{k}: {baseline_metrics['median_amount_max_topk']:.2f}")
    lines.append(f"- Eligibility rate: {baseline_metrics['eligibility_rate']:.4f}")
    lines.append(f"- Ranking stability: {baseline_metrics['ranking_stability']}")
    lines.append("")
    lines.append("## Leaderboard (Top 20)")
    lines.append("")
    lines.append(
        "| Rank | Config ID | NDCG | Coverage | Sponsor Div | Source Div | Median Amount | Stable |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for index, result in enumerate(leaderboard[:20], start=1):
        metrics = result["metrics"]
        lines.append(
            "| "
            f"{index} | `{result['config_id']}` | {metrics['ndcg_at_k']:.4f} | "
            f"{metrics['coverage_at_k']:.4f} | {metrics['sponsor_diversity_at_k']:.4f} | "
            f"{metrics['source_diversity_at_k']:.4f} | {metrics['median_amount_max_topk']:.2f} | "
            f"{metrics['ranking_stability']} |"
        )
    lines.append("")
    lines.append("## Best Config")
    lines.append("")
    lines.append(f"- Config: `{best_result['config_id']}`")
    lines.append(f"- Stage 2 weights: `{json.dumps(best_result['stage2_weights'], sort_keys=True)}`")
    lines.append(f"- Stage 3 weights: `{json.dumps(best_result['stage3_weights'], sort_keys=True)}`")
    lines.append(f"- Amount utility mode: `{best_result['amount_utility_mode']}`")
    lines.append(f"- NDCG@{k}: {best_metrics['ndcg_at_k']:.4f}")
    lines.append(f"- Coverage@{k}: {best_metrics['coverage_at_k']:.4f}")
    lines.append(f"- Sponsor diversity@{k}: {best_metrics['sponsor_diversity_at_k']:.4f}")
    lines.append(f"- Source diversity@{k}: {best_metrics['source_diversity_at_k']:.4f}")
    lines.append(f"- Median amount_max in top-{k}: {best_metrics['median_amount_max_topk']:.2f}")
    lines.append(f"- Eligibility rate: {best_metrics['eligibility_rate']:.4f}")
    lines.append(f"- Ranking stability: {best_metrics['ranking_stability']}")
    lines.append("")
    lines.append("## Delta vs Baseline")
    lines.append("")
    lines.append(
        f"- NDCG@{k}: {_format_delta(best_metrics['ndcg_at_k'], baseline_metrics['ndcg_at_k'])}"
    )
    lines.append(
        f"- Coverage@{k}: {_format_delta(best_metrics['coverage_at_k'], baseline_metrics['coverage_at_k'])}"
    )
    lines.append(
        "- Sponsor diversity@"
        f"{k}: {_format_delta(best_metrics['sponsor_diversity_at_k'], baseline_metrics['sponsor_diversity_at_k'])}"
    )
    lines.append(
        "- Source diversity@"
        f"{k}: {_format_delta(best_metrics['source_diversity_at_k'], baseline_metrics['source_diversity_at_k'])}"
    )
    lines.append(
        "- Median amount_max in top-"
        f"{k}: {_format_delta(best_metrics['median_amount_max_topk'], baseline_metrics['median_amount_max_topk'])}"
    )
    lines.append(
        f"- Eligibility rate: {_format_delta(best_metrics['eligibility_rate'], baseline_metrics['eligibility_rate'])}"
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Proxy relevance labels are heuristic labels from the existing golden harness, not human judgments.")
    lines.append("- Evaluation uses a saved parquet snapshot only; this experiment never triggers live ingest.")
    lines.append("- The selection objective is deterministic: NDCG@K, then coverage@K, then sponsor diversity@K, then lower amount-dominance penalty, then config ID.")
    lines.append("- Next steps: inspect per-profile top-K IDs in the JSON artifact before promoting weights broadly.")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.k < 1:
        raise ValueError("--k must be at least 1.")
    if args.max_configs < 1:
        raise ValueError("--max_configs must be at least 1.")

    snapshot_path = _resolve_snapshot_path(args.snapshot)
    outdir = _resolve_outdir(args.outdir)
    artifacts_dir = outdir / "artifacts"
    timestamp = datetime.now(tz=UTC)
    timestamp_label = timestamp.strftime("%Y%m%d_%H%M%S")

    snapshot_df = pd.read_parquet(snapshot_path)
    students = get_golden_students()
    profile_caches = _prepare_profile_caches(
        snapshot_df,
        students,
        similarity_mode=args.similarity_mode,
        model_name=args.model_name,
    )
    configs = generate_candidate_configs(max_configs=args.max_configs)

    evaluation_results: list[dict[str, Any]] = []
    for config in configs:
        first_run = _run_config_once(profile_caches, config=config, k=args.k)
        second_run = _run_config_once(profile_caches, config=config, k=args.k)
        metrics = _metrics_summary(first_run=first_run, second_run=second_run, k=args.k)
        evaluation_results.append(
            {
                **config.to_dict(),
                "metrics": metrics,
                "per_profile_topk_ids": first_run["ordered_ids"],
            }
        )

    baseline_result = evaluation_results[0]
    leaderboard = sorted(
        evaluation_results,
        key=lambda result: _leaderboard_sort_key(result, baseline_result["metrics"]),
    )
    best_result = leaderboard[0]

    report_path = outdir / f"weight_tuning_{timestamp_label}.md"
    artifact_path = artifacts_dir / f"weight_tuning_{timestamp_label}.json"

    report_text = _report_markdown(
        timestamp_label=timestamp.isoformat(),
        snapshot_path=snapshot_path,
        snapshot_count=int(len(snapshot_df)),
        k=args.k,
        seed=args.seed,
        config_count=len(evaluation_results),
        similarity_mode=args.similarity_mode,
        model_name=args.model_name if args.similarity_mode == "embeddings" else None,
        baseline_result=baseline_result,
        best_result=best_result,
        leaderboard=leaderboard,
    )
    _write_markdown(report_path, report_text)

    artifact_payload = {
        "generated_at": timestamp.isoformat(),
        "snapshot_path": str(snapshot_path),
        "snapshot_count": int(len(snapshot_df)),
        "k": int(args.k),
        "seed": int(args.seed),
        "similarity_mode": args.similarity_mode,
        "model_name": args.model_name if args.similarity_mode == "embeddings" else None,
        "baseline_metrics": baseline_result["metrics"],
        "best_config": {
            "config_id": best_result["config_id"],
            "stage2_weights": best_result["stage2_weights"],
            "stage3_weights": best_result["stage3_weights"],
            "amount_utility_mode": best_result["amount_utility_mode"],
            "similarity_mode": args.similarity_mode,
            "model_name": args.model_name if args.similarity_mode == "embeddings" else None,
            "metrics": best_result["metrics"],
        },
        "per_profile_topk_ids_best_config": best_result["per_profile_topk_ids"],
        "all_configs": leaderboard,
    }
    write_json_atomic(artifact_payload, artifact_path)

    best_weights_payload = {
        "stage2_weights": best_result["stage2_weights"],
        "stage3_weights": best_result["stage3_weights"],
        "amount_utility_mode": best_result["amount_utility_mode"],
        "similarity_mode": args.similarity_mode,
        "model_name": args.model_name if args.similarity_mode == "embeddings" else None,
        "snapshot_used": str(snapshot_path),
        "timestamp": timestamp.isoformat(),
    }
    write_json_atomic(best_weights_payload, BEST_WEIGHTS_PATH)

    print(f"Report written to {report_path}")
    print(f"Artifact written to {artifact_path}")
    print(f"Best weights written to {BEST_WEIGHTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
