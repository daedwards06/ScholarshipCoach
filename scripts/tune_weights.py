from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.eval.golden_students import GoldenStudent, get_golden_students
from src.eval.metrics import amount_distribution_stats, compute_ndcg_at_k, coverage_at_k, eligibility_precision
from src.eval.relevance import (
    RelevanceConfig,
    calibrate_similarity_threshold,
    get_similarity_threshold,
    proxy_relevance_label,
)
from src.io.snapshotting import get_latest_snapshot_path, write_json_atomic
from src.rank.stage1_eligibility import apply_eligibility_filter
from src.rank.stage2_scoring import score_stage2
from src.rank.stage3_rerank import rerank_stage3
from src.rank.weights import Stage2Weights, Stage3Weights
from src.win_model.infer import get_latest_model_path, load_latest_model

ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_K = 10
DEFAULT_MAX_CONFIGS = 200
DEFAULT_OUTDIR = ROOT_DIR / "reports" / "weight_tuning"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
CALIBRATION_TARGET_SHARE = 0.25
DEFAULT_SELECTION_OBJECTIVE = "relevance"
DEFAULT_OBJECTIVE_WEIGHTS_TEXT = "ndcg=0.7,coverage=0.2,ev=0.1"
DEFAULT_PARETO_METRICS_TEXT = "ndcg,coverage,ev"
DEFAULT_PARETO_TOP_N = 20
BLENDED_METRIC_NAMES = ("ndcg", "coverage", "ev")
BEST_WEIGHTS_FILENAMES = {
    "relevance": "best_weights_relevance.json",
    "blended": "best_weights_blended.json",
    "pareto": "best_weights_pareto.json",
}
BEST_WEIGHTS_LATEST_PATH = PROCESSED_DIR / "best_weights_latest.json"
_NA_METRIC_TEXT = "N/A"
_EPSILON = 1e-9


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--use-win-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use expected_value_norm from the trained win model as the Stage 3 ev signal.",
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
        action="store_true",
        default=False,
        help="Use a deterministic calibrated threshold for the active similarity mode for this run only.",
    )
    parser.add_argument(
        "--selection-objective",
        choices=("relevance", "blended", "pareto"),
        default=DEFAULT_SELECTION_OBJECTIVE,
        help="Primary selection objective. Defaults to relevance.",
    )
    parser.add_argument(
        "--objective-weights",
        type=str,
        default=DEFAULT_OBJECTIVE_WEIGHTS_TEXT,
        help=(
            "Blended-objective weights as comma-separated key=value pairs. "
            f"Defaults to '{DEFAULT_OBJECTIVE_WEIGHTS_TEXT}'."
        ),
    )
    parser.add_argument(
        "--pareto-metrics",
        type=str,
        default=DEFAULT_PARETO_METRICS_TEXT,
        help=(
            "Comma-separated metrics for Pareto dominance. "
            f"Defaults to '{DEFAULT_PARETO_METRICS_TEXT}'."
        ),
    )
    parser.add_argument(
        "--pareto-top-n",
        type=int,
        default=DEFAULT_PARETO_TOP_N,
        help=f"Number of Pareto configs to show in reports. Defaults to {DEFAULT_PARETO_TOP_N}.",
    )
    return parser.parse_args(argv)


def _resolve_snapshot_path(snapshot: Path | None) -> Path:
    if snapshot is not None:
        return snapshot if snapshot.is_absolute() else ROOT_DIR / snapshot
    latest = get_latest_snapshot_path(PROCESSED_DIR)
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
                "p_win": _safe_float(row.get("p_win")),
                "expected_value": _safe_float(row.get("expected_value")),
                "expected_value_norm": _safe_float(row.get("expected_value_norm")),
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


def _default_ev_metrics() -> dict[str, float | None]:
    return {
        "avg_p_win_topk": None,
        "median_p_win_topk": None,
        "avg_expected_value_topk": None,
        "median_expected_value_topk": None,
        "avg_expected_value_norm_topk": None,
    }


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _topk_ev_metrics(topk_df: pd.DataFrame, *, use_win_model: bool) -> dict[str, float | None]:
    if not use_win_model:
        return _default_ev_metrics()
    required_columns = {"p_win", "expected_value", "expected_value_norm"}
    if topk_df.empty or not required_columns.issubset(set(topk_df.columns)):
        return _default_ev_metrics()

    p_win = pd.to_numeric(topk_df["p_win"], errors="coerce").dropna()
    expected_value = pd.to_numeric(topk_df["expected_value"], errors="coerce").dropna()
    expected_value_norm = pd.to_numeric(topk_df["expected_value_norm"], errors="coerce").dropna()
    if p_win.empty or expected_value.empty or expected_value_norm.empty:
        return _default_ev_metrics()

    return {
        "avg_p_win_topk": float(p_win.mean()),
        "median_p_win_topk": float(p_win.median()),
        "avg_expected_value_topk": float(expected_value.mean()),
        "median_expected_value_topk": float(expected_value.median()),
        "avg_expected_value_norm_topk": float(expected_value_norm.mean()),
    }


def _prepare_profile_caches(
    snapshot_df: pd.DataFrame,
    students: list[GoldenStudent],
    *,
    similarity_mode: str,
    model_name: str,
    use_win_model: bool,
    win_model: Any | None,
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
                profile=student.profile,
                weights=Stage3Weights.baseline(),
                use_win_model=use_win_model,
                win_model=win_model,
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


def _rerank_profile_cache(
    cache: ProfileCache,
    config: WeightConfig,
    *,
    use_win_model: bool,
) -> pd.DataFrame:
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
    ev_column = "expected_value_norm" if use_win_model else "ev_proxy_norm"
    reranked_df["final_score"] = (
        (config.stage3_weights.stage2 * reranked_df["stage2_score"])
        + (
            config.stage3_weights.urgency
            * pd.to_numeric(reranked_df["urgency_boost"], errors="coerce").fillna(0.0)
        )
        + (
            config.stage3_weights.ev
            * pd.to_numeric(reranked_df.get(ev_column), errors="coerce").fillna(0.0)
        )
    )
    reranked_df = reranked_df.sort_values(
        by=["final_score", "_deadline_sort", "scholarship_id"],
        ascending=[False, True, True],
        na_position="last",
        kind="mergesort",
    )
    return reranked_df.reset_index(drop=True)


def _resolved_relevance_config(
    profile_caches: list[ProfileCache],
    *,
    similarity_mode: str,
    relevance_config: RelevanceConfig,
    calibrate_thresholds: bool,
) -> tuple[RelevanceConfig, dict[str, Any]]:
    calibrated_threshold: float | None = None
    if calibrate_thresholds:
        calibrated_threshold = calibrate_similarity_threshold(
            [cache.component_df for cache in profile_caches],
            target_share=CALIBRATION_TARGET_SHARE,
        )

    effective_config = relevance_config
    if calibrated_threshold is not None:
        if similarity_mode == "tfidf":
            effective_config = RelevanceConfig(
                label_mode=relevance_config.label_mode,
                tfidf_threshold=float(calibrated_threshold),
                embed_threshold=relevance_config.embed_threshold,
                strict_requires_all_of=relevance_config.strict_requires_all_of,
            )
        elif similarity_mode == "embeddings":
            effective_config = RelevanceConfig(
                label_mode=relevance_config.label_mode,
                tfidf_threshold=relevance_config.tfidf_threshold,
                embed_threshold=float(calibrated_threshold),
                strict_requires_all_of=relevance_config.strict_requires_all_of,
            )

    labeling = {
        "similarity_mode": similarity_mode,
        "label_mode": effective_config.label_mode,
        "tfidf_threshold": float(effective_config.tfidf_threshold),
        "embed_threshold": float(effective_config.embed_threshold),
        "active_threshold": float(get_similarity_threshold(similarity_mode, effective_config)),
        "calibration_enabled": bool(calibrate_thresholds),
        "calibrated_threshold": float(calibrated_threshold) if calibrated_threshold is not None else None,
        "calibration_target_similarity_only_share": CALIBRATION_TARGET_SHARE if calibrate_thresholds else None,
    }
    return effective_config, labeling


def _run_config_once(
    profile_caches: list[ProfileCache],
    *,
    config: WeightConfig,
    k: int,
    use_win_model: bool,
    similarity_mode: str,
    relevance_config: RelevanceConfig,
) -> dict[str, Any]:
    per_profile_results: list[dict[str, Any]] = []
    per_profile_topk: dict[str, list[dict[str, Any]]] = {}
    ordered_ids: dict[str, list[str]] = {}
    relevance_labels: dict[str, list[int]] = {}
    per_profile_metrics: dict[str, dict[str, Any]] = {}

    for cache in profile_caches:
        profile_k = min(k, len(cache.eligible_df))

        if profile_k > 0:
            reranked_df = _rerank_profile_cache(cache, config, use_win_model=use_win_model)
            topk_df = reranked_df.head(profile_k).copy()
        else:
            topk_df = pd.DataFrame()

        records = _topk_records(topk_df, profile_k)
        per_profile_topk[cache.student.student_id] = records
        ordered_ids[cache.student.student_id] = [record["scholarship_id"] for record in records]
        relevance_labels[cache.student.student_id] = [
            proxy_relevance_label(
                row,
                cache.student,
                similarity_mode=similarity_mode,
                cfg=relevance_config,
            )
            for _, row in topk_df.iterrows()
        ]
        ev_metrics = _topk_ev_metrics(topk_df, use_win_model=use_win_model)
        per_profile_metrics[cache.student.student_id] = {
            "student_id": cache.student.student_id,
            "eligible_count": int(len(cache.eligible_df)),
            "topk_count": int(profile_k),
            **ev_metrics,
        }
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
        "per_profile_metrics": per_profile_metrics,
    }


def _metrics_summary(
    *,
    first_run: dict[str, Any],
    second_run: dict[str, Any],
    k: int,
    use_win_model: bool,
) -> dict[str, Any]:
    coverage = coverage_at_k(first_run["per_profile_topk"], k=k)
    sponsor_diversity = _field_diversity_at_k(first_run["per_profile_topk"], field="sponsor", k=k)
    source_diversity = _field_diversity_at_k(first_run["per_profile_topk"], field="source", k=k)
    amount_stats = amount_distribution_stats(first_run["per_profile_topk"], k=k)
    eligibility = eligibility_precision(first_run["per_profile_results"])
    ndcg = compute_ndcg_at_k(first_run["relevance_labels"], k=k)
    profile_metrics = list(first_run["per_profile_metrics"].values())

    aggregate_ev_metrics = _default_ev_metrics()
    if use_win_model:
        for metric_name in aggregate_ev_metrics:
            values = [
                float(metric[metric_name])
                for metric in profile_metrics
                if metric.get(metric_name) is not None
            ]
            aggregate_ev_metrics[metric_name] = _mean_or_none(values)

    return {
        "ndcg_at_k": float(ndcg) if isinstance(ndcg, float) else 0.0,
        "coverage_at_k": float(coverage["coverage_at_k"]),
        "sponsor_diversity_at_k": float(sponsor_diversity["diversity_at_k"]),
        "source_diversity_at_k": float(source_diversity["diversity_at_k"]),
        "median_amount_max_topk": float(amount_stats["median"]),
        "eligibility_rate": float(eligibility["eligibility_precision"]),
        "ranking_stability": first_run["ordered_ids"] == second_run["ordered_ids"],
        **aggregate_ev_metrics,
        "details": {
            "k": k,
            "coverage": coverage,
            "sponsor_diversity": sponsor_diversity,
            "source_diversity": source_diversity,
            "amount_distribution": amount_stats,
            "eligibility": eligibility,
            "per_profile_metrics": first_run["per_profile_metrics"],
        },
    }


def _amount_penalty(metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> float:
    baseline_median = float(baseline_metrics["median_amount_max_topk"])
    if baseline_median <= 0.0:
        return 0.0
    threshold = baseline_median * 1.5
    return max(0.0, float(metrics["median_amount_max_topk"]) - threshold)


def _relevance_sort_key(result: dict[str, Any], baseline_metrics: dict[str, Any]) -> tuple[Any, ...]:
    metrics = result["metrics"]
    return (
        not bool(metrics["ranking_stability"]),
        -float(metrics["ndcg_at_k"]),
        -float(metrics["coverage_at_k"]),
        -float(metrics["sponsor_diversity_at_k"]),
        _amount_penalty(metrics, baseline_metrics),
        result["config_id"],
    )


def _normalize_metric_token(token: str) -> str:
    normalized = token.strip().lower().replace("@k", "").replace("_at_k", "")
    aliases = {
        "ndcg": "ndcg",
        "coverage": "coverage",
        "ev": "ev",
        "expected_value": "ev",
        "expected_value_norm": "ev",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported metric '{token}'. Supported metrics: ndcg, coverage, ev."
        )
    return aliases[normalized]


def _metric_value(metrics: dict[str, Any], metric_name: str) -> float:
    if metric_name == "ndcg":
        return float(metrics["ndcg_at_k"])
    if metric_name == "coverage":
        return float(metrics["coverage_at_k"])
    if metric_name == "ev":
        return float(metrics.get("avg_expected_value_norm_topk") or 0.0)
    raise ValueError(f"Unsupported metric '{metric_name}'.")


def _resolve_objective_weights(
    weights_text: str,
    *,
    use_win_model: bool,
    strict: bool = True,
) -> dict[str, float] | None:
    weights = {name: 0.0 for name in BLENDED_METRIC_NAMES}
    if not weights_text.strip():
        raise ValueError("--objective-weights may not be empty.")

    for part in weights_text.split(","):
        token = part.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"Invalid objective weight '{token}'. Expected comma-separated key=value pairs."
            )
        raw_name, raw_value = token.split("=", 1)
        metric_name = _normalize_metric_token(raw_name)
        weights[metric_name] = float(raw_value)

    total = sum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(
            f"--objective-weights must sum to 1.0 within tolerance; received {total:.6f}."
        )

    if not use_win_model and weights["ev"] > 0.0:
        if strict:
            raise ValueError(
                "Blended objective requested a non-zero ev weight, but --use-win-model is disabled."
            )
        return None

    return {name: float(value) for name, value in weights.items()}


def _resolve_pareto_metrics(metrics_text: str, *, use_win_model: bool) -> list[str]:
    if not metrics_text.strip():
        raise ValueError("--pareto-metrics may not be empty.")

    metrics: list[str] = []
    for token in metrics_text.split(","):
        metric_name = _normalize_metric_token(token)
        if metric_name == "ev" and not use_win_model:
            continue
        if metric_name not in metrics:
            metrics.append(metric_name)

    if not metrics:
        raise ValueError("No valid Pareto metrics remain after applying current win-model settings.")
    return metrics


def _blended_score(metrics: dict[str, Any], objective_weights: dict[str, float]) -> float:
    return sum(objective_weights[name] * _metric_value(metrics, name) for name in BLENDED_METRIC_NAMES)


def _with_blended_scores(
    results: list[dict[str, Any]],
    *,
    objective_weights: dict[str, float] | None,
) -> None:
    for result in results:
        result["objective_score"] = (
            None
            if objective_weights is None
            else _blended_score(result["metrics"], objective_weights)
        )


def _blended_sort_key(result: dict[str, Any]) -> tuple[Any, ...]:
    metrics = result["metrics"]
    return (
        -float(result.get("objective_score") or 0.0),
        -float(metrics["ndcg_at_k"]),
        -float(metrics["coverage_at_k"]),
        -_metric_value(metrics, "ev"),
        result["config_id"],
    )


def _dominates(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    metric_names: list[str],
) -> bool:
    left_values = [_metric_value(left["metrics"], metric_name) for metric_name in metric_names]
    right_values = [_metric_value(right["metrics"], metric_name) for metric_name in metric_names]
    at_least_as_good = all(
        left_value >= right_value - _EPSILON
        for left_value, right_value in zip(left_values, right_values)
    )
    strictly_better = any(
        left_value > right_value + _EPSILON
        for left_value, right_value in zip(left_values, right_values)
    )
    return at_least_as_good and strictly_better


def _pareto_sort_key(result: dict[str, Any]) -> tuple[Any, ...]:
    metrics = result["metrics"]
    return (
        -float(metrics["ndcg_at_k"]),
        -float(metrics["coverage_at_k"]),
        -_metric_value(metrics, "ev"),
        result["config_id"],
    )


def _pareto_front(
    results: list[dict[str, Any]],
    *,
    metric_names: list[str],
) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for candidate in results:
        if any(
            _dominates(other, candidate, metric_names=metric_names)
            for other in results
            if other is not candidate
        ):
            continue
        front.append(candidate)
    return sorted(front, key=_pareto_sort_key)


def _pareto_knee_point(
    pareto_front: list[dict[str, Any]],
    *,
    metric_names: list[str],
) -> dict[str, Any]:
    if not pareto_front:
        raise ValueError("Pareto front is empty.")

    metric_ranges: dict[str, tuple[float, float]] = {}
    for metric_name in metric_names:
        values = [_metric_value(result["metrics"], metric_name) for result in pareto_front]
        metric_ranges[metric_name] = (min(values), max(values))

    def _knee_score(result: dict[str, Any]) -> float:
        squared_sum = 0.0
        for metric_name in metric_names:
            minimum, maximum = metric_ranges[metric_name]
            value = _metric_value(result["metrics"], metric_name)
            if math.isclose(maximum, minimum, abs_tol=_EPSILON):
                normalized = 1.0
            else:
                normalized = (value - minimum) / (maximum - minimum)
            squared_sum += normalized * normalized
        return math.sqrt(squared_sum)

    for result in pareto_front:
        result["pareto_knee_score"] = _knee_score(result)

    return sorted(
        pareto_front,
        key=lambda result: (-float(result["pareto_knee_score"]), *_pareto_sort_key(result)),
    )[0]


def _format_delta(current: float, baseline: float) -> str:
    return f"{current - baseline:+.4f}"


def _format_metric(value: float | None, *, decimals: int = 4, na_text: str = _NA_METRIC_TEXT) -> str:
    if value is None:
        return na_text
    return f"{float(value):.{decimals}f}"


def _best_config_snapshot(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "config_id": result["config_id"],
        "stage2_weights": result["stage2_weights"],
        "stage3_weights": result["stage3_weights"],
        "amount_utility_mode": result["amount_utility_mode"],
        "metrics": result["metrics"],
        "objective_score": result.get("objective_score"),
        "pareto_knee_score": result.get("pareto_knee_score"),
    }


def _append_metric_section(lines: list[str], *, metrics: dict[str, Any], k: int, use_win_model: bool) -> None:
    lines.append(f"- NDCG@{k}: {metrics['ndcg_at_k']:.4f}")
    lines.append(f"- Coverage@{k}: {metrics['coverage_at_k']:.4f}")
    lines.append(f"- Sponsor diversity@{k}: {metrics['sponsor_diversity_at_k']:.4f}")
    lines.append(f"- Source diversity@{k}: {metrics['source_diversity_at_k']:.4f}")
    lines.append(f"- Median amount_max in top-{k}: {metrics['median_amount_max_topk']:.2f}")
    lines.append(f"- Eligibility rate: {metrics['eligibility_rate']:.4f}")
    lines.append(f"- Ranking stability: {metrics['ranking_stability']}")
    if use_win_model:
        lines.append(f"- Mean p_win in top-{k}: {_format_metric(metrics.get('avg_p_win_topk'))}")
        lines.append(f"- Median p_win in top-{k}: {_format_metric(metrics.get('median_p_win_topk'))}")
        lines.append(
            f"- Mean expected_value in top-{k}: {_format_metric(metrics.get('avg_expected_value_topk'), decimals=2)}"
        )
        lines.append(
            f"- Median expected_value in top-{k}: {_format_metric(metrics.get('median_expected_value_topk'), decimals=2)}"
        )
        lines.append(
            f"- Mean expected_value_norm in top-{k}: {_format_metric(metrics.get('avg_expected_value_norm_topk'))}"
        )


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
    selection_best_result: dict[str, Any],
    relevance_best_result: dict[str, Any],
    best_config_per_objective: dict[str, Any],
    leaderboard: list[dict[str, Any]],
    selection_objective: str,
    objective_weights: dict[str, float] | None,
    pareto_metrics: list[str],
    pareto_front: list[dict[str, Any]],
    pareto_top_n: int,
    similarity_mode: str = "tfidf",
    model_name: str | None = None,
    use_win_model: bool = False,
    win_model_path: Path | None = None,
    labeling: dict[str, Any] | None = None,
) -> str:
    baseline_metrics = baseline_result["metrics"]
    best_metrics = selection_best_result["metrics"]
    labeling = labeling or {}

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
    lines.append(f"- Label mode: `{labeling.get('label_mode', 'hybrid')}`")
    lines.append(f"- TF-IDF relevance threshold: {float(labeling.get('tfidf_threshold', 0.12)):.4f}")
    lines.append(f"- Embeddings relevance threshold: {float(labeling.get('embed_threshold', 0.30)):.4f}")
    lines.append(f"- Active relevance threshold: {float(labeling.get('active_threshold', 0.0)):.4f}")
    lines.append(f"- Calibration enabled: {bool(labeling.get('calibration_enabled', False))}")
    lines.append(f"- Selection objective: `{selection_objective}`")
    if objective_weights is not None:
        lines.append(f"- Blended objective weights: `{json.dumps(objective_weights, sort_keys=True)}`")
    lines.append(f"- Pareto metrics: `{','.join(pareto_metrics)}`")
    lines.append(f"- Pareto top-N in report: {pareto_top_n}")
    if bool(labeling.get("calibration_enabled", False)):
        calibrated_threshold = labeling.get("calibrated_threshold")
        if calibrated_threshold is None:
            lines.append("- Calibrated threshold used: n/a")
        else:
            lines.append(f"- Calibrated threshold used: {float(calibrated_threshold):.4f}")
    if similarity_mode == "embeddings" and model_name:
        lines.append(f"- Embedding model: `{model_name}`")
    lines.append(f"- Use win model: {use_win_model}")
    if use_win_model:
        lines.append(f"- Win model path: `{win_model_path}`" if win_model_path is not None else "- Win model path: latest")
        lines.append("- Stage 3 ev field: `expected_value_norm`")
    else:
        lines.append("- Stage 3 ev field: `ev_proxy_norm`")
    lines.append("")
    lines.append("## Baseline Metrics")
    lines.append("")
    lines.append(f"- Config: `{baseline_result['config_id']}`")
    _append_metric_section(lines, metrics=baseline_metrics, k=k, use_win_model=use_win_model)
    lines.append("")
    lines.append("## Best Config By Objective")
    lines.append("")
    objective_key_map = {
        "relevance": "relevance_best",
        "blended": "blended_best",
        "pareto": "pareto_knee_best",
    }
    for objective_name in ("relevance", "blended", "pareto"):
        result = best_config_per_objective.get(objective_key_map[objective_name])
        if result is None:
            lines.append(f"- {objective_name}: unavailable")
            continue
        metrics = result["metrics"]
        summary = (
            f"`{result['config_id']}`"
            f" (NDCG={metrics['ndcg_at_k']:.4f}, Coverage={metrics['coverage_at_k']:.4f}, "
            f"EV Norm={_format_metric(metrics.get('avg_expected_value_norm_topk'))}"
        )
        if result.get("objective_score") is not None:
            summary += f", Objective={float(result['objective_score']):.4f}"
        summary += ")"
        lines.append(f"- {objective_name}: {summary}")
    lines.append("")
    lines.append("## Best Config Under Selection Objective")
    lines.append("")
    lines.append(f"- Objective: `{selection_objective}`")
    lines.append(f"- Config: `{selection_best_result['config_id']}`")
    lines.append(
        f"- Stage 2 weights: `{json.dumps(selection_best_result['stage2_weights'], sort_keys=True)}`"
    )
    lines.append(
        f"- Stage 3 weights: `{json.dumps(selection_best_result['stage3_weights'], sort_keys=True)}`"
    )
    lines.append(f"- Amount utility mode: `{selection_best_result['amount_utility_mode']}`")
    if selection_best_result.get("objective_score") is not None:
        lines.append(f"- Objective score: {float(selection_best_result['objective_score']):.4f}")
    if selection_best_result.get("pareto_knee_score") is not None:
        lines.append(f"- Pareto knee score: {float(selection_best_result['pareto_knee_score']):.4f}")
    _append_metric_section(lines, metrics=best_metrics, k=k, use_win_model=use_win_model)
    if selection_objective != "relevance":
        lines.append("")
        lines.append("## Best Relevance Config (Comparison)")
        lines.append("")
        lines.append(f"- Config: `{relevance_best_result['config_id']}`")
        _append_metric_section(lines, metrics=relevance_best_result["metrics"], k=k, use_win_model=use_win_model)
    lines.append("")
    lines.append("## Leaderboard (Top 20)")
    lines.append("")
    if selection_objective == "blended":
        lines.append(
            "| Rank | Config ID | Objective | NDCG | Coverage | EV Norm | Mean EV | Mean p_win | Stable |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for index, result in enumerate(leaderboard[:20], start=1):
            metrics = result["metrics"]
            lines.append(
                "| "
                f"{index} | `{result['config_id']}` | {float(result.get('objective_score') or 0.0):.4f} | "
                f"{metrics['ndcg_at_k']:.4f} | {metrics['coverage_at_k']:.4f} | "
                f"{_format_metric(metrics.get('avg_expected_value_norm_topk'))} | "
                f"{_format_metric(metrics.get('avg_expected_value_topk'), decimals=2)} | "
                f"{_format_metric(metrics.get('avg_p_win_topk'))} | {metrics['ranking_stability']} |"
            )
    else:
        lines.append(
            "| Rank | Config ID | NDCG | Coverage | EV Norm | Mean EV | Mean p_win | Sponsor Div | Stable |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for index, result in enumerate(leaderboard[:20], start=1):
            metrics = result["metrics"]
            lines.append(
                "| "
                f"{index} | `{result['config_id']}` | {metrics['ndcg_at_k']:.4f} | "
                f"{metrics['coverage_at_k']:.4f} | {_format_metric(metrics.get('avg_expected_value_norm_topk'))} | "
                f"{_format_metric(metrics.get('avg_expected_value_topk'), decimals=2)} | "
                f"{_format_metric(metrics.get('avg_p_win_topk'))} | "
                f"{metrics['sponsor_diversity_at_k']:.4f} | {metrics['ranking_stability']} |"
            )
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
    if use_win_model:
        lines.append(
            f"- Mean expected_value_norm@{k}: "
            f"{_format_delta(float(best_metrics.get('avg_expected_value_norm_topk') or 0.0), float(baseline_metrics.get('avg_expected_value_norm_topk') or 0.0))}"
        )
        lines.append(
            f"- Mean expected_value@{k}: "
            f"{_format_delta(float(best_metrics.get('avg_expected_value_topk') or 0.0), float(baseline_metrics.get('avg_expected_value_topk') or 0.0))}"
        )
    if selection_objective == "pareto":
        lines.append("")
        lines.append("## Pareto Front")
        lines.append("")
        lines.append("- Non-dominated configs are sorted by NDCG, then coverage, then EV norm.")
        lines.append("- Knee point maximizes normalized distance from the origin across the active Pareto metrics.")
        lines.append("")
        lines.append("| Rank | Config ID | NDCG | Coverage | EV Norm | Knee Score |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for index, result in enumerate(pareto_front[:pareto_top_n], start=1):
            metrics = result["metrics"]
            lines.append(
                "| "
                f"{index} | `{result['config_id']}` | {metrics['ndcg_at_k']:.4f} | "
                f"{metrics['coverage_at_k']:.4f} | {_format_metric(metrics.get('avg_expected_value_norm_topk'))} | "
                f"{_format_metric(result.get('pareto_knee_score'))} |"
            )
        lines.append("")
        lines.append(f"- Selected knee config: `{selection_best_result['config_id']}`")
        lines.append(
            "- Rationale: highest normalized multi-metric distance on the non-dominated front, "
            "with deterministic tie-breaks on NDCG, coverage, EV, then config ID."
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Proxy relevance labels are heuristic labels from the existing golden harness, not human judgments.")
    lines.append("- Evaluation uses a saved parquet snapshot only; this experiment never triggers live ingest.")
    lines.append("- Best weights are versioned per objective and `best_weights_latest.json` points to the file selected by this run.")
    lines.append("- Next steps: inspect per-profile top-K IDs and per-profile EV metrics in the JSON artifact before promoting weights broadly.")
    return "\n".join(lines) + "\n"


def _best_weights_path_for_objective(processed_dir: Path, objective: str) -> Path:
    if objective not in BEST_WEIGHTS_FILENAMES:
        raise ValueError(f"Unsupported objective '{objective}'.")
    return processed_dir / BEST_WEIGHTS_FILENAMES[objective]


def _write_best_weights_artifacts(
    *,
    processed_dir: Path,
    objective: str,
    best_result: dict[str, Any],
    use_win_model: bool,
    similarity_mode: str,
    model_name: str | None,
    labeling: dict[str, Any],
    snapshot_path: Path,
    timestamp: datetime,
) -> tuple[Path, Path]:
    best_weights_path = _best_weights_path_for_objective(processed_dir, objective)
    best_weights_payload = {
        "objective": objective,
        "config_id": best_result["config_id"],
        "stage2_weights": best_result["stage2_weights"],
        "stage3_weights": best_result["stage3_weights"],
        "amount_utility_mode": best_result["amount_utility_mode"],
        "use_win_model": bool(use_win_model),
        "similarity_mode": similarity_mode,
        "model_name": model_name,
        "label_mode": labeling["label_mode"],
        "thresholds": {
            "tfidf_threshold": labeling["tfidf_threshold"],
            "embed_threshold": labeling["embed_threshold"],
            "active_threshold": labeling["active_threshold"],
        },
        "calibration_enabled": labeling["calibration_enabled"],
        "snapshot_used": str(snapshot_path),
        "timestamp": timestamp.isoformat(),
        "metrics": best_result["metrics"],
        "objective_score": best_result.get("objective_score"),
        "pareto_knee_score": best_result.get("pareto_knee_score"),
    }
    if labeling["calibrated_threshold"] is not None:
        best_weights_payload["calibrated_threshold"] = labeling["calibrated_threshold"]
    write_json_atomic(best_weights_payload, best_weights_path)

    latest_pointer_path = processed_dir / BEST_WEIGHTS_LATEST_PATH.name
    write_json_atomic(
        {
            "objective": objective,
            "path": str(best_weights_path),
            "timestamp": timestamp.isoformat(),
            "config_id": best_result["config_id"],
            "use_win_model": bool(use_win_model),
            "similarity_mode": similarity_mode,
        },
        latest_pointer_path,
    )
    return best_weights_path, latest_pointer_path


def main() -> int:
    args = parse_args()
    if args.k < 1:
        raise ValueError("--k must be at least 1.")
    if args.max_configs < 1:
        raise ValueError("--max_configs must be at least 1.")
    if args.pareto_top_n < 1:
        raise ValueError("--pareto-top-n must be at least 1.")

    snapshot_path = _resolve_snapshot_path(args.snapshot)
    outdir = _resolve_outdir(args.outdir)
    artifacts_dir = outdir / "artifacts"
    timestamp = datetime.now(tz=UTC)
    timestamp_label = timestamp.strftime("%Y%m%d_%H%M%S")

    snapshot_df = pd.read_parquet(snapshot_path)
    students = get_golden_students()
    active_win_model: Any | None = load_latest_model() if args.use_win_model else None
    active_win_model_path: Path | None = get_latest_model_path() if args.use_win_model else None
    relevance_config = RelevanceConfig(
        label_mode=args.label_mode,
        tfidf_threshold=args.tfidf_threshold,
        embed_threshold=args.embed_threshold,
    )
    profile_caches = _prepare_profile_caches(
        snapshot_df,
        students,
        similarity_mode=args.similarity_mode,
        model_name=args.model_name,
        use_win_model=args.use_win_model,
        win_model=active_win_model,
    )
    effective_relevance_config, labeling = _resolved_relevance_config(
        profile_caches,
        similarity_mode=args.similarity_mode,
        relevance_config=relevance_config,
        calibrate_thresholds=args.calibrate_thresholds,
    )
    configs = generate_candidate_configs(max_configs=args.max_configs)

    evaluation_results: list[dict[str, Any]] = []
    for config in configs:
        first_run = _run_config_once(
            profile_caches,
            config=config,
            k=args.k,
            use_win_model=args.use_win_model,
            similarity_mode=args.similarity_mode,
            relevance_config=effective_relevance_config,
        )
        second_run = _run_config_once(
            profile_caches,
            config=config,
            k=args.k,
            use_win_model=args.use_win_model,
            similarity_mode=args.similarity_mode,
            relevance_config=effective_relevance_config,
        )
        metrics = _metrics_summary(
            first_run=first_run,
            second_run=second_run,
            k=args.k,
            use_win_model=args.use_win_model,
        )
        evaluation_results.append(
            {
                **config.to_dict(),
                "metrics": metrics,
                "per_profile_topk_ids": first_run["ordered_ids"],
                "per_profile_metrics": first_run["per_profile_metrics"],
            }
        )

    baseline_result = evaluation_results[0]
    relevance_leaderboard = sorted(
        evaluation_results,
        key=lambda result: _relevance_sort_key(result, baseline_result["metrics"]),
    )
    relevance_best = relevance_leaderboard[0]

    blended_weights = _resolve_objective_weights(
        args.objective_weights,
        use_win_model=args.use_win_model,
        strict=args.selection_objective == "blended",
    )
    _with_blended_scores(evaluation_results, objective_weights=blended_weights)
    blended_leaderboard = sorted(evaluation_results, key=_blended_sort_key) if blended_weights is not None else []
    blended_best = blended_leaderboard[0] if blended_leaderboard else None

    pareto_metric_names = _resolve_pareto_metrics(
        args.pareto_metrics,
        use_win_model=args.use_win_model,
    )
    pareto_front = _pareto_front(evaluation_results, metric_names=pareto_metric_names)
    pareto_knee_best = _pareto_knee_point(pareto_front, metric_names=pareto_metric_names)

    if args.selection_objective == "relevance":
        selection_best_result = relevance_best
        report_leaderboard = relevance_leaderboard
    elif args.selection_objective == "blended":
        if blended_best is None:
            raise ValueError("Blended selection was requested, but blended scoring is unavailable.")
        selection_best_result = blended_best
        report_leaderboard = blended_leaderboard
    else:
        selection_best_result = pareto_knee_best
        report_leaderboard = relevance_leaderboard

    report_path = outdir / f"weight_tuning_{timestamp_label}.md"
    artifact_path = artifacts_dir / f"weight_tuning_{timestamp_label}.json"

    best_config_per_objective = {
        "relevance_best": _best_config_snapshot(relevance_best),
        "blended_best": _best_config_snapshot(blended_best),
        "pareto_knee_best": _best_config_snapshot(pareto_knee_best),
    }

    report_text = _report_markdown(
        timestamp_label=timestamp.isoformat(),
        snapshot_path=snapshot_path,
        snapshot_count=int(len(snapshot_df)),
        k=args.k,
        seed=args.seed,
        config_count=len(evaluation_results),
        similarity_mode=args.similarity_mode,
        model_name=args.model_name if args.similarity_mode == "embeddings" else None,
        use_win_model=args.use_win_model,
        win_model_path=active_win_model_path,
        baseline_result=baseline_result,
        selection_best_result=selection_best_result,
        relevance_best_result=relevance_best,
        best_config_per_objective=best_config_per_objective,
        leaderboard=report_leaderboard,
        selection_objective=args.selection_objective,
        objective_weights=blended_weights,
        pareto_metrics=pareto_metric_names,
        pareto_front=pareto_front,
        pareto_top_n=args.pareto_top_n,
        labeling=labeling,
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
        "use_win_model": bool(args.use_win_model),
        "win_model_path": str(active_win_model_path) if active_win_model_path is not None else None,
        "labeling": labeling,
        "objective_settings": {
            "selection_objective": args.selection_objective,
            "objective_weights": blended_weights,
            "pareto_metrics": pareto_metric_names,
            "pareto_top_n": int(args.pareto_top_n),
        },
        "baseline_metrics": baseline_result["metrics"],
        "best_config": {
            "config_id": selection_best_result["config_id"],
            "stage2_weights": selection_best_result["stage2_weights"],
            "stage3_weights": selection_best_result["stage3_weights"],
            "amount_utility_mode": selection_best_result["amount_utility_mode"],
            "use_win_model": bool(args.use_win_model),
            "similarity_mode": args.similarity_mode,
            "model_name": args.model_name if args.similarity_mode == "embeddings" else None,
            "metrics": selection_best_result["metrics"],
            "objective_score": selection_best_result.get("objective_score"),
            "pareto_knee_score": selection_best_result.get("pareto_knee_score"),
        },
        "best_config_per_objective": best_config_per_objective,
        "per_profile_topk_ids_best_config": selection_best_result["per_profile_topk_ids"],
        "per_profile_metrics_best_config": selection_best_result["per_profile_metrics"],
        "pareto_front": [_best_config_snapshot(result) for result in pareto_front],
        "all_configs": report_leaderboard,
    }
    write_json_atomic(artifact_payload, artifact_path)

    best_weights_path, latest_pointer_path = _write_best_weights_artifacts(
        processed_dir=PROCESSED_DIR,
        objective=args.selection_objective,
        best_result=selection_best_result,
        use_win_model=bool(args.use_win_model),
        similarity_mode=args.similarity_mode,
        model_name=args.model_name if args.similarity_mode == "embeddings" else None,
        labeling=labeling,
        snapshot_path=snapshot_path,
        timestamp=timestamp,
    )

    print(f"Report written to {report_path}")
    print(f"Artifact written to {artifact_path}")
    print(f"Best weights written to {best_weights_path}")
    print(f"Latest weights pointer written to {latest_pointer_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
