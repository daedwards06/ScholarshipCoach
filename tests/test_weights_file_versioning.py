from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.tune_weights import _write_best_weights_artifacts


def test_write_best_weights_artifacts_versions_files_and_updates_pointer(tmp_path: Path) -> None:
    timestamp = datetime(2026, 3, 1, 4, 0, tzinfo=UTC)
    best_result = {
        "config_id": "cfg-1",
        "stage2_weights": {"tfidf": 0.7, "amount": 0.1, "keyword": 0.1, "effort": 0.1},
        "stage3_weights": {"stage2": 0.9, "urgency": 0.05, "ev": 0.05},
        "amount_utility_mode": "log",
        "metrics": {"ndcg_at_k": 0.8},
        "objective_score": 0.77,
        "pareto_knee_score": None,
    }
    labeling = {
        "label_mode": "hybrid",
        "tfidf_threshold": 0.12,
        "embed_threshold": 0.30,
        "active_threshold": 0.30,
        "calibration_enabled": False,
        "calibrated_threshold": None,
    }

    best_path, latest_path = _write_best_weights_artifacts(
        processed_dir=tmp_path,
        objective="blended",
        best_result=best_result,
        use_win_model=True,
        similarity_mode="embeddings",
        model_name="all-MiniLM-L6-v2",
        labeling=labeling,
        snapshot_path=tmp_path / "snapshot.parquet",
        timestamp=timestamp,
    )

    assert best_path == tmp_path / "best_weights_blended.json"
    assert latest_path == tmp_path / "best_weights_latest.json"

    best_payload = json.loads(best_path.read_text(encoding="utf-8"))
    latest_payload = json.loads(latest_path.read_text(encoding="utf-8"))

    assert best_payload["objective"] == "blended"
    assert best_payload["config_id"] == "cfg-1"
    assert latest_payload["objective"] == "blended"
    assert Path(latest_payload["path"]) == best_path
