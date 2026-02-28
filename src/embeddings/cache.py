from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.embeddings import model as embedding_model

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
EMBEDDING_TEXT_FIELDS = (
    "title",
    "sponsor",
    "description",
    "eligibility_text",
    "essay_prompt",
)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if pd.isna(value):
        return ""
    return str(value).strip()


def resolve_processed_dir(processed_dir: Path | None = None) -> Path:
    if processed_dir is None:
        return DEFAULT_PROCESSED_DIR
    return processed_dir if processed_dir.is_absolute() else ROOT_DIR / processed_dir


def sanitize_model_name(model_name: str) -> str:
    resolved_name = embedding_model.resolve_model_name(model_name)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", resolved_name).strip("_")


def build_embedding_text_fingerprint(record: Mapping[str, Any]) -> str:
    return " | ".join(_normalize_text(record.get(field)) for field in EMBEDDING_TEXT_FIELDS)


def build_embedding_text(record: Mapping[str, Any]) -> str:
    return " ".join(
        text
        for text in (_normalize_text(record.get(field)) for field in EMBEDDING_TEXT_FIELDS)
        if text
    ).strip()


def compute_embedding_key(record: Mapping[str, Any]) -> str:
    scholarship_id = _normalize_text(record.get("scholarship_id"))
    payload = f"{scholarship_id}|{build_embedding_text_fingerprint(record)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def ensure_embedding_keys(df: pd.DataFrame) -> pd.DataFrame:
    keyed_df = df.copy()
    keyed_df["embedding_key"] = [
        compute_embedding_key(row) for _, row in keyed_df.iterrows()
    ]
    return keyed_df


def embedding_store_path(model_name: str, *, processed_dir: Path | None = None) -> Path:
    resolved_dir = resolve_processed_dir(processed_dir)
    model_dir = resolved_dir / "embeddings" / sanitize_model_name(model_name)
    return model_dir / "embeddings.npz"


def load_embedding_store(
    model_name: str,
    *,
    processed_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    store_path = embedding_store_path(model_name, processed_dir=processed_dir)
    if not store_path.exists():
        return {}

    with np.load(store_path, allow_pickle=False) as payload:
        keys = payload["embedding_key"]
        vectors = payload["vectors"]

    return {
        str(embedding_key): np.asarray(vector, dtype=np.float32)
        for embedding_key, vector in zip(keys.tolist(), vectors, strict=True)
    }


def update_embedding_store(
    model_name: str,
    new_rows_df: pd.DataFrame,
    *,
    processed_dir: Path | None = None,
) -> None:
    if new_rows_df.empty:
        return

    required_columns = {"embedding_key", "scholarship_id", "vector"}
    missing_columns = required_columns.difference(new_rows_df.columns)
    if missing_columns:
        raise ValueError(
            "new_rows_df is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    store_path = embedding_store_path(model_name, processed_dir=processed_dir)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: dict[str, tuple[str, np.ndarray]] = {}
    if store_path.exists():
        with np.load(store_path, allow_pickle=False) as payload:
            for embedding_key, scholarship_id, vector in zip(
                payload["embedding_key"].tolist(),
                payload["scholarship_id"].tolist(),
                payload["vectors"],
                strict=True,
            ):
                existing_rows[str(embedding_key)] = (
                    str(scholarship_id),
                    np.asarray(vector, dtype=np.float32),
                )

    for _, row in new_rows_df.iterrows():
        existing_rows[str(row["embedding_key"])] = (
            _normalize_text(row["scholarship_id"]),
            np.asarray(row["vector"], dtype=np.float32),
        )

    ordered_keys = sorted(existing_rows)
    if not ordered_keys:
        return

    scholarship_ids = np.array(
        [existing_rows[key][0] for key in ordered_keys],
        dtype=str,
    )
    vectors = np.vstack([existing_rows[key][1] for key in ordered_keys]).astype(np.float32)
    temp_path = store_path.parent / f"{store_path.name}.tmp"
    with temp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            embedding_key=np.array(ordered_keys, dtype=str),
            scholarship_id=scholarship_ids,
            vectors=vectors,
        )
    temp_path.replace(store_path)


def ensure_embedding_store_for_df(
    df: pd.DataFrame,
    model_name: str,
    *,
    processed_dir: Path | None = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    if df.empty:
        keyed_df = df.copy()
        if "embedding_key" not in keyed_df.columns:
            keyed_df["embedding_key"] = pd.Series(dtype="object")
        return keyed_df

    keyed_df = ensure_embedding_keys(df)
    store = load_embedding_store(model_name, processed_dir=processed_dir)
    missing_mask = ~keyed_df["embedding_key"].isin(store)
    if not bool(missing_mask.any()):
        return keyed_df

    missing_df = keyed_df.loc[missing_mask].reset_index(drop=True)
    texts = [
        build_embedding_text(row)
        for _, row in missing_df.iterrows()
    ]
    vectors = embedding_model.embed_texts(
        texts,
        model_name=model_name,
        batch_size=batch_size,
    )
    new_rows_df = pd.DataFrame(
        {
            "embedding_key": missing_df["embedding_key"].tolist(),
            "scholarship_id": missing_df["scholarship_id"].astype(str).tolist(),
            "vector": [vector for vector in vectors],
        }
    )
    update_embedding_store(model_name, new_rows_df, processed_dir=processed_dir)
    return keyed_df
