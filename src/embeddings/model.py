from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
else:
    SentenceTransformer = Any

DEFAULT_MODEL_NAME: Final[str] = "all-MiniLM-L6-v2"
MODEL_ALIASES: Final[dict[str, str]] = {
    DEFAULT_MODEL_NAME: "sentence-transformers/all-MiniLM-L6-v2",
}

_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def resolve_model_name(model_name: str) -> str:
    normalized = str(model_name or DEFAULT_MODEL_NAME).strip()
    if not normalized:
        normalized = DEFAULT_MODEL_NAME
    return MODEL_ALIASES.get(normalized, normalized)


def get_model(model_name: str) -> SentenceTransformer:
    try:
        from sentence_transformers import SentenceTransformer as SentenceTransformerImpl
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "sentence-transformers is required for embedding mode. "
            "Install project dependencies in .venv before building embedding caches."
        ) from exc

    resolved_name = resolve_model_name(model_name)
    cached = _MODEL_CACHE.get(resolved_name)
    if cached is not None:
        cached.eval()
        return cached

    model = SentenceTransformerImpl(resolved_name, device="cpu")
    model.eval()
    _MODEL_CACHE[resolved_name] = model
    return model


def embed_texts(texts: list[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    model = get_model(model_name)
    model.eval()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    return (array / norms).astype(np.float32, copy=False)
