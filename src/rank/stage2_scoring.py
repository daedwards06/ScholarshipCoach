from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def _get_profile_value(profile: Any, key: str) -> Any:
    if isinstance(profile, Mapping):
        return profile.get(key)
    return getattr(profile, key, None)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        if pd.isna(value):
            return ""
        return str(value).strip()
    return str(value).strip()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, Iterable):
        values: list[str] = []
        for item in value:
            text = _normalize_text(item)
            if text:
                values.append(text)
        return values
    text = _normalize_text(value)
    return [text] if text else []


def _tokenize(value: Any) -> set[str]:
    normalized = _normalize_text(value).lower()
    return set(re.findall(r"[a-z0-9]+", normalized))


def build_student_profile_text(profile: Any) -> str:
    major = _normalize_text(_get_profile_value(profile, "major"))
    interests = _as_list(_get_profile_value(profile, "interests"))
    keywords = _as_list(_get_profile_value(profile, "keywords"))
    extracurriculars = _as_list(_get_profile_value(profile, "extracurriculars"))
    goals = _normalize_text(_get_profile_value(profile, "goals"))

    parts = [
        major,
        " ".join(interests),
        " ".join(keywords),
        " ".join(extracurriculars),
        goals,
    ]
    return " ".join(part for part in parts if part).strip()


def build_scholarship_text(row: pd.Series) -> str:
    fields = [
        _normalize_text(row.get("title")),
        _normalize_text(row.get("description")),
        _normalize_text(row.get("eligibility_text")),
        _normalize_text(row.get("essay_prompt")),
        _normalize_text(row.get("sponsor")),
    ]
    return " ".join(field for field in fields if field).strip()


def compute_tfidf_similarity(student_text: str, scholarship_texts: list[str]) -> np.ndarray:
    if not scholarship_texts:
        return np.array([], dtype=float)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    corpus = [student_text, *scholarship_texts]
    matrix = vectorizer.fit_transform(corpus)
    student_vector = matrix[0]
    scholarship_matrix = matrix[1:]

    similarities = (scholarship_matrix @ student_vector.T).toarray().ravel()
    return np.clip(similarities, 0.0, 1.0)


def _resolve_amount(row: pd.Series) -> float:
    amount_max = row.get("amount_max")
    if amount_max is not None and not pd.isna(amount_max):
        return float(amount_max)
    amount_min = row.get("amount_min")
    if amount_min is not None and not pd.isna(amount_min):
        return float(amount_min)
    return 0.0


def _compute_amount_utility(df: pd.DataFrame) -> np.ndarray:
    raw_amounts = np.array([_resolve_amount(row) for _, row in df.iterrows()], dtype=float)
    max_amount = float(raw_amounts.max()) if raw_amounts.size else 0.0
    if max_amount <= 0.0:
        return np.zeros(len(df), dtype=float)
    return np.clip(raw_amounts / max_amount, 0.0, 1.0)


def _profile_keyword_tokens(profile: Any) -> set[str]:
    keywords = _as_list(_get_profile_value(profile, "keywords"))
    interests = _as_list(_get_profile_value(profile, "interests"))
    return _tokenize(" ".join([*keywords, *interests]))


def _compute_keyword_overlap(df: pd.DataFrame, profile: Any) -> np.ndarray:
    profile_tokens = _profile_keyword_tokens(profile)
    if not profile_tokens:
        return np.zeros(len(df), dtype=float)

    overlaps: list[float] = []
    denominator = float(len(profile_tokens))
    for _, row in df.iterrows():
        scholarship_tokens = set()
        scholarship_tokens.update(_tokenize(row.get("title")))
        for keyword in _as_list(row.get("keywords")):
            scholarship_tokens.update(_tokenize(keyword))
        overlap_count = len(profile_tokens.intersection(scholarship_tokens))
        overlaps.append(overlap_count / denominator)
    return np.clip(np.array(overlaps, dtype=float), 0.0, 1.0)


def _compute_effort_penalty(df: pd.DataFrame) -> np.ndarray:
    penalties: list[float] = []
    for _, row in df.iterrows():
        penalties.append(1.0 if bool(row.get("essay_required")) else 0.0)
    return np.array(penalties, dtype=float)


def score_stage2(eligible_df: pd.DataFrame, profile: Any) -> pd.DataFrame:
    scored_df = eligible_df.copy()

    student_text = build_student_profile_text(profile)
    scholarship_texts = [build_scholarship_text(row) for _, row in scored_df.iterrows()]

    tfidf_sim = compute_tfidf_similarity(student_text=student_text, scholarship_texts=scholarship_texts)
    amount_utility = _compute_amount_utility(scored_df)
    keyword_overlap = _compute_keyword_overlap(scored_df, profile)
    effort_penalty = _compute_effort_penalty(scored_df)

    stage2_score = (
        0.70 * tfidf_sim
        + 0.20 * amount_utility
        + 0.10 * keyword_overlap
        - 0.10 * effort_penalty
    )

    scored_df["tfidf_sim"] = np.clip(tfidf_sim, 0.0, 1.0)
    scored_df["amount_utility"] = np.clip(amount_utility, 0.0, 1.0)
    scored_df["keyword_overlap"] = np.clip(keyword_overlap, 0.0, 1.0)
    scored_df["effort_penalty"] = np.clip(effort_penalty, 0.0, 1.0)
    scored_df["stage2_score"] = np.clip(stage2_score, 0.0, 1.0)

    return scored_df
