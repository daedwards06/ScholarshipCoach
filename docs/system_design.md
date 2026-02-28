# System Design

ScholarshipCoach is a deterministic local ranking pipeline:

1. Ingest live scholarship sources into `data/raw/`
2. Normalize records into a stable dataframe schema
3. Build a saved snapshot parquet in `data/processed/`
4. Apply Stage 1 eligibility filters
5. Apply Stage 2 scoring
6. Apply Stage 3 reranking
7. Surface results in Streamlit and the offline evaluation harness

## Stage 2 Text Similarity

Stage 2 now supports two local-only text similarity modes:

- `tfidf`: the existing sparse `TfidfVectorizer` comparison
- `embeddings`: local sentence-transformer embeddings using `all-MiniLM-L6-v2`

Both modes use the same deterministic text inputs:

- Student text: major, interests, keywords, extracurriculars, goals
- Scholarship text: title, sponsor, description, eligibility text, essay prompt

The `Stage2Weights.tfidf` field name is retained for backward compatibility, but it now weights the active Stage 2 text similarity signal (`text_sim`) regardless of mode.

## Embedding Cache Artifact

Embedding mode is local-only and cached on disk for repeatable evaluation and tuning runs.

- Snapshot parquet stores only `embedding_key`
- Dense vectors live separately in:
  - `data/processed/embeddings/<model_name_sanitized>/embeddings.npz`

Each `embedding_key` is a SHA1 hash of:

- `scholarship_id`
- a stable text fingerprint built from title, sponsor, description, eligibility text, and essay prompt

If any of those text fields change, the key changes and the vector is recomputed.

## Reproducibility

- Snapshot rows stay deterministically sorted by `scholarship_id`
- Embedding store rows are written in sorted `embedding_key` order
- The sentence-transformer model is loaded once per process, kept in eval mode, run on CPU, and returns normalized vectors
- Re-running evaluation or tuning against the same saved snapshot and cached embeddings should produce the same ordering
