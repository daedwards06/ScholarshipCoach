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

The `Stage2Weights.text_sim` field weights the active Stage 2 text similarity signal regardless of mode. JSON files written with the old `tfidf` key are still accepted by `from_mapping()` for backward compatibility.

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

## Optional Win Probability Model

Stage 3 now has an optional local-only win probability layer for portfolio demonstration.

- Training uses synthetic labels only; there are no real won/lost outcomes in this project
- Pairwise features are deterministic and built from profile fit, Stage 2 similarity, deadline timing, essay effort, and award size
- The model predicts `p_win` for each `(student, scholarship)` pair
- Expected value is `expected_value = p_win * amount_value`
- When the win model is enabled, the Stage 3 `ev` weight applies to `expected_value_norm`
- When the win model is disabled, the existing `ev_proxy_norm` path is unchanged

This model is illustrative. It should not be treated as a real outcome predictor or a guarantee of scholarship success.

### Framing: a calibration/recovery demonstration on a known generator

Because the labels come from a transparent logistic generator (`src/win_model/synthetic.py`, `GENERATOR_COEFFICIENTS` and `p_true`), the honest claim for this component is not "it forecasts award outcomes" but "it is a calibration/EV pipeline that provably recovers its known generator." The training report (`train_report_*.json`) carries a `recovery` section proving this:

- `recovery.p_true` — Pearson correlation and mean absolute error between the predicted `p_win` and the generator's latent `p_true` on the held-out test split.
- `recovery.coefficients` — per-feature comparison of the base model's learned logistic coefficients against the generator coefficients. Magnitudes differ (the base model is fit on standardised features) but the *sign* is directly comparable, so `direction_consistent` flags whether each learned effect points the same way as the generator. Features not used by the generator report `direction_consistent = null`.

### Why the Platt calibrator slot stays

The base model is a linear `LogisticRegression`, and the generator is linear, so an additional Platt (logistic) calibration step is mathematically near-redundant today — it cannot improve a base model that is already well-calibrated in-family. It is retained deliberately for **pipeline realism**: the calibrator slot is the seam where isotonic or Platt scaling becomes load-bearing the moment the base model is swapped for a non-linear estimator (e.g. gradient-boosted trees) or the synthetic labels are replaced with real award outcomes. Keeping the slot wired and exercised in every training run means that swap is a one-line change, not a pipeline redesign. This is a conscious choice over swapping to a non-linear base model now, which would trade a clean linear recovery proof for calibration that "visibly does something" but no longer demonstrates exact generator recovery.
