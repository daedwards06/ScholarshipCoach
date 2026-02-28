# Offline Evaluation

`scripts/evaluate_golden_students.py` evaluates the ranking pipeline on a saved snapshot parquet. It does not ingest live data.

## Inputs and Outputs

- Input snapshot:
  - `--snapshot <path>` to evaluate a specific saved snapshot
  - If omitted, the script loads the latest `data/processed/scholarships_snapshot_YYYYMMDD.parquet`
- Similarity mode:
  - `--similarity-mode tfidf|embeddings`
- Embedding model:
  - `--model-name all-MiniLM-L6-v2`
- Golden student set:
  - `src/eval/golden_students.py`
  - Includes diverse profiles across GPA, state, major, education level, and citizenship
- Weights:
  - `--use-best-weights` defaults to true when `data/processed/best_weights.json` exists
  - `--no-use-best-weights` forces baseline defaults
  - `--weights <path>` overrides with a specific weights file
- Outputs:
  - Markdown report: `reports/golden_eval_YYYYMMDD_HHMMSS.md`
  - Raw artifact JSON: `reports/artifacts/golden_eval_YYYYMMDD_HHMMSS.json`

## Evaluation Flow

For each golden profile, the script runs:

1. Stage 1 eligibility filter (`apply_eligibility_filter`)
2. Stage 2 scoring (`score_stage2`)
3. Stage 3 rerank (`rerank_stage3`)

Stage 2 can run in either:

- `tfidf` mode (default)
- `embeddings` mode using the local `all-MiniLM-L6-v2` sentence-transformer

Embedding mode uses a local cache at:

- `data/processed/embeddings/<model_name_sanitized>/embeddings.npz`

Saved snapshots keep only `embedding_key` values so the main parquet stays small; dense vectors live in the sidecar cache artifact.

Top-K is computed per profile as:

- `K = min(10, len(eligible_df))`
- If `eligible_df` is empty, the profile gets an empty recommendation list (no crash path).

## Metrics

- Eligibility precision:
  - `eligible_count / total_count`
  - Includes ineligible reason-code breakdown from `ineligible_df["reasons"]`
- Coverage@K:
  - unique scholarships recommended across all profiles at K
- Amount distribution (top-K lists):
  - mean/median/max of `amount_max`
- Ranking stability:
  - pipeline run twice on the same snapshot/profiles
  - asserts identical ordered `scholarship_id` lists for each profile
  - for embedding mode, this includes deterministic reuse of cached normalized embeddings from disk
- NDCG@K:
  - computed only when relevance labels are available
  - otherwise reported as `"N/A"`

## Proxy Relevance Labels (Offline Heuristic)

For portfolio-friendly offline evaluation, relevance labels are heuristic (not human annotations):

- Label `2` (high):
  - major/state/education match AND keyword overlap > 0
- Label `1` (medium):
  - keyword overlap > 0 OR text similarity above threshold
- Label `0` (low):
  - otherwise

This is an evaluation-only proxy. It is not a production relevance model.

## Limitations with Small Catalogs

When snapshots are small (for example 8 records):

- coverage and amount metrics can be noisy or compressed
- NDCG variation is limited by low candidate diversity
- many profiles may have small or zero eligible sets

The harness is intentionally robust for these cases and still produces deterministic artifacts. Metric interpretability improves as the catalog grows.
