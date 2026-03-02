📚 Scholarship Coach

A Decision-Aware Scholarship Recommendation & Optimization System

🚀 Overview

Scholarship Coach is a multi-stage, decision-aware recommendation system that ranks scholarships for a student profile using:

Deterministic ingestion + snapshot versioning

Semantic ranking (TF-IDF + transformer embeddings)

Calibrated synthetic win probability modeling

Multi-objective weight tuning (Relevance, Blended, Pareto)

Expected value optimization (p(win) × award)

Snapshot-based offline evaluation harness

This project demonstrates applied recommender systems, ranking optimization, probabilistic modeling, and multi-objective selection in a fully reproducible pipeline.

🧠 System Architecture

Stage 0 — Candidate Retrieval

Source ingestion with runtime caps + caching

Canonical ID generation

Snapshot versioning (parquet)

Change tracking (json deltas)

Stage 1 — Eligibility Filtering

Major match

State match

Education level match

Deadline enforcement

Stage 2 — Semantic Scoring

TF-IDF similarity OR

SentenceTransformer embeddings (all-MiniLM-L6-v2)

Keyword overlap

Effort penalty

Log-normalized amount utility

Stage 3 — Decision Reranking

Weighted Stage 2 score

Deadline urgency boost

Optional win model:

p(win) (calibrated logistic regression)

Expected value = p(win) × award

Deterministic tie-breaking

📊 Evaluation Framework

Evaluation is fully offline and snapshot-based.

Proxy relevance labels are heuristic and configurable:

hybrid (keyword OR similarity threshold)

no_similarity (structured + keyword only)

> **Note:** Metrics are offline proxies computed on snapshot data. They improve with catalog
> diversity — results below are from a **163-record catalog** (March 2026 ingest).
> Baseline and Relevance-Optimized rows are historical reference points from a 160-record
> snapshot; Pareto-Selected reflects the current catalog. Run
> `python scripts/evaluate_golden_students.py` to compute up-to-date metrics.

| Configuration | NDCG@10 | Coverage@10 | Notes |
|---|---:|---:|---|
| Baseline (Default Weights) | 0.29 | 0.21 | 160-record catalog, no win model |
| Relevance-Optimized (Grid Search, 150 configs) | 0.57 | 0.45 | 160-record catalog, win model |
| Pareto-Selected (Relevance + Coverage + EV) | **0.61** | **0.40** | 163-record catalog, win model |

Additional metrics for current Pareto-Selected config (163-record catalog):

Mean p(win) in Top-10: ~0.52

Eligibility Precision: 0.83

Mean Expected Value in Top-10: ~$9,640

All experiments are:

Deterministic

Snapshot-driven

Versioned per objective

Reproducible via CLI flags

🏗 Multi-Objective Tuning

Supports three optimization modes:

Objective	Description
relevance	Maximizes NDCG + Coverage
blended	Weighted sum of NDCG, Coverage, EV
pareto	Non-dominated front + knee selection

Weight profiles are versioned:

data/processed/
  best_weights_relevance.json
  best_weights_blended.json
  best_weights_pareto.json
  best_weights_latest.json

The Streamlit UI allows switching between weight profiles.

🎯 Win Probability Model

A calibrated logistic regression model trained on synthetic labels:

Features include:

Major / state / education match

GPA above minimum

Keyword overlap

Semantic similarity

Days to deadline

Award size (competition proxy)

Essay requirement

Outputs:

p_win

expected_value

expected_value_norm

The win model is optional and fully transparent in UI.

💻 Running the Project
1️⃣ Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
2️⃣ Ingest scholarships
python scripts\run_ingest.py --max-listing-pages 10 --max-detail-pages 300 --max-runtime-seconds 1800
3️⃣ Evaluate ranking
python scripts\evaluate_golden_students.py --k 10 --similarity-mode embeddings --label-mode hybrid
4️⃣ Tune weights
python scripts\tune_weights.py --k 10 --similarity-mode embeddings --use-win-model --selection-objective pareto
5️⃣ Launch UI
streamlit run app/main.py
🔬 Design Principles

Deterministic ranking

Snapshot-only evaluation (no live API calls during scoring)

Versioned artifacts

Configurable objectives

Transparent modeling

Modular pipeline design

📂 Project Structure (High-Level)
src/
  ingest/
  rank/
  eval/
  embeddings/
  win_model/
scripts/
  run_ingest.py
  evaluate_golden_students.py
  tune_weights.py
app/
  main.py
data/
  raw/
  processed/
reports/
  weight_tuning/
  golden_eval/
📌 Future Improvements

Additional scholarship connectors (increase diversity)

Constrained portfolio optimization (knapsack-style selection)

Real-world outcome data integration

Fairness analysis across demographic slices

Production deployment (Docker + cloud)

👤 Author

Dominique Edwards
Data Scientist | Decision Systems | Applied ML
