<div align="center">

# ScholarshipCoach

### A Decision-Aware Scholarship Recommendation & Optimization System

[![CI](https://github.com/daedwards06/ScholarshipCoach/actions/workflows/ci.yml/badge.svg)](https://github.com/daedwards06/ScholarshipCoach/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-73%25-brightgreen)](https://github.com/daedwards06/ScholarshipCoach)
[![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

---

**ScholarshipCoach** is a multi-stage, end-to-end recommendation pipeline that ranks scholarships for a student profile using semantic similarity, calibrated win probability modeling, and multi-objective weight tuning. Built as a portfolio-grade applied ML project demonstrating recommender systems, ranking optimization, and probabilistic decision-making in a fully reproducible, snapshot-driven pipeline.

---

## Table of Contents

- [System Architecture](#-system-architecture)
- [Evaluation Results](#-evaluation-results)
- [Multi-Objective Tuning](#-multi-objective-tuning)
- [Win Probability Model](#-win-probability-model)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Design Principles](#-design-principles)
- [Future Work](#-future-work)
- [Author](#-author)

---

## System Architecture

The pipeline consists of four sequential stages:

```
Raw Sources
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 0 — Candidate Retrieval                           │
│  Source ingestion · Canonical ID generation              │
│  Snapshot versioning (Parquet) · Change tracking (JSON)  │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1 — Eligibility Filtering                         │
│  Major · State · Education level · Deadline enforcement  │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 2 — Semantic Scoring                              │
│  TF-IDF or SentenceTransformer (all-MiniLM-L6-v2)       │
│  Keyword overlap · Effort penalty · Award utility        │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 3 — Decision Reranking                            │
│  Weighted composite score · Deadline urgency boost       │
│  Optional win model: p(win) · Expected value = p × award │
└──────────────────────────────────────────────────────────┘
```

---

## Evaluation Results

Evaluation is fully **offline and snapshot-based**. Proxy relevance labels are heuristic and configurable (`hybrid` = keyword OR similarity threshold; `no_similarity` = structured + keyword only).

> **Note:** Metrics are computed on snapshot data and improve with catalog diversity. Results below are from a **163-record catalog** (March 2026 ingest). Baseline and Relevance-Optimized rows are historical reference points from a 160-record snapshot. Run `python scripts/evaluate_golden_students.py` to compute up-to-date metrics.

| Configuration | NDCG@10 | Coverage@10 | Notes |
|:---|---:|---:|:---|
| Baseline (Default Weights) | 0.29 | 0.21 | 160-record catalog, no win model |
| Relevance-Optimized (Grid Search, 150 configs) | 0.57 | 0.45 | 160-record catalog, win model |
| **Pareto-Selected (Relevance + Coverage + EV)** | **0.61** | **0.40** | 163-record catalog, win model |

> **Coverage@k here is a cross-profile _diversity_ ratio, not catalog coverage.** It is
> `unique recommended scholarships / total recommended slots` summed across all golden
> profiles — how distinct each profile's top-k list is, not what fraction of the catalog
> gets surfaced. A value near 1.0 means profiles are served largely different scholarships.

**Additional metrics — Pareto-Selected config (163-record catalog):**

| Metric | Value |
|:---|---:|
| Mean p(win) in Top-10 | ~0.52 |
| Eligibility Precision | 0.83 |
| Mean Expected Value in Top-10 | ~$9,640 |

All experiments are deterministic, snapshot-driven, versioned per objective, and reproducible via CLI flags.

**Human-labeled check** (2 hand-labeled profiles, 44 rows, 51-record June 2026 snapshot):

| Profile | Human NDCG@10 |
|:---|---:|
| `nc_cs_rising_sophomore` (CS) | 0.98 |
| `golden_tx_nursing_ug_us` (nursing) | 0.71 |
| **2-profile mean** | **0.85** |

Across both profiles: **29 / 44 (66%)** exact human↔proxy label agreement, **0** sharp
disagreements (0↔2).

Human NDCG is a **non-circular** measurement — labels are hand-judged, so they do not share features
with the ranker. Two honest signals fall out of it:

- The ranker orders scholarships **much better for the CS student (0.98) than the nursing student
  (0.71)** — this catalog and the weight tuning skew technical, so a nursing student is served less
  well. Single-profile (CS-only) reporting would have hidden this.
- The gap is **not** an artifact of labeling standard: under a strict reading (only `2`s count) it
  *widens* to 1.00 vs 0.63. The same scholarship flips across profiles as expected — the *Brave of
  Heart Nursing* award is a `0` for the CS student and a `2` for the nursing student.

These are *not* directly comparable to the proxy NDCG above (averaged across 9 golden profiles on a
larger catalog). Reproduce with `--human-labels`; see
[`docs/evaluation.md`](docs/evaluation.md#human-labeled-evaluation).

---

## Limitations & Evaluation Honesty

This is a portfolio project, and the headline metrics come with caveats worth stating plainly.
See [`docs/evaluation.md`](docs/evaluation.md) for the full methodology.

- **Proxy labels share features with the ranker.** Relevance labels are built from
  `keyword_overlap` and text similarity — the same signals Stage 2 scores on. Tuning weights
  to maximize NDCG against those labels is therefore partly self-fulfilling: the tuner can gain
  by upweighting the features the labels are made of. As a check, `--cross-label-check` scores
  the *same* ranking under both label heuristics (`hybrid` and `no_similarity`) and reports NDCG
  side by side; gains that survive the switch are more believable. A small **human-labeled eval set
  now exists** (hand-judged for a CS and a nursing profile, 44 pairs), scoring **mean NDCG@10 ≈ 0.85**
  (CS 0.98 / nursing 0.71) — a non-circular headline the proxy can't game.
- **Building the human set exposed a real proxy weakness.** The top label (2) was being awarded to
  scholarships that publish *no* major restriction — so a nursing award could rate as highly relevant
  for a CS student. The label-2 rule now requires an **explicit major match**
  (`require_major_match_for_label2`, default on), which removed every sharp human↔proxy disagreement.
  The trade-off: relevant-but-unrestricted awards (e.g. a general STEM scholarship) now cap at label 1,
  reflecting the proxy's dependence on structured major metadata. `scripts/check_label_agreement.py`
  flags the remaining human↔proxy gaps.
- **The win model is synthetic.** `p_win` and expected value are trained on labels from a
  transparent heuristic generator, not real award outcomes. They demonstrate a calibration/EV
  pipeline — they are not validated outcome forecasts.
- **The catalog is small.** With ~160 records, coverage and amount statistics are noisy and NDCG
  variation is limited by low candidate diversity. Metric interpretability improves as the catalog
  grows.

---

## Multi-Objective Tuning

Supports three optimization modes selectable at runtime:

| Objective | Description |
|:---|:---|
| `relevance` | Maximizes NDCG + Coverage |
| `blended` | Weighted sum of NDCG, Coverage, and Expected Value |
| `pareto` | Non-dominated front selection with knee-point picker |

Weight profiles are versioned and stored in `data/processed/`:

```
data/processed/
├── best_weights_relevance.json
├── best_weights_blended.json
├── best_weights_pareto.json
└── best_weights_latest.json
```

The Streamlit UI allows live switching between weight profiles.

---

## Win Probability Model

A **calibration / expected-value pipeline demonstration**, not an outcome forecast. The labels
come from a transparent logistic *generator* (`src/win_model/synthetic.py`), so the defensible
claim is not "this predicts who wins" but "this pipeline provably recovers its known generator."
A calibrated logistic regression (Platt scaling on top of a scaled `LogisticRegression`) is
trained on those synthetic labels, and every training run writes a **recovery check** to the
report proving the claim:

- **`p_true` recovery** — Pearson correlation and mean absolute error between predicted `p_win`
  and the generator's latent `p_true` on the held-out test split.
- **Coefficient recovery** — the learned logistic coefficients are compared to the generator
  coefficients; magnitudes differ (features are standardized) but the *signs* line up, so each
  learned effect points the same direction as the generator that produced the labels.

The Platt calibrator is intentionally near-redundant for today's linear base model; it is kept as
the seam where isotonic/Platt scaling becomes load-bearing once a non-linear base model or real
award outcomes replace the synthetic labels (see `docs/system_design.md`).

**Input features:**

| Feature | Type |
|:---|:---|
| Major / state / education level match | Binary |
| GPA above minimum threshold | Binary |
| Keyword overlap | Continuous |
| Semantic similarity score | Continuous |
| Days to deadline | Continuous |
| Award size (competition proxy) | Continuous |
| Essay requirement | Binary |

**Outputs per scholarship candidate:**

- `p_win` — calibrated win probability
- `expected_value` — p(win) × award amount
- `expected_value_norm` — normalized for cross-cohort comparison

The win model is **optional** and fully surfaced in the UI, so users can inspect and override it.

---

## Quick Start

### 1 — Set up environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### 2 — Ingest scholarships

```powershell
python scripts\run_ingest.py \
  --max-listing-pages 10 \
  --max-detail-pages 300 \
  --max-runtime-seconds 1800
```

### 3 — Evaluate ranking quality

```powershell
python scripts\evaluate_golden_students.py \
  --k 10 \
  --similarity-mode embeddings \
  --label-mode hybrid
```

### 4 — Tune weights

```powershell
python scripts\tune_weights.py \
  --k 10 \
  --similarity-mode embeddings \
  --use-win-model \
  --selection-objective pareto
```

### 5 — Launch the UI

```powershell
streamlit run app/main.py
```

**Deployment note:** The app theme is configured in `.streamlit/config.toml` with a dark palette optimized for readability. No additional setup is required — Streamlit will automatically apply the theme on launch.

---

## Project Structure

```
ScholarshipCoach/
├── src/
│   ├── ingest/          # Scrapers, canonical ID gen, snapshot versioning
│   ├── rank/            # Stages 1–3: eligibility, scoring, reranking
│   ├── eval/            # Offline evaluation harness (NDCG, coverage)
│   ├── embeddings/      # SentenceTransformer wrapper + caching
│   └── win_model/       # Logistic regression win probability model
├── scripts/
│   ├── run_ingest.py
│   ├── evaluate_golden_students.py
│   └── tune_weights.py
├── app/
│   └── main.py          # Streamlit dashboard
├── data/
│   ├── raw/             # Snapshot parquet files (git-ignored)
│   └── processed/       # Weight profiles, evaluation reports
├── tests/               # pytest suite
├── docs/plans/          # Implementation plans
└── pyproject.toml
```

---

## Design Principles

- **Deterministic ranking** — same inputs always produce the same output
- **Snapshot-only evaluation** — no live API calls during scoring; results are always reproducible
- **Versioned artifacts** — weight profiles and evaluation snapshots are tracked per experiment
- **Configurable objectives** — switch between relevance, blended, and Pareto modes via CLI flag or UI
- **Transparent modeling** — win model features and probabilities are fully exposed, not black-boxed
- **Modular pipeline** — each stage can be tested and swapped independently

---

## Future Work

- Additional scholarship connectors to increase catalog diversity
- Constrained portfolio optimization (knapsack-style multi-scholarship selection)
- Real-world outcome data integration to replace synthetic win labels
- Fairness analysis across demographic and geographic slices
- Production deployment (Docker + cloud hosting)

---

## Author

**Dominique Edwards**
Data Scientist · Decision Systems · Applied ML

[![GitHub](https://img.shields.io/badge/GitHub-daedwards06-181717?logo=github&logoColor=white)](https://github.com/daedwards06)
