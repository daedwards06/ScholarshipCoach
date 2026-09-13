# System Design

ScholarshipCoach is a deterministic local ranking pipeline wrapped in a family-facing
application. Nothing in the ranking path touches the network, and nothing automated writes to
the catalog without a person confirming it.

1. Curate records into `data/catalog/records/`; ingest feeds and scrapes into `data/raw/`
2. Normalize records into a stable dataframe schema
3. Build a saved snapshot parquet in `data/processed/`
4. Apply Stage 1 eligibility filters
5. Bucket survivors onto the student's timeline
6. Apply Stage 2 scoring
7. Apply Stage 3 reranking
8. Surface results in Streamlit (student / parent / operator modes) and the offline
   evaluation harness, with application state in `data/private/coach.db`

## The Curated Catalog

The catalog is the project's primary data asset, not a scrape artifact. One JSON file per award
under `data/catalog/records/`, committed to git because award terms are public information.

`data/catalog/schema.json` is the contract; `python scripts/validate_catalog.py` enforces it in
CI. Required fields are `catalog_id`, `title`, `source_url`, `status`, `trust`, `provenance`.
Optional fields cover the eligibility axes Stage 1 reads: `education_level`, `grade_levels`,
`majors_allowed`, `states_allowed`, `counties_allowed`, `min_gpa`, `citizenship`, `need_based`,
`first_gen_only`, `gender`, `heritage`, `military_family`, `disability`, `religion`,
`employer_restricted`, `membership_required`, `min_test_scores`, plus `requirements`,
`renewal_terms`, `cycle` and `notes`.

Schema changes are additive. An older snapshot, an older golden profile, and an older test all
keep working after a new field is introduced.

### Stable IDs across cycles

A curated record's `scholarship_id` is `sha1("catalog|" + catalog_id)` — the slug alone. Deadline,
amount and title revisions between cycles do not change it, so a recurring award stays one row
that the tracker, outcomes log and re-verification can follow year to year. Scraped and feed
sources have no `catalog_id` and keep content hashing over title, sponsor, amounts, deadline and
source domain.

### Trust levels

`trust` records how much a record has been checked, and Stage 1 carries it through to the card:

- `verified_local` — hand-curated and confirmed by a person
- `structured_feed` — from a licensed structured API feed, trusted without per-record confirmation
- `aggregator` — scraped from a listing page; needs confirmation before it counts
- `unverified` — everything else

`provenance` travels with every record. The catalog schema requires `added_on` and `source_kind`
(`sponsor_site`, `school_counselor`, `community_foundation`, `employer`, `membership_org`,
`structured_feed`, `aggregator`, `other`) and carries `verified_on` / `verified_by` once a person
has checked it. Feed records additionally carry `license` and the `attribution` string that
license requires, so the credit travels with the row into every snapshot.

### Source configuration

Which connectors run is configuration, not code. `data/catalog/sources.json` holds an `enabled`
flag and a dated note per source, so disabling a broken connector is a one-line change with its
reason recorded next to it.

A connector that returns zero records where it previously returned some fails the ingest run
visibly rather than reporting `succeeded` — the failure mode that let the catalog collapse
unnoticed before 2026-09-12.

## The Inbox (Confirm Queue)

Automation proposes; a person confirms. Feeds, URL prefill and annual re-verification never write
to `data/catalog/records/`. They write proposals into `data/catalog/inbox/`, and a person accepts
or rejects each one from the parent-mode UI.

A proposal is a partial catalog record plus `proposal` metadata: its kind, the day it was created,
and a field-level diff against the record it would replace. `confirm()` is the only path from the
inbox into `records/`, and it validates against `data/catalog/schema.json` first — a proposal that
would write an invalid record stays in the queue instead of corrupting the catalog.

Proposal ids are deterministic, so re-running a feed or a re-verification pass replaces the pending
proposal for an award rather than stacking duplicates.

Prefill sits behind an `Extractor` protocol. Today the only implementation is deterministic regex
over fetched page text. A generative-LLM extractor plugs into the same seam without touching the
queue.

## Eligibility Timeline

Most awards repeat on the same calendar, so an award that is closed today is usually a target for
a later cycle rather than a dead link. After Stage 1, `src/rank/timeline.py` places each survivor
in one bucket:

| Bucket | Meaning |
|---|---|
| `now` | Open, deadline ahead, the student is eligible this cycle |
| `next_cycle` | Recurring; this cycle has passed but the next one is projected |
| `senior_year` | Gated on a grade level the student has not reached |
| `expired` | Non-recurring and past |
| `not_applicable` | Eligible on no cycle within the student's remaining school years |

For a recurring award whose listed deadline has passed, the module projects the next cycle's date
from the listed month and day. Bucketing is a pure function of catalog plus profile plus today.

## Persistence

Two stores, split by shape rather than by sensitivity — both are private, both are git-ignored
under `data/private/`.

**The profile is a JSON document** (`src/profile/store.py`), one file per student at
`data/private/students/<student_id>.json`, because it is one document per student with no growth.
`data/demo/student_demo.json` is a committed fictional profile, loaded with a visible banner when
no private profile exists, so a fresh clone runs.

**Everything relational is SQLite** (`src/store/db.py`) in `data/private/coach.db`: standard
library only, no server. Tables: `students`, `applications`, `checklist_items`, `essays`,
`essay_versions`, `essay_links`, `recommenders`, `recommendation_requests`, `outcomes`,
`colleges`, `settings`.

Schema changes are numbered SQL files under `src/store/migrations/`. Opening a connection applies
whichever migrations the file has not seen and records them in `schema_migrations`, so a database
created by an older build catches up on the next open and re-opening an up-to-date one is a no-op.

## Modes

Three views over one profile and one database. The family shares a household machine, so the gate
is a selector, not an account system.

| Mode | Surface |
|---|---|
| `student` | The daily view: what is due, what to write, the ranked list |
| `parent` | Adds catalog curation, the inbox, colleges and money, outcomes, settings; reads essays without editing them |
| `operator` | Pipeline controls — weight profiles, similarity mode, the win model. Hidden unless `operator_enabled` is set |

A PIN in git-ignored `.streamlit/secrets.toml` turns the parent selector into a soft lock. With no
secrets file there is no PIN and the selector is open, which is what a trusting household wants by
default. Operator mode stays off by default so the tuning surface never greets a student.

Every mode decision function is pure and unit-tested without Streamlit; only the render helpers
touch the page.

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
