# Offline Evaluation

`scripts/evaluate_golden_students.py` evaluates the ranking pipeline on a saved snapshot parquet. It does not ingest live data.

## Inputs and Outputs

- Input snapshot:
  - `--snapshot <path>` to evaluate a specific saved snapshot
  - If omitted, the script loads the latest `data/processed/scholarships_snapshot_YYYYMMDD.parquet`
- Similarity mode:
  - `--similarity-mode tfidf|embeddings`
- Label mode:
  - `--label-mode hybrid|no_similarity`
- Proxy relevance thresholds:
  - `--tfidf-threshold 0.12`
  - `--embed-threshold 0.30`
- Optional calibration:
  - `--calibrate-thresholds` reports a deterministic suggested threshold for the active similarity mode
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

Optional win-model flags:

- `--use-win-model` enables `p_win` and expected-value-aware Stage 3 reranking
- `--win-model-path <path>` uses a specific saved model; otherwise the latest saved model is used
- `--train-win-model` trains a fresh deterministic synthetic-label model on the selected snapshot before evaluation

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
  - explicit major match AND compatible state/education AND keyword overlap > 0
  - "explicit major match" means the scholarship lists the student's major; a
    scholarship open to all majors no longer earns label 2 on keyword overlap
    alone. Set `require_major_match_for_label2=False` (or the eval flag
    `--no-require-major-match-for-label2`) to restore the older behavior.
- Label `1` (medium), `hybrid` mode:
  - keyword overlap > 0 OR text similarity above the active mode-specific threshold
- Label `1` (medium), `no_similarity` mode:
  - keyword overlap > 0
- Label `0` (low):
  - otherwise

Mode-specific thresholds exist because embedding similarity scores are typically distributed differently than TF-IDF scores. Using separate defaults avoids inflating proxy relevance labels in embedding mode and keeps offline comparisons more interpretable.

`--calibrate-thresholds` is an evaluation-only reporting aid. It looks at eligible items with `keyword_overlap == 0`, computes a per-profile similarity cutoff that leaves roughly the top 25% labeled by similarity alone, and reports the median suggested threshold across profiles. The script does not auto-save or auto-apply that suggested value.

This is an evaluation-only proxy. It is not a production relevance model.

## Human-Labeled Evaluation

The proxy labels above share features with the ranker (keyword overlap, text similarity), so proxy NDCG is partly self-fulfilling. The strongest answer to that critique is a small set of hand-judged labels. The harness supports this end to end:

1. **Generate a worksheet.** Sample across a profile's *eligible* set (not just the top-K, to avoid ranking bias in the label set):

   ```powershell
   python scripts/make_labeling_worksheet.py --profile nc_cs_rising_sophomore --n 60
   ```

   This writes `data/eval/labeling_worksheet_nc_cs_rising_sophomore.csv` with `scholarship_id`, `title`, `sponsor`, `amount`, `deadline`, `source_url` (for when the snippet is too thin to judge from), a description snippet, and an empty `label` column.

   To label for the **real student** rather than a golden persona, use `--student` instead of `--profile`. It reads `data/private/students/<student_id>.json` and falls back to the committed demo profile with a printed warning when no private profile exists:

   ```powershell
   python scripts/make_labeling_worksheet.py --student --top-ranked --n 20
   ```

   `--top-ranked` takes the ranked top-N the student actually sees (Stage 1 → 2 → 3) instead of a random draw across the eligible set. Twenty ranked rows is a realistic ask of a real person, and it is the right sample for measuring *ordering*. It is the wrong sample for measuring *recall*: a top-N label set cannot reveal a good award the ranker buried. Eligible-set sampling therefore stays the default, and `--top-ranked` is the opt-in.

2. **Label by hand.** Fill the `label` column using the **same 0/1/2 rubric as the proxy**:
   - `2` (high): strong fit — major/field, level, and topic all clearly match.
   - `1` (medium): plausible fit — related field or general-eligibility award worth applying to.
   - `0` (low): poor fit — off-topic or only nominally eligible.

   Leave a row's `label` blank to exclude it; blank rows are dropped, not scored as 0.

3. **(Optional) Sanity-check your labels against the proxy.** The proxy heuristic and a human should broadly agree; a large gap is either a proxy blind spot worth documenting or a labeling slip worth fixing:

   ```powershell
   python scripts/check_label_agreement.py --human-labels data/eval/labeling_worksheet_nc_cs_rising_sophomore.csv
   ```

   It recomputes each scholarship's proxy label, prints per-row `|human - proxy|` deltas worst-first, and marks *sharp* disagreements (delta ≥ 2, i.e. a 0-vs-2 flip). It is a diagnostic, not a gate — the proxy is not ground truth — but `--fail-on-sharp` makes it exit non-zero for CI use.

4. **Re-run eval against the human labels.** The CSV needs `profile_id`, `scholarship_id`, and `label` columns (the worksheet already carries `profile_id`):

   ```powershell
   python scripts/evaluate_golden_students.py --k 10 --human-labels data/eval/human_labels_sample.csv
   ```

   The report adds a **Human-Labeled NDCG Check** section: NDCG@k is computed over only the labeled scholarships, in ranked order, and shown alongside the proxy NDCG. Because human labels do not share features with the ranker, this is a more defensible headline metric.

`data/eval/human_labels_sample.csv` is a small committed fixture that exercises this path; replace or extend it with a fully labeled worksheet to report a real human-judged NDCG.

### Human labels are pinned to the snapshot they were made on

Human labels join to the ranking on `scholarship_id`, so a label set is only valid against
catalogs that still contain those ids. Two things break that join: a change to canonical id
generation, and ordinary catalog turnover.

Both have happened. `data/eval/human_labels.csv` (44 pairs across 2 profiles) was labeled on the
**51-record 2026-06-27 snapshot**. Task 1.1 then re-keyed curated records to hash `catalog_id`
alone, and the catalog itself turned over when Bold.org was disabled and the Open Scholarships
feed was added. Overlap between that label set and the current snapshot is **zero**, and only 5
of the 44 pairs survive even by title match — far too few to carry an NDCG@10.

So the historical human NDCG is reported against its own snapshot, which reproduces exactly:

```powershell
python scripts/evaluate_golden_students.py --k 10 `
  --human-labels data/eval/human_labels.csv `
  --snapshot data/processed/scholarships_snapshot_20260627.parquet
```

Run against the current snapshot, the same command correctly reports `Human NDCG@K: N/A` and
`Profiles with labels: 0` rather than silently scoring a partial join. That is the intended
behavior: unlabeled rows are excluded, not zero-filled.

**The fix is re-labeling, not re-keying.** Rewriting the committed ids by title match would make
the number reappear without anyone having judged the current catalog. Generate a fresh worksheet
against the current snapshot and label it.

## Limitations with Small Catalogs

When snapshots are small (for example 8 records):

- coverage and amount metrics can be noisy or compressed
- NDCG variation is limited by low candidate diversity
- many profiles may have small or zero eligible sets

The harness is intentionally robust for these cases and still produces deterministic artifacts. Metric interpretability improves as the catalog grows.

## Synthetic Win Probability Notes

The optional win model uses synthetic labels generated from a transparent heuristic logit. This keeps the project local-only and reproducible while making `p_win` and `expected_value` visible in reports and the Streamlit UI.

- `p_win` is illustrative, not a validated outcome forecast
- `expected_value = p_win * amount_value`
- When enabled, Stage 3 uses normalized expected value as the `ev` signal
- Reports add top-K summaries for average and median `p_win` and expected value

## Real Outcomes

The synthetic labels above exist because the project had no real ones. Every application the
student actually submits produces one: won or lost, on a pair the ranker already scored. Those
are logged from the first application so the win model, or its replacement, can be judged
honestly later rather than retrofitted from memory.

### What is logged

The tracker writes an `outcomes` row when the student records a decision (`src/store/repo.py`,
`src/store/tracker.record_outcome`). `scripts/export_outcomes.py` joins it to the application and
the snapshot and writes `data/private/eval/outcomes.csv`:

```powershell
python scripts/export_outcomes.py
```

One row per **submitted** application -- saved and skipped awards are not trials and are not
exported. Each row carries:

- `student_id`, `catalog_id`, `title`
- `cycle_year` -- the award cycle, keyed on the deadline, so repeat applications to the same
  award in different years stay separate rows
- `submitted`, `submitted_on`, `status`
- `result` (`won` / `lost` / `pending`), `amount_awarded`, `decided_on`
- every column of `FEATURE_COLUMNS` (`src/win_model/features.py`), rebuilt **as of
  `submitted_on`** rather than today -- a deadline that has since passed was weeks away when the
  student applied, and using the present-day value would leak the outcome into the features
- `features_available` -- `False` when the award has dropped out of the snapshot. Those rows keep
  *empty* feature columns, never zeros: a zero-filled row is a wrong training example, not a
  missing one, and dropping it silently would hide how much of the history no longer joins.

The file is under `data/private/` and git-ignored, like the profile and the database.

### Minimum count before training on it

Do not train on this file early. `FEATURE_COLUMNS` has 13 features; the usual floor of roughly
ten decided events per feature puts a fitted win model at **130+ decided outcomes**, and the
minority class (`won`) has to carry its share of that. One student applying to twenty or thirty
awards a year will not reach it, and a model fit on twenty rows would be memorizing which essays
got written that spring.

So the file earns its keep in three stages, in order:

1. **Under ~30 decided outcomes -- description only.** Report the base rate (won / decided) and
   the realized total awarded. Nothing is fit. This is the state the project is in.
2. **From ~30 decided outcomes -- an evaluation set, not yet a training set.** Ask whether the
   ranker put won awards above lost ones: NDCG or simple AUC over the student's own submissions,
   with `result` as the label. That measures the ranking the family actually used, against
   outcomes it did not influence, and it is the first defensible answer to "does this thing work".
3. **At 130+ decided outcomes across students and cycles -- the earliest point a fitted model is
   defensible**, and even then it is reported against the synthetic baseline rather than instead
   of it.

### How it would replace synthetic labels

`src/win_model/` trains on a deterministic synthetic logit today. Real outcomes slot into the same
feature contract, so the swap is a label source change, not a rewrite:

- Feed rows with `features_available = True` and `result` in (`won`, `lost`) as `y`, and the
  `FEATURE_COLUMNS` values as `X`. Rows with `result = pending` are censored, not negative --
  exclude them.
- Hold out by `cycle_year`, not at random. Awards within one cycle share deadlines, essays and the
  student's effort that season, so a random split leaks.
- Keep the synthetic model as the baseline to beat. If the real-label model does not beat it on
  held-out cycles, the honest report is that there is not enough outcome data yet -- which is the
  expected answer for a long time.

Until then `p_win` stays what the note above says it is: illustrative, operator-mode only, and not
shown to the family.
