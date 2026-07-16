# Evaluation Reports (curated)

The pipeline writes a fresh timestamped report on every eval and tuning run. Those
run-specific files are **git-ignored** (see the `reports/` block in [.gitignore](../.gitignore)),
so this directory keeps only a small, hand-picked set that illustrates how the system is
measured and how it improved over time. Regenerate any of these locally with the commands in
[docs/evaluation.md](../docs/evaluation.md).

| File | What it shows |
|------|---------------|
| [`golden_eval_20260222_054100.md`](golden_eval_20260222_054100.md) | **Early baseline.** The first golden-student run on an 8-record scaffold snapshot — eligibility precision 1.00 but NDCG@8 = 0.0, i.e. the ranker had almost nothing to order yet. Kept as the "before" reference point. |
| [`golden_eval_20260716_014219.md`](golden_eval_20260716_014219.md) | **Current golden-student eval.** 51-record snapshot, embeddings similarity, `hybrid` labels: NDCG@10 = 0.72, Coverage@10 = 0.23, eligibility precision 0.45. The headline offline metric the README cites. |
| [`weight_tuning/weight_tuning_20260629_235450.md`](weight_tuning/weight_tuning_20260629_235450.md) | **Weight tuning / grid search.** 200 Stage-2/Stage-3 weight configs evaluated with the win model, Pareto-selected on `ndcg,coverage,ev`. Baseline config NDCG@10 = 0.63 vs. the tuned Pareto frontier. |

Each markdown report has a machine-readable twin under [`artifacts/`](artifacts/) (and
[`weight_tuning/artifacts/`](weight_tuning/artifacts/)) — the raw JSON the report is rendered
from, one per kept report.
