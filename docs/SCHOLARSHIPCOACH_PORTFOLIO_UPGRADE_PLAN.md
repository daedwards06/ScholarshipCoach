# ScholarshipCoach Portfolio Upgrade Plan â€” Tier 1, 2 & 3

> Generated: 2026-03-01 | Based on full project audit (B+ â†’ A roadmap)
> Target executor: Claude Sonnet 4.6 via Copilot agent mode
> Estimated total effort: 4 phases, ~10-12 focused sessions
> Prerequisite: 45/45 tests passing, conda environment active, GitHub remote configured

---

## Table of Contents

1. [Phase 0: Quick-Wins & Hygiene (Tier 1)](#phase-0-quick-wins--hygiene-tier-1)
2. [Phase 1: Code Quality & Test Infrastructure (Tier 2)](#phase-1-code-quality--test-infrastructure-tier-2)
3. [Phase 2: Data Pipeline & Ranking Science (Tier 3)](#phase-2-data-pipeline--ranking-science-tier-3)
4. [Phase 3: Portfolio Showpieces (Tier 3)](#phase-3-portfolio-showpieces-tier-3)

---

## Phase 0: Quick-Wins & Hygiene (Tier 1)

**Goal:** Eliminate the most visible gaps â€” stale data, missing repo hygiene files, trivial code smells â€” so the project passes a 30-second recruiter scan.

### Task 0.1: Run a Fresh Full Ingest

**Why:** The current snapshot has only 28 scholarships (down from 160) due to a Feb 28 ingest that dropped 139 records. All evaluation metrics are near-meaningless at this catalog size. A fresh, full ingest restores the candidate universe and makes every downstream metric credible.

**Checklist:**
- [x] Run full ingest with generous runtime/page caps
- [x] Verify snapshot record count is >= 100
- [x] Inspect `changes_YYYYMMDD.json` delta for sanity (no massive drops)
- [x] Verify embedding cache updates for new records
- [x] Run golden eval against the new snapshot to re-baseline metrics

**Results (2026-03-02):**
- Snapshot: `scholarships_snapshot_20260302.parquet` — **163 records** (up from 28)
- Delta: `changes_20260302.json` — added=135, removed=0, changed=0 ✓
- Embedding cache: `embeddings.npz` updated to 290 KB ✓
- Eval report: `reports/golden_eval_20260302_031321.md`
  - NDCG@10: **0.574**, Coverage@10: **0.4375**, Eligibility precision: **0.834**
  - Win model trained: `win_model_20260302_031250.joblib`

**Command:**

```powershell
python scripts\run_ingest.py --max-listing-pages 20 --max-detail-pages 500 --max-runtime-seconds 3600
python scripts\evaluate_golden_students.py --k 10 --similarity-mode embeddings --label-mode hybrid --use-win-model --train-win-model
```

---

### Task 0.2: Add LICENSE and Repo Hygiene Files

**Why:** A GitHub repo without a LICENSE is legally "all rights reserved" â€” it signals to reviewers that the author doesn't understand open-source norms. Missing these files is the fastest way to look unpolished.

**Checklist:**
- [x] Create `LICENSE` (MIT License, author: Dominique Edwards)
- [x] Update `.gitignore` to include `data/raw/`, `*.parquet`, win model artifacts, `.conda/`
- [x] Verify `data/raw/` and large binary artifacts are NOT committed to git history

**Prompt for Claude Sonnet 4.6:**

```
Create a standard MIT LICENSE file for this project.

CONTEXT:
- Author: Dominique Edwards
- Year: 2026
- Project: ScholarshipCoach

Also update .gitignore to add:
- data/raw/          (cached HTTP responses â€” don't commit)
- *.parquet          (snapshots are build artifacts)
- data/processed/win_model/   (trained model binaries)
- data/processed/embeddings/  (embedding cache)
- .conda/            (conda environment)
- student_profile.json (personal data)
- reports/ingest_runs/ (run-specific logs)

Do NOT add these to .gitignore (they should be committed):
- data/processed/best_weights*.json  (versioned tuning results)
- data/processed/changes_*.json      (delta reports)
```

---

### Task 0.3: Remove `sys.path` Hacks from All Scripts

**Why:** Every script (`run_ingest.py`, `evaluate_golden_students.py`, `tune_weights.py`, `build_snapshot.py`, `app/main.py`) starts with `sys.path.insert(0, ROOT_DIR)`. This is unnecessary because `pip install -e .` is already configured via `pyproject.toml`. The hack triggers 34 of the 37 ruff lint errors (`E402 module-import-not-at-top-of-file`). Removing it cleans up the lint report and signals proper Python packaging knowledge.

**Checklist:**
- [x] Remove `sys.path.insert(...)` blocks from all 5 files
- [x] Move imports to the top of each file (resolving E402 errors)
- [x] Fix the 2 unused imports and 1 shadowed variable flagged by ruff
- [x] Verify `pip install -e .` is active in the environment
- [x] Run `ruff check src/ scripts/ app/ tests/` â€” target 0 errors
- [x] Run `pytest tests/ -q` â€” all 45 tests still pass

**Prompt for Claude Sonnet 4.6:**

```
Remove all sys.path manipulation from the project and fix resulting lint errors.

CONTEXT:
- 5 files have sys.path.insert(0, ROOT_DIR) patterns:
  * scripts/run_ingest.py
  * scripts/evaluate_golden_students.py
  * scripts/tune_weights.py
  * scripts/build_snapshot.py
  * app/main.py
- The project is already pip-installable via pyproject.toml with `pip install -e .`
- tests/conftest.py also has sys.path manipulation
- ruff reports 34 E402 errors (imports after sys.path), 2 F401 (unused imports), 1 F402 (shadowed)

REQUIREMENTS:
1. Remove the sys.path.insert() block from each file (usually lines 1-15).
2. Move all imports to the top of each file in standard order: stdlib â†’ third-party â†’ local.
3. Update import paths if needed â€” the package root is `src/`, so imports like 
   `from src.rank.stage1_eligibility import ...` should work when installed editably.
4. Fix the 2 unused imports (identify and remove or use).
5. Fix the 1 F402 shadowed variable.
6. Update tests/conftest.py to remove its sys.path hack.
7. Run: ruff check src/ scripts/ app/ tests/ â†’ 0 errors
8. Run: pytest tests/ -q â†’ 45 passed
```

---

### Task 0.4: Add Your Student as a Golden Profile

**Why:** The 8 golden profiles are synthetic test personas â€” none represent your actual student. Adding your real student's profile makes the evaluation personally meaningful and ensures the ranking pipeline is tested against the use case you actually care about.

**Checklist:**
- [x] Add a 9th `GoldenStudent` entry to `src/eval/golden_students.py`
- [x] Profile: NC, Computer Science / Computer Engineering, GPA 3.0â€“3.5, 9thâ€“10th grade
- [x] Include realistic interests, keywords, extracurriculars, and goals
- [x] Run golden eval and verify the new profile produces reasonable recommendations
- [x] Review the top-10 scholarships returned â€” do they make sense for your student?

**Prompt for Claude Sonnet 4.6:**

```
Add a real student golden profile to the evaluation harness.

CONTEXT:
- File: src/eval/golden_students.py
- Contains 8 GoldenStudent entries (frozen dataclasses) returned by get_golden_students()
- Each profile has: student_id, description, profile (StudentProfile), interests, keywords,
  extracurriculars, goals
- StudentProfile fields: gpa, state, major, education_level, citizenship, today
- All existing profiles use today=date(2026, 2, 22)

REQUIREMENTS:
1. Add a 9th GoldenStudent with these characteristics:
   - student_id: "nc_cs_rising_sophomore"
   - description: "NC high school student interested in CS/CE, rising sophomore, 3.25 GPA"
   - State: "NC" (North Carolina)
   - Major: "Computer Science" (also relevant: Computer Engineering)
   - GPA: 3.25
   - Education level: "high school"
   - Citizenship: "US"
   - today: date(2026, 2, 22)  (match the other profiles)
   - Interests: ("programming", "robotics", "game development", "cybersecurity", "math")
   - Keywords: ("STEM", "computer science", "engineering", "technology", "coding", "software")
   - Extracurriculars: ("robotics club", "math team", "coding bootcamp", "volunteer tutoring")
   - Goals: "Pursuing a degree in Computer Science or Computer Engineering with 
     interest in software development and cybersecurity"

2. Place the new entry at the END of the list in get_golden_students().
3. Match the exact frozen dataclass pattern used by existing entries.
4. Run: pytest tests/ -q â†’ all tests pass
5. Run eval: python scripts\evaluate_golden_students.py --k 10 --similarity-mode embeddings --label-mode hybrid --use-win-model
```

---

### Task 0.5: Update README Metrics to Reflect Reality

**Why:** The README currently shows NDCG@10: 0.57, Coverage@10: 0.45 â€” but the latest actual eval shows 0.39 and 0.20 respectively (on a 28-record snapshot). Stale metrics undermine credibility. Either update to current values or clearly label them with dataset context.

**Checklist:**
- [x] Update the Evaluation Framework section in README.md
- [x] Add dataset-size context (e.g., "160-record catalog" vs "28-record catalog")
- [x] After Task 0.1 (fresh ingest), re-run eval and update metrics to the fresh numbers
- [x] Add a note that metrics are snapshot-dependent and improve with catalog size

**Prompt for Claude Sonnet 4.6:**

```
Update the README.md evaluation metrics to be accurate and honest.

CONTEXT:
- README currently claims:
  * Baseline NDCG@10: 0.29, Coverage@10: 0.21
  * Relevance-Optimized NDCG@10: 0.57, Coverage@10: 0.45
  * Pareto-Selected NDCG@10: 0.56, Coverage@10: 0.51
- These numbers were from the 160-record snapshot (Feb 22)
- Latest eval on the 28-record snapshot shows NDCG@10: 0.39, Coverage@10: 0.20
- After Task 0.1 fresh ingest, new eval numbers will be available

REQUIREMENTS:
1. Read the latest eval report in reports/ for current metrics.
2. Update the metrics table in README.md with accurate numbers.
3. Add context about catalog size affecting metrics:
   "Metrics are offline proxies computed on snapshot data. They improve with catalog 
    diversity â€” results below are from a [N]-record catalog."
4. If fresh ingest from Task 0.1 is done, use those numbers. Otherwise use a placeholder
   format like "Run `evaluate_golden_students.py` to compute current metrics."
5. Keep the three-tier table (Baseline, Relevance-Optimized, Pareto-Selected) but make
   the numbers accurate.
```

---

## Phase 1: Code Quality & Test Infrastructure (Tier 2)

**Goal:** Eliminate the top code smells (duplicate `_normalize_text`, missing docstrings, weak typing), strengthen the test suite, and add CI â€” the marks of professional engineering.

### Task 1.1: Extract Shared `_normalize_text` into a Common Module

**Why:** `_normalize_text()` is independently defined in **7 files** with slightly different behavior. This is the single biggest DRY violation in the codebase. A portfolio reviewer scanning the project with `grep` will see this immediately. Consolidating it signals that you understand code reuse and have actively maintained the codebase.

**Checklist:**
- [ ] Create `src/text_utils.py` with a canonical `normalize_text()` function
- [ ] Audit all 7 implementations to reconcile behavioral differences
- [ ] Replace all 7 private copies with imports from `src/text_utils.py`
- [ ] Add unit tests in `tests/test_text_utils.py` covering None, empty, whitespace, casing
- [ ] Run full test suite â€” all tests pass
- [ ] Run ruff â€” 0 errors

**Prompt for Claude Sonnet 4.6:**

```
Extract the duplicated _normalize_text function into a shared module.

CONTEXT:
- _normalize_text() is independently defined in 7 files:
  1. src/rank/stage1_eligibility.py  â€” lowercases, strips, handles None
  2. src/rank/stage2_scoring.py      â€” lowercases, strips, handles None  
  3. src/eval/relevance.py           â€” lowercases, strips, handles None
  4. src/normalize/canonical_id.py   â€” lowercases, strips, handles None
  5. src/embeddings/cache.py         â€” lowercases, strips, handles None
  6. src/win_model/features.py       â€” lowercases, strips, handles None
  7. src/rank/stage3_rerank.py       â€” via _resolve_deadline (implicit normalization)

- Each variant has slightly different behavior:
  * Some return None for non-string input, others return ""
  * Some strip only whitespace, others also collapse internal whitespace
  * canonical_id.py does lowercasing for hash stability
  * stage2_scoring.py has a separate _normalize_text just for text concatenation

REQUIREMENTS:
1. Read ALL 7 implementations carefully to understand the differences.

2. Create src/text_utils.py with:
   def normalize_text(value: Any, *, default: str = "") -> str:
       """Normalize a text value: coerce to string, lowercase, strip whitespace.
       
       Returns `default` for None or non-string inputs.
       """
   
   def normalize_list(value: Any) -> list[str]:
       """Normalize a list-like value of strings: coerce, lowercase, strip each element."""

3. Replace all 7 private _normalize_text with:
   from src.text_utils import normalize_text

4. Keep _normalize_list consolidation in the same module (also duplicated in 
   stage1_eligibility.py and relevance.py).

5. Create tests/test_text_utils.py:
   - test_normalize_none_returns_default
   - test_normalize_empty_string
   - test_normalize_strips_and_lowercases
   - test_normalize_non_string_types (int, float, list)
   - test_normalize_list_with_various_inputs
   - test_normalize_list_with_none

6. Run: pytest tests/ -q â†’ all tests pass (45 + new tests)
7. Run: ruff check src/ â†’ 0 errors
```

---

### Task 1.2: Add Docstrings to All Public Functions

**Why:** Across 20 source modules and ~80 public functions, only 4 files have any docstrings at all. This is the single biggest gap for portfolio presentation. Recruiters and interviewers will open a random source file â€” the first thing they look for is a docstring explaining what the function does. Missing docstrings signal "this was autocompleted, not authored."

**Checklist:**
- [ ] Add one-liner or multi-line docstrings to all public functions and classes in `src/`
- [ ] Add module-level docstrings to key modules (`stage1_eligibility.py`, `stage2_scoring.py`, etc.)
- [ ] Add class-level docstrings to all dataclasses
- [ ] Follow Google-style docstring format (Args/Returns/Raises sections where appropriate)
- [ ] Do NOT add docstrings to private helper functions (prefixed with `_`) unless complex
- [ ] Run tests to ensure nothing broke (docstrings are code too)

**Prompt for Claude Sonnet 4.6:**

```
Add docstrings to all public functions, classes, and key modules in src/.

CONTEXT:
- 20 source files in src/ with ~80 public functions and ~10 classes
- Only 4 files currently have any docstrings (weights.py, canonical_id.py, schema.py, base.py)
- All type hints are already present â€” docstrings should complement, not duplicate them
- This is a portfolio project â€” docstrings are read by recruiters/interviewers

REQUIREMENTS:
1. Read every file in src/ and identify all public functions (no _ prefix) and classes.

2. For each public function, add a Google-style docstring:
   def apply_eligibility_filter(df: pd.DataFrame, profile: StudentProfile) -> tuple[...]:
       """Split scholarships into eligible and ineligible sets based on hard filters.
       
       Applies deadline, GPA, state, major, education level, and citizenship checks.
       Each ineligible scholarship is tagged with all applicable reason codes.
       
       Args:
           df: Scholarship DataFrame with normalized columns.
           profile: Student profile with eligibility attributes.
       
       Returns:
           Tuple of (eligible_df, ineligible_df). The ineligible frame includes
           a 'reasons' column with lists of reason codes like 'GPA_BELOW_MIN'.
       """

3. For each class, add a class-level docstring:
   @dataclass(frozen=True, slots=True)
   class Stage2Weights:
       """Immutable weight configuration for Stage 2 scoring components.
       
       All weights must be in [0, 1]. The 'tfidf' weight applies to the active
       text similarity signal regardless of whether TF-IDF or embeddings mode is used.
       """

4. Add module-level docstrings to these key modules:
   - src/rank/stage1_eligibility.py: "Hard eligibility filtering with reason codes."
   - src/rank/stage2_scoring.py: "Composite scoring with text similarity, amount utility, and keyword overlap."
   - src/rank/stage3_rerank.py: "Decision-aware reranking with urgency and expected value signals."
   - src/io/snapshotting.py: "Atomic snapshot and delta artifact management."
   - src/eval/metrics.py: "Offline evaluation metrics for ranking quality."

5. Keep docstrings concise â€” one-liners for simple helpers, multi-line for complex logic.

6. Do NOT add docstrings to private functions (prefixed with _) unless the logic is 
   non-obvious (e.g., _compute_urgency_boost with its exponential decay formula).

7. Run: pytest tests/ -q â†’ all tests pass
```

---

### Task 1.3: Strengthen Type Safety for `profile` Objects

**Why:** The `profile` parameter is passed through Stage 1, Stage 2, Stage 3, features, and relevance modules â€” but it's typed as `Any` everywhere. Functions use duck typing (`getattr`, `isinstance(dict)`) to extract fields. A `Protocol` or shared type alias would catch bugs at edit-time and signal type-discipline to reviewers.

**Checklist:**
- [ ] Define a `ProfileLike` Protocol in `src/rank/weights.py` or a new `src/types.py`
- [ ] Update Stage 2, Stage 3, win_model/features, and eval/relevance to use `ProfileLike`
- [ ] Ensure both `StudentProfile` (dataclass) and `dict` satisfy the Protocol
- [ ] Run mypy or pyright on the rank modules to verify type consistency
- [ ] All tests pass

**Prompt for Claude Sonnet 4.6:**

```
Improve type safety for the profile parameter passed throughout the pipeline.

CONTEXT:
- StudentProfile is a dataclass in src/rank/stage1_eligibility.py with fields:
  gpa, state, major, education_level, citizenship, today (all Optional)
- GoldenStudent.as_stage2_profile() returns a dict with additional fields:
  interests, keywords, extracurriculars, goals (plus the StudentProfile fields)
- Multiple modules accept profile as Any and use duck typing to read fields:
  * stage2_scoring.py: _get_profile_value(profile, field) checks dict/getattr
  * stage3_rerank.py: rerank_stage3(profile=...) typed as Any
  * win_model/features.py: build_pair_features(profile, ...) typed as Any
  * eval/relevance.py: proxy_relevance_label(student=...) uses dict-style access

REQUIREMENTS:
1. Create src/types.py with:
   from typing import Protocol, runtime_checkable
   
   @runtime_checkable
   class ProfileLike(Protocol):
       """Protocol for objects that can serve as a student profile.
       
       Satisfied by both StudentProfile (dataclass) and dict-based profiles
       from GoldenStudent.as_stage2_profile().
       """
       # Define the minimal interface that all callers need

2. The tricky part: some callers read fields via getattr (dataclass), others via 
   dict key access. The Protocol should document this clearly. Options:
   a. Require all callers to use a helper function (already exists: _get_profile_value)
   b. Make the Protocol define __getitem__ for dict-like access
   c. Keep the Protocol loose but document the contract

3. Update function signatures in:
   - src/rank/stage2_scoring.py: score_stage2(profile: ProfileLike | dict, ...)
   - src/rank/stage3_rerank.py: rerank_stage3(profile: ProfileLike | dict, ...)
   - src/win_model/features.py: build_pair_features(profile: ProfileLike | dict, ...)

4. Verify StudentProfile and dict both satisfy the Protocol where used.

5. Run: pytest tests/ -q â†’ all tests pass
6. Run: ruff check â†’ 0 errors
```

---

### Task 1.4: Consolidate Test Fixtures into `conftest.py`

**Why:** `tests/conftest.py` currently only adds `sys.path`. `StudentProfile`, sample DataFrames, and `GoldenStudent` instances are constructed inline in 8+ test files with significant duplication. Centralising shared fixtures signals mature test engineering.

**Checklist:**
- [ ] Create shared fixtures: `sample_profile`, `sample_scholarship_df`, `sample_golden_student`
- [ ] Create a `scholarship_row_factory` fixture for building custom rows
- [ ] Refactor 8+ test files to use conftest fixtures instead of inline setup
- [ ] Add `@pytest.mark.parametrize` to at least 5 test functions that currently test single cases
- [ ] All 45+ tests still pass

**Prompt for Claude Sonnet 4.6:**

```
Consolidate test fixtures into conftest.py and add parametrize decorators.

CONTEXT:
- tests/conftest.py currently only has sys.path setup (after Task 0.3, maybe nothing at all)
- 22 test files, 45 tests, zero @pytest.mark.parametrize usage
- Many tests independently create:
  * StudentProfile(gpa=3.5, state="CA", major="Computer Science", ...)
  * pd.DataFrame with scholarship columns (scholarship_id, title, amount_max, deadline, ...)
  * GoldenStudent instances
  * Stage2Weights.baseline() / Stage3Weights.baseline()

REQUIREMENTS:
1. Read ALL test files to catalog repeated setup patterns.

2. Create fixtures in tests/conftest.py:

   @pytest.fixture
   def sample_profile() -> StudentProfile:
       """Standard student profile for testing: CA CS major, 3.5 GPA."""
       return StudentProfile(gpa=3.5, state="CA", major="Computer Science",
                             education_level="undergraduate", citizenship="US",
                             today=date(2026, 2, 22))

   @pytest.fixture
   def sample_scholarship_df() -> pd.DataFrame:
       """5-row scholarship DataFrame covering common test scenarios."""
       # Include: varied deadlines, amounts, majors, states, essay requirements

   @pytest.fixture
   def baseline_stage2_weights() -> Stage2Weights:
       return Stage2Weights.baseline()

   @pytest.fixture
   def baseline_stage3_weights() -> Stage3Weights:
       return Stage3Weights.baseline()

3. Refactor test files to use these fixtures where the inline setup matches.
   Keep inline setup when it's test-SPECIFIC (edge cases, boundary conditions).

4. Add @pytest.mark.parametrize to at least these functions:
   - test_canonical_id.py: parametrize field-change sensitivity (title, deadline, sponsor, amount)
   - test_eligibility_rules.py: parametrize each reason code individually
   - test_explainability_helpers.py: parametrize format_amount_range combos
   - test_stage3_rerank.py: parametrize urgency with various deadline distances
   - test_relevance_labeling.py: parametrize threshold boundary cases

5. Run: pytest tests/ -v â†’ all pass, count should be > 45 (new parametrized cases)
```

---

### Task 1.5: Add GitHub Actions CI Workflow

**Why:** No CI means no green badge, no automated test runs on push, and no proof that the code actually works. A simple pytest + ruff workflow takes 5 minutes to set up and is the most visible "this person knows professional workflows" signal on a GitHub portfolio.

**Checklist:**
- [ ] Create `.github/workflows/ci.yml` with pytest + ruff on push/PR
- [ ] Pin Python 3.11 or 3.12 to match local dev
- [ ] Cache pip dependencies for speed
- [ ] Add status badge to README.md
- [ ] Verify workflow runs green after push

**Prompt for Claude Sonnet 4.6:**

```
Create a GitHub Actions CI workflow for the project.

CONTEXT:
- Python 3.12 (local development environment)
- Testing: pytest (45+ tests, ~17s runtime)
- Linting: ruff (configured in pyproject.toml, line-length=100, target=py311)
- Dependencies in pyproject.toml [project.dependencies] and [project.optional-dependencies.dev]
- No Docker, no cloud services needed
- Tests do not require network access or large model files

REQUIREMENTS:
1. Create .github/workflows/ci.yml:

   name: CI
   on:
     push:
       branches: [main]
     pull_request:
       branches: [main]
   
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: "3.12"
             cache: "pip"
         - run: pip install -e ".[dev]"
         - run: ruff check src/ scripts/ app/ tests/
         - run: pytest tests/ -q --tb=short

2. Add badge to README.md (right after the title):
   ![CI](https://github.com/daedwards06/ScholarshipCoach/actions/workflows/ci.yml/badge.svg)

3. Note: Some tests use sentence-transformers which download a model on first run.
   Either:
   a. Mock the model in tests (already done in test_stage2_embeddings_mode.py)
   b. Add a pip install for sentence-transformers in CI
   c. Skip embedding tests in CI with a marker

4. The CI should complete in < 2 minutes.
```

---

### Task 1.6: Add `pytest-cov` Coverage Measurement

**Why:** You can't improve what you don't measure. Adding coverage reporting identifies untested code paths and the badge gives instant credibility on the README. Combined with CI, it shows continuous quality monitoring.

**Checklist:**
- [ ] Add `pytest-cov` to dev dependencies in `pyproject.toml`
- [ ] Configure coverage in `pyproject.toml` (`[tool.coverage.run]` and `[tool.coverage.report]`)
- [ ] Run `pytest --cov=src --cov-report=term-missing` to establish baseline
- [ ] Add coverage badge to README (via codecov.io or local generation)
- [ ] Identify top 3 modules with lowest coverage for future improvement

**Prompt for Claude Sonnet 4.6:**

```
Set up pytest-cov for test coverage measurement.

CONTEXT:
- 45+ tests across 22 test files
- Source code in src/ (~1,750 LOC across 20 modules)
- pyproject.toml already has [tool.pytest.ini_options]
- No current coverage configuration

REQUIREMENTS:
1. Add pytest-cov to pyproject.toml dev dependencies:
   [project.optional-dependencies]
   dev = ["pytest", "ruff", "pytest-cov"]

2. Add coverage configuration to pyproject.toml:
   [tool.coverage.run]
   source = ["src"]
   omit = ["src/scholarshipcoach.egg-info/*"]
   
   [tool.coverage.report]
   show_missing = true
   skip_empty = true
   fail_under = 50

3. Update pytest addopts:
   [tool.pytest.ini_options]
   addopts = "-q --cov=src --cov-report=term-missing:skip-covered"

4. Run: pytest tests/ â†’ observe coverage report
5. Report the baseline coverage percentage and the 3 lowest-covered modules.
6. Add a note to README: "Coverage: X% (run `pytest --cov` for details)"
```

---

## Phase 2: Data Pipeline & Ranking Science (Tier 3)

**Goal:** Fix the data quality issues killing metric credibility (dead keyword overlap, $0 scholarships ranking high), add a second data source for real aggregation value, and re-tune weights on a healthy catalog.

### Task 2.1: Fix $0-Amount Scholarships Ranking Highly

**Why:** Scholarships with `amount_max = 0` or `None` are ranking #1 in some profiles because the win model assigns them high `p_win` (0.72) while `expected_value = 0`. The ranking over-weights non-monetary signals. For your student's actual use, recommending $0 scholarships is worse than useless â€” it wastes time.

**Checklist:**
- [ ] Add an `amount_min_filter` to Stage 1 or Stage 3 that penalizes `amount = 0/None`
- [ ] Option A: Filter them out in Stage 1 as `AMOUNT_MISSING_OR_ZERO` reason code
- [ ] Option B: Apply a heavy penalty in Stage 3 so they sink to the bottom
- [ ] Update win model features to cap `p_win` contribution when amount is zero
- [ ] Add test cases: $0 scholarship should rank below any scholarship with a positive amount
- [ ] Run golden eval to verify impact

**Prompt for Claude Sonnet 4.6:**

```
Fix $0-amount scholarships ranking unreasonably high.

CONTEXT:
- Some scholarships have amount_max = 0 or None (e.g., "Anna Marjorie MacRae", 
  "A. F. Robertson Family Memorial Fund")
- Win model assigns them p_win = 0.72 but expected_value = p_win * 0 = $0
- They still rank #1 for some profiles because stage2_score and urgency_boost dominate
- For the user, recommending a $0 scholarship is worse than useless

ANALYSIS NEEDED:
1. Read src/rank/stage3_rerank.py to understand how final_score is computed
2. Read src/rank/stage2_scoring.py to see how amount_utility handles 0/None
3. Read src/win_model/features.py to see how amount affects p_win
4. Read src/rank/stage1_eligibility.py to see current filter rules

REQUIREMENTS:
- Choose the cleanest approach (filter in Stage 1 or penalize in Stage 3):
  * If filtering: add "AMOUNT_MISSING_OR_ZERO" reason code to Stage 1
  * If penalizing: ensure expected_value = 0 â†’ ev_signal = 0, AND add a 
    multiplicative penalty to final_score when amount is missing/zero
  * Either way, a $0 scholarship should never rank above a scholarship with 
    a real dollar amount, all else being roughly equal

- I prefer the Stage 1 filter approach (cleaner, explicit) unless there's a good 
  reason to keep them in the pipeline.

- Add a test: create two scholarships identical except amount (one $0, one $5000).
  Assert the $5000 one ranks higher.

- Run: pytest tests/ -q â†’ all pass
- Run golden eval to show the ranking improvement
```

---

### Task 2.2: Fix Dead `keyword_overlap` Feature

**Why:** `keyword_overlap` is 0.0 across nearly all 80 profile-scholarship pairs in the latest eval. This means your student's interests and keywords have zero influence on ranking. The feature is dead weight. The root cause is likely that scholarship records don't populate `keywords` or the keyword extraction logic doesn't tokenize properly.

**Checklist:**
- [ ] Diagnose why keyword_overlap is zero (schema field `keywords`, extraction logic)
- [ ] Fix the root cause:
  - If scholarships have empty `keywords`: improve ingest parser to extract meaningful keywords
  - If tokenization is the issue: fix the matching logic in `_compute_keyword_overlap`
  - If fields are named differently: fix the field mapping
- [ ] Verify keyword_overlap is non-zero for at least 30% of eligible pairs after fix
- [ ] Add a test that validates keyword_overlap > 0 for a known-matching pair
- [ ] Run golden eval to measure NDCG impact

**Prompt for Claude Sonnet 4.6:**

```
Diagnose and fix the dead keyword_overlap feature.

CONTEXT:
- keyword_overlap is 0.0000 for nearly all profile-scholarship pairs in golden eval
- This means the weight applied to keyword_overlap (Stage2Weights.keyword) has no effect
- The feature is supposed to measure overlap between student keywords/interests and 
  scholarship keywords/description terms

DIAGNOSIS STEPS:
1. Read src/rank/stage2_scoring.py â€” find _compute_keyword_overlap and _profile_keyword_tokens
2. Check what the function uses as "scholarship keywords" â€” the `keywords` column? 
   Or tokenized description?
3. Read a sample scholarship from the snapshot to see if `keywords` is populated:
   - Load data/processed/scholarships_snapshot_20260228.parquet (or latest)
   - Print the `keywords` column for 5 records
4. Read the ingest source parser to see how keywords are populated:
   - src/ingest/sources/scholarship_america_live.py
5. Read golden_students.py to see what keywords the profiles define

ROOT CAUSES (likely one or more):
A. Scholarship `keywords` column is empty/None for all records
B. Profile keywords don't match scholarship vocabulary
C. Tokenization logic strips or normalizes away matches
D. The column being read doesn't exist in the DataFrame

FIX:
- If the scholarship `keywords` field is empty: modify the keyword computation to 
  ALSO tokenize from description + title + eligibility_text (fall back to text tokens)
- If it's a vocabulary mismatch: use broader matching (stemming, or check substrings)
- Add a test with a known-matching pair: student keyword "STEM", scholarship with 
  "STEM" in description â†’ keyword_overlap > 0

Run: pytest tests/ -q â†’ all pass
```

---

### Task 2.3: Add a Second Scholarship Data Source

**Why:** With only one data source (Scholarship America), the system can't demonstrate real aggregation value. Adding a second connector proves the `BaseSource` abstraction works, increases the candidate universe (critical for your student), and showcases data engineering skills. A good portfolio project has at least 2-3 sources showing the pipeline handles heterogeneous data.

**Checklist:**
- [ ] Research available scholarship APIs/sites with scrapable listing pages
- [ ] Implement a new source extending `BaseSource` in `src/ingest/sources/`
- [ ] Register the new source in `src/ingest/registry.py`
- [ ] Ensure the new source produces records conforming to `NormalizedScholarshipRecord`
- [ ] Add polite rate limiting, retry, and caching consistent with `PoliteHttpClient`
- [ ] Add at least one parse test with mock data
- [ ] Run full ingest and verify deduplication across sources
- [ ] Snapshot should grow significantly (> 200 records ideal)

**Prompt for Claude Sonnet 4.6:**

```
Add a second scholarship data source to the ingestion pipeline.

CONTEXT:
- Current: only ScholarshipAmerica (scholarship_america_live.py, ~856 lines)
- Architecture: BaseSource ABC with fetch() + parse(), PoliteHttpClient, raw caching
- Registry: src/ingest/registry.py returns [ScholarshipAmericaLiveSource()]
- Schema: NormalizedScholarshipRecord with 22 fields including canonical_id
- Deduplication: by canonical scholarship_id (SHA-1 of title + sponsor + amounts + deadline + domain)

REQUIREMENTS:
1. Choose an appropriate second source. Good candidates:
   a. Scholarships.com â€” has a browsable directory by category
   b. Bold.org â€” has a public scholarship listing at bold.org/scholarships/
   c. GoingMerry â€” has a browseable listing
   d. CollegeBoard BigFuture â€” scholarship search
   
   Pick the one with the most accessible listing (least anti-scraping protection).
   Must be a PUBLIC scholarship listing site (no login required).

2. Create src/ingest/sources/<source_name>.py:
   - Inherit from BaseSource
   - Implement fetch() â†’ list[RawResponse] (paginated listing)
   - Implement parse(raw_content, fetched_at=) â†’ list[dict]
   - Map fields to NormalizedScholarshipRecord format
   - Handle missing/malformed fields gracefully (None, not crash)
   - Add polite request delay (â‰¥1 second between requests)

3. Update src/ingest/sources/__init__.py to export the new class.
4. Update src/ingest/registry.py to include the new source.

5. Add tests/test_<source_name>_parsing.py:
   - Test against a saved sample response (store in tests/resources/)
   - Verify schema compliance, stable IDs, field population

6. Test full ingest: python scripts/run_ingest.py --max-listing-pages 5 --max-detail-pages 50

7. Important: respect robots.txt and add appropriate caching. Include a polite 
   User-Agent string identifying this as a personal project.
```

---

### Task 2.4: Re-tune Weights on the Healthy Catalog

**Why:** The current Pareto weights were tuned on the 160-record snapshot (Feb 22). After fixing the data issues (fresh ingest, keyword overlap fix, $0 amount handling, second source), the catalog will be significantly different. Stale weights on new data leads to suboptimal rankings.

**Checklist:**
- [ ] Run fresh ingest (Task 0.1) + second source (Task 2.3) first
- [ ] Train a new win model on the expanded catalog
- [ ] Tune weights with all three objectives: relevance, blended, pareto
- [ ] Compare metrics before/after tuning (include in weight tuning report)
- [ ] Update `best_weights.json` and `best_weights_latest.json`
- [ ] Run final golden eval with the new weights and save the report

**Command sequence:**

```powershell
# After fresh ingest is done:
python scripts\tune_weights.py --k 10 --similarity-mode embeddings --label-mode hybrid --use-win-model --max_configs 200 --selection-objective pareto
python scripts\evaluate_golden_students.py --k 10 --similarity-mode embeddings --label-mode hybrid --use-win-model --use-best-weights
```

---

### Task 2.5: Rename `Stage2Weights.tfidf` to `text_sim`

**Why:** The `Stage2Weights.tfidf` field is used for both TF-IDF and embedding similarity, making the field name misleading. A portfolio reviewer reading `weights.tfidf` will think it's only TF-IDF. This was already noted in the `weights.py` docstring but never fixed. Small rename, big clarity.

**Checklist:**
- [ ] Rename `Stage2Weights.tfidf` â†’ `Stage2Weights.text_sim`
- [ ] Update all references in `stage2_scoring.py`, `tune_weights.py`, `evaluate_golden_students.py`
- [ ] Update JSON serialization/deserialization in weights files
- [ ] Add backward-compatible loading (accept both `tfidf` and `text_sim` keys in JSON)
- [ ] Update existing `best_weights*.json` files
- [ ] All tests pass

**Prompt for Claude Sonnet 4.6:**

```
Rename Stage2Weights.tfidf to text_sim for clarity.

CONTEXT:
- Stage2Weights is a frozen dataclass in src/rank/weights.py with field:
  tfidf: float  # Actually used for BOTH tfidf and embeddings similarity
- The system_design.md already acknowledges this: "The Stage2Weights.tfidf field 
  name is retained for backward compatibility, but it now weights the active 
  Stage 2 text similarity signal (text_sim) regardless of mode."
- References exist in: stage2_scoring.py, tune_weights.py, evaluate_golden_students.py,
  app/main.py, and best_weights*.json files

REQUIREMENTS:
1. Rename the field in src/rank/weights.py:
   @dataclass(frozen=True, slots=True)
   class Stage2Weights:
       text_sim: float   # was: tfidf
       amount: float
       keyword: float
       effort: float

2. Update from_mapping() to accept BOTH "tfidf" and "text_sim" keys for 
   backward compatibility:
   text_sim = payload.get("text_sim", payload.get("tfidf", cls.baseline().text_sim))

3. Update to_dict() to write "text_sim" (new canonical name).

4. Find and update ALL references:
   - src/rank/stage2_scoring.py (active_weights.tfidf â†’ active_weights.text_sim)
   - scripts/tune_weights.py (grid generation, config display)
   - scripts/evaluate_golden_students.py (weight display)
   - app/main.py (UI display)
   - docs/system_design.md (update the backward compat note)

5. Update data/processed/best_weights*.json files:
   - Change "tfidf" key to "text_sim" in the stage2 section

6. Run: pytest tests/ -q â†’ all pass
7. Run: ruff check â†’ 0 errors
```

---

## Phase 3: Portfolio Showpieces (Tier 3)

**Goal:** Create the artifacts that make this project stand out in an interview â€” a pipeline walkthrough notebook, Streamlit screenshots in the README, and a proper deployment posture.

### Task 3.1: Create Pipeline Walkthrough Notebook

**Why:** No notebooks exist in this project. A "pipeline walkthrough" notebook that loads the snapshot, runs each stage step-by-step, visualises score distributions, and explains design decisions is the highest-impact portfolio artifact after the README. Interviewers will open it.

**Checklist:**
- [ ] Create `notebooks/01_pipeline_walkthrough.ipynb`
- [ ] Section 1: Load snapshot, show dataset summary stats and schema
- [ ] Section 2: Stage 1 â€” demonstrate eligibility filtering for a sample profile, show reason codes
- [ ] Section 3: Stage 2 â€” show scoring breakdown, visualise component contributions
- [ ] Section 4: Stage 3 â€” show reranking, urgency boost curves, EV computation
- [ ] Section 5: Win model â€” show p_win distribution, expected value scatter
- [ ] Section 6: Weight tuning â€” show Pareto front visualisation
- [ ] Section 7: Full pipeline end-to-end for your student's profile
- [ ] Add markdown narrative cells explaining each design decision
- [ ] Notebook runs end-to-end in < 30 seconds

**Prompt for Claude Sonnet 4.6:**

```
Create a pipeline walkthrough notebook demonstrating the full ranking system.

CONTEXT:
- No notebooks exist yet in this project
- Pipeline: Stage 0 (load snapshot) â†’ Stage 1 (eligibility filter) â†’ Stage 2 (scoring) 
  â†’ Stage 3 (reranking with optional win model)
- Key modules:
  * src/rank/stage1_eligibility.py: apply_eligibility_filter(df, profile)
  * src/rank/stage2_scoring.py: score_stage2(eligible_df, profile, weights=...)
  * src/rank/stage3_rerank.py: rerank_stage3(scored_df, today, profile=..., weights=...)
  * src/eval/golden_students.py: get_golden_students() â†’ 9 profiles
  * src/rank/weights.py: Stage2Weights, Stage3Weights
  * src/win_model/: train, infer, features
- Snapshot: data/processed/scholarships_snapshot_YYYYMMDD.parquet
- Best weights: data/processed/best_weights_pareto.json

REQUIREMENTS:
1. Create notebooks/01_pipeline_walkthrough.ipynb with these sections:

   ## 1. Data Overview
   - Load the latest snapshot parquet
   - Show: shape, column names, dtypes
   - Histogram: amount_max distribution
   - Bar chart: deadline distribution (by month)
   - Table: 5 sample records (title, amount, deadline, state)
   - Key insight: catalog diversity and coverage

   ## 2. Stage 1 â€” Eligibility Filtering
   - Pick one golden profile (the NC CS student from Task 0.4)
   - Run apply_eligibility_filter()
   - Show: eligible count vs ineligible count
   - Bar chart: reason code breakdown from ineligible_df["reasons"]
   - Table: top rejected scholarships with their reason codes
   - Key insight: "X% filtered, most common reason is Y"

   ## 3. Stage 2 â€” Scoring Components
   - Run score_stage2() on eligible set
   - Show the 4 component columns: text_sim, amount_utility, keyword_overlap, effort_penalty
   - Stacked bar chart: component contributions for top-10
   - Scatter: text_sim vs amount_utility colored by stage2_score
   - Show the weight configuration being used
   - Key insight: "Text similarity dominates, effort penalty has marginal impact"

   ## 4. Stage 3 â€” Decision Reranking
   - Run rerank_stage3() with win model enabled
   - Show: urgency_boost curve (plot exp(-days/30) over 0-90 days)
   - Table: top-10 with final_score, stage2_score, urgency_boost, p_win, expected_value
   - Before/after comparison: Stage 2 rank vs Stage 3 rank (what moved?)
   - Key insight: "Urgency and EV reorder X of the top 10"

   ## 5. Win Probability Model
   - If model is available, load it and show feature importances
   - Scatter: p_win vs amount_max for eligible scholarships
   - Histogram: p_win distribution
   - Key insight: "Model correctly assigns higher p_win to better-fit scholarships"

   ## 6. Weight Tuning & Pareto Front
   - Load a weight tuning report JSON from reports/weight_tuning/artifacts/
   - Scatter: NDCG vs Coverage for all configs, highlight Pareto front
   - Show the selected knee point
   - Key insight: "Pareto selection balances relevance and coverage"

   ## 7. Full Pipeline for Your Student
   - Run the complete pipeline for the NC CS student profile
   - Show final top-10 recommendations as a clean table
   - For each: title, amount, deadline, p_win, expected_value, why_matched
   - Key insight: personalized actionable output

2. Use matplotlib with a dark theme (plt.style.use('dark_background')).
3. Each section: 2-3 markdown narrative cells, then code cells, then insight cell.
4. Keep code cells short (one concept per cell).
5. Total runtime < 30 seconds.
```

---

### Task 3.2: Add Streamlit Screenshots to README

**Why:** The Streamlit app is the most visual asset in the project, but it's invisible in the README. Recruiters browse GitHub â€” they won't clone and run the app. 2-3 screenshots show the app exists, works, and looks professional.

**Checklist:**
- [ ] Take 3 screenshots of the running Streamlit app:
  1. Main ranking view with top-10 scholarship cards
  2. Sidebar profile configuration
  3. Explainability view (expanded scholarship with signal breakdown)
- [ ] Save to `docs/images/` directory
- [ ] Add screenshots to README.md under a "Screenshots" section
- [ ] Ensure images are reasonably sized (< 500KB each)

**Prompt for Claude Sonnet 4.6:**

```
Add a Screenshots section to the README with image placeholders.

CONTEXT:
- The Streamlit app (app/main.py) has these visible features:
  * Sidebar: profile entry, weights profile selector, similarity mode toggle
  * Main area: ranked scholarship cards with scores, amounts, deadlines
  * Expandable cards: signal breakdown, explanation text
  * Win model section: p_win, expected_value columns

REQUIREMENTS:
1. Create the docs/images/ directory.

2. Add a "Screenshots" section to README.md between "Running the Project" and 
   "Design Principles" sections:

   ## ðŸ“¸ Screenshots

   ### Ranked Scholarship Recommendations
   ![Scholarship Rankings](docs/images/ranking_view.png)
   *Top-10 ranked scholarships with scores, amounts, and deadline urgency*

   ### Student Profile Configuration
   ![Profile Setup](docs/images/profile_sidebar.png)
   *Sidebar for entering student profile and selecting weight configurations*

   ### Explainability & Signal Breakdown
   ![Explainability](docs/images/explainability.png)
   *Expanded view showing why a scholarship was ranked â€” text similarity, 
   keyword overlap, win probability, and expected value*

3. Create a placeholder file docs/images/.gitkeep so the directory is committed.

4. Add a note: "Screenshots are from a local run. Launch the app with 
   `streamlit run app/main.py` to explore interactively."

NOTE: The actual screenshots must be captured manually by the user. This task 
creates the README structure and image placeholders. After running the app, 
the user will save screenshots to docs/images/ with the specified filenames.
```

---

### Task 3.3: Refactor `tune_weights.py` into Submodules

**Why:** At 1,353 lines, `scripts/tune_weights.py` is the longest file in the project and the most complex. It contains grid generation, evaluation, Pareto front computation, blended scoring, and Markdown report generation â€” all in one file. Splitting it into focused modules proves you can manage complexity in a real system.

**Checklist:**
- [ ] Create `src/tuning/` package
- [ ] Extract `src/tuning/grid.py` â€” `generate_candidate_configs()`, `WeightConfig`
- [ ] Extract `src/tuning/pareto.py` â€” `_dominates()`, `_pareto_front()`, `_pareto_knee_point()`
- [ ] Extract `src/tuning/objectives.py` â€” blended scoring, objective weight resolution
- [ ] Extract `src/tuning/reporting.py` â€” `_report_markdown()`, metric formatting helpers
- [ ] Slim `scripts/tune_weights.py` to CLI parsing + orchestration (< 200 lines)
- [ ] All existing tests pass
- [ ] Tuning script produces identical output

**Prompt for Claude Sonnet 4.6:**

```
Refactor tune_weights.py (1,353 lines) into focused submodules.

CONTEXT:
- scripts/tune_weights.py contains:
  * CLI parsing (~50 lines)
  * WeightConfig dataclass + grid generation (~100 lines)
  * ProfileCache dataclass + preparation (~80 lines)
  * _run_config_once evaluation (~60 lines)
  * Metric aggregation (~40 lines)
  * Pareto front: _dominates, _pareto_front, _pareto_knee_point (~80 lines)
  * Blended scoring: _resolve_objective_weights, _blended_score, _with_blended_scores (~60 lines)
  * Reporting: _report_markdown and helpers (~200+ lines)
  * Weight artifact writing (~60 lines)
  * main() orchestration (~150 lines)
  * Supporting utility functions (~400+ lines)

REQUIREMENTS:
1. Create src/tuning/__init__.py (empty or with key exports)

2. Create src/tuning/grid.py:
   - Move WeightConfig, generate_candidate_configs()
   - Export from __init__.py

3. Create src/tuning/pareto.py:
   - Move _dominates(), _pareto_front(), _pareto_knee_point()
   - Make them public (remove _ prefix): dominates, pareto_front, pareto_knee_point
   - Add docstrings explaining the Pareto algorithm

4. Create src/tuning/objectives.py:
   - Move _resolve_objective_weights(), _blended_score(), _with_blended_scores()
   - Make public interfaces

5. Create src/tuning/reporting.py:
   - Move _report_markdown() and its formatting helpers
   - Move _write_best_weights_artifacts()

6. Slim scripts/tune_weights.py to:
   - Imports from src/tuning/*
   - parse_args()
   - ProfileCache + _prepare_profile_caches (specific to this script's data flow)
   - _run_config_once (depends on ProfileCache)
   - main() orchestration
   - Target: < 300 lines

7. Add tests/test_pareto.py:
   - Test _dominates with known-answer cases
   - Test _pareto_front returns only non-dominated points
   - Test _pareto_knee_point selection

8. Run: pytest tests/ -q â†’ all pass
9. Run the tuning script to verify identical output
```

---

### Task 3.4: Create `.streamlit/config.toml` for Deployment Readiness

**Why:** Streamlit Community Cloud (or any deployment) uses `.streamlit/config.toml` for theme and server configuration. Having it pre-configured shows deployment awareness and ensures the app looks consistent regardless of where it runs.

**Checklist:**
- [ ] Create `.streamlit/config.toml` with theme matching the app's dark design
- [ ] Add deployment notes to README or a `docs/DEPLOYMENT.md`
- [ ] Test that the app respects the config: `streamlit run app/main.py`
- [ ] Add Streamlit Community Cloud badge placeholder to README

**Prompt for Claude Sonnet 4.6:**

```
Create Streamlit configuration and deployment docs.

CONTEXT:
- App: app/main.py (Streamlit)
- No .streamlit/ directory exists
- App uses dark theme with custom colors visible in the CSS injection
- No environment variables or secrets needed

REQUIREMENTS:
1. Create .streamlit/config.toml:
   [theme]
   primaryColor = "#4ECDC4"
   backgroundColor = "#0E1117"
   secondaryBackgroundColor = "#1A1A2E"
   textColor = "#FAFAFA"
   font = "sans serif"
   
   [server]
   headless = true
   maxUploadSize = 5

2. Add to .gitignore (if not already):
   .streamlit/secrets.toml

3. Add Streamlit badge to README.md:
   [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/daedwards06/ScholarshipCoach/main/app/main.py)

4. Brief note in README under "Running the Project":
   "This app can be deployed to Streamlit Community Cloud. See .streamlit/config.toml 
   for theme configuration."
```

---

## Execution Order & Dependencies

```
Phase 0 (Quick-Wins â€” Tier 1) â€” Do first, no dependencies:
  0.1 Fresh ingest          (independent â€” do first to unblock downstream)
  0.2 LICENSE + .gitignore   (independent)
  0.3 Remove sys.path hacks  (independent â€” cleans up lint)
  0.4 Add real student profile (independent)
  0.5 Update README metrics   (depends on 0.1 completion)

Phase 1 (Code Quality â€” Tier 2) â€” After Phase 0:
  1.1 Extract _normalize_text (independent)
  1.2 Add docstrings          (independent, best after 1.1 so the new module gets docs)
  1.3 Type safety Protocol    (independent)
  1.4 Conftest fixtures        (independent)
  1.5 GitHub Actions CI        (depends on 0.3 for clean ruff)
  1.6 pytest-cov               (independent, best after 1.4 for accurate baseline)

Phase 2 (Data & Ranking â€” Tier 3) â€” After Phase 0, can overlap Phase 1:
  2.1 Fix $0-amount ranking    (independent)
  2.2 Fix keyword_overlap      (independent)
  2.3 Add second data source   (independent, but biggest effort)
  2.4 Re-tune weights          (depends on 0.1, 2.1, 2.2, ideally 2.3)
  2.5 Rename tfidf â†’ text_sim  (independent, do before 2.4 so new weights use new name)

Phase 3 (Portfolio Showpieces â€” Tier 3) â€” After Phases 1 & 2:
  3.1 Pipeline walkthrough notebook  (best after 2.1-2.4 so data is healthy)
  3.2 README screenshots             (best after all code changes)
  3.3 Refactor tune_weights.py       (independent)
  3.4 Streamlit config + deployment   (best last â€” final polish before deploying)
```

---

## Quick Reference: Session Prompts

| Session | Tasks | Est. Time | Priority |
|---------|-------|-----------|----------|
| Session 1 | 0.1 (Fresh ingest) + 0.2 (LICENSE) + 0.3 (sys.path) | 45-60 min | **Critical** |
| Session 2 | 0.4 (Student profile) + 0.5 (README metrics) | 30-45 min | **Critical** |
| Session 3 | 1.1 (normalize_text) + 1.3 (Type Protocol) | 60-90 min | High |
| Session 4 | 1.2 (Docstrings) | 60-90 min | High |
| Session 5 | 1.4 (Conftest) + 1.5 (CI) + 1.6 (Coverage) | 60-90 min | High |
| Session 6 | 2.1 ($0-amount fix) + 2.2 (keyword_overlap fix) + 2.5 (rename tfidf) | 60-90 min | High |
| Session 7 | 2.3 (Second data source) | 90-120 min | High |
| Session 8 | 2.4 (Re-tune weights) | 30-45 min | High |
| Session 9 | 3.1 (Pipeline notebook) | 90-120 min | High |
| Session 10 | 3.2 (Screenshots) + 3.3 (Refactor tune_weights) | 60-90 min | Medium |
| Session 11 | 3.4 (Deployment config) + Final review | 30-45 min | Medium |

---

## Success Criteria

After all phases, the project should:

1. **Data:** Snapshot with 150+ scholarships from 2+ sources; no $0-amount junk in top-10
2. **Metrics:** NDCG@10 and Coverage@10 computed on a healthy catalog with honest README values
3. **Code Quality:** 0 ruff errors; docstrings on all public functions; no `_normalize_text` duplication
4. **Type Safety:** `ProfileLike` Protocol replaces `Any` for profile parameters
5. **Testing:** Shared conftest fixtures; parametrized tests; 60+ test cases; coverage measured
6. **CI/CD:** GitHub Actions green badge on README; ruff + pytest on every push
7. **Keyword Feature:** `keyword_overlap` contributes non-zero signal for 30%+ of eligible pairs
8. **Portfolio Artifacts:** Pipeline walkthrough notebook; Streamlit screenshots in README; LICENSE
9. **Personal Use:** Your student's real profile in golden eval; recommendations are actionable
10. **Grade:** Solid A for a Data Science portfolio project â€” demonstrates ranking systems, evaluation methodology, and software engineering maturity
