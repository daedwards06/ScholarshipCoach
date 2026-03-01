# MARS Portfolio Upgrade Plan — Tier 2 & Tier 3

> Generated: 2026-02-08 | Based on B+ → A+ audit roadmap  
> Target executor: Claude Opus 4.6 / Sonnet 4.5 via Copilot agent mode  
> Estimated total effort: 3 phases, ~8-10 focused sessions  
> Prerequisite: All Tier 1 quick-wins completed (210 tests passing, ruff, codecov, label fixes, user-embedding bug)

---

## Table of Contents

1. [Phase 1: App Architecture & Modularity (Tier 2)](#phase-1-app-architecture--modularity-tier-2)
2. [Phase 2: Recommendation Science & Explainability (Tier 3)](#phase-2-recommendation-science--explainability-tier-3)
3. [Phase 3: Portfolio Showpieces (Tier 3)](#phase-3-portfolio-showpieces-tier-3)

---

## Phase 1: App Architecture & Modularity (Tier 2)

**Goal:** Break the 1,931-line `app/main.py` monolith into focused modules, wire up the dead theme system, modernise Streamlit performance with `@st.fragment`, and consolidate test infrastructure.

### Task 1.1: Split `app/main.py` into Focused Modules

**Why:** A 1,931-line single-file Streamlit app is the number-one code smell a portfolio reviewer will notice. Splitting it into sidebar, state management, display, and pipeline-runner modules proves you understand separation of concerns and makes the codebase navigable.

**Checklist:**
- [x] Create `app/sidebar.py` — extract sidebar sections (profile, personalization, search, filters, help): ~L505–1160
- [x] Create `app/state.py` — extract session-state initialisation, query-param helpers, `_qp_get()`, `_ratings_signature()`: ~L219–456, L457–504, L752
- [x] Create `app/display.py` — extract CSS injection, header rendering, card orchestration, diversity panel: ~L232–375, L1160–1931
- [x] Create `app/pipeline_runner.py` — extract the glue that calls `scoring_pipeline` and feeds results to display: ~L1160–1400
- [x] Update `app/main.py` to import and wire the extracted modules (<300 lines target)
- [x] Move `_coerce_genres()` at L1653 to use the canonical `src/utils/parsing.py` import (if it exists from the earlier plan's Task 5.1)
- [x] Verify all 210+ existing tests still pass
- [x] Verify the app runs end-to-end: `streamlit run app/main.py`

**Prompt for Claude Sonnet 4.5:**

```
I need to split app/main.py (1,931 lines) into focused modules to eliminate the monolith.

CONTEXT:
- app/main.py is the Streamlit entry point for MARS — it handles sidebar, state, CSS, recommendations, and display
- The scoring logic already lives in src/app/scoring_pipeline.py (extracted previously)
- Current major sections by line range:
  L1–147: Imports
  L148–166: init_bundle(), load_personas() (cached loaders)
  L167–231: Theme init, page config, session state defaults
  L232–375: CSS injection (<style> block), header rendering
  L376–504: Mode selector, query-param helpers, first-run experience
  L505–744: Sidebar — User Profile (load/save/manage, watchlist, taste profile)
  L744–935: Sidebar — Personalization toggle & controls
  L935–1019: Sidebar — Search & Seeds
  L1019–1139: Sidebar — Filters & Display options
  L1139–1160: Sidebar — Help & FAQ
  L1160–1931: Main content — recommendation execution, card rendering, diversity panel

- Key functions in main.py: init_bundle() L148, load_personas() L153, _qp_get() L457,
  _ratings_signature() L752, _pop_pct_for_anime_id() L1220, _is_in_training() L1230,
  _coerce_genres() L1653
- Components already live in src/app/components/ (cards.py, diversity.py, help.py, etc.)
- Theme tokens in src/app/theme.py (currently unused — separate task to wire up)

REQUIREMENTS:
1. Create app/sidebar.py:
   - Move ALL sidebar rendering code (~L505–1160)
   - Function: render_sidebar(bundle, personas) -> dict with all sidebar state
   - The dict should contain: selected seeds, filters, personalization settings, profile data
   - Import components from src/app/components/ as needed

2. Create app/state.py:
   - Move session-state initialization (~L219–231)
   - Move _qp_get() and query-param handling (~L457–504)
   - Move _ratings_signature() (~L752)
   - Function: init_session_state() — sets all defaults
   - Function: get_query_param(key, default) — wraps _qp_get

3. Create app/display.py:
   - Move CSS injection block (~L232–375) into inject_css()
   - Move header rendering into render_header()
   - Move card rendering orchestration, diversity panel rendering
   - Move helper functions: _pop_pct_for_anime_id(), _is_in_training(), _coerce_genres()

4. Create app/pipeline_runner.py:
   - Move the code that calls scoring_pipeline functions and collects results
   - Function: run_recommendations(bundle, sidebar_state) -> list[dict]
   - This is the glue between sidebar selections and src/app/scoring_pipeline

5. Reduce app/main.py to <300 lines:
   - Imports from the new modules
   - Page config (st.set_page_config)
   - init_bundle() and load_personas() (keep cached functions here)
   - Wire: init_state → render_sidebar → run_recommendations → display
   - First-run experience logic (~L484–504) stays here or moves to state.py

6. Do NOT change any business logic — this is purely structural.
7. Run all tests after the split: pytest tests/ -x
8. Run the app to verify: streamlit run app/main.py
```

---

### Task 1.2: Wire Up `src/app/theme.py` Design Tokens

**Why:** `theme.py` defines a complete design-token system (colors, spacing, typography, elevation) but the CSS in `main.py` hardcodes hex values directly. This is dead code — a code reviewer will see `from src.app.theme import get_theme` followed by zero usages. Either use it or remove it.

**Checklist:**
- [x] Audit the CSS `<style>` block in `app/main.py` (or `app/display.py` after Task 1.1) for hardcoded colors/spacing
- [x] Replace hardcoded values with references to `get_theme()` tokens
- [x] Update `inject_css()` to accept a `theme` dict and interpolate tokens via f-string or `.format()`
- [x] Add at least one test verifying that `get_theme()` returns expected keys
- [x] Verify visual appearance is unchanged after wiring

**Prompt for Claude Sonnet 4.5:**

```
I need to wire up the dead theme.py design-token system so the app CSS uses it.

CONTEXT:
- src/app/theme.py (89 lines) defines:
  COLORS: brand (#FF6B6B, #4ECDC4, #2C3E50), surfaces (bg #0E1117, card #1A1A2E, etc.),
          text (primary #FAFAFA, secondary #B0B0B0), semantic (success, warning, error, info),
          data-viz (6 colors)
  SPACING: xs=4, sm=8, md=16, lg=24, xl=32, xxl=48
  TYPE_SCALE: sm=0.875rem, base=1rem, lg=1.25rem, xl=1.5rem, xxl=2rem, display=3rem
  ELEVATION: 3 levels of box-shadow
  get_theme() → returns dict with all token groups

- app/main.py imports get_theme() at L112, calls it at L217, but NEVER uses the returned theme dict
- The CSS <style> block (~L232–375) hardcodes colors like #0E1117, #1a1a2e, #e0e0e0, etc.
- These hardcoded values overlap significantly with theme.py tokens

REQUIREMENTS:
1. Read src/app/theme.py fully to map every token.

2. Read the CSS <style> block in app/main.py (or app/display.py if Task 1.1 is done).

3. For each hardcoded color/spacing value in the CSS:
   - Find the matching theme token
   - Replace the hardcoded value with a Python f-string variable: e.g., `background: {theme['colors']['surfaces']['bg']}`

4. Refactor the CSS injection to be a function:
   def inject_css(theme: dict) -> None:
       css = f"""<style>
       .stApp {{ background-color: {theme['colors']['surfaces']['bg']}; }}
       ...
       </style>"""
       st.markdown(css, unsafe_allow_html=True)

5. If any hardcoded colors DON'T have a matching token, add the token to theme.py.

6. Add tests/test_theme.py:
   - test_get_theme_returns_all_groups: assert {"colors", "spacing", "type_scale", "elevation"} ⊆ keys
   - test_colors_has_brand_keys: assert "brand" in theme["colors"]
   - test_spacing_values_are_ints: assert all(isinstance(v, int) for v in theme["spacing"].values())

7. Verify the app looks identical before and after — no visual regression.
```

---

### Task 1.3: Modernise Streamlit with `@st.fragment`

**Why:** Every widget interaction triggers a full rerun of all 1,931 lines (or the equivalent after splitting). `@st.fragment` (Streamlit ≥1.33) lets sidebar controls, filter panels, and the diversity panel rerun independently — noticeably faster UX and a modern Streamlit best-practice to showcase.

**Checklist:**
- [x] Verify Streamlit version ≥1.33 in requirements.txt (current: 1.38 — good)
- [x] Wrap the sidebar rendering function in `@st.fragment`
- [x] Wrap the diversity/explanation panel in `@st.fragment`
- [x] Wrap the filter controls in `@st.fragment`
- [x] Test that changing a filter does NOT re-execute the full scoring pipeline
- [x] Measure before/after rerun latency with `st.session_state["_last_rerun_ms"]` timing

**Prompt for Claude Sonnet 4.5:**

```
I need to add @st.fragment decorators to reduce full-page reruns in my Streamlit app.

CONTEXT:
- Streamlit version: 1.38 (requirements.txt) — @st.fragment is available (added in 1.33)
- Currently, every widget interaction causes a full rerun of the entire app
- After Task 1.1, the app has: sidebar.py, state.py, display.py, pipeline_runner.py
  (If Task 1.1 is NOT done yet, work with the monolithic app/main.py)
- The scoring pipeline (src/app/scoring_pipeline.py) is the expensive part (~200-500ms)
- Sidebar filter changes (genre, type, year, top-N) should NOT re-run the pipeline — 
  they should only re-filter the existing results

REQUIREMENTS:
1. Identify which sections can be fragments (rerun independently):
   a. Sidebar rendering — widget state changes shouldn't rerun main content
   b. Filter panel — genre/type/year filters rerun only the display, not scoring
   c. Diversity/explanation panel — collapses/expands without rerunning anything
   d. Rating buttons on cards — submitting a rating shouldn't rerun the full pipeline

2. For each fragment:
   - Add @st.fragment decorator to the rendering function
   - Ensure session_state is used for cross-fragment communication
   - Use st.rerun() only when a fragment change SHOULD trigger a full pipeline rerun
     (e.g., changing seeds or toggling personalization)

3. Implementation pattern:
   @st.fragment
   def render_filter_panel():
       genre = st.multiselect("Genre", options=GENRE_LIST, key="genre_filter")
       # ... other filters
       # Store in session_state for display.py to read
       st.session_state["active_filters"] = {"genre": genre, ...}
   
   @st.fragment  
   def render_diversity_panel(results):
       with st.expander("Diversity Metrics"):
           # ... render metrics
           # This never needs to trigger a pipeline rerun

4. Add timing instrumentation:
   import time
   start = time.perf_counter()
   # ... main pipeline
   st.session_state["_pipeline_ms"] = (time.perf_counter() - start) * 1000
   # Show in debug: st.caption(f"Pipeline: {st.session_state['_pipeline_ms']:.0f}ms")

5. Do NOT fragment the initial pipeline execution — that must run in the main flow.

6. Test manually: change a genre filter → observe that only the card display updates,
   not the full pipeline.
```

---

### Task 1.4: Consolidate Test Fixtures into `conftest.py`

**Why:** `tests/conftest.py` currently only adds `sys.path`. Many test files independently create the same fixtures (mock metadata DataFrames, dummy models, sample ratings dicts). Centralising fixtures into `conftest.py` with `@pytest.fixture` eliminates duplication, makes tests more readable, and signals mature test engineering.

**Checklist:**
- [x] Audit all test files for repeated setup patterns (DataFrames, model mocks, rating dicts)
- [x] Create shared fixtures in `tests/conftest.py`: `sample_metadata`, `mock_mf_model`, `mock_knn_model`, `sample_ratings`, `sample_seeds`
- [x] Refactor existing tests to use conftest fixtures instead of inline setup
- [x] Add a `pipeline_result_factory` fixture for tests that need `PipelineResult` instances
- [x] Verify all 210+ tests still pass after fixture consolidation
- [x] Remove duplicated setup code from individual test files

**Prompt for Claude Sonnet 4.5:**

```
I need to consolidate repeated test fixtures into tests/conftest.py.

CONTEXT:
- tests/conftest.py currently only has sys.path setup (10 lines, no fixtures)
- There are 22 test files with 210+ tests
- Many tests independently create:
  • Mock metadata DataFrames with columns like anime_id, title, genres, type, episodes, score, etc.
  • Dummy MF model objects with P, Q, item_to_index, index_to_item, global_mean
  • Dummy kNN model objects
  • Sample rating dicts {anime_id: rating}
  • Sample seed sets {anime_id, ...}
  • PipelineResult-like dicts with scores and metadata

REQUIREMENTS:
1. Read ALL test files in tests/ to catalog repeated setup patterns. Focus on:
   - DataFrame creation (pd.DataFrame with anime metadata columns)
   - Model mock objects (SimpleNamespace or similar with MF attributes)
   - Rating dict literals
   - Seed set/list literals

2. Create fixtures in tests/conftest.py:

   @pytest.fixture
   def sample_metadata() -> pd.DataFrame:
       """Minimal metadata with 10 anime covering common test scenarios."""
       # Include diverse: genres, types (TV/Movie/OVA), score ranges, episode counts
       # IDs: 1-10 for easy assertion
   
   @pytest.fixture
   def mock_mf_model():
       """Mock MF model with P, Q matrices for 10 items."""
       # SimpleNamespace with: P (10x8 random), Q (10x8 random), 
       # item_to_index, index_to_item, global_mean=7.0
   
   @pytest.fixture
   def mock_knn_model():
       """Mock kNN model with precomputed similarities for 10 items."""
   
   @pytest.fixture
   def sample_ratings() -> dict[int, float]:
       """5 sample user ratings for anime IDs 1-5."""
       return {1: 9.0, 2: 8.0, 3: 7.0, 4: 6.0, 5: 5.0}
   
   @pytest.fixture
   def sample_seeds() -> list[int]:
       """3 seed anime IDs."""
       return [1, 2, 3]

3. Refactor existing tests to use these fixtures:
   - Replace inline DataFrame creation with `sample_metadata` parameter
   - Replace inline model mocks with `mock_mf_model` / `mock_knn_model`
   - Only keep inline setup when it's test-SPECIFIC (unusual edge case)

4. Run pytest -x after EACH file refactored to catch breakage early.

5. The goal is DRY test code — not abstracting away test clarity. If a test's setup 
   communicates something important about the test case, keep it inline.
```

---

## Phase 2: Recommendation Science & Explainability (Tier 3)

**Goal:** Add intra-list diversity reranking (the most impactful missing scientific feature) and create a pipeline walkthrough notebook that demonstrates ML depth for portfolio reviewers.

### Task 2.1: Implement Maximal Marginal Relevance (MMR) Diversification

**Why:** The current pipeline optimises purely for relevance — there is no intra-list diversity mechanism. This means recommending Naruto can return 10 battle-shounen shows. MMR (Carbonell & Goldstein, 1998) is the gold-standard reranking method that trades off relevance and diversity. Implementing it signals you understand beyond-accuracy evaluation and real-world recommendation challenges.

**Checklist:**
- [x] Add `mmr_rerank()` to `src/app/diversity.py`
- [x] Integrate MMR as an optional post-Stage-2 reranking step in `src/app/scoring_pipeline.py`
- [x] Add a `diversity_lambda` parameter (0.0 = pure relevance, 1.0 = max diversity, default 0.3)
- [x] Wire a Streamlit slider in the sidebar for diversity control
- [x] Add unit tests for `mmr_rerank()` with known-answer cases
- [x] Add integration test: same seeds, λ=0 vs λ=0.5 → different genre distributions
- [x] Update the diversity panel to show pre/post-MMR coverage comparison

**Prompt for Claude Sonnet 4.5:**

```
I need to implement Maximal Marginal Relevance (MMR) diversification for my recommendation pipeline.

CONTEXT:
- src/app/diversity.py currently computes diversity METRICS (genre coverage, Gini, ILD) 
  but does NO diversity-based reranking
- src/app/scoring_pipeline.py has run_seed_based_pipeline() and run_personalized_pipeline()
  that return PipelineResult with scored & ranked items
- Post-processing happens after Stage 2: franchise cap → blend → display filters → top-N
- Genre data is in metadata["genres"] as pipe-delimited strings (e.g., "Action|Adventure|Drama")
- The pipeline already has a "diversity_emphasized" weight profile in constants.py
  (DIVERSITY_EMPHASIZED_WEIGHTS = {"mf": 0.80, "knn": 0.18, "pop": 0.02})
- Synopsis embeddings (TF-IDF + SVD, 512-dim) are available in the artifacts bundle

REQUIREMENTS:
1. Add to src/app/diversity.py:

   def mmr_rerank(
       candidates: list[dict],
       similarity_fn: Callable[[dict, dict], float],
       lambda_param: float = 0.3,
       top_n: int = 10,
   ) -> list[dict]:
       """Maximal Marginal Relevance reranking (Carbonell & Goldstein, 1998).
       
       Iteratively selects items that maximize:
         MMR(i) = λ * relevance(i) - (1 - λ) * max_similarity(i, selected)
       
       Args:
           candidates: Scored items with "score" key (relevance) and feature vectors
           similarity_fn: Pairwise similarity function between two candidate dicts
           lambda_param: Trade-off (0=max diversity, 1=pure relevance)
           top_n: Number of items to select
       
       Returns:
           Reranked list of top_n items
       """

2. Implement two similarity functions for MMR:
   a. genre_jaccard_similarity(a, b) — Jaccard on genre sets
   b. embedding_cosine_similarity(a, b) — cosine on synopsis embedding vectors
   
   Default to genre Jaccard (fast, interpretable). Allow embedding cosine if embeddings 
   are available in the candidate dicts.

3. Integrate into scoring_pipeline.py:
   - Add optional mmr_lambda parameter to run_seed_based_pipeline() and run_personalized_pipeline()
   - Apply MMR reranking AFTER franchise cap but BEFORE final top-N selection
   - When mmr_lambda is None or 1.0, skip MMR (pure relevance — backward compatible)

4. Wire UI in the sidebar (or app/sidebar.py if Task 1.1 is done):
   - Add st.slider("Diversity", 0.0, 1.0, 0.3, 0.1, key="mmr_lambda")
   - Under "Advanced Settings" or "Recommendation Controls" section
   - Tooltip: "Lower = more similar results, Higher = more diverse genres"

5. Create tests/test_mmr.py:
   - test_mmr_lambda_1_preserves_relevance_order: λ=1.0 → same order as input
   - test_mmr_lambda_0_maximizes_diversity: λ=0.0 → genres are spread out
   - test_mmr_selects_correct_count: len(result) == top_n
   - test_mmr_empty_candidates: empty input → empty output
   - test_mmr_fewer_than_top_n: 3 candidates, top_n=10 → returns 3
   - test_mmr_genre_jaccard: verify Jaccard computation for known genre sets
   - test_mmr_integration_different_lambda: run pipeline with λ=0 and λ=0.5, 
     verify different genre distributions in output

6. Update the diversity panel (src/app/components/diversity.py) to show:
   - "Diversity λ: 0.3" badge
   - Genre coverage before vs after MMR

7. Reference in docstrings: Carbonell, J., & Goldstein, J. (1998). 
   "The use of MMR, diversity-based reranking for reordering documents and producing summaries."
```

---

### Task 2.2: Create Pipeline Walkthrough Notebook

**Why:** Only 1 notebook exists (`01_eda_data_quality.ipynb`). A "pipeline walkthrough" notebook that loads models, runs each stage step-by-step, visualises score distributions and embedding spaces, and explains design decisions is the highest-impact portfolio artifact after the README. Interviewers will open it.

**Checklist:**
- [x] Create `notebooks/02_pipeline_walkthrough.ipynb`
- [x] Section 1: Load data & models, show dataset summary stats
- [x] Section 2: Stage 0 demo — show candidate generation for a seed, visualise overlap between generators
- [x] Section 3: Stage 1 demo — show admission gating, plot score distributions pre/post filtering
- [x] Section 4: Stage 2 demo — show hybrid scoring, plot per-signal contributions
- [x] Section 5: Post-processing — show franchise cap, MMR (if Task 2.1 done), final ranking
- [x] Section 6: Embedding space visualisation — t-SNE/UMAP of synopsis embeddings colored by genre
- [x] Section 7: Ablation — run pipeline with/without each signal, table of metric impacts
- [x] Add markdown cells with clear narrative explaining decisions and trade-offs
- [x] Ensure notebook runs end-to-end in <60 seconds

**Prompt for Claude Sonnet 4.5:**

```
I need to create a pipeline walkthrough notebook that demonstrates the full recommendation 
pipeline step-by-step with visualizations. This is a KEY portfolio artifact.

CONTEXT:
- Only 1 notebook exists: notebooks/01_eda_data_quality.ipynb (EDA)
- The recommendation pipeline lives in src/app/scoring_pipeline.py with stages:
  Stage 0: Candidate generation (~3K items from neural neighbors, metadata overlap, popularity)
  Stage 1: Shortlist (~600 items via semantic admission, type/episode gates, confidence gating)
  Stage 2: Reranking (hybrid CF: 0.93 MF + 0.07 kNN, genre/theme/studio overlap, 
           synopsis similarity from 3 modalities, obscurity penalty, quality-scaled neural)
  Post: Franchise cap, personalization blend, display filters, top-N
- Models in models/: mf_sgd, item_knn_sklearn, synopsis_tfidf, synopsis_neural_embeddings
- Artifacts loader: src/app/artifacts_loader.py builds the full artifact bundle
- Constants in src/app/constants.py
- Data in data/processed/anime_metadata.parquet

REQUIREMENTS:
1. Create notebooks/02_pipeline_walkthrough.ipynb with these sections:

   ## 1. Setup & Data Overview
   - Load artifacts using build_artifacts() from src/app/artifacts_loader
   - Show: dataset shape, column types, score distribution histogram
   - Show: model file sizes, embedding dimensions

   ## 2. Stage 0 — Candidate Generation
   - Pick seed: "Steins;Gate" (anime_id 9253) — popular, well-connected
   - Run each Stage 0 generator separately:
     * Neural neighbors (show top-20 with similarity scores)
     * Metadata overlap (show overlap counts)
     * Popularity backfill (show safety net candidates)
   - Venn diagram or bar chart showing overlap between generators
   - Key insight: "Neural retrieval finds X% of final candidates"

   ## 3. Stage 1 — Shortlist
   - Show: how many candidates survive each gate (semantic, type, episode)
   - Plot: score distribution before vs after Stage 1 filtering (overlaid histograms)
   - Table: example rejected candidates and WHY they were rejected

   ## 4. Stage 2 — Hybrid Scoring
   - Show the full scoring formula breakdown for the top-5 results
   - Bar chart: per-signal contribution (MF, kNN, genre overlap, synopsis sim, etc.)
   - Scatter plot: MF score vs final score colored by genre
   - Key insight: "MF dominates at 93% weight but content signals break ties"

   ## 5. Post-Processing
   - Show franchise cap in action (which items got capped?)
   - Show final ranking vs pre-cap ranking
   - If MMR is implemented, show diversity before/after

   ## 6. Embedding Space
   - Load synopsis embeddings (neural, 384-dim)
   - Reduce to 2D with t-SNE or UMAP
   - Scatter plot colored by primary genre (top-8 genres only for readability)
   - Highlight the seed and its recommendations in a different marker
   - Key insight: "Genre clusters are visible; recommendations span X clusters"

   ## 7. Quick Ablation
   - Run pipeline with signals removed one at a time:
     * Full pipeline (baseline)
     * No MF (kNN + content only)
     * No kNN (MF + content only)
     * No synopsis similarity
   - Table: NDCG@10 or just qualitative ranking differences
   - Key insight: "Removing MF causes X; removing synopsis causes Y"

2. Use matplotlib/seaborn for plots with clean styling (dark background to match the app theme).
3. Each section should have 2-3 markdown cells of narrative before the code.
4. Keep code cells short — one concept per cell.
5. Total notebook should run in <60 seconds on a machine with the model files.
6. Final cell: summary table of key architecture decisions and their impact.
```

---

### Task 2.3: Create Model Card Document

**Why:** Model cards (Mitchell et al., 2019) are an industry best-practice for documenting ML models. Having one for your recommendation system signals awareness of responsible AI practices and model governance. It's a low-effort, high-impression-value document.

**Checklist:**
- [x] Create `docs/MODEL_CARD.md` following the standard format
- [x] Document: model architecture, training data, evaluation metrics, limitations, ethical considerations
- [x] Include real numbers from experiments/metrics/ and reports/
- [x] Add a "Limitations & Known Issues" section (honest — shows maturity)
- [x] Link from README.md

**Prompt for Claude Sonnet 4.5:**

```
I need to create a Model Card for the MARS recommendation system.

CONTEXT:
- MARS is a hybrid anime recommendation engine
- Models: FunkSVD MF (64 factors, SGD, 310K+ ratings from 73K users, 13K+ anime),
  Item-kNN (k=40, cosine, sklearn NearestNeighbors), 
  Synopsis TF-IDF + SVD (512 dims), Neural sentence embeddings (all-MiniLM-L6-v2)
- Hybrid weights: MF 93.08%, kNN 6.63%, Popularity 0.30%
- Training data: MyAnimeList dataset via Kaggle (Hernan4444/anime-recommendation-database-2020)
- Evaluation reports in reports/phase4_evaluation.md and reports/phase4_ablation.md
- Constants in src/app/constants.py (424 lines)
- Known limitations:
  * Cold-start: new users with <5 ratings get popularity fallback
  * English-only: synopsis similarity only works for English-language synopses
  * Popularity bias: MF inherits popularity bias from collaborative signal
  * Temporal: no time-aware decay, older highly-rated anime dominate
  * Demographics: MAL user base skews male, 15-25 age range

REQUIREMENTS:
1. Create docs/MODEL_CARD.md following the standard Model Card format (Mitchell et al., 2019):

   # Model Card: MARS Recommendation System

   ## Model Details
   - Developer, date, version, type, architecture summary
   - Link to paper/method for FunkSVD, MMR if implemented

   ## Intended Use
   - Primary: anime discovery for MAL users
   - Out of scope: production deployment, commercial use

   ## Training Data
   - Source, size, date range, preprocessing steps
   - User/item statistics

   ## Evaluation Data
   - Train/test split method
   - Held-out evaluation protocol

   ## Metrics
   - NDCG@K, MAP@K, Precision@K, Recall@K (pull real numbers from reports/)
   - Beyond-accuracy: Coverage, Gini Index, ILD
   - If numbers aren't available, add TODO placeholders

   ## Ethical Considerations
   - Popularity bias amplification
   - Demographic representation
   - Filter bubble risk
   - Content sensitivity (anime genres include violence, mature themes)

   ## Limitations & Known Issues
   - Cold-start behavior
   - English-only synopses  
   - ~50 hand-tuned constants
   - No A/B testing (offline eval only)
   - No temporal modeling

   ## Recommendations
   - Use cases where the model performs well vs poorly
   - Suggested improvements

2. Read reports/phase4_evaluation.md and reports/phase4_ablation.csv for real metric numbers.

3. Keep it honest — the "Limitations" section is where maturity shows. Don't hide weaknesses.

4. Add a link in README.md: "See [Model Card](docs/MODEL_CARD.md) for detailed model documentation."

5. Total length: 150-250 lines. Concise but complete.
```

---

## Phase 3: Portfolio Showpieces (Tier 3)

**Goal:** Deploy the app publicly and add production-readiness signals that distinguish a portfolio project from a homework assignment.

### Task 3.1: Deploy to Streamlit Community Cloud

**Why:** A live URL in your README is the single biggest credibility signal for a Streamlit project. Recruiters won't clone and run your code — they'll click a link. Streamlit Community Cloud is free, integrates directly with GitHub, and handles the infrastructure.

**Checklist:**
- [x] Verify all data files needed at runtime are committed or downloadable (parquet, joblib models)
- [x] Check model file sizes — Streamlit Cloud has a 1GB repo limit and 1GB RAM limit
- [x] Create `.streamlit/config.toml` with appropriate settings (theme, server config)
- [x] Add `packages.txt` if any system-level dependencies are needed (not needed for this project)
- [x] Ensure `requirements.txt` has pinned versions and no unnecessary packages
- [x] Test local deployment first: `streamlit run app/main.py` in a fresh venv
- [ ] Deploy to Streamlit Community Cloud via GitHub (ready to deploy - manual step)
- [x] Add the live URL to README.md as a prominent badge/button (placeholder URL added)
- [x] Test the deployed app: first load, search, recommendations, all 3 modes (after deployment)

**Prompt for Claude Sonnet 4.5:**

```
I need to prepare the project for deployment to Streamlit Community Cloud.

CONTEXT:
- App entry point: app/main.py
- Models in models/ directory (7 joblib files — check sizes)
- Data in data/processed/ (parquet files)
- requirements.txt exists but may have dev-only dependencies
- Currently no .streamlit/ config directory
- The app loads artifacts at startup via src/app/artifacts_loader.py
- Artifacts loader reads from relative paths based on project root
- Some imports do sys.path manipulation to find src/

REQUIREMENTS:
1. Check model file sizes:
   - List all files in models/ and data/processed/ with sizes
   - If total > 800MB, implement LFS or a download script
   - If total < 800MB, they can be committed directly

2. Create .streamlit/config.toml:
   [theme]
   primaryColor = "#FF6B6B"
   backgroundColor = "#0E1117"
   secondaryBackgroundColor = "#1A1A2E"
   textColor = "#FAFAFA"
   font = "sans serif"
   
   [server]
   maxUploadSize = 5
   enableStaticServing = true

3. Create a deployment-ready requirements.txt:
   - Remove dev-only packages (pytest, ruff, pytest-cov) — these go in requirements-dev.txt
   - Pin all versions for reproducibility
   - Minimize: only packages imported at runtime
   - Keep: streamlit, pandas, numpy, scikit-learn, sentence-transformers, joblib

4. Create requirements-dev.txt:
   - All testing and linting packages
   - Update CI to use: pip install -r requirements.txt -r requirements-dev.txt

5. Verify paths work from repo root:
   - Artifacts loader should use Path(__file__).resolve().parents[N] / "data/..." 
   - NOT hardcoded absolute paths
   - NOT os.getcwd() (unreliable in cloud)

6. Add to README.md (right after the title, before description):
   [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/YOUR_GITHUB_USER/MyAnimeRecommendationSystem/main/app/main.py)
   (Use a placeholder URL — the actual URL is created after deployment)

7. Create a DEPLOYMENT.md in docs/ with:
   - Step-by-step deployment instructions
   - Environment variables to set
   - Known deployment gotchas (model loading time, memory limits)
   - How to update after code changes

8. Test locally first:
   python -m venv .deploy-test
   pip install -r requirements.txt
   streamlit run app/main.py
```

---

## Execution Order & Dependencies

```
Phase 1 (Architecture — Tier 2):
  1.1 Split main.py ──→ 1.2 Wire theme.py (uses display.py from 1.1)
  1.1 Split main.py ──→ 1.3 @st.fragment (works on split modules)
  1.4 Consolidate conftest (independent)

Phase 2 (Science — Tier 3):
  2.1 MMR diversity (depends on scoring_pipeline.py, not on Phase 1)
  2.2 Pipeline notebook (independent, but best after 2.1 for MMR content)
  2.3 Model card (independent)

Phase 3 (Deployment — Tier 3):
  3.1 Streamlit Cloud (best after all code changes — do last)
```

---

## Quick Reference: Session Prompts

| Session | Tasks | Est. Time | Priority |
|---------|-------|-----------|----------|
| Session 1 | 1.1 (Split main.py) | 90-120 min | High |
| Session 2 | 1.2 (Wire theme.py) + 1.4 (Conftest fixtures) | 45-60 min | High |
| Session 3 | 1.3 (@st.fragment) | 45-60 min | Medium |
| Session 4 | 2.1 (MMR diversity) | 90-120 min | High |
| Session 5 | 2.2 (Pipeline notebook) | 90-120 min | High |
| Session 6 | 2.3 (Model card) | 30-45 min | Medium |
| Session 7 | 3.1 (Deploy to Streamlit Cloud) | 60-90 min | High |
| Session 8 | Final review & polish | 30-45 min | — |

---

## Success Criteria

After all phases, the project should:

1. **Architecture**: `app/main.py` < 300 lines; sidebar, state, display, pipeline-runner in separate modules
2. **Theme**: CSS uses design tokens from `theme.py` — zero hardcoded colors
3. **Performance**: `@st.fragment` eliminates full reruns for filter/collapse interactions
4. **Tests**: Shared fixtures in `conftest.py`; no duplicated setup across test files
5. **Diversity**: MMR reranking with user-controllable λ slider; genre coverage visibly improved
6. **Notebook**: Pipeline walkthrough with visualisations runs end-to-end; demonstrates ML depth
7. **Model Card**: `docs/MODEL_CARD.md` with honest limitations section
8. **Deployment**: Live Streamlit Cloud URL in README badge — one click to demo
9. **Grade**: Solid A / A+ for a Data Science portfolio project
