# ScholarshipCoach Family Product Plan — From Portfolio Pipeline to a Tool the Family Uses

> Generated: 2026-09-12 | Scope decided with the project owner over a product review
> Companion to `SCHOLARSHIPCOACH_PORTFOLIO_UPGRADE_PLAN.md`, `SCHOLARSHIPCOACH_UI_REDESIGN_PLAN.md`,
> `SCHOLARSHIPCOACH_EVAL_CREDIBILITY_PLAN.md`, and `SCHOLARSHIPCOACH_LLM_EXTRACTION_PLAN.md`
> Executor: Claude via Claude Code | Est. effort: 4 phases, 19 tasks + a future-enhancements list
> Hard deadline the plan is built around: the student's senior-year application season (fall 2028)

---

## Why this plan exists

A product review on 2026-09-12 measured what the app actually shows the student it was built
for, using the latest snapshot (`scholarships_snapshot_20260813.parquet`) and the
`nc_cs_rising_sophomore` golden profile with today's date:

| Finding | Evidence |
|---|---|
| The catalog is 35 records | Down from 166 in June; Bold.org returned **0** records and the ingest report still said `succeeded` |
| 22 of 35 are closed | `eligibility_text` begins `Status: Closed`; the parser never reads the status field |
| No parser has ever populated majors, GPA minimum, or education level | Those columns are 0 non-null in **every** snapshot in the repo, so Stage 1 is effectively a state-plus-deadline filter |
| A missing deadline passes Stage 1 | Closed awards with no parsed deadline rank as eligible with "Unknown deadline" |
| The student's top 5 are the 5 hand-typed static-feed entries | The next 9 are closed Scholarship America programs, several restricted to children of specific companies' employees |
| The canonical ID hashes the deadline | A recurring award gets a new `scholarship_id` every year, so nothing can be tracked across cycles |

The conclusion: **the ranking engine is not the bottleneck; the catalog is.** Everything
downstream (evaluation harness, weight tuning, win model, CI) is sound engineering that only
pays off once a few hundred real, open, correctly-structured awards sit behind it.

The second conclusion is about the job. For a family the job is not "rank 35 things". It is:
find awards the student can actually apply to, know when they are due, track what each one
needs, reuse essays, and keep outside scholarships in proportion to institutional and state aid,
which is where the large dollars are.

**Data acquisition reality (verified 2026-09-12):** there is no open dataset of US private or
local scholarships. Aggregators keep listings proprietary. The one openly licensed directory
(Open Scholarships, CC BY 4.0) is Nevada-focused today but has a schema close to ours. CFNC's
scholarship search filters on exactly our axes (county, deadline month, need/merit, essay and
recommendation requirements) but renders dynamically with no automated-access terms. Local and
regional awards (school counselor list, community foundations, employer, church, credit union)
have the smallest applicant pools and appear in no aggregator. The catalog is therefore
**curated first, fed second, scraped last**, and the ingest automation in this plan is
deterministic: structured feeds, regex prefill, content-hash re-verification.

**Decisions taken with the owner:**
- Portfolio and family product are the same repo: general in shape, specific in content. Code,
  schema, and roles never know the student exists; her data lives in private, git-ignored files.
- One student profile, two modes (student / parent) over the same data. Multi-student modeled
  from day one even though there is one row.
- No generative LLM in this plan. The LLM extraction module stays in the repo, flag off, and
  plugs into the prefill extractor interface as a future enhancement (Phase 5).
- The synthetic win model and expected value leave the family-facing views. The code stays
  behind the operator toggle and the README reframes it.
- Every feature passes one test: does it help the student by senior fall? General features that
  fail the test wait.

**Relationship to other plans:**
- **LLM Extraction Plan Tasks 6, 7, 8, 9 are deferred** to Phase 5 here. Do not execute them
  from that plan.
- **Portfolio Plan Task 4.1** (catalog growth gate) is served by Phase 1–2 here.
- **UI Redesign Plan Task 6** (README screenshots) is executed **after** Phase 3 of this plan,
  from that plan, against the family product UI.

---

## Design principles for this plan

1. **Catalog before ranking.** No ranking or evaluation work until Phase 1–2 give it real
   records. Metrics in the README are stale until Task 4.3 replaces them.
2. **Automation proposes, a person confirms.** Feeds, prefill, and re-verification write to an
   inbox. Only confirmed records (or records from a trusted structured feed) count as eligible.
3. **Deterministic core, offline always.** Stage 1–3 and evaluation stay pure functions of
   catalog plus profile. Network happens only in ingest and prefill. No LLM calls anywhere.
4. **Additive schema changes.** New record and profile fields are optional. Existing snapshots,
   golden profiles, and tests keep working after every task.
5. **Private data never enters git.** Student profiles, the tracker database, essays, and
   outcomes live under `data/private/` (git-ignored). A demo profile ships for anyone cloning.
6. **Student owns, parent oversees.** Parent mode adds curation, finances, and settings. It does
   not edit the student's essays.
7. **Green gate every task.** `python -m pytest tests/ -q` stays green and
   `ruff check src/ scripts/ app/ tests/` stays at 0 errors after every task.

---

# Phase 1 — Catalog and Profile Foundations

## Task 1.1: Curated Catalog Schema, Stable IDs, and Source

**Why:** The static feed is the only source producing usable records, and it has 5. It becomes
the primary data asset: one JSON file per award, committed to git (public information), with
provenance, status, cycle, requirements, and trust. A stable `catalog_id` slug replaces the
deadline-dependent hash so an award keeps its identity across years.

**Preflight Files:**
- `src/normalize/schema.py` (`NormalizedScholarshipRecord`, 22 fields, all additive changes go here)
- `src/normalize/canonical_id.py` (`generate_scholarship_id` hashes deadline — the continuity problem)
- `src/ingest/sources/static_feed.py` (`StaticFeedSource`, `_map_item` — the base for the new source)
- `src/ingest/registry.py` (`register_sources`)
- `src/io/snapshotting.py` (`prepare_snapshot_df`, `build_delta` — confirm new columns survive)
- `data/static_feed/scholarships.json` (5 records to migrate)
- `tests/test_static_feed_parsing.py`, `tests/conftest.py` (`sample_scholarship_df`, `scholarship_row_factory`)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python scripts/run_ingest.py --max-listing-pages 0 --max-detail-pages 0   # curated source only; snapshot has >= 5 records with new columns
```

**Checklist:**
- [x] Create `data/catalog/records/<catalog_id>.json`, one file per award, and
      `data/catalog/schema.json` (JSON Schema) describing: `catalog_id` (slug), `title`,
      `sponsor`, `source_url`, `description`, `eligibility_text`, `amount_min`, `amount_max`,
      `deadline`, `cycle` (`recurring`, `opens_month`, `deadline_month`), `status`
      (`open|upcoming|closed|unknown`), `education_level`, `grade_levels` (e.g. `["12"]`),
      `majors_allowed`, `states_allowed`, `counties_allowed`, `min_gpa`, `citizenship`,
      `need_based`, `first_gen_only`, `gender`, `heritage`, `military_family`, `disability`,
      `religion`, `employer_restricted`, `membership_required`, `min_test_scores`
      (`sat`/`act`), `requirements` (`essay`, `essay_prompts[]`, `recommendation_letters`,
      `transcript`, `fafsa`, `video_or_portfolio`, `interview`), `renewal_terms`,
      `trust` (`verified_local|structured_feed|aggregator|unverified`), `provenance`
      (`added_on`, `verified_on`, `verified_by`, `source_kind`), `notes`
- [x] Add the new optional fields to `NormalizedScholarshipRecord` and confirm
      `prepare_snapshot_df` / `build_delta` carry them (list and dict columns serialize)
- [x] Add a `catalog_id`-based ID path: `generate_scholarship_id` accepts `catalog_id` and, when
      present, hashes only that (no deadline); scrapers keep current behavior
- [x] Replace `StaticFeedSource` with `CuratedCatalogSource` (source name `curated_catalog`)
      reading every file in `data/catalog/records/`, validating against the schema, skipping and
      logging invalid files; register it first in `register_sources`
- [x] Migrate the 5 static-feed records into catalog files with `cycle`, `status`, `trust`,
      `requirements`, and `provenance` filled from their listings; delete `data/static_feed/`
- [x] `scripts/validate_catalog.py`: validates all records against the schema, checks slug
      uniqueness and URL shape; exit non-zero on any error; wired into CI
- [x] Tests: schema validation (valid, missing required, bad enum), stable ID across a deadline
      change, invalid file skipped without failing the source
- [x] Tests + ruff green

---

## Task 1.2: Student Profile Expansion, Private Storage, and Demo Profile

**Why:** Many awards are keyed on axes the profile does not have (need, first-gen, county,
heritage, gender, service hours, test scores, intended colleges). The profile also needs a
graduation year for the timeline, and the UI grade labels ("High School Senior") must map onto
the level vocabulary Stage 1 uses ("high school"). Multi-student is modeled now because it is
cheap now and painful later.

**Preflight Files:**
- `src/rank/stage1_eligibility.py` (`StudentProfile` dataclass)
- `src/types.py` (`ProfileLike` protocol)
- `app/main.py` (`_default_profile`, `_build_stage1_profile`, `_build_stage2_profile`, the
  sidebar `selectbox("Grade Level", ...)` options, `PROFILE_PATH`)
- `src/eval/golden_students.py` (`GoldenStudent`, must keep working unchanged)
- `src/rank/taxonomy.py` (`_EDUCATION_ADJACENCY`, `education_level_matches`)
- `.gitignore` (`student_profile.json` line; `data/private/` goes here)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
streamlit run app/main.py   # visual: demo profile loads; new fields render grouped; Run still works
```

**Checklist:**
- [x] Extend `StudentProfile` with optional fields: `student_id`, `graduation_year`,
      `grade_level` (`9|10|11|12|college_1..4`), `county`, `high_school`, `financial_need`,
      `first_gen`, `gender`, `heritage` (list), `military_family`, `disability`, `religion`,
      `parent_employers` (list), `memberships` (list), `service_hours`, `sat`, `act`,
      `intended_colleges` (list), `essay_ready` (bool)
- [x] Add `src/profile/store.py`: load/save `data/private/students/<student_id>.json`; list
      students; the app's `PROFILE_PATH` moves here; `data/private/` added to `.gitignore`
- [x] Ship `data/demo/student_demo.json` (fictional, committed) and load it when no private
      profile exists, with a visible "Demo profile" banner
- [x] Map UI grade labels to `education_level` + `grade_level` (e.g. "High School Senior" →
      `high school`, `12`); keep `key=` session-state bindings; `_build_stage2_profile` passes
      `extracurriculars` from the profile instead of `[]`
- [x] Group the sidebar form: Academic, About you (identity/need axes, each optional with a
      "prefer not to say" default), Activities, Colleges
- [x] Tests: profile round-trip, demo fallback, grade-label mapping, golden students untouched
- [x] Tests + ruff green

---

## Task 1.3: Stage 1 Rules for the New Axes, Status, and Unverified Flags

**Why:** With catalog fields and profile fields in place, Stage 1 can stop being a
state-plus-deadline filter. Each new axis gets a reason code. A missing profile value must not
silently pass: it passes but is recorded as unverified so the card can say "confirm you meet:
financial need".

**Preflight Files:**
- `src/rank/stage1_eligibility.py` (`_row_reasons`, `apply_eligibility_filter`)
- `src/rank/taxonomy.py` (`majors_match`, `education_level_matches`)
- `app/helpers.py` (`reasons_to_text`)
- `tests/test_eligibility_rules.py`, `tests/conftest.py`

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python scripts/evaluate_golden_students.py --k 10   # still runs; eligibility precision reported
```

**Checklist:**
- [x] New reason codes: `STATUS_CLOSED_NONRECURRING`, `COUNTY_NOT_ALLOWED`,
      `GRADE_LEVEL_MISMATCH`, `NEED_BASED_NOT_MET`, `FIRST_GEN_ONLY`, `GENDER_RESTRICTED`,
      `HERITAGE_RESTRICTED`, `MILITARY_FAMILY_ONLY`, `DISABILITY_RESTRICTED`,
      `RELIGION_RESTRICTED`, `EMPLOYER_RESTRICTED`, `MEMBERSHIP_REQUIRED`,
      `TEST_SCORE_BELOW_MIN`
- [x] `status == closed` with `cycle.recurring == false` and no future deadline → ineligible;
      closed-but-recurring passes Stage 1 and is handled by the timeline (Task 1.4)
- [x] A restriction the profile cannot answer (value `None`) passes and is appended to a new
      `unverified_axes` list column; `reasons_to_text` gains a companion
      `unverified_to_text`
- [x] `apply_eligibility_filter` returns the same `(eligible_df, ineligible_df)` shape; the
      new column is present on both
- [x] Tests: each new code fires and clears; unverified path; recurring-closed passes;
      existing tests unchanged
- [x] Tests + ruff green

**As built — contracts Task 1.4 depends on:**
- A recurring award (`cycle.recurring`, falling back to `is_recurring`) no longer emits
  `DEADLINE_PASSED` when its listed deadline is past: that date is a cycle date, not an expiry,
  so the row survives Stage 1 for the timeline to bucket. Unknown recurrence is unchanged, and
  only the curated catalog sets recurrence today — every scraper hardcodes `is_recurring: None`.
- `GRADE_LEVEL_MISMATCH` fires only when **every** value in `grade_levels` is behind the
  student's current grade. An award aimed at a later grade stays eligible as a future target.
  Measured on the 5-record catalog: a literal membership check passes a high-school senior on
  1 of 5 awards, because the 4 `college_1`–`college_4` awards are exactly what a senior applies
  to for freshman-year money.
- **Deviation, approved 2026-09-12:** `_compute_urgency_boost` clamped negative days to zero,
  scoring a past deadline `exp(0) = 1.0` — the maximum boost — while the card rendered
  "⏰ Passed". The first contract above makes past-deadline rows reachable in Stage 3, so the
  clamp was fixed here instead of in Task 1.4: a past deadline now scores 0.0 urgency.

---

## Task 1.4: Eligibility Timeline (Now / Next Cycle / Senior Year / Not Applicable)

**Why:** Nearly every award repeats on the same calendar. A sophomore's catalog is a plan for
junior and senior year, not a list of currently open links. This turns closed listings from
noise into future targets and gives the app its core view.

**Preflight Files:**
- `src/rank/stage3_rerank.py` (`_resolve_deadline`, `_compute_days_to_deadline`,
  `_compute_urgency_boost`, `rerank_stage3`)
- `src/rank/stage1_eligibility.py` (`StudentProfile.graduation_year`, `grade_level` from Task 1.2;
  the recurring and grade-level contracts under Task 1.3 "As built")
- `data/catalog/schema.json` (`cycle`, `grade_levels`, `status`)
- `app/main.py` (`_get_urgency_indicator`, `_render_scholarship_card`)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
```

**Checklist:**
- [x] Add `src/rank/timeline.py` with `classify_timeline(df, profile, today) -> df` adding
      `timeline_bucket` (`now|next_cycle|senior_year|expired|not_applicable`) and
      `projected_deadline` (from `deadline_month` and the next cycle year when the listed
      deadline has passed and `cycle.recurring` is true)
- [x] Bucket rules use `grade_levels` on the award against the student's grade in the cycle
      year (a "seniors only" award for a current sophomore → `senior_year`); awards the student
      has already aged out of never arrive here, since Stage 1 filters them as
      `GRADE_LEVEL_MISMATCH`, so `not_applicable` need not re-check that case
- [x] `rerank_stage3` accepts an optional `timeline_bucket` filter and, by default, ranks the
      `now` bucket; urgency uses `projected_deadline` when `deadline` is past — a past deadline
      already scores 0.0 urgency (Task 1.3 deviation), so what is left here is supplying the
      projected date so a recurring award is urgent against its *next* cycle rather than zero
- [x] Cards show the bucket and projected deadline instead of "Unknown deadline"
- [x] Tests: each bucket with hand-built rows; projection math across a year boundary;
      non-recurring closed → `expired`
- [x] Tests + ruff green

**As built — contracts later tasks depend on:**
- `classify_timeline` runs between Stage 1 and Stage 2, so `timeline_bucket` and
  `projected_deadline` ride through scoring on every row and are available on the ranked frame.
- Bucket precedence: `expired` (past or closed, not recurring) is terminal; otherwise the grade
  check overrides the deadline bucket, giving `senior_year` when the student's grade in the
  cycle year is behind every allowed grade, and `not_applicable` when they will have aged out
  by the next cycle (a grade-12 award whose next cycle falls after this senior graduates).
- `project_next_deadline` treats `cycle.deadline_month` as the authority on the month and takes
  the day from the listed deadline only when the months agree, otherwise the 1st; Feb 29 is
  clamped to the last day of the month in a common year.
- `rerank_stage3(timeline_bucket="now")` is the default, but the filter is a no-op on frames
  without the column, so every pre-timeline caller (tests, `evaluate_golden_students`,
  `tune_weights`) is unchanged. Pass `timeline_bucket=None` to rank every bucket.
- The app's new "Timeline" selectbox is read when **Run** is pressed, so switching buckets
  requires a re-run; the Phase 4 planning surface is expected to replace it.

---

# Phase 2 — Ingest Automation Without an LLM

## Task 2.1: Connector Health, Status Parsing, and Loud Zero-Record Failures

**Why:** Bold.org has returned 0 records since at least June and the report says `succeeded`.
Scholarship America emits closed programs and a junk "Scholarship Status" record. Silent
failure is worse than no connector.

**Preflight Files:**
- `scripts/run_ingest.py` (`run_ingest` source loop ~L400–475, status logic ~L555–600,
  `_normalize_records`)
- `src/ingest/sources/scholarship_america_live.py` (detail parsing; `_NON_DETAIL_TITLE_HINTS`)
- `src/ingest/sources/bold_org.py` (`_records_from_next_data`, `_records_from_html_cards`)
- `src/ingest/registry.py`
- `reports/ingest_runs/ingest_20260813T224157Z.json` (the current report shape)
- `tests/test_run_ingest.py`, `tests/test_source_parsing.py`, `tests/test_bold_org_parsing.py`

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python scripts/run_ingest.py --max-listing-pages 1 --max-detail-pages 20 --max-runtime-seconds 300   # report shows per-source counts and health
```

**Checklist:**
- [x] Scholarship America: parse `Status: Open|Closed` into `status`; drop records whose title
      is a page chrome string (extend `_NON_DETAIL_TITLE_HINTS` with "scholarship status");
      set `trust = aggregator`
- [x] Ingest report gains `health` per source: `records_this_run`, `records_prior_run`,
      `zero_record_regression` (true when prior > 0 and now == 0); any regression sets the
      source `status` to `failed` with `error = "zero_records"` and the run status to `partial`
- [x] `register_sources` reads an `enabled` flag per connector from
      `data/catalog/sources.json`; Bold.org ships `enabled: false` with a note until rewritten
- [x] Operator area in the app shows the health table and a warning banner on any regression
- [x] Tests: regression detection, disabled source skipped, status parsing, junk title dropped
- [x] Tests + ruff green

**As built — contracts later tasks depend on:**
- `data/catalog/sources.json` is the kill switch: `{"sources": {"<name>": {"enabled": bool,
  "note": str}}}`. A connector missing from the file is enabled, and an unreadable or malformed
  file enables everything — a broken config must never silently stop the ingest. `bold_org`
  ships `enabled: false`, so it is no longer in `attempted` and never trips a regression.
- `register_sources(config_path=None)` and the new `load_source_settings` /
  `disabled_sources` helpers all take an optional config path, so tests point at a tmp file.
- Per-source health lives at `sources.details[].health`; the run-level roll-ups are
  `sources.zero_record_regressions` (list of names), `sources.disabled` (`[{source, note}]`)
  and `sources.disabled_count`. A regression also appends a `guardrail_warnings` line.
- `records_prior_run` comes from the most recent readable `reports/ingest_runs/ingest_*.json`,
  matched by source name; it is `None` on a first run, which is never a regression.
- Regression demotes only the source (`status = "failed"`, `error = "zero_records"`); the run
  status still falls out of the existing derivation, so it is `partial` whenever any records
  survived and `failed` when none did.
- `_is_non_detail_title` is checked twice in `parse_detail_html` — once on the extracted
  `<title>` and again on `resolved_title`, because the live "Scholarship Status" page has no
  usable title and reaches the check only through the URL-slug fallback.
- Scholarship America records now carry `status` (`open|closed|upcoming|unknown`, or `None`
  when the page has no `Status:` label) and `trust = "aggregator"`, so Stage 1's existing
  closed-award rule now fires on live aggregator rows.

---

## Task 2.2: Open Scholarships Structured-Feed Connector

**Why:** The only openly licensed, machine-readable scholarship directory found. Coverage is
thin for NC today, but the connector is cheap, the license (CC BY 4.0) is clean, and the
schema maps almost one-to-one. It is also the model for contributing NC records back later.

**Preflight Files:**
- `src/ingest/base.py` (`BaseSource`, `RawResponse`)
- `src/ingest/sources/curated_catalog.py` (from Task 1.1; the mapping pattern)
- `src/ingest/http.py` (`PoliteHttpClient`)
- `README.md` (attribution goes in a new "Data sources" subsection)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python scripts/run_ingest.py --max-listing-pages 0 --max-detail-pages 0   # feed records appear with trust=structured_feed
```

**Checklist:**
- [x] `src/ingest/sources/open_scholarships.py`: fetch `/api/scholarships?state=NC` (and
      national records where the API exposes them), map fields including `availability` →
      `status` and `provenance.source_url`; `trust = structured_feed`
- [x] Verify the live endpoint and field names once by hand before writing the mapper; record
      the verified shape in a fixture JSON under `tests/resources/`
- [x] CC BY 4.0 attribution line in README "Data sources" and in the record `provenance`
- [x] Tests with the fixture: mapping, empty response, malformed response
- [x] Tests + ruff green

**As built — contracts later tasks depend on:**
- Base URL is `https://scholarships.grudged.io` (not `openscholarships.*`). Verified by hand
  2026-09-12: `/meta` gives the license and the required attribution string, `/healthz` the
  record count, `/api/scholarships` (also served at `/scholarships`) the query endpoint.
- Response envelope is `{total, limit, offset, license, attribution, results[]}`; `limit` is
  capped at 500 (422 above that) and `offset` pages. `?state=NC` returns NC records **plus**
  national ones — today that is 102 records, all national: the feed carries zero NC-specific
  awards (only `NV` appears in `/meta.states`), so this connector is national coverage for now.
- Record shape: `id`, `name`, `sponsor`, `sponsor_type`, `type`, optional `summary`, `award`
  (`amount_min`, `amount_max`, `currency`, `basis`, `renewable`, `notes`), `deadline` (`type`,
  `date`, `notes`, optional `opens`), `eligibility` (`residency`, `education_level`,
  `fields_of_study`, `gpa_min`, `citizenship`, `other`, `tags`), `geo` (`state`, `scope`,
  `counties`), `links`, `provenance`, `status`, `availability`, optional `review_flags`.
  Pinned in `tests/resources/open_scholarships_sample.json`.
- Mapping decisions that later tasks should not re-litigate: `residency = ["US"]` is dropped
  from `states_allowed` (a national award is not state-restricted, and keeping it would fail
  every record at Stage 1's state check); `education_level` is set only when the feed's list
  collapses to one Stage 1 level, otherwise `None`; `need_based` is `True` only for
  `award.basis == "need"` (`"merit-need"` stays `None`); `is_recurring`/`cycle.recurring` come
  from `deadline.type` (`annual`/`rolling` → true), not from `award.renewable`, which is a
  different question.
- Every record's `provenance` carries `source_url`, `license` and `attribution` alongside the
  curated-catalog keys, so the CC BY 4.0 credit travels with the data into the snapshot.
- `OpenScholarshipsSource(state=..., api_url=..., page_limit=..., max_pages=...)` is
  constructor-configurable so tests can drive it with a stub client and no network.
- `data/catalog/sources.json` gains `open_scholarships` (enabled). `register_sources()` now
  returns 4 connectors — `tests/test_source_registry.py` asserts that count.

---

## Task 2.3: Deterministic URL Prefill and the Extractor Interface

**Why:** Hand entry is the backbone, so it must be fast. Paste a URL, get a mostly-filled form.
The regex helpers already exist in two connectors; they move to one shared module. The
`Extractor` protocol is the seam where the LLM extractor plugs in later without touching the
form or the queue.

**Preflight Files:**
- `src/ingest/sources/bold_org.py` (`_MONEY_PATTERN`, `_ISO_DATE_PATTERN`, `_LONG_DATE_PATTERN`,
  `_strip_html`, `_extract_amount`, `_parse_deadline`)
- `src/ingest/sources/scholarship_america_live.py` (`_US_STATES_AND_TERRITORIES`,
  `_LABEL_BLOCK_PATTERN`, `_SCRIPT_STYLE_PATTERN`)
- `src/ingest/http.py` (`PoliteHttpClient.get`)
- `src/llm/extraction.py` (the field contract the LLM extractor already honors — the protocol
  must be satisfiable by it)
- `src/text_utils.py`

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -c "from src.ingest.prefill import prefill_from_url; print(prefill_from_url('https://www.nctech.org/about/education-foundation/scholarships.html'))"
```

**Checklist:**
- [x] `src/ingest/extract_common.py`: shared money, date (ISO, US, long-form), state,
      county (NC list), essay/recommendation/transcript keyword patterns, HTML-to-text
- [x] Both scrapers import from it; behavior-preserving (existing parsing tests unchanged)
- [x] `src/ingest/prefill.py`: `Extractor` protocol
      (`extract(title, text) -> ExtractionResult`), `RegexExtractor` implementation, and
      `prefill_from_url(url, client) -> PrefillResult` returning `title` (og:title / `<title>`),
      `sponsor` (og:site_name), `description` (meta), amount candidates, date candidates,
      state/county hits, requirement flags, the cleaned page text, and per-field confidence
- [x] Never guess: fields with no evidence are `None`; multiple date candidates are returned
      as a list for the form to choose from
- [x] Tests with fixture HTML (a foundation page, an aggregator page, a page with no signals)
- [x] Tests + ruff green

---

**As built — contracts later tasks depend on:**
- `src/ingest/extract_common.py` owns every shared pattern and helper. Public names (no
  leading underscore): `TAG_PATTERN`, `WS_PATTERN`, `SCRIPT_STYLE_PATTERN`, `MONEY_PATTERN`,
  `ISO_DATE_PATTERN`, `US_DATE_PATTERN`, `LONG_DATE_PATTERN`, `LABEL_BLOCK_PATTERN`,
  `GPA_PATTERN`, `US_STATES_AND_TERRITORIES`, `NC_COUNTIES`, and the functions
  `strip_html`, `html_to_text`, `meta_content`, `tag_text`, `extract_amounts`,
  `extract_amount_range`, `parse_date_candidates`, `parse_first_date`,
  `extract_field_value`, `find_states`, `find_nc_counties`, `find_min_gpa`,
  `find_requirement_flags`.
- `strip_html` and `html_to_text` are different on purpose: only `html_to_text` drops
  `<script>`/`<style>` bodies first. Page-level parsing must use `html_to_text`, or tracking
  JSON leaks dates and dollar figures into the extraction.
- `parse_first_date` is format-priority (ISO > `m/d/Y` > long form, regardless of position) and
  is what the two scrapers use; `parse_date_candidates` is position-ordered, deduped, and
  returns every date, and is what prefill uses so the form can offer a choice.
  `LONG_DATE_PATTERN` now treats the comma as optional for both callers.
- `find_states` matches full state names only --- never two-letter codes, because `IN`, `OR`,
  `ME` and `OK` are ordinary words. `find_nc_counties` requires the literal word "County" and
  returns the bare name (`"Wake"`), matching Stage 1's `_normalize_county` comparison.
- `find_requirement_flags(text) -> (requirements, evidence)`. `requirements` is shaped like
  the catalog's `requirements` object and carries only keys with positive evidence --- an
  absent key means "the page did not say", never `False`. A requirement negated earlier in its
  own sentence ("No essay is required") is dropped from both dicts.
  `recommendation_letters` is an int only when a count is stated; a countless mention appears
  in `evidence` alone, so the form asks a person for the number.
- `src/ingest/prefill.py` exposes `Extractor` (a `runtime_checkable` Protocol with `name` and
  `extract(title, text) -> ExtractionResult`), `RegexExtractor` (`name = "regex"`),
  `ExtractionResult`, `PrefillResult`, `prefill_from_html(html, *, url, extractor=None)` and
  `prefill_from_url(url, client=None, *, extractor=None)`.
- `ExtractionResult.deadline` is `None` unless there is exactly one candidate --- ambiguity is
  never resolved by picking the first. Same idea for `confidence`: a field missing from that
  dict was not extracted at all.
- Confidence is a function of *how* the evidence was found, not of how plausible the value
  looks: labeled + single candidate `0.85`, labeled + several `0.6`, unlabeled + single `0.6`,
  unlabeled + several `0.35`; `og:title` `0.9`, `<h1>` `0.75`, `<title>` `0.6`; `og:site_name`
  sponsor `0.7`; meta description `0.8`; labeled state `0.7` vs. body-scan state `0.35`;
  county `0.6`; GPA `0.6`; each requirement `0.6` under the key `requirements.<name>`.
- `prefill_from_url` never raises: a fetch failure returns a `PrefillResult` with `error` set
  and every field empty, so a bad paste opens a blank form. `client` is anything with
  `get_text(url) -> str`; a `PoliteHttpClient` is built and closed automatically when omitted.
- `PrefillResult.to_form_dict()` is the flat payload Task 2.4's confirm queue consumes. It
  carries both `deadline_candidates` and the single `deadline`, plus `amount_candidates`,
  `requirements`, `requirement_evidence`, `confidence`, `extractor` and `error`.
- `extraction_result_from_fields(fields, *, confidence=0.5)` converts the validated mapping
  from `src/llm/extraction.py::parse_extraction` into an `ExtractionResult`. That is the proof
  the protocol is satisfiable by the LLM extractor --- it plugs in by passing `extractor=` to
  `prefill_from_url`, with no change to the form or the queue.
- Both scrapers now import from `extract_common` and their 17 existing parsing tests pass
  unchanged. `bold_org._extract_amount` keeps its bare-number fallback (a Bold.org JSON field
  may hold `1000` rather than `"$1,000"`); `scholarship_america_live` keeps `_to_text`,
  `_extract_field_value`, `_extract_deadline`, `_extract_amount_range` and `_extract_states`
  as thin aliases so its call sites and tests are untouched.

## Task 2.4: Inbox and Confirm Queue

**Why:** This is the rule that keeps automation from polluting the catalog: everything
automated proposes, a person confirms. Feeds, prefill, and re-verification all write proposals
here. Only `records/` (plus `structured_feed` trust) enters the eligible catalog.

**Preflight Files:**
- `data/catalog/schema.json`, `src/ingest/sources/curated_catalog.py` (Task 1.1)
- `src/ingest/prefill.py` (Task 2.3)
- `scripts/run_ingest.py` (where feed records are normalized)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python scripts/catalog_inbox.py list
```

**Checklist:**
- [x] `data/catalog/inbox/<proposal_id>.json`: a partial record plus `proposal` metadata
      (`kind: prefill|feed|reverify|manual`, `created_on`, `diff` against an existing record
      when applicable)
- [x] `src/catalog/inbox.py`: `propose()`, `list_proposals()`, `confirm(proposal_id, edits)`
      (validates against the schema, writes to `records/`, removes the proposal),
      `reject(proposal_id, reason)`
- [x] `scripts/catalog_inbox.py` CLI: `list`, `show`, `confirm`, `reject`
- [x] `CuratedCatalogSource` ignores `inbox/`; ingest never promotes proposals on its own
- [x] Tests: propose → confirm round-trip, schema failure on confirm, reject, inbox ignored by
      the source
- [x] Tests + ruff green

---

**As built — contracts later tasks depend on:**
- `src/catalog/inbox.py` exposes `INBOX_DIR` (`data/catalog/inbox/`), `REJECTED_DIR`
  (`inbox/rejected/`), `PROPOSAL_KINDS`, `Proposal`, `ProposalError`, `propose`,
  `list_proposals`, `load_proposal`, `confirm` and `reject`. Every function takes
  `inbox_dir` / `records_dir` overrides, so tests and the UI never touch the real catalog.
- A proposal file is `{"proposal": {proposal_id, kind, created_on, diff, notes}, "record": {...}}`.
  `record` is a *partial* catalog record --- `propose` deliberately does not schema-validate,
  because automation is allowed to propose something incomplete and a person fills the gaps.
- `confirm(proposal_id, edits)` is the only path into `records/`. `edits` is merged over the
  proposed record at the top level (nested objects like `requirements` are replaced whole,
  not deep-merged), the result must pass `validate_catalog_record`, and on failure nothing is
  written and the proposal stays in the queue. A `reverify` proposal that carries only the
  changed fields therefore needs the full record supplied as `edits` at confirm time.
- Proposal ids are deterministic: `f"{kind}-{slug}"` from `catalog_id`, else `title`, else
  `source_url`. Re-proposing the same award **replaces** its pending proposal rather than
  stacking duplicates, which is what makes a monthly re-verification pass safe to re-run.
  Any id --- supplied or derived --- must match the catalog slug pattern, so a proposal can
  never be written or read outside the inbox directory.
- `diff` compares only the fields the proposal carries against the existing record, in the
  `{field: {"old": ..., "new": ...}}` shape `build_delta` already uses. An empty diff means a
  brand-new award or a proposal that changes nothing.
- `reject(proposal_id, reason)` requires a non-empty reason and archives the proposal to
  `inbox/rejected/<proposal_id>.json` with `rejected_on` and `reason` added, rather than
  deleting it --- so a later re-verification pass can see a person already said no.
- `list_proposals` logs and skips an unreadable or malformed file so one bad write never
  hides the queue.
- The inbox is a *sibling* of `records/`, and `iter_catalog_files` globs `records/*.json`
  non-recursively, so `CuratedCatalogSource` and `scripts/validate_catalog.py` never see a
  proposal. Nothing in the ingest path calls `confirm`.
- `scripts/catalog_inbox.py` has `main(argv)` (so it is testable) and the subcommands `list`,
  `show`, `confirm`, `reject`, plus `--inbox-dir` / `--records-dir`. `confirm` takes
  `--set FIELD=VALUE` (value parsed as JSON when possible, else kept as text) and
  `--edits <file.json>`; `--set` wins over the file. A `ProposalError` prints `ERROR: ...`
  and exits 1.
- `data/catalog/inbox/*.json` and `inbox/rejected/` are git-ignored (the directory is kept by
  a `.gitkeep`): proposals are local working state, not a committed asset.

---

## Task 2.5: Annual Re-verification

**Why:** Deadlines move, awards close, links die. Re-fetching each record's URL on a schedule
and diffing is the automation that keeps a curated catalog honest year over year.

**Preflight Files:**
- `src/catalog/inbox.py` (Task 2.4), `src/ingest/prefill.py` (Task 2.3)
- `src/io/snapshotting.py` (`build_delta` — reuse the diff style)
- `src/ingest/http.py`

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python scripts/verify_catalog.py --max-records 5
```

**Checklist:**
- [x] `scripts/verify_catalog.py`: for each record, fetch `source_url`, compute a content hash
      of the cleaned text, detect dead link (4xx/5xx), detect closed/open status text, run the
      prefill extractor and compare `deadline`, amounts, and requirement flags to the record
- [x] Unchanged → update `provenance.verified_on`; changed → write an inbox proposal of kind
      `reverify` with the diff; dead link → proposal with `status: unknown`
- [x] `--since-days` and `--max-records` flags; polite rate limit; summary report under
      `reports/catalog_verify/`
- [x] `docs/operations.md`: how to schedule it (Windows Task Scheduler entry, monthly)
- [x] Tests with a fake client: unchanged, changed deadline, dead link
- [x] Tests + ruff green

**As built — contracts later tasks depend on:**
- Logic lives in `src/catalog/verify.py`; `scripts/verify_catalog.py` is a thin CLI with
  `main(argv)`. `verify_catalog(...)` takes `records_dir`, `inbox_dir`, `reports_dir`, `client`,
  `extractor`, `since_days`, `max_records`, `today`, `requests_per_second`, `write_report`, so
  tests drive the whole pass with a stub client and tmp dirs.
- Four outcomes: `unchanged` (stamp `provenance.verified_on` + `verified_by: verify_catalog`),
  `changed` (a `reverify` proposal carrying the merged record), `dead_link` (4xx/5xx --- a
  proposal with `status: unknown`), and `error` (no HTTP status: SSL failure, timeout, offline
  laptop). Only `dead_link` proposes; an unreachable host is reported, never queued.
- A change needs a *labeled* value: `deadline` and the amounts count only at
  `MIN_CHANGE_CONFIDENCE` (0.7), which `RegexExtractor` awards to a single value read from a
  `Deadline:`/`Amount:` label. This is what stops a sponsor's "$450,000 awarded since 1998"
  from rewriting `amount_max`. A labeled value does overwrite a `null` field.
- Requirement flags count only when the record states a value and the page contradicts it;
  a flag the record leaves `null` is never filled, because those regexes are word-presence
  matches, not labeled fields. `status` comes from a `Status:` label or unambiguous
  closed/open wording --- never a bare "open" in body copy.
- `_stamp_verified_on` patches the two provenance values *as text* so a stamp does not reflow
  a hand-formatted record; it falls back to a full rewrite only when that patch does not apply.
  `trust` is never raised --- `verified_by: verify_catalog` is a machine check, not a person.
- Proposal ids are `reverify-<catalog_id>`, so a monthly pass replaces its own pending
  proposal for an award instead of stacking duplicates.
- Reports: `reports/catalog_verify/catalog_verify_<UTC stamp>.json` (git-ignored), carrying
  per-record outcome, HTTP status, `content_hash`, and diff. The next run reads the newest
  readable report for prior hashes and reports `content_changed`, which flags a page that was
  edited without any extracted field moving.
- Scheduling and the queue workflow are documented in `docs/operations.md`; exit status is 0
  for any completed pass (dead links are output, not failure) and 1 only on an empty catalog.

---

# Phase 3 — The Family Product

## Task 3.1: Persistence Layer (SQLite)

**Why:** Session state plus one JSON file cannot hold a tracker, essays, recommenders, or
outcomes. SQLite from the standard library, one file under `data/private/`, no server.

**Preflight Files:**
- `src/profile/store.py` (Task 1.2)
- `app/main.py` (`_ensure_session_state` — what currently lives in session state)
- `.gitignore`

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
```

**Checklist:**
- [x] `src/store/db.py`: connection factory for `data/private/coach.db`, schema migrations as
      numbered SQL files under `src/store/migrations/`, applied on open
- [x] Tables: `students`, `applications` (`student_id`, `catalog_id`, `status`, `notes`,
      `created_at`, `updated_at`), `checklist_items`, `essays`, `essay_links`
      (award prompt ↔ essay), `recommenders`, `recommendation_requests`, `outcomes`
      (`amount_awarded`, `paid_to`, `renewal_terms`), `colleges`, `settings`
- [x] `src/store/repo.py`: typed functions per table; no SQL in the app layer
- [x] Tests on a temp database: migrations idempotent, CRUD per table, cascade on student delete
- [x] Tests + ruff green

---

## Task 3.2: Student Mode and Parent Mode

**Why:** One profile, two views. Student mode is what is due and what to write. Parent mode adds
curation, finances, outcomes, and settings. A trusted household starts with a toggle; a PIN is a
settings change later.

**Preflight Files:**
- `app/main.py` (`main()` layout; the `Advanced / Operator` expander pattern from UI Plan Task 3)
- `src/store/repo.py` (Task 3.1)
- `.streamlit/config.toml`, `.gitignore` (`.streamlit/secrets.toml` already ignored)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
streamlit run app/main.py   # visual: mode switch; parent-only sections hidden in student mode
```

**Checklist:**
- [x] `app/modes.py`: `Mode = student|parent|operator`; a sidebar selector; `operator` keeps
      today's Advanced expander and is hidden unless `settings.operator_enabled`
- [ ] Student view: This Week, My Applications, Essays, Recommenders
- [ ] Parent view: everything in student view read-only for essays, plus Catalog & Inbox,
      Timeline, Colleges & Money, Outcomes, Settings
- [x] Optional PIN for parent mode read from `.streamlit/secrets.toml`; absent → no PIN
- [x] Tests: mode gating helpers (pure functions), PIN check
- [x] Tests + ruff green

---

## Task 3.3: Application Tracker

**Why:** The core coaching loop: save an award, plan it, work it, submit it, record the result.

**Preflight Files:**
- `app/main.py` (`_render_scholarship_card`)
- `src/store/repo.py`, `app/modes.py`
- `data/catalog/schema.json` (`requirements` → checklist template)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
streamlit run app/main.py   # visual: Save on a card creates an application; checklist auto-populates
```

**Checklist:**
- [ ] Card gains a Save button; saved awards appear under My Applications with status
      `saved|planning|in_progress|submitted|won|lost|skipped`
- [ ] Checklist items generated from the award's `requirements` (essay per prompt, letters
      count, transcript, FAFSA, video/portfolio, interview) plus free-form items
- [ ] Notes per application; submitted date; outcome entry writes to `outcomes`
- [ ] This Week view: checklist items and deadlines within 14 days across all applications
- [ ] Tests: checklist generation from requirements; status transitions; This Week query
- [ ] Tests + ruff green

---

## Task 3.4: Essay Bank and Recommenders

**Why:** Reuse is the coaching skill. A few themed essays cover most prompts, and letters need
asking early with their own deadlines.

**Preflight Files:**
- `src/store/repo.py` (`essays`, `essay_links`, `recommenders`, `recommendation_requests`)
- `app/modes.py`

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
```

**Checklist:**
- [ ] Essays: title, theme tags (`challenge|leadership|why_major|community|identity|other`),
      body, word count (computed), version history kept as rows
- [ ] Award prompts link to essays; a prompt with no linked essay shows as an open checklist
      item; an essay reused N times shows N
- [ ] Recommenders: name, role, email; requests per application with `asked_on`,
      `due_on`, `received_on`; This Week includes recommender due dates
- [ ] Student mode edits essays; parent mode reads them
- [ ] Tests: word count, link/unlink, reuse count, request lifecycle
- [ ] Tests + ruff green

---

## Task 3.5: Timeline View, Milestones, and Calendar Export

**Why:** The bucket model from Task 1.4 becomes the planning surface, and the family's real
calendar gets the dates.

**Preflight Files:**
- `src/rank/timeline.py` (Task 1.4)
- `src/store/repo.py`
- `app/modes.py`

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
```

**Checklist:**
- [ ] `data/milestones.json` (committed, general): FAFSA opening, CSS Profile, common early
      action / early decision windows, PSAT window, with `grade_level` applicability; editable
      per family in `settings`
- [ ] Timeline page: month grid by school year with award projected deadlines, application
      due dates, recommender due dates, and milestones; buckets `now|next_cycle|senior_year`
      as tabs
- [ ] ICS export of saved applications and milestones (`scripts/export_calendar.py` and a
      download button)
- [ ] Tests: ICS content, milestone applicability by grade
- [ ] Tests + ruff green

---

## Task 3.6: Family-Facing Ranking (Hide the Win Model, Show Effort and Trust)

**Why:** A teen will read "52% chance of winning" as a fact, and it is synthetic. Fit,
deadline, award size, effort, and pool size are the honest signals.

**Preflight Files:**
- `app/main.py` (`_render_scholarship_card`, `_topk_win_model_summary`, the `use_win_model`
  checkbox)
- `app/helpers.py` (`explain_ranked_row`, `unverified_to_text` from Task 1.3)
- `src/rank/stage3_rerank.py` (`_compute_effort_cost`, `rerank_stage3`)
- `src/rank/weights.py` (`Stage3Weights`)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python scripts/evaluate_golden_students.py --k 10   # ranking still deterministic
```

**Checklist:**
- [ ] `p_win`, `expected_value`, and the Win Model Summary render only in operator mode;
      `use_win_model` defaults off and is operator-only
- [ ] Effort shown as counts: essays (from `requirements.essay_prompts`), letters, extras;
      `_compute_effort_cost` uses those counts when present, falling back to `essay_required`
- [ ] `Stage3Weights` gains `local_boost` applied to `trust == verified_local` and to
      `counties_allowed` / `states_allowed` restricted awards (small pools); default small,
      documented
- [ ] `explain_ranked_row` adds "Local award, smaller applicant pool" and "N essays, M letters"
      lines; drops the expected-value line outside operator mode
- [ ] Unverified axes from Task 1.3 render as "Confirm you meet: …" on the card
- [ ] Tests: effort counts, local boost, explanation lines
- [ ] Tests + ruff green

---

## Task 3.7: Colleges and Money View

**Why:** Outside scholarships are the small lever. The college list with net price estimates
and each school's outside-award policy keeps the family looking at the whole picture.

**Preflight Files:**
- `src/store/repo.py` (`colleges`, `outcomes`)
- `app/modes.py` (parent view)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
```

**Checklist:**
- [ ] Colleges table UI: name, in-state flag, sticker price, net price estimate (manual entry
      from the school's calculator), merit aid notes, outside-award displacement policy note,
      application deadline type
- [ ] Money summary: total won (from `outcomes`), by year, renewal conditions due, and net
      price minus won per college
- [ ] Parent mode only
- [ ] Tests: summary math
- [ ] Tests + ruff green

---

## Task 3.8: Catalog Entry and Inbox UI

**Why:** The front door for data. Paste a URL, review the prefill, confirm. Review feed and
re-verification proposals in the same place.

**Preflight Files:**
- `src/ingest/prefill.py` (Task 2.3), `src/catalog/inbox.py` (Task 2.4)
- `src/rank/taxonomy.py` (`_MAJOR_FAMILIES` → dropdown options)
- `src/ingest/extract_common.py` (state and county lists → dropdown options)
- `app/modes.py` (parent view)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
streamlit run app/main.py   # visual: paste URL → prefilled form → confirm → award appears after re-ingest
```

**Checklist:**
- [ ] Add Award page: URL input → prefill → form with dropdowns for level, grade levels,
      major families, states, NC counties, requirement flags, cycle months, trust; date and
      amount candidates offered as choices; manual entry works with no URL
- [ ] Inbox page: proposals listed by kind with diff view; Confirm (with edits) / Reject
- [ ] After confirm, a "Rebuild snapshot" button runs the curated source only (no scrapers)
      so the award is rankable immediately
- [ ] Tests: form → record validation path (pure function), diff rendering helper
- [ ] Tests + ruff green

---

# Phase 4 — Hosting, What-If, and Portfolio Reconciliation

## Task 4.1: Family Hosting

**Why:** The student will use this on a phone. It needs to be reachable at home without being
public.

**Preflight Files:**
- `.streamlit/config.toml`, `app/modes.py` (PIN), `docs/operations.md` (Task 2.5)
- `README.md` (Quick Start)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
streamlit run app/main.py --server.address 0.0.0.0   # reachable from a phone on the home network
```

**Checklist:**
- [ ] Decide and document one of: home PC on the LAN with the parent PIN, or a private
      network overlay (Tailscale) for off-network access; no public deployment
- [ ] Phone-width pass on student view: cards, This Week, essay editor usable at ~400px
- [ ] `docs/operations.md`: start on boot, backups of `data/private/` (zip to a second
      location weekly), restore steps
- [ ] Tests + ruff green

---

## Task 4.2: What-If Profile View

**Why:** Sophomore year is for building the record. "What would a 3.5 GPA or 100 service hours
unlock" is decision support the pipeline can nearly give already, and it is the one genuinely
new analytic feature in this plan.

**Preflight Files:**
- `src/rank/stage1_eligibility.py`, `src/rank/timeline.py`
- `app/modes.py` (both views)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
```

**Checklist:**
- [ ] `src/rank/whatif.py`: `whatif_eligibility(df, profile, overrides) -> summary` returning
      newly eligible awards, newly ineligible, and total award dollars unlocked, by bucket
- [ ] UI: sliders/inputs for GPA, SAT/ACT, service hours, first-gen/need flags; shows the
      delta list with reasons cleared
- [ ] Tests: GPA threshold unlock, test-score unlock, no-op override
- [ ] Tests + ruff green

---

## Task 4.3: Evaluation and README Reconciliation

**Why:** The README's metrics were measured on a catalog that no longer exists, and the
project's framing is now "a decision-support product with a recommender inside." The honest
portfolio story is the measured pivot.

**Preflight Files:**
- `README.md` (Evaluation Results, Limitations, Win Probability Model, Future Work, Author)
- `docs/evaluation.md`, `docs/system_design.md`
- `scripts/evaluate_golden_students.py`, `scripts/make_labeling_worksheet.py`,
  `data/eval/human_labels.csv`
- `docs/plans/SCHOLARSHIPCOACH_UI_REDESIGN_PLAN.md` (Task 6 screenshots run after this)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python scripts/evaluate_golden_students.py --k 10 --similarity-mode embeddings --label-mode hybrid --cross-label-check
python scripts/evaluate_golden_students.py --k 10 --human-labels data/eval/human_labels.csv
```

**Checklist:**
- [ ] Re-run golden eval on the curated catalog once it passes 100 confirmed records; replace
      README metric tables and catalog-size notes; keep the historical rows labeled as such
- [ ] Generate a labeling worksheet for the student's real profile and have her label her
      top 20; add to `human_labels.csv`; report human NDCG for her profile
- [ ] README repositioning: product summary first, architecture second, "Data sources" with
      attribution, "Family product vs. portfolio library" section, win model reframed as a
      retired-from-product calibration demo
- [ ] `docs/decisions.md`: dated log starting with the 2026-09-12 review findings and each
      scope decision in this plan
- [ ] `docs/system_design.md`: catalog, inbox, timeline, store, and modes sections
- [ ] Tests + ruff green

---

## Task 4.4: Outcomes as Evaluation Data

**Why:** Real won/lost outcomes are the labels the project has never had. Logging them from
day one makes the win model, or its replacement, honest later.

**Preflight Files:**
- `src/store/repo.py` (`outcomes`, `applications`)
- `src/win_model/features.py` (`build_pair_features` — the feature contract outcomes must join to)
- `docs/evaluation.md`

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python scripts/export_outcomes.py
```

**Checklist:**
- [ ] `scripts/export_outcomes.py`: writes `data/private/eval/outcomes.csv` with
      `student_id`, `catalog_id`, cycle year, submitted flag, result, amount, plus the pair
      features at submission time
- [ ] `docs/evaluation.md`: "Real outcomes" section — what is logged, minimum count before
      any model is trained on it, and how it would replace synthetic labels
- [ ] Tests: export shape on a temp database
- [ ] Tests + ruff green

---

# Phase 5 — Future Enhancements (not scheduled)

Recorded so they are not lost. None are on the path to senior fall.

- **LLM extractor behind the `Extractor` protocol** (LLM Plan Tasks 6, 8, 9): paste-in
  prefill and feed enrichment with a paid provider; the confirm queue and cache already
  fit. Reactivate the LLM plan from Task 8 if this is picked up.
- **Bold.org and CFNC listing crawls** as best-effort connectors behind the health check and
  a per-site terms-of-use review.
- **Match-email ingestion**: per-sender parsers for Bold.org / BigFuture digest emails from an
  exported mailbox folder; a live mailbox integration only if the family wants it.
- **Application portfolio selection**: choose which applications to spend limited hours on
  given effort estimates from the essay bank and pool-size proxies (the README's knapsack
  future work, now with real inputs).
- **Contribute verified NC records to Open Scholarships** under its schema and provenance
  rules.
- **Second student** onboarding flow, when relevant.

---

## Execution Order

```
Phase 1  1.1 catalog schema + IDs → 1.2 profile → 1.3 Stage 1 axes → 1.4 timeline
Phase 2  2.1 health + status → 2.3 prefill → 2.4 inbox → 2.5 re-verify   (2.2 feed any time after 1.1)
Phase 3  3.1 store → 3.2 modes → 3.3 tracker → 3.8 entry UI → 3.4 essays → 3.5 timeline UI
         → 3.6 family ranking → 3.7 colleges
Phase 4  4.1 hosting → 4.2 what-if → 4.3 eval + README (then UI Plan Task 6 screenshots) → 4.4 outcomes
```

Phase 1 and 2 are the catalog. The student can start using the app after Task 3.3 (tracker)
plus 3.8 (entry UI); everything after that improves a working tool.

## Success Criteria

1. A person can paste a URL, confirm a prefilled record, and see the award ranked in the
   student's `now` or `next_cycle` bucket within two minutes, with no LLM and no scraper.
2. A connector that returns zero records where it previously returned some fails the ingest
   run visibly. No closed, non-recurring award ranks as eligible.
3. The student's view answers "what is due this week and what do I need to write" from a phone,
   and nothing private is in git.
4. The README's metrics describe the catalog that exists, the student's own labels are part of
   the evaluation, and the win model is not visible outside operator mode.
5. `python -m pytest tests/ -q` green and `ruff check` at 0 after every task.
