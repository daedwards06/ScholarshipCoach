# ScholarshipCoach Catalog Integrity Plan — From a Green Checklist to a Catalog the Family Can Trust

> Generated: 2026-09-13 | Scope from a post-implementation review of the Family Product Plan
> Companion to `SCHOLARSHIPCOACH_FAMILY_PRODUCT_PLAN.md` (all 19 tasks executed, 4 CI gates green)
> Executor: Claude via Claude Code, with three owner-dependent data tasks called out as such
> Est. effort: 3 phases, 13 tasks

---

## Why this plan exists

The Family Product Plan was reviewed on 2026-09-13 against the code, the current snapshot
(`scholarships_snapshot_20260912.parquet`, 107 records), the local database, and an offline
run of the in-app rebuild. Every task is checked off and `pytest`, `ruff`, `mypy`, and
`validate_catalog.py` are green. The gaps are in the data and in the seams between tasks.

| Finding | Evidence |
|---|---|
| "Rebuild snapshot" deletes the feed records | Curated-only run in a temp dir wrote a **5-row** snapshot, delta `removed: 102`; the >50% guardrail warned and wrote anyway |
| The trust rule is documented, not enforced | README and design principle 2 say only confirmed or trusted-feed records count as eligible; Stage 1 never reads `trust` |
| All 5 curated records are `trust: unverified`, `status: unknown` | Migrated in Task 1.1, never hand-confirmed; nothing in the catalog qualifies for the Task 3.6 local boost |
| Outcomes is a placeholder page | Listed in the parent nav, routes to `_render_pending_section` |
| Cross-source duplicate | "Dell Scholars Program" (feed) and "Dell Technologies Scholars Program" (curated) both rank |
| HTTP 403 is treated as a dead link | The one inbox proposal proposes `status: unknown` for a sponsor page that blocked the fetch |
| Zero local or NC-specific awards | All 102 feed records are national; no record has a county |
| The Now bucket is mostly unknowns | Demo student: 93 of 101 eligible land in `now`; 79 of those are `status: unknown` with no deadline |
| Stage 1 barely filters a real profile | 6 of 107 rejected for the demo student |
| Scholarship America is enabled but fetches nothing | `max_listing_pages 0` every run; the zero-record regression warning fires every run |
| The family has not used it | No `data/private/students/`; `coach.db` has 0 applications, essays, recommenders, outcomes |
| `app/main.py` is 2,850 lines | Every section inline; the router is the last 40 lines |

**Decisions taken:**
- Fix the destructive defect first. Nothing else in this plan is worth doing on a snapshot
  that a button can wipe.
- Confirm the five records by hand before enforcing trust, so enforcement does not empty the
  catalog on the day it lands.
- Local awards are seeded through the Add Award page, not by hand-writing JSON, so the seeding
  exercises the product's own front door.
- Owner-dependent tasks (confirming records, seeding local awards, onboarding the student) are
  written as tasks with a Claude half and an owner half. Claude drafts and validates; the owner
  confirms. Nothing gets `trust: verified_local` because Claude read a web page.
- No new features. Every task here either fixes a defect, fills a data gap, or removes debt
  that blocks the first two.

**Relationship to other plans:**
- Family Product Plan Task 3.2 has two unchecked items that Task 3.3 here closes.
- Family Product Plan Task 4.3's blocked item (student labels) is Task 2.3 here.
- UI Redesign Plan Task 6 (README screenshots) runs from Task 3.3 here, after Task 3.1.

---

## Design principles for this plan

1. **A rebuild never loses a record it did not re-fetch.** Sources that did not run carry their
   prior rows forward. A record count collapse refuses to write unless forced.
2. **Trust is a Stage 1 axis.** `unverified` and `aggregator` records are ineligible with a
   reason code, the same as a wrong state. `verified_local` and `structured_feed` pass.
3. **Unknown is not Now.** An award with no deadline, no cycle month, and `status: unknown`
   needs a date before it competes for the student's attention.
4. **A blocked fetch is not a dead link.** 403, 429, and 5xx mean "check by hand", never a
   status change proposal.
5. **Owner confirms, Claude drafts.** Same as the Family Product Plan's principle 2, applied to
   the plan's own data work.
6. **Green gate every task.** All four CI commands after every task; the app-visual command
   where the task touches the UI.

---

# Phase 1 — Stop the Bleeding

## Task 1.1: Rebuild Carries Forward, Guardrail Refuses

**Why:** The "Rebuild snapshot" button runs the curated source alone and the snapshot writer
builds only from the records that ran. Confirming one award in the app replaces the 107-record
snapshot with a 5-record one. The >50% drop guardrail logs a warning and writes anyway, which is
the exact failure mode the Family Product Plan opened with.

**Preflight Files:**
- `scripts/run_ingest.py` (`run_ingest` `only_sources` at ~L499, `_build_guardrail_warnings` at ~L362, the write block at ~L610–L680)
- `src/io/snapshotting.py` (`build_and_write_snapshot`, `find_prior_snapshot`, `prepare_snapshot_df`, `build_delta`)
- `app/main.py` (`_render_rebuild_snapshot` at ~L2502)
- `tests/test_run_ingest.py`, `tests/test_snapshotting.py`

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python scripts/validate_catalog.py
python -c "from scripts.run_ingest import run_ingest; r = run_ingest(only_sources=['curated_catalog']); print(r['records']['snapshot_total'], r['delta_counts'])"   # expect 107, removed 0
```

**Checklist:**
- [x] `run_ingest(only_sources=...)`: after normalizing the records that ran, load the prior
      snapshot and append every prior row whose `source` is not in `only_sources` and whose
      source is still enabled; the report gains `carried_forward: {source: count}`
- [x] `_build_guardrail_warnings` result gains a `blocking` flag; a >50% drop is blocking
      unless `run_ingest(..., force=True)` (CLI `--force`); a blocked run writes the report and
      raw cache, skips snapshot and delta, and sets `artifact_notes.snapshot_skip_reason`
- [x] `_render_rebuild_snapshot` shows the carried-forward counts in its success message and
      surfaces a blocked run as an error, not a silent no-op
- [x] Tests: curated-only run on a temp `processed_dir` seeded with a two-source prior
      snapshot keeps the other source's rows; disabled source is not carried; >50% drop
      refuses to write; `force=True` writes
- [x] All four CI commands green

---

## Task 1.2: Hand-Confirm the Five Curated Records *(owner-dependent)*

**Why:** The five migrated records carry `trust: unverified`, `status: unknown`, and (for
three of them) no `verified_on`. They are the only records the local boost, the "Confirm you
meet" path, and the effort explanations were built for, and none of them qualifies. Task 1.3
enforces trust; run this first so it does not empty the catalog.

**Preflight Files:**
- `data/catalog/records/*.json` (5 files)
- `data/catalog/inbox/reverify-dell-technologies-scholars-program.json` (the 403 proposal)
- `scripts/catalog_inbox.py` (`confirm --set`, `reject`)
- `data/catalog/schema.json` (`status`, `trust`, `provenance.verified_by` vocabularies)

**Validation Commands:**
```powershell
python scripts/validate_catalog.py
python -m pytest tests/ -q
python -c "import json,glob; [print(json.load(open(f))['catalog_id'], json.load(open(f))['status'], json.load(open(f))['trust']) for f in glob.glob('data/catalog/records/*.json')]"   # no 'unknown', no 'unverified'
```

**Checklist:**
- [x] Claude: for each record, open `source_url`, draft the current `deadline`, `cycle`,
      `status`, `amount_*`, `requirements`, and `grade_levels` as an inbox proposal of kind
      `reverify` with the diff; note any page that is dead, blocked, or renamed
      *(2026-09-13: 5 `reverify` proposals queued. Only 2 of 5 pages were readable. AFCEA
      renamed — award listing moved to `/afcea-educational-foundation/scholarships`;
      `amount_min` 2500 -> 1500, deadline and cycle month dropped to null (page publishes
      neither). Dell moved — old URL now 404 (not 403); live page `dellscholars.org/scholarship/`
      contradicts the record's December cycle, so `deadline_month` 12 -> 2, `opens_month`
      null -> 12, `recommendation_letters` null -> 1, `deadline` -> null, `status` -> upcoming.
      Collegiate Inventors: TLS certificate expired. SWE: HTTP 403. Google Generation:
      JavaScript-rendered, empty response. Per design principle 4 the three unreadable pages
      got proposals carrying notes and no field changes — a blocked fetch proposed nothing.)*
- [x] Owner: review each proposal on the Inbox page; confirm with `trust: verified_local` and
      `provenance.verified_by: <owner>` or reject with a reason; a dead award is deleted from
      `records/`, not left `unknown`
      *(2026-09-19: owner confirmed AFCEA, Dell and SWE and rejected two. Collegiate Inventors
      rejected "Unsafe website" (the expired TLS certificate) and Google Generation rejected
      "Not really a scholarship website"; both records deleted from `records/`, so the catalog
      is 3 records. All three survivors stamped `trust: verified_local`,
      `provenance.verified_by: daedwards06`, `verified_on: 2026-09-19`. Note `reject()` archives
      the proposal but does not delete the record — the deletion is a separate step.)*
      *(2026-09-19, follow-up: the owner reports the SWE page showed no deadline, so
      `deadline` 2027-02-15 -> null and `cycle.deadline_month` 2 -> null — both derived from the
      unverified static-feed migration, not from the page. SWE and AFCEA therefore carry no date
      at all and are Task 1.6 `needs_date` candidates; only Dell has a projectable cycle. Final
      state: 3 records, all `verified_local`, 2 of 3 with no usable deadline.)*
- [x] Resolve the Dell proposal: the 403 is bot-blocking; the owner checks the page in a
      browser and confirms or rejects on what it says
      *(2026-09-19: the premise was wrong — the old URL now returns 404, not 403. Confirmed
      against the live page at `dellscholars.org/scholarship/`: `source_url` replaced,
      `deadline_month` 12 -> 2, `opens_month` null -> 12, `recommendation_letters` null -> 1,
      `deadline` -> null, `status` -> upcoming.)*
- [x] Rebuild the snapshot (Task 1.1 path) and confirm the five records show the local boost
      line in their explanations
      *(2026-09-19: rebuilt curated-only — 3 parsed, 102 carried forward from `open_scholarships`,
      105 total vs 107 prior, guardrail not triggered, delta removed=2 changed=2. All three
      curated rows render "Local award, smaller applicant pool" via `explain_ranked_row`.
      Three records, not five: two were rejected and deleted.)*
- [x] All four CI commands green
      *(2026-09-19: pytest 655 passed / exit 0 / coverage 89.15%, ruff clean, mypy clean on
      62 files, validate_catalog passes on 3 records. Deleting the two rejected records broke
      13 tests in `tests/test_curated_catalog.py`, which used the live
      `google-generation-scholarship.json` as its fixture and asserted its pre-confirmation
      values; the fixture is now an inlined literal so curating the catalog can no longer
      break the suite.)*

---

## Task 1.3: Enforce the Trust Rule in Stage 1

**Why:** Design principle 2 of the Family Product Plan says only confirmed records or trusted
structured-feed records count as eligible. The README repeats it. Stage 1 has no trust check;
the only reader of `trust` is the Stage 3 local boost. An `aggregator` scrape that returns
tomorrow would rank as eligible on the day it lands.

**Preflight Files:**
- `src/rank/stage1_eligibility.py` (`_evaluate_row` reason-code sequence, `apply_eligibility_filter`)
- `src/rank/whatif.py` (`whatif_eligibility` must not report a trust rejection as something the student can change)
- `tests/test_eligibility_rules.py`, `tests/conftest.py` (`scholarship_row_factory` has no `trust`; missing must pass)
- `README.md` "Data sources" table (the trust column)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python scripts/validate_catalog.py
python scripts/evaluate_golden_students.py --k 10 --similarity-mode embeddings --label-mode hybrid   # unchanged on the current snapshot (no aggregator rows)
```

**Checklist:**
- [x] New reason code `TRUST_UNCONFIRMED`: fires when `trust` is `unverified` or `aggregator`;
      `verified_local` and `structured_feed` pass; a missing or null `trust` passes (pre-catalog
      fixtures and golden snapshots)
      *(2026-09-19: fires first in the `_row_reasons` sequence — it is a record-quality gate, not
      a student axis, so it reads before anything profile-dependent. `UNCONFIRMED_TRUST_VALUES`
      and `TRUST_UNCONFIRMED_CODE` live in `src/rank/stage1_eligibility.py`; the check goes
      through `normalize_text`, so a mixed-case `Aggregator` is caught too.)*
- [x] `whatif_eligibility` lists trust rejections under a separate "needs confirmation" count,
      not as a profile-changeable axis
      *(2026-09-19: `WhatIfSummary.needs_confirmation` counts awards the overridden profile clears
      on every axis except `TRUST_UNCONFIRMED`. Note a trust rejection can never reach an award's
      `reasons`: trust is profile-independent, so a trust-blocked row is ineligible in both the
      base and the what-if run and therefore appears in neither `newly_eligible` nor
      `newly_ineligible`. Filtering the code out of the reason lists would have been unreachable
      code, so the count is the whole mechanism. The What If page renders it as a caption.)*
- [x] Operator sidebar gains an "Include unconfirmed records" toggle (operator mode only) that
      disables the rule for pipeline inspection
      *(2026-09-19: `apply_eligibility_filter(..., include_unconfirmed=)`, keyword-only, default
      `False`. The app gates it on `operator_mode` at the call site as well as hiding the widget,
      so a stale session-state value cannot leak the rule off in student or parent mode.)*
- [x] README "Data sources" table: trust column reads what the records say after Task 1.2, and
      the sentence "Only confirmed records, or records from a trusted structured feed, count as
      eligible" names the reason code
      *(2026-09-19: curated count 5 -> 3 and the snapshot reference moved to
      `scholarships_snapshot_20260919.parquet` (105 records). The paragraph about the 0912 run's
      zero listing pages was rewritten, since the 0919 snapshot is a curated-only rebuild with
      102 rows carried forward, not that run.)*
- [x] Tests: each trust value; missing column; operator override
      *(2026-09-19: 4 tests in `tests/test_eligibility_rules.py` (7 parametrized trust values
      including mixed case and empty string, absent column, operator override, and trust leading a
      multi-code sequence) and 3 in `tests/test_whatif.py`.)*
- [x] All four CI commands green
      *(2026-09-19: pytest 668 passed / 0 failed / exit 0 / coverage 89.17%, ruff clean, mypy
      clean on 62 files, validate_catalog passes on 3 records. Golden eval re-run at k=10
      embeddings/hybrid: no `TRUST_UNCONFIRMED` in the ineligible reason breakdown, and the Stage 1
      eligible set is byte-identical with the rule on and off for all 9 golden students — the
      snapshot holds 102 `structured_feed` + 3 `verified_local` and zero unconfirmed rows, so the
      rule is a no-op on today's data, as the task predicted.)*

---

## Task 1.4: Cross-Source Dedupe, Curated Wins

**Why:** `_dedupe_records` keys on `scholarship_id` plus normalized `source_url`. A curated
record and its feed twin have different ids and usually different URLs, so both survive. The
student sees two Dell rows. When a feed award is confirmed into the catalog, the feed copy
should disappear, not compete.

**Preflight Files:**
- `scripts/run_ingest.py` (`_dedupe_records` at ~L173, `_normalize_url_for_dedupe`)
- `src/ingest/sources/curated_catalog.py` (what a curated row carries: `catalog_id`, `source_url`, `title`)
- `src/text_utils.py` (existing title normalization helpers)
- `src/io/snapshotting.py` (`build_delta`: superseded rows should show as `removed` with a reason)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python scripts/run_ingest.py --max-listing-pages 0 --max-detail-pages 0
python -c "import pandas as pd; from pathlib import Path; from src.io.snapshotting import get_latest_snapshot_path; df = pd.read_parquet(get_latest_snapshot_path(Path('data/processed'))); print(df[df.title.str.contains('Dell', case=False)][['title','source']])"   # one row, curated
```

**Checklist:**
- [x] Dedupe key becomes `(normalized source host + path)` OR `(normalized title, sponsor)`;
      normalization strips scheme, `www.`, trailing slash, query, and casefolds; title
      normalization strips punctuation, "the", "program", "scholarship(s)"
      *(2026-09-19: `_collapse_cross_source_duplicates` in `scripts/run_ingest.py` unions rows by
      `_url_match_key` (host + path off `_normalize_url_for_dedupe`, which already strips scheme,
      `www.`, query and trailing slash) and by `(normalize_title_for_match(title),
      normalize_text(sponsor))`. The new `normalize_title_for_match` in `src/text_utils.py`
      casefolds, collapses punctuation runs to spaces and drops `the`/`program`/`scholarship(s)`.
      A bare host with no path yields no key, so awards sharing a sponsor home page never merge.)*
- [x] Precedence by `trust`: `verified_local` > `structured_feed` > `aggregator` > `unverified`;
      ties keep the earlier source in `register_sources` order
      *(2026-09-19: `TRUST_PRECEDENCE` plus a `register_sources` rank map; unknown or empty trust
      ranks below `unverified`, and the final tiebreak is `scholarship_id` so the winner never
      depends on the order the sources ran in.)*
- [x] The winning row records the losers in a new `superseded_ids` list column; the run report
      gains `superseded: {source: count}`
      *(2026-09-19: `SUPERSEDED_IDS_COLUMN` added to `src/io/snapshotting.py` `OPTIONAL_COLUMNS`
      and coerced to a list per row in `prepare_snapshot_df`; carried-forward rows keep theirs.
      The report carries `records.superseded` / `records.superseded_total` and the CLI prints them.
      `build_delta` marks a superseded removal with `removed_reason: superseded` and the winner's
      id.)*
- [x] Curated record `notes` or a new optional `aliases` field in the schema lets a record name
      a known feed title it supersedes when normalization cannot match (schema change is
      additive; `validate_catalog.py` accepts it)
      *(2026-09-19: `aliases` added to `data/catalog/schema.json` as an optional unique string
      array, mapped in `CuratedCatalogSource._map_item`, and added to `CATALOG_COLUMNS` /
      `CATALOG_LIST_COLUMNS`. `validate_catalog.py` is schema-driven and passes on 3 records.)*
- [x] Tests: URL match, title match, precedence, alias match, no false merge on two distinct
      awards from one sponsor
      *(2026-09-19: 7 new tests in `tests/test_run_ingest.py` — URL match keeping the curated row,
      normalized title+sponsor match, trust precedence across three sources, registry-order
      tiebreak, alias match, two distinct awards from one sponsor left alone, and two awards
      sharing only a sponsor home page left alone.)*
- [x] All four CI commands green
      *(2026-09-19: pytest 675 passed / exit 0 / coverage 89.14%, ruff clean, mypy clean on 62
      files, validate_catalog passes on 3 records. `run_ingest.py --max-listing-pages 0
      --max-detail-pages 0` superseded the `open_scholarships` Dell row with the curated record
      (`superseded: {'open_scholarships': 1}`), and the snapshot now holds one Dell row, from
      `curated_catalog`.)*

---

## Task 1.5: Blocked Is Not Dead

**Why:** `verify_record` turns any HTTP-status exception into `OUTCOME_DEAD_LINK` and writes a
`reverify` proposal that sets `status: unknown`. Sponsor sites return 403 to scripted fetches
routinely. The scheduled monthly pass would, over a year, propose demoting every bot-guarded
record in the catalog.

**Preflight Files:**
- `src/catalog/verify.py` (`OUTCOME_*` at L58–61, `_fetch` at ~L395, `verify_record` at ~L232, `_dead_link_record` at ~L515)
- `scripts/verify_catalog.py` (summary counts, exit code)
- `docs/operations.md` ("What it found / What it does" table)
- `tests/test_catalog_verify.py`

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python scripts/verify_catalog.py --since-days 0 --max-records 5
```

**Checklist:**
- [x] New `OUTCOME_BLOCKED` for 401, 403, 429, and any 5xx; no proposal is written; the report
      row carries the status and a `check_by_hand: true` flag
      *(2026-09-19: `OUTCOME_BLOCKED` plus `_DEAD_LINK_STATUSES` / `_BLOCKED_STATUSES` and a
      public `classify_http_status` in `src/catalog/verify.py`. `verify_record` returns
      `(result, None)` for a blocked status, so nothing reaches `_record_proposal`.
      `RecordVerification` gained `check_by_hand: bool`, carried in `to_dict`, and
      `VerificationReport` gained a `blocked` property and a `blocked` key in `counts`.)*
- [x] `OUTCOME_DEAD_LINK` only for 404 and 410; other 4xx map to `OUTCOME_ERROR`
      *(2026-09-19: `classify_http_status` returns `dead_link` only for 404/410, `blocked` for
      401/403/429/>=500, and `error` for everything else with a status — 400 no longer proposes
      `status: unknown`.)*
- [x] Summary and `docs/operations.md` table gain the blocked row; the CLI prints blocked URLs
      at the end so the owner can open them in a browser
      *(2026-09-19: the counts line now reads `... dead_link=N blocked=N error=N`, per-record
      rows carry a `[check by hand]` flag, and the run ends with an `HTTP <status> <url>` list.
      `docs/operations.md` splits the old "Link is dead (4xx/5xx)" row into dead (404, 410),
      blocked (401, 403, 429, 5xx), and other-status rows, with a paragraph on why.)*
- [x] Tests with the fake client: 403 → blocked, no proposal; 404 → dead link proposal;
      503 → blocked; 404 after a prior blocked run still proposes
      *(2026-09-19: 4 new tests in `tests/test_catalog_verify.py` — blocked parametrized over
      401/403/429/500/503 (asserting no proposal, the on-disk record still `status: open`, and
      the `blocked` report list), 400 → error, 403-then-404 still proposing `status: unknown`,
      and a CLI test asserting `blocked=1`, `dead_link=0`, and the printed URL. The existing
      dead-link test is parametrized over 404/410.)*
- [x] All four CI commands green
      *(2026-09-19: pytest 684 collected / exit 0 / coverage 89.17%, ruff clean, mypy clean on
      62 source files, validate_catalog passes on 3 records. `verify_catalog.py --since-days 0
      --max-records 5` fetched all 3 curated records live: `unchanged=3 changed=0 dead_link=0
      blocked=0 error=0`.)*

---

## Task 1.6: A "Needs a Date" Bucket

**Why:** For the demo student, 93 of 101 eligible awards land in `now`, and 79 of those have
`status: unknown` and no deadline. The Now bucket is supposed to be what the student can act
on. An award with no date on record is not that; it is an award someone should look up.

**Preflight Files:**
- `src/rank/timeline.py` (`_row_bucket` at ~L124, `project_next_deadline`, bucket vocabulary)
- `src/rank/stage3_rerank.py` (`timeline_bucket` default `"now"`)
- `app/main.py` (`_render_find_section` bucket filter, `_timeline_deadline` at ~L592, `_render_timeline_section` tabs at ~L1823)
- `src/store/calendar_feed.py` (award events must skip the new bucket)
- `tests/test_timeline_buckets.py`, `tests/test_timeline_calendar.py`

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
streamlit run app/main.py   # visual: Find shows Now with only dated awards; "Needs a date" tab lists the rest with a "check the source" line
```

**Checklist:**
- [ ] New bucket `needs_date`: no `deadline`, no `cycle.deadline_month`, and `status` in
      `unknown|null`; an award with `status: open` and no date stays in `now` (the sponsor says
      it is open) but the card says "Deadline not on record"
- [ ] Find section: bucket selector gains "Needs a date"; default stays Now; the count of
      awards waiting in `needs_date` is shown as a caption under the results so the family
      knows what the catalog is hiding
- [ ] Timeline page: a "Needs a date" tab, sorted by amount, each row with a "Look it up"
      link to `source_url`
- [ ] Calendar export skips `needs_date`
- [ ] Tests: each rule; `open` with no date stays `now`; feed record with a cycle month but no
      date projects and stays out of `needs_date`
- [ ] All four CI commands green

---

## Task 1.7: Split the AFCEA Umbrella Into Per-Award Records

**Why:** `afcea-stem-scholarship` is not one award. The Educational Foundation listing carries
**13 separately-named awards** behind their own detail pages, and the single record averages
them into eligibility that is true of no individual award: `amount 1500–5000`, no GPA floor,
all four college years, `status: unknown`, no deadline. Stage 1 acts on that. Two pages read on
2026-09-19 show what the umbrella hides:

- `/stem-majors-scholarships` — **"Sophomores or juniors only. Minimum 3.0 GPA."** US citizen,
  full-time, four-year, STEM major. STEM Major $2,500 · Cyber Security $5,000 · Student Member
  $2,500. Status on the page: **"Applications closed!"**
- `/oracle-women-leadership-scholarship` — requires **active-duty military service**, 2.8 GPA,
  $1,500. A hard no for a civilian undergraduate, and today it inflates the umbrella's range.

The first of those matches the real student line for line; the second cannot apply to her. One
record cannot express both, so the umbrella is a correctness defect, not just missing detail.
The detail pages also publish a **status** even where they publish no date, so splitting moves
records off `unknown` without inventing anything.

**Depends on:** nothing hard. Task 1.6 is a read-time classification, so records created before
it are bucketed correctly once it lands; running 1.3 first means the new records are confirmed
once, under the trust rule, rather than confirmed and then re-reasoned about.

**Preflight Files:**
- `data/catalog/records/afcea-stem-scholarship.json` (the umbrella being replaced)
- `src/catalog/inbox.py` (`propose`, `confirm`), `scripts/catalog_inbox.py`
- `data/catalog/schema.json` (`catalog_id` is identity — a split mints new ids, it does not rename)
- `src/rank/stage1_eligibility.py` (which axes actually filter: `grade_levels`, `min_gpa`,
  `military_family`, `citizenship`)
- `app/main.py` (`_render_add_award_page`, `_render_proposal`)

**Validation Commands:**
```powershell
python scripts/validate_catalog.py
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -c "import json,glob; rs=[json.load(open(f)) for f in glob.glob('data/catalog/records/*.json')]; nodate=[r for r in rs if not r.get('deadline') and not (r.get('cycle') or {}).get('deadline_month')]; print(len(rs),'records;',sum(1 for r in rs if r['trust']=='verified_local'),'verified_local;',len(nodate),'with no date')"
```

**Checklist:**
- [ ] Claude: fetch all 13 AFCEA detail pages; for each, draft a `prefill` proposal with its own
      `catalog_id`, real `amount_min`/`amount_max`, `grade_levels`, `min_gpa`, `citizenship`,
      `military_family`, `status`, and requirements; note which pages publish no date
- [ ] Skip, with the reason recorded, the awards that cannot apply to an undergraduate student
      (STEM Teachers, Shrader Graduate, chapter-administered programs) rather than adding
      records the family will never act on
- [ ] Owner: confirm or reject each proposal on the Inbox page; `trust: verified_local` only
      for a page the owner opened
- [ ] Retire `afcea-stem-scholarship`: delete it once its awards exist as records, so the
      averaged eligibility stops ranking. Its `catalog_id` is never reused
- [ ] Verify against the real profile that the STEM Major award is eligible and the
      military-only and teacher-only awards are rejected with a reason code
- [ ] Rebuild the snapshot; confirm the record count rises and the guardrail does not fire
- [ ] All four CI commands green

---

## Task 1.8: The Machine Pass Must Not Sign a Person's Name

**Why:** `data/catalog/schema.json` defines `trust: verified_local` as *"a person opened
source_url and confirmed the record on `verified_on`."* `verified_on` / `verified_by` are,
by the schema's own words, the record of a **human** confirmation. But `_verified_provenance`
in `src/catalog/verify.py` overwrites both unconditionally, so a machine pass that learns
nothing erases who last confirmed the record.

This is not hypothetical. Running Task 1.5's own validation command on 2026-09-19 rewrote all
three curated records from `verified_by: daedwards06` to `verified_by: verify_catalog` — on the
same day a person hand-confirmed them in Task 1.2. The change was reverted rather than
committed, but the behavior is still in the code, and the scheduled monthly task will do it
again to every record in the catalog, unattended.

`docs/operations.md` already promises the weaker half of this: *"`trust` is never raised:
`verified_by: verify_catalog` means a machine re-read the page ... which is a weaker claim than
`trust: verified_local` — that one still requires a person."* The gap is that nothing stops the
machine from **lowering** the attribution, which is how the stronger claim's evidence gets lost.
Success Criterion 2 ("Every record ... is `verified_local` with a `verified_on` date") is only
meaningful while `verified_on` still means what the schema says it means.

**The fix:** give the machine pass its own two fields. `checked_on` / `checked_by` record that a
script re-read the page; `verified_on` / `verified_by` stay untouched and keep meaning a person
did. Scheduling then runs off whichever is later, so a machine check still defers the next fetch
by `since_days` and the monthly pass costs no more than it does today.

**Depends on:** Task 1.5 (done). No hard dependency the other way, but land it before Task 2.2
seeds NC awards and before any unattended run of the scheduled task, or the loss recurs.

**Preflight Files:**
- `src/catalog/verify.py` (`_verified_provenance` at ~L527, `_stamp_verified_on` at ~L479,
  `_VALUE_PATTERNS` at ~L107, `_record_proposal` at ~L550, `_due_records` / `_verified_on` at ~L453)
- `data/catalog/schema.json` (`provenance` block — `additionalProperties: false`, so new keys
  must be declared to validate)
- `src/catalog/entry.py` (~L392: the Add Award form rebuilds `provenance` from scratch and would
  silently drop the new keys on an edit)
- `src/ingest/sources/curated_catalog.py` (`_PROVENANCE_KEYS` at L45 — what reaches the snapshot)
- `docs/operations.md` (the "What it found / What it does" table and the `trust` paragraph)
- `tests/test_catalog_verify.py`

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python scripts/validate_catalog.py
python scripts/verify_catalog.py --since-days 0 --max-records 5
git diff --stat data/catalog/records/   # provenance edits only; no verified_by churn
```

**Checklist:**
- [ ] `provenance.checked_on` / `checked_by` added to `data/catalog/schema.json` as optional
      keys (additive; `additionalProperties: false` requires declaring them);
      `validate_catalog.py` passes
- [ ] `_verified_provenance` writes only `checked_on` / `checked_by: verify_catalog` and never
      touches `verified_on` / `verified_by`; `_VALUE_PATTERNS` and `_stamp_verified_on` patch
      the new keys, keeping the hand-formatted-file guarantee
- [ ] `_due_records` schedules on the later of `verified_on` and `checked_on`, so a machine
      check still defers the next fetch by `since_days`
- [ ] The changed-page path (`_record_proposal`) stamps `checked_*` on the proposed record, so
      confirming a `reverify` proposal does not erase who last confirmed it
- [ ] `entry.py` round-trips `checked_on` / `checked_by` instead of dropping them when the owner
      edits a record in the app; `curated_catalog._PROVENANCE_KEYS` carries them to the snapshot
- [ ] `docs/operations.md` table and `trust` paragraph say which pair the script writes and that
      it never writes the human pair
- [ ] Tests: an unchanged page keeps a human `verified_by` and sets `checked_by`; a never-verified
      record gets `checked_*` and leaves `verified_*` null; `--since-days` defers on `checked_on`
      alone; a `reverify` proposal preserves `verified_by`
- [ ] Re-run the live pass over the three curated records and confirm they keep
      `verified_by: daedwards06` (this is the regression that motivated the task)
- [ ] All four CI commands green

---

# Phase 2 — A Catalog With Local Awards

## Task 2.1: Scholarship America — Decide and Quiet the Alarm

**Why:** `sources.json` enables the aggregator scrape, every run passes `max_listing_pages 0`,
and the health check fires the zero-record regression warning every time. A warning that fires
on every run stops being read, which defeats Task 2.1 of the Family Product Plan.

**Preflight Files:**
- `data/catalog/sources.json`
- `scripts/run_ingest.py` (health block, `caps_hit`, `guardrail_warnings`)
- `src/ingest/registry.py` (`register_sources`, `disabled_sources`)
- `README.md` "Data sources" table and the paragraph under it
- `tests/test_source_registry.py`, `tests/test_run_ingest.py`

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python scripts/run_ingest.py --max-listing-pages 0 --max-detail-pages 0   # no guardrail warning for a source that was capped to zero pages
```

**Checklist:**
- [ ] Health: a source whose `caps_hit` includes `max_listing_pages` with a zero cap reports
      `status: skipped` and `health.zero_record_regression: false`; regression only fires when
      the source actually attempted a fetch
- [ ] Owner decision recorded in `sources.json` note and `docs/decisions.md`: either disable
      `scholarship_america` until its records go through the inbox, or run it with pages and
      keep it `aggregator` (which Task 1.3 makes ineligible until confirmed)
- [ ] README paragraph under the table rewritten to match the decision
- [ ] Tests: capped-to-zero source is skipped, not regressed; a real zero after a prior
      non-zero still regresses
- [ ] All four CI commands green

---

## Task 2.2: Seed North Carolina Local Awards *(owner-dependent)*

**Why:** The Family Product Plan's thesis is that local awards have the smallest applicant
pools and appear in no aggregator. The catalog has zero of them, and zero records with a county.
This is the highest-leverage data task left, and it is also the first real exercise of the URL
prefill, the inbox, and the Add Award page.

**Preflight Files:**
- `app/main.py` (`_render_add_award_page`, `_render_prefill_input`)
- `src/ingest/prefill.py`, `src/catalog/entry.py` (`validate_form`)
- `src/ingest/extract_common.py` (NC county list)
- `data/demo/student_demo.json` (county `Guilford` for the demo; the real profile's county is the target)
- `docs/operations.md` (add a "Where local awards come from" section)

**Validation Commands:**
```powershell
python scripts/validate_catalog.py
python -m pytest tests/ -q
python -c "import json,glob; rs=[json.load(open(f)) for f in glob.glob('data/catalog/records/*.json')]; print(len(rs), 'records;', sum(1 for r in rs if r.get('counties_allowed') or r.get('states_allowed')==['NC']), 'NC or county-scoped')"   # >= 20 records, >= 15 NC or county-scoped
```

**Checklist:**
- [ ] Owner: provide the source list — the high school counselor's scholarship page or
      handout, the county community foundation, the family's credit union, employer(s), and
      church or civic organizations; Claude cannot know these
- [ ] Claude: for each source, paste the URL into Add Award, review what prefill produced,
      record what it missed (feeds a later prefill task), and submit as `manual` proposals
      with `trust: unverified`
- [ ] Claude: add the NC statewide programs with public pages (NCSEAA-administered awards,
      Golden LEAF, NC Sheriffs' Association, and the like) the same way
- [ ] Owner: confirm each proposal on the Inbox page, setting `trust: verified_local` only
      after opening the page
- [ ] At least 15 records with `states_allowed: ["NC"]` or a non-empty `counties_allowed`;
      every one has a `deadline` or a `cycle.deadline_month`, so none lands in `needs_date`
- [ ] `docs/operations.md` "Where local awards come from": the source list, how often each is
      re-checked, and that the counselor list changes every fall
- [ ] All four CI commands green

---

## Task 2.3: Onboard the Real Student and Label Her Top 20 *(owner-dependent)*

**Why:** There is no private profile, the database has no applications, and the human-label
item in Family Product Plan Task 4.3 is blocked on exactly this. Success criteria 3 and 4 of
that plan are untested until the student has used the app once.

**Preflight Files:**
- `src/profile/store.py` (save path `data/private/students/<student_id>.json`)
- `app/main.py` (`_render_profile_sidebar`, save button)
- `scripts/make_labeling_worksheet.py` (`--student --top-ranked --n 20`)
- `src/eval/human_labels.py` (label file format, `profile_id` join)
- `scripts/evaluate_golden_students.py` (`--human-labels`)
- `docs/operations.md` "Phone width"

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python scripts/make_labeling_worksheet.py --student --top-ranked --n 20
python scripts/evaluate_golden_students.py --k 10 --human-labels data/private/eval/human_labels_student.csv
```

**Checklist:**
- [ ] Owner with the student: fill the sidebar profile on a phone in student mode, save it, open
      This Week and Find; note anything that did not fit at phone width
- [ ] Owner with the student: save at least three awards from ranked cards, so My Applications
      and This Week have rows
- [ ] Claude: `make_labeling_worksheet.py --student` writes to `data/private/eval/` (private
      profile, private worksheet); the evaluator accepts a labels file from that path
- [ ] Student: label her top 20 (0/1/2 scale, same as `human_labels.csv`)
- [ ] Claude: report human NDCG@10 for her profile in `docs/evaluation.md` under "Real
      outcomes", without the labels or her profile entering git; the README cites the number
      and says the labels are private
- [ ] Phone-width findings from the first bullet become checklist items on Task 3.2 or a
      new task, not silently fixed
- [ ] All four CI commands green

---

# Phase 3 — App and Repo Hygiene

## Task 3.1: A Real Outcomes Page

**Why:** Outcomes is in the parent nav and renders a placeholder. Outcome entry exists inside
application detail, `repo.list_outcomes` exists, and the money view already sums them. The
page should list them.

**Preflight Files:**
- `app/main.py` (`_render_pending_section`, `_PENDING_SECTION_NOTES`, `_render_outcome_form` at ~L1262, `_render_money_summary` at ~L1963)
- `src/store/repo.py` (`list_outcomes`, `Outcome`, `list_applications`)
- `src/store/money.py` (`is_won`, `won_amount`, `award_year`)
- `scripts/export_outcomes.py` (the CSV the page should offer)
- `app/modes.py` (`PARENT_ONLY_SECTIONS`)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
streamlit run app/main.py   # visual: parent mode → Outcomes lists rows, totals by year, export button
```

**Checklist:**
- [ ] Outcomes page: table of outcomes joined to application title, cycle year, result,
      amount, renewal terms; totals by year; a "Download outcomes.csv" that calls the export
      script's function
- [ ] Empty state says where outcomes are entered (application detail) rather than showing an
      empty table
- [ ] `_render_pending_section` and `_PENDING_SECTION_NOTES` deleted
- [ ] Tests: the page's row-building helper (pure) on a temp database with two outcomes
- [ ] All four CI commands green

---

## Task 3.2: Split `app/main.py` Into Section Modules

**Why:** 2,850 lines, every section inline, the router in the last 40. Every UI task in this
plan and the next edits this file. Section modules make phone-width and mode work reviewable
and let tests import one section without the rest. Behavior-preserving.

**Preflight Files:**
- `app/main.py` (whole file; the `_render_*_section` functions are the seams)
- `app/helpers.py`, `app/modes.py`
- `tests/test_app_modes.py`, `tests/test_phone_layout.py`, `tests/test_explainability_helpers.py`, `tests/test_colleges_money.py`, `tests/test_whatif.py` (import paths that must keep working)
- `.github/workflows/*.yml` (ruff and mypy scopes)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/   # new: the app package gets type-checked once it is modular
streamlit run app/main.py   # visual: every section renders in every mode as before
```

**Checklist:**
- [ ] `app/sections/` with one module per section: `find`, `this_week`, `applications`,
      `essays`, `recommenders`, `timeline`, `colleges_money`, `catalog_inbox`, `what_if`,
      `outcomes`, `settings`; shared state helpers in `app/state.py`; sidebar in
      `app/sidebar.py`
- [ ] `app/main.py` is the router and page config only, under 150 lines
- [ ] `_apply_rank_filters` parameter `essay_required_only` renamed `no_essay_only` to match
      its behavior and its call site
- [ ] `python -m mypy app/` at 0 errors and added to CI
- [ ] No behavior change: the visual pass covers every section in student, parent, and
      operator mode
- [ ] All four CI commands green

---

## Task 3.3: Plan and README Bookkeeping

**Why:** Family Product Plan Task 3.2 has two unchecked items that Tasks 3.3–3.7 delivered.
UI Redesign Plan Task 6 (screenshots) was scheduled after Phase 3 of that plan and has not
run. The README "Data sources" table's trust column described the records as
`verified_local` before Task 1.2 made that true.

**Preflight Files:**
- `docs/plans/SCHOLARSHIPCOACH_FAMILY_PRODUCT_PLAN.md` (Task 3.2 checklist, Task 4.3 blocked item)
- `docs/plans/SCHOLARSHIPCOACH_UI_REDESIGN_PLAN.md` (Task 6)
- `README.md` ("Data sources", "Screenshots" absent)
- `docs/system_design.md` (sections for the new bucket and trust rule)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
git status   # docs/images/ tracked, each image < 500KB
```

**Checklist:**
- [ ] Family Product Plan Task 3.2: check the student-view and parent-view items, with a note
      that Outcomes landed in Catalog Integrity Task 3.1
- [ ] Family Product Plan Task 4.3: check the labels item once Task 2.3 here is done, with a
      pointer
- [ ] Execute UI Redesign Plan Task 6 from that plan: three screenshots (ranked cards in
      student mode, This Week on a phone-width window, the Inbox with a proposal diff),
      `docs/images/`, README "Screenshots" section
- [ ] `docs/system_design.md`: `needs_date` bucket, `TRUST_UNCONFIRMED`, carry-forward
      rebuild, cross-source dedupe precedence
- [ ] All four CI commands green

---

## Task 3.4: Re-measure and Log

**Why:** After Tasks 1.2–1.6 and 2.2 the catalog and the filter are different. The README's
numbers are from 2026-09-13 on 107 records with 88 unknown-status rows. Measure again and
write the decision-log entry for this review so the pivot is dated.

**Preflight Files:**
- `README.md` ("Evaluation results", "Limitations")
- `docs/decisions.md`
- `docs/evaluation.md`
- `scripts/evaluate_golden_students.py`
- `data/demo/student_demo.json` (the per-profile rejection count the README should quote)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python scripts/evaluate_golden_students.py --k 10 --similarity-mode embeddings --label-mode hybrid --cross-label-check
python -c "import json,pandas as pd,dataclasses; from pathlib import Path; from src.rank.stage1_eligibility import StudentProfile, apply_eligibility_filter; from src.io.snapshotting import get_latest_snapshot_path; df=pd.read_parquet(get_latest_snapshot_path(Path('data/processed'))); d=json.load(open('data/demo/student_demo.json')); f={x.name for x in dataclasses.fields(StudentProfile)}; e,i=apply_eligibility_filter(df, StudentProfile(**{k:v for k,v in d.items() if k in f})); print(len(e),'eligible',len(i),'ineligible'); print(i['reasons'].explode().value_counts())"
```

**Checklist:**
- [ ] README "Evaluation results": new table on the new snapshot, old 2026-09-13 rows moved
      under "Historical metrics" with their snapshot named
- [ ] README "Limitations": the demo-profile Stage 1 rejection breakdown, the count of
      `needs_date` awards, the count of `verified_local` records, and the count of NC or
      county-scoped records, each from the validation command output
- [ ] `docs/decisions.md`: dated entry "2026-09-13 review: the checklist was green and the
      catalog was not" with the findings table from this plan's preamble and the decisions
      list
- [ ] `docs/evaluation.md`: the trust rule and the `needs_date` bucket as things the harness
      now sees
- [ ] All four CI commands green

---

## Execution Order

```
Phase 1  1.1 rebuild + guardrail → 1.2 confirm the five → 1.3 trust rule → 1.7 AFCEA split
         → 1.4 dedupe → 1.5 blocked vs dead → 1.8 checked_by vs verified_by → 1.6 needs_date
Phase 2  2.1 Scholarship America → 2.2 seed NC local awards → 2.3 onboard the student
Phase 3  3.1 outcomes page → 3.2 split main.py → 3.3 bookkeeping + screenshots → 3.4 re-measure
```

Task 1.1 first, alone, before anything else touches the snapshot. Task 1.2 before 1.3 so
enforcement lands on a catalog with confirmed rows. Task 1.7 (added 2026-09-19) after 1.3 so the
records it mints are confirmed once under the trust rule, and before 1.4 so dedupe is exercised
against the real per-award catalog rather than one umbrella row; it has no hard dependency on
1.6, whose bucketing is applied at read time. Task 1.8 (added 2026-09-19) after 1.5, since both
rework the same verification pass, and before the scheduled monthly task is left to run
unattended — every unattended pass until it lands erases a human `verified_by`. Task 2.2 after
1.6 so seeded records are checked against the `needs_date` rule as they land. Task 3.2 after 3.1 so the split does not
carry a placeholder. Task 3.4 last.

## Success Criteria

1. Confirming an award in the app and rebuilding leaves every feed record in place; a run that
   would halve the catalog refuses to write.
2. Every record in `data/catalog/records/` is `verified_local` with a `verified_on` date, and
   at least 15 are NC or county-scoped. No `unverified` or `aggregator` record ranks as
   eligible.
3. The demo student's Now bucket contains only awards with a deadline on record or a
   projectable cycle; the rest wait in "Needs a date" with a link to look them up.
4. The real student has a private profile, saved applications, and 20 labels; her human
   NDCG@10 is reported without her data entering git.
5. `app/main.py` is a router; every section is a module; `mypy app/` is in CI.
6. All four CI commands green after every task.
