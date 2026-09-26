# Decision Log

Dated record of the choices that shaped the project, with the evidence behind each.
Newest first. A decision stays here after it is reversed; the reversal is a new entry.

---

## 2026-09-26 — Scholarship America is disabled; its awards go through the inbox

**Context.** `sources.json` enabled the `scholarship_america` scrape, but every ingest ran with
`--max-listing-pages 0`, so the connector stopped before its first request. The health check could
not tell that from a broken scraper and raised `Source 'scholarship_america' returned 0 records but
returned 13 on the prior run` on every run. An alarm that fires every time stops being read.

**Decision.** Disable `scholarship_america` in `sources.json`. Awards found on its site are
brought in through the catalog inbox and confirmed there, like any other lead. Separately, the
health check now reports a source capped to zero listing pages as `status: skipped`, never as a
zero-record regression, and looks past skipped runs when finding the prior count, so a real zero
after a skipped run still fails.

**Rejected.** Keeping it enabled and running it with listing pages. Its rows carry
`trust=aggregator` and are ineligible until confirmed (Catalog Integrity Task 1.3), so the scrape
would add fetch load and terms exposure for rows the student cannot see until someone confirms
them by hand — the same work the inbox already does.

---

## 2026-09-26 — One record per award, even when the sponsor publishes them on one page

**Context.** `afcea-stem-scholarship` was one `verified_local` record standing in for AFCEA's whole
Educational Foundation listing: `amount 1500–5000`, no GPA floor, all four college years,
`military_family: null`, `status: unknown`. The listing is 13 separately-named awards. Reading all
13 detail pages on 2026-09-19 showed the averaged row was true of none of them — the STEM Majors
page requires sophomore-or-junior standing and a 3.0 GPA, while the Oracle Leadership page
requires the applicant's own active-duty service at a 2.8 GPA and $1,500. Stage 1 filters on
exactly those axes, so the umbrella was a correctness defect, not merely missing detail.

**Decision.** Retire `afcea-stem-scholarship` (its `catalog_id` is never reused) and mint ten
per-award records, each with the eligibility its own page publishes. The real student's profile
now gets four eligible AFCEA awards and six `MILITARY_FAMILY_ONLY` rejections where it previously
got one averaged row.

**Skipped, with reasons, rather than recorded** — five listing entries no undergraduate can act on.
They are also archived under `data/catalog/inbox/rejected/`, but that directory is git-ignored, so
the reasons live here:

| Entry | Why it is not a record |
|---|---|
| STEM Teachers Scholarships | Page states "Undergraduate students are not eligible"; needs a second-semester graduate program and a 3.5 graduate GPA. |
| Shrader Graduate Scholarship | Graduate-only, $3,000, minimum 3.5 GPA. |
| Cathy E. Johnston Memorial | Graduate-only (Asian Studies / Indo-Pacific security) plus mid-career intelligence professionals; "Application coming soon!", no amount or materials published. |
| Brad A. Logan Memorial Award | $2,000 restricted to a current senior at one Pennsylvania high school, applied for through that school's Schoology. |
| Chapter Scholarship Programs | A directory of chapters that each run their own program — no amount, eligibility, status or deadline of its own. Individual chapter awards belong in the catalog one at a time. |

**Why the skips are not just low-ranking records.** A record the family will never act on still
costs a person's attention every time it surfaces in a review or a re-verification pass. Recording
the reason once is cheaper than re-deciding it annually.

**A dedupe rule this exposed.** Task 1.4's cross-source dedupe treated a shared normalized URL
(host + path) as proof that two rows are the same award. Three of these awards — STEM Major
($2,500), Cyber Security ($5,000) and Student Member (AFCEA membership required) — are published
on one page, so that rule silently collapsed them back into one row: the umbrella defect
reappearing one layer down. A URL is now treated as an identity key only while it maps to at most
one row per source; a source offering two rows for one URL is asserting that the page holds
sibling awards, and matching falls back to title, sponsor and aliases. A feed row still collapses
onto the curated award it duplicates.

**Known limitation.** `military_family` is the only service-connection axis Stage 1 filters on, so
it carries both "your family served" (iWorks) and "you serve" (Oracle, ROTC, War Veterans, Susan
Lawrence). The rejection is correct for a civilian undergraduate either way; splitting the axis
would be a schema change and is not yet worth it.

---

## 2026-09-13 — Report the catalog that exists, not the one the metrics were measured on

**Context.** The README's headline table (NDCG@10 0.61, Coverage@10 0.40) was measured on a
163-record March 2026 catalog. That catalog is gone: Bold.org was disabled, Scholarship America
returned nothing usable, and the Open Scholarships feed replaced both.

**Decision.** Re-measure on the current 107-record catalog and replace the tables. Old rows stay
in the README under an explicit "historical" heading with the snapshot they came from, because
deleting them would hide the pivot rather than document it.

**Measured (2026-09-13, `scholarships_snapshot_20260912.parquet`, 9 golden profiles, K=10,
embeddings mode, win model off):** baseline weights NDCG@10 0.744, tuned weights 0.831;
coverage 0.400 → 0.467; eligibility precision 0.628 both.

---

## 2026-09-13 — The human-labeled eval set is pinned to its snapshot, and says so

**Context.** `data/eval/human_labels.csv` (44 hand-judged pairs, 2 profiles) joins on
`scholarship_id`. Task 1.1 made curated records hash `catalog_id` instead of content, and the
catalog itself turned over. Overlap between those 22 award ids and the current snapshot is **0**;
only 5 awards survive even by title match, which is too thin to carry an NDCG@10.

**Decision.** Do not re-key the committed labels by title to make the number reappear. Keep the
file as the June-27 artifact it is, report its NDCG (0.848, reproducible with
`--snapshot data/processed/scholarships_snapshot_20260627.parquet`) as historical, and treat
re-labeling against the current catalog as the real fix.

**Consequence.** There is no current human-judged headline until the student labels her own list.
`scripts/make_labeling_worksheet.py --student --top-ranked --n 20` generates that worksheet.

---

## 2026-09-13 — A label worksheet can target the real student, and her ranked list

**Context.** The worksheet script only accepted golden personas, and always sampled across the
eligible set to avoid ranking bias. Neither fits asking a real teenager for twenty judgements.

**Decision.** Add `--student` (reads `data/private/students/`, falls back to the committed demo
profile with a printed warning) and `--top-ranked` (the ranked prefix she actually sees).

**Trade-off, stated because it is real.** A top-20 label set can measure the order of what was
shown but cannot reveal an award the ranker missed. Eligible-set sampling stays the default.

---

## 2026-09-12 — Product review: the catalog is the bottleneck, not the ranker

Measured against `scholarships_snapshot_20260813.parquet` with the `nc_cs_rising_sophomore`
profile:

| Finding | Evidence |
|---|---|
| Catalog had collapsed to 35 records | Down from 166 in June |
| Bold.org returned 0 records, silently | The ingest report still said `succeeded` |
| 22 of 35 awards were closed | `eligibility_text` began `Status: Closed`; no parser read it |
| Majors, GPA minimum and education level were never populated | 0 non-null in **every** snapshot in the repo — Stage 1 was a state-plus-deadline filter |
| A missing deadline passed Stage 1 | Closed awards ranked as eligible with "Unknown deadline" |
| The student's top 5 were the 5 hand-typed static entries | The next 9 were closed, several restricted to specific employers' children |
| The canonical id hashed the deadline | A recurring award got a new `scholarship_id` every year, so nothing tracked across cycles |

**Decision.** Everything downstream of ingest was sound engineering with nothing to stand on.
Stop ranking work; fix the catalog first. Recorded as design principle 1 of the Family Product
Plan: no ranking or evaluation work until Phases 1–2 supply real records.

**Verification (2026-09-13).** Stage 1 now rejects on the axes that were empty: across 963
profile×award pairs, `EDUCATION_LEVEL_MISMATCH` 336, `GPA_BELOW_MIN` 21, `MAJOR_NOT_ALLOWED` 12,
`CITIZENSHIP_MISMATCH` 2.

---

## 2026-09-12 — Portfolio and family product are one repo

**Decision.** General in shape, specific in content. Code, schema and roles never know the
student exists; her data lives in git-ignored `data/private/`. A fictional demo profile
(`data/demo/student_demo.json`) ships so a fresh clone runs.

**Why.** Two repos would mean two pipelines, and the family one would rot.

---

## 2026-09-12 — No generative LLM in this plan

**Decision.** The LLM extraction module stays in the repo with its flag off. Ingest automation is
deterministic: structured feeds, regex prefill, content-hash re-verification. The LLM extractor
is a Phase 5 candidate behind the `Extractor` protocol; LLM Plan Tasks 6–9 are deferred, not
executed.

**Why.** Per-record cost and non-determinism against a catalog the family has to trust, for an
extraction job regex already does on the feeds that matter. The `Extractor` seam means picking it
up later is a plug-in, not a rewrite.

---

## 2026-09-12 — Automation proposes, a person confirms

**Decision.** Feeds, URL prefill and re-verification write proposals to `data/catalog/inbox/`,
never to `data/catalog/records/`. `confirm()` is the only path in, and it validates against
`data/catalog/schema.json` first. Only confirmed records, or records from a trusted structured
feed, count as eligible.

**Why.** Automated extraction is wrong often enough to matter, and a wrong record costs the
family an application.

---

## 2026-09-12 — The win model leaves the family-facing views

**Decision.** `p_win` and expected value are removed from student and parent views. The code
stays, reachable behind the operator toggle, and the README reframes it as a
calibration/expected-value demonstration on a known synthetic generator — retired from the
product, kept as a portfolio artifact.

**Why.** Its labels come from `src/win_model/synthetic.py`, not award outcomes. Showing a
student a number that looks like her odds, generated by a heuristic, is worse than showing
nothing. Task 4.4 logs real outcomes; if enough accumulate, this reverses.

---

## 2026-09-12 — Stable ids keyed on `catalog_id`

**Decision.** A curated record's `scholarship_id` is `sha1("catalog|" + catalog_id)`. Deadline,
amount and title revisions no longer change it. Scraped sources keep content hashing.

**Why.** A recurring award has to be one row across cycles for the tracker, outcomes and
re-verification to mean anything.

**Known cost.** It orphaned the June human-label set (see 2026-09-13 above). Accepted: cross-cycle
tracking is load-bearing for the product; 44 labels are replaceable.

---

## 2026-09-12 — Curated first, fed second, scraped last

**Context.** Verified the same day: no open dataset of US private or local scholarships exists.
Aggregators keep listings proprietary. Open Scholarships (CC BY 4.0) is Nevada-focused but has a
near-compatible schema. CFNC filters on exactly our axes but renders dynamically with no
automated-access terms. Local awards — counselor lists, community foundations, employers,
churches, credit unions — have the smallest applicant pools and appear in no aggregator.

**Decision.** Hand-curated JSON records under `data/catalog/records/`, one file per award, are the
primary data asset. Structured feeds second. Scrapes last, behind a health check and a per-site
terms review.

---

## 2026-09-12 — One profile, two modes; multi-student modeled anyway

**Decision.** Student and parent modes over one profile and one database. Parent mode adds
curation, money and settings, and reads essays without editing them. The file layout
(`data/private/students/<student_id>.json`) and the `students` table are multi-student from day
one.

**Why.** The selector is a household convention, not an account system — the family shares a
machine. The storage layout is the expensive part to change later, so it was done now.

---

## 2026-09-12 — Every feature passes the senior-fall test

**Decision.** A feature ships if it helps the student by fall 2028. General features that fail
the test wait in Phase 5.

---

## 2026-09-12 — No public deployment

**Decision.** The app runs on a home PC, reachable from a phone on the LAN, gated by a parent PIN
in git-ignored `.streamlit/secrets.toml`. Off-network access goes through a private network
overlay (Tailscale). No hosted URL.

**Why.** `data/private/` holds a real student's profile, essays and application history.
