# ScholarshipCoach Freshness & Today Plan — New Things Arrive, and She Is Told

> Generated: 2026-11-13 | Scope from an owner review of the server's timers and the catalog
> (Part A), and of `scripts/screenshot_app.py` captures at 400px and 1366px (Part B). Every owner
> finding was re-checked against the code, the 40 curated records and
> `scholarships_snapshot_20261011.parquet` before this plan relied on it; corrections are in
> "What the check found"
> Owns: discovery, the watch list, the digest, the Today page, card hierarchy, the phone tab bar.
> The Roles & Product Plan still owns the role model and the go/no-go gate; the Design Refresh
> Plan still owns the theme and the final sweep; the Cloud Hosting Plan still owns the server
> Executor: Claude via Claude Code, with owner-dependent tasks called out as such. Each task is
> marked **judgment** or **routine** (see "Which model runs what")
> Est. effort: 5 phases, 15 tasks. No database migration is planned

---

## Owner follow-ups (open)

Owner steps left behind by tasks that are otherwise done. Claude: when starting any task from
this plan, list the items below to the owner, and say which are due. Remove an item once its
task's checklist records it.

| Due | Task | What | Done when |
|---|---|---|---|
| Any time | 2.4 | Add the family's own pages (her school's counselor page, employers, congregations) to `data/private/watchlist.json` on the server, each with a dated `terms_note`; then run `check_watchlist.py --dry-run` there | The dry run lists them with `[private]`; the Task 2.4 owner item is checked |
| After Sun 2026-11-22 05:30 UTC | 2.4 | Second watch run (the timer's): read the proposal count in `journalctl -u scholarshipcoach-watch -n 50`, clear the Inbox and time it | Both numbers are in Task 2.4's notes; over ten minutes means lowering the caps first |
| Optional | 2.4 | Read NC State's and Girls Who Code's terms in a browser; if they allow it, Claude fills `terms_note` and enables the four disabled entries | The entries are enabled, or the reason they stay off is in `notes` |

---

## Why this plan exists

Two problems that are one. New opportunities do not arrive unless the owner runs ingest by hand,
and when something does arrive, nothing in the app tells her. The app stores things well and
announces nothing: no screen answers "what matters today", and nothing on any screen has changed
since her last visit, because nothing upstream is changing either.

Part A makes the catalog move on its own, inside the existing rule (automation proposes, a person
confirms). Part B gives that movement somewhere to show up: a Today page, a weekly digest, and
cards that say what is urgent.

### What the check found

Each owner finding, checked on 2026-11-13 on the dev PC. The server's own state (its snapshot,
its inbox, its verify reports) was not readable from here; Task 1.1 starts by reading it.

| # | Owner finding | Verdict | Evidence |
|---|---|---|---|
| V1 | Timers exist for backup, catalog sync and verify; nothing runs discovery | **Confirmed** | `deploy/` holds three timers: backup 03:30 daily, catalog-sync 04:15 daily, verify first Sunday 07:00. `scripts/run_ingest.py` is called by no unit |
| V2 | Open Scholarships rows are carried forward from the 2026-09-12 pull | **Confirmed, date differs** | All 99 feed rows in the 2026-10-11 snapshot carry `first_seen_at` and `last_seen_at` of 2026-09-19. The feed has not been fetched for eight weeks either way |
| V3 | `PROPOSAL_KINDS` has `feed`, the inbox has a "From a feed" badge, nothing writes one | **Confirmed, and wider** | `propose()` is called from `src/catalog/verify.py` (`reverify`) and the Add Award form (`manual`). Nothing writes `feed` **or `prefill`**: both kinds exist only in tests |
| V4 | 33 of 40 curated records have no deadline, so the timeline and Stage 3 cannot place most of the catalog | **Half right** | 33 of 40 have no `deadline`: true. But 30 of those 33 have `cycle.deadline_month`, so `timeline.project_next_deadline` places them and Stage 3 scores urgency against `projected_deadline`. Only 3 curated records have no date signal at all. What a projected date cannot do: it shows as "~Mar 1", it never becomes a to-do date (A.7 notes), and it is a month, not a day. The rows that truly cannot be placed are the **feed** rows: 82 of 99 have no deadline and no cycle month |
| V5 | Every `kind=program` record is hand-curated; no source produces programs | **Confirmed** | 6 programs, all `curated_catalog`. Prefill reads neither `cost` nor `grade_levels`, the two fields a program card needs |
| V6 | After a confirm, the UI tells the parent to rebuild by hand | **Confirmed** | `app/sections/catalog_inbox.py` lines 494–547: "Rebuild the snapshot to rank it." A rebuild takes 0.4 s (`reports/ingest_runs/ingest_20261011T140043Z.json`) |
| V7 | Verify needs extending to propose deadlines when a new cycle's date appears | **Partly built already** | `verify.compare_record` already proposes a deadline against a `null` field, when the page states one under a label with confidence ≥ 0.7; `cycle_checkpoint` already re-reads a recurring record around its usual opening. What is missing is in the extractor: year-less dates ("March 1"), the open date taken for the deadline, page-chrome dates (`docs/operations.md`, "What prefill missed") |
| V8 | *(not raised)* Rejections are forgotten | **New finding** | `inbox.reject` says rejections are kept "so a later re-verification pass can tell that a person already looked". Nothing reads `inbox/rejected/`. A rejected proposal comes back on the next run. Harmless monthly, unusable weekly |
| V9 | *(not raised)* Feed rows cannot reach her page | **New finding** | 0 of 99 feed rows name a grade. Opportunities lists only records with an explicit grade match, and Find sends grade-less rows to "Check the grade first". So a feed row reaches **her** only after a person confirms it into a curated record with grades. That is what a `feed` proposal is for |
| V10 | No screen answers "what matters today"; overdue PSAT below a neutral card | **Confirmed** | My Path renders groups in stored `position` order; the seeded "Take the PSAT" (due 2026-10-15, not started) is drawn as a gray "Not started" row, identical to "Calculus · 12th grade". The red only appears under "Coming up", below the checklist |
| V11 | Requirement cards show status twice | **Confirmed, on rows she owns** | `my_path._render_requirement` draws a status `st.badge` and, for course and test rows, a `st.segmented_control` with the same value selected |
| V12 | Opportunities opens with "Nothing is open to your grade right now"; cards look alike; buttons repeat | **Confirmed** | The Open now empty state is the first thing drawn. `money_badge_text` gives "Amount not listed" the same badge as "$20,000". Every card draws two buttons and an expander |
| V13 | No feedback on completion | **Mostly confirmed** | Six `st.toast` calls exist (Save, Suggest, Add to my to-dos, dates saved). Ticking a to-do and setting a requirement to Met, the two completions that matter, give nothing |
| V14 | No dollars-in-play total, no "new since your last visit", no countdown | **Confirmed** | No such element in `app/`. `activity_days` holds one row per person per day, which is enough to know her last visit day without a migration |
| V15 | Phone navigation is `»` plus bottom-of-page links | **Confirmed** | `docs/decisions.md` 2026-09-30 and 2026-10-10 (R3): the top bar folds into the sidebar at 400px and stays open after a pick |
| V16 | Parent Find is one narrow column with no sort | **Confirmed** | `find.render` draws `filtered.head(shown)` in rank order in one column; the only controls are filters |

Two consequences shape the plan:

- **The feed timer will be quiet.** Open Scholarships is the only enabled feed. It is
  Nevada-focused, 82 of its 99 rows have no date and none names a grade. A weekly run over it is
  cheap and worth having (V2), but it will not be where her new content comes from. The watch
  list (Task 2.3) is. The plan is ordered and sized on that expectation.
- **"New" on her Today page is what a parent confirmed, not what a feed proposed.** She is kept
  out of the catalog and the inbox by the role model. Proposals are "new" for parents; a
  confirmed record open to her grade, a to-do from her family, or a newly posted date is "new"
  for her.

### Relationship to other plans

- **Roles & Product Plan.** This plan changes its D5 (student home is My Path) to Today, and
  pulls a narrowed version of its Task B.4 ("Bringing People Back") ahead of the gate: a digest
  for parents now, hers only by owner decision (D3). It closes that plan's recorded overruns for
  journeys P2 and P3. Its success criterion 2 is kept: where she stands is still on the home
  screen at 0 taps. Its fallback is kept: every student-facing piece here still works if the gate
  removes My Path and Opportunities (principle 6).
- **Onboarding and the gate.** Cloud Hosting Task 3.1 is still unchecked, so the family has not
  onboarded and the four-week window has not started. Changing her home page inside that window
  would change what the gate measures. D1 decides the order.
- **Design Refresh Plan.** Its Task 4.1 was left the question of the folded phone nav (R3); Task
  4.2 here answers it. Its 4.1 sweep and 4.2 README screenshots still run last, after this plan.
  The ≤ 260px opportunity-card target left "pending the owner's call" is settled by Task 4.1 here.
- **Cloud Hosting Plan.** Adds units under `deploy/`. `update.sh` does not install units;
  `bootstrap.sh` does, as root. So every task that adds a unit has an owner step on the server.
- **LLM Extraction Plan.** Its Tasks 6–9 stay deferred. Task 2.5 here is a measurement only.
- **Family Product / Catalog Integrity Plans.** "Automation proposes, a person confirms" and the
  zero-record alarm are used as they are.

---

## Owner decisions (defaults used unless changed before the task named)

| # | Decision | Default | Alternatives | Before |
|---|---|---|---|---|
| D1 | When the student-facing work (Today, tab bar, card changes) reaches her | **Before onboarding.** She has not onboarded; the gate should measure the version she will keep, as A.10 argued. Costs onboarding the length of Phases 3–4 | Onboard now on today's app and hold Phases 3–4 until the gate is held; or ship Phase 3 only before onboarding | 3.1 |
| D2 | Digest delivery | **In the app, plus an email with no content**: "3 things this week" and the tailnet link, sent from the server by SMTP. Award titles, deadlines and her to-dos never leave the tailnet | See "Digest delivery options" below | 3.2 |
| D3 | Who gets a digest nudge | **Parents only until the gate is held.** The gate counts the days she opens the app on her own; a weekly nudge to her would count as asked. Today is her digest | Her too, from day one, with the gate reading adjusted | 3.2 |
| D4 | Which feed rows become proposals | Rows that are new or whose deadline, status or amount changed since the last run, that a curated record does not already cover, that were not rejected with the same content, and that pass Stage 1 for the saved profile with grade ignored. At most 15 a run; the rest are counted, not queued. The first run records a baseline and proposes nothing | Propose the screened backlog on the first run (about 60–75 proposals at once); or no profile screen | 2.1 |
| D5 | Rebuild after confirm | **Both**: the app rebuilds right after a confirm (0.4 s), and a 03:50 timer rebuilds when the records are newer than the snapshot | Timer only (new records appear the next morning) | 1.1 |
| D6 | The encouraging number on Today | **"N done this school year"**: to-dos she ticked, requirements met, programs applied to. Never money, never a streak. Her path progress ("5 of 12 on track") stays as the link line under it | Path progress as the number; done this month | 3.3 |
| D7 | Parent landing page | A **This Week** page (the digest, live) becomes the Parent home; My Path stays first under "Her path" | Keep My Path (read-only) as the Parent home and reach the digest from the nudge link only | 3.1 |
| D8 | Family-specific watch pages | A **private overlay** on the server (`data/private/watchlist.json`) for her school's counselor page, employers and congregations. The committed list holds only pages that say nothing about her | One committed list (rejected: her school's page in a public repo identifies her school) | 2.3 |
| D9 | LLM extraction | **Measure only** (Task 2.5), on the dev PC, against hand-confirmed records. Adoption is a separate decision made on the numbers | Skip the measurement; the 2026-09-12 "no LLM" decision stands untouched | 2.5 |
| D10 | Phone tab bar if the CSS approach fails its spike | A link row at the top of every student page (not fixed) | Wait for a Streamlit release that supports it; keep today's home links | 4.2 |

### Digest delivery options (D2)

The app has no public address, so anything that pushes must either stay on the tailnet or carry
nothing worth reading.

| Option | What leaves the tailnet | Needs | Verdict |
|---|---|---|---|
| **A. In-app digest + content-free email** *(default)* | "3 things this week" and a link that only opens on the tailnet. The provider learns the family's addresses and that a digest exists | One SMTP account; addresses and credentials in `/etc/scholarshipcoach/` (never in git) | Reaches people where they already look. Reverses Roles B.4's "no email" for a message with no content |
| B. In-app only | Nothing | Nothing | Safest. Depends on being remembered, which is finding R4 |
| C. Calendar subscription (`.ics` over the tailnet) | Nothing, if the calendar app fetches over Tailscale | Roles B.4 as written; each phone's calendar must reach the tailnet name. Not verified on iOS | Good for dates, cannot say "2 proposals waiting". Stays in Roles B.4 |
| D. Self-hosted push (ntfy on the server) | To be verified: iOS delivery for a self-hosted ntfy server is usually relayed through a public service | A second service to run, an app on each phone | Most moving parts for one message a week. Not scheduled |
| E. Full-content email | Titles, deadlines, her to-dos, to a mail provider | SMTP | Rejected: a third party holds her schedule |

---

## Design principles for this plan

1. **Automation proposes, a person confirms.** No timer writes under `data/catalog/records/`
   except verify's existing `checked_on` stamp. `confirm()` stays the only way in. Tested, not
   promised (Task 2.2).
2. **Timers never deploy and never migrate.** No timer runs `update.sh`, `git pull`, `pip` or a
   restart. `src/store/db.connect` applies migrations, so a timer that reads `coach.db` opens it
   read-only without that call, and stops with a clear message when the schema is behind.
   Otherwise the first connection after an unattended update would be a timer's.
3. **Nothing private is pushed, and the guard is not widened.** Proposals, watch state and digests
   live in git-ignored paths. `deploy/privacy_check.sh` still gates everything the sync commits.
   Nothing identifying her or her school goes in a committed file, a unit, or a commit message.
4. **A quiet queue.** Every automated source has a deterministic proposal id, remembers
   rejections, and has a cap. A queue nobody can clear in ten minutes a week is a failed design.
5. **Health is reported where people look.** A source that fails or returns nothing shows in the
   digest in plain words, using the existing `skipped` / zero-record-regression rules. A unit
   that fails silently is how the catalog fell from 166 records to 35.
6. **Today survives the fallback.** Each Today section is built from one source (to-dos,
   requirements, opportunities). If the gate removes My Path and Opportunities from Student mode,
   Today still renders from To-dos alone.
7. **The design system is the spec, and is edited in the same commit.** Changes to the card and
   the home page are written into `design-system/scholarshipcoach/` when they land. CLAUDE.md's
   styling order is binding: theme, then native layout, then CSS in `phone_width_css()`.
8. **Seen at 400px in light and dark, measured by journey.** Every UI task adds or updates a
   journey in `scripts/screenshot_app.py` with a tap budget ("Journeys this plan adds").
9. **The senior-fall test.** Each task says how it helps her by fall 2028. What fails is under
   "Not scheduled".
10. **Green gate every task:** the five CI commands in CLAUDE.md.

---

## Journeys this plan adds

Taps at 400px, excluding typing, counted as the Roles & Product Plan counts them.

| # | Who | Journey | Budget | Task |
|---|---|---|---|---|
| S8 | Student | Open the app → see what is overdue or due soon, then what is new | 0 | 3.3 |
| S9 | Student | Finish the top item on Today (tick it) and see it acknowledged | 1 | 3.3, 3.4 |
| S10 | Student | A new program on Today → on her to-dos | 1 | 3.3 |
| S11 | Student | From any student page to any other top page | 1 (2 on the D10 fallback) | 4.2 |
| S1 | Student | *(revised)* Where she stands for her first target school | 0: the path line on Today; the checklist is 1 tap | 3.3 |
| P2 | Parent | *(revised)* Add a program → it is on her Opportunities, no rebuild tap | form + 1 | 1.1 |
| P5 | Parent | Open the app → proposals waiting, deadlines in 30 days, findings | 0 | 3.1 |
| P6 | Parent | From the digest to a waiting proposal's confirm form | 1 | 3.1 |
| P7 | Parent | Re-sort Find by deadline or amount | 1 | 4.3 |
| P3 | Parent | *(revised)* See what she did this week | 0 from This Week | 3.1 |

---

# Phase 1 — Close the Loop That Already Exists

Both tasks improve what is already built and deployed. Neither needs the other phases.

## Task 1.1: Rebuild After Confirm, and at 03:50 *(routine)*

**Why:** V6. A confirmed record is invisible until someone presses a button, and the button's
delta report then waits for the next night. Senior-fall: every award a parent adds between now
and 2028 reaches her page the moment it is confirmed.

**Preflight Files:**
- `app/sections/catalog_inbox.py` (`_render_rebuild_snapshot`, the three "Rebuild the snapshot"
  notices, the `st.cache_data.clear()` note)
- `scripts/run_ingest.py` (`run_ingest(only_sources=...)`, `_carry_forward_prior_records`,
  `_build_guardrail_warnings`), `app/state.py` (`load_snapshot_cached`, `active_snapshot_path`)
- `deploy/catalog_sync.sh`, `deploy/update.sh`, `deploy/bootstrap.sh` (the timer install loop),
  the three existing `.timer` / `.service` pairs
- `tests/test_deploy_config.py`, `scripts/screenshot_app.py` (journey P2)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/screenshot_app.py --journeys --scratch-db
```

**Checklist:**
- [x] *Owner-assisted, first:* read the server's state and record it in the task notes (counts
      only): pending proposals, the newest verify report's outcomes, the snapshot's date, whether
      any unit is failed (`systemctl --failed`) *(read after the code was written; see the notes)*
- [x] One function builds the curated-only snapshot (the call the button makes today), used by
      the button, by confirm, and by a new `scripts/rebuild_snapshot.py`
- [x] Confirming a proposal or saving Add Award rebuilds at once; the notice reads "Confirmed.
      It's on her Opportunities page now." A blocked guardrail shows the existing error and keeps
      the old snapshot *(the blocked branch is written but was not walked; see the notes)*
- [x] Snapshot caches are keyed on the file's path **and** modification time, so a rebuild made
      by another process reaches every open session without a restart (the A.10 bug, one layer
      out)
- [x] `scripts/rebuild_snapshot.py --if-stale`: rebuilds only when a record file is newer than
      the snapshot; exit 0 when there is nothing to do, non-zero when a guardrail blocks
- [x] `deploy/scholarshipcoach-rebuild.service` + `.timer`: daily at 03:50, after the 03:30
      backup and before the 04:15 sync, `Persistent=true`, so the delta report rides the same
      night's sync. Added to `bootstrap.sh`'s install loop
- [x] The "Rebuild snapshot" button stays, in the page's closing expander, for a blocked run
- [x] `tests/test_deploy_config.py`: the rebuild timer sits between backup and sync; the unit
      runs as `coach` from the repo root
- [x] Journey P2 re-walked: form + 1, within budget (was 5 + form) *(measured 0 + form)*
- [x] *Owner, on the server:* install the new unit (re-run the idempotent `bootstrap.sh` as the
      admin login) and confirm `systemctl list-timers 'scholarshipcoach-*'` lists four
      *(2026-11-14: the two files installed by hand, which is bootstrap's timer step without its
      restart; four timers listed: backup 03:30, rebuild 03:50, catalog-sync 04:15, verify
      first Sunday 07:00)*
- [x] All five CI commands green

**Task notes (2026-11-14):**
- *Server state (2026-11-14, read by the owner after the code was written).* 0 pending
  proposals. Newest snapshot `scholarships_snapshot_20261011.parquet`: nothing has rebuilt on
  the server in five weeks. Newest verify report 2026-11-01: 40 records, 8 due, all 8
  unchanged (0 changed, 0 dead link, 0 blocked, 0 error). No `scholarshipcoach-*` unit failed;
  the only failed unit was `user@0.service`, root's user manager, which is unrelated to the app.
  None of this changed the code.
- *Installed (2026-11-14).* Deployed `63abf8b`, then the rebuild unit and timer installed and
  enabled as the admin login. Read the unit's journal with `sudo journalctl -u
  scholarshipcoach-rebuild`: the admin login is not in `adm`, so without `sudo` the journal
  shows "No entries".
- *The notice.* A program gets the plan's wording. A scholarship gets "Confirmed. It's ranked in
  Find now.", since Opportunities lists a scholarship only when it names her grade. When the
  rebuild fails or a guardrail blocks it, the notice is an error: the record is confirmed, the
  run's own reason follows, the prior snapshot is untouched. That branch was not exercised in
  the app; the script's non-zero exit on a block is tested.
- *Caches.* `load_snapshot_cached`, Find's ranking, Opportunities and the sidebar delta take the
  file's modification time in their key, and `catalog_inbox` no longer clears `st.cache_data`.
  `state.active_snapshot_path` now looks up the newest snapshot on every run and no longer
  keeps the path a session first saw: the nightly rebuild writes a new dated file, which a
  remembered path would never reach. The walk shows this across sessions (her browser was open
  before P2's confirm wrote a new snapshot, and her Opportunities showed the program). A
  same-day rewrite of one file is covered by the key, not by the walk.
- *Stale.* `--if-stale` also rebuilds when there is no snapshot, and when the records folder
  itself is newer than the snapshot, which is how a removed record shows. A merged pull and the
  verify pass's `checked_on` stamp both touch record files, so the night after either rebuilds.
- *P2 as walked.* Form, then her Opportunities shows the program: 0 + form against 1 + form.
  Suggesting it to her is not part of the revised journey and still costs a nav + 1 from Catalog
  & Inbox; the walk now does it as S3's setup step, since S3 reads the suggested card. P3 and P4
  are unchanged (2 against 1, 1 + form against form); Task 3.1 revises P3.

---

## Task 1.2: Deadlines From the Monthly Verify Pass *(judgment)*

**Why:** V4, V7, V8. Thirty curated records are placed by a month, not a date, so none of them
can become a dated to-do. The verify pass already visits each page when the new round is due; it
fails to read the date. Senior-fall: by fall 2028 every award she applies to should have had its
real date proposed by the machine and confirmed by a parent, not typed from memory.

**Preflight Files:**
- `src/catalog/verify.py` (`compare_record`, `cycle_checkpoint`, `MIN_CHANGE_CONFIDENCE`,
  `_record_proposal`), `src/catalog/inbox.py` (`reject`, `_default_proposal_id`)
- `src/ingest/extract_common.py` (`parse_date_candidates`, `extract_field_value`),
  `src/ingest/prefill.py` (`RegexExtractor`, `_candidate_confidence`)
- `docs/operations.md` ("What prefill missed", "Monthly catalog re-verification")
- `src/rank/timeline.py` (`project_next_deadline`), the verify and prefill tests

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/verify_catalog.py --since-days 0 --inbox-dir $env:TEMP\coach_inbox --reports-dir $env:TEMP\coach_verify
```

**Checklist:**
- [x] A fixture set first: saved page text for the failure cases already written down in
      `docs/operations.md` (year-less "March 1"; an open date beside a labeled deadline; a
      sidebar post date; "first Monday in February"), each with the date a person would read
- [x] Year-less dates: a labeled month and day with no year becomes a candidate dated to its
      next occurrence, but only when it agrees with the record's `cycle.deadline_month`. It is
      proposed with a note saying the year was inferred
- [x] A date under a deadline label outranks any other date on the page; a date in a sentence
      about opening ("opens", "will open", "available") is never the deadline
- [x] A proposed deadline must be in the future and within 14 months. A past date is not a new
      cycle
- [x] When the page's new date and the record's `cycle.deadline_month` disagree, the proposal
      carries both the deadline and the corrected month, and its note says so
- [x] Rejections are remembered (V8): a proposal whose changed fields equal those of a rejected
      proposal with the same id is not written again; it is counted in the report as
      `previously_rejected`. Applies to every proposal kind *(the Add Award form opts out; see
      the notes)*
- [x] Measured and recorded in the task notes: of the fixtures, how many dates are read correctly
      before and after; and on a dry run over all 40 records (to temp directories), how many
      deadline proposals result and how many of those a person agrees with. A wrong deadline
      proposal is worse than none: if precision on the dry run is under 9 in 10, tighten until it
      is, and record what was left unread
- [x] `docs/operations.md`: the "What prefill missed" items this fixes are marked fixed, the rest
      stay
- [x] All five CI commands green

**Task notes (2026-11-14):**
- *Fixtures.* `tests/resources/verify_deadline_fixtures.json`: ten page texts in the shape of the
  real pages. Four are the `docs/operations.md` cases (year-less "DEADLINE March 1", an open date
  beside a labeled deadline, sidebar post dates, "first Monday in February"). Six are guards: a
  deadline in another month, "closes on" under a label, last cycle's date still posted, several
  different labeled deadlines on one page, a year-less date in another month, and a calendar
  that puts the date before its label. Each records the date a person would read, and a test
  asserts what verify proposes for each one.
- *Fixtures measured.* Before: of the 8 with a date to read, 2 read correctly, 6 missed, 0 wrong.
  After: 6 read correctly, 2 left unread on purpose (the year-less date in another month, and
  "2027 JAN 12 Application Deadline"), 0 wrong. The 2 with nothing to read propose nothing, before
  and after.
- *Dry run, all 40 records.* Every page was fetched once into a scratch cache, and verify ran
  over a temp copy of the records. Before: 0 deadline proposals. The first version of the change
  proposed 12, and a person would agree with 1. Ten AFCEA pages date the transcript-mailing
  cutoff with "no later than May 1, 2027", and Duke's page dates the immunization upload with
  "no later than June 1, 2027". Tightened so that "due", "no later than" and "closes on" count
  only in a sentence (capped at 100 characters, because bulleted pages have no full stops) that
  says "deadline" or "application", or with a colon. "Deadline" itself still needs no sentence.
  After tightening: 1 deadline proposal (NCAE MLK Jr., "first Monday in February", proposed as
  2027-02-01 with the inferred-year note), and a person agrees: 1 of 1. The live run behind the
  validation command matches it.
- *Why so few today.* Most dated pages still show last cycle's date ("Deadline March 2, 2026" on
  the Gaston pages, March 13, 2026 for Space Grant). Those are now read, and refused by the past-date
  rule. They should start proposing when the sponsors post the 2027 dates, around December–January,
  which is when the cycle checkpoints re-read them.
- *Left unread.* Dell ("2027 JAN 12 Application Deadline": the date comes before its label, and
  the record's month, 2, disagrees, which it would correct). SWE (two different labeled deadlines
  for different scholarships: ambiguous, so nothing is proposed). Golden LEAF ("March 1, 2027
  Deadline to submit": the date comes before the label, but it already matches the record). An
  unlabeled year-less date is never read.
- *Rejections.* `inbox.propose` refuses (`PreviouslyRejectedError`) when `inbox/rejected/` holds a
  file under the same id with the same changed fields: the diff's new values, or the whole record
  for a new award, leaving out `provenance` because verify stamps a fresh `checked_on` every
  run. The Add Award form passes `respect_rejections=False`: there a person enters and confirms
  in one step, and an earlier rejection must not block them. Verify counts these as
  `previously_rejected` in the report and the CLI line, and does not stamp `checked_on`, since the
  page still disagrees.
- *Side effect on the Add Award form.* An opening date ("Applications open ...") is no longer
  offered as a deadline candidate. Two tests that expected it were updated.
- *Validation command.* `verify_catalog.py` as listed in this task leaves `--records-dir` at its
  default, so it stamps `checked_on` into the committed records. It was run with `--records-dir`
  pointed at a temp copy. On that run the AFCEA NC chapter page returned 404, although it had
  fetched normally about twenty minutes earlier. That dead-link proposal comes from the site, not
  from this change.

---

# Phase 2 — Discovery

## Task 2.1: Feed Rows Become Proposals *(judgment)*

**Why:** V3, V9. A feed row can rank for a parent, but it cannot reach her until a person gives
it a grade and a date. Nothing asks a person to. Senior-fall: modest, and stated as such — the
one enabled feed is small and mostly undated. The value is that the path exists and is quiet, so
adding a better feed later is configuration.

**Preflight Files:**
- `src/ingest/sources/open_scholarships.py`, `src/ingest/registry.py`, `data/catalog/sources.json`
- `scripts/run_ingest.py` (`_collapse_cross_source_duplicates`, `_apply_source_health`,
  `_load_prior_source_counts`), `src/catalog/inbox.py`
- `src/rank/stage1_eligibility.py` (the grade check and `unverified`), `src/profile/store.py`
- `tests/test_catalog_inbox.py` (the `feed` tests that already exist)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
```

**Checklist:**
- [x] `src/catalog/discovery.py`, pure where it can be: given the feed rows of this run, the
      state of the last run and the curated records, return the rows to propose (D4) and the
      counts of everything left out, by reason
- [x] Proposal id `feed-<source>-<source id>`, so a re-run replaces its own proposal and two
      sources never collide. The record carries a suggested `catalog_id`, `trust: unverified`,
      the feed's provenance and license line, and the fields the feed gave; the diff is against
      the curated record when one matches
- [x] Left out, each with a counted reason: covered by a curated record (the existing
      cross-source match); rejected before with the same content (Task 1.2's rule); fails the
      profile screen; over the cap
- [x] State in a git-ignored file under `reports/` (row fingerprints from the last run). A first
      run with no state writes the baseline and proposes nothing
- [x] The profile screen runs Stage 1 with the saved profile and ignores the grade axis, so an
      award she can apply to in two years still counts. With no saved profile it screens nothing.
      Code and tests use the demo profile; nothing about the real one is written anywhere but
      the git-ignored inbox
- [x] The inbox shows the feed badge, the source, and "why this was proposed" (new, or which
      field changed)
- [x] Tests: deterministic ids across two runs; a changed deadline re-proposes, an unchanged row
      does not; rejected-then-unchanged stays out, rejected-then-changed returns; the cap; the
      baseline run; a row covered by a curated record
- [x] All five CI commands green

**Task notes (2026-11-14):**
- *Shape.* `plan_feed_proposals` is the pure part (rows, last run's fingerprints, curated
  records, a Stage 1 profile, a rejection predicate); `run_feed_discovery` loads state and
  records, writes proposals and the new state, and has a `dry_run` for Task 2.2. Nothing calls
  it yet: Task 2.2's `scripts/discover.py` does.
- *The cross-source match moved.* `_collapse_cross_source_duplicates` and its URL helpers moved
  from `scripts/run_ingest.py` to `src/ingest/dedupe.py` unchanged, so `src/` does not import
  from `scripts/`. Discovery runs it over the curated records plus this run's changed rows.
- *Covered vs. diffed.* A row that matches a curated record is left out as
  `covered_by_curated` when every deadline, status or amount the feed states equals the
  record's. When the feed states a different one, the proposal is the curated record with the
  feed's values applied (its catalog id, its trust), so the inbox diff is against it. A feed
  `status: unknown` or an empty field never counts as a disagreement.
- *Left-out reasons* in the result: `baseline`, `unchanged`, `covered_by_curated`,
  `previously_rejected`, `failed_profile_screen`, `over_cap`. A rejected proposal that would be
  over the cap counts as rejected, not over the cap.
- *Cap order.* Soonest dated deadline first, then undated, then by id, so the 15 kept are the
  urgent ones.
- *State* is per source in `reports/discovery/feed_state.json` (now git-ignored): row key to
  deadline, status and amounts. A source missing from the file is a baseline run, so adding a
  second feed later baselines only that feed.
- *Provenance and license.* The schema's `provenance` allows no license field, so the record
  keeps `source_kind: structured_feed` and the feed's `added_on`, and the attribution line goes
  in the record's `notes`, where it survives a confirm. `verified_on` / `verified_by` are left
  null: the feed's own check is not a person opening the page.
- *Screen.* Stage 1 with `include_unconfirmed=True` (a person confirms anyway) and
  `GRADE_LEVEL_MISMATCH` ignored. Run over the sample feed with the demo profile: the two
  Nevada rows fail the state check, the two national ones pass.
- *Inbox.* Proposals carry optional `source` and `why` metadata. A feed proposal shows the feed
  badge, "Source: Open Scholarships" and "Why this was proposed", tested with `AppTest`. Not yet
  seen at 1366px in the real app: the inbox is a parent page and Task 2.2's first run is the
  first time a real feed proposal exists.
- *Not measured.* No live fetch of the feed in this task. Task 2.2 runs it, and its first run on
  the server writes the baseline only.

---

## Task 2.2: Weekly Discovery Timer and Health *(routine)*

**Why:** V1, V2. Senior-fall: the feed's own corrections (a date posted, an award closed) reach
the snapshot within a week instead of whenever someone remembers.

**Preflight Files:**
- `scripts/run_ingest.py` (`main`, exit codes, the ingest report), Task 2.1's module
- `deploy/` units and `bootstrap.sh`, `tests/test_deploy_config.py`
- `docs/operations.md` ("Catalog sync", "Deploying an update")

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/discover.py --dry-run
```

**Checklist:**
- [x] `scripts/discover.py`: runs ingest for the enabled feed sources with today's guardrails
      (the > 50% drop still blocks the write), then Task 2.1 over the result. Writes
      `reports/discovery/discovery_<UTC stamp>.json`: per source `succeeded` / `skipped` /
      `failed`, records, the zero-record regression flag, proposals written, and the left-out
      counts. `--dry-run` fetches and reports, writes nothing
- [x] Exit code: 0 for a completed pass, including one that found nothing new; non-zero when a
      source that returned records before returns none, or a guardrail blocks. systemd then shows
      the unit failed, and the digest says so (Task 3.1)
- [x] `deploy/scholarshipcoach-discover.service` + `.timer`: Sundays 05:00, `Persistent=true`,
      as `coach`, in `bootstrap.sh`'s loop
- [x] `tests/test_deploy_config.py`, for every script a timer starts: it contains no `git pull`,
      `pip`, `systemctl restart` or `update.sh`; nothing under `scripts/` or `src/catalog/` that a
      timer reaches calls `confirm(`; nothing a timer reaches writes under
      `data/catalog/records/` except verify's stamp (principles 1 and 2)
- [x] `docs/operations.md`: a "Weekly discovery" section in the style of the verify section:
      what it does, what each outcome means, how to run it by hand, where the report is
- [x] *Owner, on the server:* install the unit; one manual run; the baseline is written and the
      inbox is unchanged
      *(2026-11-14: deployed `3b20822`; the two files installed by hand as root; five timers
      listed, discover next Sun 05:00 UTC. Manual run: 98 rows, `succeeded`, baseline, nothing
      proposed, exit 0; `feed_state.json` and the report written as `coach`; inbox 0 before and
      after)*
- [x] All five CI commands green

**Task notes (2026-11-14):**
- *Rows before dedupe.* `run_ingest` gained `parsed_rows` (filled with each source's rows before
  the cross-source collapse, which would otherwise drop a feed row that disagrees with a curated
  record) and `dry_run` (fetch, parse and guardrails; no raw cache, snapshot, delta or report).
  Ingest runs once per pass, not once for the snapshot and again for discovery.
- *Which sources.* `FEED_SOURCES` in the script (Open Scholarships today), intersected with what
  `sources.json` enables. Scrapers are not feeds even when re-enabled.
- *Exit 1* when any feed source is `failed` (a raised fetch, or the zero-record regression) or
  the guardrail blocks. A first-ever fetch that raises is a failure too: it is not a completed
  pass. A source that did not succeed, or any source under a blocked guardrail, is not planned,
  so its state stays and the next good run compares against the last good one.
- *Guards.* The deploy tests read every `*.timer`'s unit, follow the shell scripts it runs and
  the Python modules those import, and check: no `git pull` / `pip` / `systemctl restart` /
  `update.sh` in shell code or Python string constants; no call to `confirm`; and the only
  functions that write a path built from the records directory are `inbox.confirm` and verify's
  `_stamp_checked_on`. A further test asserts the detector finds exactly those two, so a clean
  pass is not a blind one. `tests/test_discover.py` runs the script end to end on the sample
  feed and checks the real records directory is byte-for-byte unchanged.
- *Measured.* `python scripts/discover.py --dry-run` against the live feed on the dev PC: 98
  rows, `succeeded`, baseline (no state on the dev PC), nothing written.
- *Installed (2026-11-14).* The admin login's `sudo` password was not to hand, and `coach`'s
  sudo covers only the app's restart, so the unit went in from `ssh root@coach` (Tailscale SSH
  allows root without a password). The run still goes as `coach`, per the unit.

---

## Task 2.3: The Watch List *(judgment)*

**Why:** V5, and the first consequence above. Local awards and summer programs are in no feed.
They are on a few dozen pages that change a few times a year, and the family already knows which
pages (`docs/operations.md`, "Where local awards come from"). Today each one is re-opened by hand
in the month someone remembers. Senior-fall: this is where the awards with the smallest applicant
pools, and every program she can do before senior year, will come from.

**Preflight Files:**
- `docs/operations.md` ("Where local awards come from", "What prefill missed")
- `src/ingest/prefill.py` (`Extractor`, `prefill_from_html`, `PrefillResult`),
  `src/ingest/http.py`, `src/catalog/verify.py` (`content_hash`, `classify_http_status`,
  `load_prior_hashes`)
- `src/catalog/entry.py`, `data/catalog/schema.json` (`kind`, `cost`, `grade_levels`)
- `deploy/privacy_check.sh`, `deploy/backup.sh` (what `data/private/` backs up)
- `docs/decisions.md` (2026-09-12 "Curated first, fed second, scraped last"; 2026-09-26 "One
  record per award")

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/check_watchlist.py --dry-run
```

**Checklist:**
- [x] `data/catalog/watchlist.json`, committed, with a small schema checked by
      `scripts/validate_catalog.py`: per entry an id, URL, label, `page` (`single` award or
      `listing`), `kind` (`scholarship` or `program`), the months it is worth checking, a
      `terms_note` with the day a person read the site's terms, `enabled`, notes. An entry with
      no `terms_note` does not run *(committed empty; Task 2.4 adds the entries)*
- [x] `data/private/watchlist.json` (D8): same shape, server only, merged at run time, backed up
      with the rest of `data/private/`. A test asserts the loader reads both and that the
      committed file is the only one in git. The committed list never holds her school's page
      *(guarded by the privacy term lists; see the notes)*
- [x] `src/catalog/watch.py`: fetch politely (the verify pass's rate), hash the cleaned text,
      compare with the last run. Unchanged pages cost one request and write nothing
- [x] A changed `single` page goes through `prefill_from_html` and becomes a `prefill` proposal
      (the kind that exists and nothing writes, V3), diffed against the record that shares its
      URL, if any
- [x] A changed `listing` page is diffed by its links: links that are new, on the same site, and
      match no record or rejected proposal are fetched and prefilled, at most 10 a page. One
      record per award, never one for the listing (2026-09-26 decision)
- [x] What cannot be read is reported, not guessed: a PDF, a postback portal, a blocked fetch, or
      a change with nothing extractable becomes a finding ("this page changed; open it") for the
      digest, with no proposal
- [x] **Programs are first-class:** an entry with `kind: program` proposes a program record. The
      regex extractor learns the fields a program card shows and prefill does not read today:
      cost ("free", "tuition", "$1,550"), grades ("rising 10th–12th graders"), and session dates.
      Measured against the six confirmed program records and their live pages, per field, before
      and after; recorded in the task notes
- [x] Deterministic ids `prefill-<watch id>-<slug>`, the rejection rule from Task 1.2, the cap,
      and a `reports/watchlist/` report in the shape of the verify report
- [x] The inbox says which watch entry a proposal came from
- [x] Tests: unchanged page writes nothing; a changed single page proposes once across two runs;
      a listing gains a link; a rejected link stays out; a PDF becomes a finding; a private
      entry's proposal is written only to the git-ignored inbox
- [x] All five CI commands green

**Task notes (2026-11-14):**
- *Shape.* `src/catalog/watch.py` loads both lists (a file that fails its schema contributes no
  entries), checks each entry that is enabled, has a `terms_note` and is in season, and writes
  `prefill` proposals through `inbox.propose`, so Task 1.2's rejection rule applies unchanged.
  `scripts/check_watchlist.py` is the CLI; exit 1 only when a list is unreadable or invalid.
  The committed list is empty: entries need a person to read each site's terms (Task 2.4).
- *First sight is a baseline*, as for the feed (D4): a newly watched page proposes nothing until
  it changes. A page to read now goes through Add Award.
- *Findings.* PDF, postback portal (a changed listing with no new links whose page carries
  `__doPostBack` / `__VIEWSTATE`), refused fetch, dead link, nothing extractable (no title, or no
  date, amount or grade), and one the checklist did not name: a changed page whose readable
  fields still match its record ("changed, still matches the record; open it"). A network error
  is `error` with no finding, as in verify.
- *Records never name the entry.* The watch entry is in the proposal's `source`
  (`watchlist:<id>`) and `why`, which stay in the git-ignored inbox; `confirm` writes only the
  record. A private entry therefore leaves nothing in a committed record.
- *Her school's page.* No public test can name her school, so `scripts/validate_catalog.py`
  checks each committed entry against `data/private/privacy_names.txt` and
  `privacy_abbrevs.txt` when they exist (the dev PC, the server; CI skips it), the same lists
  `privacy_check.sh` uses, and prints only the entry id. It passes on the dev PC today.
- *Cap and retries.* 15 proposals a run, 10 links a listing. A link over either cap, or one whose
  write failed, is left out of the saved links, so the next change follows it; a single page
  over the cap keeps its old hash, so the next run sees the change again.
- *New-record deadline.* Last cycle's date still on a page is left out of a new record and named
  in its notes (Task 1.2's past-date rule), found on the live Gaston page below.
- *Program fields.* `find_cost`, `find_grade_levels` and `find_session_dates` in
  `extract_common`, read from the page's `<main>` without nav, header, footer or aside blocks.
  NCSSM's menu says "Summer programs for rising 5th- through 12th-graders" on every page: read
  from the whole page, Summer Ventures' grades come out 9–11 instead of 10–11. A grade is the
  grade when applying ("rising 11th" → 10), and rising wording beats current wording for the same
  students. Cost: one labeled amount ("Cost of Attendance", "Program fees", "tuition", "price")
  or free wording; a bare "fee", a rate ("/week", "per night") and "extra / application /
  deposit" charges are not the cost; two answers leave it empty. The Add Award form now seeds the
  grades; it does not seed Cost.
- *Measured, six program records against their live pages (fetched 2026-11-14).* Before: prefill
  read none of the three fields (0 of 6 each). After:
  - Grades: 6 of 6 match the records.
  - Cost: 4 read, 0 wrong. Girls Who Code 0, Summer Ventures 0, NC State $1,550 all match. Duke
    reads $6,300, the 2027 residential fee the page now posts, where the record still has null
    ("to be announced"). Left unread: the NCSSM Accelerator page states no cost (the record's
    $1,925 came from elsewhere), and UNC Charlotte gives only "$450/week".
  - Session dates: NC State 4 of 4, UNC Charlotte 1 of 1 (the "No camps June 29 – July 3" gap is
    skipped), Duke 3 of 3 plus "June 12 – July 23" from a typo on the page itself. The other three
    pages state no session dates, and none was invented.
- *Live run (scratch directories, real records read-only).* Four entries: Duke, NC State, Summer
  Ventures, and the Gaston foundation listing. Run 1: four baselines. Then the saved hashes were
  staled and three of the listing's 61 saved links removed, standing in for a week's change.
  Run 2: Duke proposed `cost null → 6300` with its session dates in the notes. NC State and
  Summer Ventures were findings ("still matches the record"). From the listing, Myers was
  recognized as already in the catalog, and Bohanan was proposed as a new scholarship ($2,500,
  last cycle's March 2, 2026 deadline left out). The third link's write failed on the dev PC's
  260-character path limit (the scratch path is long). That error is now logged and the run goes
  on, and the slug is the page's last path segment, which keeps ids short. Run 3: all four
  unchanged, one request each.

---

## Task 2.4: Watch List Timer and First Entries *(routine, owner-dependent)*

**Why:** A watch list with nothing on it watches nothing. Senior-fall: as Task 2.3.

**Preflight Files:**
- Task 2.3's files, `docs/operations.md` (the Gaston and statewide tables)
- `deploy/` units and `bootstrap.sh`, `tests/test_deploy_config.py`
- `docs/operations.md` "Scope rule" (awards that need Mecklenburg residency or a CMS school are
  out of scope)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/check_watchlist.py --dry-run
```

**Checklist:**
- [x] `deploy/scholarshipcoach-watch.service` + `.timer`: Sundays 05:30, after discovery and
      before the monthly verify, `Persistent=true`; in `bootstrap.sh`'s loop; covered by Task
      2.2's deploy tests
- [x] Claude drafts the committed entries from pages already named in `docs/operations.md` and
      the six program records' sponsors: the county foundation's listing, the statewide portal's
      dates page, the seven statewide awards, each program's page, two or three university
      pre-college listings from different institutions so no one school stands out. Each opened
      once and its terms read; `terms_note` filled or the entry left disabled with the reason
- [ ] *Owner:* confirms the committed entries, and adds the private ones on the server (her
      school's counselor page, employers, congregations)
- [ ] First run on the server: the baseline is written, nothing proposed. Second run a week
      later: the proposal count and the time to clear the queue are recorded here. If clearing it
      takes more than ten minutes, lower the caps before anything else is added (principle 4)
- [x] `docs/operations.md`: "Where local awards come from" now points at the watch list, and says
      which sources are still by hand (the postback portal, paper forms)
- [x] All five CI commands green

**Task notes (2026-11-14):**
- *Timer.* Sundays 05:30, `Persistent=true`, `TimeoutStartSec=1h`; sixth job in `bootstrap.sh`'s
  loop. `tests/test_deploy_config.py` pins the time between discovery and verify, and its
  timer-reach checks now cover `check_watchlist.py` (no confirm, no record writes, no deploy).
- *Entries drafted: 17.* Each page and its robots.txt fetched once with the bot's user agent;
  every robots.txt allows its page (AFCEA NC has none). Terms read where the site links them
  (NCAE, UNC Charlotte) and privacy pages elsewhere; none has a clause on automated access.
  Enabled: 12. Disabled with the reason in `notes`: the three ncsu.edu entries (Space Grant,
  engineering camps, pre-college listing; ncsu.edu's legal pages answer scripts with an empty
  202), Girls Who Code (terms render only with JavaScript), and PENC (403 bot challenge, yearly
  PDFs). "The statewide portal's dates page" was read as FFTC's resources page, which states the
  next cycle's dates before the postback portal opens.
- *Two entries went private.* `validate_catalog.py` found a privacy term in two drafted entries,
  so they live in `data/private/watchlist.json` on the dev PC instead of the committed list; the
  owner decides whether that stays. The pre-college listings committed are NCSSM's (enabled) and
  NC State's (disabled), with Duke's page as a single entry.
- *Dry run, 2026-11-14:* 17 entries, 9 baseline, 8 skipped (5 disabled, 3 out of season),
  nothing written.
- *Owner, 2026-11-15:* confirmed the 17 entries (15 committed, 2 private). Deployed `9393ca1`;
  the private list copied to the server and read there (dry run: 17 entries, both private ones
  marked). The family's own pages are not added yet, so the owner item stays open.
- *First run on the server, 2026-11-15* (by hand, `--ignore-months`, so the three out-of-season
  entries also have a baseline): 12 baseline, 5 skipped (disabled), nothing proposed, exit 0.
  Report `reports/watchlist/watchlist_20261115T152627Z.json`. The timer went in from the root
  login over Tailscale SSH; `systemctl list-timers` shows its next run, the second run, on
  Sunday 2026-11-22 at 05:30 UTC.

---

## Task 2.5: Measure an LLM Extractor on Confirmed Records *(judgment, optional, dev PC only)*

**Why:** D9. The regex extractor's misses are documented and some will survive Tasks 1.2 and 2.3.
The 2026-09-12 decision kept the LLM extractor off for cost and non-determinism, with the
`Extractor` protocol as the seam. This task produces the numbers that decision would need to
change; it changes nothing by itself. Senior-fall: passes only as a measurement, and only because
the watch list makes extraction quality the bottleneck. Skip it if Task 2.4's second run shows
the regex proposals are good enough to confirm quickly.

**Preflight Files:**
- `scripts/evaluate_llm_extraction.py` (its gold is parser output, by its own docstring),
  `src/llm/extraction.py`, `src/llm/client.py`, `src/llm/cache.py`
- `src/ingest/prefill.py` (`extraction_result_from_fields`, `Extractor`)
- `docs/decisions.md` (2026-09-12 "No generative LLM in this plan"),
  `docs/plans/SCHOLARSHIPCOACH_LLM_EXTRACTION_PLAN.md` (Tasks 6–9, deferred)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/evaluate_llm_extraction.py --gold curated
```

**Checklist:**
- [ ] `--gold curated`: the gold is the 40 hand-confirmed records, each compared with what an
      extractor reads from that record's live page. This is the first gold in the project that a
      person confirmed; the existing mode scores the LLM against the parser it would replace
- [ ] Both extractors scored on the same pages, per field: deadline, amounts, grades, cost,
      counties, GPA. Reported: right, wrong, and left blank. A wrong value and a blank are never
      added together
- [ ] Fabrication check: any extracted deadline or amount whose digits do not appear in the page
      text counts as fabricated, reported separately
- [ ] Cost and time per page, and whether a cached re-run is byte-identical
- [ ] Only public page text is sent. Runs on the dev PC with the owner's key; the server gets no
      key and no LLM call in this plan
- [ ] The result goes in `docs/decisions.md` either way. The bar for a later adoption task: zero
      fabricated deadlines, and more right deadlines and program fields than the regex extractor
      on the same pages. Adoption would be fill-only into proposals, never into records
- [ ] All five CI commands green

---

# Phase 3 — The Digest and Today (one deploy)

Tasks 2.1–2.2 and 3.1–3.3 are pushed and deployed together: Today's "new" section and the
digest's "waiting" section are empty without proposals and confirms flowing, and proposals with
nowhere to be announced are the problem this plan started from.

## Task 3.1: The Digest, and This Week for Parents *(judgment)*

**Why:** R4 in the Roles review, and V14. One builder answers "what changed and what is due" for
each person; three places show it (her Today, the parents' This Week, the weekly nudge).
Senior-fall: a missed deadline is the parents' main worry, and today nothing surfaces one unless
someone opens the right page.

**Preflight Files:**
- `src/store/todos.py` (`assemble_rows`, `todo_bucket`), `src/store/tracker.py` (`this_week`),
  `src/store/requirements.py`, `src/store/repo.py` (`list_activity`), `src/store/db.py`
  (`connect` applies migrations)
- `src/rank/opportunities.py`, `src/catalog/inbox.py` (`list_proposals`), the verify, discovery
  and watch reports
- `app/modes.py` (`pages_for_mode`, `SECTION_LABELS`), `app/sections/todos.py` ("Done recently")
- `design-system/scholarshipcoach/MASTER.md` ("Empty state", "Deadline states"),
  `pages/parent-mode.md`

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/screenshot_app.py --theme both --scratch-db --seed
python scripts/screenshot_app.py --journeys --scratch-db
```

**Checklist:**
- [ ] `src/digest.py`, pure: `build_digest(role, today, since, ...) -> Digest` with sections
      **Due** (overdue first, then the next 30 days, across to-dos, requirement dates, saved
      applications and posted program deadlines), **New** (since `since`), and for parents
      **Waiting on you** (pending proposals by kind, oldest first) and **Findings** (blocked
      pages to open by hand, dead links, changed pages nothing could read, and any source or
      unit that failed, in plain words)
- [ ] "New" for her: records open to her grade whose `provenance.added_on` is after `since`;
      to-dos someone else created; a deadline that went from projected to posted. "New" for a
      parent: proposals created, and what she finished. `since` is the last day before today on
      which that person opened the app, from `activity_days`; 14 days when there is none
- [ ] Her digest never carries the inbox, money totals, or anything from a page her role cannot
      open. A parent's never carries an unshared essay's title or a to-do's notes
- [ ] `app/sections/this_week.py` (Parent, D7): the digest, live. Each row links to where it is
      acted on; "Waiting on you" opens the proposal. What she did this week is listed open, not
      in a collapsed expander (closes P3). `pages_for_mode` and `SECTION_LABELS` gain the page
- [ ] `scripts/build_digest.py`: writes `data/private/digests/digest_<date>_<role>.json` (backed
      up nightly). It opens `coach.db` read-only without `apply_migrations`; when migrations are
      pending it writes nothing and exits non-zero with "update first" (principle 2). Tested with
      a database one migration behind
- [ ] `deploy/scholarshipcoach-digest.service` + `.timer`: Sundays 08:00, after discovery, the
      watch list and (on first Sundays) verify; in `bootstrap.sh`'s loop and Task 2.2's tests
- [ ] Empty digest: "Nothing new this week" with the next dated item, per MASTER. Never a green
      box
- [ ] Journeys P5 (0 taps), P6 (1), P3 (0) added to `scripts/screenshot_app.py` with a seeded
      proposal and a seeded finding; 400px and 1366px, light and dark, looked at
- [ ] Tests: section contents per role; `since` from activity; nothing private in the wrong
      role's digest; the read-only connection
- [ ] All five CI commands green

---

## Task 3.2: Send the Nudge *(routine, owner-dependent; after D2 and D3)*

**Why:** A digest nobody is told about is a page. Senior-fall: as Task 3.1.

**Preflight Files:**
- Task 3.1's script and unit, `deploy/backup.sh` (how `/etc/scholarshipcoach/restic.env` is read)
- `docs/operations.md`, `tests/test_deploy_config.py`

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/build_digest.py --send --dry-run
```

**Checklist:**
- [ ] With D2 at its default: `build_digest.py --send` sends each recipient one plain-text email
      through SMTP (Python's standard library; no new dependency). Subject and body hold a count
      and the tailnet link, and nothing else. A test asserts the body contains no title, date,
      name or amount from the digest
- [ ] Recipients and credentials in `/etc/scholarshipcoach/digest.env`, mode 600, owned by
      `coach`, referenced by path only and absent from the repo (the `restic.env` pattern and
      test)
- [ ] Recipients by role; with D3 at its default only parents are listed
- [ ] Nothing to say, nothing sent. A failed send exits non-zero and is a finding in the next
      digest; it never blocks the digest file
- [ ] `docs/operations.md`: what is sent, what never is, how to stop it (remove the file)
- [ ] *Owner:* SMTP account, `digest.env` written on the server, one test send received on a
      phone, and the link opens the app on the tailnet
- [ ] If D2 is changed to in-app only, this task is closed as not needed
- [ ] All five CI commands green

---

## Task 3.3: Today — Her Home Page *(judgment)*

**Why:** V10, V14. My Path is a checklist of eight equal cards; the one that is overdue looks like
the one due in two years. Today is one screen in the order a person would say it: what is late or
close, what is new, how it is going. My Path becomes the detail behind it. Senior-fall: this is
the page that has to work in fall 2028, when the overdue thing is an application.

**Preflight Files:**
- `src/digest.py` (Task 3.1), `app/sections/my_path.py` (`coming_up`, `_render_coming_up`,
  the home links), `app/sections/todos.py` (`render_row`, `_tick`),
  `app/sections/opportunities.py` (`_render_card`, `_add_to_todos`)
- `app/modes.py` (`pages_for_mode`, `render_home_links`), `app/main.py`
  (`_forget_unservable_path`: the home page is served only at `/`)
- `design-system/scholarshipcoach/pages/student-mode.md` ("My Path (home)", "Layout overrides"),
  `MASTER.md` ("Product and tone", "Empty state")
- `scripts/screenshot_app.py` (`journeys`, `seed_path`, `SEED_REQUIREMENTS`: the seeded PSAT is
  already overdue)
- Roles & Product Plan: D5, success criterion 2, the gate's fallback table

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/screenshot_app.py --theme both --scratch-db --seed
python scripts/screenshot_app.py --journeys --scratch-db
```

**Checklist:**
- [ ] `app/sections/today.py`, Student home, in this order:
      1. **Due**: overdue rows first (the only red on the page), then the next 14 days. Each is
         a to-do row she can tick here. At most five; "N more" links to To-dos
      2. **New since <day>**: at most three, each with a "New" badge: a program or award open to
         her grade (with "Add to my to-dos" on the row), a to-do from her family, a date that
         was posted. Absent when there is nothing new; never an empty box
      3. **The number** (D6) as one bordered `st.metric`, and under it one link line with her
         path progress ("Your path: 5 of 12 on track") to My Path
      4. Links to the other student pages, until Task 4.2 replaces them
- [ ] At 400×860 in both themes: the first Due row, or "Nothing due in the next two weeks" with
      the next dated item, is visible without scrolling
- [ ] The words follow `student-mode.md`: relative time first, spoken to her, no money as
      pressure, no operator vocabulary, nothing about the inbox
- [ ] Fallback (principle 6): with My Path and Opportunities removed from `pages_for_mode`,
      Today renders Due and the number from To-dos alone. Tested by building the page list that
      way
- [ ] `pages_for_mode`: Today first in the Student list; My Path second. `/my-path` now serves
      My Path and `/today` is the path `_forget_unservable_path` clears. Old bookmarks still open
      something sensible; checked in Edge
- [ ] My Path loses "Coming up" (Today has it) and keeps the checklist and Activities & Honors
- [ ] Parent mode: Today appears under "Her path" as her view, read-only, with the "Her view"
      badge
- [ ] `student-mode.md` gains a "Today (home)" section and the layout rule moves from My Path to
      Today; `MASTER.md`'s nav icon list gains Today (`today`, retired with This Week, returns)
- [ ] Journeys S8 (0), S9 (1), S10 (1) and the revised S1 (0) added and walked; the seed gains a
      record confirmed "yesterday" and a parent to-do so New is populated
- [ ] Tests: section order and caps; overdue outranks due soon; New is absent when empty; the
      fallback page list
- [ ] All five CI commands green

---

## Task 3.4: Small Feedback Moments *(routine)*

**Why:** V13. Ticking the thing she came to do changes a checkbox and nothing else. Senior-fall:
small, but it is the difference between a list and something that noticed.

**Preflight Files:**
- `app/sections/todos.py` (`_tick`), `app/sections/my_path.py` (`_set_status`),
  `app/sections/today.py`, the six existing `st.toast` calls
- `design-system/scholarshipcoach/MASTER.md` ("Voice and formatting", "Alerts", "Motion")

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/screenshot_app.py --journeys --scratch-db
```

**Checklist:**
- [ ] Ticking a to-do and setting a requirement to Met each give one `st.toast`, in MASTER's
      voice ("Done. 2 left this week."), from a pure, tested text function. Un-ticking says
      nothing
- [ ] The last item of the week gets its own line ("That's everything for this week."). No
      balloons, no confetti, no streaks, no exclamation marks
- [ ] "New" badges (blue, with an icon) on Opportunities cards and To-do rows that are new since
      her last visit, from the same `since` as the digest. They clear on the visit after
- [ ] Journey S9 asserts the toast text
- [ ] All five CI commands green

---

# Phase 4 — Hierarchy and Navigation

## Task 4.1: Card Hierarchy on My Path, Opportunities and Find *(judgment)*

**Why:** V10, V11, V12. Every card has the same weight, so the page says nothing. Senior-fall: the
same cards carry her senior-year applications.

**Preflight Files:**
- `app/sections/my_path.py` (`_render_requirement`, `_render_path`),
  `src/store/requirements.py` (`group_requirements`, `due_by_text`)
- `app/sections/opportunities.py` (`_render_card`, `card_badges`, `money_badge_text`, `render`),
  `app/sections/find.py` (`_render_award_card`, `award_badges`)
- `app/helpers.py` (`deadline_badge`, `phone_width_css`, the card selectors)
- `design-system/scholarshipcoach/MASTER.md` ("Award card", "Deadline states"), both `pages/*.md`
- `tests/test_phone_layout.py`

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/screenshot_app.py --theme both --scratch-db --seed
python scripts/screenshot_app.py --journeys --scratch-db
```

**Checklist:**
- [ ] **Status once.** On a requirement she can set, the segmented control is the status and the
      badge is dropped; on every other row the badge stays and there is no control
- [ ] **Urgency by position first.** Within a My Path group: overdue, then dated soonest, then
      undated. A dated requirement shows the shared `deadline_badge`, so an overdue PSAT is red
      at the top of "This semester" and cannot look like Calculus. "Next year" and "Senior year"
      are collapsed expanders with counts; "This semester" is open
- [ ] **Tint only if it can be done from theme tokens.** A left border on overdue and due-soon
      cards through a keyed container and `phone_width_css()`, if the status colors can be
      referenced without a hex value in Python or CSS. If they cannot, position and the badge
      carry it, and the task notes say so. Color never carries the meaning alone
- [ ] **Opportunities does not open on an empty box.** With nothing open now, the page leads
      with one line ("Nothing to apply to this week. Next: <title> opens Feb 1") and goes
      straight to Opens later, soonest first
- [ ] **A known amount is emphasis; an unknown one is silence.** "Amount not listed" and "Cost
      not listed" leave the card face on both pages and are stated in the expander. On Find, a
      card with no amount has no h3; the title carries it. On her card a known amount or "Free"
      stays a badge (her headline is still the next step and its date)
- [ ] **One action on the face.** Her card: "Add to my to-dos" until it is on her to-dos, then
      Apply. Find: Save until saved, then Apply. The other actions (Apply before saving, "Suggest
      to her") move into the expander
- [ ] Measured at 400px, recorded: median card height on Opportunities and Find before and
      after. The student card's ≤ 260px target is settled here: met, or revised in
      `student-mode.md` with the measurement, as Find's was
- [ ] `MASTER.md` and both page files updated in the same commit (card diagrams, the one-action
      rule, where unknown amounts go)
- [ ] Journeys re-walked. Moving "Suggest to her" into the expander adds a tap to P1 (2 + nav
      becomes 3 + nav, still within its budget); S3 and S10 unchanged. Any overrun recorded
- [ ] Tests: sort order within a group; badge or control, never both; no "not listed" badge on a
      card face; `test_phone_layout.py` for any new selector
- [ ] All five CI commands green

---

## Task 4.2: A Bottom Tab Bar on the Phone *(judgment: a spike with a verdict)*

**Why:** V15, R3. Three taps to change page, on the device she uses.

**Said plainly:** Streamlit 1.54 has no bottom tab bar. `st.navigation` draws a top bar or a
sidebar, and at 400px it folds into the sidebar whatever is asked. What can be built is a keyed
container of `st.page_link`s, pinned to the bottom of the screen by a `position: fixed` rule in
`phone_width_css()`. That is a rule about Streamlit's page structure, which is not a public API,
and it has not been tried in this app. Known risks: an ancestor element that breaks fixed
positioning; the bar scrolling or jumping when the phone's keyboard opens in the essay editor;
iPhone's home-indicator area covering it; the bar going grey for a moment on every rerun; five
labels not fitting 400px. So this task is a time-boxed spike that ends in a written verdict, not
a feature that is assumed to work.

**Preflight Files:**
- `app/modes.py` (`render_home_links`, `pages_for_mode`), `app/main.py`, `app/helpers.py`
  (`phone_width_css`, its docstring on degrading safely)
- `docs/decisions.md` (2026-09-30, 2026-10-10), `tests/test_phone_layout.py`
- `design-system/scholarshipcoach/MASTER.md` ("Navigation and page chrome", "Where styling
  lives"), `pages/student-mode.md` ("Layout overrides")

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/screenshot_app.py --theme both --scratch-db --seed
python scripts/screenshot_app.py --journeys --scratch-db
```

**Checklist:**
- [ ] Spike, one session: `main.py` draws one `st.container(key="tabbar", horizontal=True)` after
      the page runs, in Student mode, with icon-and-label links to Today, To-dos, Opportunities,
      Essays and More. `phone_width_css()` pins `.st-key-tabbar` to the bottom below 640px, with
      the page's bottom padding raised to clear it and `env(safe-area-inset-bottom)` respected.
      Above 640px the container is hidden and the top bar is unchanged
- [ ] Pass bar, all of it, in Edge at 400px in both themes **and** on one real iPhone and one
      real Android phone over the tailnet: the bar stays put while scrolling; it is not over the
      last card; it does not cover the essay editor or its Save button with the keyboard open;
      each tab is ≥ 44px and its label is readable; the current page is marked by more than
      color; tapping changes page in one tap; nothing scrolls sideways; with the CSS removed the
      links fall back to an ordinary row at the end of the page
- [ ] No `st-emotion-cache-*` selector, no script injection, no component
- [ ] **Verdict recorded in `docs/decisions.md`** with the screenshots' findings. On a pass: the
      bar ships, the home-links row leaves the student pages, journey S11 is 1 tap. On a fail:
      D10's fallback ships instead (the same links as an ordinary row at the top of every student
      page, S11 is 2 taps counting the scroll as none), the fixed rule is deleted, and the
      decision entry says which check failed
- [ ] "More" opens a small page or popover listing What If, Recommenders and My Profile; decided
      in the spike by which is one tap and survives a rerun
- [ ] `student-mode.md` and `MASTER.md` describe what shipped
- [ ] Tests: the stylesheet scopes the rule to the phone breakpoint and the keyed container
- [ ] All five CI commands green

---

## Task 4.3: A Wider, Sortable Find for Parents *(routine)*

**Why:** V16, V14. Parents compare awards on a laptop, in one phone-width column, in an order
they cannot change. Senior-fall: Find is the parents' weekly page through 2028.

**Preflight Files:**
- `app/sections/find.py` (`render`, `rank_snapshot`, `_render_award_card`, `SHOWN_STATE_KEY`),
  `app/sections/applications.py`
- `design-system/scholarshipcoach/pages/parent-mode.md` ("Find Scholarships", "Spacing
  overrides": a dataframe over cards past about eight rows)
- `src/store/tracker.py` (application statuses)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/screenshot_app.py --theme both --scratch-db --seed
python scripts/screenshot_app.py --journeys --scratch-db
```

**Checklist:**
- [ ] A sort control above the results: Best match (today's order, the default), Deadline,
      Amount. Sorting reorders what Stage 3 returned; it never re-ranks or changes a score.
      Undated and unlisted values sort last. A pure, tested function
- [ ] At 1366px the cards fill the width in two columns; at 400px one, as now
      (`st.columns` stacks below 640px by itself)
- [ ] A Cards / List switch (`st.segmented_control`); List is a `st.dataframe` with title,
      sponsor, amount (currency), deadline (date), effort and a link, sortable by its headers
- [ ] "Dollars in play" as a bordered `st.metric` above the results and on Applications: the sum
      of the top amount of every saved application not yet decided, with the count beside it
      ("$41,500 across 6 applications"). Unlisted amounts are counted as "+ 2 not listed", never
      as $0. Parent pages only. With nothing saved, the empty state, not "$0"
- [ ] Journey P7 (1 tap) added; 1366px and 400px, light and dark, looked at
- [ ] `parent-mode.md` updated
- [ ] All five CI commands green

---

# Phase 5 — Close Out

## Task 5.1: Sweep, Docs, Bookkeeping *(routine)*

**Why:** Four new units, three new reports and a new home page that the docs do not describe are how the
next session works from the wrong picture.

**Preflight Files:**
- `README.md` ("What it does", "Data sources", "Quick start" step 6, "Future work")
- `docs/operations.md`, `docs/system_design.md`, `docs/decisions.md`, `CLAUDE.md` (plan table)
- `design-system/scholarshipcoach/MASTER.md` ("Pre-delivery checklist")
- Roles & Product Plan (D5, B.4, the P2/P3 overrun notes), Design Refresh Plan (Task 4.1)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/screenshot_app.py --theme both --scratch-db --seed
python scripts/screenshot_app.py --journeys --scratch-db
```

**Checklist:**
- [ ] MASTER's pre-delivery checklist run on every page this plan added or changed, both widths,
      both themes
- [ ] Every journey in "Journeys this plan adds" within budget, or the overrun recorded
- [ ] `docs/operations.md`: one table of every timer in order (03:30 backup, 03:50 rebuild, 04:15
      sync; Sundays 05:00 discovery, 05:30 watch list, 08:00 digest; first Sunday 07:00 verify),
      what each writes, and what a failure of each looks like in the digest
- [ ] `README.md`: the catalog row, the timers in step 6, and "Future work" (the watch list
      leaves "On the path"; what stays unscheduled matches this plan's list)
- [ ] Roles & Product Plan: D5 marked changed, B.4 marked partly moved here, the P2 and P3
      overruns marked closed with the task that closed them. Design Refresh Plan Task 4.1: the R3
      item points at Task 4.2's verdict
- [ ] The private-list `git grep` (Roles & Product Plan Task A.10) re-run over this plan's
      commits; the committed watch list read once more for anything that points at her
- [ ] All five CI commands green

---

## Not scheduled (fails the senior-fall test, or waits on a measurement)

| Idea | Why it waits |
|---|---|
| Adopting LLM extraction in prefill, the watch list or ingest | Waits on Task 2.5's numbers and a decision made on them. The 2026-09-12 decision stands until then |
| A crawler or feed for programs | No open source of programs exists (V5). The watch list covers the pages that matter to one student; a general crawler serves students this app does not have |
| Re-enabling the Bold.org or Scholarship America scrapes, or a CFNC crawl | Unchanged from `docs/decisions.md`: rows arrive `aggregator` and ineligible, at a cost in fetch load and terms exposure |
| Looking up dates for the 82 undated feed rows by machine | Most are out-of-state or closed to her. A row worth a date is worth confirming, which Task 2.1 already offers |
| Auto-confirming anything, however trusted the source | Breaks principle 1. A wrong record costs the family an application |
| Deploying or updating from a timer | Breaks principle 2 (`docs/operations.md`, "Deploying an update") |
| Push notifications through a public service, or full-content email | D2: her schedule would sit with a third party |
| Self-hosted push (ntfy) | D2 option D: a second service to run for one message a week |
| Per-person calendar subscription | Stays in Roles & Product Plan Task B.4, past the gate |
| Streaks, points, confetti, a "level" | MASTER: encouraging, not cute. A streak is pressure, and the gate measures whether she comes back without it |
| Dollars shown to her | `student-mode.md`: money is a fact on a card, never pressure. Dollars in play is a Parent number |
| Ranking opportunities | Roles & Product Plan Task B.2, past the gate |
| A native app or installable PWA shell | "Add to Home Screen" already exists; Streamlit is the stack through 2028 unless the gate says otherwise |
| A notification center or per-item read state | `activity_days` gives "since your last visit" without a migration. Per-item state would be a page-level trail of what she looked at, which D6 of the Roles plan ruled out |

---

## Which model runs what

| Task | Kind | Why |
|---|---|---|
| 1.1 Rebuild after confirm + timer | Routine | A function extracted, a cache key, a unit, tests that mirror existing ones |
| 1.2 Deadlines from verify | **Judgment** | Extraction rules where a wrong date is worse than none; a precision bar to hold |
| 2.1 Feed proposals | **Judgment** | What to propose and what to leave out decides whether the queue is usable |
| 2.2 Discovery timer + health | Routine | Script wiring, unit, deploy tests, a docs section |
| 2.3 Watch list | **Judgment** | Schema, the private overlay, listing diffs, program fields, privacy |
| 2.4 Watch timer + entries | Routine (owner-dependent) | A unit and a list drafted from pages already documented |
| 2.5 LLM measurement | **Judgment** (optional) | Evaluation design; what counts as fabricated |
| 3.1 Digest + This Week | **Judgment** | What each role may see; the read-only database rule |
| 3.2 Send the nudge | Routine (owner-dependent) | Standard-library SMTP, an env file, one test |
| 3.3 Today | **Judgment** | The page's order, words and fallback |
| 3.4 Feedback moments | Routine | Toast text functions and a badge |
| 4.1 Card hierarchy | **Judgment** | Design-system changes across three pages |
| 4.2 Bottom tab bar | **Judgment** | A spike whose verdict has to be honest |
| 4.3 Wider, sortable Find | Routine | A sort function, columns, a dataframe, a metric |
| 5.1 Sweep and docs | Routine | Checklists and documentation |

A routine task that turns up a design question stops and brings it back, and does not answer it.

---

## Execution Order

```
Any time, independent (may deploy as soon as each is green):
  1.1 rebuild after confirm + 03:50 timer      (routine)
  1.2 deadlines from verify + rejection memory (judgment)

Discovery, merged to main but held from the server:
  2.1 feed proposals                           (judgment; uses 1.2's rejection rule)
  2.2 discovery timer + health                 (routine)
  2.3 watch list                               (judgment)
  2.4 watch timer + first entries              (routine; owner adds the private entries)
  2.5 LLM measurement                          (optional; only if 2.4's second run says so)

Digest and Today, then ONE deploy for 2.1–2.2 and 3.1–3.3 together:
  3.1 digest builder + This Week + digest timer (judgment)
  3.3 Today                                     (judgment)
  3.2 send the nudge                            (routine; after D2, D3)
  3.4 feedback moments                          (routine)
      — deploy; owner installs the new units; first Sunday run watched —

Hierarchy and navigation:
  4.1 card hierarchy                           (judgment)
  4.2 tab bar spike → verdict                  (judgment)
  4.3 Find wide + sortable                     (routine)

  5.1 sweep, docs, bookkeeping                 (routine)

Then, across plans (D1 at its default):
  Cloud Hosting 3.1 onboarding → four weeks → the Roles & Product Plan's gate
  Design Refresh 4.1 sweep → 4.2 README screenshots
```

1.1 first because it is the smallest change with the most visible effect, and its cache fix is
needed by every later timer that rewrites the snapshot. 1.2 before 2.1 and 2.3 because all three
share the rejection rule, and it is cheapest to get right on the monthly pass. 2.1–2.2 are held
from the server until 3.1 and 3.3 are ready, so proposals never pile up unannounced and Today
never ships with an empty "New". 2.3–2.4 can deploy with that batch or a week behind it; the
first watch run only writes a baseline. 3.1 before 3.3 because Today is a view of the digest. 4.1
after 3.3 because Today reuses the rows 4.1 restyles, and restyling them first would mean doing
it twice. 4.2 after 3.3 because the bar's first tab is Today. 5.1 last, once the screens stop
changing.

No task adds a file under `src/store/migrations/`. If one turns out to need it, that is a change
to this plan, flagged before the push (CLAUDE.md Step 6).

## Success Criteria

1. A record confirmed in the app is on her Opportunities page with no further tap, and its delta
   report is pushed by the same night's sync.
2. The enabled feed is fetched weekly and the watch list checked weekly, without anyone running
   a command; a run that fails or returns nothing is named in that week's digest.
3. Re-running any automated source twice writes no duplicate proposal, and a rejected proposal
   does not return unless the page changed.
4. No timer writes a catalog record, deploys, or applies a migration; `tests/test_deploy_config.py`
   proves each.
5. Clearing a normal week's inbox takes a parent under ten minutes.
6. At least one program and one local award reach the catalog through the watch list in its
   first cycle, each confirmed by a person.
7. The share of curated records with a posted date for their next round rises each cycle,
   measured in the verify report; no wrong deadline is confirmed from a machine proposal.
8. At 400×860, her home page shows the most urgent thing she has, or the next dated one, without
   scrolling, and an overdue item is never drawn like one due in two years.
9. Something on her home page is different from her last visit whenever something she can act on
   was added, and says so.
10. On a phone she reaches any top student page in one tap, or the decision log says why that
    could not be built and what shipped instead.
11. No committed file, unit, digest email or commit message identifies her or her first target
    school; the watch list in git holds no page that points at her.
12. Every journey this plan adds is walked by `scripts/screenshot_app.py --journeys` within its
    budget, or its overrun is recorded.
13. All five CI commands green after every task.
