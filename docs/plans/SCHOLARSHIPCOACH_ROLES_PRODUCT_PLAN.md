# ScholarshipCoach Roles & Product Plan — Her Path to College, the Family's Money View

> Generated: 2026-10-03 | Scope from a roles and UX review: the current role model questioned
> against the code, the catalog and the snapshot, then today's journeys walked at 400px against a
> scratch copy of `coach.db` with a 10th-grade profile
> Owns **what** exists (roles, pages, what each page is for, data and catalog changes). The Design
> Refresh Plan owns **how it looks**; the Cloud Hosting Plan owns **where it runs**
> Executor: Claude via Claude Code, with owner-dependent tasks called out as such
> Est. effort: Phase A 11 tasks → go/no-go gate → Phase B 5 tasks (only past the gate)

---

## Why this plan exists

The app was built for a senior who applies for scholarships while her parents curate the list.
The student it serves is a 10th grader (class of 2029, NC, CS with computer engineering as the
alternate). Most awards are closed to her for two more years, and scholarships are mostly the
parents' worry. On 2026-09-27 the owner leaned toward a different split: Student mode as her path
to college, starting from her first target school, and Parent mode as the money and strategy view.
This review tested that lean against the code and the catalog, and walked every current journey at
400px.

**Privacy rule for this plan and everything it produces:** the GitHub repo is public. No
committed file (plans, docs, design system, catalog commit messages, screenshots) names her first
target school or anything that identifies her. Committed text says "her first target school". The
school, its requirements and its thresholds live in `data/private/` (the `colleges` table and its
requirement rows in `coach.db`). Catalog records for that school's own programs or scholarships
are fine as ordinary NC records. Nothing committed links them to her, and they are seeded
alongside records from other NC institutions so no one school stands out.

### Review findings

Walked at 400×860 with a scratch-DB journey walker built on `scripts/screenshot_app.py`
(scratchpad only, not committed; Task A.10 promotes it). The profile was set to 10th grade,
GPA 3.25, NC, Computer Science, in the scratch copy only. Snapshot `20260926`: 133 records,
34 curated.

| # | Sev. | Finding | Evidence |
|---|---|---|---|
| R1 | Critical | Student mode is built for a senior | Curated grade scopes: 18 records grade 12 only, 13 college only, 3 unscoped; none name grade 10. Student nav is This Week → Find → Applications → Essays → Recommenders → What If |
| R2 | Critical | Find tells a 10th grader "Apply now" for senior-only awards | 12 "Apply now" cards for the grade-10 profile, among them a senior-only science competition and a college scholarship for seniors. 102 of 133 snapshot rows (the feed) have no `grade_levels`. Stage 1 only flags grade as unverified when the *profile* grade is blank, so an unscoped record passes silently |
| R3 | High | Changing page at 400px takes 3 taps | At phone width the top nav folds behind `»`; after a destination is picked the sidebar stays open over the page and must be closed. Measured on every navigation in every journey |
| R4 | High | Nothing brings anyone back | This Week: green "Nothing due in the next two weeks." with a saved award 33 days out not shown. No notification; the `.ics` is a download or a script run, not a subscription |
| R5 | High | Her real near-term work has no home | Junior-year course registration (which advanced-math course decides her CS track), PSAT, summer program applications and GPA targets are not awards, so none of them can be saved, dated or ticked off |
| R6 | High | A parent-added award never reaches her | Catalog & Inbox confirm → "Rebuild snapshot" → she must open Find and press Run. Nothing tells her it is there |
| R7 | Medium | What If is not tied to a goal | GPA 3.25 → 3.70 "opens" 4 awards, two shown as "$0" (amount unknown), none connected to her first target school. The school's program GPA minimum and its full-cost scholarship's GPA minimum appear nowhere in the app |
| R8 | Medium | Steps and prompts are empty for most awards | A saved feed award says "No requirements recorded for this award"; Essays says "Every prompt on your open applications has an essay" when there are no prompts at all, so essay reuse has nothing to work on |
| R9 | Medium | One soft lock, no authorship, every draft visible | One family PIN; nothing records who added or finished anything; parents read every essay draft, including first drafts |
| R10 | Medium | First run is a 28-field form and a Run button | Profile page is 28 inputs (3,061px); Save prints a server file path; Find is empty until "Run Scholarship Coach"; pipeline wording (snapshot, weights, Top-N, `eligible=117 ineligible=16`) is shown to the student |

What already works and must be kept: grade-aware timeline buckets (`src/rank/timeline.py`), the
milestone list with grade scoping (`data/milestones.json`, Settings overrides), What If's
re-run of Stage 1 with overrides, the essay version history, the `colleges` table's
outside-award policy and net-price fields, and numbered migrations that apply on connect with a
backup before every server update (Cloud Hosting Task 1.2).

### Today's journeys at 400px (measured 2026-10-03)

Tap counts exclude typing. A page change costs 3 taps (R3).

| Journey (today) | Taps | Dead ends | What she must remember that the app could | What brings each person back |
|---|---|---|---|---|
| First run: profile → matches | 8 (profile 3, save 1, Find 3, Run 1) + 28 fields | Save shows a file path; Find empty until Run; 12 "Apply now" cards, several wrong for her grade | Which of the 28 fields matter | Nothing |
| Find → save → apply → track | 6 to an open application, then status changes | "No requirements recorded" for feed awards; Apply leaves the app with no return path | What the award asks for; to come back and mark it submitted | A deadline inside 14 days, if she opens This Week |
| Weekly check-in | 0 | "Nothing due in the next two weeks." with no next item and no link | That a deadline 15+ days out exists | Nothing; she has to think of opening it |
| Essay reuse | 3 to Essays | No prompts to link on feed awards; the empty message reads as success | Which draft fits which prompt | Nothing |
| Recommender request | ~9 (page 3, add 2, expand 1, pick award 2, add request 1) | None, once an award is saved | When to ask, and to follow up | A letter due date inside 14 days |
| Parent adds an award → student finds out | Parent: paste URL, read, 4 tabs, confirm, rebuild | The student is never told | That a parent added something | Nothing |
| Result → Colleges & Money | Status through to submitted, record result, page 3 | Colleges & Money shows "$0 / 0 / 0" until a college exists | Which college the money counts against | Nothing |

For a 10th grader these journeys are mostly empty. The only awards she can act on now are
competitions and the few unscoped records, and R2 shows she cannot tell which ones those are.

---

## Role model (decided 2026-10-03)

Three models were weighed:

| | A: Parent-led, student to-do list | B: Today's shared tracker | **C: Her path + parent money view (chosen)** |
|---|---|---|---|
| Student sees | To-dos her parents assign, her essays | Everything: find, save, apply, track | Her first target school's checklist, what to do in the next two years, opportunities open to her grade, What If toward her goals, shared to-dos, her essays |
| Parent sees | Everything; assigns tasks | Everything plus curation and money | Scholarships (Find, Applications, Outcomes), Colleges & Money, Timeline, catalog; her path read-only; shared to-dos |
| Why she would open it | She wouldn't; it is a chore list | Not until senior year | It answers her question: am I on track for that school? |
| Beyond the UI | One migration (`todos`) | None | Migrations (`todos`, requirements, essay sharing, activity); catalog `kind`; opportunities bypass Stage 2/3; R2 fix in Stage 1 |
| Main risk | The gate would measure compliance, not engagement | Two years of empty screens | It is the parents' theory; her school may provide a tool; content needs curating; Stage 3 does not fit opportunities |

**Chosen: C, ordered so it falls back to A.** Phase A ships what A and C share first: shared
to-dos, the parent money view, and the new page grouping. Only then does it add the parts that are
C's alone: My Path, Opportunities, and What If tied to her goals. If the gate shows she does not open
the app on her own, the student-path pages are removed and the app keeps model A. No data or
parent work is thrown away.

How the risks are handled:

- **The parents' theory, not hers.** No observation notes existed at review time. Task A.0 (owner)
  is a short conversation with her before My Path content is written, and the gate measures her
  behaviour, not the parents' impression.
- **Her school may already provide a planning tool.** Her high school is not yet known. My Path is
  kept narrow: one target school's requirements plus the family's own to-dos. It is not a general
  college planner. Phase B Task B.5 reconciles with the school's tool once that is known.
- **Opportunity content needs curating.** Phase A has exactly one opportunity type (summer
  programs) entered through the existing catalog form and re-checked by the existing monthly
  verify job. More types are Phase B.
- **Stage 3 does not fit non-scholarship opportunities.** It scores on amount, expected value and
  deadline urgency, so a $0 program always sinks. In Phase A, programs never enter Stage 2/3: they
  are filtered by grade and listed by date. Ranking them is Phase B Task B.2.

### Roles

| | Student (10th grade, phone, evenings) | Parents (laptop, weekly) |
|---|---|---|
| Cares about | Grades, classes, friends, activities; "am I on track for that school?" | Paying for college, not missing a deadline, which schools and which money |
| Worries about | Being nagged or watched; looking behind | A missed deadline; not knowing what she is doing; net price |
| Actually does | Picks next year's classes, takes the PSAT, maybe a summer program, writes for school | Curates awards, tracks applications and money, sets family dates, nudges |
| Would open it on her own if | It shows where she stands against her goal in one screen; ticking things off is quick; it is hers, not a surveillance channel | It is their only place for money and dates |
| Owns | Her to-dos, requirement progress (courses taken, scores), essays, recommenders | The catalog, the scholarship tracker, colleges and money, family milestones, target-school requirement definitions |
| Sees of the other | Parent to-dos assigned to "family" or to her | Her path read-only; her to-dos; essays she shared |
| Kept out of | Catalog, money, scholarship ranking internals, Settings | Her unshared essay drafts; editing her essays |

### Access (decided 2026-10-03)

- **Per-person identity from Tailscale, PIN as the fallback.** `tailscale serve` adds
  `Tailscale-User-Login` and `Tailscale-User-Name` headers to requests from devices owned by a
  signed-in tailnet user (not tagged devices). Streamlit 1.54 exposes request headers as
  `st.context.headers`. A `[family]` table in `.streamlit/secrets.toml` (server only, never
  committed) maps each login to `student` or `parent`. A student login gets Student mode only, even
  with the PIN; a parent login gets Parent mode without the PIN. A missing header or an unmapped
  login (local development, a tagged device) falls back to today's PIN behaviour.
- **Not yet verified on the server.** That headers reach Streamlit's websocket through
  `tailscale serve` on this server is Task A.2's first checklist item. If they do not, the PIN stays
  the only gate and A.2 records why.
- **Consequences:** her phone must sign in to her own Tailscale account (Cloud Hosting D3, now
  required). A shared household laptop signed in as a parent shows Parent mode. Authorship ("added
  by Dad") comes from the login, or from the mode when it falls back to the PIN.
- **Essays are private until she shares them.** Parents see an essay only after she marks it
  "Share with parents". The migration makes existing drafts private. This is only enforceable with
  identity: under the PIN fallback, Parent mode still hides unshared drafts, but a student who
  knows the PIN could open Parent mode.

---

## Owner decisions (defaults used unless changed before Task A.1)

| # | Decision | Default | Alternatives |
|---|---|---|---|
| D1 | Role model | **C, ordered to fall back to A** (decided 2026-10-03) | A parent-led; B today's tracker |
| D2 | Access | **Tailscale identity, family PIN as fallback** (decided 2026-10-03) | PIN only; identity only |
| D3 | Essay visibility | **Private until she shares** (decided 2026-10-03); existing drafts become private | Parents read all drafts, read-only (today) |
| D4 | First opportunity type | Summer programs (`kind: program`) | Competitions; pre-college courses |
| D5 | Student home page | My Path once A.6 ships; To-dos until then | To-dos always first |
| D6 | Measuring "opens it on her own" | Days opened per person, counted in `coach.db`, and **she is told** it is counted | Self-report at the gate only |
| D7 | Gate window and bar | 4 weeks after Hosting Task 3.1 onboarding; bar in "Go/no-go gate" | 6 weeks; a different bar |
| D8 | Where target-school content lives | Entered in the browser on the server into `coach.db` (one copy of record) | A private JSON seed under `data/private/` |
| D9 | Senior-year switch | Applications and Recommenders return to Student mode from spring of grade 11 (grade-driven, Phase B) | Owner flips it by hand in Settings |

---

## Design principles for this plan

1. **Nothing committed identifies her or her first target school.** See the privacy rule above.
   Every task that commits data or screenshots checks it; screenshots use the demo profile.
2. **Fallback-first order.** What models A and C share ships before what only C needs, so a failed
   gate costs pages, not data.
3. **Show her only what she can act on now.** A student surface never says "Apply now" for
   something her grade cannot apply to. Later things are labelled with when they open.
4. **Every person has a reason to come back.** Each page answers "what's next" and links to it; no
   success-green "nothing here" (MASTER "Empty state").
5. **Additive, tested migrations.** New tables and columns only; each migration is tested against
   a database at the previous schema; the server takes a backup before applying it (Cloud Hosting
   Task 1.2).
6. **The scholarship pipeline stays as it is**, apart from the R2 grade fix. Opportunities bypass
   Stage 2/3 in Phase A.
7. **Built with the Study Hall design system.** `design-system/scholarshipcoach/` is the spec; a
   deviation is a design-system edit in the same commit. The CLAUDE.md "UI/UX rules" styling order
   is binding.
8. **Seen at 400px, measured by journey.** Each page task re-walks its journeys and states tap
   counts against the budget in "Journeys after Phase A".
9. **Green gate every task:** the five CI commands in CLAUDE.md.

---

## Page list and grouping (replaces Design Refresh Task 1.3's provisional list)

`app/modes.py` `pages_for_mode` and `SECTION_LABELS` carry this; nothing else should need to change
when a page is added (Design Refresh Task 1.3's contract).

**Student** (`layout="centered"`). Five top destinations, the rest under **More**:

| Group | Page | Key / URL | Purpose (caption) | Built in |
|---|---|---|---|---|
| — | **My Path** (home) | `my_path` / `my-path` | Where you stand for your first-choice school, and what's next | A.6 |
| — | To-dos | `todos` / `to-dos` | Things to do, from you and your family | A.4 |
| — | Opportunities | `opportunities` / `opportunities` | Programs and awards open to you now, and what opens later | A.7 |
| — | Essays | `essays` | Your drafts. Parents see only the ones you share | A.9 |
| More | What If | `what_if` | See what a higher GPA or a test score opens up | A.8 |
| More | Recommenders | `recommenders` | The people who'll write your letters | existing |
| More | My Profile | `profile` | What the app uses to match you | existing |

Until A.6 ships, To-dos is the home page (D5).

**Parent** (`layout="wide"`), three groups:

| Group | Pages |
|---|---|
| Her path | My Path (read-only), To-dos, Opportunities, Essays (shared drafts only, read-only), What If, Recommenders |
| Money | Find Scholarships, Applications, Colleges & Money, Timeline, Outcomes |
| Family tools | Catalog & Inbox, Settings, Profile |

**Removed or moved:** This Week is replaced by To-dos. Its due items (application deadlines,
checklist steps, letters) appear in To-dos as linked rows, so nothing it showed is lost. Find
Scholarships and Applications move to Parent until the senior-year switch (D9, Phase B). Old
`/this-week` and `/find` bookmarks in Student mode land on the home page through the existing
hidden-page redirect in `app/main.py`.

**Timeline owner (settles Design Refresh Task 3.2's dependency):** Timeline is a Parent page in
the Money group showing every date: award deadlines, family milestones, her dated to-dos and
requirement deadlines. She sees her own dated items on My Path ("Coming up") and in To-dos, not on
a separate Timeline. Colleges & Money holds the target schools, and each college gains a
Requirements section (A.6) that My Path reads.

---

## Journeys after Phase A (derived from the role model)

Budgets are taps at 400px, excluding typing. A page change counts 1 when it is a `st.page_link`
from the current page, 3 through the folded nav (R3).

| # | Who | Journey | Budget | Brings them back |
|---|---|---|---|---|
| S1 | Student | Open the app → see where she stands for her first target school and the next 3 things | 0 | Next to-do with a date; a requirement "this semester" |
| S2 | Student | Tick a to-do or a requirement done | 1 | Parent sees it done (P3) |
| S3 | Student | Parent-suggested program → read it → "Add to my to-dos" | 2 from My Path | The program's application date in To-dos |
| S4 | Student | Ask a parent for something ("Pay the PSAT fee") → it shows on their To-dos | 2 + typing | Seeing it marked done |
| S5 | Student | What If: move GPA → see which target-school thresholds and awards it reaches | 1 from My Path | The gap on My Path ("0.25 to the program minimum") |
| S6 | Student | Write an essay → share it with parents | 2 from Essays | A parent comment arrives as a to-do |
| P1 | Parent | Weekly: Find (results on arrival) → save an award → assign a step to her | 3 + nav | Deadlines in Applications and Timeline |
| P2 | Parent | Add a program or award in Catalog & Inbox → it appears in her Opportunities with a "From your family" badge | form + 1 | Her "Added to my to-dos" shown on the parent To-dos |
| P3 | Parent | See what she did this week (done to-dos, requirement progress) without opening her essays | 1 from home | Her activity |
| P4 | Parent | Enter or edit the first target school's requirements in Colleges & Money | form | Her progress on My Path |

---

# Phase A — The Light Version

Built with the Study Hall design system. Tasks A.1–A.5 are what models A and C share; A.6–A.9 are
C's alone; A.10 closes the phase. Hosting Task 3.1 onboards the family after A.10.

## Task A.0: A Short Conversation With Her *(owner-dependent, no code)*

**Why:** The role model is the parents' theory until she has had a say. Ten minutes before My Path
content is written costs less than a month of the gate.

**Preflight Files:**
- This plan's "Roles" table and "Journeys after Phase A"

**Validation Commands:** none; the notes are the output.

**Checklist:**
- [ ] Owner asks her, without showing the app first: what she'd want to know about getting into
      her first-choice school; whether her school already gives her a planning tool (and its name);
      what she'd never want her parents to see; whether she'd open something like this on her own
- [ ] Owner tells her the app will count the days each person opens it (D6), and why
- [ ] Notes kept in `data/private/` (not committed); anything that changes D4–D9 or the page list
      is brought back before Task A.6 starts

---

## Task A.1: Migration Path and the Shared Data Model

**Why:** Every Phase A page needs new tables, and the server applies migrations automatically on
the first connection after an update. The path has to be proved before anything lands on the
family's database.

**Preflight Files:**
- `src/store/db.py` (`apply_migrations`, `connect`; migrations apply on connect, in name order)
- `src/store/migrations/0001_initial.sql` … `0004_colleges_money.sql`
- `src/store/repo.py`, `src/store/tracker.py` (row types, `this_week`)
- `deploy/update.sh`, `deploy/backup.sh` (backup before pull; the update log names new migrations)
- `tests/` covering the store (`grep -l "open_db\|apply_migrations" tests/`)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
```

**Checklist:**
- [x] Migration path checked and written down in the task notes: `deploy/update.sh` runs
      `deploy/backup.sh` before `git pull` and stops on failure; `apply_migrations` rolls back a
      failing file; `executescript` commits per statement, so a half-applied file is possible. Each
      new migration is therefore written to be safe to re-run (`IF NOT EXISTS`; `ALTER TABLE ADD
      COLUMN` only in a file that touches nothing else)
- [x] `src/store/migrations/0005_shared_todos.sql`: `todos` (`id`, `student_id` FK cascade,
      `title`, `notes`, `due_on`, `assignee` in `student|parent|family`, `created_by`,
      `created_role`, `done_on`, `done_by`, `source_kind` in
      `manual|application|checklist|letter|requirement|opportunity|milestone`, `source_ref`,
      `created_at`, `updated_at`), index on (`student_id`, `done_on`)
- [x] `0006_college_requirements.sql`: `college_requirements` (`id`, `college_id` FK cascade,
      `category` in `course|gpa|test|application|scholarship|program|other`, `label`, `target`,
      `due_by` (grade or date), `status` in `not_started|in_progress|met|not_needed`, `notes`,
      `source_url`, `verified_on`, `position`, timestamps); `colleges.priority INTEGER` (1 = first
      target) in its own file, `0007_college_priority.sql`, per the ALTER rule above
- [x] `0008_essay_sharing.sql`: `essays.shared_with_parents INTEGER NOT NULL DEFAULT 0` (existing
      drafts become private, per D3)
- [x] `0009_activity.sql`: `activity_days` (`login_or_role`, `day`, `pages_opened`, primary key
      on the first two); one row per person per day, no page-level trail
- [x] `src/store/repo.py` (or a small `src/store/todos.py`): typed rows and functions for todos,
      requirements, essay sharing and activity; pure helpers (`todo_bucket`, `requirement_progress`)
      unit-tested
- [x] Tests: each migration applies to a database built at the 0004 schema with rows in every
      table, and existing rows survive (counts before = after); re-opening is a no-op
- [x] All five CI commands green

**Task notes (2026-10-03):**
- *Migration path, as checked.* `deploy/update.sh` runs `deploy/backup.sh` (a `.backup` copy of
  `coach.db` plus `integrity_check`, then restic) before `git pull` and exits on failure, then
  prints the migration files the pull added. `src/store/db.py` applies unapplied files in name
  order on the first `connect` after the restart; a failing file raises `MigrationError` after
  `rollback()` and is not recorded in `schema_migrations`. But `executescript` autocommits each
  statement, so the rollback cannot undo statements that already ran: a file can be half-applied.
- *Rule that follows.* A migration file either holds only `CREATE … IF NOT EXISTS` statements
  (re-runnable from the top) or holds exactly one `ALTER TABLE … ADD COLUMN` (all or nothing).
  That is why `colleges.priority` moved out of 0006 into `0007_college_priority.sql`, renumbering
  essay sharing to 0008 and activity to 0009. Recovery from a half-applied file is then: fix the
  cause, restart; nothing needs hand-editing.
- *Where the code went.* Todos and `todo_bucket` (groups `overdue`, `this_week` ≤ 7 days,
  `coming_up` ≤ 60 days, `later`, `someday`, `done`) in `src/store/todos.py`; requirements,
  `requirement_progress` (`met` of everything not `not_needed`), `first_target_college`,
  `set_essay_shared` / `list_essays(shared_only=True)` and `record_activity` in
  `src/store/repo.py`. Tests: `tests/test_store_shared_model.py`.
- *Deploy note.* Pushing this adds five files under `src/store/migrations/`; the next server
  update migrates the family database after its backup.

---

## Task A.2: Identity From Tailscale, PIN as Fallback

**Why:** D2. Shared to-dos need an author, private essays need someone to be private from, and the
gate needs to know who opened the app.

**Preflight Files:**
- `app/modes.py` (`resolve_mode`, `render_mode_selector`, `parent_pin_from_secrets`)
- `app/main.py`, `app/state.py`
- `docs/plans/SCHOLARSHIPCOACH_CLOUD_HOSTING_PLAN.md` (D3, Task 2.2 `tailscale serve`, Task 3.1)
- `tests/test_app_modes.py`

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
```

**Checklist:**
- [ ] *Owner-assisted, first:* on the server, confirm the headers reach Streamlit through
      `tailscale serve`. Temporarily show, in Operator mode only, which of
      `Tailscale-User-Login` / `Tailscale-User-Name` are present in `st.context.headers`, and open the
      HTTPS name from two different family logins. Record the outcome here (header names only,
      never values in a committed file). If they are absent, stop the identity half and keep the
      PIN; record why
- [ ] `app/identity.py`: pure `role_for_headers(headers, family_map) -> Identity | None`;
      `family_map` read from `st.secrets["family"]` (absent → `{}`); logins compared case-folded
- [ ] `resolve_mode` takes the identity: a student login → Student mode only (PIN prompt hidden); a
      parent login → Parent mode with no PIN; no identity → today's PIN path unchanged
- [ ] The sidebar shows "Signed in as <name>" when identity is present; the mode switch is hidden
      for a student login
- [ ] Each session records one `activity_days` row per person per day (D6)
- [ ] Tests: header parsing, unmapped login falls back to PIN, a student login cannot reach Parent
      even with the PIN in session state, missing secrets table
- [ ] `docs/operations.md`: a short "Who sees what" note on the `[family]` secrets table (example
      logins are placeholders)
- [ ] All five CI commands green

---

## Task A.3: Page List and Grouping

**Why:** Design Refresh Task 1.3 built the navigation mechanism over provisional groups. This task
sets the groups from "Page list and grouping" and moves the scholarship tracker to Parent.

**Preflight Files:**
- This plan's "Page list and grouping"
- `app/modes.py` (`SECTION_LABELS`, `SECTION_ICONS`, `SECTION_CAPTIONS`, `STUDENT_SECTIONS`,
  `pages_for_mode`), `app/main.py` (`SECTION_RENDERERS`, hidden-page redirect)
- `design-system/scholarshipcoach/MASTER.md` ("Navigation and page chrome"), both `pages/*.md`
- `scripts/screenshot_app.py` (`MODE_SECTIONS` comes from `nav_sections`)
- `docs/decisions.md` (2026-09-30 nav entry)
- `tests/test_app_modes.py`

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/screenshot_app.py --theme both --scratch-db --seed
```

**Checklist:**
- [ ] `pages_for_mode` returns the Student list (ungrouped top pages + "More") and the Parent
      groups "Her path" / "Money" / "Family tools", with only the pages built so far; later tasks add
      theirs by editing this function and `SECTION_LABELS` only
- [ ] Find Scholarships and Applications leave the Student list; labels "Find Scholarships" and
      "Applications" in Parent; This Week stays until A.4 replaces it
- [ ] A student deep link to `/find` or `/applications` lands on the student home, not an error
- [ ] R3 checked: if the sidebar still stays open after a pick at 400px, record it for Design
      Refresh Task 4.1 in `docs/decisions.md`; independently, the student home links each top page
      with `st.page_link` so every student page is one tap from home
- [ ] Tests: groups per mode, no duplicates, parent-only pages absent from Student, every page has
      an icon, URL path and caption
- [ ] 400px sweep looked at for both modes
- [ ] All five CI commands green

---

## Task A.4: Shared To-dos (replaces This Week)

**Why:** R4, R5, R6. One list both roles write to, with an author and a date, is the piece models A
and C share, and the place a parent suggestion reaches her. *(Absorbs Design Refresh Task 2.2.)*

**Preflight Files:**
- `app/sections/this_week.py`, `src/store/tracker.py` (`this_week`, `DueItem`)
- `src/store/milestones.py` (`milestones_for_grade`)
- `design-system/scholarshipcoach/MASTER.md` ("Empty state", "Alerts"),
  `pages/student-mode.md` ("To-dos")
- Design Refresh Plan Task 2.2 checklist (moved here)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/screenshot_app.py --theme both --scratch-db
python scripts/screenshot_app.py --theme both --scratch-db --seed
```

**Checklist:**
- [ ] `app/components.py` (or `app/helpers.py`): `render_empty_state(message, *, next_up=None,
      link_page=None, link_label=None)` per MASTER
- [ ] `app/sections/todos.py`: groups **Overdue**, **This week**, **Coming up** (next 60 days),
      **Someday** (no date); each row shows title, who it's for, the author ("from Mom"), a date
      badge, and one checkbox. Today's `tracker.this_week` items (application deadlines, checklist
      steps, letters) appear as linked rows with their source; ticking a checklist step ticks it in
      the application too
- [ ] Add form: title, optional date, "for" (me / parents / family) in Student mode; (her /
      parents / family) in Parent mode; the author comes from A.2's identity, or the mode
- [ ] Grade-scoped milestones for her grade (PSAT, course registration added in Settings) appear in
      Coming up once each school year, as rows she can tick
- [ ] Empty state: "Nothing to do this week" + the nearest dated item + a link to Opportunities
      (or Find in Parent mode); no `st.success` for empty
- [ ] The "this_week" key is removed; `todos` takes its place as the Student home until A.6
- [ ] Measured at 400px: the first row or the empty state's "Next up" is visible without scrolling;
      S2 = 1 tap, S4 ≤ 2 taps
- [ ] Tests: bucketing by date, author from identity and from mode, assignee filtering, checklist
      step two-way tick
- [ ] All five CI commands green

---

## Task A.5: Parent Money View — Find, Applications, Grade-Correct Results

**Why:** R2, R10, and the parents' weekly job. Find becomes a parent page that ranks on arrival,
and it stops calling senior-only awards "Apply now" for a 10th grader. *(Absorbs Design Refresh
Task 2.1 and the Applications item of Task 2.3.)*

**Preflight Files:**
- `app/sections/find.py`, `app/sections/applications.py`, `app/state.py`
  (`load_snapshot_cached`), `app/helpers.py` (`phone_width_css`, `explain_ranked_row`)
- `src/rank/stage1_eligibility.py` (grade check, `unverified`), `src/rank/timeline.py`
  (`_row_bucket`)
- `design-system/scholarshipcoach/MASTER.md` ("Award card"), `pages/parent-mode.md` ("Find
  Scholarships", "Applications")
- Design Refresh Plan Tasks 2.1 and 2.3 checklists (moved here)
- `tests/test_phone_layout.py`, `tests/test_application_tracker.py`, the Stage 1 tests

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/evaluate_golden_students.py
python scripts/screenshot_app.py --theme both --scratch-db --seed
```

**Checklist:**
- [ ] R2: in Stage 1, a record with no `grade_levels` for a profile below grade 12 adds
      `grade_level` to `unverified` (the existing "confirm you meet" path) rather than passing
      silently; `timeline.py` never buckets such a record as "Apply now" for that profile. Golden
      students re-run; any metric change explained in the task notes
- [ ] Find ranks on arrival with the saved profile (`st.cache_data` keyed on profile + snapshot),
      spinner "Finding matches…"; filters in a collapsed `st.expander("Filters",
      icon=":material/tune:")`
- [ ] Operator-only: snapshot box, "Pipeline Execution", weights, Top-N slider, the
      `eligible=… ineligible=…` box, "Signal details", win-model summary; Parent gets 25 + "Show more"
- [ ] Card per MASTER/parent-mode "Award card" (money headline); keyed
      `st.container(border=True, key=f"award_{catalog_id}")`; Save / Apply in one row; a third action
      "Suggest to her" creates a To-do for her (`source_kind=application`)
- [ ] `phone_width_css()` scopes the full-width-button rule so `.st-key-award_*` buttons stay in a
      row; `test_phone_layout.py` extended
- [ ] Applications: expander label is the title; first line a status badge, a deadline badge and
      "N steps left" (`steps_left_text`, tested) with `st.progress`
- [ ] Measured at 1366px and 400px: typical card ≤ 260px tall at 400px; P1 within budget
- [ ] All five CI commands green

At this point the app is model A in full: parents run money and scholarships, she has a shared
to-do list and her essays.

---

## Task A.6: My Path — Her First Target School's Checklist

**Why:** R5, R7, and the reason she would open the app. One screen: where she stands against her
first target school, what to do this semester, what's coming up.

**Preflight Files:**
- Task A.0 notes (in `data/private/`)
- `app/sections/colleges_money.py`, `src/store/money.py`, `src/store/repo.py` (colleges)
- `src/profile/grade_levels.py` (grade sequence, graduation year)
- `design-system/scholarshipcoach/pages/student-mode.md` ("My Path")

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/screenshot_app.py --theme both --scratch-db --seed
```

**Checklist:**
- [ ] Colleges & Money (Parent): each college gets a "Requirements" section to add, edit, reorder
      and verify requirement rows (category, label, target, due by, source URL, verified on); a
      college can be marked first target (`priority = 1`)
- [ ] `app/sections/my_path.py` (Student home): header "Your path to <college name from the
      database>"; a progress line ("5 of 12 on track"); requirements grouped **This semester**,
      **Next year**, **Senior year**, **Done**; she can set status on course/test rows; GPA rows
      compare the profile GPA with the target and show the gap; dated rows feed To-dos
      (`source_kind=requirement`); "Coming up": the next 3 dated items with page links
- [ ] Parent sees My Path read-only under "Her path"
- [ ] Empty state when no first target exists: Student "Your family hasn't added a school yet";
      Parent links to Colleges & Money
- [ ] Generic only in code and tests: requirement categories and grade logic. Test fixtures use a
      made-up college. No real school's name, thresholds or dates in any committed file
- [ ] *Owner, on the server in the browser (one copy of record):* first target school entered with
      its requirements, re-verified against the school's pages (the 2026-09-27 research is for the
      2026–27 cycle). Unclear rules (for example, how a program test minimum applies to
      test-optional applicants) entered as `notes` with status `not_started`, not guessed
- [ ] Measured at 400px: S1 = 0 taps; the progress line and the first "This semester" item are
      visible without scrolling
- [ ] Tests: grouping by grade and date, GPA gap text, read-only in Parent, requirement → to-do
- [ ] All five CI commands green

---

## Task A.7: Opportunities — One Kind Beyond Scholarships

**Why:** R1, R5. Summer programs are what she can actually apply to at 15. They live in the same
catalog so the inbox, the form and the monthly verify job cover them, but they never enter
scholarship ranking.

**Preflight Files:**
- `data/catalog/schema.json`, `src/normalize/catalog_schema.py`, `src/catalog/entry.py`,
  `app/sections/catalog_inbox.py` (form fields), `scripts/validate_catalog.py`
- `src/ingest/` curated source (how catalog fields reach the snapshot)
- `src/rank/stage1_eligibility.py`, `src/rank/timeline.py`, `app/sections/find.py`
- `src/catalog/verify.py` (monthly re-verification)
- `design-system/scholarshipcoach/pages/student-mode.md` ("Opportunities", "Opportunity card")

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/screenshot_app.py --theme both --scratch-db --seed
```

**Checklist:**
- [ ] Catalog schema: `kind` enum `scholarship | program`, default `scholarship` when absent (every
      existing record stays valid unchanged); carried into the snapshot as a column
- [ ] Catalog & Inbox form: "Kind" on the Basics tab; for `program` the money tab is optional and
      the card wording changes ("Cost" / "Free" instead of an award amount)
- [ ] Find (Parent) and Stage 2/3 exclude `kind=program`; win-model features never see programs
- [ ] `app/sections/opportunities.py`: programs and scholarships that pass Stage 1 for her grade
      **with an explicit grade match**, sorted by application date; groups **Open now**, **Opens
      later** (with the grade or month it opens), **Passed this year**; each card has "Add to my
      to-dos" (`source_kind=opportunity`) and Apply ↗; a violet "From your family" badge when a parent
      suggested it
- [ ] Seed 6–10 NC summer programs for rising 10th–12th graders from several institutions (UNC
      system and private schools), entered through the form, `trust` per the existing rules, URLs
      verified. Commit messages list catalog IDs only and say nothing about her
- [ ] Measured at 400px: S3 ≤ 2 taps from My Path; P2 adds a record that shows on her page after
      the existing rebuild
- [ ] Tests: schema default, program excluded from ranking, explicit-grade filter, to-do creation
- [ ] All five CI commands green

---

## Task A.8: What If as the GPA → Goals Bridge

**Why:** R7. What If already re-runs eligibility with a changed GPA or score. It should also say
which of her own targets that reaches. *(Absorbs Design Refresh Task 2.3's What If item.)*

**Preflight Files:**
- `app/sections/what_if.py`, `src/rank/whatif.py`
- Task A.6's requirement rows (`category` `gpa` / `test`)
- `design-system/scholarshipcoach/pages/student-mode.md` ("What If")

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/screenshot_app.py --theme both --scratch-db --seed
```

**Checklist:**
- [ ] Result first: "Your goals" lists each GPA/test requirement of the first target school with
      reached / not yet at the slider value ("3.50: reaches the program minimum; 0.25 short of the
      full-cost scholarship")
- [ ] Below it, awards that open up, in friendly terms; "$0" for an unknown amount becomes "Amount
      not listed"; awards for a later grade say when they open
- [ ] My Path's GPA row links here with the target pre-set (S5 = 1 tap)
- [ ] Results render above the controls or directly under the changed control at 400px
- [ ] Tests: goal reach text at, below and above each threshold; unknown amount wording
- [ ] All five CI commands green

---

## Task A.9: Essays She Shares, Recommenders

**Why:** D3, R8. *(Absorbs Design Refresh Task 2.3's Essays and Recommenders items.)*

**Preflight Files:**
- `app/sections/essays.py`, `app/sections/recommenders.py`, `src/store/essays.py`
- `app/modes.py` (`essays_read_only`, `can_edit_essays`)
- `design-system/scholarshipcoach/pages/student-mode.md` ("Essays", "Recommenders"),
  `pages/parent-mode.md` ("Parent view of Essays")

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/screenshot_app.py --theme both --scratch-db --seed
```

**Checklist:**
- [ ] Student: a "Share with parents" toggle per essay with a caption saying what it does; shared
      essays show a "Shared" badge
- [ ] Parent: only shared essays listed, as text with a gray "Read-only" badge; an unshared count
      ("2 drafts not shared") without titles; a "Suggest a change" action creates a To-do for her
      that links the essay
- [ ] "Challenge: 1" counts replaced with readable theme chips or removed; "Prompts waiting for an
      essay" shows only when a saved application has a prompt, and otherwise says nothing (R8)
- [ ] Editor ≥ 12 rows with a word-count caption
- [ ] Recommenders: one row per person with "2 in / 1 waiting"; a request with a due date creates
      a To-do for her ("Ask Ms. … by …")
- [ ] Tests: parent cannot list or open an unshared essay by ID; sharing toggles; the
      `can_edit_essays` tests still pass
- [ ] All five CI commands green

---

## Task A.10: Phase A Sweep, Journey Harness, Ready to Onboard

**Why:** The gate needs a version that is not about to change, and journeys that can be re-walked
the same way after it.

**Preflight Files:**
- `scripts/screenshot_app.py`, `app/screenshot_entry.py`
- This plan's "Journeys after Phase A"
- `design-system/scholarshipcoach/MASTER.md` ("Pre-delivery checklist")
- `docs/plans/SCHOLARSHIPCOACH_CLOUD_HOSTING_PLAN.md` Task 3.1

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
- [ ] `scripts/screenshot_app.py --journeys`: walks S1–S6 and P1–P4 step by step at 400px on the
      scratch DB, using the demo profile set to a 10th grader, writes one PNG per step and prints
      taps per journey. It never submits the catalog form outside a scratch copy of
      `data/catalog/records/`
- [ ] Every journey within its budget, or the overrun recorded with the reason
- [ ] MASTER pre-delivery checklist run on every page this plan added or moved
- [ ] No committed file names her first target school (`git grep -i` for the school's name and
      abbreviations, run locally from a private list, returns nothing)
- [ ] Hosting Task 3.1 told it can start: deploy, then onboarding
- [ ] All five CI commands green

---

# Go/No-Go Gate

**When:** about four weeks after Cloud Hosting Task 3.1 onboards the family (D7). Ideally the window
covers her school's course-registration season, when My Path has the most to say.

**Evidence:**
- `activity_days` for her login (D6): days she opened the app, and how many of those followed a
  parent's to-do for her on the same day (asked) versus not (on her own)
- To-dos she finished, and to-dos she created
- Requirement rows she updated on My Path
- A 10-minute conversation with her, run by the owner, with the same questions as Task A.0

**Bar (default; owner may change before onboarding):**

| Outcome | Signal | Next |
|---|---|---|
| **Go** | She opened it on her own on 6+ days of the 28 and changed something on My Path or To-dos herself | Phase B |
| **Partial** | She opened it mostly when asked, but finished her to-dos there | Fall back to model A: remove My Path, Opportunities and What If from Student mode; keep To-dos and Essays; Phase B limited to B.4 |
| **No-go** | Under 3 days and the to-dos were done elsewhere | Parent-only tool: Student mode reduced to Essays; record the reason in `docs/decisions.md` |

The result and the evidence are recorded in this plan and `docs/decisions.md` before any Phase B
task starts.

---

# Phase B — Only Past the Gate

Scoped, not specified. Each task gets full Preflight / Checklist detail when the gate passes, from
what the month showed.

## Task B.1: More Opportunity Kinds and Their Upkeep

**Why:** Competitions, pre-college courses and internships are the rest of what she can do before
senior year.

**Preflight Files:** `data/catalog/schema.json`, `src/catalog/verify.py`, `scripts/catalog_inbox.py`,
Task A.7's notes

**Validation Commands:** the five CI commands

**Checklist:**
- [ ] `kind` gains `competition | course | internship` as the gate evidence supports
- [ ] Monthly verify reports program and competition dates per kind; a "re-check each spring" rule
      for recurring programs
- [ ] Feed sources for programs considered only if curation load measured in Phase A says so

## Task B.2: Ranking for Opportunities

**Why:** Stage 3's urgency + expected value assumes money. Opportunities need fit + timing.

**Preflight Files:** `src/rank/stage2_scoring.py`, `src/rank/stage3_rerank.py`, `src/rank/weights.py`,
`src/eval/golden_students.py`

**Validation Commands:** the five CI commands + `python scripts/evaluate_golden_students.py`

**Checklist:**
- [ ] A separate opportunity scorer (Stage 2 similarity + deadline proximity, no expected value, no
      win model), or a Stage 3 weight profile per kind; chosen by measurement on a small labelled set
- [ ] Scholarship rankings unchanged (golden students identical)

## Task B.3: The College Side

**Why:** One target school answers "am I on track"; a list answers "where should I apply".

**Preflight Files:** `app/sections/colleges_money.py`, `app/sections/my_path.py`,
`src/store/money.py`

**Validation Commands:** the five CI commands + the screenshot sweep

**Checklist:**
- [ ] Several target schools, requirements compared side by side (Parent), My Path switchable
      between them (Student)
- [ ] Each college's application and merit deadlines feed To-dos and Timeline
- [ ] Net price per school against won awards (existing Colleges & Money math)

## Task B.4: Bringing People Back

**Why:** R4. Without a nudge the app depends on being remembered.

**Preflight Files:** `src/store/calendar_feed.py`, `scripts/export_calendar.py`,
`deploy/` (serve config)

**Validation Commands:** the five CI commands

**Checklist:**
- [ ] Per-person calendar subscription (`.ics` URL served over the tailnet, behind identity) with
      her to-dos and requirement dates; parents' with money dates
- [ ] No email or public push service unless the owner decides otherwise ("never public")

## Task B.5: Senior-Year Switch and Her School's Tool

**Why:** D9, and the open question of what her high school already provides.

**Preflight Files:** `app/modes.py`, Task A.0 notes

**Validation Commands:** the five CI commands

**Checklist:**
- [ ] From spring of grade 11, Find (her view, grade-correct), Applications and Recommenders
      return to Student mode
- [ ] Once her high school and its planning tool are known: link out to it from My Path for what
      it covers, and drop any duplicate from this app

---

## Execution Order

```
Now, in parallel with Hosting Phase 2 wrap-up and Design Refresh 3.1:
  A.0 owner conversation (no code)
  A.1 migrations + data model
  A.2 identity (needs the server for its first item)

Then, in order:
  A.3 page list + grouping
  A.4 shared to-dos           ← absorbs Refresh 2.2
  A.5 parent money view       ← absorbs Refresh 2.1 + 2.3 (Applications)
      — model A complete here —
  A.6 My Path                 (after A.0's notes)
  A.7 Opportunities
  A.8 What If bridge          ← absorbs Refresh 2.3 (What If)
  A.9 essays sharing          ← absorbs Refresh 2.3 (Essays, Recommenders); may run any time after A.2
  A.10 sweep + journey harness

Design Refresh 3.2 (Timeline, Colleges & Money, Outcomes): after A.6
Hosting 3.1 onboarding: after A.10 is deployed
  → about four weeks of real use → Go/No-Go Gate
Phase B (only on Go; B.4 also on Partial)
Last, across everything: Design Refresh 4.1 sweep → 4.2 screenshots + docs
```

A.1 comes first because every page writes to its tables, and the server migrates on the first
request after an update. A.2 precedes A.4 so to-dos carry an author from their first row. A.3
before the page tasks so each one only adds its own entry. A.4 and A.5 before A.6 so a failed gate
falls back to a finished model A. A.10 last, so the family onboards to a version that is not about
to change.

Every task that adds a file under `src/store/migrations/` is flagged when pushed: the server update
migrates the family database irreversibly, after a backup (CLAUDE.md Step 6).

## Success Criteria

1. A 10th-grade profile never sees "Apply now" on an award her grade cannot apply to, in any mode.
2. Student home answers "where do I stand for my first target school, and what's next" with
   0 taps, and every student page is 1 tap from home at 400px.
3. A parent suggestion (award, program, essay comment, letter) reaches her To-dos with the
   parent's name on it, and its completion is visible to the parent.
4. Parents cannot list or open an essay she has not shared.
5. With identity on, her login cannot reach a Parent page; without identity, the PIN behaves as
   before.
6. Every migration applied to a database at the previous schema keeps every existing row.
7. Programs never enter Stage 2/3; scholarship golden-student results are unchanged except where
   the R2 grade fix explains a change.
8. No committed file names her first target school or identifies her.
9. The go/no-go gate is held with recorded evidence before any Phase B task starts.
10. All five CI commands green after every task.
