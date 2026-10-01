# ScholarshipCoach Design Refresh Plan — "Study Hall": Calm, Phone-First, Light and Dark

> Generated: 2026-09-26 | Scope from a `ui-ux-pro-max` audit of every section in Student and
> Parent mode at 400px and 1366px, and a choice between three design directions
> Design system: `design-system/scholarshipcoach/MASTER.md` + `pages/student-mode.md`,
> `pages/parent-mode.md` (read these before any task; page files override Master)
> Executor: Claude via Claude Code | Est. effort: 5 phases, 11 tasks
> **Updated 2026-09-27:** the owner is moving toward a "student path + parent money view" role
> model (see "Relationship to other plans"). Phase 2 is **on hold** and expected to move to the
> Roles & Product Plan; Task 1.3 is narrowed to the navigation *mechanism*; Execution Order
> rewritten across all three active plans.

---

## Why this plan exists

The UI Redesign Plan (2026-07) turned a developer console into a card UI, and the Family Product
Plan added modes, a tracker, essays and a timeline. The app now *works* for the family. It does not
yet *feel* like something a high-school student opens on her phone every day. On 2026-09-26 every
section was screenshotted in Student and Parent mode at 400×860 and 1366×900. That was 36 captures
against the real app, plus a second set against a scratch copy of `coach.db` with 3 saved awards,
1 essay and 1 recommender, so populated states could be seen without writing to `data/private/`.
The captures were checked against the `ui-ux-pro-max` guidelines.

### Audit findings

| # | Sev. | Finding | Evidence | Guideline |
|---|---|---|---|---|
| F1 | Critical | Primary button label fails contrast | White on rendered `#2D9F98` = **3.2:1** (config `#4ECDC4`) | `color-contrast` 4.5:1 |
| F2 | High | Phone navigation hidden and mixed with a form | Sections are a sidebar radio behind `»`, above a ~30-field profile form. Changing section takes 3 taps (open, pick, close). No URL per section | `persistent-nav`, `deep-linking`, `nav-hierarchy` |
| F3 | High | Global title block wastes the phone's first screen | `st.title("Scholarship Coach")` + "Find scholarships matched to your profile" on every page is ~200px of the 860px viewport, and the caption is wrong on 10 of 11 pages | `content-priority`, `visual-hierarchy` |
| F4 | High | Pipeline vocabulary on a student surface | Find shows "Active snapshot: …parquet", "Pipeline Execution", "Active ranking weights: Latest", "Top-N results", "Pipeline complete: eligible=106 ineligible=27", "Signal details" | tone; `progressive-disclosure` |
| F5 | High | Find shows nothing until "Run" is pressed | Every visit starts with a form and an empty page | `empty-states`, `primary-action` |
| F6 | High | Award cards are ~600px tall on a phone | 21 results → a **12,000px** page. Each reason is its own paragraph; Save and Apply are stacked full-width | `touch-density`, `whitespace-balance` |
| F7 | High | Emoji as status icons | `urgency_indicator()` returns 🔴🟡🟢⏰⚠️ with inline hex; "📊 Win Model Summary"; "✓ Saved" | `no-emoji-icons`, `color-semantic` |
| F8 | Medium | Alarm language contradicts the calm tone | Timeline marks *"FAFSA opens"* and *"CSS Profile opens"* **"URGENT (≤7 days)"** in red; "Passed" gets ⏰ | tone; `color-not-decorative-only` |
| F9 | Medium | Contradictory reasons | A card with a 40-day deadline and a "Later" badge says "Deadline soon, boosted for urgency" (`urgency_boost = e^(-40/30) ≈ 0.26 > 0`) | `error-clarity` |
| F10 | Medium | Dark-only theme, off-palette surfaces | Purple-navy `#1A1A2E` surface under a teal primary; default Streamlit-blue info boxes; no light theme; default Source Sans | `dark-mode-pairing`, `color-semantic` |
| F11 | Medium | Empty states lead nowhere | This Week: green success box "Nothing due in the next two weeks." with no next item or link. Colleges & Money: "$0 / 0 / 0" | `empty-states` |
| F12 | Medium | Weak card hierarchy, raw formats | Amount has the same weight as deadline; ISO dates ("Due on or before 2026-10-10"); cryptic counts ("Challenge: 1", "Saved — 0/3 done") | `visual-hierarchy`, `number-formatting` |
| F13 | Medium | Parent forms are one long column | Catalog & Inbox "Add an award" is ~3,000px on desktop and ~5,000px on a phone | `progressive-disclosure`, `field-grouping` |
| F14 | Low | Expert switch leads Settings | "Show operator tools" is the first control on the page | `destructive-nav-separation` |

What already works and must be kept: the 44px tap-target and 16px input rules in
`phone_width_css()`, the PIN gate, read-only essays for parents, the Timeline's month-grouped cards,
and the plain-English reasons from `explain_ranked_row()`.

### Directions considered (2026-09-26)

| | A — Study Hall **(chosen)** | B — Sunrise | C — Night Desk |
|---|---|---|---|
| Palette | Evolved teal, light + dark following system | Cream + indigo-violet, amber deadlines | Dark-first navy + mint, gold wins |
| Type | Nunito / Nunito Sans | Outfit / Work Sans | Inter only |
| Why / why not | Continuity with today's teal, calm and friendly, lowest risk; least distinctive | Most encouraging personality; biggest change, cream is less neutral for dense Parent forms | Closest to today, good at night; weakest fit for "encouraging", dark-first in daylight |

**Decisions taken:**
- **Direction A, Study Hall**, with every token contrast-measured in
  `design-system/scholarshipcoach/MASTER.md`. Dark primary is the mid-tone `#11807A` (white label
  4.8:1); a bright `#5FE0CD` is used only as the dark-mode link color.
- **Light and dark, following the device.** Both are defined in `[theme.light]` / `[theme.dark]`.
- **Top navigation via `st.navigation(position="top")`**, one URL per section, profile on its own
  page, sidebar reduced to the mode switch.
- **Fonts self-hosted** via `[[theme.fontFaces]]` + static serving; no Google Fonts requests.
- **Per-mode layout:** Student `layout="centered"`, comfortable spacing; Parent `layout="wide"`,
  dense spacing and dataframes. The theme is global in Streamlit, so the modes differ in layout,
  density, and copy, not color.

**Relationship to other plans:**
- **UI Redesign Plan:** complete except its screenshot follow-up. This plan supersedes its Task 1
  theme values (`#4ECDC4` / `#0E1117` / `#1A1A2E`) and re-takes its README screenshots in Task 4.2.
- **Cloud Hosting Plan:** runs in parallel. This plan touches `.streamlit/config.toml` and
  `app/`; hosting touches `deploy/` and `docs/operations.md`. Task 1.1 already added
  `enableStaticServing = true` to `[server]` (fonts in `app/static/`), and the hosting plan's
  Task 1.1 now checks that the server picks it up. Task 4.1's real-phone check uses the hosted
  address once hosting Phase 2 is live.
- **Roles & Product Plan** (`docs/plans/SCHOLARSHIPCOACH_ROLES_PRODUCT_PLAN.md`, to be written by
  the roles/UX review): after 2026-09-27 discussions the leading role model is **Student mode as
  her path to college** (starting from her first target school and intended CS major) with
  **Parent mode as the money and strategy view**, tested first as a light version. Ownership
  between the plans:
  - *Roles & Product Plan* owns **what** exists: roles, the page list and grouping, what each
    page is for and shows, data and catalog changes.
  - *This plan* owns **how it looks**: theme, badges, components, page chrome, phone-width rules,
    the screenshot harness, and the final sweep.
  - So Phase 2 (student page content) is on hold and expected to move there, and the
    design-system student-card rules ("money is the headline") are revisited by that plan. Under
    the leading model, money-first cards become a `parent-mode.md` rule.
- **Operator mode** keeps every pipeline control removed from Student view (F4). Nothing is
  deleted, only moved.

---

## Owner decisions (defaults used unless changed before Task 1.1)

| # | Decision | Default | Alternatives |
|---|---|---|---|
| D1 | Direction | **A — Study Hall** (decided 2026-09-26) | B, C above |
| D2 | Light/dark | Follow the device's system setting | Force light; force dark |
| D3 | Navigation | `st.navigation(position="top")` | Sidebar `st.navigation` with sections (fallback if top overflows at 400px) |
| D4 | Find on arrival | *Moved to the Roles & Product Plan with Task 2.1* (was: rank automatically with the saved profile, cached) | Keep the Run button |
| D5 | Fonts | Nunito + Nunito Sans, self-hosted `.woff2` in `app/static/fonts/` | System font stack (zero files, less personality) |

---

## Design principles for this plan

1. **The design system is the spec.** Each task reads `MASTER.md` and the relevant `pages/*.md`
   file first. A deviation is a design-system edit made in the same commit, not a one-off.
2. **Styling order is binding** (CLAUDE.md "UI/UX rules"): `config.toml` theme → native layout →
   CSS in `phone_width_css()` targeting `data-testid` / `.st-key-*`. No `st-emotion-cache-*`.
3. **Presentation only.** Stage 1/2/3 output is unchanged for the same inputs. The one ranking-adjacent
   edit is wording in `explain_ranked_row()` (F9), which does not change scores.
4. **Decisions stay pure and tested.** New helpers (deadline badge, friendly dates, page lists)
   are plain functions with unit tests, as `app/modes.py` already is. Only `render_*` code touches
   Streamlit.
5. **Seen at 400px, in both themes, before it is done.** Every UI task ends with a screenshot
   sweep from the Task 0.1 harness, light and dark, and the screenshots are looked at.
6. **Green gate every task:** the five CI commands in CLAUDE.md.

---

# Phase 0 — Screenshot Harness

## Task 0.1: Repeatable Phone/Desktop Screenshots

**Why:** The audit's 36 captures came from a throwaway script. Every later task has to be checked
at 400px in light and dark. Without a harness that check gets skipped, and phone-width bugs
reach the family.

**Preflight Files:**
- `docs/operations.md` ("Phone width")
- `app/modes.py` (`SECTION_LABELS`, `sections_for_mode`)
- `src/store/db.py`, `src/profile/store.py` (`DEFAULT_DB_PATH`, `PRIVATE_DIR`, `STUDENTS_DIR`)
- `pyproject.toml` (`[project.optional-dependencies]`)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/screenshot_app.py --out $env:TEMP\coach_shots --scratch-db
```

**Checklist:**
- [x] `pyproject.toml`: new optional group `ui = ["playwright==1.63.0"]`, pinned like `dev`, not
      installed by CI
- [x] `scripts/screenshot_app.py`: starts the app on a free port, drives the system Edge through
      Playwright (`channel="msedge"`, so no browser download), walks every section of Student and
      Parent mode at 400×860 and 1366×900, measures the scroll height of `[data-testid="stMain"]`
      and resizes before capturing, and writes `<width>_<mode>_<section>_<theme>.png`
- [x] `--theme light|dark|both` via Playwright `color_scheme`
- [x] `--scratch-db`: copies `data/private/` to a temp dir and points `DEFAULT_DB_PATH` /
      `STUDENTS_DIR` there through a wrapper entry script, so a sweep never writes the family's
      database. Default **on**; `--live-db` must be passed explicitly
- [x] `--seed`: through the UI, saves the top 3 Find results, adds one essay and one recommender
      (in the scratch DB), so populated states are captured
- [x] Output goes outside the repo by default; screenshots are not committed except Task 4.2's
      README set
- [x] `docs/operations.md` "Phone width" gains a "Checking it" paragraph naming the script
- [x] All five CI commands green

---

# Phase 1 — Foundations

## Task 1.1: Theme Tokens, Light + Dark, Self-Hosted Fonts

**Why:** F1 and F10. One file re-skins every page. Doing it first means every later task is built
and screenshotted against the final palette.

**Preflight Files:**
- `design-system/scholarshipcoach/MASTER.md` ("Color tokens", "Typography", "Shape, spacing,
  elevation")
- `.streamlit/config.toml`
- `app/helpers.py` (`phone_width_css`)
- `tests/test_phone_layout.py`

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
- [x] `.streamlit/config.toml`: `[theme]` holds shared keys (`font`, `headingFont`,
      `baseFontSize = 16`, `headingFontSizes`, `headingFontWeights`, `baseRadius`,
      `buttonRadius`, `showWidgetBorder = true`); `[theme.light]` and `[theme.dark]` hold every
      color in MASTER "Core" and "Status" (`<c>Color`, `<c>BackgroundColor`, `<c>TextColor` for
      red/orange/yellow/green/blue/violet/gray), plus `linkColor` and `borderColor`;
      `[theme.light.sidebar]` / `[theme.dark.sidebar]` set the sidebar background. Every option name
      is checked against Streamlit 1.54 (`streamlit config show`); an unknown key fails silently
- [x] Nunito (700, 800) and Nunito Sans (400, 600, 700) `.woff2` under `app/static/fonts/` with
      `OFL.txt`; `[[theme.fontFaces]]` entries; `[server] enableStaticServing = true`
- [x] `tests/test_theme_config.py`: parses `config.toml` and asserts WCAG ratios per theme:
      text/background ≥ 4.5, gray text/background ≥ 4.5, white/primary ≥ 4.5, primary/background
      ≥ 3.0, border/background ≥ 3.0, each status text/background pair ≥ 4.5. It also asserts that
      no key points at `fonts.googleapis.com`. The ratio function lives in the test, not `src/`
- [x] Sampled from a screenshot, not trusted from config: the rendered primary button fill and
      its label give ≥ 4.5:1 in both themes (Streamlit may shade the configured primary)
- [x] Browser network log during a sweep shows no request to `fonts.googleapis.com` /
      `fonts.gstatic.com`
- [x] Sweep reviewed in light and dark at 400px and 1366px; no unreadable element in either
- [x] All five CI commands green

---

## Task 1.2: One Status Vocabulary — Deadline Badges, Friendly Dates, No Emoji

**Why:** F7, F8, F9, F12. `urgency_indicator()` feeds This Week, Find, Applications and Timeline
with emoji and inline hex, and has no idea a milestone *opening* is not a deadline.

**Preflight Files:**
- `design-system/scholarshipcoach/MASTER.md` ("Deadline states", "Iconography", "Voice and
  formatting")
- `app/helpers.py` (`urgency_indicator`, `explain_ranked_row`, `format_amount_range`,
  `timeline_deadline`)
- `app/sections/this_week.py`, `find.py`, `applications.py`, `timeline.py`
- `src/rank/stage3_rerank.py` (`_compute_urgency_boost`, read-only)
- `tests/test_explainability_helpers.py`

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
- [x] `app/helpers.py`: pure `deadline_badge(days_until, *, kind="deadline"|"milestone",
      projected=False, deadline=None) -> DeadlineBadge(label, icon, color)` implementing MASTER
      "Deadline states"; milestones never return red/orange or the word "urgent"
- [x] `friendly_date(d, today)` → `Fri, Oct 10` within the school year, `Oct 10, 2027` beyond it;
      "Unknown" amounts render as "Amount not listed"
- [x] `urgency_indicator` removed; the four sections render `st.badge(label, icon=, color=)`
- [x] `explain_ranked_row`: the "Deadline soon" reason appears only when the deadline is ≤ 14
      days away; the wording changes, the scores do not
- [x] Emoji removed from app chrome: "📊 Win Model Summary", "✓ Saved" (→ green "Saved" badge),
      "⚠️" strings. `grep` for common emoji ranges in `app/` returns only user-data paths
- [x] Tests: every `deadline_badge` state; milestone never red; `friendly_date` year boundary;
      the F9 case (40 days → no "soon")
- [x] All five CI commands green

---

## Task 1.3: Page Chrome and Top Navigation

**Why:** F2 and F3. Navigation is the frame every other screen sits in, and fixing it after the
sections are restyled would mean restyling twice.

**Scope (2026-09-27):** the *mechanism* only, built over **today's sections**: top navigation,
one URL per page, per-mode layout, no global title, the profile page. Which pages exist, how they
are grouped, and what they are called belong to the Roles & Product Plan. Build it so that a later
page-list change is an edit to `pages_for_mode` and `SECTION_LABELS` and nothing else. This task
can run in parallel with the roles/UX review.

**Preflight Files:**
- `design-system/scholarshipcoach/MASTER.md` ("Navigation and page chrome"), both `pages/*.md`
  ("Layout overrides")
- `app/main.py`, `app/modes.py`, `app/sidebar.py`, `app/state.py`
- `app/screenshot_entry.py`, `scripts/screenshot_app.py` (`section_url`; the harness navigates
  through the state this task replaces)
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
- [x] `app/main.py` builds `st.Page` objects (title, `:material/…:` icon, stable `url_path`) from
      the existing `SECTION_RENDERERS` and calls `st.navigation(..., position="top")` with only the
      pages the resolved mode may see. A deep link to a parent-only URL in Student mode lands on
      This Week, not an error
- [x] Nav lists today's sections for each mode, in the order `sections_for_mode` gives, plus My
      Profile. `pages_for_mode` may return grouped sections (a `dict` for `st.navigation`), but
      the grouping chosen here is provisional; the final page list and grouping ("More",
      "Student views" / "Family tools" in `pages/*.md`) come from the Roles & Product Plan
- [x] Checked at 400px in Student mode (7 items) and Parent mode (12 items): the top bar does not
      wrap into more than one row or push content below the fold. If it does, group the pages
      (Parent first) or fall back to D3's sidebar alternative, and record which and why in
      `docs/decisions.md`
- [x] `st.set_page_config(layout="centered")` in Student mode, `"wide"` in Parent/Operator
- [x] The global `st.title("Scholarship Coach")` and caption are removed; each section opens with
      its own h1 and a one-line purpose caption
- [x] Profile form moves from the sidebar to a "My Profile" page (`app/sections/profile.py`); the
      sidebar keeps only the mode switch, PIN prompt and mode caption (plus operator tools in
      Operator mode)
- [x] Screenshot harness updated in the same task: `app/screenshot_entry.py` currently opens a
      page by writing `?shot_section=` into `SECTION_STATE_KEY`, which `st.navigation` no longer
      reads. `section_url()` in `scripts/screenshot_app.py` opens each page by its `url_path` (keeping
      `shot_mode` for the mode and PIN unlock); a full `--theme both --scratch-db --seed` sweep
      captures every page of both modes, and each capture is the requested page, not This Week
- [x] Pure page-list functions in `app/modes.py` (`pages_for_mode`) are unit-tested: every section
      has a URL path and icon, no duplicates, parent-only pages absent from Student, PIN still gates
      Parent. Existing `test_app_modes.py` tests updated rather than deleted
- [x] All five CI commands green

---

# Phase 2 — Student Surfaces

> **ON HOLD (2026-09-27) — expected to move to the Roles & Product Plan.** These tasks assume the
> student's job is applying for scholarships. The leading role model makes Student mode her path
> to college and moves money-first views to Parent mode, which changes what Find, This Week and
> the tracker pages are for. Do not implement them from this plan. When the Roles & Product Plan
> is written it should absorb each task (marking it here "MOVED → <plan> Task N", the convention
> the Portfolio → UI Redesign handoff used) or release it back unchanged. The checklists below are
> kept as a starting point. The visual pieces (compact card layout, badge rows, one primary action,
> `render_empty_state`, phone-width CSS scoping) still follow this plan's design system wherever
> they end up.

## Task 2.1: Find — Results First, Compact Award Cards

**Why:** F4, F5, F6. Find is where the student decides what to apply for. Today she meets a
pipeline form, an empty page, and then 12,000px of cards.

**Preflight Files:**
- `design-system/scholarshipcoach/MASTER.md` ("Award card"), `pages/student-mode.md` ("Find
  Scholarships")
- `app/sections/find.py`, `app/state.py` (`load_snapshot_cached`), `app/helpers.py`
  (`phone_width_css`)
- `tests/test_phone_layout.py`

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
- [ ] Student and Parent mode rank on arrival with the saved profile (`st.cache_data` keyed on the
      profile and snapshot); a spinner with plain words ("Finding matches…") covers the first run
- [ ] Filters (search, "When can I apply", minimum amount, no-essay only) live in a collapsed
      `st.expander("Filters", icon=":material/tune:")`; results update when they change
- [ ] Operator-only: the snapshot info box, "Pipeline Execution", ranking weights, the Top-N slider,
      the "Pipeline complete: eligible=… ineligible=…" box, "Signal details", the win-model summary.
      Student and Parent get 25 results and a "Show more" button
- [ ] Card per MASTER: keyed `st.container(border=True, key=f"award_{catalog_id}")`; amount as the
      headline; effort, local and eligibility as badges; reasons on one line; Save / Apply in one
      horizontal row with one primary; details in a "More about this award" expander
- [ ] `phone_width_css()`: the "buttons 100% wide" rule is scoped so buttons inside
      `.st-key-award_*` stay in their row; `test_phone_layout.py` extended for the new selector
- [ ] Measured from the 400px screenshot: a typical card ≤ 260px tall (was ~600px); 25 cards
      ≤ 7,000px total
- [ ] Header reads "N matches for you"; a zero-result state names the filter to relax
- [ ] All five CI commands green

---

## Task 2.2: This Week and Empty States

**Why:** F11. This Week is the page the student opens most. When nothing is due it currently
says so in a success box and stops.

**Preflight Files:**
- `design-system/scholarshipcoach/MASTER.md` ("Empty state", "Alerts"), `pages/student-mode.md`
  ("This Week")
- `app/sections/this_week.py`, `applications.py`, `outcomes.py`, `colleges_money.py`, `essays.py`
- `src/store/` (the due-items query This Week uses)

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
- [ ] `app/helpers.py` (or a small `app/components.py`): `render_empty_state(message, *,
      next_up=None, link_page=None, link_label=None)`: bordered container, one sentence, an
      optional "Next up" line, one `st.page_link`
- [ ] This Week: compact due rows (title, what's due, deadline badge, one action); the empty state
      shows the nearest future deadline across saved awards and links to Find
- [ ] Empty states replace bare `st.info`/`st.success` on Applications, Essays, Recommenders,
      Outcomes; Colleges & Money shows the empty state instead of a "$0 / 0 / 0" metric row
- [ ] No `st.success` remains for anything but confirming an action the user just took
- [ ] Both sweeps (empty and seeded) reviewed at 400px: the first item or the empty state is
      visible without scrolling
- [ ] All five CI commands green

---

## Task 2.3: Applications, Essays, Recommenders, What If

**Why:** F12. Cryptic counts and status packed into expander labels make the tracker hard to scan
on a phone.

**Preflight Files:**
- `design-system/scholarshipcoach/pages/student-mode.md` ("My Applications", "Essays",
  "Recommenders", "What If")
- `app/sections/applications.py`, `essays.py`, `recommenders.py`, `what_if.py`
- `tests/test_application_tracker.py`

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
- [ ] Applications: expander label is the title; first line inside is a status badge, a deadline
      badge and "N steps left" (a pure, tested `steps_left_text`), with a `st.progress` bar
- [ ] Essays: "Challenge: 1" replaced with readable theme counts or removed; "Prompts waiting for an
      essay" → "Needs an essay" cards; editor text area ≥ 12 rows with a word-count caption
- [ ] Recommenders: one row per person with a "2 in / 1 waiting" badge
- [ ] What If: the "N more awards open up" result renders above or directly beside the changed
      control at 400px
- [ ] Parent mode: Essays shows a gray "Read-only" badge and renders drafts as text (per
      `parent-mode.md`); the edit path stays blocked (`can_edit_essays` tests still pass)
- [ ] All five CI commands green

---

# Phase 3 — Parent Surfaces

## Task 3.1: Catalog & Inbox and Settings

**Why:** F13, F14. The add-an-award form is the parents' most-used tool and the longest page in the
app.

**Preflight Files:**
- `design-system/scholarshipcoach/pages/parent-mode.md` ("Catalog & Inbox", "Settings")
- `app/sections/catalog_inbox.py`, `app/sections/settings.py`
- `scripts/validate_catalog.py` (the record the form writes must still validate)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
python scripts/screenshot_app.py --theme both --scratch-db
```

**Checklist:**
- [x] "Add an award" split into `st.tabs` *inside* the existing `st.form("catalog_entry_form")`:
      Basics · Money & dates · Who it's for · Requirements. The submit button stays in the form but
      below the tabs, so it is visible from every tab. "Read the page" stays the lead action
- [x] The dict `_award_form_values()` returns keeps the same keys, and its widgets stay keyless
      (prefill and proposal loading depend on that; see its docstring). The record still goes
      through `catalog_entry.validate_form` unchanged
- [x] Inbox list uses the MASTER status badges
- [x] Settings: Family milestones first; "Show operator tools" moves into a closing "Advanced"
      expander
- [x] Page height at 1366px ≤ 1,600px for the Add form's first tab (was ~3,000px for the whole
      form)
- [x] All five CI commands green

---

## Task 3.2: Timeline, Colleges & Money, Outcomes

**Why:** F8 on the page where it is loudest, plus dense views parents compare across a year.

**Depends on (2026-09-27):** the Roles & Product Plan deciding who the Timeline belongs to
(parent-only, shared, or split into her milestones and the family's money dates) and what
Colleges & Money holds once target schools exist. Run this task after that
decision. Task 3.1 has no such dependency.

**Preflight Files:**
- `design-system/scholarshipcoach/pages/parent-mode.md` ("Timeline", "Colleges & Money",
  "Outcomes")
- `app/sections/timeline.py`, `colleges_money.py`, `outcomes.py`
- `src/store/milestones.py`, `src/store/calendar_feed.py` (event kinds)

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
- [ ] Timeline entries use `deadline_badge(kind="milestone")` for milestones (blue "Opens"),
      deadline states for awards, gray "Passed"; no "URGENT" anywhere in `app/`
- [ ] `st.segmented_control` Cards / List; List is a `st.dataframe` with date, what, kind and status
      columns via `column_config`
- [ ] Colleges & Money: bordered metrics in a horizontal row once data exists; currency
      `column_config` in any table
- [ ] Outcomes: `st.dataframe` with a currency column when populated
- [ ] `.ics` export unchanged (existing calendar tests pass)
- [ ] All five CI commands green

---

# Phase 4 — Verify and Document

## Task 4.1: Full Sweep and Design-System Reconciliation

**Why:** Tasks land one page at a time. Drift between pages is only visible when they are seen
side by side.

**When (2026-09-27):** after the Roles & Product Plan's page work ships, so the sweep and
reconciliation cover the pages the family will actually use. It includes pages that plan built
with this design system.

**Preflight Files:**
- `design-system/scholarshipcoach/MASTER.md` ("Pre-delivery checklist"), both `pages/*.md`
- This plan's audit table (F1–F14)

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
- [ ] Every MASTER pre-delivery item checked against the sweep, light and dark, 400px and 1366px
- [ ] F1–F14 each re-checked and marked fixed, or listed in `docs/decisions.md` with the reason it
      was not
- [ ] Anything built differently from the design system is either fixed or written back into
      `MASTER.md` / `pages/*.md`, so the files describe the app as shipped
- [ ] One real phone check (the student's, over the family's normal access path): nav, Find, This
      Week, essay editor focus without zoom
- [ ] All five CI commands green

---

## Task 4.2: Screenshots, Docs, Bookkeeping

**Why:** The README screenshots and `docs/operations.md` describe the old dark console. Left as
they are, the next reader, or the next session, works from the wrong picture.

**Preflight Files:**
- `README.md` ("Screenshots")
- `docs/images/` (`ranked_cards.png`, `this_week_phone.png`, `inbox_proposal.png`)
- `docs/operations.md` ("Phone width")
- `docs/decisions.md`
- `CLAUDE.md` (plan table, "UI/UX rules")

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
```

**Checklist:**
- [ ] The three README screenshots retaken from a seeded scratch-DB sweep (demo profile only, no
      family data), same filenames, light theme; one dark-theme phone shot added
- [ ] `docs/operations.md` "Phone width": top nav instead of the sidebar hamburger, the screenshot
      harness, light/dark follows the device
- [ ] `docs/decisions.md`: entry for Direction A, the rejected B/C with one line each, and D2–D5
      as taken
- [ ] `CLAUDE.md` "UI/UX rules" points at `design-system/scholarshipcoach/` as the source of truth
      for tokens and components
- [ ] All five CI commands green

---

## Execution Order

Done: 0.1 harness → 1.1 theme + fonts → 1.2 status vocabulary.

Revised 2026-09-27 across the three active plans (this one, Cloud Hosting, and the Roles &
Product Plan to be written):

```
Now, in parallel:
  Hosting Phase 1–2: repo scripts, server, backups, restore drill (no app changes)
  Refresh 1.3: nav/chrome mechanism over today's pages
  Refresh 3.1: catalog form tabs, Settings order
  Owner: observation session with the student + parent notes (no code)
  Owner: the first target school's merit awards and grade-scoped milestones entered
    via Catalog & Inbox and Settings (re-verify dates; researched values are for the
    2026–27 cycle)

Then: roles/UX review → writes the Roles & Product Plan
  Decides the role model, page list and grouping, parent visibility of essays,
  identity vs family PIN; absorbs or releases Refresh Phase 2

Roles & Product Plan, Phase A (light version), built with this design system
  Refresh 3.2 once the Timeline's owner is settled

Hosting 3.1 family onboarding (after the identity decision and Phase A ship)
  → about a month of real use → the Roles & Product Plan's go/no-go gate for its Phase B

Last, across everything:
  Refresh 4.1 full sweep + reconcile → 4.2 screenshots + docs
```

Within this plan the original reasons still hold. 0.1 came first because every task is checked
with it. 1.1 came before page work so nothing is tuned against the old palette. 1.2 came before
page tasks because they render its badges. 1.3 comes before any page content because the chrome
and layout mode change what fits at 400px. 4.2 runs last, after the screenshots stop changing.

## Success Criteria

1. Every text/background pair in both themes, including the primary button label as rendered,
   measures ≥ 4.5:1 (`tests/test_theme_config.py` plus a pixel sample).
2. At 400px the student reaches any Student section in one tap from any other, and each section has
   its own URL.
3. On This Week at 400×860, the first due item or the empty state's "Next up" is visible without
   scrolling.
4. A typical card on any list page is ≤ 260px tall at 400px. (Whether Find ranks on arrival, and
   what Student mode lists at all, moved to the Roles & Product Plan with Phase 2.)
5. No emoji status icons, no "URGENT", and no pipeline vocabulary on any Student or Parent page;
   Operator mode keeps every control it had.
6. No request leaves the server for fonts.
7. `design-system/scholarshipcoach/` describes the app as shipped.
8. All five CI commands green after every task.
