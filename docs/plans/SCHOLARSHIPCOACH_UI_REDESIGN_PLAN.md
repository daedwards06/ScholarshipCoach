# ScholarshipCoach UI Redesign Plan — Card UI, Single Page

> Generated: 2026-07-01 | Scope decided with the project owner
> Companion to `SCHOLARSHIPCOACH_PORTFOLIO_UPGRADE_PLAN.md` (this file owns the app/UI track)
> Executor: Claude via Claude Code | Est. effort: 5-6 focused tasks

---

## Why this plan exists

The Streamlit app (`app/main.py`) currently presents as a **developer console, not a
product**: raw variable-name labels (`gpa`, `education_level`, `profile_keywords
(comma-separated)`), a pipeline-jargon subtitle ("Ingest -> Snapshot/Delta -> Stage 1 ->
Stage 2 -> Stage 3"), operator tooling (ingest, win-model training, weights-JSON selectors)
in the main flow, and results rendered as a spreadsheet + raw `st.json(...)` score dumps.

For a portfolio project the app is the most visual asset, so this drift matters. Taking
README screenshots of the current UI would make the project look *less* finished than the
code actually is — which is why the screenshots task (previously Portfolio Plan Task 3.2) is
pulled into this plan and sequenced **last**, after the redesign.

**Chosen scope:** card redesign, single page. Keep one page; rewrite the results area as
scholarship cards; humanize the profile form; collapse operator controls into an "Advanced"
area; add a theme. (Multipage was considered and declined.)

**Relationship to the Portfolio Upgrade Plan:**
- Portfolio Plan **Task 3.4** (`.streamlit/config.toml`) is **absorbed** into Task 1 below.
- Portfolio Plan **Task 3.2** (README screenshots) is **moved** here as Task 6.
- Both are cross-referenced in the Portfolio Plan and should not be executed there.

---

## Design principles for this redesign

1. **User-first surface, operator-second.** A student sees: enter profile → get ranked
   scholarships with plain-English reasons. Ingest / win-model / weights internals live
   behind an "Advanced" expander, present but out of the way.
2. **Behavior-preserving.** The pipeline (`apply_eligibility_filter` → `score_stage2` →
   `rerank_stage3`) and session-state flow are **not** changed. This is a presentation-layer
   refactor. Ranking output must be identical for the same inputs.
3. **Reuse existing helpers.** `explain_ranked_row()` and `format_amount_range()` in
   `app/helpers.py` already produce human-readable output — the cards consume them rather
   than re-implementing.
4. **Keep the escape hatch.** Raw component scores stay reachable in a per-card "signal
   details" expander, so nothing debuggable is lost.
5. **Green gate every task.** `python -m pytest tests/ -q` stays green and
   `ruff check src/ scripts/ app/ tests/` stays at 0 errors after every task.

---

## Task 1: Theme + Page Config (`.streamlit/config.toml`)

**Why:** A theme is the highest visual-payoff / lowest-risk change — it re-skins the entire
app with no logic touched. Doing it first means every later task is built and screenshotted
against the final look. This absorbs Portfolio Plan Task 3.4.

**Preflight Files:**
- `app/main.py` (the `st.set_page_config(...)` call at the top of `main()`)
- `.gitignore` (confirm whether `.streamlit/secrets.toml` needs ignoring)
- `README.md` ("Running the Project" section — where a deployment note will go)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
streamlit run app/main.py   # visual: dark theme + accent applied app-wide
```

**Checklist:**
- [x] Create `.streamlit/config.toml` with a dark theme (accent `#4ECDC4`, bg `#0E1117`,
      secondary bg `#1A1A2E`, text `#FAFAFA`) and `[server] headless = true`
- [x] Add `.streamlit/secrets.toml` to `.gitignore` if not already ignored
- [x] Confirm `st.set_page_config` still sets a sensible `page_title` / `layout`
- [x] Add a one-line deployment note to README pointing at the config
- [x] App loads with the theme; tests + ruff green

---

## Task 2: Humanize the Profile Form

**Why:** The sidebar inputs read like database columns. Human labels, a user-facing
title/subtitle, and light grouping make the app legible to a non-engineer in five seconds —
the single cheapest credibility win in the UI.

**Preflight Files:**
- `app/main.py` (title/caption at ~L384-385; sidebar profile inputs at ~L389-403)
- `app/helpers.py` (no change — just confirm nothing here hard-codes labels)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
streamlit run app/main.py   # visual: friendly labels, no raw field names
```

**Checklist:**
- [x] Replace the page title/caption with user-facing copy (e.g. "Scholarship Coach —
      scholarships matched to your profile"), drop the pipeline-stage subtitle
- [x] Relabel every profile input to plain English: GPA, State, Intended Major, Grade Level,
      Interests / Keywords, Goals — with helpful placeholders/help text
- [x] Keep all `key=` session-state bindings identical (labels change, keys do not)
- [x] Group related inputs (e.g. academic vs. interests) for scannability
- [x] Tests + ruff green; Save/Load Profile still works

---

## Task 3: Split User Mode from Operator Mode

**Why:** Ingest, win-model training, and the weights-profile JSON selector are *your*
operational controls, not a student's. Moving them into a collapsed "Advanced / Operator"
area declutters the main flow without removing any capability.

**Preflight Files:**
- `app/main.py` (sidebar Similarity ~L425, Ranking Weights ~L438, Win Model ~L471; main-area
  Data Update ~L530, and the status/`st.json` dumps at ~L585-611)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
streamlit run app/main.py   # visual: operator controls collapsed, pipeline still runnable
```

**Checklist:**
- [x] Move Similarity mode, Ranking Weights profile, and Win Model controls into a collapsed
      `st.expander("Advanced / Operator")` (sidebar or a clearly separated section)
- [x] Move the "Run Update (Ingest)" / "Use Latest Snapshot" controls and ingest/delta
      reports into the same operator area
- [x] Remove the raw `st.write(status_payload)` / `st.json(weights)` dumps from the default
      view; relocate them under the operator expander
- [x] Default (collapsed) view shows: profile → "Run Scholarship Coach" → results
- [x] All controls still function; session-state keys unchanged; tests + ruff green

---

## Task 4: Scholarship Card Component (Results)

**Why:** This is the centerpiece. The current `st.dataframe` + JSON expander reads as a data
console. Cards — award amount, deadline urgency, and a plain-English "why this matches you" —
turn the output into something a student (and a recruiter) immediately understands.

**Preflight Files:**
- `app/main.py` (results block ~L664-740: the dataframe render, the per-row expander loop,
  and `_apply_rank_filters`)
- `app/helpers.py` (`explain_ranked_row`, `format_amount_range`)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
streamlit run app/main.py   # visual: ranked cards replace the dataframe; ordering identical
```

**Checklist:**
- [x] Add a `_render_scholarship_card(row)` helper in `app/main.py` (or `app/helpers.py`)
      rendering: title, award amount (`format_amount_range`), deadline with an urgency
      indicator, a "Why this matches you" list (`explain_ranked_row`), and an Apply/source
      link
- [x] Replace the top-N `st.dataframe` render with a loop of cards (preserve existing rank
      order and the top-N / search / deadline / amount filters)
- [x] Verify the same inputs produce the same ordering as before (behavior-preserving)
- [x] Keep the win-model top-k summary available but unobtrusive (operator area or a small
      caption)
- [x] Tests + ruff green

---

## Task 5: Signal Details + Excluded-List Polish

**Why:** Power users (and you) still want the raw component scores and the Stage-1 exclusion
reasons. Tucking them into per-card expanders and a tidy excluded section keeps the default
view clean while losing no information.

**Preflight Files:**
- `app/main.py` (per-row component-score block ~L706-740; excluded-scholarships block
  ~L742-769)
- `app/helpers.py` (`reasons_to_text`)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
streamlit run app/main.py   # visual: signal details behind a per-card expander
```

**Checklist:**
- [x] Put the raw component-score JSON behind a per-card "Signal details" expander
- [x] Keep the "Why ranked" sentences visible on the card; raw numbers only in the expander
- [x] Restyle the "Excluded Scholarships (Stage 1 Reasons)" section for readability; keep the
      reason-code filter
- [x] Nothing debuggable is lost vs. the old view; tests + ruff green

---

## Task 6: Screenshots → README (moved from Portfolio Plan Task 3.2)

**Why:** With the redesign done, screenshots finally represent the project well. Recruiters
browse GitHub without cloning — 3 screenshots prove the app exists, works, and looks
professional. (Originally Portfolio Upgrade Plan Task 3.2.)

**Preflight Files:**
- `README.md` (insertion point between "Running the Project" and "Design Principles")
- `app/main.py` (to caption the redesigned sections accurately — cards, not a table)

**Validation Commands:**
```powershell
streamlit run app/main.py   # capture screenshots manually
```
(Save to `docs/images/` as `ranking_view.png`, `profile_sidebar.png`, `explainability.png`,
each < 500 KB.)

**Checklist:**
- [ ] Create `docs/images/` (with `.gitkeep`)
- [ ] Capture 3 screenshots of the redesigned app: card ranking view, profile form,
      signal-details / explainability expander
- [ ] Add a "Screenshots" section to README with the three images and captions that match
      the **new** card UI (not the old dataframe)
- [ ] Add a note: launch with `streamlit run app/main.py` to explore interactively
- [ ] Images committed and render on GitHub

---

## Execution Order

```
1  Theme + config          (do first — reskins everything, absorbs Portfolio 3.4)
2  Humanize profile form    (independent)
3  Split user/operator mode (independent; behavior-preserving reorg)
4  Card component           (centerpiece; after 1-3 so cards inherit theme + clean layout)
5  Signal details polish    (after 4 — expanders hang off the cards)
6  Screenshots → README     (last — needs the finished UI; was Portfolio 3.2)
```

## Success Criteria

1. Default view is a student-legible flow: profile → "Run Scholarship Coach" → ranked cards.
2. No raw variable-name labels; no pipeline-jargon subtitle; no `st.json` status dumps in the
   default view.
3. Results are cards with amount, deadline urgency, and plain-English match reasons; raw
   scores reachable via a per-card expander.
4. Operator controls (ingest, win-model, weights) are present but collapsed.
5. Themed via `.streamlit/config.toml`.
6. Ranking output is identical to pre-redesign for the same inputs (presentation-only change).
7. `python -m pytest tests/ -q` green and `ruff check src/ scripts/ app/ tests/` at 0 errors
   after every task.
8. Three README screenshots reflect the redesigned card UI.
