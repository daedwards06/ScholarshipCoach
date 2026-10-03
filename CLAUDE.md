# ScholarshipCoach — Claude Code Instructions

## Task Execution Protocol

When told **"Implement Task X.Y"** (with or without a specific plan file named), follow
this protocol exactly:

### Locating the plan
- All plan files live in `docs/plans/` and match the pattern `*_PLAN.md`.
- If the user names a specific plan (e.g. "from the Portfolio Upgrade Plan"), find that file.
- If only one plan exists in `docs/plans/`, use it automatically.
- If multiple plans exist and none is specified, list them and ask which one to use.

### Step 1 — Preflight (read before writing any code)
1. Open the identified plan file and locate the task by ID (e.g. "Task 1.2").
2. Read every file listed in the task's **Preflight Files** section.
3. After reading, write a short confirmation (2–4 sentences): what the current state is,
   what the gap is, and what you are about to change. Do not start coding until this is done.

### Step 2 — Implement
- Follow the task's **Checklist** and **Prompt** section as the specification.
- Prefer editing existing files over creating new ones.
- Write no comments unless the WHY is non-obvious.
- Do not add features, refactor, or abstract beyond what the task requires.

### Step 3 — Validate
- Run every command listed in the task's **Validation Commands** section.
- Show the full terminal output to the user.
- If any command fails, fix the issue before proceeding.

### Step 4 — Update the plan
- For each checklist item you can verify (test passed, file created, command ran green),
  change `- [ ]` to `- [x]` in the plan file.
- Only check off items you can confirm — do not speculatively mark things done.

### Step 5 — Deploy (only after a push the user asked for)
- Committing and pushing still need the user's say-so. Once `main` is pushed, offer to deploy;
  run it only when the user agrees.
- Deploy with `.\scripts\deploy.ps1` (PowerShell). It waits for green CI on the pushed commit,
  then runs `deploy/update.sh` on the server over Tailscale SSH. Show its full output.
- If it fails, report the failing step from the output. Do not retry with `-SkipCi`, and do not
  SSH in to hand-fix the server, unless the user says to.
- Mention it explicitly when the push adds a file under `src/store/migrations/`: the update
  migrates the family database irreversibly (after a backup).

---

## Plan Files

| File | Description |
|------|-------------|
| `docs/plans/SCHOLARSHIPCOACH_PORTFOLIO_UPGRADE_PLAN.md` | B+ → A portfolio upgrade (Phases 0–4) |
| `docs/plans/SCHOLARSHIPCOACH_UI_REDESIGN_PLAN.md` | Streamlit card-UI redesign (owns the app/UI track; absorbs Portfolio Tasks 3.2 & 3.4) |
| `docs/plans/SCHOLARSHIPCOACH_EVAL_CREDIBILITY_PLAN.md` | Evaluation credibility, matching quality, CI/repo hardening (from 2026-07-05 project review; Task 5 refines Portfolio Task 2.1) |
| `docs/plans/SCHOLARSHIPCOACH_LLM_EXTRACTION_PLAN.md` | Generative-LLM structured extraction at the ingest boundary (cached, fill-only, measured vs. parser gold; optional UI explanations) |
| `docs/plans/SCHOLARSHIPCOACH_FAMILY_PRODUCT_PLAN.md` | Family product track (from 2026-09-12 product review): curated catalog + stable IDs, expanded profile/eligibility axes, timeline buckets, deterministic ingest automation, tracker/essays/modes, hosting; defers LLM Plan Tasks 6–9 |
| `docs/plans/SCHOLARSHIPCOACH_CATALOG_INTEGRITY_PLAN.md` | Post-implementation fixes (from 2026-09-13 review of the Family Product Plan): rebuild carry-forward + blocking guardrail, trust enforced in Stage 1, cross-source dedupe, blocked-vs-dead verification, `needs_date` bucket, NC local award seeding, student onboarding, Outcomes page, `app/main.py` split; closes Family Plan 3.2/4.3 leftovers and UI Plan Task 6 |
| `docs/plans/SCHOLARSHIPCOACH_CLOUD_HOSTING_PLAN.md` | Move hosting off the home PC to a small cloud server reachable only over Tailscale (from 2026-09-26 hosting discussion): `deploy/` service/bootstrap/update scripts, nightly restic backups + restore drill, server-side catalog edits flow back via a `server` branch PR, family phone onboarding; supersedes the home-PC hosting in `docs/operations.md`. Updated 2026-09-27: backup before every update (migrations auto-apply), Task 3.1 onboarding waits on the Roles & Product Plan's access decision and Phase A |
| `docs/plans/SCHOLARSHIPCOACH_DESIGN_REFRESH_PLAN.md` | "Study Hall" design refresh (from 2026-09-26 `ui-ux-pro-max` audit at 400px/1366px): screenshot harness, light+dark contrast-tested theme with self-hosted fonts, emoji-free deadline badges, top `st.navigation`, results-first Find with compact cards, empty states, Parent form/timeline density; spec lives in `design-system/scholarshipcoach/` (MASTER + student/parent overrides); supersedes UI Redesign Plan theme values. Updated 2026-09-27: Phase 2 (student pages) ON HOLD pending the Roles & Product Plan; Task 1.3 is nav mechanism only |
| `docs/plans/SCHOLARSHIPCOACH_ROLES_PRODUCT_PLAN.md` | Roles & product (from 2026-10-03 roles/UX review): Student mode as her path to her first target school (My Path checklist, shared To-dos, Opportunities with catalog `kind: program`, What If as GPA → goals bridge), Parent mode as the money view (Find, Applications, Colleges & Money, Timeline); Tailscale identity with PIN fallback; essays private until shared; ordered to fall back to a parent-led model; go/no-go gate after a month of real use before Phase B. Owns the page list; absorbs Design Refresh Phase 2. Public repo: never name the target school in committed files |

*(Add new plan files to this table as they are created.)*

---

## Project Context

- **What it is:** A multi-stage scholarship recommendation system built for a real student
  (NC, Computer Science / Computer Engineering, rising sophomore, GPA 3.25).
- **Pipeline:** Ingest → Stage 1 (eligibility filter) → Stage 2 (semantic scoring) →
  Stage 3 (decision reranking with optional win model).
- **Source code:** `src/` — editable install via `pip install -e .` (`pyproject.toml`).
- **Tests:** `pytest tests/ -q` — must stay green after every task.
- **Lint:** `ruff check src/ scripts/ app/ tests/` — must stay at 0 errors.
- **Types:** `python -m mypy src/` and `python -m mypy app/` — must stay at 0 errors.
- **Catalog:** `python scripts/validate_catalog.py` — must pass.

## UI/UX rules

- The app is Streamlit (1.54). Never generate React, Vue, Svelte, Tailwind, shadcn, or standalone
  HTML pages/components. Do not use the `ui-styling`, `design`, `brand`, `banner-design`, or
  `slides` skills for this app.
- Use `ui-ux-pro-max` for design decisions only: palette, typography, spacing, UX guidelines,
  chart choices, anti-patterns. It has no Streamlit stack; ignore its stack-specific code.
- Apply designs in this order:
  1. `.streamlit/config.toml` `[theme]` — colors, fonts, radius (check option names against 1.54)
  2. Native layout — `st.columns`, `st.container`, `st.tabs`, `st.expander`, `st.badge`, etc.
  3. CSS only where theming can't reach, added to `phone_width_css()` in `app/helpers.py`
     (the single injected stylesheet). Target `data-testid` attributes or `.st-key-<key>`
     classes from keyed containers — never `st-emotion-cache-*` class names.
- Small inline HTML fragments via `st.markdown(unsafe_allow_html=True)` are allowed only when no
  native element does the job; never interpolate user or catalog text into them unescaped.
- Student surfaces are checked at ~400px width (see `docs/operations.md` "Phone width").
- Design source of truth: `design-system/scholarshipcoach/MASTER.md`, with overrides in
  `design-system/scholarshipcoach/pages/` (`student-mode.md`, `parent-mode.md`) that win over
  MASTER for their mode. Tokens there are translated into config.toml / the CSS helper, not
  pasted as-is. The design system decides *what*; these rules decide *how*.

## Environment

- Python 3.12, Windows 11 / PowerShell.
- Active virtualenv is `.venv/` in the project root — `conda` is **not** on PATH in this shell.
- Package is editable-installed: `from src.rank.stage1_eligibility import ...` works.
- Large artifacts (`.parquet`, win model `.joblib`, embeddings `.npz`) are git-ignored —
  do not commit them.

### Exact validation commands

These are exactly what CI runs. A task is not validated until all of them are green —
running only tests and lint is how a mypy break reached `main` after Task 1.4.

```powershell
# Tests — use python -m pytest, not bare pytest (bare pytest may not resolve in this shell)
python -m pytest tests/ -q

# Lint — ruff is a standalone binary, not a Python module; do NOT use python -m ruff
ruff check src/ scripts/ app/ tests/

# Type check — CI runs this and it is not covered by ruff
python -m mypy src/
python -m mypy app/

# Curated catalog schema validation — CI runs this whenever catalog records change
python scripts/validate_catalog.py
```
