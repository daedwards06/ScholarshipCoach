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

---

## Plan Files

| File | Description |
|------|-------------|
| `docs/plans/SCHOLARSHIPCOACH_PORTFOLIO_UPGRADE_PLAN.md` | B+ → A portfolio upgrade (Phases 0–4) |
| `docs/plans/SCHOLARSHIPCOACH_UI_REDESIGN_PLAN.md` | Streamlit card-UI redesign (owns the app/UI track; absorbs Portfolio Tasks 3.2 & 3.4) |

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

## Environment

- Python 3.12, Windows 11 / PowerShell.
- Active virtualenv is `.venv/` in the project root — `conda` is **not** on PATH in this shell.
- Package is editable-installed: `from src.rank.stage1_eligibility import ...` works.
- Large artifacts (`.parquet`, win model `.joblib`, embeddings `.npz`) are git-ignored —
  do not commit them.

### Exact validation commands

```powershell
# Tests — use python -m pytest, not bare pytest (bare pytest may not resolve in this shell)
python -m pytest tests/ -q

# Lint — ruff is a standalone binary, not a Python module; do NOT use python -m ruff
ruff check src/ scripts/ app/ tests/
```
