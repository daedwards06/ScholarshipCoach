# How to Run Plan Tasks

Quick reference for executing tasks from any plan in `docs/plans/` using either
Claude Code or GitHub Copilot.

---

## Where Plans Live

All plan files are in `docs/plans/` and follow the `*_PLAN.md` naming convention.

| Plan | File |
|------|------|
| ScholarshipCoach Portfolio Upgrade | `docs/plans/SCHOLARSHIPCOACH_PORTFOLIO_UPGRADE_PLAN.md` |

> To keep plans out of the public repo, add `docs/plans/` to `.gitignore`.

---

## Using Claude Code

Claude Code auto-loads `CLAUDE.md` at session start, so the full protocol
(preflight → implement → validate → update checkboxes) runs automatically.

**One plan in docs/plans/:**
```
Implement Task 1.2 from the plan.
```

**Multiple plans — name it:**
```
Implement Task 2.1 from the Portfolio Upgrade plan.
```

That's all you need to type. Claude Code finds the plan, reads the preflight files,
confirms understanding, implements, runs the validation commands, and updates the
checkboxes — without further prompting.

---

## Using GitHub Copilot (Agent Mode in VS Code)

`.github/copilot-instructions.md` loads the protocol automatically in Agent mode.
You still need to attach the plan file manually with `#file:`.

> **Note:** Each new Copilot chat window loses context. You need the `#file:` reference
> again at the start of every new chat — it does not carry over between sessions.

**One plan:**
```
#file:docs/plans/SCHOLARSHIPCOACH_PORTFOLIO_UPGRADE_PLAN.md

Implement Task 1.2 from the plan.
```

**Multiple plans — attach the specific file:**
```
#file:docs/plans/FEATURE_ROADMAP_PLAN.md

Implement Task 2.1 from the plan.
```

Copilot then follows the same protocol: reads the preflight files, confirms
understanding, implements, validates, and updates checkboxes.

---

## What the Protocol Does (Both Tools)

1. **Preflight** — reads every file listed in the task's `Preflight Files` section
   before touching any code.
2. **Confirm** — writes a 2–4 sentence summary of current state and what will change.
   No code is written until this step is done.
3. **Implement** — follows the task's Checklist and Prompt as the specification.
4. **Validate** — runs the task's `Validation Commands` and shows the full output.
5. **Update** — checks off completed items in the plan file (`- [ ]` → `- [x]`).

---

## If the AI Skips a Step

The protocol only works if the AI actually follows it. If it jumps straight to writing
code without confirming understanding first, push back:

```
Stop — you skipped the preflight. Read the Preflight Files listed in the task first,
then tell me what the current state is before writing any code.
```

The confirmation step (Step 2) should look something like this before any code appears:

> "Current state: `_normalize_text` is privately defined in 7 files with slightly
> different behavior. The gap: no shared implementation exists. I'm about to create
> `src/text_utils.py` with a canonical `normalize_text()` function and replace all
> 7 private copies with imports from it."

If you don't see something like that before the code, the preflight was skipped.

---

## Writing New Plan Tasks

For the protocol to work, every task in a plan needs these two sections in addition
to the standard Why / Checklist / Prompt structure:

```markdown
**Preflight Files:**
- `path/to/file1.py`   (one-line note on why this file matters)
- `path/to/file2.py`

**Validation Commands:**
\```powershell
pytest tests/ -q
ruff check src/ scripts/ app/ tests/
\```
(Note what to look for in the output — e.g. "all 45 tests pass", "> 0 keyword_overlap values")
```

Without `Preflight Files`, the AI has no anchor for the preflight step and may skip it.
Without `Validation Commands`, there is no exit condition — "done" becomes subjective.

---

## Adding a New Plan

1. Create the file in `docs/plans/` with a `_PLAN.md` suffix.
2. Add a row to the Plan Files table in:
   - `CLAUDE.md`
   - `.github/copilot-instructions.md`
   - The table above in this file.
3. Each task in the plan should have a **Preflight Files** and **Validation Commands**
   section so the protocol has everything it needs.
