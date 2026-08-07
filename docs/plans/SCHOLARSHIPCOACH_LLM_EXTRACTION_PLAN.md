# ScholarshipCoach LLM Extraction Plan — Structured Extraction at the Ingest Boundary

> Generated: 2026-07-06 | Scope decided with the project owner (career-feedback follow-up)
> Companion to `SCHOLARSHIPCOACH_PORTFOLIO_UPGRADE_PLAN.md`, `SCHOLARSHIPCOACH_UI_REDESIGN_PLAN.md`,
> and `SCHOLARSHIPCOACH_EVAL_CREDIBILITY_PLAN.md`
> Executor: Claude via Claude Code | Est. effort: 6 focused tasks + 1 optional stretch

---

## Why this plan exists

The project's NLP is currently encoder-only (sentence-transformer embeddings). Career
feedback identified that a **generative LLM layer** would close the "LLM applications"
portfolio gap — and the highest-value form is **LLM-powered structured extraction**:
parsing messy scholarship listings into the clean eligibility fields the pipeline
already expects (`min_gpa`, `states_allowed`, `majors_allowed`, `education_level`,
`deadline`, amounts, `essay_required`).

This is not a bolt-on demo. It attacks the project's actual binding constraint:

- The catalog is ~163 records and ingest parsing is brittle hand-rolled regex/HTML
  (`src/ingest/sources/bold_org.py` money/date patterns, etc.). Many records have
  empty structured fields that Stage 1 eligibility depends on.
- Unstructured-document → structured-data is one of the most employable LLM use cases
  (it is what financial firms do with filings and disclosures), which serves the
  project's decision-systems framing.
- The deterministic parsers already produced structured fields for the existing
  catalog — **built-in gold labels** for measuring extraction accuracy per field.
  A measured LLM feature beats an unmeasured one in any interview.

**Non-negotiable architectural constraint:** the LLM lives strictly at the ingest
boundary (Stage 0). Extractions are cached as versioned artifacts keyed by content
hash (same pattern as the embeddings store). Everything downstream of the snapshot —
Stage 1/2/3, evaluation, tuning — remains exactly as deterministic and offline as it
is today. The reproducibility story survives intact: *the LLM runs once at ingest;
the ranking pipeline never depends on a live API.*

**Relationship to other plans:**
- Catalog growth toward the 300+ gate on Portfolio Plan **Task 4.1** is served by this
  plan (enrichment makes previously-unusable records rankable) but connector expansion
  itself stays with Portfolio **Task 2.3**-style work.
- No dependency on the Eval Credibility Plan, but its Tasks 1–2 (honesty + metrics
  tests) are recommended first so evaluation infrastructure is trustworthy before new
  data flows in.
- **Privacy:** no student profile data is ever sent to an LLM API in Tasks 1–6.
  Only public scraped scholarship text leaves the machine. The optional Task 7 is the
  single exception and is explicitly opt-in with a minimal payload.

---

## Design principles for this plan

1. **LLM at the edge, determinism at the core.** LLM calls happen only during ingest,
   results are content-hash cached, and the snapshot parquet remains the single source
   of truth for everything downstream.
2. **Fill, never overwrite.** LLM enrichment only fills fields the deterministic
   parsers left empty. Parser-extracted values always win. Provenance is recorded.
3. **Validate everything.** Every LLM output is schema-validated and range-checked
   before it touches a record. An invalid extraction degrades to "field stays empty,"
   never to "garbage enters the snapshot."
4. **No key, no problem.** Without an API key the feature is cleanly disabled: ingest,
   tests, and CI all run green with zero network calls. All tests use fake clients.
5. **Measure against gold.** Extraction accuracy is reported per field against the
   deterministic parsers' outputs before the feature is advertised in the README.
6. **Lean dependencies.** The client is a thin HTTP wrapper over the existing
   `requests` dependency targeting OpenAI-compatible chat-completions endpoints
   (Gemini's OpenAI-compat endpoint, Groq, OpenRouter all speak this shape) — no
   provider SDK lock-in.
7. **Green gate every task.** `python -m pytest tests/ -q` green and
   `ruff check src/ scripts/ app/ tests/` at 0 errors after every task.

---

## Task 1: LLM Client Foundation (`src/llm/client.py`)

**Why:** Everything else needs a provider-agnostic, testable way to call a chat model.
A thin wrapper over `requests` against an OpenAI-compatible endpoint keeps the free-tier
options open (Gemini / Groq / OpenRouter) and shows API-level competence rather than
SDK plumbing.

**Preflight Files:**
- `src/ingest/http.py` (existing HTTP patterns: timeouts, retry/backoff conventions to mirror)
- `pyproject.toml` (dependencies — confirm `requests` present; add an `llm` extras group only if needed)
- `.gitignore` (ensure `.env` is ignored before any key handling exists)
- `.github/workflows/ci.yml` (confirm CI needs no secrets — tests must pass keyless)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
```

**Checklist:**
- [x] Create `src/llm/__init__.py` and `src/llm/client.py`: an `LlmClient` with a
      single `complete(system: str, user: str) -> str` method posting to an
      OpenAI-compatible `/chat/completions` endpoint via `requests`
- [x] Configuration via env vars: `SCHOLARSHIPCOACH_LLM_API_KEY`,
      `SCHOLARSHIPCOACH_LLM_BASE_URL`, `SCHOLARSHIPCOACH_LLM_MODEL`; a
      `client_from_env() -> LlmClient | None` factory returns `None` (feature
      disabled) when the key is absent — no exception, no network call
- [x] Bounded retries with backoff on 429/5xx (mirror `PoliteHttpClient` conventions);
      hard timeout; requests are temperature-0 for repeatability
- [x] The client accepts an injectable transport/session so tests run with a fake —
      zero network in the test suite
- [x] Add `.env` to `.gitignore`; never log the API key
- [x] `tests/test_llm_client.py`: env factory (present/absent key), retry path,
      timeout path, payload shape — all against the fake transport
- [x] Document provider setup (Gemini OpenAI-compat / Groq / OpenRouter base URLs,
      free-tier note with "verify current limits at signup") in `docs/llm_extraction.md` (new)
- [x] Tests + ruff green

---

## Task 2: Extraction Prompt + Schema Validation (`src/llm/extraction.py`)

**Why:** The extraction contract is the heart of the feature: one prompt, strict JSON
out, and a validator that coerces or rejects every field before it can touch a record.
The prompt is versioned so cached extractions invalidate when it changes.

**Preflight Files:**
- `src/normalize/schema.py` (`NormalizedScholarshipRecord` — the target field set and types)
- `src/ingest/sources/bold_org.py` (`_extract_amount`, `_parse_deadline` — the regex
  behavior the LLM complements; also a source of realistic messy-text fixtures)
- `src/text_utils.py` (normalization helpers for state/education-level vocab)
- `src/rank/stage1_eligibility.py` (how each structured field is consumed — defines
  what "valid" means, e.g. state codes, education-level strings)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
```

**Checklist:**
- [x] Define `EXTRACTION_PROMPT_VERSION` (bump on any prompt change — cache key input)
      and a system/user prompt asking for STRICT JSON with exactly these nullable
      fields: `deadline` (ISO date), `amount_min`, `amount_max`, `min_gpa`,
      `states_allowed` (2-letter codes), `majors_allowed`, `education_level`,
      `citizenship`, `essay_required`, `keywords` — with "null when not stated,
      never guess" instructions
- [x] `parse_extraction(raw: str) -> dict`: tolerant JSON recovery (strip code fences,
      find the JSON object), then per-field validation: dates must parse ISO and be
      plausible (2020–2040), GPA in [0, 5], amounts ≥ 0 and `min ≤ max`, states
      mapped to known 2-letter codes, education level mapped to the pipeline's
      existing vocabulary; any field failing validation is dropped to `None`
- [x] `extract_fields(client, *, title, description, eligibility_text) -> dict`:
      composes prompt → `client.complete` → `parse_extraction`; returns `{}` on any
      client error (extraction is always best-effort)
- [x] `tests/test_llm_extraction.py` with canned LLM responses (valid JSON, fenced
      JSON, malformed JSON, out-of-range values, hallucinated extra fields) — no
      network, no real client
- [x] Tests + ruff green

---

## Task 3: Content-Hash Extraction Cache (`src/llm/cache.py`)

**Why:** The cache is what reconciles a nondeterministic API with the project's
determinism principle. Keyed by content + model + prompt version, it guarantees an
unchanged listing is extracted exactly once, re-runs are free and reproducible, and
the snapshot build never depends on live API behavior.

**Preflight Files:**
- `src/embeddings/cache.py` (the pattern to mirror: fingerprint → key → sidecar store,
  `ensure_*` idempotent API)
- `docs/system_design.md` ("Embedding Cache Artifact" section — the new cache gets a
  parallel section later, in Task 6)
- `.gitignore` (`data/processed/embeddings/` precedent — add the new artifact dir)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
```

**Checklist:**
- [x] Cache key = SHA1 of (source text fingerprint: title + description +
      eligibility_text, model name, `EXTRACTION_PROMPT_VERSION`)
- [x] Store extractions as JSON under
      `data/processed/llm_extractions/<model_sanitized>/<key>.json` including the
      validated fields plus metadata (`extracted_at`, model, prompt version)
- [x] `get_or_extract(client, cache_dir, record_fields) -> dict`: cache hit returns
      stored result with **no API call**; miss calls `extract_fields`, validates,
      writes, returns; a `client=None` (disabled) with a cache miss returns `{}`
- [x] Add `data/processed/llm_extractions/` to `.gitignore` (build artifact, same
      policy as embeddings)
- [x] `tests/test_llm_extraction_cache.py`: hit avoids the client (assert fake client
      not called), miss writes-then-reads identically, prompt-version bump changes the
      key, disabled-client behavior
- [x] Tests + ruff green

---

## Task 4: Enrichment Pass in Ingest (`--llm-enrich`)

**Why:** This wires extraction into the pipeline where it pays off: records whose
structured eligibility fields are empty (so Stage 1 can't reason about them) get
filled from their own description/eligibility text. Fill-only semantics and provenance
keep the deterministic parsers authoritative.

**Preflight Files:**
- `scripts/run_ingest.py` (`parse_args` ~L29–46, `_normalize_records` ~L95,
  `run_ingest` ~L175 — where the enrichment pass inserts, after parse/normalize and
  before snapshot build)
- `src/io/snapshotting.py` (`REQUIRED_COLUMNS`, snapshot build — new provenance column
  must survive the snapshot round-trip)
- `src/llm/cache.py`, `src/llm/client.py` (Tasks 1–3 outputs)
- `tests/test_run_ingest.py` (existing ingest test patterns to extend)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
# Keyless smoke run — enrichment silently disabled, ingest still green:
python scripts/run_ingest.py --max-listing-pages 1 --max-detail-pages 5 --llm-enrich
```

**Checklist:**
- [ ] Add `--llm-enrich` (default off) and `--llm-max-calls <n>` (default e.g. 100,
      free-tier guardrail) to `run_ingest.py`
- [ ] Enrichment pass: for each normalized record with at least one empty target
      field AND non-empty description/eligibility text, call `get_or_extract`
      (cache-first) and fill **only** the empty fields — parser values are never
      overwritten
- [ ] Record provenance: an `llm_enriched_fields` column (list of field names filled
      by the LLM, empty list otherwise) that survives snapshot write/read
- [ ] Respect `--llm-max-calls` for actual API calls (cache hits don't count);
      log a clear summary: records scanned / cache hits / API calls / fields filled
- [ ] With no API key: `--llm-enrich` logs "LLM enrichment disabled (no key)" and the
      run completes normally (cache-only fills still apply)
- [ ] Ingest report JSON gains an `llm_enrichment` block with the summary counts
- [ ] Tests (fake client): fill-only semantics, provenance column, max-calls cap,
      disabled path, snapshot round-trip of the new column
- [ ] Tests + ruff green

---

## Task 5: Extraction Quality Evaluation (Gold = Deterministic Parsers)

**Why:** This is what elevates the feature from "I called an API" to "I measured an
LLM system." The existing catalog has fields the regex parsers extracted successfully —
free gold labels. Blind the LLM to those fields, re-extract from raw text, and score
per-field agreement.

**Preflight Files:**
- `src/io/snapshotting.py` (`load_latest_snapshot_df` — evaluation input)
- `src/llm/extraction.py`, `src/llm/cache.py` (extraction path under test)
- `scripts/evaluate_golden_students.py` (report-writing conventions: markdown to
  `reports/`, JSON artifact alongside)
- `reports/` naming conventions (timestamped files)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
# Requires SCHOLARSHIPCOACH_LLM_API_KEY (run manually, not in CI):
python scripts/evaluate_llm_extraction.py --sample-size 60
```

**Checklist:**
- [ ] Create `scripts/evaluate_llm_extraction.py`: sample N snapshot records where the
      parsers populated at least one target field; run LLM extraction on their raw
      text (cache-aware); compare per field
- [ ] Metrics per field: exact-match accuracy for scalars (deadline, gpa, amounts
      within $1, education_level, essay_required); set precision/recall for list
      fields (states, majors, keywords); plus abstention rate (LLM said null where
      gold has a value) and hallucination rate (LLM gave a value where gold is null —
      reported separately and honestly labeled *unverified*, since gold-null may mean
      "parser missed it," not "not stated")
- [ ] Write `reports/llm_extraction_eval_<timestamp>.md` + JSON artifact with the
      per-field table, sample size, model, and prompt version
- [ ] Unit-test the comparison/metric logic with fixture frames (no network); the
      live run stays manual
- [ ] Run the live evaluation once with a real key; keep the resulting report as a
      curated artifact (it feeds the README table in Task 6)
- [ ] Tests + ruff green

---

## Task 6: Documentation + README Repositioning

**Why:** The portfolio payoff. The README currently describes an embeddings-only
system; after Tasks 1–5 the project is a genuine LLM application with a measured
extraction layer — and the docs must present it with the same honesty standard as the
rest of the project.

**Preflight Files:**
- `README.md` (architecture diagram ~L39–68, Design Principles ~L220–228, Future Work)
- `docs/system_design.md` (add an "LLM Extraction Layer" section parallel to the
  embedding-cache section)
- `docs/llm_extraction.md` (created in Task 1 — flesh out end-to-end)
- The curated evaluation report from Task 5 (source of the accuracy table)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
```

**Checklist:**
- [ ] README: add "LLM-Assisted Extraction" section — what it does, fill-only +
      validate + cache design, the per-field accuracy table from Task 5, and an
      honest note that accuracy is measured against parser outputs (not human gold)
- [ ] README: update the Stage 0 box in the architecture diagram (mention LLM
      enrichment, cached + optional) and amend Design Principles: determinism is
      preserved because LLM calls happen only at ingest and are content-hash cached;
      the ranking pipeline remains fully offline
- [ ] README: setup subsection — env vars, provider options (Gemini / Groq /
      OpenRouter OpenAI-compatible endpoints), "runs fine without a key" note
- [ ] `docs/system_design.md`: LLM extraction layer section (cache key anatomy,
      prompt versioning, fill-only semantics, provenance column)
- [ ] Remove/adjust any now-stale "local-only" phrasing in docs so no doc contradicts
      the new architecture (ranking stays local-only; ingest is optionally LLM-assisted)
- [ ] Tests + ruff green

---

## Task 7 (OPTIONAL / STRETCH): Grounded Ranking Explanations in the UI

**Why:** A small generative feature that makes demos feel alive: turn each card's
component scores into 2–3 plain-English sentences. Strictly *explanation grounded in
existing scores* — not essay drafting (integrity optics, unmeasurable) and not free-form
advice. Skip freely if Tasks 1–6 already tell the story.

**Preflight Files:**
- `app/helpers.py` (`explain_ranked_row` — the deterministic explanation the LLM
  version augments, and the fallback when disabled)
- `app/main.py` (card render + "Signal details" expander; operator/Advanced area for
  the toggle)
- `src/llm/client.py`, `src/llm/cache.py`

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
streamlit run app/main.py   # visual: toggle off by default; cards unchanged when off
```

**Checklist:**
- [ ] Opt-in toggle in the Advanced/operator area, default OFF; without a key the
      toggle is hidden or disabled with a hint
- [ ] Prompt receives ONLY: scholarship title, the component scores/signals, and
      non-identifying profile fields (major, education level, interest keywords) —
      never name, GPA, state, or the saved profile JSON
- [ ] Output constrained to explaining the provided signals ("high keyword overlap,
      deadline in 12 days"), rendered on the card above the deterministic signal list;
      `explain_ranked_row` remains the fallback
- [ ] Responses cached (profile-fields hash + scholarship id + scores hash) so
      re-renders don't re-call the API
- [ ] Tests with a fake client: payload contains no restricted fields; disabled path
      renders identically to today
- [ ] Tests + ruff green

---

## Execution Order

```
1  Client foundation        (everything depends on it)
2  Extraction + validation  (depends on 1)
3  Content-hash cache       (depends on 2 — needs the prompt version)
4  Ingest enrichment pass   (depends on 3)
5  Extraction evaluation    (depends on 4; needs one manual keyed run)
6  Docs + README            (last of core — needs 5's accuracy table)
7  UI explanations          (optional; any time after 3)
```

## Success Criteria

1. `pip install -e .` + no API key = today's behavior exactly: ingest, tests, CI all
   green with zero network calls to any LLM.
2. With a key, `--llm-enrich` fills empty structured fields from listing text —
   validated, fill-only, provenance-tracked — and an unchanged listing never triggers
   a second API call (content-hash cache).
3. Ranking output for a given snapshot is bit-identical whether or not the LLM was
   used to build it — determinism lives in the snapshot, not the API.
4. A per-field extraction accuracy report exists in `reports/`, measured against the
   deterministic parsers, with abstention/hallucination rates reported honestly.
5. The README presents the project as an LLM application with a measured extraction
   layer, without weakening any existing reproducibility claim.
6. No student profile data is sent to any API in the core feature (Task 7, if built,
   is opt-in with a minimal non-identifying payload).
7. `python -m pytest tests/ -q` green and `ruff check src/ scripts/ app/ tests/` at
   0 errors after every task.
