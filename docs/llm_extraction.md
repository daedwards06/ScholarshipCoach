# LLM-Assisted Extraction

ScholarshipCoach uses a generative LLM at the **ingest boundary (Stage 0) only**:
messy scholarship listings are parsed into the clean structured eligibility fields
the ranking pipeline already expects. Extractions are validated and content-hash
cached, so everything downstream of the snapshot stays deterministic and offline.

> **This feature is optional.** Without an API key it is cleanly disabled — ingest,
> tests, and CI all run green with zero LLM network calls. The ranking pipeline
> never depends on a live API.

## Client foundation (Task 1)

`src/llm/client.py` provides `LlmClient`, a thin wrapper over `requests` targeting
any OpenAI-compatible `/chat/completions` endpoint. It has a single method:

```python
LlmClient.complete(system: str, user: str) -> str
```

- Requests are **temperature 0** for repeatability.
- Transient failures (HTTP 429, 5xx, and network errors) are retried with
  exponential back-off, mirroring `src.ingest.http.PoliteHttpClient` conventions.
- A hard connect/read timeout is always applied.
- The transport/session is injectable, so the test suite runs with a fake and
  makes **zero network calls**.
- The API key is sent as a bearer token and is **never logged**.

## Extraction contract (Task 2)

`src/llm/extraction.py` holds the extraction contract: one versioned prompt, a
strict-JSON response, and a validator that runs before any value can reach a record.

```python
extract_fields(client, title=..., description=..., eligibility_text=...) -> dict
parse_extraction(raw: str) -> dict
```

The model is asked for exactly ten nullable fields — `deadline`, `amount_min`,
`amount_max`, `min_gpa`, `states_allowed`, `majors_allowed`, `education_level`,
`citizenship`, `essay_required`, `keywords` — under a "null when not stated, never
guess" instruction.

Every response is then recovered and validated:

- **Tolerant recovery.** Code fences are stripped and a brace-balanced JSON object
  is located inside any surrounding prose. Unrecoverable output yields `{}`.
- **Per-field validation.** Deadlines must parse as ISO dates in 2020–2040; GPA must
  fall in `[0, 5]`; amounts must be non-negative with `min ≤ max`; states map to
  known two-letter codes; education level and citizenship map onto the vocabulary
  Stage 1 already reasons about (`high school` / `undergraduate` / `graduate`;
  `us` / `permanent resident` / `international`).
- **Drop, never guess.** Unknown keys are discarded and any field failing validation
  is dropped from the result. The returned dict therefore contains *only* fields with
  a usable value, which is exactly what the fill-only enrichment pass consumes.
- **Best-effort.** Any client error returns `{}` — a flaky provider can never fail an
  ingest run.

`EXTRACTION_PROMPT_VERSION` is an input to the extraction cache key, so bumping it on
a prompt or validation change invalidates stale cached extractions.

## Configuration

The client is entirely environment-driven:

| Env var | Purpose | Default |
|---------|---------|---------|
| `SCHOLARSHIPCOACH_LLM_API_KEY` | Provider API key. **Required** to enable the feature. | — (feature disabled) |
| `SCHOLARSHIPCOACH_LLM_BASE_URL` | Endpoint root (the `/chat/completions` suffix is appended). | Gemini OpenAI-compat endpoint |
| `SCHOLARSHIPCOACH_LLM_MODEL` | Model identifier. | `gemini-2.0-flash` |

`client_from_env()` returns an `LlmClient` when the key is present, or `None`
(feature disabled — no exception, no network call) when it is absent or blank.

Store secrets in a local `.env` file (git-ignored) — never commit keys.

## Provider options

Any provider exposing an OpenAI-compatible chat-completions surface works. Common
free-tier options (**verify current limits and endpoints at signup — they change**):

| Provider | `SCHOLARSHIPCOACH_LLM_BASE_URL` | Example model |
|----------|--------------------------------|---------------|
| Google Gemini (OpenAI-compat) | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-2.0-flash` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.1-8b-instant` |
| OpenRouter | `https://openrouter.ai/api/v1` | (see OpenRouter model list) |

### Example (PowerShell)

```powershell
$env:SCHOLARSHIPCOACH_LLM_API_KEY = "your-key-here"
$env:SCHOLARSHIPCOACH_LLM_BASE_URL = "https://api.groq.com/openai/v1"
$env:SCHOLARSHIPCOACH_LLM_MODEL = "llama-3.1-8b-instant"
```

## Design guarantees

1. **LLM at the edge, determinism at the core.** LLM calls happen only during
   ingest; the snapshot parquet remains the single source of truth downstream.
2. **No key, no problem.** Absent a key the feature is a no-op; nothing breaks.
3. **Never log the key.** The bearer token is never written to logs.

Later tasks build on this foundation: extraction prompt + schema validation
(Task 2), a content-hash cache (Task 3), the ingest enrichment pass (Task 4),
quality evaluation against the deterministic parsers (Task 5), and documentation
(Task 6).
