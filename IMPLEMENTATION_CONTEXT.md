ScholarshipCoach – Portfolio-Grade Data Science System

1. Project Overview
ScholarshipCoach is a personal-use, portfolio-visible data science system that:
  • Ingests scholarship data from live sources
  • Normalizes and snapshots structured artifacts (parquet)
  • Applies a multi-stage ranking pipeline
  • Provides explainable eligibility filtering
  • Produces deterministic evaluation outputs
This is NOT a chatbot wrapper.
This is a multi-stage ranking system with evaluation harness and reproducible artifacts.

2. High-Level Architecture
Data Flow
Ingest Sources
→ Raw Cache (data/raw/)
→ Normalized Records
→ Snapshot (parquet)
→ Delta Report (JSON)
→ Stage 1 Eligibility Filter
→ Stage 2 Scoring
→ Stage 3 Rerank
→ Streamlit UI + Evaluation Harness

3. Technical Constraints
Environment
  • Python 3.11+
  • Windows (VS Code)
  • Local filesystem only (no database for MVP)
  • Virtual environment located at .venv
Required Libraries
  • pandas
  • pyarrow
  • scikit-learn
  • streamlit
  • pytest
  • requests
No heavy frameworks.
No cloud infra.
No external databases.

4. Artifact Rules (Critical)
This project must produce deterministic, versionable artifacts.
Snapshot
Each ingest run produces:

data/processed/scholarships_snapshot_YYYYMMDD.parquet
Requirements:
  • Stable schema
  • Deterministic ordering
  • No random row ordering
  • Use engine="pyarrow"
Delta Report
Each run must also produce:

data/processed/changes_YYYYMMDD.json
Structure:

{
  "added": [...],
  "removed": [...],
  "changed": [
    {
      "scholarship_id": "...",
      "fields_changed": {
        "deadline": {"old": "...", "new": "..."},
        "amount_max": {"old": 5000, "new": 6000}
      }
    }
  ]
}
Delta must compare against most recent prior snapshot.

5. Normalized Scholarship Schema
All records must conform to a strict normalized schema.
Required fields:
  • scholarship_id (string, canonical, stable hash)
  • source (string)
  • source_id (nullable string)
  • source_url (string)
  • title (string)
  • sponsor (nullable string)
  • description (nullable string)
  • eligibility_text (nullable string)
  • deadline (date, nullable)
  • amount_min (float, nullable)
  • amount_max (float, nullable)
  • is_recurring (bool, nullable)
  • states_allowed (list[string], nullable)
  • majors_allowed (list[string], nullable)
  • min_gpa (float, nullable)
  • citizenship (nullable string)
  • education_level (nullable string)
  • essay_required (bool, nullable)
  • essay_prompt (nullable string)
  • keywords (list[string], nullable)
  • first_seen_at (timestamp)
  • last_seen_at (timestamp)
Schema must remain stable unless explicitly migrated.

6. Canonical ID Rules
scholarship_id must be:
  • Deterministic
  • Stable across runs
  • SHA1 hash of normalized:
    ○ lowercased title
    ○ sponsor
    ○ amount range
    ○ deadline
    ○ source domain
Changing any of those fields must change the ID.
Tests must validate:
  • identical input → identical ID
  • changed title → different ID
  • changed deadline → different ID

7. Multi-Stage Ranking Pipeline
Stage 0 – Candidate Universe
Input:
  • Latest snapshot parquet
Output:
  • Full candidate dataframe
No ranking happens here.

Stage 1 – Hard Eligibility Filter
Input:
  • Scholarship dataframe
  • Student profile dict
Output:
  • eligible_df
  • ineligible_df (with reason codes)
Reason codes must include:
  • GPA_BELOW_MIN
  • DEADLINE_PASSED
  • STATE_NOT_ALLOWED
  • MAJOR_NOT_ALLOWED
  • EDUCATION_LEVEL_MISMATCH
  • CITIZENSHIP_MISMATCH
Each rejected scholarship must list all applicable reasons.
No silent filtering.

Stage 2 – Baseline Scoring
Score components:
  1. TF-IDF similarity between:
    ○ student profile summary text
    ○ title + description + eligibility_text + essay_prompt
  2. Amount utility (normalized)
  3. Essay effort penalty (if essay_required)
  4. Keyword overlap boost
Composite score must be deterministic.

Stage 3 – Rerank
Apply:
  • Deadline urgency boost
  • Expected value proxy:

EV = amount_max * (1 / effort_penalty)
Final ranked dataframe must include:
  • final_score
  • component breakdown columns

8. Explainability Requirements
For each ranked scholarship, UI must show:
  • Why matched (top contributing signals)
  • If excluded, show reason codes
  • Deadline
  • Amount
  • Source link
No opaque ranking.

9. Evaluation Harness
Must include:
scripts/evaluate_golden_students.py
Golden students defined in:
src/eval/golden_students.py
Metrics:
  • Eligibility precision
  • NDCG@10
  • Coverage
  • Distribution of award amounts recommended
Evaluation must run on saved snapshot (not live ingest).

10. Ingestion Rules
Each source must:
  • Respect polite rate limiting
  • Cache raw responses in:
data/raw/<source>/<timestamp>.<ext>
  • Fail independently (one source failure must not kill pipeline)
  • Use retry with exponential backoff
  • Never crash silently
Use pathlib.Path everywhere.
No hardcoded slashes.

11. Windows-Specific Requirements
  • Always use pathlib.Path
  • Write parquet to temp file, then atomic rename
  • No open file handles left behind
  • Tests run via:

.\.venv\Scripts\Activate.ps1
pytest -q

  • Virtual environment must exist at .venv.
  • Codex must not run tests outside this environment.
  • If .venv does not exist, create it before running tests.

12. Code Quality Standards
  • Type hints required
  • No global state
  • Clear logging
  • No random seeds unless fixed
  • No non-deterministic ordering
All tasks must end by running tests.

13. Definition of Done (MVP)
MVP is complete when:
  • Repo skeleton exists
  • Snapshot + delta work
  • One live ingestion connector works
  • Stage 1 eligibility filter returns reason codes
  • Stage 2 ranking works
  • Stage 3 rerank works
  • Streamlit app displays ranked scholarships
  • Evaluation harness runs successfully
  • All tests pass

14. Explicit Non-Goals (MVP)
  • No authentication
  • No database
  • No cloud deployment
  • No LLM-based essay scoring (future phase)
  • No win-probability model (future phase)

15. Implementation Order (Strict)
Codex must implement in this order:
  1. Repo skeleton + pyproject
  2. Schema + canonical_id + tests
  3. Snapshot + delta logic
  4. Ingestion framework
  5. One working source connector
  6. Stage 1 eligibility filter
  7. Stage 2 scoring
  8. Stage 3 rerank
  9. Streamlit UI
  10. Evaluation harness
No skipping ahead.

16. Behavior Instructions for Codex
  • Always run tests before finishing a task.
  • Never remove fields from schema without migration note.
  • Never introduce randomness.
  • Do not refactor unrelated modules.
  • Keep functions small and testable.
