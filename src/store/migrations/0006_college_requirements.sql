-- Roles Plan Task A.1 (requirements behind My Path).
--
-- A target school's requirements are rows under its colleges row, entered by
-- the parents on the server and never committed: the school is personal data
-- here, so it lives only in coach.db.  due_by holds either a grade ("11") or
-- an ISO date, because "take precalculus by junior year" has no date until the
-- course schedule exists.  verified_on records when someone last checked the
-- school's own page, since admissions rules change every cycle.

CREATE TABLE IF NOT EXISTS college_requirements (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    college_id  INTEGER NOT NULL REFERENCES colleges (id) ON DELETE CASCADE,
    category    TEXT NOT NULL DEFAULT 'other'
                CHECK (category IN ('course', 'gpa', 'test', 'application',
                                    'scholarship', 'program', 'other')),
    label       TEXT NOT NULL,
    target      TEXT NOT NULL DEFAULT '',
    due_by      TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'not_started'
                CHECK (status IN ('not_started', 'in_progress', 'met', 'not_needed')),
    notes       TEXT NOT NULL DEFAULT '',
    source_url  TEXT NOT NULL DEFAULT '',
    verified_on TEXT,
    position    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_college_requirements_college
    ON college_requirements (college_id);
