-- Family product tables: students and everything that hangs off a student.
-- Every child row cascades on delete, so removing a student removes their
-- applications, essays, recommenders, outcomes and colleges with them.

CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    name       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS applications (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL REFERENCES students (student_id) ON DELETE CASCADE,
    catalog_id TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'saved',
    notes      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (student_id, catalog_id)
);

CREATE INDEX IF NOT EXISTS idx_applications_student ON applications (student_id);

CREATE TABLE IF NOT EXISTS checklist_items (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id INTEGER NOT NULL REFERENCES applications (id) ON DELETE CASCADE,
    label          TEXT NOT NULL,
    done           INTEGER NOT NULL DEFAULT 0,
    due_on         TEXT,
    position       INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_checklist_items_application
    ON checklist_items (application_id);

CREATE TABLE IF NOT EXISTS essays (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL REFERENCES students (student_id) ON DELETE CASCADE,
    title      TEXT NOT NULL,
    body       TEXT NOT NULL DEFAULT '',
    word_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_essays_student ON essays (student_id);

-- An award prompt reuses an essay: the prompt text belongs to the pairing,
-- not to the essay, because the same draft answers several prompts.
CREATE TABLE IF NOT EXISTS essay_links (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    essay_id       INTEGER NOT NULL REFERENCES essays (id) ON DELETE CASCADE,
    application_id INTEGER NOT NULL REFERENCES applications (id) ON DELETE CASCADE,
    prompt         TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    UNIQUE (essay_id, application_id)
);

CREATE TABLE IF NOT EXISTS recommenders (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL REFERENCES students (student_id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    role       TEXT NOT NULL DEFAULT '',
    email      TEXT NOT NULL DEFAULT '',
    notes      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_recommenders_student ON recommenders (student_id);

CREATE TABLE IF NOT EXISTS recommendation_requests (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    recommender_id INTEGER NOT NULL REFERENCES recommenders (id) ON DELETE CASCADE,
    application_id INTEGER NOT NULL REFERENCES applications (id) ON DELETE CASCADE,
    status         TEXT NOT NULL DEFAULT 'planned',
    requested_on   TEXT,
    due_on         TEXT,
    submitted_on   TEXT,
    notes          TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    UNIQUE (recommender_id, application_id)
);

CREATE TABLE IF NOT EXISTS outcomes (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id INTEGER NOT NULL UNIQUE
                   REFERENCES applications (id) ON DELETE CASCADE,
    result         TEXT NOT NULL DEFAULT 'pending',
    amount_awarded REAL,
    paid_to        TEXT NOT NULL DEFAULT '',
    renewal_terms  TEXT NOT NULL DEFAULT '',
    decided_on     TEXT,
    notes          TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS colleges (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id         TEXT NOT NULL REFERENCES students (student_id) ON DELETE CASCADE,
    name               TEXT NOT NULL,
    status             TEXT NOT NULL DEFAULT 'considering',
    cost_of_attendance REAL,
    aid_offered        REAL,
    notes              TEXT NOT NULL DEFAULT '',
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_colleges_student ON colleges (student_id);

-- App-wide switches (parent mode, operator tools), not per student.
CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
