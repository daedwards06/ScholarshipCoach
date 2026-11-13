-- Roles Plan Task A.11: what she applied to, got into, and does.
--
-- A program from the catalog keeps its result here (applied, accepted, ...),
-- one row per application cycle, so a program she was not accepted to can be
-- applied to again next year as a new row.  Clubs, awards and jobs she enters
-- by hand live here too (catalog_id NULL), so Activities & Honors is one list.
-- Named experiences, not activities: activity_days (0009) already means app
-- logins.
--
-- Every statement is IF NOT EXISTS: executescript commits per statement, so a
-- file interrupted half-way must be safe to run again.

CREATE TABLE IF NOT EXISTS experiences (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id     TEXT NOT NULL REFERENCES students (student_id) ON DELETE CASCADE,
    catalog_id     TEXT,
    title          TEXT NOT NULL,
    organization   TEXT NOT NULL DEFAULT '',
    kind           TEXT NOT NULL DEFAULT 'other'
                   CHECK (kind IN ('program', 'club', 'award', 'job', 'volunteer', 'other')),
    status         TEXT NOT NULL DEFAULT 'active'
                   CHECK (status IN ('applied', 'accepted', 'waitlisted', 'not_accepted',
                                     'declined', 'completed', 'active')),
    starts_on      TEXT,
    ends_on        TEXT,
    term           TEXT NOT NULL DEFAULT '',
    hours_per_week REAL,
    notes          TEXT NOT NULL DEFAULT '',
    created_by     TEXT NOT NULL DEFAULT '',
    created_role   TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_experiences_catalog_term
    ON experiences (student_id, catalog_id, term) WHERE catalog_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_experiences_student ON experiences (student_id);
