-- Task 3.4 (essay bank and recommenders).
--
-- A theme tag is what makes the bank reusable: the student writes a handful of
-- themed drafts and answers most prompts out of them, so the tag is how they
-- find the right draft for a new prompt.
--
-- Drafts are rows, not a diff log.  essays holds the current text and
-- essay_versions holds every version as it was saved, so a student who cuts a
-- paragraph to hit a 500-word limit can read the long one back.
--
-- asked_on / received_on replace requested_on / submitted_on: a letter is
-- received, and "submitted" already means the student sent the application.

ALTER TABLE essays ADD COLUMN theme TEXT NOT NULL DEFAULT 'other';

CREATE TABLE IF NOT EXISTS essay_versions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    essay_id   INTEGER NOT NULL REFERENCES essays (id) ON DELETE CASCADE,
    title      TEXT NOT NULL,
    body       TEXT NOT NULL DEFAULT '',
    word_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_essay_versions_essay ON essay_versions (essay_id);

ALTER TABLE recommendation_requests RENAME COLUMN requested_on TO asked_on;
ALTER TABLE recommendation_requests RENAME COLUMN submitted_on TO received_on;
