-- Roles Plan Task A.1 (shared to-dos, replacing This Week).
--
-- One list both roles write to.  assignee says who the row is for and
-- created_by / created_role say who added it, because "from Mom" is half of
-- what makes a parent's suggestion read as a suggestion rather than a chore.
-- created_by is the Tailscale login when one is known, else empty, and
-- created_role is the mode it was added from, so authorship survives the PIN
-- fallback.
--
-- source_kind / source_ref point a row back at what generated it (a checklist
-- step, a requirement, a program), so ticking it can tick the source too.
--
-- Every statement is IF NOT EXISTS: executescript commits per statement, so a
-- file interrupted half-way must be safe to run again.

CREATE TABLE IF NOT EXISTS todos (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id   TEXT NOT NULL REFERENCES students (student_id) ON DELETE CASCADE,
    title        TEXT NOT NULL,
    notes        TEXT NOT NULL DEFAULT '',
    due_on       TEXT,
    assignee     TEXT NOT NULL DEFAULT 'student'
                 CHECK (assignee IN ('student', 'parent', 'family')),
    created_by   TEXT NOT NULL DEFAULT '',
    created_role TEXT NOT NULL DEFAULT '',
    done_on      TEXT,
    done_by      TEXT NOT NULL DEFAULT '',
    source_kind  TEXT NOT NULL DEFAULT 'manual'
                 CHECK (source_kind IN ('manual', 'application', 'checklist', 'letter',
                                        'requirement', 'opportunity', 'milestone')),
    source_ref   TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_todos_student_done ON todos (student_id, done_on);
