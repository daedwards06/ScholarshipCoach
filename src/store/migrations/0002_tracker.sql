-- Task 3.3 (application tracker): the columns My Applications and This Week
-- read without a snapshot loaded.
--
-- title, source_url and deadline are copies of catalog fields on purpose.  A
-- saved award has to stay readable when the pipeline has not been run this
-- session, and the deadline the student planned against is the one they saw,
-- not whatever the catalog says after the next re-verification pass.

ALTER TABLE applications ADD COLUMN title TEXT NOT NULL DEFAULT '';
ALTER TABLE applications ADD COLUMN source_url TEXT NOT NULL DEFAULT '';
ALTER TABLE applications ADD COLUMN deadline TEXT;
ALTER TABLE applications ADD COLUMN submitted_on TEXT;
