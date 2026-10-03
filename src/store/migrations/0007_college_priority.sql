-- Roles Plan Task A.1: 1 marks her first target school, which My Path reads.
-- NULL means unranked.  An ALTER cannot be guarded with IF NOT EXISTS, so it
-- sits alone in its file: if it fails, nothing in the file has applied.

ALTER TABLE colleges ADD COLUMN priority INTEGER;
