-- Roles Plan Task A.1 (owner decision D3): essays are private until she shares
-- them.  The default makes every existing draft private on upgrade.  Alone in
-- its file for the same reason as 0007.

ALTER TABLE essays ADD COLUMN shared_with_parents INTEGER NOT NULL DEFAULT 0;
