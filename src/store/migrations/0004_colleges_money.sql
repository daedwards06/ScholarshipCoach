-- Task 3.7 (colleges and money view).
--
-- Outside scholarships are the small lever; the college list is the big one.
-- cost_of_attendance stays the sticker price, but the number a family can act
-- on is the school's own net price calculator output, which no public feed
-- carries -- so net_price_estimate is typed in by hand and wins over
-- sticker minus aid whenever it is set.
--
-- outside_award_policy is the column that makes this view worth building: a
-- school that displaces its own grant dollar for dollar turns a won
-- scholarship into no money at all, and the family needs that written down
-- next to the school before the student spends a weekend on the essay.

ALTER TABLE colleges ADD COLUMN in_state INTEGER NOT NULL DEFAULT 0;
ALTER TABLE colleges ADD COLUMN net_price_estimate REAL;
ALTER TABLE colleges ADD COLUMN merit_aid_notes TEXT NOT NULL DEFAULT '';
ALTER TABLE colleges ADD COLUMN outside_award_policy TEXT NOT NULL DEFAULT '';
ALTER TABLE colleges ADD COLUMN deadline_type TEXT NOT NULL DEFAULT '';
