-- Roles Plan Task A.1 (owner decision D6): days opened, per person.
--
-- One row per person per day and a page counter, deliberately no page-level
-- trail: the go/no-go gate needs "did she open it on her own", and she is told
-- this is counted.  login_or_role is the Tailscale login when known, else the
-- mode ("student" / "parent") under the PIN fallback.

CREATE TABLE IF NOT EXISTS activity_days (
    login_or_role TEXT NOT NULL,
    day           TEXT NOT NULL,
    pages_opened  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (login_or_role, day)
);
