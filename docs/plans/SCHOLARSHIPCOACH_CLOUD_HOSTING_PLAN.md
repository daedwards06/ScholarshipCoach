# ScholarshipCoach Cloud Hosting Plan — Off the Home PC, Still Off the Public Internet

> Generated: 2026-09-26 | Scope from a hosting discussion after the Catalog Integrity Plan
> Supersedes the "Family hosting" section of `docs/operations.md` (home PC on the LAN)
> Executor: Claude via Claude Code, with owner-dependent provisioning tasks called out as such
> Est. effort: 3 phases, 8 tasks | Running cost: one small cloud server (~$4.54/month, billed 12 months upfront) + backup storage (cents)

---

## Why this plan exists

`docs/operations.md` settles hosting as "the home PC, bound to the LAN, behind the family PIN".
Walking through that on 2026-09-26 turned up three problems:

| Finding | Evidence |
|---|---|
| LAN access has never been tested | The app has only been launched from a terminal on the dev PC and opened on that PC |
| The home PC blocks it as configured | Wi-Fi profile "Korriban" is `Public`; two enabled inbound **Block** rules exist for `anaconda3\python.exe` on the Public profile, and `.venv` runs through that interpreter (`pyvenv.cfg`); NordVPN is connected (NordLynx) |
| The design needs a PC that is always on | The app, the weekly backup, and the monthly re-verification are all Windows scheduled tasks on the home PC; the owner does not want the PC on 24/7 |

What the app needs from a host, measured the same day:

| Need | Measured |
|---|---|
| Memory | `sentence-transformers` (torch + `all-MiniLM-L6-v2`), so budget 1–2 GB resident |
| Durable disk | `data/private/coach.db` is SQLite, written by the app; ~125 KB today |
| Other artifacts | `data/processed/` ~3.5 MB of snapshots, plus git-ignored `embeddings/`, `win_model/`, `llm_extractions/` |
| Personal data | Schema holds the student's name, essays and every version of them, recommenders' names and emails, award amounts, college costs and aid offers |

**Decisions taken:**
- **A rented Linux server, reachable only over Tailscale.** Always on, no hardware, no code
  changes. No public port, no public DNS record, no Tailscale Funnel. The "never public" rule in
  `docs/operations.md` still holds; only the machine changes.
- **Streamlit binds to `127.0.0.1` on the server.** `tailscale serve` is the only way in, and it
  gives the family an HTTPS name (`https://coach.<tailnet>.ts.net`) instead of an IP and a port.
- **Nightly encrypted backups off the server**, and a restore drill before the family relies on it.
- **The server is where catalog edits happen once it is live.** Add Award and inbox confirms on
  the server commit to a `server` branch and reach `main` through a PR, so CI still validates
  every catalog change.
- **The home PC drops out.** Its three scheduled tasks are unregistered at cutover. The Windows
  network/firewall fixes from the 2026-09-26 discussion are not needed and are not made.

**Considered and not taken:**
- *A small always-on box at home (mini PC / Raspberry Pi 5)*: same result, data stays in the
  house, but a one-time hardware purchase and one more device to maintain. Still a fine fallback;
  every file this plan writes works on it unchanged.
- *Streamlit Community Cloud*: free, but its disk is wiped on restart, so `coach.db` would have to
  move to a hosted database (a code change and a third-party holder of the essays), and torch
  risks its memory limit.
- *Cloudflare Tunnel + Access*: a public hostname behind an email login. Lets people without an
  app in, but reverses "never public" and still needs an always-on host.

**Relationship to other plans:**
- Replaces the hosting half of Family Product Plan Task 4.1; its PIN and phone-width work stand.
- **Design Refresh Plan** (updated 2026-09-27): runs in parallel; nothing here touches `app/` UI.
  Its Task 1.1 already put `[server] enableStaticServing = true` in `.streamlit/config.toml` and
  self-hosted fonts in `app/static/fonts/`. The service must therefore run from the repo root, so
  that config and static directory are picked up (Task 1.1 checks this).
- **Roles & Product Plan** (`docs/plans/SCHOLARSHIPCOACH_ROLES_PRODUCT_PLAN.md`, to be written by
  the roles/UX review; leading model: Student mode as her path to college, Parent mode as the
  money view). Two dependencies run through this plan:
  - *Schema changes on a live database.* `src/store/db.py` applies numbered migrations from
    `src/store/migrations/` automatically on connect, so the first request after an update
    migrates `coach.db` on the server. Task 1.2 makes `update.sh` take a backup before it pulls.
  - *Identity.* That plan decides whether parent access stays a family PIN or becomes per-person
    identity, for example the signed-in Tailscale user that `tailscale serve` can pass to the app.
    This must be decided before Task 3.1 onboards phones, and it constrains D3.
  - *Timing.* Task 3.1 onboarding waits for that plan's light version (its Phase A) to ship, so the
    family's first weeks on their phones are the real-use test that plan's go/no-go gate needs.

---

## Owner decisions (defaults used unless changed before Task 2.1)

| # | Decision | Default | Alternatives |
|---|---|---|---|
| D1 | Provider and size | **OVHcloud US VPS-1** (2 vCore, 4 GB RAM, 40 GB NVMe, IPv4 included), US East (Vint Hill, VA), 12-month upfront | Hetzner CX23 in Germany (~$7.09/mo, no US plan); DigitalOcean / Linode 4 GB ($24/mo); a home mini PC (~$250 upfront) |
| D2 | OS | Ubuntu 24.04 LTS (ships Python 3.12, matching CI) | Debian 12 + deadsnakes Python |
| D3 | Tailscale accounts | One family tailnet with **one account per person**; check the free plan's current user limit against the number of family members | Everyone on one shared login (rules out per-person identity if the Roles & Product Plan chooses it) |
| D4 | Backup destination | `restic` to Backblaze B2 (encrypted client-side; first 10 GB free) | OVHcloud Object Storage; nightly pull to a home machine |
| D5 | Catalog edits | Server commits to a `server` branch, owner merges by PR | Catalog edits only on the dev PC; server catalog read-only |

Check current prices and plan names at signup; the figures here are approximate.

*(2026-09-27: D1 moved from Hetzner to OVHcloud. Hetzner's 15 June 2026 price adjustment
took its US 4 GB plan (CPX21) from $13.99 to $37.49/mo plus IPv4, and its cheap CX line is
EU-only. VPS-1 at $4.54/mo requires the 12-month upfront term; check the renewal price at
checkout. Whatever the provider, the server needs a public IPv4, because GitHub has no IPv6
and `bootstrap.sh` clones from it. OVH's included daily VPS backup does not replace restic:
it is kept by the same provider and holds only the last 24 hours.)*

---

## Design principles for this plan

1. **Nothing listens on a public interface.** Streamlit on `127.0.0.1`; SSH through Tailscale
   SSH; the cloud firewall allows only Tailscale's UDP port inbound. A port scan of the public IP
   finds nothing.
2. **The server is rebuildable from the repo plus one backup.** Every server file lives in
   `deploy/`, and a fresh server comes up from `deploy/bootstrap.sh` + a restic restore.
3. **A backup is not real until a restore has been done.** Task 2.3 restores onto the server and
   opens the app before the family is told the address.
4. **Owner provisions, Claude writes.** Claude writes every script, unit and runbook; the owner
   creates accounts, pays, and runs the commands that need their credentials.
5. **Green gate every task** that changes the repo: the four CI commands.

---

# Phase 1 — Server Configuration in the Repo

Everything in this phase is written and tested on the dev PC. No server is needed yet.

## Task 1.1: Service, Bootstrap and Update Scripts

**Why:** A server configured by hand from a chat log cannot be rebuilt. The service definition,
the install steps, and the update routine belong in the repo next to the code they run.

**Preflight Files:**
- `docs/operations.md` ("Family hosting", "Start on boot" sections)
- `.streamlit/config.toml` (`[server]` section; `headless = true`, address left unset on purpose)
- `.github/workflows/ci.yml` (the CPU-only torch install step)
- `pyproject.toml` (dependencies)
- `.gitignore` (what the server must receive by copy rather than by clone)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
```

**Checklist:**
- [x] `deploy/scholarshipcoach.service`: systemd unit running
      `.venv/bin/streamlit run app/main.py --server.address 127.0.0.1 --server.port 8501` as a
      non-root `coach` user from `/srv/scholarshipcoach`; `Restart=always`
- [x] `deploy/bootstrap.sh` (idempotent, run once as root on a fresh Ubuntu 24.04): packages
      (`python3.12-venv`, `git`, `sqlite3`, `restic`, `ufw`, `unattended-upgrades`), a 2 GB
      swap file, the `coach` user, the clone, the venv with CPU-only torch installed before
      `pip install -e .` (same order as CI), `ufw` default-deny inbound with `tailscale0` and
      UDP 41641 allowed, the unit installed and enabled
- [x] `deploy/update.sh` (run as `coach`): refuses to run with uncommitted changes outside
      `data/catalog/records/`, runs the catalog sync from Task 1.3 first, `git pull` from
      `origin/main`, `pip install -e .`, restarts the service, and fails loudly unless
      `curl -fsS http://127.0.0.1:8501/_stcore/health` answers `ok` within 60 seconds
      *(2026-09-27: calls `deploy/catalog_sync.sh` when present; until Task 1.3 adds it,
      stops if `records/` has edits rather than pulling over them)*
- [x] `tests/test_deploy_config.py`: the unit's `ExecStart` binds `--server.address 127.0.0.1`
      and nothing in `deploy/` contains `0.0.0.0` or `tailscale funnel` — a guard against a
      future edit exposing the app on the server's public interface
- [x] The unit sets `WorkingDirectory=/srv/scholarshipcoach`, so `.streamlit/config.toml` (theme,
      `enableStaticServing`) and `app/static/` are found; `update.sh`'s health check also fetches
      one self-hosted font file under `/app/static/fonts/` and requires HTTP 200
- [x] The `[server]` comment in `.streamlit/config.toml` that mentions passing
      `--server.address 0.0.0.0` is rewritten to describe the server setup (127.0.0.1 behind
      `tailscale serve`), since 0.0.0.0 is now the thing the deploy test forbids
- [x] All four CI commands green

---

## Task 1.2: Nightly Encrypted Backup and Restore Scripts

**Why:** On the home PC the backup was a weekly zip to a second drive. On a rented server the
disk belongs to someone else and can disappear with a billing mistake, so the backup has to
leave the machine, be encrypted before it does, and be restorable by a script.

**Preflight Files:**
- `scripts/backup_private.ps1` (current backup, retention of 8)
- `docs/operations.md` ("Back up `data/private/`" and "Restore" sections)
- `src/store/db.py` (`PRIVATE_DIR`, how `coach.db` is opened, `apply_migrations` on connect)
- `src/store/migrations/` (numbered SQL files applied automatically)
- `deploy/update.sh` (from Task 1.1)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
```

**Checklist:**
- [x] `deploy/backup.sh`: takes a consistent copy with `sqlite3 coach.db ".backup ..."` (safe
      while the app is running, unlike a file copy), then `restic backup` of that copy,
      `data/private/students/`, `data/catalog/inbox/`, and `.streamlit/secrets.toml`; retention
      `--keep-daily 14 --keep-weekly 8 --keep-monthly 12`; reads repository and password from
      `/etc/scholarshipcoach/restic.env` (mode 600, never in git)
- [x] `deploy/restore.sh <snapshot-id|latest>`: stops the service, moves `data/private/` aside
      to `private_before_restore_<stamp>`, restores, checks `coach.db` opens and prints row
      counts per table, starts the service
- [x] `deploy/scholarshipcoach-backup.service` + `.timer`: nightly at 03:30 server time,
      `Persistent=true`
- [x] `deploy/update.sh` runs `deploy/backup.sh` before `git pull` and stops if the backup
      fails. The pull can bring new files under `src/store/migrations/`, and `src/store/db.py`
      applies them on the first connection after the restart, so an update may change `coach.db`
      irreversibly. The Roles & Product Plan is expected to add migrations. The update log names
      any migration files the pull added
- [x] `tests/test_deploy_config.py` extended: `restic.env` is referenced by path only and is
      not present in the repo; `update.sh` calls `backup.sh` before `git pull`
- [x] All four CI commands green
      *(2026-09-27: `restore.sh` runs as root and fetches the snapshot before stopping the
      service; `inbox/` and `secrets.toml` are restored only when the server has none.
      `bootstrap.sh` now creates `/etc/scholarshipcoach/` and enables the backup timer)*

---

## Task 1.3: Scheduled Re-verification and Catalog Sync

**Why:** The monthly re-verification is a Windows scheduled task today, and catalog edits made
in the app write to git-tracked files. On the server both need a schedule, and the edits need a
path back to `main` that still passes CI.

**Preflight Files:**
- `scripts/verify_catalog.py` (flags, exit codes)
- `docs/operations.md` ("Monthly catalog re-verification", "Schedule it monthly")
- `src/catalog/inbox.py` (`confirm` writes into `data/catalog/records/`)
- `.gitignore` (inbox and verify reports are local state)

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
```

**Checklist:**
- [x] `deploy/scholarshipcoach-verify.service` + `.timer`: `scripts/verify_catalog.py` on the
      first Sunday of each month, `Persistent=true`
- [x] `deploy/catalog_sync.sh`: if `data/catalog/records/` has changes, runs
      `scripts/validate_catalog.py`, commits only that directory to the `server` branch with a
      message listing the changed `catalog_id`s, and pushes; stops and reports rather than
      committing when validation fails or anything outside `records/` is dirty
- [x] `deploy/scholarshipcoach-catalog-sync.service` + `.timer`: nightly, after the backup
- [x] Push credential documented as a repo-scoped **deploy key with write access**, held only
      by the `coach` user; no personal token on the server
- [x] All four CI commands green
      *(2026-09-27: the server checkout lives on a local `server` branch, so `update.sh` now
      merges `origin/main` into it instead of `--ff-only`. `catalog_sync.sh` pushes only when
      the server holds record changes main lacks, so a squash-merged, auto-deleted branch is
      not recreated. `bootstrap.sh` creates the branch, the deploy key (printed for the owner),
      an SSH push URL, and enables the verify and catalog-sync timers)*

---

# Phase 2 — Stand It Up *(owner-dependent)*

## Task 2.1: Provision and Lock Down the Server *(owner-dependent)*

**Why:** The accounts, the payment method, and the first root login belong to the owner.
Lockdown happens before any student data reaches the machine.

**Preflight Files:**
- This plan's Owner decisions table
- `deploy/bootstrap.sh`

**Validation Commands** (from the dev PC, NordVPN disconnected if it interferes with Tailscale):
```powershell
tailscale status                                   # server listed, online
ssh coach@<server-tailscale-name> "systemctl is-enabled scholarshipcoach"
Test-NetConnection <server-public-ip> -Port 22     # TcpTestSucceeded : False (after port 22 closed)
Test-NetConnection <server-public-ip> -Port 8501   # TcpTestSucceeded : False
```

**Checklist:**
- [x] Owner: provider account (D1), server created with Ubuntu 24.04 (D2) and the owner's SSH key
- [x] Owner: Tailscale on the server (`tailscale up --ssh`), on the dev PC, and in the family
      tailnet (D3)
- [x] Owner: `deploy/bootstrap.sh` run as root; Claude reviews its output
- [x] Owner: once Tailscale SSH is confirmed working, OVH Edge Network Firewall enabled on the
      VPS's IPv4 with no port 22 rule; password login disabled. The Edge firewall is stateless,
      so it needs reply rules for the server's own outbound traffic: 0 TCP *established*,
      1 UDP source port 53 (DNS), 2 UDP source port 123 (NTP), 3 UDP destination port 41641
      (Tailscale), 4 ICMP, 19 refuse IPv4. It does not cover IPv6; `ufw` from `bootstrap.sh`
      does, and is the rule that matters on both
- [x] Both `Test-NetConnection` checks against the public IP fail
      *(2026-09-27: ports 22 and 8501 on the public IPv4 both `TcpTestSucceeded : False`;
      Tailscale connects direct on 41641; ICMP answers by design)*
      *(2026-09-27: OVH VPS-1, Ubuntu 24.04.5 LTS, Tailscale name `coach`. `sshd -T` shows
      `passwordauthentication no`; the `ubuntu` password is locked. Bootstrap log clean, `ufw`
      default-deny inbound with only `tailscale0` and 41641/udp allowed on v4 and v6, 2 GB swap,
      service active, three timers scheduled. The backup timer fails nightly until Task 2.3
      creates `restic.env`; the deploy key is not yet on GitHub)*

---

## Task 2.2: Move the Data and Serve It *(owner-dependent)*

**Why:** The app is useless on the server without the private directory, the PIN, and the
git-ignored artifacts it ranks with.

**Preflight Files:**
- `.gitignore` (git-ignored artifacts under `data/processed/`)
- `app/state.py` (`PROCESSED_DIR`, which snapshot is loaded)
- `.streamlit/secrets.toml` (exists on the dev PC; contents not read by Claude)

**Validation Commands:**
```bash
# on the server, as coach
systemctl is-active scholarshipcoach                     # active
ss -tlnp | grep 8501                                     # 127.0.0.1:8501 only
curl -fsS http://127.0.0.1:8501/_stcore/health           # ok
tailscale serve status                                   # https://coach.<tailnet>.ts.net -> 127.0.0.1:8501
```

**Checklist:**
- [ ] Stop the app on the dev PC first, so `coach.db` is not written during the copy
- [ ] Copy over Tailscale (`scp`): `data/private/`, `.streamlit/secrets.toml` (then `chmod 600`),
      the latest `data/processed/scholarships_snapshot_*.parquet`, `data/processed/embeddings/`,
      `data/processed/win_model/`, `data/processed/llm_extractions/`
- [ ] Server hostname set to `coach` in the Tailscale admin console; `tailscale serve --bg 8501`
- [ ] From the dev PC browser: the HTTPS name loads, ranking returns results, Parent mode asks
      for the PIN, and an edit in My Applications survives a `systemctl restart`
- [ ] "Rebuild snapshot" run once on the server; the result carries every record forward
      (fetches from a datacenter IP may come back `blocked` more often than from home; that is
      the Catalog Integrity Plan's "check by hand" path, not a failure)

---

## Task 2.3: Backups Live, Restore Drilled *(owner-dependent)*

**Why:** Design principle 3. The family does not get the address until a restore has worked.

**Preflight Files:**
- `deploy/backup.sh`, `deploy/restore.sh`

**Validation Commands:**
```bash
sudo systemctl start scholarshipcoach-backup.service && journalctl -u scholarshipcoach-backup -n 20
restic snapshots                                         # at least one snapshot
sudo deploy/restore.sh latest                            # prints row counts, service active again
systemctl list-timers 'scholarshipcoach-*'               # backup, verify, catalog-sync scheduled
```

**Checklist:**
- [ ] Owner: B2 bucket (D4), application key scoped to that bucket, `/etc/scholarshipcoach/restic.env`
      written, `restic init` run
- [ ] Owner: restic password stored somewhere off the server (password manager); without it the
      backups cannot be read
- [ ] One manual backup, then a full restore with row counts matching the pre-restore counts
- [ ] Owner: deploy key added to the GitHub repo; one `catalog_sync.sh` run pushes (or reports
      nothing to push)
- [ ] All three timers enabled and listed

---

# Phase 3 — Family On, Home PC Off

## Task 3.1: Family Onboarding *(owner-dependent)*

**Why:** The goal is a home-screen icon that works from school, not a server that works from
the dev PC.

**Prerequisites (2026-09-27):** the Roles & Product Plan has (1) decided how parent access works
(family PIN or per-person identity) and (2) shipped its light version (Phase A). Onboarding
starts the month of real use that plan's go/no-go gate depends on. Onboarding earlier just
shows the family the version that is about to change.

**Preflight Files:**
- `docs/operations.md` ("Set the family PIN first", "Phone width")
- The Roles & Product Plan's identity / access decision

**Validation Commands:** none automated; the checklist is the test.

**Checklist:**
- [ ] Tailscale installed and signed in on each family phone
- [ ] Each phone opens `https://coach.<tailnet>.ts.net` **on cellular data, Wi-Fi off**, the test
      that proves it is not reaching anything at home
- [ ] "Add to Home Screen" on each phone
- [ ] Access works as the Roles & Product Plan decided: with the family PIN, Student mode opens
      without it and Parent mode asks for it; with per-person identity, each family member's own
      Tailscale login lands in the right mode and the student cannot reach parent-only pages
- [ ] Any phone that also runs a VPN app (NordVPN): confirm which one wins, since phones
      generally allow one active VPN at a time; note the outcome in `docs/operations.md`

---

## Task 3.2: Docs, Bookkeeping, and Retiring the Home PC

**Why:** `docs/operations.md` tells the reader to run the app on the home PC. Left as is, the
next session follows it.

**Preflight Files:**
- `docs/operations.md` (whole "Family hosting" section, re-verification scheduling)
- `README.md` (any hosting or scheduling mentions)
- `CLAUDE.md` (plan table)
- `scripts/backup_private.ps1`

**Validation Commands:**
```powershell
python -m pytest tests/ -q
ruff check src/ scripts/ app/ tests/
python -m mypy src/
python -m mypy app/
python scripts/validate_catalog.py
Get-ScheduledTask -TaskName "ScholarshipCoach*" -ErrorAction SilentlyContinue   # nothing listed
```

**Checklist:**
- [ ] `docs/operations.md`: "Family hosting" rewritten around the server — the reachability
      table (Tailscale only; public never), update routine (`deploy/update.sh`), backup and
      restore via restic, the catalog `server` branch flow; the Windows Task Scheduler sections
      moved to a short "Running locally for development" note
- [ ] `scripts/backup_private.ps1` removed, or kept and labeled as the local-dev backup, with
      the docs matching
- [ ] `README.md` hosting mentions updated
- [ ] `CLAUDE.md` plan table gains this plan
- [ ] Owner: the three Windows scheduled tasks unregistered on the home PC
- [ ] All four CI commands green

---

## Execution Order

```
Phase 1  1.1 service + bootstrap + update → 1.2 backup + restore → 1.3 verify timer + catalog sync
Phase 2  2.1 provision + lock down → 2.2 move data + serve → 2.3 backups live + restore drill
Phase 3  3.1 family onboarding → 3.2 docs + retire home PC
```

Phase 1 needs no server and can land before the owner signs up anywhere. Task 2.1 before 2.2 so
no student data reaches a machine that still has a public SSH port. Task 2.3 before 3.1 so the
family never depends on a server whose backups have not been restored once. Task 3.2 last, after
the home PC's tasks are no longer the thing keeping the app alive.

**Across plans (2026-09-27):** Phases 1–2 run now, in parallel with Design Refresh Tasks 1.3 and
3.1 and the roles/UX review. Task 3.1 waits for the Roles & Product Plan's access decision and its
Phase A, per the prerequisites in that task. Task 1.2's backup-before-update must be in place
before any Roles & Product Plan migration reaches the server.

**One copy of record.** From Task 2.2 on, the server's `data/private/` and catalog records are
the live copies. Owner data entry (target-school awards and milestones, profile edits) happens in the
browser against `https://coach.<tailnet>.ts.net`, never against a dev-PC run. Local development
uses `scripts/screenshot_app.py --scratch-db` or a copy, so the two databases never diverge.
Before Task 2.2, entry on the dev PC is fine; Task 2.2 copies it over.

## Success Criteria

1. The app is reachable at `https://coach.<tailnet>.ts.net` from each family phone on cellular
   data, with the home PC switched off.
2. A port scan of the server's public IP finds no open TCP port.
3. A nightly encrypted backup exists off the server, and a restore from it has been done and
   checked by row count.
4. A catalog record confirmed in the app on the server reaches `main` through a PR that CI
   validated.
5. No scheduled task for ScholarshipCoach remains on the home PC; `docs/operations.md` describes
   the server, not the PC.
6. All four CI commands green after every task that changes the repo.
