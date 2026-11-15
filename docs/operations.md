# Operations

Running this app for one family, and keeping the curated catalog behind it honest. Nothing here
deploys to the public internet, and every catalog change proposes; a person confirms.

## Family hosting

**The decision: the app runs on the home PC, bound to the LAN, behind the family PIN. There is
no public deployment.** Off-network access, when it is wanted, comes from a private network
overlay (Tailscale) rather than from opening a port.

`data/private/` holds a real student's profile, essays, application history and outcomes. That
is the whole reason for the rule: a hosted URL is a standing invitation to index, scrape, or
guess at a teenager's file. A LAN-bound process is reachable by the phones in the house and by
nothing else, and it needs no certificate, no account system, and no paid tier.

| | Reachable from | Setup | Use it when |
|---|---|---|---|
| **LAN (default)** | Any device on the home Wi-Fi | One scheduled task | Always — this is the normal mode |
| **Tailscale (optional)** | The family's own devices, anywhere | Install on PC + phone, sign in | The student wants it from school or a friend's house |
| **Public hosting** | Everyone | — | Never |

### Set the family PIN first

The PIN turns the Parent and Operator selectors into a soft lock; Student mode is always open,
so the daily surface never asks for anything. Create `.streamlit/secrets.toml` (git-ignored):

```toml
parent_pin = "4417"
```

With no secrets file there is no PIN and every mode is open — a fine default for a trusting
household, a poor one once the app is reachable from every phone on the network. Set it before
the first LAN start.

### Start it on the LAN

```powershell
.\.venv\Scripts\streamlit.exe run app\main.py --server.address 0.0.0.0
```

`--server.address 0.0.0.0` is deliberately not baked into `.streamlit/config.toml`: a bare
`streamlit run` during development should stay on localhost, and LAN exposure should be
something a person typed. Find the address the phones use:

```powershell
Get-NetIPAddress -AddressFamily IPv4 -PrefixOrigin Dhcp | Select-Object IPAddress, InterfaceAlias
```

Open `http://<that address>:8501` on the phone. Reserve the address in the router's DHCP
settings so the bookmark does not rot, and allow the port once, for private networks only:

```powershell
New-NetFirewallRule -DisplayName "ScholarshipCoach (LAN)" -Direction Inbound `
    -LocalPort 8501 -Protocol TCP -Action Allow -Profile Private
```

`-Profile Private` is the load-bearing part — it keeps the rule from following a laptop onto
public Wi-Fi.

### Start on boot (Windows Task Scheduler)

Register once, from the project root, in an elevated PowerShell:

```powershell
$streamlit = Join-Path $PWD ".venv\Scripts\streamlit.exe"
$action  = New-ScheduledTaskAction -Execute $streamlit `
    -Argument "run app\main.py --server.address 0.0.0.0" -WorkingDirectory $PWD
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero)

Register-ScheduledTask -TaskName "ScholarshipCoach app" -Action $action -Trigger $trigger `
    -Settings $settings -Description "Serve the family scholarship app on the home network."
```

`-ExecutionTimeLimit ([TimeSpan]::Zero)` means no time limit — the default three days would
otherwise kill a long-running server mid-week. `-RestartCount 3` brings it back after a crash.

```powershell
Get-ScheduledTaskInfo -TaskName "ScholarshipCoach app"
Stop-ScheduledTask    -TaskName "ScholarshipCoach app"     # take it down
Start-ScheduledTask   -TaskName "ScholarshipCoach app"     # bring it back
```

### Who sees what (`[family]` in secrets)

On the server, `tailscale serve` tells the app who is signed in (the `Tailscale-User-Login`
header). A `[family]` table in the server's `.streamlit/secrets.toml` maps each login to a role:

```toml
parent_pin = "4417"

[family]
"student-login@example.com" = "student"
"parent-login@example.com" = "parent"
```

- A **student** login gets Student mode only. There is no mode switch and no PIN prompt, and
  the PIN cannot open Parent mode from her devices.
- A **parent** login opens in Parent mode with no PIN, and can still switch to Student to see
  her view (and to Operator when it is turned on in Settings).
- **No header or an unmapped login** (local development, a tagged device, a login missing from
  the table) falls back to the PIN exactly as above.

Logins are compared ignoring case. The sidebar shows "Signed in as …" whenever a login matched,
which is also the quick check that a new device is mapped. Pages opened are counted per login
per day (per mode under the PIN fallback); she is told this is counted. Keep real logins out of
the repo: the table lives only in the server's secrets file. Restart the service
(`sudo systemctl restart scholarshipcoach`) after editing it.

After a `git pull` that changes dependencies, restart the task — a running Streamlit process
does not pick up a new `pip install`.

### Off-network access (Tailscale)

Install Tailscale on the home PC and on the student's phone, and sign both into the same
account. The PC gets a stable `100.x.y.z` address, so `http://100.x.y.z:8501` works from
anywhere the phone has data, with no port forwarded and no DNS record. The app still needs
`--server.address 0.0.0.0` to accept the connection, and the PIN still gates Parent mode. This
is the only sanctioned way to reach the app from outside the house.

### Deploying an update (cloud server)

The app now runs on the cloud server (`coach` on the tailnet; see the Cloud Hosting Plan), which
serves whatever its checkout has merged from `main`. After pushing to `main`, from the dev PC:

```powershell
.\scripts\deploy.ps1
```

It refuses unless local `main` is exactly `origin/main`, waits for the GitHub CI run on that
commit and stops if it is not green, then runs `deploy/update.sh` over Tailscale SSH
(`ssh coach@coach`). `update.sh` backs up, merges `main`, reinstalls, restarts and
health-checks; its output is printed, and a failure exits non-zero. `-SkipCi` skips the CI wait
for emergencies only. Set `GITHUB_TOKEN` if the unauthenticated API rate limit (60/hour) is hit.

Deploy after a push, not necessarily after every task: each update restarts the app, so batch
when the family is using it. Deploy promptly when the push adds a file under
`src/store/migrations/`, so the backup sits next to the change. Never deploy from a GitHub
Action or a server-side timer: one needs a tailnet key in a public repo, the other applies
migrations with nobody watching.

### Catalog sync

Catalog edits made on the server (Add Award, inbox confirms, `checked_on` stamps, "Rebuild
snapshot" delta reports) are committed nightly by `deploy/catalog_sync.sh` to the `server`
branch and pushed; merge them into `main` by PR. The repo is public, so the push publishes the
edit before anyone reviews it.

**Privacy check.** Before it commits anything, the sync runs `deploy/privacy_check.sh` on the
changed records and delta reports. It reads two lists on the server only, owned by `coach`,
mode 600, git-ignored under `data/private/`:

- `privacy_names.txt`: the student's first target school's own names, and her first name, one
  per line, case-insensitive
- `privacy_abbrevs.txt`: the school's abbreviations, matched as whole words only

These are not the dev PC's lists. Those serve the occasional full-history audit (Roles & Product
Plan Task A.10), where an extra term only costs a line to read, so they can stay broad. Here
every term holds syncs, so the server copy leaves out:

- place names the local catalog legitimately contains (they would hold every new local award,
  and a city alone does not point to her)
- the family surname (the repo's authorship makes it public already, and it would hold any
  award that shares it)
- the audit's control line (`Example State University`)

Write them as `coach`, then `chmod 600`. Save them without a UTF-8 byte-order mark: the check
refuses such a file, since it would match nothing.

The check reads only the lines a change adds, plus whole new files, so an existing record that
names a college or a county passes when the verify run stamps its `checked_on`. A delta report
is always a new file and copies whole records, so a rebuild that touches a record naming the
school as a sponsor is held as well.

On a hit it prints only the file names, never the term, and nothing is committed or pushed. A
missing or empty list holds the sync the same way. `deploy/update.sh` runs the sync when records
are dirty, so a held record also stops an update until it is resolved; that is intended.

To resolve a hit:

1. **The record mentions her** (a "for her" note, a remark on her school's award): fix the
   record in Catalog & Inbox. The next nightly sync, or the next update, commits it.
2. **The record is a public award that merely names the school as its sponsor** and says
   nothing about her: the owner reads it (and any delta report the sync named) and commits the
   named files by hand on the server, as `coach`: `git add -- <file> && git commit -m
   "Catalog: <id>"`. The next sync pushes them. This is the only way past the guard, and a deliberate one.

Drill after copying the lists: add one term to a scratch record and run
`deploy/privacy_check.sh data/catalog/records/<id>.json`. It should exit 1 and name the file. A
`checked_on`-only change to a record that already names a place should exit 0. Then
`git checkout -- data/catalog/records/` (or delete the scratch file).

### Phone width

The student surfaces — To-dos, the essay editor, What If — are checked at ~400px.
Streamlit stacks `st.columns` to full width below 640px on its own; `phone_width_css()` in
`app/helpers.py` adds what stacking does not: 44px tap targets on buttons and expander headers,
a 16px floor on input text so mobile Safari stops zooming when the essay editor takes focus, and
trimmed page padding. Navigation is `st.navigation(position="top")`, one URL per page (a bookmark
to `/essays` opens Essays). On a desktop it is a top bar; on a phone Streamlit folds it into the
sidebar behind the `»` button, above the Student/Parent switch. The profile form is its own
"My Profile" page, not a sidebar panel. `client.toolbarMode = "minimal"` keeps the developer toolbar off the
student's screen.

**Checking it.** `scripts/screenshot_app.py` photographs every Student and Parent page (opened by its URL) at
400×860 and 1366×900, in light and dark, as `<width>_<mode>_<section>_<theme>.png`. It needs the
`ui` extra (`pip install -e ".[ui]"`) and drives the installed Edge, so nothing is downloaded. It
runs against a scratch copy of `data/private/` unless `--live-db` is passed; `--seed` saves three
Find results and adds an essay and a recommender first, so populated pages are captured too.
Screenshots go to the temp folder by default and are not committed.

```powershell
python scripts/screenshot_app.py --out $env:TEMP\coach_shots --scratch-db --seed
python scripts/screenshot_app.py --theme dark --sections find todos
```

### Back up `data/private/`

`data/private/` is git-ignored, which means git is not a backup. It holds `coach.db` (awards,
checklists, essays and their version history, recommenders, outcomes) and `students/<id>.json`.
Zip it weekly to a second location — an external drive or a synced folder, not another
directory on the same disk:

```powershell
$target = "D:\Backups\ScholarshipCoach"
New-Item -ItemType Directory -Force $target | Out-Null
Compress-Archive -Path "data\private\*" `
    -DestinationPath "$target\private_$(Get-Date -Format yyyyMMdd).zip" -Force
```

Scheduled weekly, from the project root, in an elevated PowerShell:

```powershell
$backup  = Join-Path $PWD "scripts\backup_private.ps1"
$action  = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$backup`"" `
    -WorkingDirectory $PWD
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 6am

Register-ScheduledTask -TaskName "ScholarshipCoach private backup" -Action $action `
    -Trigger $trigger -Settings (New-ScheduledTaskSettingsSet -StartWhenAvailable) `
    -Description "Weekly zip of data/private/ to a second location."
```

`scripts/backup_private.ps1` is that snippet plus a `-Keep` retention sweep (eight zips by
default); keeping it in a file rather than an inline `-Command` string is what makes the quoting
survive Task Scheduler. Its `-Target` defaults to `D:\Backups\ScholarshipCoach` — pass the drive
this machine actually has.

SQLite is copied while the app may be running, so take the backup when nobody is using it —
6am Sunday, an hour ahead of the re-verification task — and treat a restored `coach.db` as
"last week's" rather than "the exact moment it was zipped".

Keep the last few zips and one copy of `.streamlit/secrets.toml`. Those two, plus the repo, are
the whole system: everything else is regenerated by a snapshot rebuild.

### Restore

1. Stop the app so nothing holds the database open:
   ```powershell
   Stop-ScheduledTask -TaskName "ScholarshipCoach app"
   ```
2. Move the current directory aside rather than deleting it — a bad restore is recoverable, an
   overwrite is not:
   ```powershell
   Rename-Item data\private ("private_before_restore_" + (Get-Date -Format yyyyMMdd_HHmm))
   ```
3. Unpack the chosen zip into a fresh `data/private/`:
   ```powershell
   New-Item -ItemType Directory -Force data\private | Out-Null
   Expand-Archive -Path "D:\Backups\ScholarshipCoach\private_20260913.zip" `
       -DestinationPath data\private -Force
   ```
4. Check the database opens and carries the rows you expect:
   ```powershell
   python -c "from src.store.db import open_db; conn = open_db().__enter__(); print(conn.execute('select count(*) from applications').fetchone()[0], 'applications')"
   ```
5. Start the app and confirm Applications and Essays look right:
   ```powershell
   Start-ScheduledTask -TaskName "ScholarshipCoach app"
   ```
6. Once the restore is confirmed, delete the `private_before_restore_*` directory.

The curated catalog under `data/catalog/records/` is in git and needs no backup; only
`data/private/` and `.streamlit/secrets.toml` are unrecoverable if the disk dies.

---

## Monthly catalog re-verification

`scripts/verify_catalog.py` re-fetches each due record's `source_url`, hashes the cleaned page
text, re-runs the deterministic extractor, and compares `deadline`, amounts, `status`, and the
requirement flags against the record.

| What it found | What it does |
|---|---|
| Page agrees with the record | Stamps `provenance.checked_on` (and `checked_by: verify_catalog`) in place; `verified_on` / `verified_by` are left as they were |
| Page disagrees | Writes a `reverify` proposal to `data/catalog/inbox/` with the field-level diff; the proposed record carries the `checked_*` stamp and the existing `verified_*` pair unchanged |
| Link is dead (404, 410) | Writes a `reverify` proposal with `status: unknown` |
| Host refused the fetch (401, 403, 429, 5xx) | Reports `blocked` with `check_by_hand: true` — no proposal; the URLs are printed at the end of the run to open in a browser |
| Any other HTTP status | Reports an `error` — no proposal |
| Host unreachable, timeout | Reports an `error` — no proposal, because an offline laptop is not news |

Sponsor sites answer scripted fetches with 401, 403, 429, or a WAF's 5xx routinely, and those
pages are almost always alive in a browser. Only a 404 or 410 is read as a dead link; everything
else refusing the fetch is `blocked` and waits for a person, so a year of monthly passes cannot
quietly demote every bot-guarded record in the catalog.

A record's content is never edited by the script beyond that `checked_*` stamp. The script
writes only `checked_on` / `checked_by`, and never `verified_on` / `verified_by`: those two
record when a person last opened the page and confirmed the record, which is what
`trust: verified_local` rests on, and a machine pass must not overwrite that name. `trust` is
never raised either — `checked_by: verify_catalog` means a machine re-read the page and found
nothing contradicting the record, which is a weaker claim. A record is due when the later of
`verified_on` and `checked_on` is at least `--since-days` old, so a machine check still defers
the next fetch.

A recurring record (`cycle.recurring: true`) is also due around its usual opening, whatever
`--since-days` says: once in the 30 days before the 1st of `cycle.opens_month`, and once in the
45 days after it. A look inside each window satisfies it. With no opening month, applications are
assumed to open two months before `cycle.deadline_month`. That way a monthly run re-reads an annual
program's page when its new round is likely to be posted, not up to a year later.

A deadline is proposed only when the page states exactly one under a label, and only when it falls
after the run date and within 14 months: last cycle's date still on the page is not a new cycle.
"Due" and "no later than" count as labels only in a sentence about the deadline or the application,
because pages use them for transcript mailing, immunization forms and payments too. A labeled date
with no year is dated to its next occurrence when its month is the record's `cycle.deadline_month`,
and the proposal's note says the year was inferred. A dated deadline in a different month than
`cycle.deadline_month` proposes the corrected month alongside it, with a note.

A change a person rejected is not proposed again: when the proposal id and the changed fields
(ignoring the `checked_*` stamp) match a file in `inbox/rejected/`, the run counts it as
`previously_rejected` instead of writing it. This holds for every proposal kind; only the Add
Award form, where a person enters and confirms at once, skips the check. A change to different
values is proposed as usual.

### Run it by hand

```powershell
# Everything not verified in the last year, polite 0.5 req/s
python scripts/verify_catalog.py

# Quick smoke run: five records, ignore when they were last verified
python scripts/verify_catalog.py --since-days 0 --max-records 5
```

Flags: `--since-days` (default 365), `--max-records`, `--requests-per-second` (default 0.5),
`--records-dir`, `--inbox-dir`, `--reports-dir`.

Each run writes `reports/catalog_verify/catalog_verify_<UTC stamp>.json` with per-record
outcome, HTTP status, content hash, and diff. The next run reads the newest report to fill in
`content_changed`, so a page that was edited without changing any extracted field is still
visible in the report.

### Work the queue afterwards

```powershell
python scripts/catalog_inbox.py list
python scripts/catalog_inbox.py show reverify-afcea-stem-major-scholarship
python scripts/catalog_inbox.py confirm reverify-afcea-stem-major-scholarship --set trust=verified_local
python scripts/catalog_inbox.py reject reverify-some-award --reason "page moved; re-adding by hand"
```

`confirm` is the only path from the inbox into `data/catalog/records/`, and it validates against
`data/catalog/schema.json` first. Proposals and rejections are git-ignored local working state.

### Schedule it monthly (Windows Task Scheduler)

Register the task once, from the project root, in an elevated PowerShell:

```powershell
$python  = Join-Path $PWD ".venv\Scripts\python.exe"
$script  = Join-Path $PWD "scripts\verify_catalog.py"
$action  = New-ScheduledTaskAction -Execute $python -Argument $script -WorkingDirectory $PWD
$trigger = New-ScheduledTaskTrigger -Weekly -WeeksInterval 4 -DaysOfWeek Sunday -At 7am
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

Register-ScheduledTask -TaskName "ScholarshipCoach catalog re-verification" `
    -Action $action -Trigger $trigger -Settings $settings -Description `
    "Re-fetch each curated record's source page and queue changes in the catalog inbox."
```

`-StartWhenAvailable` makes a missed run (laptop asleep) fire at the next boot instead of being
skipped. Because `--since-days` defaults to 365, running monthly costs almost nothing: only the
records that have aged past a year are fetched, and a full catalog naturally spreads itself
across the calendar.

Check on it:

```powershell
Get-ScheduledTaskInfo -TaskName "ScholarshipCoach catalog re-verification"
Start-ScheduledTask   -TaskName "ScholarshipCoach catalog re-verification"   # run now
Unregister-ScheduledTask -TaskName "ScholarshipCoach catalog re-verification" -Confirm:$false
```

The task's exit code is 0 for any completed pass — dead links, changed pages, and blocked hosts
are the output of a healthy run, not a failure — and 1 only when the catalog directory holds no records. Read
what a run found in `reports/catalog_verify/`, or from the inbox.

---

## Weekly discovery

`scripts/discover.py` refreshes the feed sources (today only Open Scholarships; `FEED_SOURCES`
in the script, and only those enabled in `data/catalog/sources.json`) and proposes what changed.
On the server, `scholarshipcoach-discover.timer` runs it Sundays at 05:00 as `coach`.

It runs ingest for the feed sources alone, with the usual guardrails: the curated rows are
carried forward from the prior snapshot, and a record drop of more than half blocks the
snapshot write. Then, per source that succeeded, `src/catalog/discovery.py` writes `feed`
proposals to `data/catalog/inbox/`: rows that are new, or whose deadline, status or amount
changed since the last run, that the curated catalog does not already carry with the same
values, that were not rejected before with the same content, and that pass Stage 1 for the
saved profile with the grade ignored. At most 15 a run; the rest are counted in the report, not
queued. The first run for a source records a baseline (`reports/discovery/feed_state.json`) and
proposes nothing.

It never confirms a proposal and never writes under `data/catalog/records/`;
`tests/test_deploy_config.py` checks that for everything a timer reaches. A written snapshot
leaves a `data/processed/changes_*.json` delta report, which the nightly catalog sync commits
after its privacy check, as it does for "Rebuild snapshot".

| Outcome | What it means | Exit |
|---|---|---|
| `completed`, a source `succeeded` with proposals | New or changed awards wait in Catalog & Inbox with the feed badge and "why this was proposed" | 0 |
| `completed`, nothing proposed | The feed had nothing new for her, or this was the baseline run. Healthy | 0 |
| A source `failed` | The fetch or parse raised, or a source that returned records before returned none (`zero_record_regression: true`). That source is not planned and its state is kept, so the next good run compares against the last good one | 1 |
| Guardrail blocked | The snapshot was not written and no source was planned. The prior snapshot stays | 1 |

A non-zero exit leaves the unit failed in `systemctl --failed`; the weekly digest (Task 3.1 of
the Freshness & Today Plan) will say so in plain words.

### Run it by hand

```powershell
# Fetch and report only: no snapshot, proposals, state or report are written
python scripts/discover.py --dry-run

# A real pass, as the timer runs it
python scripts/discover.py
```

On the server, as `coach`: `sudo systemctl start scholarshipcoach-discover.service`, then
`journalctl -u scholarshipcoach-discover -n 50`. Each real run writes
`reports/discovery/discovery_<UTC stamp>.json`: per source its status (`succeeded`, `partial`,
`skipped` or `failed`), records, the zero-record regression flag, proposals written with their
ids, and the left-out counts by reason (`baseline`, `unchanged`, `covered_by_curated`,
`previously_rejected`, `failed_profile_screen`, `over_cap`). The ingest report it ran is in
`reports/ingest_runs/`.

### Install the timer on the server

`deploy/bootstrap.sh` installs it with the other timers, and is safe to re-run. By hand, as root
after an update has merged the unit files:

```bash
cd /srv/scholarshipcoach
install -m 644 deploy/scholarshipcoach-discover.service deploy/scholarshipcoach-discover.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now scholarshipcoach-discover.timer
systemctl list-timers scholarshipcoach-discover.timer
```

## Watch list

Local awards and summer programs are in no feed; they sit on a few dozen pages that change a few
times a year. `scripts/check_watchlist.py` re-reads those pages and proposes what changed. On the
server, `scholarshipcoach-watch.timer` runs it Sundays at 05:30 as `coach`, after discovery and
before the first-Sunday verify. It never confirms a proposal and never writes under
`data/catalog/records/`.

**Two lists, one shape.** `data/catalog/watchlist.json` is committed and holds only pages that say
nothing about her (the county foundation, statewide programs). Her school's counselor page,
employers and congregations go in `data/private/watchlist.json` on the server, which is merged at
run time and backed up with the rest of `data/private/`. `scripts/validate_catalog.py` checks the
committed list against `data/catalog/watchlist.schema.json` and, where the privacy term lists
exist, refuses an entry that names one (it prints the entry id, never the term).

Each entry has an `id`, `url`, `label`, `page` (`single`: one award or program; `listing`: a page
of links, one per award), `kind` (`scholarship` or `program`), the `months` it is worth checking,
`enabled`, `notes`, and a `terms_note` with the day a person read the site's terms and robots.txt.
An entry whose `terms_note` is null does not run.

| What it found | What it does |
|---|---|
| First sight of a page | Records a baseline; proposes nothing. To have a new page read now, paste it into Add Award |
| Page unchanged | One request; nothing written |
| `single` page changed, no record shares its URL | A `prefill` proposal for a new record, `trust: unverified` |
| `single` page changed, a record shares its URL | A `prefill` proposal with the fields the page now contradicts (the verify pass's rules, plus a program's cost and grades) |
| `single` page changed, everything read still matches the record | Finding: "changed, still matches the record; open it" |
| `listing` page changed | Each new same-site link that matches no record and no rejected proposal is fetched and prefilled, at most 10 a page; one proposal per award, never one for the listing. Links over the 10 are kept unsaved, so the next change follows them |
| `listing` changed with no new link | Finding (or "postback portal" when its links are form buttons) |
| A PDF, a refused fetch (401/403/429/5xx), a dead link (404/410), or a page with no title or no date, amount or grade | Finding with the URL to open; no proposal |
| Host unreachable | `error`; no finding, because an offline laptop is not news |

At most 15 proposals a run; the rest are counted (`over_cap`) and asked again on the next change.
Proposal ids are `prefill-<watch id>-<page slug>`, so a re-run replaces its own proposal, and a
change a person rejected is not proposed again (Task 1.2's rule). A new record never carries the
page's last-cycle deadline: a date already past is left out and named in the record's notes.

The inbox shows "From the watch list" and the watch entry's id. The record itself never names the
entry, so a confirmed record carries nothing from a private one; the entry id lives only in the
git-ignored inbox file. A program page's session dates go in the record's notes (the catalog has
no session field).

### Run it by hand

```powershell
# Fetch and report only: no proposals, state or report are written
python scripts/check_watchlist.py --dry-run

# Every enabled entry, whatever the month
python scripts/check_watchlist.py --ignore-months
```

Each real run writes `reports/watchlist/watchlist_<UTC stamp>.json` (per entry: outcome,
proposals, findings, left-out counts by reason: `known_record`, `previously_rejected`,
`over_link_cap`, `over_cap`) and the page hashes and listing links in
`reports/watchlist/watch_state.json`. Both are git-ignored. Exit 0 when the pass ran, whatever it
found; 1 when a watch list file is unreadable or invalid, which leaves the unit failed in
`systemctl --failed`.

On the server, as `coach`: `sudo systemctl start scholarshipcoach-watch.service`, then
`journalctl -u scholarshipcoach-watch -n 50`.

### The committed entries

Drafted 2026-11-14 from the pages named below and the six program records' sponsors. Each page and
its robots.txt was opened once and the site's terms or privacy page read; what was found is in the
entry's `terms_note`. Four entries are disabled with the reason in `notes`: the three on ncsu.edu
(Space Grant, the engineering camps, the pre-college listing), whose legal pages answer scripts
with an empty 202, and Girls Who Code, whose terms render only with JavaScript. Read those terms
in a browser, then fill `terms_note` and set `enabled`. PENC stays disabled (403 to scripts,
yearly PDFs). Two more drafted entries matched the privacy term lists, so they went to
`data/private/watchlist.json` on the dev PC instead; copy that file to the server with the rest of
`data/private/`.

### Install the timer on the server

`deploy/bootstrap.sh` installs it with the other timers. By hand, as root after an update has
merged the unit files:

```bash
cd /srv/scholarshipcoach
install -m 644 deploy/scholarshipcoach-watch.service deploy/scholarshipcoach-watch.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now scholarshipcoach-watch.timer
systemctl list-timers scholarshipcoach-watch.timer
```

The first run records a baseline for every due entry and proposes nothing.

---

## Where local awards come from

Local and state awards have the smallest applicant pools and appear in no aggregator. The pages
below that can be read by a script are on the [watch list](#watch-list): FFTC's cycle dates, the
statewide awards and the program pages are re-read weekly in their months, and a change arrives as
an inbox proposal or a finding. The family's own pages (counselor, employers, congregations), and
any drafted page whose text matches the privacy term lists, go in the private overlay on the
server, `data/private/watchlist.json`.

Still by hand: FFTC's funds (a postback portal with no page per fund; the watch list only sees its
cycle dates), PENC (403 to scripts, yearly PDFs), the ncsu.edu and Girls Who Code entries until
someone reads their terms, and anything posted only as a paper or PDF form or not posted at all.
For those, paste the URL into **Catalog → Add Award**, correct what prefill got wrong, and confirm.
Nothing gets `trust: verified_local` until a person has opened the page that day.

### Local sources (the family's list)

These are the sources only the family can know. Each one gets its URL pasted into Add Award once
a year, and the counselor's list is the one that moves the most.

| Source | Where to look | Re-check |
|---|---|---|
| High school counselor's scholarship page or handout | School website / counseling office | **Every fall** (September) and again in January; the list is rebuilt each school year |
| County community foundation | The foundation's scholarship portal | Each fall, when the portal opens (usually November–January) |
| Family credit union | Member site, "scholarships" or "community" page | Each fall |
| Parents' employers | HR / benefits portal, dependent scholarship programs | Each fall; some programs change sponsors between years |
| Church and civic organizations (Rotary, Lions, Elks, VFW/American Legion posts, sororities/fraternities) | Their websites or a phone call | Each winter; many post only a paper form in January–March |

The owner fills in the actual names and URLs; a source with no web page is still worth a row with
a note of who to call.

### Gaston and Mecklenburg counties

Searched 2026-09-26. The two county foundations carry nearly all of the local money; the school
district pages mostly link back to them.

**Scope rule:** awards that require Mecklenburg County residency or enrollment in a
Charlotte-Mecklenburg school are out of scope for this family — skip them in future searches.
From FFTC, only funds open to any NC (or Carolinas) senior apply.

| Source | What it holds | Cycle | Re-check |
|---|---|---|---|
| [Gaston Community Foundation](https://www.cfgaston.org/explore-scholarships/) | 34 funds; 11 are open county-wide (Myers and Barnard at $20,000; Ragan, Gingles, Beam, Brackett, Bess Chapel, Sadler, Holmes, Locke Bell, Max J. Fowler) | Most due early March (Mar 2 in 2026); Locke Bell Mar 31 | **January** — the portal reopens for the new cycle |
| [Foundation For The Carolinas](https://fftcscholarships.communityforce.com/Funds/Search.aspx) | 161 funds; in the catalog: Trawick, Annable, Advocates for African Americans (open to any NC senior). Mecklenburg-only funds (Holcomb, Caldwell, CMS Incentive, Panthers/Sam Mills) are out of scope | 2027 cycle opens Dec 1, 2026, closes Mar 5, 2027 | **December** — dates post before the portal opens |
| [PENC Harold McKnight](https://cdn.ymaws.com/penc.org/resource/resmgr/scholarships/2025_Harold_McKnight_Scholar.pdf) | $1,000, engineering at an NC school (earlier years limited it to nine Charlotte-region counties; 2025 does not) | May 1 | **March**, for the new year's PDF |
| [Gaston County Schools](https://www.gaston.k12.nc.us/for-parents/school-counseling/scholarships) / [CMS](https://www.cmsk12.org/academics/fafsa-resources/financial-resources-scholarships) | Links to the two foundations plus national search sites | — | Each fall, with the counselor |

Not seeded, and why:
- **One-school funds.** About 15 Gaston funds, and a similar share of FFTC's, are limited to a
  single high school (Cherryville, Forestview, South Point, Ashbrook, Hunter Huss, North
  Mecklenburg, Garinger, Providence, and others). The catalog has no high-school field, so they
  would rank as eligible for every student in the county. Add the ones for her school by hand, with
  the school named in `eligibility_text`, once the school is known.
- **Employer, church, trade, nursing, teaching, and graduate-only funds** — no path for a CS/CE
  student, or tied to an employer or congregation the family has not named.
- **CapTech STEM and Logical Advantage technology scholarships** (Charlotte) — the only listing
  found is a 2015 IT-oLogy post. Ask the CMS counselor whether they still run.

FFTC fund pages have no per-fund URL (the portal is an ASP.NET postback), so those records point at
the portal search page and name the Fund ID in `notes`; search the portal by fund name to open one.
`verify_catalog.py` re-fetches the shared portal page for all of them, so a change to one fund's
details will not show up in its diff — re-check FFTC funds by hand each December.

### NC statewide programs

Drafted by Claude on 2026-09-26 as `manual` inbox proposals (`trust: unverified`) through the same
prefill path the Add Award page uses. Each waits on the Inbox page for the owner to open the source
and confirm.

| Award | Scope | Cycle | Source |
|---|---|---|---|
| Golden LEAF Colleges and Universities Scholarship | 81 rural counties | Jan → Mar 1 | scholars.goldenleaf.org |
| Aubrey Lee Brooks Scholarship | 14 counties (incl. Guilford); NC State, UNC-CH, UNCG | Jan → Mar 1 | cfnc.org (the NCSEAA page is password-protected) |
| Betsy Y. Justus NC TECH Founders Scholarship | NC; women; technology or engineering | Jan → Apr 15 | nctech.org |
| PENC Engineering Freshman Scholarship | NC; engineering | spring → May 1 | PENC application PDF (penc.org answers 403 to scripts) |
| NCAE Dr. Martin Luther King Jr. Scholarship | NC public HS seniors | first Monday in Feb | ncae.org |
| NC Scholarship for Children of Wartime Veterans | NC; military family | Feb 14 | milvets.nc.gov |
| SECU Foundation People Helping People | NC public HS; SECU members; UNC campuses | set by each school district | ncsecufoundation.org |
| NC Sheriffs' Association Criminal Justice Scholarship | NC; CJ undergraduates | Mar 31 | ncsheriffs.org (date from CFNC) |
| NC Space Grant Undergraduate Research Scholarship | NC colleges; STEM | Mar | ncspacegrant.ncsu.edu |
| North Carolina Teaching Fellows (forgivable loan) | NC; teaching | Nov 1 early / Feb 14 | myapps.northcarolina.edu |
| NC 4-H Development Fund Scholarships | NC; 4-H members | Feb 1 (county deadline earlier) | cfnc.org |
| R. Flake Shaw Scholarship | NC; agriculture | Mar | ncfb.org |
| AFCEA North Carolina Chapter graduating senior award | NC; grades 11–12 and college; STEM; 2.7 GPA | Apr 20 (2026, extended) | afceanc.org (date from BizFayetteville) |

Seven are in the catalog (Golden LEAF, NC TECH Founders, PENC Freshman, NCAE MLK Jr., Space
Grant, Teaching Fellows, AFCEA NC). The owner rejected the rest as not applying to this family:
Brooks (county), Wartime Veterans (military family), Sheriffs (criminal justice), 4-H
(membership), R. Flake Shaw (agriculture), and SECU (the family are not SECU members). Keep them
listed so a later search does not re-propose them.

Re-check the seven in **December**: most open in January, and a December pass catches the new
cycle's date before the award reaches the Now bucket.

### What prefill missed

Recorded while seeding, for a later prefill task. Every one of these was corrected by hand before
proposing. Items marked *fixed* name the task that fixed them.

- **Year-less dates.** *Fixed (Freshness Task 1.2) under a deadline label:* "Deadline March 1" and
  "no later than the first Monday in February" are read, and the verify pass dates them to their
  next occurrence when the month is the record's `cycle.deadline_month`; the proposal's note says
  the year was inferred. Still unread: a year-less date with no label, one in a month the record's
  cycle disagrees with, and calendar layouts that put the date before its label ("JAN 12
  Application Deadline"). The Add Award form still offers only dated candidates.
- **Wrong date chosen.** *Fixed (Freshness Task 1.2):* a date right after "opens", "will open",
  "open date" or "available" is never a deadline candidate, and a date right after a deadline label
  ("DEADLINE March 1, 2027", "Deadline Due", "closes on", with or without a colon) outranks every
  other date on the page. Unlabeled dates (info sessions, award announcements) still crowd the
  candidate list when the page has no label.
- **County lists.** The county pattern requires "X County" after every name, so "Alamance, Bertie,
  …, Swain or Warren county" yields only Warren, and "Cleveland, Gaston, Lincoln, Polk, and
  Rutherford Counties" yields two of five. A list in a linked PDF yields nothing. (Single-county
  pages — "Gaston County students" — were read correctly every time.)
- **Page chrome dates.** *Fixed (Freshness Task 1.2):* on the Gaston foundation's pages, sidebar
  post dates ("October 7, 2025") crowded out the labeled "Deadline March 2, 2026" because the label
  had no colon; the label is now read without one and outranks the post dates.
- **Postback portals.** FFTC's CommunityForce portal has no URL per fund, so there is nothing to
  paste into Add Award.
- **Counties and states from prose.** "a native of Guilford County" restricted a statewide award to
  Guilford; "Hollywood, Florida" in a winner's bio added Florida.
- **Site chrome as requirements.** CFNC's navigation says "FAFSA 101" on every page, so every CFNC
  listing is flagged FAFSA-required.
- **Renewal terms as entry criteria.** "maintains a 2.75 cumulative GPA" (renewal) became the minimum
  GPA.
- **Totals as award amounts.** Program lifetime totals ("more than $3,000,000", "$580,000") become
  `amount_max`.
- **PDFs.** A PDF URL is fetched and its raw bytes are treated as text; prefill reports success
  with nothing usable. (The watch list recognises a PDF and reports it as a finding instead.)
- **Program fields.** *Fixed (Freshness Task 2.3):* prefill reads a program's cost ("Cost of
  Attendance $1,550", "Program fees: $6300", "free"/"no cost"), its grades as the grade when
  applying ("rising 11th and 12th graders" → 10, 11; "second or third year of high school"), and
  its session date ranges, from the page's main content only, so the site menu's other programs
  cannot leak in. Still unread: a weekly rate ("$450/week") and any cost a page gives only in a
  linked handbook. The Add Award form now seeds the grades; it leaves Cost for a person, since a
  "free" on a scholarship page is about applying, not attending.
- **Never extracted.** Gender, military family, membership, and majors outside the vocabulary are
  not attempted.
