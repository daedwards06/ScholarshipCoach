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

After a `git pull` that changes dependencies, restart the task — a running Streamlit process
does not pick up a new `pip install`.

### Off-network access (Tailscale)

Install Tailscale on the home PC and on the student's phone, and sign both into the same
account. The PC gets a stable `100.x.y.z` address, so `http://100.x.y.z:8501` works from
anywhere the phone has data, with no port forwarded and no DNS record. The app still needs
`--server.address 0.0.0.0` to accept the connection, and the PIN still gates Parent mode. This
is the only sanctioned way to reach the app from outside the house.

### Phone width

The student surfaces — ranked cards, This Week, the essay editor — are checked at ~400px.
Streamlit stacks `st.columns` to full width below 640px on its own; `phone_width_css()` in
`app/helpers.py` adds what stacking does not: 44px tap targets on buttons and expander headers,
a 16px floor on input text so mobile Safari stops zooming when the essay editor takes focus, and
trimmed page padding. Navigation lives in the sidebar, which Streamlit collapses behind the
hamburger on a phone. `client.toolbarMode = "minimal"` keeps the developer toolbar off the
student's screen.

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
5. Start the app and confirm My Applications and Essays look right:
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

## Where local awards come from

Local and state awards have the smallest applicant pools and appear in no aggregator, so they are
entered by hand: paste the URL into **Catalog → Add Award**, correct what prefill got wrong, and
confirm. Nothing gets `trust: verified_local` until a person has opened the page that day.

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
proposing.

- **Year-less dates.** "March 1", "April 15", "February 1", and "first Monday in February" produce
  no candidate. Pages state a recurring month far more often than a dated deadline.
- **Wrong date chosen.** On CFNC pages the application-open date ("will open on Monday, January 12,
  2026") was taken as the deadline, while the labeled "DEADLINE March 1" was missed. Unrelated dates
  (info sessions, award announcements) crowd the candidate list.
- **County lists.** The county pattern requires "X County" after every name, so "Alamance, Bertie,
  …, Swain or Warren county" yields only Warren, and "Cleveland, Gaston, Lincoln, Polk, and
  Rutherford Counties" yields two of five. A list in a linked PDF yields nothing. (Single-county
  pages — "Gaston County students" — were read correctly every time.)
- **Page chrome dates.** On the Gaston foundation's pages, sidebar post dates ("October 7, 2025")
  crowd out the labeled "Deadline March 2, 2026"; one page proposed the post date as its deadline.
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
  with nothing usable.
- **Never extracted.** Gender, military family, membership, and majors outside the vocabulary are
  not attempted.
