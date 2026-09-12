# Operations

Scheduled maintenance for the curated catalog. Everything here proposes; a person confirms.

## Monthly catalog re-verification

`scripts/verify_catalog.py` re-fetches each due record's `source_url`, hashes the cleaned page
text, re-runs the deterministic extractor, and compares `deadline`, amounts, `status`, and the
requirement flags against the record.

| What it found | What it does |
|---|---|
| Page agrees with the record | Stamps `provenance.verified_on` (and `verified_by: verify_catalog`) in place |
| Page disagrees | Writes a `reverify` proposal to `data/catalog/inbox/` with the field-level diff |
| Link is dead (4xx/5xx) | Writes a `reverify` proposal with `status: unknown` |
| Host unreachable, timeout | Reports an `error` — no proposal, because an offline laptop is not news |

A record's content is never edited by the script beyond that verification stamp, and `trust` is
never raised: `verified_by: verify_catalog` means a machine re-read the page and found nothing
contradicting the record, which is a weaker claim than `trust: verified_local` — that one still
requires a person.

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
python scripts/catalog_inbox.py show reverify-afcea-stem-scholarship
python scripts/catalog_inbox.py confirm reverify-afcea-stem-scholarship --set trust=verified_local
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

The task's exit code is 0 for any completed pass — dead links and changed pages are the output
of a healthy run, not a failure — and 1 only when the catalog directory holds no records. Read
what a run found in `reports/catalog_verify/`, or from the inbox.
