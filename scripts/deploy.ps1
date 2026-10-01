# Deploy what is on origin/main to the family server, once CI has passed on it.
#
#   .\scripts\deploy.ps1            # wait for CI on the pushed commit, then update
#   .\scripts\deploy.ps1 -SkipCi    # emergencies only
#
# The server is reachable only over Tailscale; deploy/update.sh does the
# backup, pull, reinstall, restart and health check. See docs/operations.md.

param(
    [string]$Server = "coach@coach",
    [string]$UpdateScript = "/srv/scholarshipcoach/deploy/update.sh",
    [int]$TimeoutMinutes = 30,
    [int]$PollSeconds = 30,
    [switch]$SkipCi
)

$ErrorActionPreference = "Stop"

function Fail([string]$message) {
    Write-Host "deploy.ps1: $message" -ForegroundColor Red
    exit 1
}

git fetch -q origin main
if ($LASTEXITCODE -ne 0) { Fail "git fetch failed" }
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
$local = (git rev-parse HEAD).Trim()
$remote = (git rev-parse origin/main).Trim()
if ($branch -ne "main" -or $local -ne $remote) {
    Fail "local $branch ($($local.Substring(0, 7))) is not origin/main ($($remote.Substring(0, 7))); push or pull first"
}
$sha = $remote
$short = $sha.Substring(0, 7)

if ($SkipCi) {
    Write-Host "==> Skipping the CI check for $short (-SkipCi)" -ForegroundColor Yellow
} else {
    $origin = (git remote get-url origin).Trim()
    if ($origin -notmatch "github\.com[:/](?<repo>[^/]+/[^/]+?)(\.git)?$") {
        Fail "origin is not a GitHub URL: $origin"
    }
    $api = "https://api.github.com/repos/$($Matches.repo)/actions/runs?head_sha=$sha&event=push"
    # The repo is public, so no token is needed; one only raises the rate limit.
    $headers = @{ Accept = "application/vnd.github+json" }
    if ($env:GITHUB_TOKEN) { $headers.Authorization = "Bearer $env:GITHUB_TOKEN" }

    Write-Host "==> Waiting for CI on $short"
    $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
    while ($true) {
        $runs = (Invoke-RestMethod -Uri $api -Headers $headers).workflow_runs |
            Where-Object { $_.name -eq "CI" } |
            Sort-Object created_at -Descending
        $run = $runs | Select-Object -First 1
        if ($null -eq $run) {
            Write-Host "    no CI run for $short yet"
        } elseif ($run.status -ne "completed") {
            Write-Host "    CI $($run.status)  $($run.html_url)"
        } elseif ($run.conclusion -eq "success") {
            Write-Host "    CI passed  $($run.html_url)" -ForegroundColor Green
            break
        } else {
            Fail "CI $($run.conclusion) on $short; not deploying. $($run.html_url)"
        }
        if ((Get-Date) -gt $deadline) { Fail "no green CI on $short after $TimeoutMinutes minutes" }
        Start-Sleep -Seconds $PollSeconds
    }
}

Write-Host "==> Updating $Server to $short"
ssh $Server $UpdateScript
if ($LASTEXITCODE -ne 0) { Fail "update.sh exited $LASTEXITCODE on $Server; read its output above" }
Write-Host "==> Deployed $short" -ForegroundColor Green
