# Weekly backup of data/private/ -- the one directory git does not carry.
# Registered as a scheduled task in docs/operations.md. Point $Target at the
# drive this machine actually has.

param(
    [string]$Target = "D:\Backups\ScholarshipCoach",
    [int]$Keep = 8
)

$ErrorActionPreference = "Stop"

$source = Join-Path $PSScriptRoot "..\data\private"
if (-not (Test-Path $source)) {
    Write-Error "No data/private/ to back up at $source"
    exit 1
}

New-Item -ItemType Directory -Force $Target | Out-Null
$archive = Join-Path $Target ("private_" + (Get-Date -Format "yyyyMMdd") + ".zip")
Compress-Archive -Path (Join-Path $source "*") -DestinationPath $archive -Force
Write-Output "Wrote $archive"

Get-ChildItem $Target -Filter "private_*.zip" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -Skip $Keep |
    Remove-Item -Force
