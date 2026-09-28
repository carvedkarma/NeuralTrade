param(
    [Parameter(Mandatory=$true)]
    [string]$BackupDirectory,
    [string]$ProjectRoot = "."
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$backup = (Resolve-Path -LiteralPath $BackupDirectory).Path
$liveBackup = Join-Path $backup "live.py"
$liveTarget = Join-Path $root "mythos\neural\live.py"
$moduleBackup = Join-Path $backup "rithal_behavior_fix_v105_r3_3.py"
$moduleTarget = Join-Path $root "mythos\neural\rithal_behavior_fix_v105_r3_3.py"

if (-not (Test-Path -LiteralPath $liveBackup)) {
    throw "Backup live.py not found: $liveBackup"
}

Copy-Item -LiteralPath $liveBackup -Destination $liveTarget -Force
if (Test-Path -LiteralPath $moduleBackup) {
    Copy-Item -LiteralPath $moduleBackup -Destination $moduleTarget -Force
} elseif (Test-Path -LiteralPath $moduleTarget) {
    Remove-Item -LiteralPath $moduleTarget -Force
}

Write-Host "[RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3] rollback complete" -ForegroundColor Yellow
Write-Host "Restored from: $backup"
