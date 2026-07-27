param(
    [string]$ProjectRoot = ".",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Version = "RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3"
$root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$pkg = Split-Path -Parent $MyInvocation.MyCommand.Path
$live = Join-Path $root "mythos\neural\live.py"
$model = Join-Path $root "mythos\neural\model.py"
$features = Join-Path $root "mythos\features.py"
$dataEngine = Join-Path $root "data_engine_v22.py"
$moduleTarget = Join-Path $root "mythos\neural\rithal_behavior_fix_v105_r3_3.py"
$modulePayload = Join-Path $pkg "payload\mythos\neural\rithal_behavior_fix_v105_r3_3.py"
$verifier = Join-Path $pkg "Verify-RithalBehaviorV105R3_3.py"

foreach ($path in @($live, $model, $features, $dataEngine, $modulePayload, $verifier)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required file missing: $path" }
}

$running = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.CommandLine -match 'quick_start\.py' -and
    $_.CommandLine -match 'live-neural-v2'
})
if ($running.Count -gt 0) {
    $pids = ($running | ForEach-Object { $_.ProcessId }) -join ","
    throw "Stop the Rithal live engine before installing. Active Python PID(s): $pids"
}

function Get-HashMap([string[]]$Paths) {
    $map = [ordered]@{}
    foreach ($path in $Paths) {
        if (Test-Path -LiteralPath $path) {
            $map[$path] = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
        }
    }
    return $map
}

$checkpointFiles = @(Get-ChildItem -LiteralPath (Join-Path $root "checkpoints\rithal_clean") -File -Filter "mythos_neural_*.pt" -ErrorAction Stop)
if ($checkpointFiles.Count -ne 6) {
    throw "Expected exactly six Rithal checkpoints; found $($checkpointFiles.Count)."
}
$protectedPaths = @($model, $features, $dataEngine) + @($checkpointFiles.FullName)
$protectedBefore = Get-HashMap $protectedPaths

$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$backup = Join-Path $root "rithal_backups\behavior_v105_r3_3_$stamp"
New-Item -ItemType Directory -Path $backup -Force | Out-Null
Copy-Item -LiteralPath $live -Destination (Join-Path $backup "live.py") -Force
if (Test-Path -LiteralPath $moduleTarget) {
    Copy-Item -LiteralPath $moduleTarget -Destination (Join-Path $backup "rithal_behavior_fix_v105_r3_3.py") -Force
}

try {
    Copy-Item -LiteralPath $modulePayload -Destination $moduleTarget -Force

    $content = [IO.File]::ReadAllText($live)
    $content = [regex]::Replace(
        $content,
        '(?ms)\r?\n?# RITHAL_BEHAVIOR_V105_R3_3_START.*?# RITHAL_BEHAVIOR_V105_R3_3_END\r?\n?',
        "`r`n"
    )

    $oldLogger = '_rp.get("p_trend", 0.0), _rp.get("p_chop", 0.0), _rp.get("p_panic", 0.0),'
    $newLogger = '_rp.get("p_trend", 0.0), _rp.get("p_chop", 0.0), _rp.get("p_breakout", 0.0),'
    if ($content.Contains($oldLogger)) {
        $content = $content.Replace($oldLogger, $newLogger)
    } elseif (-not $content.Contains($newLogger)) {
        throw "Could not locate the rich REGIME-P logger contract in live.py"
    }

    $oldProbe = 'probe_logits = _np.asarray([1.0, 0.2, -0.4, 0.8], dtype=float)'
    $newProbe = 'probe_logits = _np.asarray([1.0, 0.2, -0.4, 0.8, -0.3], dtype=float)'
    if ($content.Contains($oldProbe)) {
        $content = $content.Replace($oldProbe, $newProbe)
    } elseif (-not $content.Contains($newProbe)) {
        throw "Could not locate the startup regime-parity probe in live.py"
    }

    $activation = @'

# RITHAL_BEHAVIOR_V105_R3_3_START
try:
    from .rithal_behavior_fix_v105_r3_3 import apply_live_patch as _rithal_v105_r33_apply_live_patch
except ImportError:
    from rithal_behavior_fix_v105_r3_3 import apply_live_patch as _rithal_v105_r33_apply_live_patch
_rithal_v105_r33_apply_live_patch(globals())
# RITHAL_BEHAVIOR_V105_R3_3_END
'@
    $content = $content.TrimEnd() + "`r`n" + $activation.TrimStart() + "`r`n"
    [IO.File]::WriteAllText($live, $content, [Text.UTF8Encoding]::new($false))

    & $Python -m py_compile $moduleTarget $live $verifier
    if ($LASTEXITCODE -ne 0) { throw "Python compile failed" }

    & $Python $moduleTarget --self-test
    if ($LASTEXITCODE -ne 0) { throw "R3.3 module self-test failed" }

    & $Python $verifier --project-root $root --source-only
    if ($LASTEXITCODE -ne 0) { throw "R3.3 source verification failed" }

    $protectedAfter = Get-HashMap $protectedPaths
    foreach ($path in $protectedBefore.Keys) {
        if ($protectedBefore[$path] -ne $protectedAfter[$path]) {
            throw "Protected source changed unexpectedly: $path"
        }
    }

    $record = [ordered]@{
        version = $Version
        installed_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        project_root = $root
        backup = $backup
        changed = @(
            "mythos\neural\live.py",
            "mythos\neural\rithal_behavior_fix_v105_r3_3.py"
        )
        protected_hashes = $protectedAfter
        contract = [ordered]@{
            regime_classes = @("TREND_UP", "TREND_DOWN", "CHOP", "BREAKOUT", "PANIC")
            regime_class_count = 5
            true_posterior = "softmax(regime_logits)"
            rank_posterior = "softmax(true_posterior)"
            canonical_rank_score_preserved = $true
            risk_adjusted_side_retained = $true
            inference_replaced = $false
            rank_prewarm_replaced = $false
            feature_preparation_replaced = $false
            process_routing_replaced = $false
            trade_manager_authority_changed = $false
            checkpoints_changed = $false
        }
    }
    $recordPath = Join-Path $root "mythos_model_instances\rithal-1-0-contract-locked\rithal_behavior_v105_r3_3_install.json"
    New-Item -ItemType Directory -Path (Split-Path -Parent $recordPath) -Force | Out-Null
    $record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $recordPath -Encoding UTF8

    Write-Host ""
    Write-Host "[$Version] INSTALL PASS" -ForegroundColor Green
    Write-Host "Project: $root"
    Write-Host "Backup:  $backup"
    Write-Host ""
    Write-Host "Restart with your existing Start-Rithal18KContractV2.ps1 command."
    Write-Host "Expected boot marker: [$Version] installed: exact 5-class regime contract"
}
catch {
    Copy-Item -LiteralPath (Join-Path $backup "live.py") -Destination $live -Force
    $savedModule = Join-Path $backup "rithal_behavior_fix_v105_r3_3.py"
    if (Test-Path -LiteralPath $savedModule) {
        Copy-Item -LiteralPath $savedModule -Destination $moduleTarget -Force
    } elseif (Test-Path -LiteralPath $moduleTarget) {
        Remove-Item -LiteralPath $moduleTarget -Force
    }
    Write-Warning "[$Version] install failed; live.py and the R3.3 module were restored."
    throw
}
