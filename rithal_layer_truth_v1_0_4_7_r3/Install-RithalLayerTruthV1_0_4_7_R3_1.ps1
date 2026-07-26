param(
    [Parameter(Mandatory = $false)]
    [string]$ProjectRoot = (Get-Location).Path
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Version = 'RITHAL_LAYER_TRUTH_V1_0_4_7_R3_1'
$Branch = 'rithal-layer-truth-v1.0.4.7-r3'
$RawBase = "https://raw.githubusercontent.com/carvedkarma/NeuralTrade/$Branch/rithal_layer_truth_v1_0_4_7_r3"
$Root = [System.IO.Path]::GetFullPath($ProjectRoot)
$Neural = Join-Path $Root 'mythos\neural'
$Runtime = Join-Path $Neural 'rithal_layer_truth_v1047.py'
$Dashboard = Join-Path $Neural 'dashboard.py'
$Settings = Join-Path $Root 'settings_override_v1.json'
$Control = Join-Path $Root 'mythos_5m_execution_control.json'
$StartScript = Join-Path $Root 'Start-RithalLayerTruthV1_0_4_7.ps1'
$StopScript = Join-Path $Root 'Stop-RithalLayerTruthV1_0_4_7.ps1'
$VerifyScript = Join-Path $Root 'Verify-RithalLayerTruthV1_0_4_7.ps1'
$RollbackScript = Join-Path $Root 'Rollback-RithalLayerTruthV1_0_4_7.ps1'
$PidFile = Join-Path $Root 'rithal_layer_truth_v1047.pid'
$Backup = Join-Path $Root ("rithal_layer_truth_backup_r31_" + (Get-Date -Format 'yyyyMMdd_HHmmss'))

function Stage([string]$Message) { Write-Host "[$Version] $Message" -ForegroundColor Cyan }

function Set-Property {
    param($Object, [string]$Name, $Value, [switch]$OnlyIfMissing)
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        $Object | Add-Member -MemberType NoteProperty -Name $Name -Value $Value
    } elseif (-not $OnlyIfMissing) {
        $Object.$Name = $Value
    }
}

function Write-JsonAtomic {
    param([string]$Path, $Value)
    $temp = "$Path.$PID.tmp"
    [System.IO.File]::WriteAllText($temp, (($Value | ConvertTo-Json -Depth 80) + [Environment]::NewLine), [System.Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temp -Destination $Path -Force
}

function Restore-Backup([string]$ManifestPath) {
    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) { return }
    $manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    foreach ($entry in @($manifest.files)) {
        if ($entry.existed -and (Test-Path -LiteralPath $entry.backup -PathType Leaf)) {
            Copy-Item -LiteralPath $entry.backup -Destination $entry.target -Force
        } elseif (-not $entry.existed -and (Test-Path -LiteralPath $entry.target)) {
            Remove-Item -LiteralPath $entry.target -Force -ErrorAction SilentlyContinue
        }
    }
}

if (-not (Test-Path -LiteralPath $Neural -PathType Container)) { throw "Invalid project root: $Root" }
if (-not (Test-Path -LiteralPath $Dashboard -PathType Leaf)) { throw "Dashboard missing: $Dashboard" }
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw 'python was not found on PATH.' }

if (Test-Path -LiteralPath $PidFile -PathType Leaf) {
    try {
        $oldPid = [int](Get-Content -LiteralPath $PidFile -Raw)
        if (Get-Process -Id $oldPid -ErrorAction SilentlyContinue) { Stop-Process -Id $oldPid -Force }
    } catch { Write-Warning "Previous sidecar stop warning: $($_.Exception.Message)" }
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Path $Backup -Force | Out-Null
$targets = @($Runtime, $Dashboard, $Settings, $Control, $StartScript, $StopScript, $VerifyScript, $RollbackScript)
$rows = @()
for ($i = 0; $i -lt $targets.Count; $i++) {
    $target = $targets[$i]
    $exists = Test-Path -LiteralPath $target -PathType Leaf
    $copy = Join-Path $Backup (('{0:D2}_' -f $i) + [System.IO.Path]::GetFileName($target))
    if ($exists) { Copy-Item -LiteralPath $target -Destination $copy -Force }
    $rows += [pscustomobject]@{ target = $target; existed = $exists; backup = $copy }
}
$ManifestPath = Join-Path $Backup 'backup_manifest.json'
Write-JsonAtomic $ManifestPath ([pscustomobject]@{ version = $Version; created_at = (Get-Date).ToUniversalTime().ToString('o'); files = $rows })

try {
    Stage 'Downloading R3 canonical runtime'
    Invoke-WebRequest -UseBasicParsing -Uri "$RawBase/rithal_layer_truth_v1047_r3.py" -OutFile $Runtime

    Stage 'Applying exact-thesis richness selector'
    $runtimeText = [System.IO.File]::ReadAllText($Runtime)
    $oldAppend = '            matches.append((timestamp_of(mapping), mapping, path))'
    $newAppend = @'
            richness = sum(1 for key in ("side", "decision_score", "expected_r_long", "expected_r_short", "p_win_long", "p_win_short", "mae_long", "mae_short", "long_edge", "short_edge", "true_regime_probs", "rank_regime_probs") if recursive_first(mapping, (key,)) is not None)
            matches.append((timestamp_of(mapping), richness, mapping, path))
'@.TrimEnd()
    $oldSort = @'
    matches.sort(key=lambda row: row[0].timestamp() if row[0] else 0.0, reverse=True)
    _, mapping, path = matches[0]
'@
    $newSort = @'
    matches.sort(key=lambda row: ((row[0].timestamp() if row[0] else 0.0), row[1]), reverse=True)
    _, _, mapping, path = matches[0]
'@
    if ($runtimeText.Contains($oldAppend)) { $runtimeText = $runtimeText.Replace($oldAppend, $newAppend) }
    if ($runtimeText.Contains($oldSort)) { $runtimeText = $runtimeText.Replace($oldSort, $newSort) }
    if (-not $runtimeText.Contains('matches.append((timestamp_of(mapping), richness, mapping, path))')) { throw 'Exact-thesis richness patch failed.' }
    [System.IO.File]::WriteAllText($Runtime, $runtimeText, [System.Text.UTF8Encoding]::new($false))

    Stage 'Preserving control schema and authority metadata while enforcing shadow-only mode'
    if (Test-Path -LiteralPath $Control -PathType Leaf) {
        try { $controlObject = Get-Content -LiteralPath $Control -Raw | ConvertFrom-Json }
        catch { throw "Invalid control JSON: $($_.Exception.Message)" }
        if ($null -eq $controlObject) { $controlObject = [pscustomobject]@{} }
    } else { $controlObject = [pscustomobject]@{} }
    Set-Property $controlObject 'schema_version' 1 -OnlyIfMissing
    Set-Property $controlObject 'module' 'RITHAL_5M_POSITION_BRAIN_CONTROLLER_V1' -OnlyIfMissing
    Set-Property $controlObject 'mode' 'SHADOW_ONLY'
    Set-Property $controlObject 'execution_enabled' $false
    Set-Property $controlObject 'scope' 'PAPER_ONLY'
    Set-Property $controlObject 'updated_at' ((Get-Date).ToUniversalTime().ToString('o'))
    Set-Property $controlObject 'updated_by' $Version
    Write-JsonAtomic $Control $controlObject

    Stage 'Publishing the locked starting profile without resetting current equity'
    if (Test-Path -LiteralPath $Settings -PathType Leaf) {
        try { $settingsObject = Get-Content -LiteralPath $Settings -Raw | ConvertFrom-Json }
        catch { throw "Invalid settings JSON: $($_.Exception.Message)" }
        if ($null -eq $settingsObject) { $settingsObject = [pscustomobject]@{} }
    } else { $settingsObject = [pscustomobject]@{} }
    $contract = [pscustomobject]@{
        schema_version = 3
        module = $Version
        instance_id = 'rithal-1-0-contract-locked'
        starting_equity_usd = 18000.0
        fixed_margin_pct = 0.10
        leverage = 10
        allocation_cap_pct = 0.60
        heat_cap_pct = 0.03
        max_open_positions = 6
        manager_mode = 'SHADOW_ONLY'
        reset_current_equity = $false
        note = 'Starting profile metadata only. Current realised equity is preserved.'
        updated_at = (Get-Date).ToUniversalTime().ToString('o')
    }
    Set-Property $settingsObject 'rithal_layer_contract_v1047' $contract
    Write-JsonAtomic $Settings $settingsObject

    Stage 'Patching Dashboard 1 read-only route and next-session policy'
    $dashboardText = [System.IO.File]::ReadAllText($Dashboard)
    $policyStart = $dashboardText.IndexOf('DEFAULT_PAPER_POLICY = {')
    if ($policyStart -ge 0) {
        $policyEnd = $dashboardText.IndexOf("`n}", $policyStart)
        if ($policyEnd -gt $policyStart) {
            $length = $policyEnd - $policyStart + 2
            $policy = $dashboardText.Substring($policyStart, $length)
            $policy = [regex]::Replace($policy, '(?m)^(\s*"per_trade_margin_pct"\s*:\s*)[^,]+,', '${1}0.10,')
            $policy = [regex]::Replace($policy, '(?m)^(\s*"leverage"\s*:\s*)[^,]+,', '${1}10,')
            $policy = [regex]::Replace($policy, '(?m)^(\s*"max_portfolio_margin_pct"\s*:\s*)[^,]+,', '${1}0.60,')
            $policy = [regex]::Replace($policy, '(?m)^(\s*"max_concurrent_positions"\s*:\s*)[^,]+,', '${1}6,')
            $dashboardText = $dashboardText.Remove($policyStart, $length).Insert($policyStart, $policy)
        }
    }
    $dashboardText = $dashboardText.Replace('truth_path = Path.cwd() / "mythos_rithal_intelligence_truth.json"', 'truth_path = Path(__file__).resolve().parents[2] / "mythos_rithal_intelligence_truth.json"')
    $marker = '# RITHAL_LAYER_TRUTH_V1_0_4_7_DASHBOARD_BRIDGE'
    if (-not $dashboardText.Contains($marker)) {
        $anchor = '    app = Flask(__name__)'
        if (-not $dashboardText.Contains($anchor)) { throw 'Dashboard Flask anchor not found.' }
        $bridge = @'
    app = Flask(__name__)
    # RITHAL_LAYER_TRUTH_V1_0_4_7_DASHBOARD_BRIDGE
    @app.get("/api/rithal-layer-truth")
    def rithal_layer_truth_v1047():
        truth_path = Path(__file__).resolve().parents[2] / "mythos_rithal_intelligence_truth.json"
        if not truth_path.is_file():
            return jsonify({"ok": False, "version": "RITHAL_LAYER_TRUTH_V1_0_4_7_R3", "status": "UNAVAILABLE", "error": "TRUTH_SIDECAR_NOT_RUNNING"}), 503
        try:
            payload = json.loads(truth_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            return jsonify({"ok": False, "version": "RITHAL_LAYER_TRUTH_V1_0_4_7_R3", "status": "UNAVAILABLE", "error": str(exc)}), 503
        response = jsonify({"ok": True, "layer_truth": payload})
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Rithal-Layer-Truth"] = "V1.0.4.7-R3"
        return response
'@
        $dashboardText = $dashboardText.Replace($anchor, $bridge)
    } else {
        $dashboardText = $dashboardText.Replace('"version": "RITHAL_LAYER_TRUTH_V1_0_4_7"', '"version": "RITHAL_LAYER_TRUTH_V1_0_4_7_R3"')
        $dashboardText = $dashboardText.Replace('response.headers["X-Rithal-Layer-Truth"] = "V1.0.4.7"', 'response.headers["X-Rithal-Layer-Truth"] = "V1.0.4.7-R3"')
    }
    [System.IO.File]::WriteAllText($Dashboard, $dashboardText, [System.Text.UTF8Encoding]::new($false))

    Stage 'Creating start, stop, verify, and rollback commands'
    $startContent = @'
param([string]$ProjectRoot=(Get-Location).Path)
$ErrorActionPreference='Stop'
$root=[System.IO.Path]::GetFullPath($ProjectRoot)
$runtime=Join-Path $root 'mythos\neural\rithal_layer_truth_v1047.py'
$pidFile=Join-Path $root 'rithal_layer_truth_v1047.pid'
$logFile=Join-Path $root 'rithal_layer_truth_v1047.log'
if(Test-Path $pidFile){try{$oldPid=[int](Get-Content $pidFile -Raw)}catch{$oldPid=0};if($oldPid -gt 0 -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)){Write-Host "Rithal layer truth already running PID=$oldPid";exit 0};Remove-Item $pidFile -Force -ErrorAction SilentlyContinue}
$p=Start-Process -FilePath 'python' -ArgumentList @($runtime,'--project-root',$root,'--interval','2') -WorkingDirectory $root -RedirectStandardOutput $logFile -RedirectStandardError ($logFile+'.err') -PassThru -WindowStyle Hidden
Start-Sleep -Milliseconds 500
if($p.HasExited){throw "Layer truth exited immediately. See $logFile.err"}
Set-Content -LiteralPath $pidFile -Value $p.Id -Encoding ascii
Write-Host "RITHAL_LAYER_TRUTH_V1_0_4_7_R3 started PID=$($p.Id)"
'@
    [System.IO.File]::WriteAllText($StartScript, $startContent, [System.Text.UTF8Encoding]::new($false))

    $stopContent = @'
param([string]$ProjectRoot=(Get-Location).Path)
$root=[System.IO.Path]::GetFullPath($ProjectRoot);$pidFile=Join-Path $root 'rithal_layer_truth_v1047.pid'
if(-not(Test-Path $pidFile)){Write-Host 'Rithal layer truth is not running.';exit 0}
try{$pidValue=[int](Get-Content $pidFile -Raw)}catch{$pidValue=0};if($pidValue -gt 0){Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue};Remove-Item $pidFile -Force -ErrorAction SilentlyContinue;Write-Host "RITHAL_LAYER_TRUTH_V1_0_4_7_R3 stopped PID=$pidValue"
'@
    [System.IO.File]::WriteAllText($StopScript, $stopContent, [System.Text.UTF8Encoding]::new($false))

    $verifyContent = @'
param([string]$ProjectRoot=(Get-Location).Path)
$ErrorActionPreference='Stop';$root=[System.IO.Path]::GetFullPath($ProjectRoot)
python (Join-Path $root 'mythos\neural\rithal_layer_truth_v1047.py') --project-root $root --verify
if($LASTEXITCODE -ne 0){throw "RITHAL_LAYER_TRUTH_V1_0_4_7_R3 verification failed with exit code $LASTEXITCODE"}
'@
    [System.IO.File]::WriteAllText($VerifyScript, $verifyContent, [System.Text.UTF8Encoding]::new($false))

    $rollbackContent = @"
`$ErrorActionPreference='Stop'
if(Test-Path -LiteralPath '$PidFile'){try{`$p=[int](Get-Content -LiteralPath '$PidFile' -Raw);Stop-Process -Id `$p -Force -ErrorAction SilentlyContinue}catch{};Remove-Item -LiteralPath '$PidFile' -Force -ErrorAction SilentlyContinue}
`$manifest=Get-Content -LiteralPath '$ManifestPath' -Raw | ConvertFrom-Json
foreach(`$entry in @(`$manifest.files)){if(`$entry.existed -and (Test-Path -LiteralPath `$entry.backup)){Copy-Item -LiteralPath `$entry.backup -Destination `$entry.target -Force}elseif(-not `$entry.existed -and (Test-Path -LiteralPath `$entry.target)){Remove-Item -LiteralPath `$entry.target -Force -ErrorAction SilentlyContinue}}
Write-Host 'RITHAL_LAYER_TRUTH_V1_0_4_7_R3 rollback complete.'
"@
    [System.IO.File]::WriteAllText($RollbackScript, $rollbackContent, [System.Text.UTF8Encoding]::new($false))

    Stage 'Compiling runtime and Dashboard 1'
    & python -m py_compile $Runtime $Dashboard
    if ($LASTEXITCODE -ne 0) { throw "Compilation failed with exit code $LASTEXITCODE" }

    Stage 'Running runtime/source verification'
    & python $Runtime --project-root $Root --verify
    if ($LASTEXITCODE -ne 0) { throw "Verification failed with exit code $LASTEXITCODE" }

    Write-Host "[$Version] INSTALLATION PASS" -ForegroundColor Green
    Write-Host 'Start:  .\Start-RithalLayerTruthV1_0_4_7.ps1' -ForegroundColor Green
    Write-Host 'Verify: .\Verify-RithalLayerTruthV1_0_4_7.ps1' -ForegroundColor Green
    Write-Host 'API:    http://127.0.0.1:8888/api/rithal-layer-truth' -ForegroundColor Green
    Write-Host "Backup: $Backup" -ForegroundColor DarkGray
}
catch {
    Write-Warning "[$Version] failed: $($_.Exception.Message)"
    Restore-Backup $ManifestPath
    throw
}
