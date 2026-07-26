param(
    [Parameter(Mandatory = $false)]
    [string]$ProjectRoot = (Get-Location).Path
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Version = 'RITHAL_LAYER_TRUTH_V1_0_4_7_R3'
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
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$Backup = Join-Path $Root "rithal_layer_truth_backup_r3_$Stamp"

function Write-Stage([string]$Message) {
    Write-Host "[$Version] $Message" -ForegroundColor Cyan
}

function Set-Property {
    param($Object, [string]$Name, $Value)
    if ($null -eq $Object.PSObject.Properties[$Name]) {
        $Object | Add-Member -MemberType NoteProperty -Name $Name -Value $Value
    } else {
        $Object.$Name = $Value
    }
}

function Write-JsonAtomic {
    param([string]$Path, $Value)
    $temp = "$Path.$PID.tmp"
    $json = $Value | ConvertTo-Json -Depth 80
    [System.IO.File]::WriteAllText($temp, $json + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temp -Destination $Path -Force
}

function Restore-Manifest {
    param([string]$ManifestPath)
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

if (-not (Test-Path -LiteralPath $Neural -PathType Container)) {
    throw "Invalid project root. Missing: $Neural"
}
if (-not (Test-Path -LiteralPath $Dashboard -PathType Leaf)) {
    throw "Dashboard backend missing: $Dashboard"
}
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw 'python was not found on PATH.'
}

if (Test-Path -LiteralPath $PidFile -PathType Leaf) {
    try {
        $oldPid = [int](Get-Content -LiteralPath $PidFile -Raw)
        if (Get-Process -Id $oldPid -ErrorAction SilentlyContinue) {
            Write-Stage "Stopping previous layer-truth sidecar PID=$oldPid"
            Stop-Process -Id $oldPid -Force -ErrorAction Stop
        }
    } catch {
        Write-Warning "Could not stop previous sidecar cleanly: $($_.Exception.Message)"
    }
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Path $Backup -Force | Out-Null
$targets = @($Runtime, $Dashboard, $Settings, $Control, $StartScript, $StopScript, $VerifyScript)
$rows = @()
for ($index = 0; $index -lt $targets.Count; $index++) {
    $target = $targets[$index]
    $exists = Test-Path -LiteralPath $target -PathType Leaf
    $backupName = ('{0:D2}_{1}' -f $index, [System.IO.Path]::GetFileName($target))
    $backupPath = Join-Path $Backup $backupName
    if ($exists) { Copy-Item -LiteralPath $target -Destination $backupPath -Force }
    $rows += [pscustomobject]@{ target = $target; existed = $exists; backup = $backupPath }
}
$ManifestPath = Join-Path $Backup 'backup_manifest.json'
Write-JsonAtomic -Path $ManifestPath -Value ([pscustomobject]@{
    version = $Version
    created_at = (Get-Date).ToUniversalTime().ToString('o')
    files = $rows
})

try {
    Write-Stage 'Downloading cross-verified R3 runtime'
    Invoke-WebRequest -UseBasicParsing -Uri "$RawBase/rithal_layer_truth_v1047_r3.py" -OutFile $Runtime

    Write-Stage 'Preserving the complete control document while enforcing SHADOW_ONLY / PAPER_ONLY'
    if (Test-Path -LiteralPath $Control -PathType Leaf) {
        try { $controlObject = Get-Content -LiteralPath $Control -Raw | ConvertFrom-Json }
        catch { throw "Existing control JSON is invalid: $($_.Exception.Message)" }
    } else {
        $controlObject = [pscustomobject]@{}
    }
    Set-Property $controlObject 'schema_version' 1
    Set-Property $controlObject 'module' 'RITHAL_5M_POSITION_BRAIN_CONTROLLER_V1'
    Set-Property $controlObject 'mode' 'SHADOW_ONLY'
    Set-Property $controlObject 'execution_enabled' $false
    Set-Property $controlObject 'scope' 'PAPER_ONLY'
    Set-Property $controlObject 'updated_at' ((Get-Date).ToUniversalTime().ToString('o'))
    Set-Property $controlObject 'updated_by' $Version
    Write-JsonAtomic -Path $Control -Value $controlObject

    Write-Stage 'Publishing the locked starting contract without resetting realised equity'
    if (Test-Path -LiteralPath $Settings -PathType Leaf) {
        try { $settingsObject = Get-Content -LiteralPath $Settings -Raw | ConvertFrom-Json }
        catch { throw "Existing settings JSON is invalid: $($_.Exception.Message)" }
    } else {
        $settingsObject = [pscustomobject]@{}
    }
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
        note = 'Starting contract metadata. Current realised paper equity is never reset by this installer.'
        updated_at = (Get-Date).ToUniversalTime().ToString('o')
    }
    Set-Property $settingsObject 'rithal_layer_contract_v1047' $contract
    Write-JsonAtomic -Path $Settings -Value $settingsObject

    Write-Stage 'Patching Dashboard 1 read-only projection and next-session paper policy'
    $dashboardText = [System.IO.File]::ReadAllText($Dashboard)
    $policyStart = $dashboardText.IndexOf('DEFAULT_PAPER_POLICY = {')
    if ($policyStart -ge 0) {
        $policyEnd = $dashboardText.IndexOf("`n}", $policyStart)
        if ($policyEnd -gt $policyStart) {
            $policyLength = $policyEnd - $policyStart + 2
            $policy = $dashboardText.Substring($policyStart, $policyLength)
            $policy = [regex]::Replace($policy, '(?m)^(\s*"per_trade_margin_pct"\s*:\s*)[^,]+,', '${1}0.10,')
            $policy = [regex]::Replace($policy, '(?m)^(\s*"leverage"\s*:\s*)[^,]+,', '${1}10,')
            $policy = [regex]::Replace($policy, '(?m)^(\s*"max_portfolio_margin_pct"\s*:\s*)[^,]+,', '${1}0.60,')
            $policy = [regex]::Replace($policy, '(?m)^(\s*"max_concurrent_positions"\s*:\s*)[^,]+,', '${1}6,')
            $dashboardText = $dashboardText.Remove($policyStart, $policyLength).Insert($policyStart, $policy)
        }
    }

    $oldTruthPath = 'truth_path = Path.cwd() / "mythos_rithal_intelligence_truth.json"'
    $newTruthPath = 'truth_path = Path(__file__).resolve().parents[2] / "mythos_rithal_intelligence_truth.json"'
    if ($dashboardText.Contains($oldTruthPath)) {
        $dashboardText = $dashboardText.Replace($oldTruthPath, $newTruthPath)
    }

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
    }
    $dashboardText = $dashboardText.Replace('"RITHAL_LAYER_TRUTH_V1_0_4_7"', '"RITHAL_LAYER_TRUTH_V1_0_4_7_R3"')
    $dashboardText = $dashboardText.Replace('"V1.0.4.7"', '"V1.0.4.7-R3"')
    [System.IO.File]::WriteAllText($Dashboard, $dashboardText, [System.Text.UTF8Encoding]::new($false))

    Write-Stage 'Creating runtime commands'
    $startContent = @'
param([string]$ProjectRoot=(Get-Location).Path)
$ErrorActionPreference='Stop'
$root=[System.IO.Path]::GetFullPath($ProjectRoot)
$runtime=Join-Path $root 'mythos\neural\rithal_layer_truth_v1047.py'
$pidFile=Join-Path $root 'rithal_layer_truth_v1047.pid'
$logFile=Join-Path $root 'rithal_layer_truth_v1047.log'
if(Test-Path $pidFile){
  try{$oldPid=[int](Get-Content $pidFile -Raw)}catch{$oldPid=0}
  if($oldPid -gt 0 -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)){Write-Host "Rithal layer truth already running PID=$oldPid";exit 0}
  Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}
$p=Start-Process -FilePath 'python' -ArgumentList @($runtime,'--project-root',$root,'--interval','2') -WorkingDirectory $root -RedirectStandardOutput $logFile -RedirectStandardError ($logFile+'.err') -PassThru -WindowStyle Hidden
Start-Sleep -Milliseconds 500
if($p.HasExited){throw "Rithal layer truth exited immediately with code $($p.ExitCode). See $logFile.err"}
Set-Content -LiteralPath $pidFile -Value $p.Id -Encoding ascii
Write-Host "RITHAL_LAYER_TRUTH_V1_0_4_7_R3 started PID=$($p.Id)"
'@
    [System.IO.File]::WriteAllText($StartScript, $startContent, [System.Text.UTF8Encoding]::new($false))

    $stopContent = @'
param([string]$ProjectRoot=(Get-Location).Path)
$root=[System.IO.Path]::GetFullPath($ProjectRoot)
$pidFile=Join-Path $root 'rithal_layer_truth_v1047.pid'
if(-not(Test-Path $pidFile)){Write-Host 'Rithal layer truth is not running.';exit 0}
try{$pidValue=[int](Get-Content $pidFile -Raw)}catch{$pidValue=0}
if($pidValue -gt 0){Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue}
Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
Write-Host "RITHAL_LAYER_TRUTH_V1_0_4_7_R3 stopped PID=$pidValue"
'@
    [System.IO.File]::WriteAllText($StopScript, $stopContent, [System.Text.UTF8Encoding]::new($false))

    $verifyContent = @'
param([string]$ProjectRoot=(Get-Location).Path)
$ErrorActionPreference='Stop'
$root=[System.IO.Path]::GetFullPath($ProjectRoot)
python (Join-Path $root 'mythos\neural\rithal_layer_truth_v1047.py') --project-root $root --verify
if($LASTEXITCODE -ne 0){throw "RITHAL_LAYER_TRUTH_V1_0_4_7_R3 verification failed with exit code $LASTEXITCODE"}
'@
    [System.IO.File]::WriteAllText($VerifyScript, $verifyContent, [System.Text.UTF8Encoding]::new($false))

    $rollbackContent = @"
`$ErrorActionPreference='Stop'
if(Test-Path -LiteralPath '$PidFile'){
  try{`$p=[int](Get-Content -LiteralPath '$PidFile' -Raw);Stop-Process -Id `$p -Force -ErrorAction SilentlyContinue}catch{}
  Remove-Item -LiteralPath '$PidFile' -Force -ErrorAction SilentlyContinue
}
`$manifest=Get-Content -LiteralPath '$ManifestPath' -Raw | ConvertFrom-Json
foreach(`$entry in @(`$manifest.files)){
  if(`$entry.existed -and (Test-Path -LiteralPath `$entry.backup)){Copy-Item -LiteralPath `$entry.backup -Destination `$entry.target -Force}
  elseif(-not `$entry.existed -and (Test-Path -LiteralPath `$entry.target)){Remove-Item -LiteralPath `$entry.target -Force -ErrorAction SilentlyContinue}
}
Write-Host 'RITHAL_LAYER_TRUTH_V1_0_4_7_R3 rollback complete.'
"@
    [System.IO.File]::WriteAllText($RollbackScript, $rollbackContent, [System.Text.UTF8Encoding]::new($false))

    Write-Stage 'Compiling modified Python files'
    & python -m py_compile $Runtime $Dashboard
    if ($LASTEXITCODE -ne 0) { throw "Python compilation failed with exit code $LASTEXITCODE" }

    Write-Stage 'Running source/state verifier'
    & python $Runtime --project-root $Root --verify
    if ($LASTEXITCODE -ne 0) { throw "Verification failed with exit code $LASTEXITCODE" }

    Write-Host "[$Version] INSTALLATION PASS" -ForegroundColor Green
    Write-Host "Start:  .\Start-RithalLayerTruthV1_0_4_7.ps1" -ForegroundColor Green
    Write-Host "Verify: .\Verify-RithalLayerTruthV1_0_4_7.ps1" -ForegroundColor Green
    Write-Host "API:    http://127.0.0.1:8888/api/rithal-layer-truth" -ForegroundColor Green
    Write-Host "Backup: $Backup" -ForegroundColor DarkGray
}
catch {
    Write-Warning "[$Version] installation failed: $($_.Exception.Message)"
    Restore-Manifest -ManifestPath $ManifestPath
    throw
}
