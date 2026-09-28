param(
    [Parameter(Mandatory = $false)]
    [string]$ProjectRoot = (Get-Location).Path
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Version = 'RITHAL_LAYER_TRUTH_V1_0_4_7'
$Branch = 'rithal-layer-truth-v1.0.4.7'
$RawBase = "https://raw.githubusercontent.com/carvedkarma/NeuralTrade/$Branch/rithal_layer_truth_v1_0_4_7"
$Root = [System.IO.Path]::GetFullPath($ProjectRoot)
$Neural = Join-Path $Root 'mythos\neural'
$InstallDir = Join-Path $Root 'rithal_layer_truth_v1_0_4_7'
$RuntimeTarget = Join-Path $Neural 'rithal_layer_truth_v1047.py'
$Dashboard = Join-Path $Neural 'dashboard.py'
$Settings = Join-Path $Root 'settings_override_v1.json'
$Control = Join-Path $Root 'mythos_5m_execution_control.json'
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$Backup = Join-Path $Root "rithal_layer_truth_backup_$Stamp"

function Write-Stage([string]$Message) {
    Write-Host "[$Version] $Message" -ForegroundColor Cyan
}

function Set-JsonProperty {
    param(
        [Parameter(Mandatory = $true)]$Object,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)]$Value
    )
    $existing = $Object.PSObject.Properties[$Name]
    if ($null -eq $existing) {
        $Object | Add-Member -MemberType NoteProperty -Name $Name -Value $Value
    } else {
        $Object.$Name = $Value
    }
}

function Write-JsonAtomic {
    param([string]$Path, $Value)
    $temp = "$Path.tmp"
    $json = $Value | ConvertTo-Json -Depth 64
    [System.IO.File]::WriteAllText($temp, $json + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temp -Destination $Path -Force
}

function Restore-Backups {
    if (-not (Test-Path -LiteralPath $Backup)) { return }
    Write-Warning "Restoring backup from $Backup"
    $manifestPath = Join-Path $Backup 'backup_manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath)) { return }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    foreach ($entry in @($manifest.files)) {
        if ($entry.existed -and (Test-Path -LiteralPath $entry.backup)) {
            Copy-Item -LiteralPath $entry.backup -Destination $entry.target -Force
        } elseif (-not $entry.existed -and (Test-Path -LiteralPath $entry.target)) {
            Remove-Item -LiteralPath $entry.target -Force
        }
    }
}

if (-not (Test-Path -LiteralPath $Neural -PathType Container)) {
    throw "Invalid Rithal project root. Missing: $Neural"
}
if (-not (Test-Path -LiteralPath $Dashboard -PathType Leaf)) {
    throw "Dashboard backend not found: $Dashboard"
}

New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
New-Item -ItemType Directory -Path $Backup -Force | Out-Null

$targets = @($RuntimeTarget, $Dashboard, $Settings, $Control)
$backupRows = @()
foreach ($target in $targets) {
    $exists = Test-Path -LiteralPath $target -PathType Leaf
    $backupPath = Join-Path $Backup ([System.IO.Path]::GetFileName($target))
    if ($exists) { Copy-Item -LiteralPath $target -Destination $backupPath -Force }
    $backupRows += [pscustomobject]@{ target = $target; existed = $exists; backup = $backupPath }
}
Write-JsonAtomic -Path (Join-Path $Backup 'backup_manifest.json') -Value ([pscustomobject]@{
    version = $Version
    created_at = (Get-Date).ToUniversalTime().ToString('o')
    files = $backupRows
})

try {
    Write-Stage 'Downloading canonical layer-truth runtime'
    Invoke-WebRequest -UseBasicParsing -Uri "$RawBase/rithal_layer_truth_v1047.py" -OutFile $RuntimeTarget

    Write-Stage 'Locking the manager control file to SHADOW_ONLY / PAPER_ONLY'
    $controlObject = [pscustomobject]@{
        schema_version = 1
        module = 'RITHAL_5M_POSITION_BRAIN_CONTROLLER_V1'
        mode = 'SHADOW_ONLY'
        execution_enabled = $false
        scope = 'PAPER_ONLY'
        updated_at = (Get-Date).ToUniversalTime().ToString('o')
        updated_by = $Version
    }
    Write-JsonAtomic -Path $Control -Value $controlObject

    Write-Stage 'Publishing the locked 18K / 10% / 10x contract without resetting the running ledger'
    if (Test-Path -LiteralPath $Settings -PathType Leaf) {
        try {
            $settingsObject = Get-Content -LiteralPath $Settings -Raw | ConvertFrom-Json
        } catch {
            throw "Existing settings_override_v1.json is invalid JSON: $($_.Exception.Message)"
        }
    } else {
        $settingsObject = [pscustomobject]@{}
    }
    $contract = [pscustomobject]@{
        schema_version = 1
        module = $Version
        instance_id = 'rithal-1-0-contract-locked'
        paper_equity_usd = 18000.0
        fixed_margin_pct = 0.10
        leverage = 10
        allocation_cap_pct = 0.60
        heat_cap_pct = 0.03
        max_open_positions = 2
        manager_mode = 'SHADOW_ONLY'
        reset_current_equity = $false
        note = 'Contract metadata only. The current realised paper wallet is never reset by this installer.'
        updated_at = (Get-Date).ToUniversalTime().ToString('o')
    }
    Set-JsonProperty -Object $settingsObject -Name 'rithal_layer_contract_v1047' -Value $contract
    Write-JsonAtomic -Path $Settings -Value $settingsObject

    Write-Stage 'Aligning the dashboard next-session paper policy and exposing canonical layer truth'
    $dashboardText = [System.IO.File]::ReadAllText($Dashboard)
    $marker = '# RITHAL_LAYER_TRUTH_V1_0_4_7_DASHBOARD_BRIDGE'
    if ($dashboardText -notmatch [regex]::Escape($marker)) {
        $oldPolicy = @'
    "per_trade_margin_pct": 0.10,
    "leverage": 5,
    "max_portfolio_margin_pct": 0.20,
    "max_concurrent_positions": 2,
'@
        $newPolicy = @'
    "per_trade_margin_pct": 0.10,
    "leverage": 10,
    "max_portfolio_margin_pct": 0.60,
    "max_concurrent_positions": 2,
'@
        if ($dashboardText.Contains($oldPolicy)) {
            $dashboardText = $dashboardText.Replace($oldPolicy, $newPolicy)
        }

        $anchor = '    app = Flask(__name__)'
        if (-not $dashboardText.Contains($anchor)) {
            throw 'Dashboard patch anchor not found: app = Flask(__name__)'
        }
        $bridge = @'
    app = Flask(__name__)
    # RITHAL_LAYER_TRUTH_V1_0_4_7_DASHBOARD_BRIDGE
    @app.get("/api/rithal-layer-truth")
    def rithal_layer_truth_v1047():
        truth_path = Path.cwd() / "mythos_rithal_intelligence_truth.json"
        if not truth_path.is_file():
            return jsonify({
                "ok": False,
                "version": "RITHAL_LAYER_TRUTH_V1_0_4_7",
                "status": "UNAVAILABLE",
                "error": "TRUTH_SIDECAR_NOT_RUNNING",
            }), 503
        try:
            payload = json.loads(truth_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            return jsonify({
                "ok": False,
                "version": "RITHAL_LAYER_TRUTH_V1_0_4_7",
                "status": "UNAVAILABLE",
                "error": str(exc),
            }), 503
        response = jsonify({"ok": True, "layer_truth": payload})
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Rithal-Layer-Truth"] = "V1.0.4.7"
        return response
'@
        $dashboardText = $dashboardText.Replace($anchor, $bridge)
        [System.IO.File]::WriteAllText($Dashboard, $dashboardText, [System.Text.UTF8Encoding]::new($false))
    }

    Write-Stage 'Creating start, stop, verify, and rollback commands'
    $startScript = @'
param([string]$ProjectRoot=(Get-Location).Path)
$ErrorActionPreference='Stop'
$root=[System.IO.Path]::GetFullPath($ProjectRoot)
$script=Join-Path $root 'mythos\neural\rithal_layer_truth_v1047.py'
$pidFile=Join-Path $root 'rithal_layer_truth_v1047.pid'
$logFile=Join-Path $root 'rithal_layer_truth_v1047.log'
if(Test-Path $pidFile){
  $oldPid=[int](Get-Content $pidFile -Raw)
  if(Get-Process -Id $oldPid -ErrorAction SilentlyContinue){ Write-Host "Rithal layer truth already running PID=$oldPid"; exit 0 }
}
$p=Start-Process -FilePath 'python' -ArgumentList @($script,'--project-root',$root,'--interval','2') -WorkingDirectory $root -RedirectStandardOutput $logFile -RedirectStandardError ($logFile+'.err') -PassThru -WindowStyle Hidden
Set-Content -LiteralPath $pidFile -Value $p.Id -Encoding ascii
Write-Host "RITHAL_LAYER_TRUTH_V1_0_4_7 started PID=$($p.Id)"
'@
    [System.IO.File]::WriteAllText((Join-Path $Root 'Start-RithalLayerTruthV1_0_4_7.ps1'), $startScript, [System.Text.UTF8Encoding]::new($false))

    $stopScript = @'
param([string]$ProjectRoot=(Get-Location).Path)
$root=[System.IO.Path]::GetFullPath($ProjectRoot)
$pidFile=Join-Path $root 'rithal_layer_truth_v1047.pid'
if(-not(Test-Path $pidFile)){Write-Host 'Rithal layer truth is not running.';exit 0}
$pidValue=[int](Get-Content $pidFile -Raw)
Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue
Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
Write-Host "RITHAL_LAYER_TRUTH_V1_0_4_7 stopped PID=$pidValue"
'@
    [System.IO.File]::WriteAllText((Join-Path $Root 'Stop-RithalLayerTruthV1_0_4_7.ps1'), $stopScript, [System.Text.UTF8Encoding]::new($false))

    $verifyScript = @'
param([string]$ProjectRoot=(Get-Location).Path)
$ErrorActionPreference='Stop'
$root=[System.IO.Path]::GetFullPath($ProjectRoot)
python (Join-Path $root 'mythos\neural\rithal_layer_truth_v1047.py') --project-root $root --verify
if($LASTEXITCODE -ne 0){throw "RITHAL_LAYER_TRUTH_V1_0_4_7 verification failed with exit code $LASTEXITCODE"}
'@
    [System.IO.File]::WriteAllText((Join-Path $Root 'Verify-RithalLayerTruthV1_0_4_7.ps1'), $verifyScript, [System.Text.UTF8Encoding]::new($false))

    $rollbackScript = @"
`$ErrorActionPreference='Stop'
`$manifest=Get-Content -LiteralPath '$Backup\backup_manifest.json' -Raw | ConvertFrom-Json
foreach(`$entry in @(`$manifest.files)){
  if(`$entry.existed -and (Test-Path -LiteralPath `$entry.backup)){Copy-Item -LiteralPath `$entry.backup -Destination `$entry.target -Force}
  elseif(-not `$entry.existed -and (Test-Path -LiteralPath `$entry.target)){Remove-Item -LiteralPath `$entry.target -Force}
}
Write-Host 'RITHAL_LAYER_TRUTH_V1_0_4_7 rollback complete.'
"@
    [System.IO.File]::WriteAllText((Join-Path $Root 'Rollback-RithalLayerTruthV1_0_4_7.ps1'), $rollbackScript, [System.Text.UTF8Encoding]::new($false))

    Write-Stage 'Compiling modified Python files'
    & python -m py_compile $RuntimeTarget $Dashboard
    if ($LASTEXITCODE -ne 0) { throw "Python compilation failed with exit code $LASTEXITCODE" }

    Write-Stage 'Running canonical verifier'
    & python $RuntimeTarget --project-root $Root --verify
    if ($LASTEXITCODE -ne 0) { throw "Runtime verification failed with exit code $LASTEXITCODE" }

    Write-Host "[$Version] INSTALLATION PASS" -ForegroundColor Green
    Write-Host "Start:  .\Start-RithalLayerTruthV1_0_4_7.ps1" -ForegroundColor Green
    Write-Host "Verify: .\Verify-RithalLayerTruthV1_0_4_7.ps1" -ForegroundColor Green
    Write-Host "API:    http://127.0.0.1:8888/api/rithal-layer-truth" -ForegroundColor Green
    Write-Host "Backup: $Backup" -ForegroundColor DarkGray
}
catch {
    Write-Error "[$Version] INSTALLATION FAILED: $($_.Exception.Message)"
    Restore-Backups
    throw
}
