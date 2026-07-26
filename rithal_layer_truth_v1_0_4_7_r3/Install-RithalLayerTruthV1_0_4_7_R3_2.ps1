param([string]$ProjectRoot = (Get-Location).Path)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Version = 'RITHAL_LAYER_TRUTH_V1_0_4_7_R3_2'
$Branch = 'rithal-layer-truth-v1.0.4.7-r3'
$RawBase = "https://raw.githubusercontent.com/carvedkarma/NeuralTrade/$Branch/rithal_layer_truth_v1_0_4_7_r3"
$Root = [System.IO.Path]::GetFullPath($ProjectRoot)
$Neural = Join-Path $Root 'mythos\neural'
$Runtime = Join-Path $Neural 'rithal_layer_truth_v1047.py'
$Dashboard = Join-Path $Neural 'dashboard.py'
$Settings = Join-Path $Root 'settings_override_v1.json'
$Control = Join-Path $Root 'mythos_5m_execution_control.json'
$StartFile = Join-Path $Root 'Start-RithalLayerTruthV1_0_4_7.ps1'
$StopFile = Join-Path $Root 'Stop-RithalLayerTruthV1_0_4_7.ps1'
$VerifyFile = Join-Path $Root 'Verify-RithalLayerTruthV1_0_4_7.ps1'
$RollbackFile = Join-Path $Root 'Rollback-RithalLayerTruthV1_0_4_7.ps1'
$PidFile = Join-Path $Root 'rithal_layer_truth_v1047.pid'
$Backup = Join-Path $Root ('rithal_layer_truth_backup_r32_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))

function Stage([string]$Text) { Write-Host "[$Version] $Text" -ForegroundColor Cyan }
function Set-Value($Object, [string]$Name, $Value, [switch]$OnlyIfMissing) {
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { $Object | Add-Member -MemberType NoteProperty -Name $Name -Value $Value }
    elseif (-not $OnlyIfMissing) { $Object.$Name = $Value }
}
function Write-Json([string]$Path, $Object) {
    $temp = "$Path.$PID.tmp"
    [IO.File]::WriteAllText($temp, (($Object | ConvertTo-Json -Depth 80) + [Environment]::NewLine), [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temp -Destination $Path -Force
}
function Restore([string]$ManifestPath) {
    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) { return }
    $manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    foreach ($item in @($manifest.files)) {
        if ($item.existed -and (Test-Path -LiteralPath $item.backup -PathType Leaf)) { Copy-Item -LiteralPath $item.backup -Destination $item.target -Force }
        elseif (-not $item.existed -and (Test-Path -LiteralPath $item.target)) { Remove-Item -LiteralPath $item.target -Force -ErrorAction SilentlyContinue }
    }
}

if (-not (Test-Path -LiteralPath $Neural -PathType Container)) { throw "Invalid Rithal root: $Root" }
if (-not (Test-Path -LiteralPath $Dashboard -PathType Leaf)) { throw "Dashboard missing: $Dashboard" }
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw 'python not found on PATH' }

if (Test-Path -LiteralPath $PidFile -PathType Leaf) {
    try { $oldPid = [int](Get-Content -LiteralPath $PidFile -Raw); Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue } catch {}
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Path $Backup -Force | Out-Null
$targets = @($Runtime,$Dashboard,$Settings,$Control,$StartFile,$StopFile,$VerifyFile,$RollbackFile)
$records = @()
for ($i=0; $i -lt $targets.Count; $i++) {
    $target = $targets[$i]
    $exists = Test-Path -LiteralPath $target -PathType Leaf
    $copy = Join-Path $Backup (('{0:D2}_' -f $i) + [IO.Path]::GetFileName($target))
    if ($exists) { Copy-Item -LiteralPath $target -Destination $copy -Force }
    $records += [pscustomobject]@{target=$target;existed=$exists;backup=$copy}
}
$Manifest = Join-Path $Backup 'backup_manifest.json'
Write-Json $Manifest ([pscustomobject]@{version=$Version;created_at=(Get-Date).ToUniversalTime().ToString('o');files=$records})

try {
    Stage 'Installing the R3 canonical truth runtime'
    Invoke-WebRequest -UseBasicParsing -Uri "$RawBase/rithal_layer_truth_v1047_r3.py" -OutFile $Runtime

    Stage 'Preserving control metadata and enforcing shadow-only paper scope'
    if (Test-Path -LiteralPath $Control -PathType Leaf) { $controlObject = Get-Content -LiteralPath $Control -Raw | ConvertFrom-Json } else { $controlObject = [pscustomobject]@{} }
    if ($null -eq $controlObject) { $controlObject = [pscustomobject]@{} }
    Set-Value $controlObject 'schema_version' 1 -OnlyIfMissing
    Set-Value $controlObject 'module' 'RITHAL_5M_POSITION_BRAIN_CONTROLLER_V1' -OnlyIfMissing
    Set-Value $controlObject 'mode' 'SHADOW_ONLY'
    Set-Value $controlObject 'execution_enabled' $false
    Set-Value $controlObject 'scope' 'PAPER_ONLY'
    Set-Value $controlObject 'updated_at' ((Get-Date).ToUniversalTime().ToString('o'))
    Set-Value $controlObject 'updated_by' $Version
    Write-Json $Control $controlObject

    Stage 'Publishing locked profile metadata without resetting realised equity'
    if (Test-Path -LiteralPath $Settings -PathType Leaf) { $settingsObject = Get-Content -LiteralPath $Settings -Raw | ConvertFrom-Json } else { $settingsObject = [pscustomobject]@{} }
    if ($null -eq $settingsObject) { $settingsObject = [pscustomobject]@{} }
    $contract = [pscustomobject]@{schema_version=3;module=$Version;instance_id='rithal-1-0-contract-locked';starting_equity_usd=18000.0;fixed_margin_pct=0.10;leverage=10;allocation_cap_pct=0.60;heat_cap_pct=0.03;max_open_positions=6;manager_mode='SHADOW_ONLY';reset_current_equity=$false;updated_at=(Get-Date).ToUniversalTime().ToString('o')}
    Set-Value $settingsObject 'rithal_layer_contract_v1047' $contract
    Write-Json $Settings $settingsObject

    Stage 'Adding the read-only dashboard route and aligning next-session policy'
    $text = [IO.File]::ReadAllText($Dashboard)
    $start = $text.IndexOf('DEFAULT_PAPER_POLICY = {')
    if ($start -ge 0) {
        $finish = $text.IndexOf("`n}",$start)
        if ($finish -gt $start) {
            $count = $finish-$start+2
            $policy = $text.Substring($start,$count)
            $policy = [regex]::Replace($policy,'(?m)^(\s*"per_trade_margin_pct"\s*:\s*)[^,]+,','${1}0.10,')
            $policy = [regex]::Replace($policy,'(?m)^(\s*"leverage"\s*:\s*)[^,]+,','${1}10,')
            $policy = [regex]::Replace($policy,'(?m)^(\s*"max_portfolio_margin_pct"\s*:\s*)[^,]+,','${1}0.60,')
            $policy = [regex]::Replace($policy,'(?m)^(\s*"max_concurrent_positions"\s*:\s*)[^,]+,','${1}6,')
            $text = $text.Remove($start,$count).Insert($start,$policy)
        }
    }
    $text = $text.Replace('truth_path = Path.cwd() / "mythos_rithal_intelligence_truth.json"','truth_path = Path(__file__).resolve().parents[2] / "mythos_rithal_intelligence_truth.json"')
    $marker = '# RITHAL_LAYER_TRUTH_V1_0_4_7_DASHBOARD_BRIDGE'
    if (-not $text.Contains($marker)) {
        $anchor = '    app = Flask(__name__)'
        if (-not $text.Contains($anchor)) { throw 'Dashboard Flask anchor not found' }
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
        $text = $text.Replace($anchor,$bridge)
    }
    [IO.File]::WriteAllText($Dashboard,$text,[Text.UTF8Encoding]::new($false))

    Stage 'Creating commands and rollback'
    $startBody = @'
param([string]$ProjectRoot=(Get-Location).Path)
$ErrorActionPreference='Stop';$root=[IO.Path]::GetFullPath($ProjectRoot);$runtime=Join-Path $root 'mythos\neural\rithal_layer_truth_v1047.py';$pidFile=Join-Path $root 'rithal_layer_truth_v1047.pid';$logFile=Join-Path $root 'rithal_layer_truth_v1047.log'
if(Test-Path $pidFile){try{$p0=[int](Get-Content $pidFile -Raw)}catch{$p0=0};if($p0 -gt 0 -and (Get-Process -Id $p0 -ErrorAction SilentlyContinue)){Write-Host "Already running PID=$p0";exit 0};Remove-Item $pidFile -Force -ErrorAction SilentlyContinue}
$p=Start-Process python -ArgumentList @($runtime,'--project-root',$root,'--interval','2') -WorkingDirectory $root -RedirectStandardOutput $logFile -RedirectStandardError ($logFile+'.err') -PassThru -WindowStyle Hidden;Start-Sleep -Milliseconds 500;if($p.HasExited){throw "Sidecar exited immediately; see $logFile.err"};Set-Content $pidFile $p.Id -Encoding ascii;Write-Host "R3 layer truth started PID=$($p.Id)"
'@
    [IO.File]::WriteAllText($StartFile,$startBody,[Text.UTF8Encoding]::new($false))
    $stopBody = @'
param([string]$ProjectRoot=(Get-Location).Path)
$root=[IO.Path]::GetFullPath($ProjectRoot);$pidFile=Join-Path $root 'rithal_layer_truth_v1047.pid';if(-not(Test-Path $pidFile)){Write-Host 'Not running';exit 0};try{$p=[int](Get-Content $pidFile -Raw);Stop-Process -Id $p -Force -ErrorAction SilentlyContinue}catch{};Remove-Item $pidFile -Force -ErrorAction SilentlyContinue;Write-Host 'R3 layer truth stopped'
'@
    [IO.File]::WriteAllText($StopFile,$stopBody,[Text.UTF8Encoding]::new($false))
    $verifyBody = @'
param([string]$ProjectRoot=(Get-Location).Path)
$ErrorActionPreference='Stop';$root=[IO.Path]::GetFullPath($ProjectRoot);python (Join-Path $root 'mythos\neural\rithal_layer_truth_v1047.py') --project-root $root --verify;if($LASTEXITCODE -ne 0){throw "R3 verification failed: $LASTEXITCODE"}
'@
    [IO.File]::WriteAllText($VerifyFile,$verifyBody,[Text.UTF8Encoding]::new($false))
    $rollbackBody = @"
`$ErrorActionPreference='Stop';if(Test-Path '$PidFile'){try{`$p=[int](Get-Content '$PidFile' -Raw);Stop-Process -Id `$p -Force -ErrorAction SilentlyContinue}catch{};Remove-Item '$PidFile' -Force -ErrorAction SilentlyContinue};`$m=Get-Content '$Manifest' -Raw|ConvertFrom-Json;foreach(`$i in @(`$m.files)){if(`$i.existed -and (Test-Path `$i.backup)){Copy-Item `$i.backup `$i.target -Force}elseif(-not `$i.existed -and (Test-Path `$i.target)){Remove-Item `$i.target -Force -ErrorAction SilentlyContinue}};Write-Host 'R3 rollback complete'
"@
    [IO.File]::WriteAllText($RollbackFile,$rollbackBody,[Text.UTF8Encoding]::new($false))

    Stage 'Compiling and verifying'
    & python -m py_compile $Runtime $Dashboard
    if ($LASTEXITCODE -ne 0) { throw "Compilation failed: $LASTEXITCODE" }
    & python $Runtime --project-root $Root --verify
    if ($LASTEXITCODE -ne 0) { throw "Verification failed: $LASTEXITCODE" }

    Write-Host "[$Version] INSTALLATION PASS" -ForegroundColor Green
    Write-Host 'Start: .\Start-RithalLayerTruthV1_0_4_7.ps1' -ForegroundColor Green
    Write-Host 'Verify: .\Verify-RithalLayerTruthV1_0_4_7.ps1' -ForegroundColor Green
    Write-Host 'API: http://127.0.0.1:8888/api/rithal-layer-truth' -ForegroundColor Green
    Write-Host "Backup: $Backup"
}
catch { Restore $Manifest; throw }
