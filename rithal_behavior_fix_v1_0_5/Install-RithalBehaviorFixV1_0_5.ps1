param(
    [Parameter(Mandatory=$false)][string]$ProjectRoot=(Get-Location).Path,
    [Parameter(Mandatory=$false)][ValidateSet('PAPER_CONTROL','SHADOW_ONLY')][string]$ManagerMode='PAPER_CONTROL'
)

$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest

$Version='RITHAL_BEHAVIOR_FIX_V1_0_5'
$Branch='rithal-behavior-fix-v1.0.5'
$RawBase="https://raw.githubusercontent.com/carvedkarma/NeuralTrade/$Branch/rithal_behavior_fix_v1_0_5"
$Root=[IO.Path]::GetFullPath($ProjectRoot)
$Neural=Join-Path $Root 'mythos\neural'
$Live=Join-Path $Neural 'live.py'
$Manager=Join-Path $Neural 'trade_manager_v6.py'
$Module=Join-Path $Neural 'rithal_behavior_fix_v105.py'
$Settings=Join-Path $Root 'settings_override_v1.json'
$Control=Join-Path $Root 'mythos_5m_execution_control.json'
$PaperPolicy=Join-Path $Neural 'mythos_paper_policy.json'
$ControllerConfig=Join-Path $Root 'mythos_model_instances\rithal-1-0-contract-locked\rithal_controller_config.json'
$VerifyFile=Join-Path $Root 'Verify-RithalBehaviorFixV1_0_5.ps1'
$RollbackFile=Join-Path $Root 'Rollback-RithalBehaviorFixV1_0_5.ps1'
$Stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$Backup=Join-Path $Root "rithal_behavior_fix_v105_backup_$Stamp"
$Manifest=Join-Path $Backup 'backup_manifest.json'
$InstallReport=Join-Path $Root 'rithal_behavior_fix_v105_install_report.json'

function Stage([string]$Text){Write-Host "[$Version] $Text" -ForegroundColor Cyan}
function Write-JsonAtomic([string]$Path,$Object){
    $temp="$Path.$PID.tmp"
    [IO.File]::WriteAllText($temp,(($Object|ConvertTo-Json -Depth 100)+[Environment]::NewLine),[Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temp -Destination $Path -Force
}
function File-Sha([string]$Path){
    if(Test-Path -LiteralPath $Path -PathType Leaf){return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()}
    return $null
}
function Checkpoint-Digest([string]$Dir){
    if(-not(Test-Path -LiteralPath $Dir -PathType Container)){return $null}
    $rows=Get-ChildItem -LiteralPath $Dir -File -Recurse | Sort-Object FullName | ForEach-Object {
        $rel=$_.FullName.Substring($Dir.Length).TrimStart('\','/')
        "$rel|$((Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant())"
    }
    $joined=[string]::Join("`n",$rows)
    $bytes=[Text.Encoding]::UTF8.GetBytes($joined)
    $sha=[Security.Cryptography.SHA256]::Create()
    try{return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-','').ToLowerInvariant()}finally{$sha.Dispose()}
}
function Restore-Backup([string]$ManifestPath){
    if(-not(Test-Path -LiteralPath $ManifestPath -PathType Leaf)){return}
    $m=Get-Content -LiteralPath $ManifestPath -Raw|ConvertFrom-Json
    foreach($item in @($m.files)){
        if($item.existed -and (Test-Path -LiteralPath $item.backup -PathType Leaf)){Copy-Item -LiteralPath $item.backup -Destination $item.target -Force}
        elseif(-not $item.existed -and (Test-Path -LiteralPath $item.target)){Remove-Item -LiteralPath $item.target -Force -ErrorAction SilentlyContinue}
    }
}
function Append-Hook([string]$Path,[string]$Marker,[string]$Hook,[string[]]$RequiredAnchors){
    $text=[IO.File]::ReadAllText($Path)
    foreach($anchor in $RequiredAnchors){if(-not $text.Contains($anchor)){throw "Required source anchor missing in $Path : $anchor"}}
    $count=([regex]::Matches($text,[regex]::Escape($Marker))).Count
    if($count -gt 1){throw "Duplicate behavior hook marker in $Path : $count"}
    if($count -eq 0){
        if(-not $text.EndsWith("`n")){ $text += "`r`n" }
        $text += "`r`n$Hook`r`n"
        [IO.File]::WriteAllText($Path,$text,[Text.UTF8Encoding]::new($false))
    }
}

if(-not(Test-Path -LiteralPath $Neural -PathType Container)){throw "Invalid project root: missing $Neural"}
foreach($required in @($Live,$Manager,(Join-Path $Neural 'rithal_runtime_settings.py'),(Join-Path $Neural 'rithal_manager_authority_contract.py'))){
    if(-not(Test-Path -LiteralPath $required -PathType Leaf)){throw "Required active source missing: $required"}
}
if(-not(Get-Command python -ErrorAction SilentlyContinue)){throw 'python was not found on PATH'}

New-Item -ItemType Directory -Path $Backup -Force|Out-Null
$targets=@($Live,$Manager,$Module,$Settings,$Control,$PaperPolicy,$ControllerConfig,$VerifyFile,$RollbackFile,$InstallReport)
$records=@()
for($i=0;$i -lt $targets.Count;$i++){
    $target=$targets[$i]
    $exists=Test-Path -LiteralPath $target -PathType Leaf
    $copy=Join-Path $Backup (('{0:D2}_' -f $i)+[IO.Path]::GetFileName($target))
    if($exists){Copy-Item -LiteralPath $target -Destination $copy -Force}
    $records += [pscustomobject]@{target=$target;existed=$exists;backup=$copy;sha256_before=(File-Sha $target)}
}
$checkpointDir=Join-Path $Root 'checkpoints\rithal_clean'
$checkpointBefore=Checkpoint-Digest $checkpointDir
Write-JsonAtomic $Manifest ([pscustomobject]@{version=$Version;created_at=(Get-Date).ToUniversalTime().ToString('o');checkpoint_digest_before=$checkpointBefore;files=$records})

try{
    Stage 'Downloading the behavior overlay'
    Invoke-WebRequest -UseBasicParsing -Uri "$RawBase/rithal_behavior_fix_v105.py" -OutFile $Module

    Stage 'Adding the engine hook after all existing overlays'
    $liveMarker='# RITHAL_BEHAVIOR_FIX_V1_0_5_LIVE_HOOK'
    $liveHook=@'
# RITHAL_BEHAVIOR_FIX_V1_0_5_LIVE_HOOK
try:
    from .rithal_behavior_fix_v105 import apply_live_patch as _rithal_v105_apply_live_patch
except ImportError:
    from rithal_behavior_fix_v105 import apply_live_patch as _rithal_v105_apply_live_patch
_rithal_v105_apply_live_patch(globals())
'@
    Append-Hook $Live $liveMarker $liveHook @('class NeuralV2Model','class LiveEngine','RITHAL_CONSOLIDATED_INTELLIGENCE_SAFETY')

    Stage 'Adding the single-manager hook after V3.3 and persistence overlays'
    $managerMarker='# RITHAL_BEHAVIOR_FIX_V1_0_5_MANAGER_HOOK'
    $managerHook=@'
# RITHAL_BEHAVIOR_FIX_V1_0_5_MANAGER_HOOK
try:
    from .rithal_behavior_fix_v105 import apply_trade_manager_patch as _rithal_v105_apply_trade_manager_patch
except ImportError:
    from rithal_behavior_fix_v105 import apply_trade_manager_patch as _rithal_v105_apply_trade_manager_patch
import sys as _rithal_v105_sys
_rithal_v105_apply_trade_manager_patch(_rithal_v105_sys.modules[__name__])
'@
    Append-Hook $Manager $managerMarker $managerHook @('class TradeManager','RITHAL_TRADE_MANAGER_V3_3_2','RITHAL_CONSOLIDATED_TRADE_MANAGER_PERSISTENCE')

    Stage 'Compiling modified source before changing authority settings'
    & python -m py_compile $Module $Live $Manager (Join-Path $Neural 'rithal_consolidated_v104.py') (Join-Path $Neural 'rithal_runtime_settings.py') (Join-Path $Neural 'rithal_manager_authority_contract.py')
    if($LASTEXITCODE -ne 0){throw "Python compilation failed: $LASTEXITCODE"}

    Stage 'Running deterministic behavior tests'
    $testOutput=& python $Module --self-test 2>&1
    if($LASTEXITCODE -ne 0){throw "Behavior self-test failed: $testOutput"}
    $testText=[string]::Join("`n",@($testOutput))
    if(-not $testText.Contains('"status": "PASS"')){throw "Behavior self-test did not report PASS: $testText"}

    Stage "Applying locked PAPER settings; manager mode=$ManagerMode"
    $configureOutput=& python $Module --configure --project-root $Root --instance-id 'rithal-1-0-contract-locked' --manager-mode $ManagerMode 2>&1
    if($LASTEXITCODE -ne 0){throw "Settings configuration failed: $configureOutput"}

    Stage 'Cross-verifying source hooks and checkpoint immutability'
    $liveText=[IO.File]::ReadAllText($Live)
    $managerText=[IO.File]::ReadAllText($Manager)
    if(([regex]::Matches($liveText,[regex]::Escape($liveMarker))).Count -ne 1){throw 'Live hook count is not exactly one'}
    if(([regex]::Matches($managerText,[regex]::Escape($managerMarker))).Count -ne 1){throw 'Manager hook count is not exactly one'}
    if($liveText.Contains('checkpoints\mythos_models') -and -not $liveText.Contains('checkpoints\rithal_clean')){
        Write-Warning 'Source contains a generic checkpoint reference; active launcher/manifest verification remains required.'
    }
    $checkpointAfter=Checkpoint-Digest $checkpointDir
    if($checkpointBefore -and $checkpointAfter -ne $checkpointBefore){throw "Checkpoint digest changed: $checkpointBefore -> $checkpointAfter"}

    Stage 'Creating verifier and rollback commands'
    $verifyContent=@'
param([string]$ProjectRoot=(Get-Location).Path)
$ErrorActionPreference='Stop';$root=[IO.Path]::GetFullPath($ProjectRoot);$neural=Join-Path $root 'mythos\neural';$module=Join-Path $neural 'rithal_behavior_fix_v105.py';$live=Join-Path $neural 'live.py';$manager=Join-Path $neural 'trade_manager_v6.py'
python -m py_compile $module $live $manager (Join-Path $neural 'rithal_consolidated_v104.py') (Join-Path $neural 'rithal_runtime_settings.py') (Join-Path $neural 'rithal_manager_authority_contract.py');if($LASTEXITCODE -ne 0){throw "compile failed: $LASTEXITCODE"}
python $module --self-test;if($LASTEXITCODE -ne 0){throw "self-test failed: $LASTEXITCODE"}
$lt=[IO.File]::ReadAllText($live);$mt=[IO.File]::ReadAllText($manager);if(([regex]::Matches($lt,[regex]::Escape('# RITHAL_BEHAVIOR_FIX_V1_0_5_LIVE_HOOK'))).Count -ne 1){throw 'live hook invalid'};if(([regex]::Matches($mt,[regex]::Escape('# RITHAL_BEHAVIOR_FIX_V1_0_5_MANAGER_HOOK'))).Count -ne 1){throw 'manager hook invalid'}
Write-Host '[RITHAL_BEHAVIOR_FIX_V1_0_5] VERIFICATION PASS' -ForegroundColor Green
'@
    [IO.File]::WriteAllText($VerifyFile,$verifyContent,[Text.UTF8Encoding]::new($false))
    $rollbackContent=@"
`$ErrorActionPreference='Stop';`$m=Get-Content -LiteralPath '$Manifest' -Raw|ConvertFrom-Json;foreach(`$item in @(`$m.files)){if(`$item.existed -and (Test-Path -LiteralPath `$item.backup)){Copy-Item -LiteralPath `$item.backup -Destination `$item.target -Force}elseif(-not `$item.existed -and (Test-Path -LiteralPath `$item.target)){Remove-Item -LiteralPath `$item.target -Force -ErrorAction SilentlyContinue}};Write-Host '[RITHAL_BEHAVIOR_FIX_V1_0_5] rollback complete; restart the engine.' -ForegroundColor Yellow
"@
    [IO.File]::WriteAllText($RollbackFile,$rollbackContent,[Text.UTF8Encoding]::new($false))

    $afterRows=@()
    foreach($target in $targets){$afterRows += [pscustomobject]@{path=$target;sha256=(File-Sha $target)}}
    $report=[pscustomobject]@{
        version=$Version;installed_at=(Get-Date).ToUniversalTime().ToString('o');project_root=$Root;instance_id='rithal-1-0-contract-locked';manager_mode=$ManagerMode
        hooks=[pscustomobject]@{live=1;manager=1};self_test='PASS';compile='PASS';checkpoint_digest_before=$checkpointBefore;checkpoint_digest_after=$checkpointAfter;checkpoint_unchanged=($checkpointBefore -eq $checkpointAfter)
        protected_contract=[pscustomobject]@{starting_equity=18000.0;per_trade_margin_pct=0.10;leverage=10;allocation_cap_pct=0.60;heat_cap_pct=0.03;live_exchange_control=$false;checkpoints='checkpoints\rithal_clean';feature_order_changed=$false;tp_sl_geometry_changed=$false;static_thresholds_changed=$false}
        changed_files=$afterRows;backup=$Backup;restart_required=$true;configure_output=[string]::Join("`n",@($configureOutput))
    }
    Write-JsonAtomic $InstallReport $report

    Write-Host "[$Version] INSTALLATION PASS" -ForegroundColor Green
    Write-Host "Backup: $Backup"
    Write-Host 'Verify: .\Verify-RithalBehaviorFixV1_0_5.ps1' -ForegroundColor Green
    Write-Host 'A controlled trading-engine restart is required before the behavior changes become active.' -ForegroundColor Yellow
}
catch{
    Write-Warning "[$Version] failed: $($_.Exception.Message)"
    Restore-Backup $Manifest
    throw
}
