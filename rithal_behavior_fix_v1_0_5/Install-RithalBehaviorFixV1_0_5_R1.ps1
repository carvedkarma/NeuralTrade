param(
    [Parameter(Mandatory=$false)][string]$ProjectRoot=(Get-Location).Path,
    [Parameter(Mandatory=$false)][ValidateSet('PAPER_CONTROL','SHADOW_ONLY')][string]$ManagerMode='PAPER_CONTROL'
)

$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest

$Version='RITHAL_BEHAVIOR_FIX_V1_0_5_R1'
$Branch='rithal-behavior-fix-v1.0.5'
$RawBase="https://raw.githubusercontent.com/carvedkarma/NeuralTrade/$Branch/rithal_behavior_fix_v1_0_5"
$Root=[IO.Path]::GetFullPath($ProjectRoot)
$Neural=Join-Path $Root 'mythos\neural'
$Live=Join-Path $Neural 'live.py'
$Manager=Join-Path $Neural 'trade_manager_v6.py'
$BaseModule=Join-Path $Neural 'rithal_behavior_fix_v105.py'
$R1Module=Join-Path $Neural 'rithal_behavior_fix_v105_r1.py'
$Settings=Join-Path $Root 'settings_override_v1.json'
$Control=Join-Path $Root 'mythos_5m_execution_control.json'
$PaperPolicy=Join-Path $Neural 'mythos_paper_policy.json'
$ControllerConfig=Join-Path $Root 'mythos_model_instances\rithal-1-0-contract-locked\rithal_controller_config.json'
$VerifyFile=Join-Path $Root 'Verify-RithalBehaviorFixV1_0_5_R1.ps1'
$RollbackFile=Join-Path $Root 'Rollback-RithalBehaviorFixV1_0_5_R1.ps1'
$InstallReport=Join-Path $Root 'rithal_behavior_fix_v105_r1_install_report.json'
$Stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$Backup=Join-Path $Root "rithal_behavior_fix_v105_r1_backup_$Stamp"
$Manifest=Join-Path $Backup 'backup_manifest.json'

function Stage([string]$Text){Write-Host "[$Version] $Text" -ForegroundColor Cyan}
function File-Sha([string]$Path){if(Test-Path -LiteralPath $Path -PathType Leaf){return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()};return $null}
function Write-JsonAtomic([string]$Path,$Object){$temp="$Path.$PID.tmp";[IO.File]::WriteAllText($temp,(($Object|ConvertTo-Json -Depth 100)+[Environment]::NewLine),[Text.UTF8Encoding]::new($false));Move-Item -LiteralPath $temp -Destination $Path -Force}
function Checkpoint-Digest([string]$Dir){
    if(-not(Test-Path -LiteralPath $Dir -PathType Container)){return $null}
    $rows=Get-ChildItem -LiteralPath $Dir -File -Recurse|Sort-Object FullName|ForEach-Object{
        $rel=$_.FullName.Substring($Dir.Length).TrimStart([char[]]'\/')
        "$rel|$((Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant())"
    }
    $bytes=[Text.Encoding]::UTF8.GetBytes([string]::Join("`n",$rows));$sha=[Security.Cryptography.SHA256]::Create()
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
function Install-Hook([string]$Path,[string]$Marker,[string]$Hook,[string[]]$RequiredAnchors){
    $text=[IO.File]::ReadAllText($Path)
    foreach($anchor in $RequiredAnchors){if(-not $text.Contains($anchor)){throw "Required source anchor missing in $Path : $anchor"}}
    # Remove the superseded base hook if a previous V1.0.5 attempt was installed.
    $basePattern='(?ms)\r?\n# RITHAL_BEHAVIOR_FIX_V1_0_5_(?:LIVE|MANAGER)_HOOK\r?\n.*?_rithal_v105_apply_(?:live_patch\(globals\(\)\)|trade_manager_patch\(_rithal_v105_sys\.modules\[__name__\]\))\r?\n'
    $text=[regex]::Replace($text,$basePattern,"`r`n")
    $count=([regex]::Matches($text,[regex]::Escape($Marker))).Count
    if($count -gt 1){throw "Duplicate R1 hook marker in $Path : $count"}
    if($count -eq 0){if(-not $text.EndsWith("`n")){$text+="`r`n"};$text+="`r`n$Hook`r`n"}
    [IO.File]::WriteAllText($Path,$text,[Text.UTF8Encoding]::new($false))
}

if(-not(Test-Path -LiteralPath $Neural -PathType Container)){throw "Invalid project root: $Root"}
$required=@($Live,$Manager,(Join-Path $Neural 'rithal_consolidated_v104.py'),(Join-Path $Neural 'rithal_runtime_settings.py'),(Join-Path $Neural 'rithal_manager_authority_contract.py'))
foreach($path in $required){if(-not(Test-Path -LiteralPath $path -PathType Leaf)){throw "Required active source missing: $path"}}
if(-not(Get-Command python -ErrorAction SilentlyContinue)){throw 'python was not found on PATH'}

New-Item -ItemType Directory -Path $Backup -Force|Out-Null
$targets=@($Live,$Manager,$BaseModule,$R1Module,$Settings,$Control,$PaperPolicy,$ControllerConfig,$VerifyFile,$RollbackFile,$InstallReport)
$records=@()
for($i=0;$i -lt $targets.Count;$i++){$target=$targets[$i];$exists=Test-Path -LiteralPath $target -PathType Leaf;$copy=Join-Path $Backup (('{0:D2}_' -f $i)+[IO.Path]::GetFileName($target));if($exists){Copy-Item -LiteralPath $target -Destination $copy -Force};$records+=[pscustomobject]@{target=$target;existed=$exists;backup=$copy;sha256_before=(File-Sha $target)}}
$CheckpointDir=Join-Path $Root 'checkpoints\rithal_clean'
$CheckpointBefore=Checkpoint-Digest $CheckpointDir
Write-JsonAtomic $Manifest ([pscustomobject]@{version=$Version;created_at=(Get-Date).ToUniversalTime().ToString('o');checkpoint_digest_before=$CheckpointBefore;files=$records})

try{
    Stage 'Downloading canonical base and R1 safety modules'
    Invoke-WebRequest -UseBasicParsing -Uri "$RawBase/rithal_behavior_fix_v105.py" -OutFile $BaseModule
    Invoke-WebRequest -UseBasicParsing -Uri "$RawBase/rithal_behavior_fix_v105_r1.py" -OutFile $R1Module

    Stage 'Installing one engine hook after all existing overlays'
    $LiveMarker='# RITHAL_BEHAVIOR_FIX_V1_0_5_R1_LIVE_HOOK'
    $LiveHook=@'
# RITHAL_BEHAVIOR_FIX_V1_0_5_R1_LIVE_HOOK
try:
    from .rithal_behavior_fix_v105_r1 import apply_live_patch as _rithal_v105_r1_apply_live_patch
except ImportError:
    from rithal_behavior_fix_v105_r1 import apply_live_patch as _rithal_v105_r1_apply_live_patch
_rithal_v105_r1_apply_live_patch(globals())
'@
    Install-Hook $Live $LiveMarker $LiveHook @('class NeuralV2Model','class LiveEngine','RITHAL_CONSOLIDATED_INTELLIGENCE_SAFETY')

    Stage 'Installing one single-manager hook after V3.3 and persistence overlays'
    $ManagerMarker='# RITHAL_BEHAVIOR_FIX_V1_0_5_R1_MANAGER_HOOK'
    $ManagerHook=@'
# RITHAL_BEHAVIOR_FIX_V1_0_5_R1_MANAGER_HOOK
try:
    from .rithal_behavior_fix_v105_r1 import apply_trade_manager_patch as _rithal_v105_r1_apply_trade_manager_patch
except ImportError:
    from rithal_behavior_fix_v105_r1 import apply_trade_manager_patch as _rithal_v105_r1_apply_trade_manager_patch
import sys as _rithal_v105_r1_sys
_rithal_v105_r1_apply_trade_manager_patch(_rithal_v105_r1_sys.modules[__name__])
'@
    Install-Hook $Manager $ManagerMarker $ManagerHook @('class TradeManager','RITHAL_TRADE_MANAGER_V3_3_2','RITHAL_CONSOLIDATED_TRADE_MANAGER_PERSISTENCE')

    Stage 'Compiling active engine, manager and contracts before authority changes'
    & python -m py_compile $BaseModule $R1Module $Live $Manager (Join-Path $Neural 'rithal_consolidated_v104.py') (Join-Path $Neural 'rithal_runtime_settings.py') (Join-Path $Neural 'rithal_manager_authority_contract.py')
    if($LASTEXITCODE -ne 0){throw "Python compilation failed: $LASTEXITCODE"}

    Stage 'Running deterministic R1 behavior tests'
    $TestOutput=& python $R1Module 2>&1
    if($LASTEXITCODE -ne 0){throw "R1 self-test failed: $([string]::Join(' ',@($TestOutput)))"}
    $TestText=[string]::Join("`n",@($TestOutput));if(-not $TestText.Contains('"status": "PASS"')){throw "R1 self-test did not report PASS: $TestText"}

    Stage "Applying locked PAPER profile and manager mode=$ManagerMode"
    $ConfigureOutput=& python $BaseModule --configure --project-root $Root --instance-id 'rithal-1-0-contract-locked' --manager-mode $ManagerMode 2>&1
    if($LASTEXITCODE -ne 0){throw "Settings configuration failed: $([string]::Join(' ',@($ConfigureOutput)))"}

    Stage 'Cross-verifying hook uniqueness, source policy and checkpoint immutability'
    $LiveText=[IO.File]::ReadAllText($Live);$ManagerText=[IO.File]::ReadAllText($Manager)
    if(([regex]::Matches($LiveText,[regex]::Escape($LiveMarker))).Count -ne 1){throw 'Live R1 hook count is not exactly one'}
    if(([regex]::Matches($ManagerText,[regex]::Escape($ManagerMarker))).Count -ne 1){throw 'Manager R1 hook count is not exactly one'}
    if(([regex]::Matches($LiveText,'RITHAL_BEHAVIOR_FIX_V1_0_5_LIVE_HOOK')).Count -ne 0){throw 'Superseded live hook remains'}
    if(([regex]::Matches($ManagerText,'RITHAL_BEHAVIOR_FIX_V1_0_5_MANAGER_HOOK')).Count -ne 0){throw 'Superseded manager hook remains'}
    $CheckpointAfter=Checkpoint-Digest $CheckpointDir
    if($CheckpointBefore -and $CheckpointAfter -ne $CheckpointBefore){throw "Checkpoint digest changed: $CheckpointBefore -> $CheckpointAfter"}

    Stage 'Creating verifier and rollback commands'
    $VerifyBody=@'
param([string]$ProjectRoot=(Get-Location).Path)
$ErrorActionPreference='Stop';$root=[IO.Path]::GetFullPath($ProjectRoot);$n=Join-Path $root 'mythos\neural';$b=Join-Path $n 'rithal_behavior_fix_v105.py';$r=Join-Path $n 'rithal_behavior_fix_v105_r1.py';$l=Join-Path $n 'live.py';$m=Join-Path $n 'trade_manager_v6.py'
python -m py_compile $b $r $l $m (Join-Path $n 'rithal_consolidated_v104.py') (Join-Path $n 'rithal_runtime_settings.py') (Join-Path $n 'rithal_manager_authority_contract.py');if($LASTEXITCODE -ne 0){throw "compile failed: $LASTEXITCODE"};python $r;if($LASTEXITCODE -ne 0){throw "self-test failed: $LASTEXITCODE"}
$lt=[IO.File]::ReadAllText($l);$mt=[IO.File]::ReadAllText($m);if(([regex]::Matches($lt,[regex]::Escape('# RITHAL_BEHAVIOR_FIX_V1_0_5_R1_LIVE_HOOK'))).Count -ne 1){throw 'live hook invalid'};if(([regex]::Matches($mt,[regex]::Escape('# RITHAL_BEHAVIOR_FIX_V1_0_5_R1_MANAGER_HOOK'))).Count -ne 1){throw 'manager hook invalid'};Write-Host '[RITHAL_BEHAVIOR_FIX_V1_0_5_R1] VERIFICATION PASS' -ForegroundColor Green
'@
    [IO.File]::WriteAllText($VerifyFile,$VerifyBody,[Text.UTF8Encoding]::new($false))
    $RollbackBody=@"
`$ErrorActionPreference='Stop';`$m=Get-Content -LiteralPath '$Manifest' -Raw|ConvertFrom-Json;foreach(`$item in @(`$m.files)){if(`$item.existed -and (Test-Path -LiteralPath `$item.backup)){Copy-Item -LiteralPath `$item.backup -Destination `$item.target -Force}elseif(-not `$item.existed -and (Test-Path -LiteralPath `$item.target)){Remove-Item -LiteralPath `$item.target -Force -ErrorAction SilentlyContinue}};Write-Host '[RITHAL_BEHAVIOR_FIX_V1_0_5_R1] rollback complete; restart the engine.' -ForegroundColor Yellow
"@
    [IO.File]::WriteAllText($RollbackFile,$RollbackBody,[Text.UTF8Encoding]::new($false))

    $Report=[pscustomobject]@{version=$Version;installed_at=(Get-Date).ToUniversalTime().ToString('o');instance_id='rithal-1-0-contract-locked';manager_mode=$ManagerMode;compile='PASS';self_test='PASS';hooks=[pscustomobject]@{live=1;manager=1};checkpoint_digest_before=$CheckpointBefore;checkpoint_digest_after=$CheckpointAfter;checkpoint_unchanged=($CheckpointBefore -eq $CheckpointAfter);protected_contract=[pscustomobject]@{starting_equity=18000.0;per_trade_margin_pct=0.10;leverage=10;allocation_cap_pct=0.60;heat_cap_pct=0.03;live_exchange_control=$false;feature_order_changed=$false;static_thresholds_changed=$false;tp_sl_geometry_changed=$false};backup=$Backup;restart_required=$true;configure_output=[string]::Join("`n",@($ConfigureOutput))}
    Write-JsonAtomic $InstallReport $Report

    Write-Host "[$Version] INSTALLATION PASS" -ForegroundColor Green
    Write-Host "Backup: $Backup"
    Write-Host 'Verify: .\Verify-RithalBehaviorFixV1_0_5_R1.ps1' -ForegroundColor Green
    Write-Host 'Controlled engine restart required to activate behavior changes.' -ForegroundColor Yellow
}catch{Write-Warning "[$Version] failed: $($_.Exception.Message)";Restore-Backup $Manifest;throw}
