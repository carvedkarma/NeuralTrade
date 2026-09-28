param(
    [Parameter(Mandatory=$false)][string]$ProjectRoot=(Get-Location).Path,
    [Parameter(Mandatory=$false)][ValidateSet('PAPER_CONTROL','SHADOW_ONLY')][string]$ManagerMode='PAPER_CONTROL'
)

$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest

$Version='RITHAL_BEHAVIOR_FIX_V1_0_5_R3'
$Branch='rithal-behavior-fix-v1.0.5'
$RawBase="https://raw.githubusercontent.com/carvedkarma/NeuralTrade/$Branch/rithal_behavior_fix_v1_0_5"
$Root=[IO.Path]::GetFullPath($ProjectRoot)
$Neural=Join-Path $Root 'mythos\neural'
$Live=Join-Path $Neural 'live.py'
$Manager=Join-Path $Neural 'trade_manager_v6.py'
$BaseModule=Join-Path $Neural 'rithal_behavior_fix_v105.py'
$R3Module=Join-Path $Neural 'rithal_behavior_fix_v105_r3.py'
$Settings=Join-Path $Root 'settings_override_v1.json'
$Control=Join-Path $Root 'mythos_5m_execution_control.json'
$PaperPolicy=Join-Path $Neural 'mythos_paper_policy.json'
$ControllerConfig=Join-Path $Root 'mythos_model_instances\rithal-1-0-contract-locked\rithal_controller_config.json'
$VerifyFile=Join-Path $Root 'Verify-RithalBehaviorFixV1_0_5_R3.ps1'
$RollbackFile=Join-Path $Root 'Rollback-RithalBehaviorFixV1_0_5_R3.ps1'
$InstallReport=Join-Path $Root 'rithal_behavior_fix_v105_r3_install_report.json'
$CheckpointDir=Join-Path $Root 'checkpoints\rithal_clean'
$Backup=Join-Path $Root ('rithal_behavior_fix_v105_r3_backup_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
$Manifest=Join-Path $Backup 'backup_manifest.json'

function Stage([string]$Text){Write-Host "[$Version] $Text" -ForegroundColor Cyan}
function File-Sha([string]$Path){if(Test-Path -LiteralPath $Path -PathType Leaf){return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()};return $null}
function Write-JsonAtomic([string]$Path,$Object){$temp="$Path.$PID.tmp";[IO.File]::WriteAllText($temp,(($Object|ConvertTo-Json -Depth 100)+[Environment]::NewLine),[Text.UTF8Encoding]::new($false));Move-Item -LiteralPath $temp -Destination $Path -Force}
function Checkpoint-Digest([string]$Dir){
    if(-not(Test-Path -LiteralPath $Dir -PathType Container)){throw "Checkpoint directory missing: $Dir"}
    $files=@(Get-ChildItem -LiteralPath $Dir -File -Recurse|Sort-Object FullName)
    if($files.Count -eq 0){throw "Checkpoint directory is empty: $Dir"}
    $rows=$files|ForEach-Object{
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
function Install-R3Hook([string]$Path,[string]$Marker,[string]$Hook,[string[]]$RequiredAnchors){
    $text=[IO.File]::ReadAllText($Path)
    foreach($anchor in $RequiredAnchors){if(-not $text.Contains($anchor)){throw "Required source anchor missing in $Path : $anchor"}}
    $patterns=@(
        '(?ms)\r?\n# RITHAL_BEHAVIOR_FIX_V1_0_5_LIVE_HOOK\r?\n.*?_rithal_v105_apply_live_patch\(globals\(\)\)\r?\n',
        '(?ms)\r?\n# RITHAL_BEHAVIOR_FIX_V1_0_5_MANAGER_HOOK\r?\n.*?_rithal_v105_apply_trade_manager_patch\(_rithal_v105_sys\.modules\[__name__\]\)\r?\n',
        '(?ms)\r?\n# RITHAL_BEHAVIOR_FIX_V1_0_5_R1_LIVE_HOOK\r?\n.*?_rithal_v105_r1_apply_live_patch\(globals\(\)\)\r?\n',
        '(?ms)\r?\n# RITHAL_BEHAVIOR_FIX_V1_0_5_R1_MANAGER_HOOK\r?\n.*?_rithal_v105_r1_apply_trade_manager_patch\(_rithal_v105_r1_sys\.modules\[__name__\]\)\r?\n',
        '(?ms)\r?\n# RITHAL_BEHAVIOR_FIX_V1_0_5_R3_(?:LIVE|MANAGER)_HOOK\r?\n.*?_rithal_v105_r3_apply_(?:live_patch\(globals\(\)\)|trade_manager_patch\(_rithal_v105_r3_sys\.modules\[__name__\]\))\r?\n'
    )
    foreach($pattern in $patterns){$text=[regex]::Replace($text,$pattern,"`r`n")}
    if(-not $text.EndsWith("`n")){$text+="`r`n"}
    $text+="`r`n$Hook`r`n"
    if(([regex]::Matches($text,[regex]::Escape($Marker))).Count -ne 1){throw "R3 hook count is not exactly one in $Path"}
    [IO.File]::WriteAllText($Path,$text,[Text.UTF8Encoding]::new($false))
}

if(-not(Test-Path -LiteralPath $Neural -PathType Container)){throw "Invalid project root: $Root"}
$required=@(
    $Live,$Manager,
    (Join-Path $Neural 'rithal_consolidated_v104.py'),
    (Join-Path $Neural 'rithal_runtime_settings.py'),
    (Join-Path $Neural 'rithal_manager_authority_contract.py')
)
foreach($path in $required){if(-not(Test-Path -LiteralPath $path -PathType Leaf)){throw "Required active source missing: $path"}}
if(-not(Get-Command python -ErrorAction SilentlyContinue)){throw 'python was not found on PATH'}
$CheckpointBefore=Checkpoint-Digest $CheckpointDir

# Authority must not change while old Python code is still loaded.
$running=@()
try{
    $running=@(Get-CimInstance Win32_Process -ErrorAction Stop|Where-Object{
        $name=[string]$_.Name;$cmd=[string]$_.CommandLine
        ($name -match '^(python|pythonw)(\.exe)?$') -and ($cmd -match '(?i)(live-neural-v2|mythos[\\/]neural[\\/]live\.py|rithal-1-0-contract-locked|Start-Rithal18KContractV2)')
    })
}catch{throw "Unable to prove that the trading engine is stopped. Run from an elevated PowerShell after closing Rithal. Detail: $($_.Exception.Message)"}
if($running.Count -gt 0){$details=($running|ForEach-Object{"PID=$($_.ProcessId) $($_.CommandLine)"}) -join "`n";throw "RITHAL_ENGINE_RUNNING. Stop the trading engine before installing actual behavior changes.`n$details"}
Stage 'Engine-stop preflight PASS'

New-Item -ItemType Directory -Path $Backup -Force|Out-Null
$targets=@($Live,$Manager,$BaseModule,$R3Module,$Settings,$Control,$PaperPolicy,$ControllerConfig,$VerifyFile,$RollbackFile,$InstallReport)
$records=@()
for($i=0;$i -lt $targets.Count;$i++){
    $target=$targets[$i];$exists=Test-Path -LiteralPath $target -PathType Leaf
    $copy=Join-Path $Backup (('{0:D2}_' -f $i)+[IO.Path]::GetFileName($target))
    if($exists){Copy-Item -LiteralPath $target -Destination $copy -Force}
    $records+=[pscustomobject]@{target=$target;existed=$exists;backup=$copy;sha256_before=(File-Sha $target)}
}
Write-JsonAtomic $Manifest ([pscustomobject]@{version=$Version;created_at=(Get-Date).ToUniversalTime().ToString('o');checkpoint_digest_before=$CheckpointBefore;files=$records})

try{
    Stage 'Downloading canonical base and R3 safety wrapper'
    Invoke-WebRequest -UseBasicParsing -Uri "$RawBase/rithal_behavior_fix_v105.py" -OutFile $BaseModule
    Invoke-WebRequest -UseBasicParsing -Uri "$RawBase/rithal_behavior_fix_v105_r3.py" -OutFile $R3Module

    Stage 'Installing one R3 engine hook after all existing overlays'
    $LiveMarker='# RITHAL_BEHAVIOR_FIX_V1_0_5_R3_LIVE_HOOK'
    $LiveHook=@'
# RITHAL_BEHAVIOR_FIX_V1_0_5_R3_LIVE_HOOK
try:
    from .rithal_behavior_fix_v105_r3 import apply_live_patch as _rithal_v105_r3_apply_live_patch
except ImportError:
    from rithal_behavior_fix_v105_r3 import apply_live_patch as _rithal_v105_r3_apply_live_patch
_rithal_v105_r3_apply_live_patch(globals())
'@
    Install-R3Hook $Live $LiveMarker $LiveHook @('class NeuralV2Model','class LiveEngine','RITHAL_CONSOLIDATED_INTELLIGENCE_SAFETY')

    Stage 'Installing one R3 single-manager hook after V3.3 and persistence overlays'
    $ManagerMarker='# RITHAL_BEHAVIOR_FIX_V1_0_5_R3_MANAGER_HOOK'
    $ManagerHook=@'
# RITHAL_BEHAVIOR_FIX_V1_0_5_R3_MANAGER_HOOK
try:
    from .rithal_behavior_fix_v105_r3 import apply_trade_manager_patch as _rithal_v105_r3_apply_trade_manager_patch
except ImportError:
    from rithal_behavior_fix_v105_r3 import apply_trade_manager_patch as _rithal_v105_r3_apply_trade_manager_patch
import sys as _rithal_v105_r3_sys
_rithal_v105_r3_apply_trade_manager_patch(_rithal_v105_r3_sys.modules[__name__])
'@
    Install-R3Hook $Manager $ManagerMarker $ManagerHook @('class TradeManager','RITHAL_TRADE_MANAGER_V3_3_2','RITHAL_CONSOLIDATED_TRADE_MANAGER_PERSISTENCE')

    Stage 'Compiling active engine, manager and authority contracts before settings change'
    & python -m py_compile $BaseModule $R3Module $Live $Manager (Join-Path $Neural 'rithal_consolidated_v104.py') (Join-Path $Neural 'rithal_runtime_settings.py') (Join-Path $Neural 'rithal_manager_authority_contract.py')
    if($LASTEXITCODE -ne 0){throw "Python compilation failed: $LASTEXITCODE"}

    Stage 'Running deterministic R3 behavior and active-signature tests'
    $SelfTest=& python $R3Module --self-test 2>&1
    if($LASTEXITCODE -ne 0){throw "R3 self-test failed: $([string]::Join(' ',@($SelfTest)))"}
    $SelfText=[string]::Join("`n",@($SelfTest));if(-not $SelfText.Contains('"status": "PASS"')){throw "R3 self-test did not report PASS: $SelfText"}

    Stage 'Cross-verifying exact active manager-authority classification'
    $AuthorityTest=& python $R3Module --authority-self-test --project-root $Root 2>&1
    if($LASTEXITCODE -ne 0){throw "Authority-contract self-test failed: $([string]::Join(' ',@($AuthorityTest)))"}
    $AuthorityText=[string]::Join("`n",@($AuthorityTest));if(-not $AuthorityText.Contains('"status": "PASS"')){throw "Authority-contract test did not report PASS: $AuthorityText"}

    Stage "Applying locked PAPER profile and manager mode=$ManagerMode"
    $ConfigureOutput=& python $R3Module --configure --project-root $Root --instance-id 'rithal-1-0-contract-locked' --manager-mode $ManagerMode 2>&1
    if($LASTEXITCODE -ne 0){throw "Settings configuration failed: $([string]::Join(' ',@($ConfigureOutput)))"}

    Stage 'Cross-verifying hook uniqueness, mode consistency and checkpoint immutability'
    $LiveText=[IO.File]::ReadAllText($Live);$ManagerText=[IO.File]::ReadAllText($Manager)
    if(([regex]::Matches($LiveText,[regex]::Escape($LiveMarker))).Count -ne 1){throw 'Live R3 hook count is not exactly one'}
    if(([regex]::Matches($ManagerText,[regex]::Escape($ManagerMarker))).Count -ne 1){throw 'Manager R3 hook count is not exactly one'}
    foreach($old in @('RITHAL_BEHAVIOR_FIX_V1_0_5_LIVE_HOOK','RITHAL_BEHAVIOR_FIX_V1_0_5_MANAGER_HOOK','RITHAL_BEHAVIOR_FIX_V1_0_5_R1_LIVE_HOOK','RITHAL_BEHAVIOR_FIX_V1_0_5_R1_MANAGER_HOOK')){if($LiveText.Contains($old) -or $ManagerText.Contains($old)){throw "Superseded hook remains: $old"}}
    $ControlObject=Get-Content -LiteralPath $Control -Raw|ConvertFrom-Json
    if($ManagerMode -eq 'PAPER_CONTROL'){
        if([string]$ControlObject.mode -ne 'PAPER_AUTOMANAGE' -or -not [bool]$ControlObject.execution_enabled){throw 'PAPER_CONTROL did not produce PAPER_AUTOMANAGE execution=true'}
    }else{
        if([string]$ControlObject.mode -ne 'SHADOW_ONLY' -or [bool]$ControlObject.execution_enabled){throw 'SHADOW_ONLY did not produce execution=false'}
    }
    $CheckpointAfter=Checkpoint-Digest $CheckpointDir
    if($CheckpointAfter -ne $CheckpointBefore){throw "Checkpoint digest changed: $CheckpointBefore -> $CheckpointAfter"}

    Stage 'Creating exact-project verifier and rollback commands'
    $VerifyBody=@"
param([string]`$ProjectRoot=(Get-Location).Path)
`$ErrorActionPreference='Stop'
`$root=[IO.Path]::GetFullPath(`$ProjectRoot);`$n=Join-Path `$root 'mythos\neural';`$b=Join-Path `$n 'rithal_behavior_fix_v105.py';`$r=Join-Path `$n 'rithal_behavior_fix_v105_r3.py';`$l=Join-Path `$n 'live.py';`$m=Join-Path `$n 'trade_manager_v6.py'
python -m py_compile `$b `$r `$l `$m (Join-Path `$n 'rithal_consolidated_v104.py') (Join-Path `$n 'rithal_runtime_settings.py') (Join-Path `$n 'rithal_manager_authority_contract.py');if(`$LASTEXITCODE -ne 0){throw "compile failed: `$LASTEXITCODE"}
python `$r --self-test;if(`$LASTEXITCODE -ne 0){throw "self-test failed: `$LASTEXITCODE"}
python `$r --authority-self-test --project-root `$root;if(`$LASTEXITCODE -ne 0){throw "authority test failed: `$LASTEXITCODE"}
`$lt=[IO.File]::ReadAllText(`$l);`$mt=[IO.File]::ReadAllText(`$m);if(([regex]::Matches(`$lt,[regex]::Escape('# RITHAL_BEHAVIOR_FIX_V1_0_5_R3_LIVE_HOOK'))).Count -ne 1){throw 'live hook invalid'};if(([regex]::Matches(`$mt,[regex]::Escape('# RITHAL_BEHAVIOR_FIX_V1_0_5_R3_MANAGER_HOOK'))).Count -ne 1){throw 'manager hook invalid'}
`$report=Get-Content -LiteralPath (Join-Path `$root 'rithal_behavior_fix_v105_r3_install_report.json') -Raw|ConvertFrom-Json
`$dir=Join-Path `$root 'checkpoints\rithal_clean';if(-not(Test-Path `$dir -PathType Container)){throw 'checkpoint directory missing'};`$rows=Get-ChildItem `$dir -File -Recurse|Sort-Object FullName|ForEach-Object{`$rel=`$_.FullName.Substring(`$dir.Length).TrimStart([char[]]'\/');"`$rel|`$((Get-FileHash -Algorithm SHA256 `$_.FullName).Hash.ToLowerInvariant())"};if(@(`$rows).Count -eq 0){throw 'checkpoint directory empty'};`$bytes=[Text.Encoding]::UTF8.GetBytes([string]::Join("`n",`$rows));`$sha=[Security.Cryptography.SHA256]::Create();try{`$digest=([BitConverter]::ToString(`$sha.ComputeHash(`$bytes))).Replace('-','').ToLowerInvariant()}finally{`$sha.Dispose()};if(`$digest -ne [string]`$report.checkpoint_digest_after){throw "checkpoint digest mismatch: `$digest"}
Write-Host '[RITHAL_BEHAVIOR_FIX_V1_0_5_R3] VERIFICATION PASS' -ForegroundColor Green
"@
    [IO.File]::WriteAllText($VerifyFile,$VerifyBody,[Text.UTF8Encoding]::new($false))
    $RollbackBody=@"
`$ErrorActionPreference='Stop';`$m=Get-Content -LiteralPath '$Manifest' -Raw|ConvertFrom-Json;foreach(`$item in @(`$m.files)){if(`$item.existed -and (Test-Path -LiteralPath `$item.backup)){Copy-Item -LiteralPath `$item.backup -Destination `$item.target -Force}elseif(-not `$item.existed -and (Test-Path -LiteralPath `$item.target)){Remove-Item -LiteralPath `$item.target -Force -ErrorAction SilentlyContinue}};Write-Host '[RITHAL_BEHAVIOR_FIX_V1_0_5_R3] rollback complete; restart the engine.' -ForegroundColor Yellow
"@
    [IO.File]::WriteAllText($RollbackFile,$RollbackBody,[Text.UTF8Encoding]::new($false))

    $Report=[pscustomobject]@{
        version=$Version;installed_at=(Get-Date).ToUniversalTime().ToString('o');instance_id='rithal-1-0-contract-locked';manager_mode=$ManagerMode
        compile='PASS';self_test='PASS';authority_contract_test='PASS';hooks=[pscustomobject]@{live=1;manager=1}
        checkpoint_digest_before=$CheckpointBefore;checkpoint_digest_after=$CheckpointAfter;checkpoint_unchanged=$true
        protected_contract=[pscustomobject]@{starting_equity=18000.0;per_trade_margin_pct=0.10;leverage=10;allocation_cap_pct=0.60;heat_cap_pct=0.03;live_exchange_control=$false;feature_order_changed=$false;static_thresholds_changed=$false;tp_sl_geometry_changed=$false}
        corrected_findings=@('active_method_signature','missing_regime_fail_closed','restored_entry_context','feature_repair_lock','true_rank_regime_separation','shadow_mode_control_consistency','checkpoint_presence_required')
        backup=$Backup;restart_required=$true;configure_output=[string]::Join("`n",@($ConfigureOutput));authority_output=$AuthorityText
    }
    Write-JsonAtomic $InstallReport $Report

    Write-Host "[$Version] INSTALLATION PASS" -ForegroundColor Green
    Write-Host "Backup: $Backup"
    Write-Host 'Verify: .\Verify-RithalBehaviorFixV1_0_5_R3.ps1' -ForegroundColor Green
    Write-Host 'Controlled engine restart required to activate behavior changes.' -ForegroundColor Yellow
}catch{
    Write-Warning "[$Version] failed: $($_.Exception.Message)"
    Restore-Backup $Manifest
    throw
}
