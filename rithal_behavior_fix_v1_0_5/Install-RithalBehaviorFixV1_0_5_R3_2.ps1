param(
    [Parameter(Mandatory=$false)][string]$ProjectRoot=(Get-Location).Path,
    [Parameter(Mandatory=$false)][ValidateSet('PAPER_CONTROL','SHADOW_ONLY')][string]$ManagerMode='PAPER_CONTROL'
)

$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest

$Version='RITHAL_BEHAVIOR_FIX_V1_0_5_R3_2'
$Branch='rithal-behavior-fix-v1.0.5'
$RawBase="https://raw.githubusercontent.com/carvedkarma/NeuralTrade/$Branch/rithal_behavior_fix_v1_0_5"
$Root=[IO.Path]::GetFullPath($ProjectRoot)
$Neural=Join-Path $Root 'mythos\neural'
$Live=Join-Path $Neural 'live.py'
$Manager=Join-Path $Neural 'trade_manager_v6.py'
$R32Module=Join-Path $Neural 'rithal_behavior_fix_v105_r3_2.py'
$R31Installer=Join-Path $Root 'Install-RithalBehaviorFixV1_0_5_R3_1.ps1'
$R31Rollback=Join-Path $Root 'Rollback-RithalBehaviorFixV1_0_5_R3_1.ps1'
$VerifyFile=Join-Path $Root 'Verify-RithalBehaviorFixV1_0_5_R3_2.ps1'
$RollbackFile=Join-Path $Root 'Rollback-RithalBehaviorFixV1_0_5_R3_2.ps1'
$ReportFile=Join-Path $Root 'rithal_behavior_fix_v105_r3_2_install_report.json'
$CheckpointDir=Join-Path $Root 'checkpoints\rithal_clean'

function Stage([string]$Text){Write-Host "[$Version] $Text" -ForegroundColor Cyan}
function Write-JsonAtomic([string]$Path,$Object){$temp="$Path.$PID.tmp";[IO.File]::WriteAllText($temp,(($Object|ConvertTo-Json -Depth 100)+[Environment]::NewLine),[Text.UTF8Encoding]::new($false));Move-Item -LiteralPath $temp -Destination $Path -Force}
function Checkpoint-Digest([string]$Dir){
    if(-not(Test-Path -LiteralPath $Dir -PathType Container)){throw "Checkpoint directory missing: $Dir"}
    $files=@(Get-ChildItem -LiteralPath $Dir -File -Recurse|Sort-Object FullName)
    if($files.Count -eq 0){throw "Checkpoint directory is empty: $Dir"}
    $rows=$files|ForEach-Object{$rel=$_.FullName.Substring($Dir.Length).TrimStart([char[]]'\/');"$rel|$((Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant())"}
    $bytes=[Text.Encoding]::UTF8.GetBytes([string]::Join("`n",$rows));$sha=[Security.Cryptography.SHA256]::Create()
    try{return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-','').ToLowerInvariant()}finally{$sha.Dispose()}
}
function Replace-Hook([string]$Path,[string]$Kind){
    $text=[IO.File]::ReadAllText($Path)
    if($Kind -eq 'LIVE'){
        $pattern='(?ms)\r?\n# RITHAL_BEHAVIOR_FIX_V1_0_5_R3_1_LIVE_HOOK\r?\n.*?_rithal_v105_r3_1_apply_live_patch\(globals\(\)\)\r?\n'
        $hook=@'

# RITHAL_BEHAVIOR_FIX_V1_0_5_R3_2_LIVE_HOOK
try:
    from .rithal_behavior_fix_v105_r3_2 import apply_live_patch as _rithal_v105_r3_2_apply_live_patch
except ImportError:
    from rithal_behavior_fix_v105_r3_2 import apply_live_patch as _rithal_v105_r3_2_apply_live_patch
_rithal_v105_r3_2_apply_live_patch(globals())
'@
    }else{
        $pattern='(?ms)\r?\n# RITHAL_BEHAVIOR_FIX_V1_0_5_R3_1_MANAGER_HOOK\r?\n.*?_rithal_v105_r3_1_apply_trade_manager_patch\(_rithal_v105_r3_1_sys\.modules\[__name__\]\)\r?\n'
        $hook=@'

# RITHAL_BEHAVIOR_FIX_V1_0_5_R3_2_MANAGER_HOOK
try:
    from .rithal_behavior_fix_v105_r3_2 import apply_trade_manager_patch as _rithal_v105_r3_2_apply_trade_manager_patch
except ImportError:
    from rithal_behavior_fix_v105_r3_2 import apply_trade_manager_patch as _rithal_v105_r3_2_apply_trade_manager_patch
import sys as _rithal_v105_r3_2_sys
_rithal_v105_r3_2_apply_trade_manager_patch(_rithal_v105_r3_2_sys.modules[__name__])
'@
    }
    $updated=[regex]::Replace($text,$pattern,$hook)
    if($updated -eq $text){throw "Expected R3.1 $Kind hook was not found in $Path"}
    $marker="# RITHAL_BEHAVIOR_FIX_V1_0_5_R3_2_${Kind}_HOOK"
    if(([regex]::Matches($updated,[regex]::Escape($marker))).Count -ne 1){throw "R3.2 $Kind hook count is not exactly one"}
    if($updated.Contains("# RITHAL_BEHAVIOR_FIX_V1_0_5_R3_1_${Kind}_HOOK")){throw "Superseded R3.1 $Kind hook remains"}
    [IO.File]::WriteAllText($Path,$updated,[Text.UTF8Encoding]::new($false))
}
function Rollback-R32{
    if(Test-Path -LiteralPath $R31Rollback -PathType Leaf){
        try{& powershell -NoProfile -ExecutionPolicy Bypass -File $R31Rollback|Out-Host}catch{Write-Warning "R3.1 rollback failed: $($_.Exception.Message)"}
    }
    foreach($path in @($R32Module,$VerifyFile,$RollbackFile,$ReportFile)){
        Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
    }
}

if(-not(Test-Path -LiteralPath $Neural -PathType Container)){throw "Invalid project root: $Root"}
if(-not(Get-Command python -ErrorAction SilentlyContinue)){throw 'python was not found on PATH'}
$CheckpointBefore=Checkpoint-Digest $CheckpointDir

try{
    Stage 'Reinstalling verified R3.1 baseline while the engine is stopped'
    Invoke-WebRequest -UseBasicParsing -Uri "$RawBase/Install-RithalBehaviorFixV1_0_5_R3_1.ps1" -OutFile $R31Installer
    & powershell -NoProfile -ExecutionPolicy Bypass -File $R31Installer -ProjectRoot $Root -ManagerMode $ManagerMode
    if($LASTEXITCODE -ne 0){throw "R3.1 baseline installer failed: $LASTEXITCODE"}

    Stage 'Installing R3.2 active-shared-scorer module'
    Invoke-WebRequest -UseBasicParsing -Uri "$RawBase/rithal_behavior_fix_v105_r3_2.py" -OutFile $R32Module
    Replace-Hook $Live 'LIVE'
    Replace-Hook $Manager 'MANAGER'

    Stage 'Compiling complete behavior chain and active source'
    $compile=@(
        (Join-Path $Neural 'rithal_behavior_fix_v105.py'),
        (Join-Path $Neural 'rithal_behavior_fix_v105_r3.py'),
        (Join-Path $Neural 'rithal_behavior_fix_v105_r3_1.py'),
        $R32Module,$Live,$Manager,
        (Join-Path $Neural 'rithal_consolidated_v104.py'),
        (Join-Path $Neural 'rithal_runtime_settings.py'),
        (Join-Path $Neural 'rithal_manager_authority_contract.py')
    )
    & python -m py_compile @compile
    if($LASTEXITCODE -ne 0){throw "Python compilation failed: $LASTEXITCODE"}

    Stage 'Running the exact rank-prewarm regression test'
    $self=& python $R32Module --self-test 2>&1
    if($LASTEXITCODE -ne 0){throw "R3.2 self-test failed: $([string]::Join(' ',@($self)))"}
    $selfText=[string]::Join("`n",@($self))
    if(-not $selfText.Contains('"status": "PASS"')){throw "R3.2 self-test did not report PASS: $selfText"}

    Stage 'Rechecking exact local manager authority classification'
    $authority=& python $R32Module --authority-self-test --project-root $Root 2>&1
    if($LASTEXITCODE -ne 0){throw "Authority test failed: $([string]::Join(' ',@($authority)))"}
    $authorityText=[string]::Join("`n",@($authority))
    if(-not $authorityText.Contains('"status": "PASS"')){throw "Authority test did not report PASS: $authorityText"}

    $CheckpointAfter=Checkpoint-Digest $CheckpointDir
    if($CheckpointAfter -ne $CheckpointBefore){throw "Checkpoint digest changed: $CheckpointBefore -> $CheckpointAfter"}

    Stage 'Creating verifier, rollback and install report'
    $VerifyBody=@"
param([string]`$ProjectRoot=(Get-Location).Path)
`$ErrorActionPreference='Stop';`$root=[IO.Path]::GetFullPath(`$ProjectRoot);`$n=Join-Path `$root 'mythos\neural';`$r=Join-Path `$n 'rithal_behavior_fix_v105_r3_2.py';`$l=Join-Path `$n 'live.py';`$m=Join-Path `$n 'trade_manager_v6.py'
python -m py_compile (Join-Path `$n 'rithal_behavior_fix_v105.py') (Join-Path `$n 'rithal_behavior_fix_v105_r3.py') (Join-Path `$n 'rithal_behavior_fix_v105_r3_1.py') `$r `$l `$m;if(`$LASTEXITCODE -ne 0){throw 'compile failed'}
python `$r --self-test;if(`$LASTEXITCODE -ne 0){throw 'rank-prewarm regression test failed'}
python `$r --authority-self-test --project-root `$root;if(`$LASTEXITCODE -ne 0){throw 'authority test failed'}
`$lt=[IO.File]::ReadAllText(`$l);`$mt=[IO.File]::ReadAllText(`$m);if(([regex]::Matches(`$lt,[regex]::Escape('# RITHAL_BEHAVIOR_FIX_V1_0_5_R3_2_LIVE_HOOK'))).Count -ne 1){throw 'live hook invalid'};if(([regex]::Matches(`$mt,[regex]::Escape('# RITHAL_BEHAVIOR_FIX_V1_0_5_R3_2_MANAGER_HOOK'))).Count -ne 1){throw 'manager hook invalid'};if(`$lt.Contains('# RITHAL_BEHAVIOR_FIX_V1_0_5_R3_1_LIVE_HOOK') -or `$mt.Contains('# RITHAL_BEHAVIOR_FIX_V1_0_5_R3_1_MANAGER_HOOK')){throw 'R3.1 hook still active'}
Write-Host '[RITHAL_BEHAVIOR_FIX_V1_0_5_R3_2] VERIFICATION PASS' -ForegroundColor Green
"@
    [IO.File]::WriteAllText($VerifyFile,$VerifyBody,[Text.UTF8Encoding]::new($false))
    $RollbackBody=@"
`$ErrorActionPreference='Stop';if(Test-Path -LiteralPath '$R31Rollback'){& powershell -NoProfile -ExecutionPolicy Bypass -File '$R31Rollback'};Remove-Item -LiteralPath '$R32Module' -Force -ErrorAction SilentlyContinue;Write-Host '[RITHAL_BEHAVIOR_FIX_V1_0_5_R3_2] rollback complete; restart the engine.' -ForegroundColor Yellow
"@
    [IO.File]::WriteAllText($RollbackFile,$RollbackBody,[Text.UTF8Encoding]::new($false))
    Write-JsonAtomic $ReportFile ([pscustomobject]@{
        version=$Version;installed_at=(Get-Date).ToUniversalTime().ToString('o');manager_mode=$ManagerMode
        root_cause='R3.1 raw regime_logits requirement bypassed active shared scorer during rank prewarm'
        correction='active _score_model_output and prewarm_rank_history preserved; risk-adjusted side applied as overlay only'
        self_test='PASS';authority_test='PASS';checkpoint_digest_before=$CheckpointBefore;checkpoint_digest_after=$CheckpointAfter;checkpoint_unchanged=$true
        restart_required=$true
    })
    Write-Host "[$Version] INSTALLATION PASS" -ForegroundColor Green
    Write-Host 'Verify: .\Verify-RithalBehaviorFixV1_0_5_R3_2.ps1' -ForegroundColor Green
    Write-Host 'Restart the engine once to load R3.2.' -ForegroundColor Yellow
}catch{
    Write-Warning "[$Version] failed: $($_.Exception.Message)"
    Rollback-R32
    throw
}
