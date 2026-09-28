[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Version = 'RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3'
$Root = [IO.Path]::GetFullPath($ProjectRoot)
$Neural = Join-Path $Root 'mythos\neural'
$Live = Join-Path $Neural 'live.py'
$SourceModule = Join-Path $PSScriptRoot 'rithal_behavior_fix_v105_r3_3.py'
$TargetModule = Join-Path $Neural 'rithal_behavior_fix_v105_r3_3.py'
$VerifyTarget = Join-Path $Root 'Verify-RithalBehaviorFixV1_0_5_R3_3.ps1'
$RollbackTarget = Join-Path $Root 'Rollback-RithalBehaviorFixV1_0_5_R3_3.ps1'
$Report = Join-Path $Root 'mythos_model_instances\rithal-1-0-contract-locked\rithal_behavior_fix_v105_r3_3_install_report.json'
$BackupRoot = Join-Path $Root ('rithal_behavior_backups\v1_0_5_r3_3_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
$Manifest = Join-Path $BackupRoot 'manifest.json'
$Python = (Get-Command python -ErrorAction Stop).Source

function Stage([string]$Message,[ConsoleColor]$Color=[ConsoleColor]::Cyan){
    Write-Host "[$Version] $Message" -ForegroundColor $Color
}
function Hash([string]$Path){
    if(Test-Path -LiteralPath $Path -PathType Leaf){return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()}
    return $null
}
function WriteJson([string]$Path,$Value){
    $parent=Split-Path -Parent $Path
    if($parent){New-Item -ItemType Directory -Force -Path $parent|Out-Null}
    $tmp="$Path.$PID.tmp"
    [IO.File]::WriteAllText($tmp,(($Value|ConvertTo-Json -Depth 100)+[Environment]::NewLine),[Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $tmp -Destination $Path -Force
}
function RestoreBackup {
    if(-not(Test-Path -LiteralPath $Manifest -PathType Leaf)){return}
    $state=Get-Content -LiteralPath $Manifest -Raw|ConvertFrom-Json
    foreach($item in @($state.files)){
        if([bool]$item.existed -and (Test-Path -LiteralPath ([string]$item.backup) -PathType Leaf)){
            $parent=Split-Path -Parent ([string]$item.target)
            if($parent){New-Item -ItemType Directory -Force -Path $parent|Out-Null}
            Copy-Item -LiteralPath ([string]$item.backup) -Destination ([string]$item.target) -Force
        }elseif(-not [bool]$item.existed -and (Test-Path -LiteralPath ([string]$item.target))){
            Remove-Item -LiteralPath ([string]$item.target) -Force -ErrorAction SilentlyContinue
        }
    }
}

if(-not(Test-Path -LiteralPath $Neural -PathType Container)){throw "Invalid project root: $Root"}
foreach($path in @(
    $Live,
    (Join-Path $Neural 'rithal_behavior_fix_v105.py'),
    (Join-Path $Neural 'rithal_behavior_fix_v105_r3.py'),
    (Join-Path $Neural 'rithal_behavior_fix_v105_r3_1.py'),
    (Join-Path $Neural 'rithal_behavior_fix_v105_r3_2.py'),
    $SourceModule
)){
    if(-not(Test-Path -LiteralPath $path -PathType Leaf)){throw "Required source missing: $path"}
}

$running=@(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue|Where-Object{
    $_.Name -match '^python(?:w)?\.exe$' -and
    ([string]$_.CommandLine).ToLowerInvariant().Contains($Root.ToLowerInvariant()) -and
    ([string]$_.CommandLine) -match 'quick_start\.py|mythos\.neural\.live|mythos\\neural\\live\.py'
})
if($running.Count){
    $ids=($running|ForEach-Object{$_.ProcessId}) -join ','
    throw "Rithal engine is still running (PID $ids). Stop only the engine with Ctrl+C, then rerun this installer."
}

$protected=@(
    (Join-Path $Neural 'rithal_behavior_fix_v105.py'),
    (Join-Path $Neural 'rithal_behavior_fix_v105_r3.py'),
    (Join-Path $Neural 'rithal_behavior_fix_v105_r3_1.py'),
    (Join-Path $Neural 'rithal_behavior_fix_v105_r3_2.py')
)
$protectedBefore=@{}
foreach($path in $protected){$protectedBefore[$path]=Hash $path}

New-Item -ItemType Directory -Force -Path $BackupRoot|Out-Null
$targets=@($Live,$TargetModule,$VerifyTarget,$RollbackTarget,$Report)
$records=@()
for($index=0;$index-lt $targets.Count;$index++){
    $target=[string]$targets[$index]
    $exists=Test-Path -LiteralPath $target -PathType Leaf
    $backup=Join-Path $BackupRoot (('{0:D2}_'-f $index)+((Split-Path $target -Leaf)-replace '[^A-Za-z0-9._-]','_'))
    if($exists){Copy-Item -LiteralPath $target -Destination $backup -Force}
    $records+=[ordered]@{target=$target;existed=$exists;backup=$backup}
}
WriteJson $Manifest ([ordered]@{version=$Version;created_at=(Get-Date).ToUniversalTime().ToString('o');files=$records})

try{
    Stage 'Running isolated scorer/inference simulation.'
    $selfTest=& $Python $SourceModule --self-test 2>&1
    $selfCode=$LASTEXITCODE
    $selfText=[string]::Join("`n",@($selfTest))
    $selfTest|ForEach-Object{Write-Host $_}
    if($selfCode-ne 0 -or -not $selfText.Contains('"status": "PASS"')){throw "R3.3 self-test failed with exit code $selfCode"}

    Stage 'Installing compatibility module without changing any existing behavior module.'
    Copy-Item -LiteralPath $SourceModule -Destination $TargetModule -Force

    $text=[IO.File]::ReadAllText($Live)
    $loggerMarker='RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3_SAFE_REGIME_LOG'
    if(-not $text.Contains($loggerMarker)){
        $old=@'
            _rp = self._get_regime_probs(sym, pred)
            L("  │ REGIME-P:   p_trend=%.3f  p_chop=%.3f  p_breakout=%.3f   (chop gate %.2f / panic gate %.2f)",
              _rp.get("p_trend", 0.0), _rp.get("p_chop", 0.0), _rp.get("p_panic", 0.0),
              self.CHOP_GATE_THRESHOLD, self.PANIC_GATE_THRESHOLD)
'@
        $new=@'
            _rp = self._get_regime_probs(sym, pred) or {}
            # RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3_SAFE_REGIME_LOG
            def _r33_regime_log_number(name):
                try:
                    value = float(_rp.get(name, float("nan")))
                    return value if np.isfinite(value) else float("nan")
                except Exception:
                    return float("nan")
            L("  │ REGIME-P:   p_trend=%.3f  p_chop=%.3f  p_breakout=%.3f   (chop gate %.2f / panic gate %.2f)",
              _r33_regime_log_number("p_trend"), _r33_regime_log_number("p_chop"),
              _r33_regime_log_number("p_breakout"),
              self.CHOP_GATE_THRESHOLD, self.PANIC_GATE_THRESHOLD)
'@
        if(-not $text.Contains($old)){throw 'Exact rich-regime logger block was not found; refusing a broad edit.'}
        $text=$text.Replace($old,$new)
    }

    $activationMarker='RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3_ACTIVATION_START'
    if(-not $text.Contains($activationMarker)){
        $activation=@'

# RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3_ACTIVATION_START
try:
    from .rithal_behavior_fix_v105_r3_3 import apply_live_patch as _rithal_v105_r33_apply_live_patch
except ImportError:
    from rithal_behavior_fix_v105_r3_3 import apply_live_patch as _rithal_v105_r33_apply_live_patch
_rithal_v105_r33_apply_live_patch(globals())
# RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3_ACTIVATION_END
'@
        $text=$text.TrimEnd()+$activation+[Environment]::NewLine
    }

    $tmp="$Live.$PID.r33.tmp"
    [IO.File]::WriteAllText($tmp,$text,[Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $tmp -Destination $Live -Force

    Stage 'Compiling the complete active calculation chain.'
    & $Python -m py_compile $Live $TargetModule @protected
    if($LASTEXITCODE-ne 0){throw "Python compilation failed with exit code $LASTEXITCODE"}

    $liveText=[IO.File]::ReadAllText($Live)
    if(([regex]::Matches($liveText,[regex]::Escape($activationMarker))).Count-ne 1){throw 'R3.3 activation marker count is not exactly one'}
    if(([regex]::Matches($liveText,[regex]::Escape($loggerMarker))).Count-ne 1){throw 'Safe regime logger marker count is not exactly one'}
    if($liveText.Contains('_rp.get("p_panic", 0.0)')){throw 'Rich logger still labels panic as breakout'}
    if(-not $liveText.Contains('_r33_regime_log_number("p_breakout")')){throw 'Rich logger does not read p_breakout'}

    foreach($path in $protected){
        if((Hash $path)-ne $protectedBefore[$path]){throw "Protected behavior source changed unexpectedly: $path"}
    }

    $verifyBody=@'
[CmdletBinding()]
param([string]$ProjectRoot=(Get-Location).Path)
$ErrorActionPreference='Stop';$root=[IO.Path]::GetFullPath($ProjectRoot);$n=Join-Path $root 'mythos\neural';$live=Join-Path $n 'live.py';$module=Join-Path $n 'rithal_behavior_fix_v105_r3_3.py'
foreach($p in @($live,$module,(Join-Path $n 'rithal_behavior_fix_v105.py'),(Join-Path $n 'rithal_behavior_fix_v105_r3.py'),(Join-Path $n 'rithal_behavior_fix_v105_r3_1.py'),(Join-Path $n 'rithal_behavior_fix_v105_r3_2.py'))){if(-not(Test-Path -LiteralPath $p -PathType Leaf)){throw "missing: $p"}}
python $module --self-test;if($LASTEXITCODE-ne 0){throw 'R3.3 self-test failed'}
python -m py_compile $live $module (Join-Path $n 'rithal_behavior_fix_v105.py') (Join-Path $n 'rithal_behavior_fix_v105_r3.py') (Join-Path $n 'rithal_behavior_fix_v105_r3_1.py') (Join-Path $n 'rithal_behavior_fix_v105_r3_2.py');if($LASTEXITCODE-ne 0){throw 'compile failed'}
$t=[IO.File]::ReadAllText($live)
foreach($m in @('RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3_ACTIVATION_START','RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3_SAFE_REGIME_LOG')){if(([regex]::Matches($t,[regex]::Escape($m))).Count-ne 1){throw "marker invalid: $m"}}
if($t.Contains('_rp.get("p_panic", 0.0)')){throw 'legacy breakout logger remains'}
Write-Host '[RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3] VERIFICATION PASS' -ForegroundColor Green
'@
    [IO.File]::WriteAllText($VerifyTarget,$verifyBody,[Text.UTF8Encoding]::new($false))

    $rollbackBody=@"
`$ErrorActionPreference='Stop';`$state=Get-Content -LiteralPath '$Manifest' -Raw|ConvertFrom-Json;foreach(`$item in @(`$state.files)){if([bool]`$item.existed -and (Test-Path -LiteralPath ([string]`$item.backup))){`$parent=Split-Path -Parent ([string]`$item.target);if(`$parent){New-Item -ItemType Directory -Force -Path `$parent|Out-Null};Copy-Item -LiteralPath ([string]`$item.backup) -Destination ([string]`$item.target) -Force}elseif(-not [bool]`$item.existed -and (Test-Path -LiteralPath ([string]`$item.target)){Remove-Item -LiteralPath ([string]`$item.target) -Force -ErrorAction SilentlyContinue}};Write-Host '[RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3] rollback complete' -ForegroundColor Yellow
"@
    [IO.File]::WriteAllText($RollbackTarget,$rollbackBody,[Text.UTF8Encoding]::new($false))

    $protectedAfter=@{}
    foreach($path in $protected){$protectedAfter[$path]=Hash $path}
    WriteJson $Report ([ordered]@{
        version=$Version
        installed_at=(Get-Date).ToUniversalTime().ToString('o')
        project_root=$Root
        root_cause='R3.2 restored the canonical scorer but left the V1.0.5 infer wrapper expecting regime_valid and scalar regime fields; its side overlay also recomputed edges without the canonical scorer MAE fields.'
        correction='Project the existing canonical true/rank posteriors into the infer contract, reuse canonical MAE-based edges, retain risk-adjusted side selection, and make rich logging None-safe.'
        changed_files=@($Live,$TargetModule,$VerifyTarget,$RollbackTarget,$Report)
        protected_behavior_hashes_before=$protectedBefore
        protected_behavior_hashes_after=$protectedAfter
        checkpoint_files_changed=$false
        feature_order_changed=$false
        raw_composite_changed=$false
        true_rank_regime_arrays_changed=$false
        thresholds_changed=$false
        fees_changed=$false
        tp_sl_changed=$false
        sizing_or_authority_settings_changed=$false
        backup=$BackupRoot
    })

    Stage 'INSTALLATION PASS' Green
    Write-Host 'Verify: .\Verify-RithalBehaviorFixV1_0_5_R3_3.ps1 -ProjectRoot .' -ForegroundColor Green
    Write-Host 'Restart the Rithal engine once. The startup log must show RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3.' -ForegroundColor Green
}catch{
    Write-Warning "[$Version] failed: $($_.Exception.Message)"
    RestoreBackup
    Write-Warning "[$Version] all edited files were restored from backup."
    throw
}
