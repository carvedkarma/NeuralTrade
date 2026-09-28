[CmdletBinding()]
param([string]$ProjectRoot=(Get-Location).Path)

$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$Version='RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3_R2'
$Root=[IO.Path]::GetFullPath($ProjectRoot)
$Neural=Join-Path $Root 'mythos\neural'
$Live=Join-Path $Neural 'live.py'
$SourceModule=Join-Path $PSScriptRoot 'rithal_behavior_fix_v105_r3_3.py'
$SourcePatcher=Join-Path $PSScriptRoot 'rithal_behavior_fix_v105_r3_3_install_patch.py'
$TargetModule=Join-Path $Neural 'rithal_behavior_fix_v105_r3_3.py'
$TargetPatcher=Join-Path $Neural 'rithal_behavior_fix_v105_r3_3_install_patch.py'
$Verify=Join-Path $Root 'Verify-RithalBehaviorFixV1_0_5_R3_3.ps1'
$Rollback=Join-Path $Root 'Rollback-RithalBehaviorFixV1_0_5_R3_3.ps1'
$Report=Join-Path $Root 'mythos_model_instances\rithal-1-0-contract-locked\rithal_behavior_fix_v105_r3_3_install_report.json'
$Backup=Join-Path $Root ('rithal_behavior_backups\v1_0_5_r3_3_r2_'+(Get-Date -Format 'yyyyMMdd_HHmmss'))
$Manifest=Join-Path $Backup 'manifest.json'
$Python=(Get-Command python -ErrorAction Stop).Source

function Stage([string]$Message,[ConsoleColor]$Color=[ConsoleColor]::Cyan){Write-Host "[$Version] $Message" -ForegroundColor $Color}
function Sha([string]$Path){if(Test-Path -LiteralPath $Path -PathType Leaf){return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()};return $null}
function JsonAtomic([string]$Path,$Value){$parent=Split-Path -Parent $Path;if($parent){New-Item -ItemType Directory -Force -Path $parent|Out-Null};$tmp="$Path.$PID.tmp";[IO.File]::WriteAllText($tmp,(($Value|ConvertTo-Json -Depth 100)+[Environment]::NewLine),[Text.UTF8Encoding]::new($false));Move-Item -LiteralPath $tmp -Destination $Path -Force}
function Restore-All {
    if(-not(Test-Path -LiteralPath $Manifest -PathType Leaf)){return}
    $m=Get-Content -LiteralPath $Manifest -Raw|ConvertFrom-Json
    foreach($item in @($m.files)){
        if([bool]$item.existed -and (Test-Path -LiteralPath ([string]$item.backup) -PathType Leaf)){
            $parent=Split-Path -Parent ([string]$item.target);if($parent){New-Item -ItemType Directory -Force -Path $parent|Out-Null}
            Copy-Item -LiteralPath ([string]$item.backup) -Destination ([string]$item.target) -Force
        }elseif(-not [bool]$item.existed -and (Test-Path -LiteralPath ([string]$item.target))){Remove-Item -LiteralPath ([string]$item.target) -Force -ErrorAction SilentlyContinue}
    }
}

if(-not(Test-Path -LiteralPath $Neural -PathType Container)){throw "Invalid project root: $Root"}
$Protected=@(
    (Join-Path $Neural 'rithal_behavior_fix_v105.py'),
    (Join-Path $Neural 'rithal_behavior_fix_v105_r3.py'),
    (Join-Path $Neural 'rithal_behavior_fix_v105_r3_1.py'),
    (Join-Path $Neural 'rithal_behavior_fix_v105_r3_2.py')
)
foreach($path in @($Live,$SourceModule,$SourcePatcher)+$Protected){if(-not(Test-Path -LiteralPath $path -PathType Leaf)){throw "Required source missing: $path"}}
$running=@(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue|Where-Object{
    $_.Name -match '^python(?:w)?\.exe$' -and ([string]$_.CommandLine).ToLowerInvariant().Contains($Root.ToLowerInvariant()) -and ([string]$_.CommandLine) -match 'quick_start\.py|mythos\.neural\.live|mythos\\neural\\live\.py'
})
if($running.Count){throw "Rithal engine is still running (PID $(($running|ForEach-Object{$_.ProcessId})-join ',')). Stop only the engine with Ctrl+C and rerun."}

$Before=@{};foreach($path in $Protected){$Before[$path]=Sha $path}
New-Item -ItemType Directory -Force -Path $Backup|Out-Null
$Targets=@($Live,$TargetModule,$TargetPatcher,$Verify,$Rollback,$Report)
$Records=@()
for($i=0;$i-lt $Targets.Count;$i++){$target=[string]$Targets[$i];$exists=Test-Path -LiteralPath $target -PathType Leaf;$copy=Join-Path $Backup (('{0:D2}_'-f $i)+((Split-Path $target -Leaf)-replace '[^A-Za-z0-9._-]','_'));if($exists){Copy-Item -LiteralPath $target -Destination $copy -Force};$Records+=[ordered]@{target=$target;existed=$exists;backup=$copy}}
JsonAtomic $Manifest ([ordered]@{version=$Version;created_at=(Get-Date).ToUniversalTime().ToString('o');files=$Records})

try{
    Stage 'Running scorer/inference and line-ending patch simulations.'
    foreach($test in @($SourceModule,$SourcePatcher)){$output=& $Python $test --self-test 2>&1;$code=$LASTEXITCODE;$text=[string]::Join("`n",@($output));$output|ForEach-Object{Write-Host $_};if($code-ne 0 -or -not $text.Contains('"status": "PASS"')){throw "Self-test failed: $test (exit $code)"}}

    Stage 'Copying the additive compatibility layer.'
    Copy-Item -LiteralPath $SourceModule -Destination $TargetModule -Force
    Copy-Item -LiteralPath $SourcePatcher -Destination $TargetPatcher -Force

    Stage 'Applying exact live activation and safe regime telemetry patch.'
    $patchOutput=& $Python $SourcePatcher --patch --live $Live 2>&1
    $patchCode=$LASTEXITCODE;$patchText=[string]::Join("`n",@($patchOutput));$patchOutput|ForEach-Object{Write-Host $_}
    if($patchCode-ne 0 -or -not $patchText.Contains('"status": "PASS"')){throw "Live patch failed with exit code $patchCode"}

    Stage 'Compiling the complete model/execution chain.'
    $Compile=@($Live,$TargetModule,$TargetPatcher)+$Protected
    & $Python -m py_compile @Compile
    if($LASTEXITCODE-ne 0){throw "Python compilation failed with exit code $LASTEXITCODE"}

    $liveText=[IO.File]::ReadAllText($Live)
    foreach($marker in @('RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3_ACTIVATION_START','RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3_SAFE_REGIME_LOG')){if(([regex]::Matches($liveText,[regex]::Escape($marker))).Count-ne 1){throw "Marker count invalid: $marker"}}
    if($liveText.Contains('_rp.get("p_panic", 0.0)')){throw 'Legacy panic-as-breakout logger remains'}
    foreach($path in $Protected){if((Sha $path)-ne $Before[$path]){throw "Protected behavior source changed: $path"}}

    $VerifyBody=@'
[CmdletBinding()]
param([string]$ProjectRoot=(Get-Location).Path)
$ErrorActionPreference='Stop';$root=[IO.Path]::GetFullPath($ProjectRoot);$n=Join-Path $root 'mythos\neural';$live=Join-Path $n 'live.py';$module=Join-Path $n 'rithal_behavior_fix_v105_r3_3.py';$patcher=Join-Path $n 'rithal_behavior_fix_v105_r3_3_install_patch.py';$protected=@((Join-Path $n 'rithal_behavior_fix_v105.py'),(Join-Path $n 'rithal_behavior_fix_v105_r3.py'),(Join-Path $n 'rithal_behavior_fix_v105_r3_1.py'),(Join-Path $n 'rithal_behavior_fix_v105_r3_2.py'))
foreach($p in @($live,$module,$patcher)+$protected){if(-not(Test-Path -LiteralPath $p -PathType Leaf)){throw "missing: $p"}}
foreach($test in @($module,$patcher)){python $test --self-test;if($LASTEXITCODE-ne 0){throw "self-test failed: $test"}}
$compile=@($live,$module,$patcher)+$protected;python -m py_compile @compile;if($LASTEXITCODE-ne 0){throw 'compile failed'}
$t=[IO.File]::ReadAllText($live);foreach($m in @('RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3_ACTIVATION_START','RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3_SAFE_REGIME_LOG')){if(([regex]::Matches($t,[regex]::Escape($m))).Count-ne 1){throw "marker invalid: $m"}};if($t.Contains('_rp.get("p_panic", 0.0)')){throw 'legacy logger remains'}
Write-Host '[RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3] VERIFICATION PASS' -ForegroundColor Green
'@
    [IO.File]::WriteAllText($Verify,$VerifyBody,[Text.UTF8Encoding]::new($false))
    $RollbackBody="`$ErrorActionPreference='Stop';`$m=Get-Content -LiteralPath '$Manifest' -Raw|ConvertFrom-Json;foreach(`$item in @(`$m.files)){if([bool]`$item.existed -and (Test-Path -LiteralPath ([string]`$item.backup)){`$parent=Split-Path -Parent ([string]`$item.target);if(`$parent){New-Item -ItemType Directory -Force -Path `$parent|Out-Null};Copy-Item -LiteralPath ([string]`$item.backup -Destination ([string]`$item.target) -Force}elseif(-not [bool]`$item.existed -and (Test-Path -LiteralPath ([string]`$item.target)){Remove-Item -LiteralPath ([string]`$item.target) -Force -ErrorAction SilentlyContinue}};Write-Host '[RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3] rollback complete' -ForegroundColor Yellow"
    [IO.File]::WriteAllText($Rollback,$RollbackBody,[Text.UTF8Encoding]::new($false))

    $After=@{};foreach($path in $Protected){$After[$path]=Sha $path}
    JsonAtomic $Report ([ordered]@{
        version=$Version;installed_at=(Get-Date).ToUniversalTime().ToString('o');project_root=$Root
        root_cause='Canonical scorer arrays were valid, but V1.0.5 infer still required regime_valid/scalar fields. R3.2 also recomputed edges with default MAE because canonical score output did not expose raw MAE fields.'
        correction='Bridge canonical posterior arrays to the existing fail-closed infer contract; reuse canonical MAE-based edges; retain risk-adjusted side; make logger None-safe and read p_breakout.'
        protected_hashes_before=$Before;protected_hashes_after=$After;checkpoint_files_changed=$false;feature_order_changed=$false;raw_composite_changed=$false;true_rank_regime_arrays_changed=$false;thresholds_changed=$false;fees_changed=$false;tp_sl_changed=$false;sizing_or_authority_settings_changed=$false;backup=$Backup
    })
    Stage 'INSTALLATION PASS' Green
    Write-Host 'Verify: .\Verify-RithalBehaviorFixV1_0_5_R3_3.ps1 -ProjectRoot .' -ForegroundColor Green
}catch{Write-Warning "[$Version] failed: $($_.Exception.Message)";Restore-All;Write-Warning "[$Version] edited files restored from backup.";throw}
