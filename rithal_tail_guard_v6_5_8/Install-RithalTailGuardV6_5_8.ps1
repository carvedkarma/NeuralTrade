[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [switch]$RunBootstrap
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Version = 'RITHAL_TAIL_GUARD_V6_5_8'
$BaseBranch = 'rithal-tail-guard-v6.5.6'
$RebaseBranch = 'rithal-tail-guard-v6.5.7'
$FreshBranch = 'rithal-tail-guard-v6.5.8'
$BaseRaw = "https://raw.githubusercontent.com/carvedkarma/NeuralTrade/$BaseBranch/rithal_tail_guard_v6_5_6"
$RebaseRaw = "https://raw.githubusercontent.com/carvedkarma/NeuralTrade/$RebaseBranch/rithal_tail_guard_v6_5_7"
$FreshRaw = "https://raw.githubusercontent.com/carvedkarma/NeuralTrade/$FreshBranch/rithal_tail_guard_v6_5_8"
$Root = [IO.Path]::GetFullPath($ProjectRoot)
$Incremental = Join-Path $Root 'mythos_incremental_enricher_v6_5.py'
$Guard = Join-Path $Root 'mythos_live_tail_guard_v6_6.py'
$Launcher = Join-Path $Root 'Start-RithalTailWatch.ps1'
$BasePatcher = Join-Path $Root 'rithal_tail_guard_v656_patch.py'
$RebasePatcher = Join-Path $Root 'rithal_tail_guard_v657_delta.py'
$FreshPatcher = Join-Path $Root 'rithal_tail_guard_v658_freshness.py'
$VerifyFile = Join-Path $Root 'Verify-RithalTailGuardV6_5_8.ps1'
$RollbackFile = Join-Path $Root 'Rollback-RithalTailGuardV6_5_8.ps1'
$ReportFile = Join-Path $Root 'data_lake\manifests\rithal_tail_guard_v6_5_8_install_report.json'
$BackupRoot = Join-Path $Root ('rithal_tail_guard_backups\v6_5_8_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
$Manifest = Join-Path $BackupRoot 'backup_manifest.json'

function Stage([string]$Message,[ConsoleColor]$Color=[ConsoleColor]::Cyan){
    Write-Host "[$Version] $Message" -ForegroundColor $Color
}
function Write-JsonAtomic([string]$Path,$Object){
    $parent=Split-Path -Parent $Path
    if($parent){New-Item -ItemType Directory -Force -Path $parent|Out-Null}
    $tmp="$Path.$PID.tmp"
    [IO.File]::WriteAllText($tmp,(($Object|ConvertTo-Json -Depth 100)+[Environment]::NewLine),[Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $tmp -Destination $Path -Force
}
function Restore-Backup {
    if(-not(Test-Path -LiteralPath $Manifest -PathType Leaf)){return}
    $m=Get-Content -LiteralPath $Manifest -Raw|ConvertFrom-Json
    foreach($item in @($m.files)){
        if([bool]$item.existed -and (Test-Path -LiteralPath $item.backup -PathType Leaf)){
            $parent=Split-Path -Parent ([string]$item.target)
            if($parent){New-Item -ItemType Directory -Force -Path $parent|Out-Null}
            Copy-Item -LiteralPath $item.backup -Destination $item.target -Force
        }elseif(-not [bool]$item.existed -and (Test-Path -LiteralPath $item.target)){
            Remove-Item -LiteralPath $item.target -Force -ErrorAction SilentlyContinue
        }
    }
}

if(-not(Test-Path -LiteralPath $Root -PathType Container)){throw "Project root missing: $Root"}
foreach($path in @($Incremental,$Guard,$Launcher)){
    if(-not(Test-Path -LiteralPath $path -PathType Leaf)){throw "Required active source missing: $path"}
}
$Python=(Get-Command python -ErrorAction Stop).Source

Stage 'Stopping only enrichment Python writers belonging to this project.' Yellow
$rootNeedle=$Root.ToLowerInvariant()
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue|Where-Object{
    if($_.Name -notmatch '^python(?:w)?\.exe$'){return $false}
    $cmd=([string]$_.CommandLine).ToLowerInvariant()
    return $cmd.Contains($rootNeedle) -and $cmd -match 'mythos_live_tail_guard_v\d+(?:_\d+)*\.py'
}|ForEach-Object{
    Stage "Stopping enrichment PID=$($_.ProcessId)" Yellow
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Milliseconds 500

New-Item -ItemType Directory -Force -Path $BackupRoot|Out-Null
$targets=@($Incremental,$Guard,$Launcher,$BasePatcher,$RebasePatcher,$FreshPatcher,$VerifyFile,$RollbackFile,$ReportFile)
$records=@()
for($i=0;$i-lt $targets.Count;$i++){
    $target=[string]$targets[$i]
    $exists=Test-Path -LiteralPath $target -PathType Leaf
    $backup=Join-Path $BackupRoot (('{0:D2}_'-f $i)+((Split-Path $target -Leaf)-replace '[^A-Za-z0-9._-]','_'))
    if($exists){Copy-Item -LiteralPath $target -Destination $backup -Force}
    $records+=[pscustomobject]@{target=$target;existed=$exists;backup=$backup}
}
Write-JsonAtomic $Manifest ([pscustomobject]@{
    version=$Version;created_at=(Get-Date).ToUniversalTime().ToString('o');files=$records
})

try{
    Stage 'Downloading the validated V6.5.6 base, V6.5.7 rebase and V6.5.8 freshness delta.'
    Invoke-WebRequest -UseBasicParsing -Uri "$BaseRaw/rithal_tail_guard_v656_patch.py" -OutFile $BasePatcher
    Invoke-WebRequest -UseBasicParsing -Uri "$RebaseRaw/rithal_tail_guard_v657_delta.py" -OutFile $RebasePatcher
    Invoke-WebRequest -UseBasicParsing -Uri "$FreshRaw/rithal_tail_guard_v658_freshness.py" -OutFile $FreshPatcher

    Stage 'Running full-context and anchored-clock regressions.'
    foreach($patcher in @($BasePatcher,$RebasePatcher,$FreshPatcher)){
        $test=& $Python $patcher --self-test 2>&1
        $testCode=$LASTEXITCODE;$testText=[string]::Join("`n",@($test));$test|ForEach-Object{Write-Host $_}
        if($testCode-ne 0 -or -not $testText.Contains('"status": "PASS"')){
            throw "Self-test failed for $patcher with exit code $testCode"
        }
    }

    Stage 'Applying cumulative exact-state repair.'
    $base=& $Python $BasePatcher --incremental $Incremental --guard $Guard --powershell $Launcher 2>&1
    $baseCode=$LASTEXITCODE;$baseText=[string]::Join("`n",@($base));$base|ForEach-Object{Write-Host $_}
    if($baseCode-ne 0 -or -not $baseText.Contains('"status": "PASS"')){throw "V6.5.6 base patch failed: $baseCode"}

    $rebase=& $Python $RebasePatcher --patch --incremental $Incremental --guard $Guard --powershell $Launcher 2>&1
    $rebaseCode=$LASTEXITCODE;$rebaseText=[string]::Join("`n",@($rebase));$rebase|ForEach-Object{Write-Host $_}
    if($rebaseCode-ne 0 -or -not $rebaseText.Contains('"status": "PASS"')){throw "V6.5.7 rebase patch failed: $rebaseCode"}

    $fresh=& $Python $FreshPatcher --patch --guard $Guard 2>&1
    $freshCode=$LASTEXITCODE;$freshText=[string]::Join("`n",@($fresh));$fresh|ForEach-Object{Write-Host $_}
    if($freshCode-ne 0 -or -not $freshText.Contains('"status": "PASS"')){throw "V6.5.8 freshness patch failed: $freshCode"}

    Stage 'Compiling active Python and parsing active PowerShell.'
    & $Python -m py_compile $BasePatcher $RebasePatcher $FreshPatcher $Incremental $Guard
    if($LASTEXITCODE-ne 0){throw "Python compilation failed: $LASTEXITCODE"}
    [void][scriptblock]::Create((Get-Content -LiteralPath $Launcher -Raw))

    $incText=[IO.File]::ReadAllText($Incremental)
    $guardText=[IO.File]::ReadAllText($Guard)
    $psText=[IO.File]::ReadAllText($Launcher)
    if(([regex]::Matches($incText,[regex]::Escape('RITHAL_TAIL_GUARD_V6_5_7_FULL_CONTEXT_LIVE_REBASE'))).Count-ne 1){throw 'Full-context live-rebase marker invalid'}
    if(-not $incText.Contains('INCREMENTAL_LIVE_OVERLAY_WITH_FULL_CONTEXT_REBASE')){throw 'Full-context rebase mode missing'}
    if($incText.Contains('canonical/reference seam parity failed')){throw 'Invalid canonical seam gate remains'}
    if(([regex]::Matches($guardText,[regex]::Escape('RITHAL_TAIL_GUARD_V6_5_8_ANCHORED_FRESHNESS'))).Count-ne 1){throw 'Anchored freshness marker invalid'}
    if(-not $guardText.Contains('anchored_latest_bar_age_min')){throw 'Anchored age telemetry missing'}
    if($guardText.Contains('and float(status.get("latest_bar_age_min", 1e9)) <= float(args.max_latest_age_minutes)')){throw 'Wall-clock age remains authoritative'}
    if(-not $guardText.Contains('and float(status.get("anchored_latest_bar_age_min", 1e9)) <= float(args.max_latest_age_minutes)')){throw 'Anchored age contract missing'}
    if(-not $guardText.Contains('"guard_version": "6.5.8"')){throw 'Guard version projection is not 6.5.8'}
    if(([regex]::Matches($psText,[regex]::Escape('RITHAL_TAIL_GUARD_V6_5_6_FRESHNESS_HANDSHAKE'))).Count-ne 1){throw 'One-bar freshness handshake marker invalid'}

    $VerifyBody=@"
[CmdletBinding()]
param([string]`$ProjectRoot=(Get-Location).Path)
`$ErrorActionPreference='Stop';`$root=[IO.Path]::GetFullPath(`$ProjectRoot);`$inc=Join-Path `$root 'mythos_incremental_enricher_v6_5.py';`$guard=Join-Path `$root 'mythos_live_tail_guard_v6_6.py';`$launcher=Join-Path `$root 'Start-RithalTailWatch.ps1';`$base=Join-Path `$root 'rithal_tail_guard_v656_patch.py';`$rebase=Join-Path `$root 'rithal_tail_guard_v657_delta.py';`$fresh=Join-Path `$root 'rithal_tail_guard_v658_freshness.py'
foreach(`$p in @(`$base,`$rebase,`$fresh)){python `$p --self-test;if(`$LASTEXITCODE-ne 0){throw "self-test failed: `$p"}}
python -m py_compile `$base `$rebase `$fresh `$inc `$guard;if(`$LASTEXITCODE-ne 0){throw 'Python compile failed'}
[void][scriptblock]::Create((Get-Content -LiteralPath `$launcher -Raw))
`$it=[IO.File]::ReadAllText(`$inc);`$gt=[IO.File]::ReadAllText(`$guard);`$pt=[IO.File]::ReadAllText(`$launcher)
if(([regex]::Matches(`$it,[regex]::Escape('RITHAL_TAIL_GUARD_V6_5_7_FULL_CONTEXT_LIVE_REBASE'))).Count-ne 1){throw 'rebase marker invalid'}
if(([regex]::Matches(`$gt,[regex]::Escape('RITHAL_TAIL_GUARD_V6_5_8_ANCHORED_FRESHNESS'))).Count-ne 1){throw 'freshness marker invalid'}
if(`$gt.Contains('and float(status.get("latest_bar_age_min", 1e9)) <= float(args.max_latest_age_minutes)')){throw 'wall-clock age gate remains'}
if(-not `$gt.Contains('and float(status.get("anchored_latest_bar_age_min", 1e9)) <= float(args.max_latest_age_minutes)')){throw 'anchored age gate missing'}
if(-not `$gt.Contains('"guard_version": "6.5.8"')){throw 'guard version invalid'}
if(([regex]::Matches(`$pt,[regex]::Escape('RITHAL_TAIL_GUARD_V6_5_6_FRESHNESS_HANDSHAKE'))).Count-ne 1){throw 'launcher marker invalid'}
Write-Host '[RITHAL_TAIL_GUARD_V6_5_8] VERIFICATION PASS' -ForegroundColor Green
"@
    [IO.File]::WriteAllText($VerifyFile,$VerifyBody,[Text.UTF8Encoding]::new($false))

    $RollbackBody=@"
`$ErrorActionPreference='Stop';`$m=Get-Content -LiteralPath '$Manifest' -Raw|ConvertFrom-Json;foreach(`$item in @(`$m.files)){if([bool]`$item.existed -and (Test-Path -LiteralPath `$item.backup)){`$parent=Split-Path -Parent ([string]`$item.target);if(`$parent){New-Item -ItemType Directory -Force -Path `$parent|Out-Null};Copy-Item -LiteralPath `$item.backup -Destination `$item.target -Force}elseif(-not [bool]`$item.existed -and (Test-Path -LiteralPath `$item.target)){Remove-Item -LiteralPath `$item.target -Force -ErrorAction SilentlyContinue}};Write-Host '[RITHAL_TAIL_GUARD_V6_5_8] rollback complete.' -ForegroundColor Yellow
"@
    [IO.File]::WriteAllText($RollbackFile,$RollbackBody,[Text.UTF8Encoding]::new($false))

    $bootstrapStatus='NOT_RUN'
    if($RunBootstrap){
        Stage 'Running one protected bootstrap with anchored validation and wall-clock rollover retry.'
        & $Launcher -ProjectRoot $Root -Once -Bootstrap
        $bootstrapCode=$LASTEXITCODE
        if($bootstrapCode-ne 0){throw "Bootstrap failed with exit code $bootstrapCode"}
        $tailReport=Join-Path $Root 'data_lake\manifests\live_tail_guard_v6_5_report.json'
        if(-not(Test-Path -LiteralPath $tailReport -PathType Leaf)){throw 'Bootstrap report missing'}
        $tail=Get-Content -LiteralPath $tailReport -Raw|ConvertFrom-Json
        if([string]$tail.publication_transaction-ne 'COMMITTED' -or -not [bool]$tail.tail_feature_contract_ok){
            throw "Bootstrap did not commit anchored contract: publication=$($tail.publication_transaction) tail_ok=$($tail.tail_feature_contract_ok)"
        }
        foreach($name in @('BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','ADAUSDT','AVAXUSDT')){
            $status=$tail.tail_status.$name
            if($null-eq $status){throw "Tail status missing: $name"}
            if([double]$status.anchored_latest_bar_age_min -gt 35.0){throw "Anchored age failed for $name"}
        }
        $bootstrapStatus='COMMITTED_PASS'
    }

    Write-JsonAtomic $ReportFile ([ordered]@{
        version=$Version;installed_at=(Get-Date).ToUniversalTime().ToString('o');project_root=$Root
        root_cause='the build used an immutable start anchor but final age validation used wall-clock completion time; crossing one candle boundary made a valid committed anchor older than 35 minutes and forced exit 2'
        correction='validate feature publication against anchor_now_ms, retain wall-clock age as telemetry, and let the existing one-bar PowerShell handshake own catch-up'
        strict_parity_columns_preserved=$true;canonical_archive_immutable=$true;bootstrap=$bootstrapStatus;backup=$BackupRoot
    })
    Stage 'INSTALLATION PASS' Green
    if($RunBootstrap){Stage 'BOOTSTRAP COMMITTED PASS' Green}
    Write-Host 'Verify: .\Verify-RithalTailGuardV6_5_8.ps1 -ProjectRoot .' -ForegroundColor Green
}catch{
    Write-Warning "[$Version] failed: $($_.Exception.Message)"
    Restore-Backup
    Write-Warning "[$Version] source files were restored from backup."
    throw
}
