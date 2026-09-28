[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [switch]$RunBootstrap
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Version = 'RITHAL_TAIL_GUARD_V6_5_6'
$Branch = 'rithal-tail-guard-v6.5.6'
$RawBase = "https://raw.githubusercontent.com/carvedkarma/NeuralTrade/$Branch/rithal_tail_guard_v6_5_6"
$Root = [IO.Path]::GetFullPath($ProjectRoot)
$Incremental = Join-Path $Root 'mythos_incremental_enricher_v6_5.py'
$Guard = Join-Path $Root 'mythos_live_tail_guard_v6_6.py'
$Launcher = Join-Path $Root 'Start-RithalTailWatch.ps1'
$Patcher = Join-Path $Root 'rithal_tail_guard_v656_patch.py'
$VerifyFile = Join-Path $Root 'Verify-RithalTailGuardV6_5_6.ps1'
$RollbackFile = Join-Path $Root 'Rollback-RithalTailGuardV6_5_6.ps1'
$ReportFile = Join-Path $Root 'data_lake\manifests\rithal_tail_guard_v6_5_6_install_report.json'
$BackupRoot = Join-Path $Root ('rithal_tail_guard_backups\v6_5_6_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
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
$targets=@($Incremental,$Guard,$Launcher,$Patcher,$VerifyFile,$RollbackFile,$ReportFile)
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
    Stage 'Downloading the exact V6.5.6 source patcher.'
    Invoke-WebRequest -UseBasicParsing -Uri "$RawBase/rithal_tail_guard_v656_patch.py" -OutFile $Patcher

    Stage 'Running synthetic stale-baseline, rolling-state, premium and freshness regressions.'
    $test=& $Python $Patcher --self-test 2>&1
    $testCode=$LASTEXITCODE;$testText=[string]::Join("`n",@($test));$test|ForEach-Object{Write-Host $_}
    if($testCode-ne 0 -or -not $testText.Contains('"status": "PASS"')){throw "V6.5.6 self-test failed: $testCode"}

    Stage 'Patching the three exact active files.'
    $patch=& $Python $Patcher --incremental $Incremental --guard $Guard --powershell $Launcher 2>&1
    $patchCode=$LASTEXITCODE;$patchText=[string]::Join("`n",@($patch));$patch|ForEach-Object{Write-Host $_}
    if($patchCode-ne 0 -or -not $patchText.Contains('"status": "PASS"')){throw "V6.5.6 source patch failed: $patchCode"}

    Stage 'Compiling active Python and parsing active PowerShell.'
    & $Python -m py_compile $Patcher $Incremental $Guard
    if($LASTEXITCODE-ne 0){throw "Python compilation failed: $LASTEXITCODE"}
    [void][scriptblock]::Create((Get-Content -LiteralPath $Launcher -Raw))

    $incText=[IO.File]::ReadAllText($Incremental)
    $guardText=[IO.File]::ReadAllText($Guard)
    $psText=[IO.File]::ReadAllText($Launcher)
    if(([regex]::Matches($incText,[regex]::Escape('RITHAL_TAIL_GUARD_V6_5_6_CANONICAL_ROLLING_REBASE'))).Count-ne 1){throw 'Incremental V6.5.6 marker invalid'}
    if(([regex]::Matches($guardText,[regex]::Escape('RITHAL_TAIL_GUARD_V6_5_6_PREMIUM_WARMUP'))).Count-ne 1){throw 'Guard V6.5.6 marker invalid'}
    if(([regex]::Matches($psText,[regex]::Escape('RITHAL_TAIL_GUARD_V6_5_6_FRESHNESS_HANDSHAKE'))).Count-ne 1){throw 'PowerShell V6.5.6 marker invalid'}
    if(-not $incText.Contains('INCREMENTAL_LIVE_OVERLAY_WITH_CANONICAL_REBASE')){throw 'Canonical rebase mode missing'}
    if(-not $guardText.Contains('requested_start_ms - 4 * BAR_15M_MS')){throw 'Premium four-bar warmup missing'}
    if(-not $psText.Contains('Get-OneBarRolloverState')){throw 'One-bar freshness handshake missing'}

    $VerifyBody=@"
[CmdletBinding()]
param([string]`$ProjectRoot=(Get-Location).Path)
`$ErrorActionPreference='Stop';`$root=[IO.Path]::GetFullPath(`$ProjectRoot);`$inc=Join-Path `$root 'mythos_incremental_enricher_v6_5.py';`$guard=Join-Path `$root 'mythos_live_tail_guard_v6_6.py';`$launcher=Join-Path `$root 'Start-RithalTailWatch.ps1';`$patcher=Join-Path `$root 'rithal_tail_guard_v656_patch.py'
python `$patcher --self-test;if(`$LASTEXITCODE-ne 0){throw 'V6.5.6 self-test failed'}
python -m py_compile `$patcher `$inc `$guard;if(`$LASTEXITCODE-ne 0){throw 'Python compile failed'}
[void][scriptblock]::Create((Get-Content -LiteralPath `$launcher -Raw))
`$it=[IO.File]::ReadAllText(`$inc);`$gt=[IO.File]::ReadAllText(`$guard);`$pt=[IO.File]::ReadAllText(`$launcher)
if(([regex]::Matches(`$it,[regex]::Escape('RITHAL_TAIL_GUARD_V6_5_6_CANONICAL_ROLLING_REBASE'))).Count-ne 1){throw 'incremental marker invalid'}
if(([regex]::Matches(`$gt,[regex]::Escape('RITHAL_TAIL_GUARD_V6_5_6_PREMIUM_WARMUP'))).Count-ne 1){throw 'guard marker invalid'}
if(([regex]::Matches(`$pt,[regex]::Escape('RITHAL_TAIL_GUARD_V6_5_6_FRESHNESS_HANDSHAKE'))).Count-ne 1){throw 'launcher marker invalid'}
Write-Host '[RITHAL_TAIL_GUARD_V6_5_6] VERIFICATION PASS' -ForegroundColor Green
"@
    [IO.File]::WriteAllText($VerifyFile,$VerifyBody,[Text.UTF8Encoding]::new($false))

    $RollbackBody=@"
`$ErrorActionPreference='Stop';`$m=Get-Content -LiteralPath '$Manifest' -Raw|ConvertFrom-Json;foreach(`$item in @(`$m.files)){if([bool]`$item.existed -and (Test-Path -LiteralPath `$item.backup)){`$parent=Split-Path -Parent ([string]`$item.target);if(`$parent){New-Item -ItemType Directory -Force -Path `$parent|Out-Null};Copy-Item -LiteralPath `$item.backup -Destination `$item.target -Force}elseif(-not [bool]`$item.existed -and (Test-Path -LiteralPath `$item.target)){Remove-Item -LiteralPath `$item.target -Force -ErrorAction SilentlyContinue}};Write-Host '[RITHAL_TAIL_GUARD_V6_5_6] rollback complete.' -ForegroundColor Yellow
"@
    [IO.File]::WriteAllText($RollbackFile,$RollbackBody,[Text.UTF8Encoding]::new($false))

    $bootstrapStatus='NOT_RUN'
    if($RunBootstrap){
        Stage 'Running one protected bootstrap through the patched freshness handshake.'
        & $Launcher -ProjectRoot $Root -Once -Bootstrap
        $bootstrapCode=$LASTEXITCODE
        if($bootstrapCode-ne 0){throw "Bootstrap failed with exit code $bootstrapCode"}
        $tailReport=Join-Path $Root 'data_lake\manifests\live_tail_guard_v6_5_report.json'
        if(-not(Test-Path -LiteralPath $tailReport -PathType Leaf)){throw 'Bootstrap report missing'}
        $tail=Get-Content -LiteralPath $tailReport -Raw|ConvertFrom-Json
        if([string]$tail.publication_transaction-ne 'COMMITTED' -or -not [bool]$tail.tail_feature_contract_ok){
            throw "Bootstrap did not commit the anchored contract: publication=$($tail.publication_transaction) tail_ok=$($tail.tail_feature_contract_ok)"
        }
        $bootstrapStatus='COMMITTED_PASS'
    }

    Write-JsonAtomic $ReportFile ([ordered]@{
        version=$Version;installed_at=(Get-Date).ToUniversalTime().ToString('o');project_root=$Root
        correction='independent extended-reference proof plus canonical live-overlay rebase; four-bar premium warmup; exact one-bar wall-clock freshness handshake'
        strict_parity_columns_preserved=$true;canonical_archive_immutable=$true;bootstrap=$bootstrapStatus;backup=$BackupRoot
    })
    Stage 'INSTALLATION PASS' Green
    if($RunBootstrap){Stage 'BOOTSTRAP COMMITTED PASS' Green}
    Write-Host 'Verify: .\Verify-RithalTailGuardV6_5_6.ps1 -ProjectRoot .' -ForegroundColor Green
}catch{
    Write-Warning "[$Version] failed: $($_.Exception.Message)"
    Restore-Backup
    Write-Warning "[$Version] all three source files were restored from backup."
    throw
}
