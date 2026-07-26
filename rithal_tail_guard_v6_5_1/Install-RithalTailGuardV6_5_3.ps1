[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [switch]$StartBackground
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Version = 'RITHAL_TAIL_GUARD_V6_5_3'
$Branch = 'rithal-tail-guard-v6.5.1'
$RawBase = "https://raw.githubusercontent.com/carvedkarma/NeuralTrade/$Branch/rithal_tail_guard_v6_5_1"
$Root = [IO.Path]::GetFullPath($ProjectRoot)
$DataRoot = Join-Path $Root 'data_lake'
$Engine = Join-Path $Root 'mythos\data_engine_v22.py'
$Launcher = Join-Path $Root 'Start-RithalTailWatch.ps1'
$Patcher = Join-Path $Root 'rithal_tail_guard_premium_fix_v653.py'
$VerifyFile = Join-Path $Root 'Verify-RithalTailGuardV6_5_3.ps1'
$RollbackFile = Join-Path $Root 'Rollback-RithalTailGuardV6_5_3.ps1'
$ReportFile = Join-Path $DataRoot 'manifests\rithal_tail_guard_v6_5_3_install_report.json'
$BackupRoot = Join-Path $Root ('rithal_tail_guard_backups\v6_5_3_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
$Manifest = Join-Path $BackupRoot 'backup_manifest.json'
$TempPatcher = Join-Path ([IO.Path]::GetTempPath()) ("rithal_tail_guard_premium_fix_v653_$PID.py")
$Symbols = 'BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,ADAUSDT,AVAXUSDT'

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
function Resolve-GuardFile {
    $items=Get-ChildItem -LiteralPath $Root -File -Filter 'mythos_live_tail_guard_v*.py' -ErrorAction SilentlyContinue|ForEach-Object{
        if($_.BaseName -match '^mythos_live_tail_guard_v(?<parts>\d+(?:_\d+)*)$'){
            $n=@($Matches.parts.Split('_')|ForEach-Object{[int]$_});while($n.Count-lt 6){$n+=0}
            [pscustomobject]@{File=$_;K0=$n[0];K1=$n[1];K2=$n[2];K3=$n[3];K4=$n[4];K5=$n[5]}
        }
    }|Sort-Object @{Expression='K0';Descending=$true},@{Expression='K1';Descending=$true},@{Expression='K2';Descending=$true},@{Expression='K3';Descending=$true},@{Expression='K4';Descending=$true},@{Expression='K5';Descending=$true},@{Expression={$_.File.LastWriteTimeUtc};Descending=$true}
    $selected=@($items|Select-Object -First 1)
    if($selected.Count-ne 1){throw 'No mythos_live_tail_guard_v*.py was found.'}
    return $selected[0].File.FullName
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
foreach($path in @($Engine,$Launcher,$DataRoot)){
    if(-not(Test-Path -LiteralPath $path)){throw "Required path missing: $path"}
}
$Python=(Get-Command python -ErrorAction Stop).Source
$Guard=Resolve-GuardFile

try{
    Stage 'Downloading the V6.5.3 repair into a temporary preflight location.'
    Invoke-WebRequest -UseBasicParsing -Uri "$RawBase/rithal_tail_guard_premium_fix_v653.py" -OutFile $TempPatcher

    Stage 'Running stale-boundary, full-state, idempotency and compile regressions.'
    $test=& $Python $TempPatcher --self-test 2>&1
    $testCode=$LASTEXITCODE;$testText=[string]::Join("`n",@($test));$test|ForEach-Object{Write-Host $_}
    if($testCode-ne 0 -or -not $testText.Contains('"status": "PASS"')){throw "V6.5.3 self-test failed: $testCode"}

    Stage "Resolving the six exact raw premium files through the active guard: $Guard"
    $pathOutput=& $Python $TempPatcher --list-premium-paths --guard $Guard --data-root $DataRoot --symbols $Symbols 2>&1
    $pathCode=$LASTEXITCODE;$pathText=[string]::Join("`n",@($pathOutput))
    if($pathCode-ne 0){throw "Raw premium path resolution failed: $pathText"}
    $premiumMap=$pathText|ConvertFrom-Json
    $premiumPaths=@()
    foreach($symbol in $Symbols.Split(',')){
        $p=[string]$premiumMap.$symbol
        if([string]::IsNullOrWhiteSpace($p)){throw "Raw premium path missing for $symbol"}
        if(-not(Test-Path -LiteralPath $p -PathType Leaf)){throw "Raw premium parquet missing for $symbol`: $p"}
        $premiumPaths+=$p
    }

    Stage 'Stopping only enrichment writers belonging to this project root.' Yellow
    $rootNeedle=$Root.ToLowerInvariant();$guardNeedle=$Guard.ToLowerInvariant()
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue|Where-Object{
        if($_.Name -notmatch '^python(?:w)?\.exe$'){return $false}
        $cmd=([string]$_.CommandLine).ToLowerInvariant()
        return $cmd.Contains($guardNeedle) -or ($cmd.Contains($rootNeedle) -and $cmd -match 'mythos_live_tail_guard_v\d+(?:_\d+)*\.py')
    }|ForEach-Object{
        Stage "Stopping project enrichment PID=$($_.ProcessId)" Yellow
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 500

    New-Item -ItemType Directory -Force -Path $BackupRoot|Out-Null
    $targets=@($Engine,$Guard,$Patcher,$VerifyFile,$RollbackFile,$ReportFile)+$premiumPaths
    $records=@()
    for($i=0;$i-lt $targets.Count;$i++){
        $target=[string]$targets[$i];$exists=Test-Path -LiteralPath $target -PathType Leaf
        $safeName=((Split-Path $target -Leaf) -replace '[^A-Za-z0-9._-]','_')
        $backup=Join-Path $BackupRoot (('{0:D2}_'-f $i)+$safeName)
        if($exists){Copy-Item -LiteralPath $target -Destination $backup -Force}
        $records+=[pscustomobject]@{target=$target;existed=$exists;backup=$backup}
    }
    Write-JsonAtomic $Manifest ([pscustomobject]@{version=$Version;created_at=(Get-Date).ToUniversalTime().ToString('o');guard=$Guard;files=$records})
    Copy-Item -LiteralPath $TempPatcher -Destination $Patcher -Force

    Stage 'Restoring the frozen V22 engine contract and future four-bar fetch warm-up.'
    $patch=& $Python $Patcher --patch-source --engine $Engine --guard $Guard 2>&1
    $patchCode=$LASTEXITCODE;$patchText=[string]::Join("`n",@($patch));$patch|ForEach-Object{Write-Host $_}
    if($patchCode-ne 0 -or -not $patchText.Contains('"status": "PASS"')){throw "V6.5.3 source patch failed: $patchCode"}

    Stage 'Compiling the exact active engine, guard and repair utility.'
    & $Python -m py_compile $Patcher $Engine $Guard
    if($LASTEXITCODE-ne 0){throw "Python compile failed: $LASTEXITCODE"}

    $engineText=[IO.File]::ReadAllText($Engine);$guardText=[IO.File]::ReadAllText($Guard)
    if(([regex]::Matches($engineText,[regex]::Escape('RITHAL_TAIL_GUARD_V6_5_3_CANONICAL_PREMIUM'))).Count-ne 1){throw 'Engine V6.5.3 marker invalid'}
    if($engineText.Contains('RITHAL_TAIL_GUARD_V6_5_1_PREMIUM_ALIGNMENT')){throw 'V6.5.1 engine semantics remain'}
    if(([regex]::Matches($guardText,[regex]::Escape('RITHAL_TAIL_GUARD_V6_5_3_PREMIUM_WARMUP'))).Count-ne 1){throw 'Guard V6.5.3 marker invalid'}

    Stage 'Canonicalizing the complete existing raw premium derivative state once.'
    $canon=& $Python $Patcher --canonicalize-premium-state --guard $Guard --data-root $DataRoot --symbols $Symbols 2>&1
    $canonCode=$LASTEXITCODE;$canonText=[string]::Join("`n",@($canon));$canon|ForEach-Object{Write-Host $_}
    if($canonCode-ne 0 -or -not $canonText.Contains('"status": "PASS"')){throw "Premium-state canonicalization failed: $canonCode"}
    $canonicalization=$canonText|ConvertFrom-Json

    Stage 'Running one protected bootstrap; success requires a committed publication.'
    & $Launcher -ProjectRoot $Root -Once -Bootstrap
    $bootstrapCode=$LASTEXITCODE
    if($bootstrapCode-ne 0){throw "Bootstrap/publication failed with exit code $bootstrapCode"}

    $tailReport=Join-Path $DataRoot 'manifests\live_tail_guard_v6_5_report.json'
    if(-not(Test-Path -LiteralPath $tailReport -PathType Leaf)){throw "Tail report missing after bootstrap: $tailReport"}
    $tail=Get-Content -LiteralPath $tailReport -Raw|ConvertFrom-Json
    $publication=[string]$tail.publication
    $tailOk=[bool]$tail.tail_feature_contract_ok
    if($publication-ne 'COMMITTED' -or -not $tailOk){throw "Publication contract not committed: publication=$publication tail_ok=$tailOk"}

    $VerifyBody=@"
[CmdletBinding()]
param([string]`$ProjectRoot=(Get-Location).Path)
`$ErrorActionPreference='Stop';`$root=[IO.Path]::GetFullPath(`$ProjectRoot);`$engine=Join-Path `$root 'mythos\data_engine_v22.py';`$patcher=Join-Path `$root 'rithal_tail_guard_premium_fix_v653.py'
`$guard=Get-ChildItem -LiteralPath `$root -File -Filter 'mythos_live_tail_guard_v*.py'|Where-Object{[IO.File]::ReadAllText(`$_.FullName).Contains('RITHAL_TAIL_GUARD_V6_5_3_PREMIUM_WARMUP')}|Select-Object -First 1
if(`$null-eq `$guard){throw 'V6.5.3 guard not found'}
python `$patcher --self-test;if(`$LASTEXITCODE-ne 0){throw 'self-test failed'}
python -m py_compile `$patcher `$engine `$guard.FullName;if(`$LASTEXITCODE-ne 0){throw 'compile failed'}
`$et=[IO.File]::ReadAllText(`$engine);`$gt=[IO.File]::ReadAllText(`$guard.FullName)
if(([regex]::Matches(`$et,[regex]::Escape('RITHAL_TAIL_GUARD_V6_5_3_CANONICAL_PREMIUM'))).Count-ne 1){throw 'engine marker invalid'}
if(([regex]::Matches(`$gt,[regex]::Escape('RITHAL_TAIL_GUARD_V6_5_3_PREMIUM_WARMUP'))).Count-ne 1){throw 'guard marker invalid'}
`$report=Get-Content -LiteralPath (Join-Path `$root 'data_lake\manifests\live_tail_guard_v6_5_report.json') -Raw|ConvertFrom-Json
if([string]`$report.publication-ne 'COMMITTED' -or -not [bool]`$report.tail_feature_contract_ok){throw 'latest publication is not committed'}
Write-Host '[RITHAL_TAIL_GUARD_V6_5_3] VERIFICATION PASS' -ForegroundColor Green
"@
    [IO.File]::WriteAllText($VerifyFile,$VerifyBody,[Text.UTF8Encoding]::new($false))

    $RollbackBody=@"
`$ErrorActionPreference='Stop';`$m=Get-Content -LiteralPath '$Manifest' -Raw|ConvertFrom-Json;foreach(`$item in @(`$m.files)){if([bool]`$item.existed -and (Test-Path -LiteralPath `$item.backup)){`$parent=Split-Path -Parent ([string]`$item.target);if(`$parent){New-Item -ItemType Directory -Force -Path `$parent|Out-Null};Copy-Item -LiteralPath `$item.backup -Destination `$item.target -Force}elseif(-not [bool]`$item.existed -and (Test-Path -LiteralPath `$item.target)){Remove-Item -LiteralPath `$item.target -Force -ErrorAction SilentlyContinue}};Write-Host '[RITHAL_TAIL_GUARD_V6_5_3] rollback complete.' -ForegroundColor Yellow
"@
    [IO.File]::WriteAllText($RollbackFile,$RollbackBody,[Text.UTF8Encoding]::new($false))

    Write-JsonAtomic $ReportFile ([ordered]@{
        version=$Version;installed_at=(Get-Date).ToUniversalTime().ToString('o');project_root=$Root;engine=$Engine;guard=$Guard
        root_cause='older chunk-boundary derivative values remained persisted in the raw premium parquet; four-bar warmup corrected only the newest fetch'
        correction='one-time full raw premium derivative canonicalization from the unchanged close series plus four hidden predecessor bars for all future incremental fetches'
        canonicalization=$canonicalization;bootstrap_exit_code=0;publication='COMMITTED';tail_feature_contract_ok=$true;backup=$BackupRoot
    })

    Stage 'INSTALLATION AND BOOTSTRAP PASS' Green
    Write-Host 'Verify: .\Verify-RithalTailGuardV6_5_3.ps1 -ProjectRoot .' -ForegroundColor Green
    if($StartBackground){
        Stage 'Starting continuous enrichment after committed bootstrap.'
        & $Launcher -ProjectRoot $Root -Background -NoRunNow
    }
}catch{
    Write-Warning "[$Version] failed: $($_.Exception.Message)"
    Restore-Backup
    Write-Warning "[$Version] engine, guard and all six raw premium files were restored from backup."
    throw
}finally{
    Remove-Item -LiteralPath $TempPatcher -Force -ErrorAction SilentlyContinue
}
