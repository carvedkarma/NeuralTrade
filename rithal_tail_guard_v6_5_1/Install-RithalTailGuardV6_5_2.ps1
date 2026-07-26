[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [switch]$StartBackground
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Version = 'RITHAL_TAIL_GUARD_V6_5_2'
$Branch = 'rithal-tail-guard-v6.5.1'
$RawBase = "https://raw.githubusercontent.com/carvedkarma/NeuralTrade/$Branch/rithal_tail_guard_v6_5_1"
$Root = [IO.Path]::GetFullPath($ProjectRoot)
$Engine = Join-Path $Root 'mythos\data_engine_v22.py'
$Launcher = Join-Path $Root 'Start-RithalTailWatch.ps1'
$Patcher = Join-Path $Root 'rithal_tail_guard_premium_fix_v652.py'
$VerifyFile = Join-Path $Root 'Verify-RithalTailGuardV6_5_2.ps1'
$RollbackFile = Join-Path $Root 'Rollback-RithalTailGuardV6_5_2.ps1'
$ReportFile = Join-Path $Root 'data_lake\manifests\rithal_tail_guard_v6_5_2_install_report.json'
$BackupRoot = Join-Path $Root ('rithal_tail_guard_backups\v6_5_2_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
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
function Resolve-GuardFile {
    $items=Get-ChildItem -LiteralPath $Root -File -Filter 'mythos_live_tail_guard_v*.py' -ErrorAction SilentlyContinue|ForEach-Object{
        if($_.BaseName -match '^mythos_live_tail_guard_v(?<parts>\d+(?:_\d+)*)$'){
            $n=@($Matches.parts.Split('_')|ForEach-Object{[int]$_});while($n.Count-lt 5){$n+=0}
            [pscustomobject]@{File=$_;K0=$n[0];K1=$n[1];K2=$n[2];K3=$n[3];K4=$n[4]}
        }
    }|Sort-Object @{Expression='K0';Descending=$true},@{Expression='K1';Descending=$true},@{Expression='K2';Descending=$true},@{Expression='K3';Descending=$true},@{Expression='K4';Descending=$true},@{Expression={$_.File.LastWriteTimeUtc};Descending=$true}
    $selected=@($items|Select-Object -First 1)
    if($selected.Count-ne 1){throw 'No mythos_live_tail_guard_v*.py was found.'}
    return $selected[0].File.FullName
}
function Restore-Backup {
    if(-not(Test-Path -LiteralPath $Manifest -PathType Leaf)){return}
    $m=Get-Content -LiteralPath $Manifest -Raw|ConvertFrom-Json
    foreach($item in @($m.files)){
        if([bool]$item.existed -and (Test-Path -LiteralPath $item.backup -PathType Leaf)){
            Copy-Item -LiteralPath $item.backup -Destination $item.target -Force
        }elseif(-not [bool]$item.existed -and (Test-Path -LiteralPath $item.target)){
            Remove-Item -LiteralPath $item.target -Force -ErrorAction SilentlyContinue
        }
    }
}

if(-not(Test-Path -LiteralPath $Root -PathType Container)){throw "Project root missing: $Root"}
foreach($path in @($Engine,$Launcher)){
    if(-not(Test-Path -LiteralPath $path -PathType Leaf)){throw "Required file missing: $path"}
}
$Python=(Get-Command python -ErrorAction Stop).Source
$Guard=Resolve-GuardFile

Stage 'Stopping enrichment writers only. Trading engine and dashboards remain untouched.' Yellow
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue|Where-Object{
    $_.Name -match '^python(?:w)?\.exe$' -and [string]$_.CommandLine -match 'mythos_live_tail_(?:fix_v5|guard_v\d+(?:_\d+)*)\.py'
}|ForEach-Object{
    Stage "Stopping enrichment PID=$($_.ProcessId)" Yellow
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Milliseconds 500

New-Item -ItemType Directory -Force -Path $BackupRoot|Out-Null
$targets=@($Engine,$Guard,$Patcher,$VerifyFile,$RollbackFile,$ReportFile)
$records=@()
for($i=0;$i-lt $targets.Count;$i++){
    $target=$targets[$i];$exists=Test-Path -LiteralPath $target -PathType Leaf
    $backup=Join-Path $BackupRoot (('{0:D2}_'-f $i)+(Split-Path $target -Leaf))
    if($exists){Copy-Item -LiteralPath $target -Destination $backup -Force}
    $records+=[pscustomobject]@{target=$target;existed=$exists;backup=$backup}
}
Write-JsonAtomic $Manifest ([pscustomobject]@{version=$Version;created_at=(Get-Date).ToUniversalTime().ToString('o');guard=$Guard;files=$records})

try{
    Stage "Selected active guard: $Guard"
    Invoke-WebRequest -UseBasicParsing -Uri "$RawBase/rithal_tail_guard_premium_fix_v652.py" -OutFile $Patcher

    Stage 'Running canonical V22 and bounded-window regression tests.'
    $test=& $Python $Patcher --self-test 2>&1
    $testCode=$LASTEXITCODE;$testText=[string]::Join("`n",@($test));$test|ForEach-Object{Write-Host $_}
    if($testCode-ne 0 -or -not $testText.Contains('"status": "PASS"')){throw "V6.5.2 self-test failed: $testCode"}

    Stage 'Restoring frozen V22 premium semantics and adding four hidden fetch warm-up bars.'
    $patch=& $Python $Patcher --engine $Engine --guard $Guard 2>&1
    $patchCode=$LASTEXITCODE;$patchText=[string]::Join("`n",@($patch));$patch|ForEach-Object{Write-Host $_}
    if($patchCode-ne 0 -or -not $patchText.Contains('"status": "PASS"')){throw "V6.5.2 source patch failed: $patchCode"}

    Stage 'Compiling the exact active engine and guard.'
    & $Python -m py_compile $Patcher $Engine $Guard
    if($LASTEXITCODE-ne 0){throw "Python compile failed: $LASTEXITCODE"}

    $engineText=[IO.File]::ReadAllText($Engine);$guardText=[IO.File]::ReadAllText($Guard)
    if(([regex]::Matches($engineText,[regex]::Escape('RITHAL_TAIL_GUARD_V6_5_2_CANONICAL_PREMIUM'))).Count-ne 1){throw 'Engine V6.5.2 marker invalid'}
    if($engineText.Contains('RITHAL_TAIL_GUARD_V6_5_1_PREMIUM_ALIGNMENT')){throw 'V6.5.1 engine semantics remain active'}
    if(([regex]::Matches($guardText,[regex]::Escape('RITHAL_TAIL_GUARD_V6_5_2_PREMIUM_WARMUP'))).Count-ne 1){throw 'Guard V6.5.2 marker invalid'}
    if(-not $guardText.Contains('_requested_start_ms - 4 * BAR_15M_MS')){throw 'Four-bar warmup is missing'}
    if(-not $guardText.Contains('out = out.loc[out["timestamp"] >= _requested_start_ms]')){throw 'Post-derivative trim is missing'}

    Stage 'Running one protected bootstrap. Installation is not considered successful unless publication commits.'
    & $Launcher -ProjectRoot $Root -Once -Bootstrap
    $bootstrapCode=$LASTEXITCODE
    if($bootstrapCode-ne 0){throw "Bootstrap/publication failed with exit code $bootstrapCode"}

    $tailReport=Join-Path $Root 'data_lake\manifests\live_tail_guard_v6_5_report.json'
    if(-not(Test-Path -LiteralPath $tailReport -PathType Leaf)){throw "Tail report missing after bootstrap: $tailReport"}
    $tail=Get-Content -LiteralPath $tailReport -Raw|ConvertFrom-Json
    $publication=[string]$tail.publication
    $tailOk=[bool]$tail.tail_feature_contract_ok
    if($publication -ne 'COMMITTED' -or -not $tailOk){throw "Bootstrap returned zero but publication contract is not committed: publication=$publication tail_ok=$tailOk"}

    $VerifyBody=@"
[CmdletBinding()]
param([string]`$ProjectRoot=(Get-Location).Path)
`$ErrorActionPreference='Stop';`$root=[IO.Path]::GetFullPath(`$ProjectRoot);`$engine=Join-Path `$root 'mythos\data_engine_v22.py';`$patcher=Join-Path `$root 'rithal_tail_guard_premium_fix_v652.py';`$guard=Get-ChildItem -LiteralPath `$root -File -Filter 'mythos_live_tail_guard_v*.py'|Where-Object{[IO.File]::ReadAllText(`$_.FullName).Contains('RITHAL_TAIL_GUARD_V6_5_2_PREMIUM_WARMUP')}|Select-Object -First 1
if(`$null-eq `$guard){throw 'V6.5.2 guard not found'}
python `$patcher --self-test;if(`$LASTEXITCODE-ne 0){throw 'self-test failed'}
python -m py_compile `$patcher `$engine `$guard.FullName;if(`$LASTEXITCODE-ne 0){throw 'compile failed'}
`$et=[IO.File]::ReadAllText(`$engine);`$gt=[IO.File]::ReadAllText(`$guard.FullName)
if(([regex]::Matches(`$et,[regex]::Escape('RITHAL_TAIL_GUARD_V6_5_2_CANONICAL_PREMIUM'))).Count-ne 1){throw 'engine marker invalid'}
if(`$et.Contains('RITHAL_TAIL_GUARD_V6_5_1_PREMIUM_ALIGNMENT')){throw 'V6.5.1 semantics remain'}
if(([regex]::Matches(`$gt,[regex]::Escape('RITHAL_TAIL_GUARD_V6_5_2_PREMIUM_WARMUP'))).Count-ne 1){throw 'guard marker invalid'}
`$report=Get-Content -LiteralPath (Join-Path `$root 'data_lake\manifests\live_tail_guard_v6_5_report.json') -Raw|ConvertFrom-Json
if([string]`$report.publication-ne 'COMMITTED' -or -not [bool]`$report.tail_feature_contract_ok){throw 'latest publication contract is not committed'}
Write-Host '[RITHAL_TAIL_GUARD_V6_5_2] VERIFICATION PASS' -ForegroundColor Green
"@
    [IO.File]::WriteAllText($VerifyFile,$VerifyBody,[Text.UTF8Encoding]::new($false))

    $RollbackBody=@"
`$ErrorActionPreference='Stop';`$m=Get-Content -LiteralPath '$Manifest' -Raw|ConvertFrom-Json;foreach(`$item in @(`$m.files)){if([bool]`$item.existed -and (Test-Path -LiteralPath `$item.backup)){Copy-Item -LiteralPath `$item.backup -Destination `$item.target -Force}elseif(-not [bool]`$item.existed -and (Test-Path -LiteralPath `$item.target)){Remove-Item -LiteralPath `$item.target -Force -ErrorAction SilentlyContinue}};Write-Host '[RITHAL_TAIL_GUARD_V6_5_2] rollback complete.' -ForegroundColor Yellow
"@
    [IO.File]::WriteAllText($RollbackFile,$RollbackBody,[Text.UTF8Encoding]::new($false))

    Write-JsonAtomic $ReportFile ([ordered]@{
        version=$Version;installed_at=(Get-Date).ToUniversalTime().ToString('o');project_root=$Root;engine=$Engine;guard=$Guard
        root_cause='bounded premium fetch calculated pct_change without the four raw warmup bars while the immutable full archive had those predecessors'
        correction='restore frozen V22 raw-derivative semantics; fetch four hidden bars; derive; trim; retain exact parity gate'
        bootstrap_exit_code=0;publication='COMMITTED';tail_feature_contract_ok=$true;backup=$BackupRoot
    })
    Stage 'INSTALLATION AND BOOTSTRAP PASS' Green
    Write-Host 'Verify: .\Verify-RithalTailGuardV6_5_2.ps1 -ProjectRoot .' -ForegroundColor Green

    if($StartBackground){
        Stage 'Starting the continuous watcher after successful committed bootstrap.'
        & $Launcher -ProjectRoot $Root -Background -NoRunNow
    }
}catch{
    Write-Warning "[$Version] failed: $($_.Exception.Message)"
    Restore-Backup
    Write-Warning "[$Version] source files restored from backup; existing enrichment publication remained protected."
    throw
}
