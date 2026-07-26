[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [switch]$RunBootstrap,
    [switch]$StartBackground
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Version = 'RITHAL_TAIL_GUARD_V6_5_1'
$Branch = 'rithal-tail-guard-v6.5.1'
$RawBase = "https://raw.githubusercontent.com/carvedkarma/NeuralTrade/$Branch/rithal_tail_guard_v6_5_1"
$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$Engine = Join-Path $ProjectRoot 'mythos\data_engine_v22.py'
$Launcher = Join-Path $ProjectRoot 'Start-RithalTailWatch.ps1'
$Patcher = Join-Path $ProjectRoot 'rithal_tail_guard_premium_fix_v651.py'
$VerifyFile = Join-Path $ProjectRoot 'Verify-RithalTailGuardV6_5_1.ps1'
$RollbackFile = Join-Path $ProjectRoot 'Rollback-RithalTailGuardV6_5_1.ps1'
$ReportFile = Join-Path $ProjectRoot 'data_lake\manifests\rithal_tail_guard_v6_5_1_install_report.json'
$BackupRoot = Join-Path $ProjectRoot ('rithal_tail_guard_backups\v6_5_1_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
$Manifest = Join-Path $BackupRoot 'backup_manifest.json'

function Write-Stage([string]$Message, [ConsoleColor]$Color = [ConsoleColor]::Cyan) {
    Write-Host "[$Version] $Message" -ForegroundColor $Color
}

function Write-JsonAtomic([string]$Path, $Object) {
    $parent = Split-Path -Parent $Path
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    $temp = "$Path.$PID.tmp"
    [IO.File]::WriteAllText(
        $temp,
        (($Object | ConvertTo-Json -Depth 100) + [Environment]::NewLine),
        [Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $temp -Destination $Path -Force
}

function Resolve-GuardFile {
    $candidates = Get-ChildItem -LiteralPath $ProjectRoot -File -Filter 'mythos_live_tail_guard_v*.py' -ErrorAction SilentlyContinue |
        ForEach-Object {
            if ($_.BaseName -match '^mythos_live_tail_guard_v(?<parts>\d+(?:_\d+)*)$') {
                $numbers = @($Matches.parts.Split('_') | ForEach-Object { [int]$_ })
                while ($numbers.Count -lt 4) { $numbers += 0 }
                [pscustomobject]@{
                    File = $_
                    K0 = $numbers[0]
                    K1 = $numbers[1]
                    K2 = $numbers[2]
                    K3 = $numbers[3]
                }
            }
        } |
        Sort-Object @{Expression='K0';Descending=$true}, @{Expression='K1';Descending=$true}, @{Expression='K2';Descending=$true}, @{Expression='K3';Descending=$true}, @{Expression={$_.File.LastWriteTimeUtc};Descending=$true}
    $selected = @($candidates | Select-Object -First 1)
    if ($selected.Count -ne 1) {
        throw 'No mythos_live_tail_guard_v*.py file was found in the project root.'
    }
    return $selected[0].File.FullName
}

function Restore-Backup {
    if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) { return }
    $data = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
    foreach ($item in @($data.files)) {
        if ([bool]$item.existed -and (Test-Path -LiteralPath $item.backup -PathType Leaf)) {
            Copy-Item -LiteralPath $item.backup -Destination $item.target -Force
        } elseif (-not [bool]$item.existed -and (Test-Path -LiteralPath $item.target)) {
            Remove-Item -LiteralPath $item.target -Force -ErrorAction SilentlyContinue
        }
    }
}

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "Project root not found: $ProjectRoot"
}
foreach ($required in @($Engine, $Launcher)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file missing: $required"
    }
}
$Python = (Get-Command python -ErrorAction Stop).Source
$Guard = Resolve-GuardFile

Write-Stage 'Stopping enrichment writers only; engine and dashboards remain untouched.' Yellow
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -match '^python(?:w)?\.exe$' -and
        [string]$_.CommandLine -match 'mythos_live_tail_(?:fix_v5|guard_v\d+(?:_\d+)*)\.py'
    } |
    ForEach-Object {
        Write-Stage "Stopping enrichment PID=$($_.ProcessId)" Yellow
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
Start-Sleep -Milliseconds 500

New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null
$targets = @($Engine, $Guard, $Patcher, $VerifyFile, $RollbackFile, $ReportFile)
$records = @()
for ($index = 0; $index -lt $targets.Count; $index++) {
    $target = $targets[$index]
    $exists = Test-Path -LiteralPath $target -PathType Leaf
    $backup = Join-Path $BackupRoot (('{0:D2}_' -f $index) + (Split-Path $target -Leaf))
    if ($exists) { Copy-Item -LiteralPath $target -Destination $backup -Force }
    $records += [pscustomobject]@{ target=$target; existed=$exists; backup=$backup }
}
Write-JsonAtomic $Manifest ([pscustomobject]@{
    version=$Version
    created_at=(Get-Date).ToUniversalTime().ToString('o')
    guard=$Guard
    files=$records
})

try {
    Write-Stage "Selected active guard: $Guard"
    Write-Stage 'Downloading the premium-alignment patcher.'
    Invoke-WebRequest -UseBasicParsing -Uri "$RawBase/rithal_tail_guard_premium_fix_v651.py" -OutFile $Patcher

    Write-Stage 'Running deterministic zero-premium, parity, idempotency and compile tests.'
    $selfTest = & $Python $Patcher --self-test 2>&1
    $selfCode = $LASTEXITCODE
    $selfText = [string]::Join("`n", @($selfTest))
    $selfTest | ForEach-Object { Write-Host $_ }
    if ($selfCode -ne 0 -or -not $selfText.Contains('"status": "PASS"')) {
        throw "Patcher self-test failed with exit code $selfCode"
    }

    Write-Stage 'Patching final-aligned premium derivation and coverage semantics.'
    $patchOutput = & $Python $Patcher --engine $Engine --guard $Guard 2>&1
    $patchCode = $LASTEXITCODE
    $patchText = [string]::Join("`n", @($patchOutput))
    $patchOutput | ForEach-Object { Write-Host $_ }
    if ($patchCode -ne 0 -or -not $patchText.Contains('"status": "PASS"')) {
        throw "Source patch failed with exit code $patchCode"
    }

    Write-Stage 'Compiling the exact active engine and guard.'
    & $Python -m py_compile $Patcher $Engine $Guard
    if ($LASTEXITCODE -ne 0) { throw "Python compilation failed: $LASTEXITCODE" }

    $engineText = [IO.File]::ReadAllText($Engine)
    $guardText = [IO.File]::ReadAllText($Guard)
    if (($engineText.Split('RITHAL_TAIL_GUARD_V6_5_1_PREMIUM_ALIGNMENT').Count - 1) -ne 1) {
        throw 'Engine premium-alignment marker count is not exactly one.'
    }
    if (-not $engineText.Contains('premium_source_covered')) {
        throw 'Engine coverage-source logic is missing.'
    }
    if (-not $engineText.Contains('pct_change(periods=4, fill_method=None)')) {
        throw 'Engine final aligned 1h premium derivation is missing.'
    }
    if (-not $guardText.Contains('pct_change(periods=4, fill_method=None)')) {
        throw 'Guard explicit no-fill premium derivation is missing.'
    }

    $VerifyBody = @"
[CmdletBinding()]
param([string]`$ProjectRoot=(Get-Location).Path)
`$ErrorActionPreference='Stop'
`$root=[IO.Path]::GetFullPath(`$ProjectRoot)
`$engine=Join-Path `$root 'mythos\data_engine_v22.py'
`$patcher=Join-Path `$root 'rithal_tail_guard_premium_fix_v651.py'
`$guard=Get-ChildItem -LiteralPath `$root -File -Filter 'mythos_live_tail_guard_v*.py' | Where-Object { [IO.File]::ReadAllText(`$_.FullName).Contains('RITHAL_TAIL_GUARD_V6_5_1_PREMIUM_FETCH') } | Select-Object -First 1
if(`$null -eq `$guard){throw 'V6.5.1 patched guard not found'}
python `$patcher --self-test
if(`$LASTEXITCODE -ne 0){throw 'self-test failed'}
python -m py_compile `$patcher `$engine `$guard.FullName
if(`$LASTEXITCODE -ne 0){throw 'compile failed'}
`$et=[IO.File]::ReadAllText(`$engine);`$gt=[IO.File]::ReadAllText(`$guard.FullName)
if((`$et.Split('RITHAL_TAIL_GUARD_V6_5_1_PREMIUM_ALIGNMENT').Count-1) -ne 1){throw 'engine marker invalid'}
if(-not `$et.Contains('premium_source_covered') -or -not `$et.Contains('pct_change(periods=4, fill_method=None)')){throw 'engine premium contract invalid'}
if(-not `$gt.Contains('pct_change(periods=4, fill_method=None)')){throw 'guard premium contract invalid'}
Write-Host '[RITHAL_TAIL_GUARD_V6_5_1] VERIFICATION PASS' -ForegroundColor Green
"@
    [IO.File]::WriteAllText($VerifyFile, $VerifyBody, [Text.UTF8Encoding]::new($false))

    $RollbackBody = @"
`$ErrorActionPreference='Stop'
`$manifest=Get-Content -LiteralPath '$Manifest' -Raw|ConvertFrom-Json
foreach(`$item in @(`$manifest.files)){
    if([bool]`$item.existed -and (Test-Path -LiteralPath `$item.backup)){Copy-Item -LiteralPath `$item.backup -Destination `$item.target -Force}
    elseif(-not [bool]`$item.existed -and (Test-Path -LiteralPath `$item.target)){Remove-Item -LiteralPath `$item.target -Force -ErrorAction SilentlyContinue}
}
Write-Host '[RITHAL_TAIL_GUARD_V6_5_1] rollback complete.' -ForegroundColor Yellow
"@
    [IO.File]::WriteAllText($RollbackFile, $RollbackBody, [Text.UTF8Encoding]::new($false))

    $report = [ordered]@{
        version=$Version
        installed_at=(Get-Date).ToUniversalTime().ToString('o')
        project_root=$ProjectRoot
        engine=$Engine
        guard=$Guard
        root_cause='precomputed premium_index_change_1h was carried through an independent asof alignment; nonzero value was incorrectly used as premium coverage'
        correction='derive 1h change from final aligned premium close; use premium_source_covered; preserve genuine zero premium observations'
        code_verification='PASS'
        bootstrap_requested=[bool]$RunBootstrap
        bootstrap_result='NOT_RUN'
        backup=$BackupRoot
    }

    if ($RunBootstrap) {
        Write-Stage 'Running one bootstrap/catch-up transaction. Existing files remain protected by the guard rollback.'
        & $Launcher -ProjectRoot $ProjectRoot -Once -Bootstrap
        $bootstrapCode = $LASTEXITCODE
        $report.bootstrap_exit_code = $bootstrapCode
        $report.bootstrap_result = if ($bootstrapCode -eq 0) { 'PASS' } else { 'FAILED_PRESERVED_PATCH' }
        if ($bootstrapCode -eq 0) {
            Write-Stage 'Bootstrap and 96-row publication contract PASS.' Green
        } else {
            Write-Stage "Code fix installed, but bootstrap returned exit code $bootstrapCode. Patched code was preserved for diagnosis." Yellow
        }
    }

    Write-JsonAtomic $ReportFile $report
    Write-Stage 'INSTALLATION PASS' Green
    Write-Host 'Verify: .\Verify-RithalTailGuardV6_5_1.ps1 -ProjectRoot .' -ForegroundColor Green

    if ($StartBackground) {
        if ($RunBootstrap -and $report.bootstrap_result -ne 'PASS') {
            Write-Stage 'Background watcher was not started because bootstrap did not pass.' Yellow
        } else {
            Write-Stage 'Starting the continuous 15-minute watcher in a separate window.'
            & $Launcher -ProjectRoot $ProjectRoot -Background -NoRunNow
        }
    }
} catch {
    Write-Warning "[$Version] installation failed: $($_.Exception.Message)"
    Restore-Backup
    throw
}
