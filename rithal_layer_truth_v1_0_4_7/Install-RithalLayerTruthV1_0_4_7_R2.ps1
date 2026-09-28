param(
    [Parameter(Mandatory = $false)]
    [string]$ProjectRoot = (Get-Location).Path
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Version = 'RITHAL_LAYER_TRUTH_V1_0_4_7_R2'
$Branch = 'rithal-layer-truth-v1.0.4.7'
$RawBase = "https://raw.githubusercontent.com/carvedkarma/NeuralTrade/$Branch/rithal_layer_truth_v1_0_4_7"
$Root = [System.IO.Path]::GetFullPath($ProjectRoot)
$BaseInstaller = Join-Path $Root 'Install-RithalLayerTruthV1_0_4_7.ps1'
$Runtime = Join-Path $Root 'mythos\neural\rithal_layer_truth_v1047.py'
$Patcher = Join-Path $Root 'rithal_layer_truth_v1_0_4_7\apply_rithal_layer_truth_v1047_r2.py'

Write-Host "[$Version] Downloading transactional base installer" -ForegroundColor Cyan
Invoke-WebRequest -UseBasicParsing -Uri "$RawBase/Install-RithalLayerTruthV1_0_4_7.ps1" -OutFile $BaseInstaller
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $BaseInstaller -ProjectRoot $Root
if ($LASTEXITCODE -ne 0) { throw "Base installer failed with exit code $LASTEXITCODE" }

New-Item -ItemType Directory -Path (Split-Path -Parent $Patcher) -Force | Out-Null
Write-Host "[$Version] Downloading deterministic R2 patcher" -ForegroundColor Cyan
Invoke-WebRequest -UseBasicParsing -Uri "$RawBase/apply_rithal_layer_truth_v1047_r2.py" -OutFile $Patcher

Write-Host "[$Version] Applying exact-binding, regime, and sizing hotfixes" -ForegroundColor Cyan
& python $Patcher $Runtime
if ($LASTEXITCODE -ne 0) { throw "R2 patcher failed with exit code $LASTEXITCODE" }

Write-Host "[$Version] Compiling and verifying installed runtime" -ForegroundColor Cyan
& python -m py_compile $Runtime
if ($LASTEXITCODE -ne 0) { throw "Runtime compilation failed with exit code $LASTEXITCODE" }
& python $Runtime --project-root $Root --verify
if ($LASTEXITCODE -ne 0) { throw "Runtime verification failed with exit code $LASTEXITCODE" }

Write-Host "[$Version] INSTALLATION PASS" -ForegroundColor Green
Write-Host 'Start:  .\Start-RithalLayerTruthV1_0_4_7.ps1' -ForegroundColor Green
Write-Host 'Verify: .\Verify-RithalLayerTruthV1_0_4_7.ps1' -ForegroundColor Green
Write-Host 'API:    http://127.0.0.1:8888/api/rithal-layer-truth' -ForegroundColor Green
