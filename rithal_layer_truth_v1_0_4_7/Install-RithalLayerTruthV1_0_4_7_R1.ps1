param(
    [Parameter(Mandatory = $false)]
    [string]$ProjectRoot = (Get-Location).Path
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Version = 'RITHAL_LAYER_TRUTH_V1_0_4_7_R1'
$Branch = 'rithal-layer-truth-v1.0.4.7'
$RawBase = "https://raw.githubusercontent.com/carvedkarma/NeuralTrade/$Branch/rithal_layer_truth_v1_0_4_7"
$Root = [System.IO.Path]::GetFullPath($ProjectRoot)
$BaseInstaller = Join-Path $Root 'Install-RithalLayerTruthV1_0_4_7.ps1'
$Runtime = Join-Path $Root 'mythos\neural\rithal_layer_truth_v1047.py'

Write-Host "[$Version] Downloading base transactional installer" -ForegroundColor Cyan
Invoke-WebRequest -UseBasicParsing -Uri "$RawBase/Install-RithalLayerTruthV1_0_4_7.ps1" -OutFile $BaseInstaller
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $BaseInstaller -ProjectRoot $Root
if ($LASTEXITCODE -ne 0) { throw "Base installer failed with exit code $LASTEXITCODE" }

if (-not (Test-Path -LiteralPath $Runtime -PathType Leaf)) {
    throw "Installed runtime missing: $Runtime"
}

Write-Host "[$Version] Applying exact-ticket binding and canonical regime projection hotfixes" -ForegroundColor Cyan
$text = [System.IO.File]::ReadAllText($Runtime)

$oldBinding = @'
        cand_trade_id = extract_trade_id(mapping)
        cand_ticket_id = extract_ticket_id(mapping)
        cand_symbol = extract_symbol(mapping)
'@
$newBinding = @'
        cand_trade_id = extract_trade_id(mapping)
        if not cand_trade_id and trade_id and trade_id in path:
            cand_trade_id = trade_id
        cand_ticket_id = extract_ticket_id(mapping)
        if not cand_ticket_id and ticket_id and ticket_id in path:
            cand_ticket_id = ticket_id
        cand_symbol = extract_symbol(mapping)
'@
if ($text.Contains($oldBinding)) {
    $text = $text.Replace($oldBinding, $newBinding)
} elseif (-not $text.Contains('if not cand_trade_id and trade_id and trade_id in path:')) {
    throw 'Exact-binding patch anchor not found.'
}

$oldThesis = @'
        if trade_id and mapped_trade == trade_id:
            matches.append((extract_timestamp(mapping), mapping, path))
        elif ticket_id and mapped_ticket == ticket_id:
'@
$newThesis = @'
        if trade_id and (mapped_trade == trade_id or trade_id in path):
            matches.append((extract_timestamp(mapping), mapping, path))
        elif ticket_id and (mapped_ticket == ticket_id or ticket_id in path):
'@
if ($text.Contains($oldThesis)) {
    $text = $text.Replace($oldThesis, $newThesis)
} elseif (-not $text.Contains('mapped_trade == trade_id or trade_id in path')) {
    throw 'Exact-thesis patch anchor not found.'
}

$oldRegime = '                    "regime": regime_truth(thesis.get("snapshot") if isinstance(thesis, dict) else position),'
$newRegime = '                    "regime": (((thesis.get("snapshot") or {}).get("regime")) if isinstance(thesis, dict) and isinstance(thesis.get("snapshot"), dict) and isinstance((thesis.get("snapshot") or {}).get("regime"), dict) else regime_truth(position)),'
if ($text.Contains($oldRegime)) {
    $text = $text.Replace($oldRegime, $newRegime)
} elseif (-not $text.Contains('else regime_truth(position)),')) {
    throw 'Canonical-regime patch anchor not found.'
}

[System.IO.File]::WriteAllText($Runtime, $text, [System.Text.UTF8Encoding]::new($false))

Write-Host "[$Version] Compiling and verifying" -ForegroundColor Cyan
& python -m py_compile $Runtime
if ($LASTEXITCODE -ne 0) { throw "Runtime compilation failed with exit code $LASTEXITCODE" }
& python $Runtime --project-root $Root --verify
if ($LASTEXITCODE -ne 0) { throw "Runtime verification failed with exit code $LASTEXITCODE" }

Write-Host "[$Version] INSTALLATION PASS" -ForegroundColor Green
Write-Host 'Start:  .\Start-RithalLayerTruthV1_0_4_7.ps1' -ForegroundColor Green
Write-Host 'Verify: .\Verify-RithalLayerTruthV1_0_4_7.ps1' -ForegroundColor Green
Write-Host 'API:    http://127.0.0.1:8888/api/rithal-layer-truth' -ForegroundColor Green
