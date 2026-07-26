param(
    [Parameter(Mandatory=$false)][string]$ProjectRoot=(Get-Location).Path,
    [Parameter(Mandatory=$false)][ValidateSet('PAPER_CONTROL','SHADOW_ONLY')][string]$ManagerMode='PAPER_CONTROL'
)

$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest

$Version='RITHAL_BEHAVIOR_FIX_V1_0_5_R2'
$Branch='rithal-behavior-fix-v1.0.5'
$Root=[IO.Path]::GetFullPath($ProjectRoot)
$Inner=Join-Path $Root 'Install-RithalBehaviorFixV1_0_5_R1.ps1'
$InnerUrl="https://raw.githubusercontent.com/carvedkarma/NeuralTrade/$Branch/rithal_behavior_fix_v1_0_5/Install-RithalBehaviorFixV1_0_5_R1.ps1"

if(-not(Test-Path -LiteralPath (Join-Path $Root 'mythos\neural\live.py') -PathType Leaf)){
    throw "Invalid Rithal project root: $Root"
}

# The current process must be stopped before settings are changed to PAPER_CONTROL.
# Otherwise the already-loaded, pre-V1.0.5 manager could hot-reload the new authority
# setting and act using the superseded age-only/P80-first-touch rules.
$running=@()
try{
    $running=@(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
        $name=[string]$_.Name
        $cmd=[string]$_.CommandLine
        $isPython=$name -match '^(python|pythonw)(\.exe)?$'
        $isRithal=$cmd -match '(?i)(live-neural-v2|mythos[\\/]neural[\\/]live\.py|rithal-1-0-contract-locked|Start-Rithal18KContractV2)'
        $isPython -and $isRithal
    })
}catch{
    throw "Unable to prove that the trading engine is stopped. Close the Rithal engine and run this installer from an elevated PowerShell. Detail: $($_.Exception.Message)"
}
if($running.Count -gt 0){
    $details=($running | ForEach-Object { "PID=$($_.ProcessId) $($_.CommandLine)" }) -join "`n"
    throw "RITHAL_ENGINE_RUNNING. Stop the trading engine before installing actual behavior changes.`n$details"
}

Write-Host "[$Version] engine-stop preflight PASS" -ForegroundColor Green
Invoke-WebRequest -UseBasicParsing -Uri $InnerUrl -OutFile $Inner
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Inner -ProjectRoot $Root -ManagerMode $ManagerMode
if($LASTEXITCODE -ne 0){throw "R1 transactional installer failed with exit code $LASTEXITCODE"}

Write-Host "[$Version] INSTALLATION PASS" -ForegroundColor Green
Write-Host 'Run .\Verify-RithalBehaviorFixV1_0_5_R1.ps1, then start the engine once.' -ForegroundColor Yellow
