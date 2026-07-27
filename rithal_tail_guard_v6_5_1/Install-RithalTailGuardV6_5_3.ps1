[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [switch]$StartBackground
)

$ErrorActionPreference='Stop'
$Root=[IO.Path]::GetFullPath($ProjectRoot)
$Target=Join-Path $Root 'Install-RithalTailGuardV6_5_3_R2.ps1'
$Url='https://raw.githubusercontent.com/carvedkarma/NeuralTrade/rithal-tail-guard-v6.5.1/rithal_tail_guard_v6_5_1/Install-RithalTailGuardV6_5_3_R2.ps1'
Write-Warning '[RITHAL_TAIL_GUARD_V6_5_3] Superseded by V6.5.3 R2. Loading the path-safe installer.'
Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Target
$args=@('-NoProfile','-ExecutionPolicy','Bypass','-File',$Target,'-ProjectRoot',$Root)
if($StartBackground){$args+='-StartBackground'}
& powershell @args
exit $LASTEXITCODE
