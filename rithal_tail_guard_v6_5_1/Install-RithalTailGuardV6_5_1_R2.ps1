[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [switch]$RunBootstrap,
    [switch]$StartBackground
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Branch = 'rithal-tail-guard-v6.5.1'
$Raw = "https://raw.githubusercontent.com/carvedkarma/NeuralTrade/$Branch/rithal_tail_guard_v6_5_1/Install-RithalTailGuardV6_5_1.ps1"
$Root = [IO.Path]::GetFullPath($ProjectRoot)
$Downloaded = Join-Path $Root 'Install-RithalTailGuardV6_5_1_Source.ps1'
$Inner = Join-Path $Root 'Install-RithalTailGuardV6_5_1_R2_Inner.ps1'

Write-Host '[RITHAL_TAIL_GUARD_V6_5_1_R2] preparing corrected transactional installer' -ForegroundColor Cyan
Invoke-WebRequest -UseBasicParsing -Uri $Raw -OutFile $Downloaded
$text = [IO.File]::ReadAllText($Downloaded)

# Single-quoted literals preserve the backtick that must remain inside the base
# installer's generated-verifier here-string.
$oldInstall = '($engineText.Split(''RITHAL_TAIL_GUARD_V6_5_1_PREMIUM_ALIGNMENT'').Count - 1)'
$newInstall = '([regex]::Matches($engineText,[regex]::Escape(''RITHAL_TAIL_GUARD_V6_5_1_PREMIUM_ALIGNMENT''))).Count'
$oldVerify = '(`$et.Split(''RITHAL_TAIL_GUARD_V6_5_1_PREMIUM_ALIGNMENT'').Count-1)'
$newVerify = '([regex]::Matches(`$et,[regex]::Escape(''RITHAL_TAIL_GUARD_V6_5_1_PREMIUM_ALIGNMENT''))).Count'

if (-not $text.Contains($oldInstall)) { throw 'Base installer marker-count expression was not found.' }
if (-not $text.Contains($oldVerify)) { throw 'Generated verifier marker-count expression was not found.' }
$text = $text.Replace($oldInstall, $newInstall).Replace($oldVerify, $newVerify)
$text = $text.Replace("`$Version = 'RITHAL_TAIL_GUARD_V6_5_1'", "`$Version = 'RITHAL_TAIL_GUARD_V6_5_1_R2'")

foreach ($token in @(
    '[regex]::Matches($engineText',
    '[regex]::Matches(`$et',
    'premium_source_covered',
    'pct_change(periods=4, fill_method=None)'
)) {
    if (-not $text.Contains($token)) { throw "R2 transformation missing token: $token" }
}
[void][scriptblock]::Create($text)
[IO.File]::WriteAllText($Inner, $text, [Text.UTF8Encoding]::new($false))

$argsList = @('-NoProfile','-ExecutionPolicy','Bypass','-File',$Inner,'-ProjectRoot',$Root)
if ($RunBootstrap) { $argsList += '-RunBootstrap' }
if ($StartBackground) { $argsList += '-StartBackground' }
& powershell @argsList
if ($LASTEXITCODE -ne 0) { throw "V6.5.1 R2 installer failed with exit code $LASTEXITCODE" }
Write-Host '[RITHAL_TAIL_GUARD_V6_5_1_R2] installation completed' -ForegroundColor Green
