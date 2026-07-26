param(
    [Parameter(Mandatory=$false)][string]$ProjectRoot=(Get-Location).Path,
    [Parameter(Mandatory=$false)][ValidateSet('PAPER_CONTROL','SHADOW_ONLY')][string]$ManagerMode='PAPER_CONTROL'
)

$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest

$Branch='rithal-behavior-fix-v1.0.5'
$Raw='https://raw.githubusercontent.com/carvedkarma/NeuralTrade/' + $Branch + '/rithal_behavior_fix_v1_0_5/Install-RithalBehaviorFixV1_0_5_R3.ps1'
$Root=[IO.Path]::GetFullPath($ProjectRoot)
$Inner=Join-Path $Root 'Install-RithalBehaviorFixV1_0_5_R3_1_Inner.ps1'
$Downloaded=Join-Path $Root 'Install-RithalBehaviorFixV1_0_5_R3_Source.ps1'

Write-Host '[RITHAL_BEHAVIOR_FIX_V1_0_5_R3_1] preparing canonical transactional installer' -ForegroundColor Cyan
Invoke-WebRequest -UseBasicParsing -Uri $Raw -OutFile $Downloaded
$text=[IO.File]::ReadAllText($Downloaded)

# Advance every generated module, marker, verifier, report and rollback name to
# R3.1, then replace release-comment anchors with exact callable assignments
# verified in the active source supplied by the user.
$text=$text.Replace('V1_0_5_R3','V1_0_5_R3_1')
$text=$text.Replace('v105_r3','v105_r3_1')
$text=$text.Replace('V105_R3','V105_R3_1')
$text=$text.Replace(
    "@('class NeuralV2Model','class LiveEngine','RITHAL_CONSOLIDATED_INTELLIGENCE_SAFETY')",
    "@('class NeuralV2Model','class LiveEngine','_rithal_v104_apply_live_patch(globals())')"
)
$text=$text.Replace(
    "@('class TradeManager','RITHAL_TRADE_MANAGER_V3_3_2','RITHAL_CONSOLIDATED_TRADE_MANAGER_PERSISTENCE')",
    "@('class TradeManager','TradeManager.observe_mark = _tmv33_observe_mark','TradeManager._decision = _tmv33_decision')"
)
# A rerun must remove both an older R3 hook and the current R3.1 hook.
$text=$text.Replace('R3_1_(?:LIVE|MANAGER)_HOOK','R3(?:_1)?_(?:LIVE|MANAGER)_HOOK')

$required=@(
    'RITHAL_BEHAVIOR_FIX_V1_0_5_R3_1',
    'rithal_behavior_fix_v105_r3_1.py',
    '_rithal_v104_apply_live_patch(globals())',
    'TradeManager.observe_mark = _tmv33_observe_mark',
    'TradeManager._decision = _tmv33_decision',
    '--authority-self-test',
    'Checkpoint directory missing'
)
foreach($token in $required){if(-not $text.Contains($token)){throw "R3.1 installer transformation missing token: $token"}}
[void][scriptblock]::Create($text)
[IO.File]::WriteAllText($Inner,$text,[Text.UTF8Encoding]::new($false))

& powershell -NoProfile -ExecutionPolicy Bypass -File $Inner -ProjectRoot $Root -ManagerMode $ManagerMode
if($LASTEXITCODE -ne 0){throw "RITHAL_BEHAVIOR_FIX_V1_0_5_R3_1 installation failed with exit code $LASTEXITCODE"}
Write-Host '[RITHAL_BEHAVIOR_FIX_V1_0_5_R3_1] canonical installation completed' -ForegroundColor Green
