# Rithal Behavior Fix V1.0.5 R3.2

Canonical incremental PAPER behavior repair for the locked Rithal instance.

**R1, R2, R3 and R3.1 are superseded. Use R3.2 only.**

R3.2 corrects the six-symbol startup quarantine introduced by R3.1. R3.1 reimplemented rank prewarm directly from a raw `regime_logits` key. The active Rithal source already had a canonical shared `_score_model_output()` and rank-prewarm path that handled the deployed regime contract. Because the raw key was not exposed in that exact form, all six rank buffers were marked `QUARANTINED_REGIME` with `regime_head_not_valid_four_class`.

R3.2 preserves the active shared scorer and active rank-prewarm implementation. Risk-adjusted side selection is now applied as a narrow post-scoring overlay, so regime truth, double-softmax rank parity and the valid breakout posterior remain owned by the active scorer.

## Protected contract

- Instance: `rithal-1-0-contract-locked`
- Checkpoints: `checkpoints\rithal_clean`
- Starting paper equity: `$18,000`
- Current realized paper equity and ledger: preserved
- Fixed margin: `10%`
- Fixed leverage: `10x`
- Allocation cap: `60%`
- Heat cap: `3%`
- Symbols/slots: six
- Entry and exit fees: unchanged
- Static deployment score thresholds: unchanged
- 168-feature order: unchanged
- Original TP/SL geometry: unchanged
- LIVE exchange authority: disabled

## R3.2 behavior

- Uses the active shared `_score_model_output()` for live inference and rank prewarm.
- Does not require a raw `regime_logits` dictionary key in the overlay.
- Preserves true regime probabilities, rank probabilities, breakout probability and active double-softmax parity.
- Applies risk-adjusted long/short side selection after canonical scoring.
- Retains R3.1 coherent feature-family expansion and process-wide repair isolation.
- Retains immutable 15-minute thesis handoff and restored-position context handling.
- Retains persistent closed-5-minute evidence for loss, mature-profit and P80 actions.
- Reuses the exact local PAPER authority classifier.
- Keeps `SHADOW_ONLY` and `PAPER_CONTROL` mode semantics consistent.

## Installation

Stop the running Rithal engine before installing. Then run from:

```text
C:\Users\muham\Downloads\mythos_v24_full
```

```powershell
$u='https://raw.githubusercontent.com/carvedkarma/NeuralTrade/rithal-behavior-fix-v1.0.5/rithal_behavior_fix_v1_0_5/Install-RithalBehaviorFixV1_0_5_R3_2.ps1'
Invoke-WebRequest -UseBasicParsing $u -OutFile .\Install-RithalBehaviorFixV1_0_5_R3_2.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\Install-RithalBehaviorFixV1_0_5_R3_2.ps1 -ProjectRoot . -ManagerMode PAPER_CONTROL
```

The installer rebuilds the verified R3.1 dependency chain while the engine is stopped, promotes the active hooks to R3.2, compiles the complete chain, runs the exact rank-prewarm regression test, retests the local authority classifier, confirms checkpoint immutability and rolls back automatically on failure.

## Verify

```powershell
.\Verify-RithalBehaviorFixV1_0_5_R3_2.ps1 -ProjectRoot .
```

Expected final line:

```text
[RITHAL_BEHAVIOR_FIX_V1_0_5_R3_2] VERIFICATION PASS
```

## Start

```powershell
.\Start-Rithal18KContractV2.ps1 -ProjectRoot .
```

After startup, the rank buffer should no longer report `QUARANTINED_REGIME` solely because a raw `regime_logits` key is absent. A genuine invalid regime output still fails closed.

## Rollback

```powershell
.\Rollback-RithalBehaviorFixV1_0_5_R3_2.ps1
```

Restart the engine after rollback.
