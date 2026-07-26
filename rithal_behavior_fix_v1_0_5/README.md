# Rithal Behavior Fix V1.0.5 R3

Canonical incremental PAPER behavior repair for the locked Rithal instance.

**R1 and R2 were withdrawn during cross-verification. Do not install them.** R3 corrects the active `live.py` method-signature mismatch, separates true and rank regime probabilities, serializes feature-family repair, fails closed on missing 15-minute intelligence, reconstructs restored-position entry context, and verifies the exact local manager-authority classifier before enabling PAPER control.

## Protected contract

- Instance: `rithal-1-0-contract-locked`
- Checkpoints: `checkpoints\rithal_clean`
- Starting paper equity: `$18,000`
- Current realized paper equity: preserved; never reset by this installer
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

## Actual PAPER behavior changes

- Direction uses risk-adjusted long/short edge instead of raw expected-R alone.
- True regime posterior and backtest/rank posterior are published separately.
- The real breakout posterior is preserved; invalid four-class output becomes `REGIME_UNKNOWN` and blocks a new entry.
- Live inference and rank prewarm use the same canonical scorer and rank-regime contract.
- Open-interest, liquidation and aggregate-trade repairs operate as coherent families.
- Feature repair is protected by a process-wide lock so one symbol cannot leak a temporary repair contract into another symbol's inference.
- The active five-argument engine/manager context signatures are preserved exactly.
- A full immutable 15-minute thesis is handed to the single 5-minute Trade Manager.
- Restored positions reconstruct entry context where possible; missing context remains explicitly unavailable.
- Controller proposed, effective and applied multipliers are separated; quantity follows the effective entry-authority multiplier already used by the active engine.
- The PAPER monthly trade-count cap is bypassed; daily/monthly R-loss guards remain active.
- Early-loss authority requires persistent closed-5-minute evidence and thesis deterioration or failed recovery.
- The 70-minute/55% mature-profit exit additionally requires stagnation, giveback and available deterioration evidence.
- P80 partial harvesting requires persistent erosion; the first 80% touch alone cannot execute.
- Existing parity-locked PAPER full-close and partial-close authority is reused only after an exact-project classifier self-test passes.

## Mandatory controlled restart

Stop the running Rithal engine before installation. R3 refuses to continue while it detects the active Python engine. This prevents the old in-memory manager from loading new authority settings.

## Install in PAPER_CONTROL

Run from `C:\Users\muham\Downloads\mythos_v24_full` after stopping the engine:

```powershell
$u='https://raw.githubusercontent.com/carvedkarma/NeuralTrade/rithal-behavior-fix-v1.0.5/rithal_behavior_fix_v1_0_5/Install-RithalBehaviorFixV1_0_5_R3.ps1'; Invoke-WebRequest -UseBasicParsing $u -OutFile .\Install-RithalBehaviorFixV1_0_5_R3.ps1; powershell -NoProfile -ExecutionPolicy Bypass -File .\Install-RithalBehaviorFixV1_0_5_R3.ps1 -ProjectRoot . -ManagerMode PAPER_CONTROL
```

For observation-only manager behavior, replace `PAPER_CONTROL` with `SHADOW_ONLY`. R3 then writes `SHADOW_ONLY` and `execution_enabled=false` consistently.

The installer:

1. proves the old engine is stopped;
2. requires a non-empty `checkpoints\rithal_clean` directory;
3. backs up every changed/generated file;
4. installs exactly one R3 engine hook and one R3 manager hook;
5. compiles the active engine, manager, settings and authority modules;
6. runs deterministic behavior and active-signature tests;
7. sends synthetic hold, mature-profit, P80 partial and loss instructions through the exact local authority classifier;
8. applies settings only after all code/authority checks pass;
9. confirms mode consistency and checkpoint immutability;
10. restores the backup automatically on any failure.

## Verify

```powershell
.\Verify-RithalBehaviorFixV1_0_5_R3.ps1
```

Expected final line:

```text
[RITHAL_BEHAVIOR_FIX_V1_0_5_R3] VERIFICATION PASS
```

## Activate

Start the engine once after verification. Dashboard restart is not required by this package.

```powershell
.\Start-Rithal18KContractV2.ps1 -ProjectRoot .
```

## Roll back

```powershell
.\Rollback-RithalBehaviorFixV1_0_5_R3.ps1
```

Restart the engine after rollback.
