# Rithal Behavior Fix V1.0.5 R2

Canonical incremental behavior repair for the locked Rithal PAPER instance.

## Protected contract

- Instance: `rithal-1-0-contract-locked`
- Checkpoints: `checkpoints\rithal_clean`
- Starting paper equity: `$18,000`
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
- Original, repaired and applied inference truth is retained in the trade record.
- Open-interest, liquidation and aggregate-trade repairs are performed coherently by family.
- Family members are neutralized only during an explicitly forced repair call.
- Invalid or missing four-class regime output is `REGIME_UNKNOWN` and blocks a new entry.
- The real breakout posterior is published; no synthetic panic class is invented.
- Rank prewarm and live decisions use the same canonical scorer.
- A full immutable 15m entry thesis is handed to the single 5m Trade Manager.
- Controller proposed, effective and applied multipliers are separated; quantity follows applied/effective authority.
- The PAPER monthly trade-count cap is bypassed; daily/monthly R-loss guards remain active.
- Early-loss authority requires persistent closed-5m evidence and thesis deterioration/recovery failure.
- The 70-minute/55% mature exit also requires stagnation, giveback and deterioration evidence.
- P80 partial harvesting requires persistent erosion; the first 80% touch alone does not execute.
- Existing parity-locked PAPER full-close and partial-close authority is reused.

## Mandatory controlled restart

Stop the currently running trading engine before installation. R2 refuses to continue when it detects the active Rithal Python process. This prevents the already-loaded old manager from hot-reloading `PAPER_CONTROL` before the repaired code is loaded.

## Install in PAPER_CONTROL

Run from `C:\Users\muham\Downloads\mythos_v24_full` after the engine is stopped:

```powershell
$u='https://raw.githubusercontent.com/carvedkarma/NeuralTrade/rithal-behavior-fix-v1.0.5/rithal_behavior_fix_v1_0_5/Install-RithalBehaviorFixV1_0_5_R2.ps1'; Invoke-WebRequest -UseBasicParsing $u -OutFile .\Install-RithalBehaviorFixV1_0_5_R2.ps1; powershell -NoProfile -ExecutionPolicy Bypass -File .\Install-RithalBehaviorFixV1_0_5_R2.ps1 -ProjectRoot . -ManagerMode PAPER_CONTROL
```

The inner transactional installer compiles the active engine and manager, runs deterministic behavior tests, checks unique hook placement, records hashes, verifies that `checkpoints\rithal_clean` did not change, and restores its timestamped backup on any failure.

## Verify

```powershell
.\Verify-RithalBehaviorFixV1_0_5_R1.ps1
```

Expected final line:

```text
[RITHAL_BEHAVIOR_FIX_V1_0_5_R1] VERIFICATION PASS
```

## Activate

Start the engine once after verification. Dashboard restart is not required by this package.

```powershell
.\Start-Rithal18KContractV2.ps1 -ProjectRoot .
```

## Roll back

```powershell
.\Rollback-RithalBehaviorFixV1_0_5_R1.ps1
```

Restart the trading engine after rollback.
