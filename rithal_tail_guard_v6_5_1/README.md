# Rithal Tail Guard V6.5.3 R2

V6.5.3 R2 is the only supported installer. The original V6.5.3 installer failed during preflight because it imported the complete active tail-guard module merely to discover six raw premium paths. The active guard has project/runtime imports and side effects, so importing it from a temporary installer location can raise a traceback before any repair begins.

R2 does not import or execute the active guard for path discovery. It resolves the frozen V22 premium location directly:

```text
data_lake\raw\binance_um\premium_15m\<SYMBOL>.parquet
```

The actual premium-state repair remains unchanged:

- recalculate only `premium_index_change_1h` across each complete existing raw premium parquet;
- preserve timestamps, premium closes, coverage flags and all unrelated columns;
- fetch four hidden predecessor bars for future incremental updates;
- keep the strict 132-column bounded/full parity gate enabled;
- require `publication=COMMITTED` and the 96-row contract before reporting success;
- restore the engine, guard and all six premium files on any failed bootstrap.

The trading engine, dashboards, model checkpoints, thresholds, ledger/equity, fees, positions, TP/SL geometry and execution authority are untouched.

## Install

```powershell
$u='https://raw.githubusercontent.com/carvedkarma/NeuralTrade/rithal-tail-guard-v6.5.1/rithal_tail_guard_v6_5_1/Install-RithalTailGuardV6_5_3_R2.ps1'
Invoke-WebRequest -UseBasicParsing $u -OutFile .\Install-RithalTailGuardV6_5_3_R2.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\Install-RithalTailGuardV6_5_3_R2.ps1 -ProjectRoot .
```

Expected successful ending:

```text
[RITHAL_TAIL_GUARD_V6_5_3_R2] INSTALLATION AND BOOTSTRAP PASS
Publication: COMMITTED
96-row contract: PASS
```

## Verify

```powershell
.\Verify-RithalTailGuardV6_5_3_R2.ps1 -ProjectRoot .
```

## Start continuous updates

```powershell
.\Start-RithalTailWatch.ps1 -ProjectRoot . -Background -NoRunNow
```

## Rollback

```powershell
.\Rollback-RithalTailGuardV6_5_3_R2.ps1
```
