# Rithal Tail Guard V6.5.3

V6.5.3 is the only supported installer in this package. V6.5.1, R2 and V6.5.2 were removed after exact-project bootstraps continued to fail strict bounded/full parity on `premium_index_change_1h`.

## Proven root cause

The underlying `premium_index_close` series passed parity, but older `premium_index_change_1h` values persisted in the raw premium parquet had been calculated at separate incremental chunk boundaries. Adding four predecessor bars corrected the newest fetch, reducing the maximum mismatch, but did not repair derivative values already stored in the earlier raw state.

V6.5.3 therefore performs two complementary corrections:

1. One time during installation, recalculate only `premium_index_change_1h` across each complete existing raw premium parquet from its unchanged, timestamp-sorted `premium_index_close` series.
2. For all later incremental refreshes, fetch four hidden 15-minute predecessor bars, calculate the one-hour derivative, and trim those hidden rows only after calculation.

The strict 132-column bounded/full parity gate remains enabled. No feature is excluded and no tolerance is weakened.

## Protected scope

The installer:

- backs up the active tail guard, `mythos\data_engine_v22.py`, and all six raw premium parquets;
- stops only enrichment writers belonging to the selected project root;
- does not stop or modify the trading engine or dashboards;
- does not change model checkpoints, thresholds, ledger/equity, fees, TP/SL geometry, positions, or execution authority;
- restores every modified source/raw file if bootstrap does not finish with `publication=COMMITTED` and the 96-row contract passing.

## Install and bootstrap

Run from the project root:

```powershell
$u='https://raw.githubusercontent.com/carvedkarma/NeuralTrade/rithal-tail-guard-v6.5.1/rithal_tail_guard_v6_5_1/Install-RithalTailGuardV6_5_3.ps1'
Invoke-WebRequest -UseBasicParsing $u -OutFile .\Install-RithalTailGuardV6_5_3.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\Install-RithalTailGuardV6_5_3.ps1 -ProjectRoot .
```

The installer automatically runs one protected bootstrap. Its only successful final result is:

```text
[RITHAL_TAIL_GUARD_V6_5_3] INSTALLATION AND BOOTSTRAP PASS
Publication: COMMITTED
96-row contract: PASS
```

## Verify

```powershell
.\Verify-RithalTailGuardV6_5_3.ps1 -ProjectRoot .
```

Expected:

```text
[RITHAL_TAIL_GUARD_V6_5_3] VERIFICATION PASS
```

## Start continuous updates

```powershell
.\Start-RithalTailWatch.ps1 -ProjectRoot . -Background -NoRunNow
```

`-NoRunNow` avoids immediately repeating the successful bootstrap and begins routine operation at the next closed 15-minute boundary.

## Rollback

```powershell
.\Rollback-RithalTailGuardV6_5_3.ps1
```
