# Rithal Tail Guard V6.5.1 R2

This package repairs the enrichment publication failure where the guard reported:

```text
publication=ROLLED_BACK
column=premium_index_change_1h
```

**Use the R2 installer only.** R2 includes the corrected Windows marker-verification path and passed the complete Python and Windows transaction suites.

## Root cause

`premium_index_change_1h` was calculated in the raw premium frame and then carried through an independent as-of merge. `premium_index_close` was aligned separately in the final engine-facing frame, so the derivative no longer necessarily described the published close series.

The previous engine also used `premium_index_close != 0` as its availability test. A genuine zero premium is valid market data, so this incorrectly reduced AVAX recent premium coverage to 91.7% even when the source row was present.

## V6.5.1 correction

- Recalculate `premium_index_change_1h` from the final aligned 15-minute `premium_index_close` series.
- Use `premium_source_covered` for `premium_index_available` when that source flag exists.
- Treat an exact zero premium as covered data.
- Explicitly use `pct_change(periods=4, fill_method=None)` in both raw and final derivations.
- Keep the guard's transactional publication, parity validation, preserve-on-failure behavior, 96-row validation, catch-up logic, and all other feature columns unchanged.
- Stop enrichment writers only; the Rithal trading engine and dashboards are not stopped.

## Install and repair the missing tail

From the project root:

```powershell
$u='https://raw.githubusercontent.com/carvedkarma/NeuralTrade/rithal-tail-guard-v6.5.1/rithal_tail_guard_v6_5_1/Install-RithalTailGuardV6_5_1_R2.ps1'
Invoke-WebRequest -UseBasicParsing $u -OutFile .\Install-RithalTailGuardV6_5_1_R2.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\Install-RithalTailGuardV6_5_1_R2.ps1 -ProjectRoot . -RunBootstrap
```

The installer backs up the active guard and `mythos\data_engine_v22.py`, runs deterministic tests, applies the source patch, compiles the exact local files, and runs one bootstrap/catch-up transaction when `-RunBootstrap` is supplied.

## Verify

```powershell
.\Verify-RithalTailGuardV6_5_1.ps1 -ProjectRoot .
```

Expected:

```text
[RITHAL_TAIL_GUARD_V6_5_1] VERIFICATION PASS
```

## Start continuous updates

After the bootstrap passes:

```powershell
.\Start-RithalTailWatch.ps1 -ProjectRoot . -Background -NoRunNow
```

`-NoRunNow` avoids immediately repeating the successful bootstrap; the watcher begins at the next closed 15-minute boundary.

## Rollback

```powershell
.\Rollback-RithalTailGuardV6_5_1.ps1
```
