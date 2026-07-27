# Rithal Tail Guard V6.5.6

V6.5.6 repairs the three proven failure paths in the current active tail stack:

1. A corrected bounded candidate was compared against a stale compact live baseline. V6.5.6 builds an independently longer-context reference, verifies the immutable canonical seam, and rebases the compact live overlay only when both strict comparisons pass.
2. The premium 1-hour derivative was calculated at the beginning of a bounded fetch without its four predecessor bars. V6.5.6 fetches those hidden rows and trims them only after the derivative is calculated.
3. A long successful bootstrap could cross a 15-minute boundary. The Python guard passed its anchored exact contract, while the outer PowerShell watcher expected the next candle and declared failure. V6.5.6 retries one exact one-bar rollover and preserves a valid committed publication instead of rolling it back.

No parity feature is excluded. The existing exclusions remain only `btc_regime` and `eth_regime`. Tolerances remain unchanged. The canonical five-year archive, model/checkpoints, ledger, settings, thresholds, fees, TP/SL geometry, dashboard and trading execution are untouched.

## Install and run one protected bootstrap

```powershell
cd C:\Users\muham\Downloads\mythos_v24_full
$u='https://raw.githubusercontent.com/carvedkarma/NeuralTrade/rithal-tail-guard-v6.5.6/rithal_tail_guard_v6_5_6/Install-RithalTailGuardV6_5_6.ps1'
Invoke-WebRequest -UseBasicParsing $u -OutFile .\Install-RithalTailGuardV6_5_6.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\Install-RithalTailGuardV6_5_6.ps1 -ProjectRoot . -RunBootstrap
```

Expected final lines:

```text
[RITHAL_TAIL_GUARD_V6_5_6] INSTALLATION PASS
[RITHAL_TAIL_GUARD_V6_5_6] BOOTSTRAP COMMITTED PASS
```

## Verify

```powershell
.\Verify-RithalTailGuardV6_5_6.ps1 -ProjectRoot .
```

## Start continuous enrichment

```powershell
.\Start-RithalTailWatch.ps1 -ProjectRoot . -Background -NoRunNow
```

The first successful stale-baseline migration reports:

```text
INCREMENTAL_LIVE_OVERLAY_WITH_CANONICAL_REBASE
```

Later cycles report:

```text
INCREMENTAL_LIVE_OVERLAY
```

## Rollback

```powershell
.\Rollback-RithalTailGuardV6_5_6.ps1
```
