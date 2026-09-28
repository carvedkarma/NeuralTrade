# Rithal Tail Guard V6.5.7

V6.5.7 supersedes V6.5.6 after the exact six-symbol bootstrap proved that canonical/reference seam equality is not a valid live-data invariant.

## Proven cause

The failing SOLUSDT seam contained a one-row correction in Binance metrics:

- `binance_oi_usd_15m` and `open_interest_value` differed in exactly one of 96 rows (`mean = max / 96`);
- the corresponding raw OI and taker-ratio fields also differed in exactly one row;
- rolling z-scores then propagated that corrected source row across their windows.

The canonical five-year enriched archive is an immutable historical output snapshot. A new reference build uses the currently corrected raw source state. Requiring those two artifacts to be identical rejects the correction itself.

## Correct architecture

When the existing live overlay still matches the current bounded build, V6.5.7 performs the ordinary incremental replacement.

When it does not match, V6.5.7:

1. builds the normal short bounded candidate;
2. builds an independent longer reference with enough warm-up to publish a complete 16,384-row live overlay;
3. requires exact short/long parity after the short build is rolling-state mature;
4. publishes the complete live overlay from the single full-context reference;
5. never splices corrected reference rows onto the canonical archive;
6. leaves the canonical five-year archive untouched.

The four-hidden-bar premium correction and exact one-bar wall-clock freshness handshake from V6.5.6 remain active. All strict parity columns and tolerances remain active; `premium_index_z_7d` and Binance metrics are not excluded.

## Install

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\RithalTailGuardV6_5_7\Install-RithalTailGuardV6_5_7.ps1 -ProjectRoot . -RunBootstrap
```

Required result:

```text
[RITHAL_TAIL_GUARD_V6_5_7] INSTALLATION PASS
[RITHAL_TAIL_GUARD_V6_5_7] BOOTSTRAP COMMITTED PASS
Publication: COMMITTED | 96-row contract: PASS
```

## Verify

```powershell
.\Verify-RithalTailGuardV6_5_7.ps1 -ProjectRoot .
```

## Resume

```powershell
.\Start-RithalTailWatch.ps1 -ProjectRoot . -Background -NoRunNow
```

## Rollback

```powershell
.\Rollback-RithalTailGuardV6_5_7.ps1
```
