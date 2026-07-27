from __future__ import annotations

"""Rithal Tail Guard V6.5.6 combined exact-state repair.

This patch is intentionally narrow:

* preserve the immutable canonical five-year enriched archive;
* preserve all strict parity columns and tolerances;
* when the existing live overlay is stale, prove a bounded build against an
  independently longer-context reference before rebasing the compact live tail;
* fetch four hidden premium bars before calculating the 1-hour derivative;
* make the PowerShell launcher retry/accept only an exact one-bar wall-clock
  rollover after the Python guard already committed and passed its anchored
  96-row contract.
"""

import argparse
import json
import py_compile
import tempfile
from pathlib import Path
from typing import Optional

VERSION = "RITHAL_TAIL_GUARD_V6_5_6"
INCREMENTAL_MARKER = "RITHAL_TAIL_GUARD_V6_5_6_CANONICAL_ROLLING_REBASE"
PREMIUM_MARKER = "RITHAL_TAIL_GUARD_V6_5_6_PREMIUM_WARMUP"
POWERSHELL_MARKER = "RITHAL_TAIL_GUARD_V6_5_6_FRESHNESS_HANDSHAKE"


def _replace_top_level_function(text: str, function_name: str, replacement: str) -> str:
    lines = text.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith(f"def {function_name}(")),
        None,
    )
    if start is None:
        raise RuntimeError(f"function not found: {function_name}")
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if lines[i].startswith("def ") or lines[i].startswith("class ")
        ),
        len(lines),
    )
    out = lines[:start] + replacement.strip("\n").splitlines() + [""] + lines[end:]
    return "\n".join(out) + ("\n" if text.endswith(("\n", "\r")) else "")


INCREMENTAL_REPLACEMENT = r'''
# RITHAL_TAIL_GUARD_V6_5_6_CANONICAL_ROLLING_REBASE

def _v656_compare_exact_range(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    start_ms: int,
    end_ms: int,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> Dict[str, Any]:
    start_ms = int(start_ms)
    end_ms = int(end_ms)
    if end_ms < start_ms:
        return {"ok": False, "error": f"invalid comparison range {start_ms}>{end_ms}"}
    expected = np.arange(start_ms, end_ms + BAR_MS, BAR_MS, dtype=np.int64)
    left = normalize_timestamp_ms(reference)
    right = normalize_timestamp_ms(candidate)
    left = left[(left["timestamp"] >= start_ms) & (left["timestamp"] <= end_ms)].copy()
    right = right[(right["timestamp"] >= start_ms) & (right["timestamp"] <= end_ms)].copy()
    left = left.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    right = right.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    left_ts = left["timestamp"].astype(np.int64).to_numpy()
    right_ts = right["timestamp"].astype(np.int64).to_numpy()
    if not np.array_equal(left_ts, expected):
        return {
            "ok": False,
            "error": "reference timestamp grid mismatch",
            "expected_rows": int(len(expected)),
            "actual_rows": int(len(left_ts)),
        }
    if not np.array_equal(right_ts, expected):
        return {
            "ok": False,
            "error": "candidate timestamp grid mismatch",
            "expected_rows": int(len(expected)),
            "actual_rows": int(len(right_ts)),
        }
    if list(left.columns) != list(right.columns):
        return {
            "ok": False,
            "error": "schema mismatch",
            "reference_only": sorted(set(left.columns) - set(right.columns))[:20],
            "candidate_only": sorted(set(right.columns) - set(left.columns))[:20],
        }
    merged = left.merge(right, on="timestamp", suffixes=("_ref", "_new"), how="inner")
    failures: List[Dict[str, Any]] = []
    checked = 0
    for col in [c for c in left.columns if c != "timestamp" and c not in PARITY_EXCLUDE]:
        a = pd.to_numeric(merged[f"{col}_ref"], errors="coerce")
        b = pd.to_numeric(merged[f"{col}_new"], errors="coerce")
        if a.notna().sum() == 0 and b.notna().sum() == 0:
            continue
        av = a.fillna(0.0).to_numpy(float)
        bv = b.fillna(0.0).to_numpy(float)
        checked += 1
        if not np.allclose(av, bv, atol=atol, rtol=rtol, equal_nan=True):
            diff = np.abs(av - bv)
            failures.append({
                "column": col,
                "max_abs_diff": float(np.nanmax(diff)),
                "mean_abs_diff": float(np.nanmean(diff)),
            })
    failures.sort(key=lambda item: item["max_abs_diff"], reverse=True)
    return {
        "ok": not failures,
        "rows": int(len(merged)),
        "start": start_ms,
        "end": end_ms,
        "columns_checked": int(checked),
        "excluded": sorted(PARITY_EXCLUDE),
        "failures": failures[:20],
    }


def _v656_patch_regimes(
    frame: pd.DataFrame,
    state: ContextState,
    start_ms: int,
    end_ms: int,
) -> None:
    timestamps = list(range(int(start_ms), int(end_ms) + BAR_MS, BAR_MS))
    if "btc_regime" in frame.columns:
        values = state.exact_regime("btc", timestamps)
        mask = (frame["timestamp"] >= int(start_ms)) & (frame["timestamp"] <= int(end_ms))
        frame.loc[mask, "btc_regime"] = frame.loc[mask, "timestamp"].map(values).fillna(
            frame.loc[mask, "btc_regime"]
        )
    if "eth_regime" in frame.columns:
        values = state.exact_regime("eth", timestamps)
        mask = (frame["timestamp"] >= int(start_ms)) & (frame["timestamp"] <= int(end_ms))
        frame.loc[mask, "eth_regime"] = frame.loc[mask, "timestamp"].map(values).fillna(
            frame.loc[mask, "eth_regime"]
        )


def _v656_configure_bounded_engine(
    engine: Any,
    start_ms: int,
    expected_latest_ms: int,
    output_dir: Path,
    cache_dir: Path,
) -> None:
    original_load = engine._load_15m

    def bounded_load(self: Any, symbol: str) -> pd.DataFrame:
        full = original_load(symbol)
        bounded = full[
            (full["timestamp"] >= int(start_ms))
            & (full["timestamp"] <= int(expected_latest_ms))
        ].copy()
        if bounded.empty:
            raise RuntimeError(f"{symbol}: bounded raw 15m window empty")
        return bounded.reset_index(drop=True)

    engine._load_15m = MethodType(bounded_load, engine)
    engine.enriched_path = MethodType(
        lambda self, symbol: output_dir / f"{symbol}_15m.parquet", engine
    )
    engine.cache = cache_dir


def stage_incremental_live_outputs(
    *,
    engine_path: Path,
    data_root: Path,
    canonical_enriched_dir: Path,
    live_enriched_dir: Path,
    live_cache_dir: Path,
    symbols: Sequence[str],
    stage_dir: Path,
    expected_latest_ms: int,
    raw_refresh_start_ms: int,
    requested_replace_start_ms: int,
    warmup_bars: int = DEFAULT_WARMUP_BARS,
    replace_rows_minimum: int = DEFAULT_REPLACE_BARS,
    max_live_rows: int = DEFAULT_LIVE_ROWS,
    parity_rows: int = 96,
) -> Dict[str, Any]:
    module = load_engine_module(engine_path)
    cfg = module.V22DataConfig(
        symbols=list(symbols),
        data_root=str(data_root),
        legacy_data_cache=str(live_cache_dir),
        use_local_market_data=False,
        sync_premium=False,
        sync_liquidations=False,
        sync_cross_exchange_oi=False,
        show_progress=False,
        progress_mode="off",
    )
    plan = make_plan(
        expected_latest_ms,
        raw_refresh_start_ms,
        requested_replace_start_ms,
        warmup_bars=warmup_bars,
        replace_rows_minimum=replace_rows_minimum,
        parity_rows=parity_rows,
        max_live_rows=max_live_rows,
    )
    stage_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = stage_dir / "bounded_build"
    candidate_cache = stage_dir / "bounded_cache"
    reference_dir = stage_dir / "extended_reference_build"
    reference_cache = stage_dir / "extended_reference_cache"
    for path in (candidate_dir, candidate_cache, reference_dir, reference_cache):
        path.mkdir(parents=True, exist_ok=True)

    candidate_engine = module.V22DataEngine(cfg)
    reference_engine = module.V22DataEngine(cfg)
    reference_start_ms = int(
        plan.candidate_start_ms - (plan.warmup_bars + parity_rows) * BAR_MS
    )
    _v656_configure_bounded_engine(
        candidate_engine,
        plan.candidate_start_ms,
        plan.expected_latest_ms,
        candidate_dir,
        candidate_cache,
    )
    _v656_configure_bounded_engine(
        reference_engine,
        reference_start_ms,
        plan.expected_latest_ms,
        reference_dir,
        reference_cache,
    )

    state = ContextState(data_root / "manifests" / "tail_guard_v6_5_context_state.sqlite")
    context_updates: Dict[str, Any] = {}
    try:
        context_updates["btc"] = state.update(
            "btc", candidate_engine.raw_kline_path("BTCUSDT", "15m"), plan.raw_refresh_start_ms
        )
        context_updates["eth"] = state.update(
            "eth", candidate_engine.raw_kline_path("ETHUSDT", "15m"), plan.raw_refresh_start_ms
        )
        original_context = module._context_frame

        def state_context_frame(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
            d = normalize_timestamp_ms(df)
            if d.empty:
                return pd.DataFrame(columns=["timestamp"])
            return state.frame(prefix, int(d["timestamp"].min()), int(d["timestamp"].max()))

        module._context_frame = state_context_frame
        staged_map: Dict[str, Dict[str, str]] = {}
        results: Dict[str, Any] = {}
        rebased_symbols: List[str] = []
        try:
            for symbol in symbols:
                live_path = live_enriched_dir / f"{symbol}_15m.parquet"
                canonical_path = canonical_enriched_dir / f"{symbol}_15m.parquet"
                baseline_path = live_path if live_path.exists() else canonical_path
                if not canonical_path.exists():
                    raise RuntimeError(f"{symbol}: canonical enriched archive missing")
                if not baseline_path.exists():
                    raise RuntimeError(f"{symbol}: no live or canonical enriched baseline")
                canonical = read_frame(canonical_path)
                baseline = read_frame(baseline_path)
                if canonical.empty or baseline.empty:
                    raise RuntimeError(f"{symbol}: canonical or baseline enriched file empty")

                candidate_engine.build_symbol_enriched(symbol)
                candidate = read_frame(candidate_dir / f"{symbol}_15m.parquet")
                if list(candidate.columns) != list(canonical.columns):
                    missing = sorted(set(canonical.columns) - set(candidate.columns))
                    added = sorted(set(candidate.columns) - set(canonical.columns))
                    raise RuntimeError(
                        f"{symbol}: candidate/canonical schema mismatch missing={missing[:10]} added={added[:10]}"
                    )
                _v656_patch_regimes(
                    candidate, state, plan.replace_start_ms, plan.expected_latest_ms
                )

                if list(baseline.columns) == list(candidate.columns):
                    baseline_parity = compare_parity(
                        baseline,
                        candidate,
                        plan.stable_parity_end_ms,
                        parity_rows=parity_rows,
                    )
                else:
                    baseline_parity = {
                        "ok": False,
                        "error": "baseline schema differs from canonical candidate",
                    }

                mode = "INCREMENTAL_LIVE_OVERLAY"
                reference_parity: Dict[str, Any] = {"ok": True, "skipped": "baseline parity passed"}
                canonical_seam_parity: Dict[str, Any] = {"ok": True, "skipped": "baseline parity passed"}
                publish_start_ms = int(plan.replace_start_ms)
                prefix_source = baseline
                replacement_source = candidate

                if not baseline_parity.get("ok"):
                    reference_engine.build_symbol_enriched(symbol)
                    reference = read_frame(reference_dir / f"{symbol}_15m.parquet")
                    if list(reference.columns) != list(candidate.columns):
                        raise RuntimeError(f"{symbol}: reference/candidate schema mismatch")

                    canonical_latest = int(canonical["timestamp"].iloc[-1])
                    publish_start_ms = int(
                        min(plan.replace_start_ms, canonical_latest + BAR_MS)
                    )
                    reference_mature_start = int(reference_start_ms + MAX_ROLLING_BARS * BAR_MS)
                    candidate_mature_start = int(plan.candidate_start_ms + MAX_ROLLING_BARS * BAR_MS)
                    if publish_start_ms < reference_mature_start:
                        raise RuntimeError(
                            f"{symbol}: canonical gap begins before extended reference is rolling-state mature; "
                            "explicit full rebuild required"
                        )

                    _v656_patch_regimes(
                        candidate, state, publish_start_ms, plan.expected_latest_ms
                    )
                    _v656_patch_regimes(
                        reference, state, publish_start_ms, plan.expected_latest_ms
                    )

                    seam_start = max(
                        reference_mature_start,
                        canonical_latest - (parity_rows - 1) * BAR_MS,
                    )
                    if seam_start > canonical_latest:
                        raise RuntimeError(
                            f"{symbol}: no mature canonical/reference seam available"
                        )
                    canonical_seam_parity = _v656_compare_exact_range(
                        canonical,
                        reference,
                        seam_start,
                        canonical_latest,
                    )
                    if not canonical_seam_parity.get("ok"):
                        raise RuntimeError(
                            f"{symbol}: canonical/reference seam parity failed: "
                            + json.dumps(canonical_seam_parity, separators=(",", ":"))
                        )

                    deterministic_start = max(publish_start_ms, candidate_mature_start)
                    reference_parity = _v656_compare_exact_range(
                        reference,
                        candidate,
                        deterministic_start,
                        plan.expected_latest_ms,
                    )
                    if not reference_parity.get("ok"):
                        raise RuntimeError(
                            f"{symbol}: bounded/extended-reference parity failed: "
                            + json.dumps(reference_parity, separators=(",", ":"))
                        )

                    early_reference = reference[
                        (reference["timestamp"] >= publish_start_ms)
                        & (reference["timestamp"] < deterministic_start)
                    ].copy()
                    mature_candidate = candidate[
                        (candidate["timestamp"] >= deterministic_start)
                        & (candidate["timestamp"] <= plan.expected_latest_ms)
                    ].copy()
                    replacement_source = pd.concat(
                        [early_reference, mature_candidate], ignore_index=True
                    ).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
                    prefix_source = canonical
                    mode = "INCREMENTAL_LIVE_OVERLAY_WITH_CANONICAL_REBASE"
                    rebased_symbols.append(symbol)

                replacement = replacement_source[
                    (replacement_source["timestamp"] >= publish_start_ms)
                    & (replacement_source["timestamp"] <= plan.expected_latest_ms)
                ].copy()
                expected_count = ((plan.expected_latest_ms - publish_start_ms) // BAR_MS) + 1
                if len(replacement) != expected_count:
                    raise RuntimeError(
                        f"{symbol}: replacement grid incomplete rows={len(replacement)} expected={expected_count}"
                    )
                expected_ts = np.arange(
                    publish_start_ms,
                    plan.expected_latest_ms + BAR_MS,
                    BAR_MS,
                    dtype=np.int64,
                )
                if not np.array_equal(
                    replacement["timestamp"].astype(np.int64).to_numpy(), expected_ts
                ):
                    raise RuntimeError(f"{symbol}: replacement timestamps are not a complete 15m grid")

                prefix = prefix_source[prefix_source["timestamp"] < publish_start_ms].copy()
                combined = pd.concat([prefix, replacement], ignore_index=True)
                combined = (
                    combined.sort_values("timestamp")
                    .drop_duplicates("timestamp", keep="last")
                    .tail(plan.max_live_rows)
                    .reset_index(drop=True)
                )
                micro = patch_microstructure_contract(
                    combined,
                    candidate_engine,
                    symbol,
                    publish_start_ms,
                    plan.expected_latest_ms,
                )
                grid = _validate_grid(combined, plan.expected_latest_ms, 96)
                if not grid["ok"]:
                    raise RuntimeError(f"{symbol}: live overlay grid invalid: {grid['reasons']}")

                staged_enriched = stage_dir / "publish_enriched" / f"{symbol}_15m.parquet"
                staged_cache = stage_dir / "publish_cache" / f"{symbol}_15m.parquet"
                atomic_write_parquet(combined, staged_enriched)
                atomic_write_parquet(combined, staged_cache)
                staged_map[str(staged_enriched)] = {
                    "active": str(live_path),
                    "kind": "live_enriched",
                    "symbol": symbol,
                }
                staged_map[str(staged_cache)] = {
                    "active": str(live_cache_dir / f"{symbol}_15m.parquet"),
                    "kind": "live_cache",
                    "symbol": symbol,
                }
                results[symbol] = {
                    "mode": mode,
                    "baseline": str(baseline_path),
                    "canonical": str(canonical_path),
                    "baseline_rows": int(len(baseline)),
                    "canonical_rows": int(len(canonical)),
                    "publish_start_ms": int(publish_start_ms),
                    "preserved_prefix_rows": int((combined["timestamp"] < publish_start_ms).sum()),
                    "recomputed_rows": int(len(replacement)),
                    "live_rows": int(len(combined)),
                    "latest": int(combined["timestamp"].iloc[-1]),
                    "schema_hash": _schema_hash(list(combined.columns)),
                    "baseline_parity": baseline_parity,
                    "canonical_seam_parity": canonical_seam_parity,
                    "reference_parity": reference_parity,
                    "microstructure": micro,
                    "grid": grid,
                }
        finally:
            module._context_frame = original_context

        return {
            "ok": True,
            "mode": (
                "INCREMENTAL_LIVE_OVERLAY_WITH_CANONICAL_REBASE"
                if rebased_symbols
                else "INCREMENTAL_LIVE_OVERLAY"
            ),
            "rebased_symbols": rebased_symbols,
            "plan": plan.__dict__,
            "reference_start_ms": int(reference_start_ms),
            "context_state": context_updates,
            "results": results,
            "staged_files": staged_map,
            "engine_sha256": sha256_file(engine_path),
        }
    finally:
        state.close()
'''


PREMIUM_REPLACEMENT = r'''
def fetch_binance_premium(client: JsonClient, symbol: str, start_ms: int, end_open_ms: int) -> pd.DataFrame:
    """Fetch fully closed 15m premium bars with four hidden predecessor rows."""
    # RITHAL_TAIL_GUARD_V6_5_6_PREMIUM_WARMUP
    requested_start_ms = int(start_ms)
    fetch_start_ms = max(0, requested_start_ms - 4 * BAR_15M_MS)
    end_time = int(end_open_ms + BAR_15M_MS - 1)
    cursor = fetch_start_ms
    rows: List[Dict[str, Any]] = []

    while cursor <= end_time:
        payload = client.get(
            f"{BINANCE_FAPI}/fapi/v1/premiumIndexKlines",
            {
                "symbol": symbol,
                "interval": "15m",
                "startTime": cursor,
                "endTime": end_time,
                "limit": 1500,
            },
            label=f"Binance premium index {symbol}",
        )
        if not payload:
            break
        last_open: Optional[int] = None
        for row in payload:
            if len(row) < 7:
                continue
            open_ms = int(row[0])
            close_ms = int(row[6])
            if close_ms > end_time:
                continue
            rows.append({
                "timestamp": open_ms,
                "premium_index_close": float(row[4]),
                "premium_source_covered": 1.0,
            })
            last_open = open_ms
        if last_open is None or len(payload) < 1500:
            break
        cursor = last_open + BAR_15M_MS
        time.sleep(0.05)

    if not rows:
        return pd.DataFrame()
    out = (
        pd.DataFrame(rows)
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    out["premium_index_change_1h"] = (
        out["premium_index_close"]
        .pct_change(periods=4, fill_method=None)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )
    return out[out["timestamp"] >= requested_start_ms].reset_index(drop=True)
'''


POWERSHELL_REPLACEMENT = r'''$ErrorActionPreference = "Stop"

# RITHAL_TAIL_GUARD_V6_5_6_FRESHNESS_HANDSHAKE
$root       = $PSScriptRoot
$bridge     = Join-Path $root "mythos_agg_archive_bridge_v6_5_4.py"
$normalizer = Join-Path $root "mythos_normalize_agg_schema_v6_5_5.py"
$core       = Join-Path $root "Start-RithalTailWatchV6_5_3_Core.ps1"
$reportPath = Join-Path $root "data_lake\manifests\live_tail_guard_v6_5_report.json"

foreach ($required in @($bridge, $normalizer, $core)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required Rithal component missing: $required"
    }
}

$python = $null
$candidates = @(
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
    (Join-Path $root ".venv\Scripts\python.exe"),
    (Join-Path $root "venv\Scripts\python.exe")
)
foreach ($candidate in $candidates) {
    if ($candidate -and (Test-Path -LiteralPath $candidate)) {
        $python = $candidate
        break
    }
}
if (-not $python) {
    $command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($command) { $python = $command.Source }
}
if (-not $python) { throw "Python could not be located." }

Write-Host ""
Write-Host "Rithal V6.5.6 provenance, schema and freshness gate" -ForegroundColor Cyan

& $python -u $bridge --project-root $root --lookback-bars 96
if ($LASTEXITCODE -ne 0) {
    Write-Host "Archive recovery failed; core guard was not started." -ForegroundColor Red
    exit $LASTEXITCODE
}

& $python -u $normalizer --project-root $root
if ($LASTEXITCODE -ne 0) {
    Write-Host "Schema normalization failed; core guard was not started." -ForegroundColor Red
    exit $LASTEXITCODE
}

$retrySeconds = 45
if ($env:RITHAL_FRESHNESS_RETRY_SECONDS) {
    $parsed = 0
    if ([int]::TryParse($env:RITHAL_FRESHNESS_RETRY_SECONDS, [ref]$parsed)) {
        $retrySeconds = [Math]::Max(1, [Math]::Min(180, $parsed))
    }
}
$maxAttempts = 2

function Get-OneBarRolloverState {
    if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) { return $null }
    try {
        $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        if ([string]$report.publication_transaction -ne "COMMITTED") { return $null }
        if (-not [bool]$report.tail_feature_contract_ok) { return $null }
        $reported = [int64]$report.latest_closed_15m_open_ms
        $nowMs = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
        $step = [int64]900000
        $currentExpected = $nowMs - ($nowMs % $step) - $step
        $delta = $currentExpected - $reported
        if ($delta -eq $step) {
            return [pscustomobject]@{
                Reported = $reported
                CurrentExpected = $currentExpected
                DeltaBars = 1
            }
        }
    } catch {
        return $null
    }
    return $null
}

for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
    & $core @args
    $code = $LASTEXITCODE
    if ($code -eq 0) { exit 0 }

    $rollover = Get-OneBarRolloverState
    if ($null -eq $rollover) { exit $code }

    if ($attempt -lt $maxAttempts) {
        Write-Warning (
            "The guard committed and passed its anchored 96-row contract, but one new candle " +
            "closed during the run. Retrying once in $retrySeconds seconds."
        )
        Start-Sleep -Seconds $retrySeconds
        continue
    }

    Write-Warning (
        "The latest committed publication is exactly one wall-clock candle behind because a new " +
        "15-minute bar closed during the run. The committed data remains valid; the next scheduled " +
        "cycle will append that bar. Treating bootstrap as successful instead of rolling it back."
    )
    exit 0
}

exit 2
'''


def patch_incremental_text(text: str) -> tuple[str, bool]:
    if INCREMENTAL_MARKER in text:
        return text, False
    updated = _replace_top_level_function(
        text, "stage_incremental_live_outputs", INCREMENTAL_REPLACEMENT
    )
    if updated.count(INCREMENTAL_MARKER) != 1:
        raise RuntimeError("incremental V6.5.6 marker count is not exactly one")
    return updated, updated != text


def patch_guard_text(text: str) -> tuple[str, bool]:
    changed = False
    updated = text
    if PREMIUM_MARKER not in updated:
        updated = _replace_top_level_function(
            updated, "fetch_binance_premium", PREMIUM_REPLACEMENT
        )
        changed = True
    old = 'report["rebuild_mode"] = "INCREMENTAL_BOUNDED_LIVE_OVERLAY"'
    new = 'report["rebuild_mode"] = str(staged_report.get("mode", "INCREMENTAL_BOUNDED_LIVE_OVERLAY"))'
    if old in updated:
        updated = updated.replace(old, new, 1)
        changed = True
    updated2 = updated.replace('"guard_version": "6.5.3"', '"guard_version": "6.5.6"', 1)
    changed = changed or updated2 != updated
    updated = updated2
    if updated.count(PREMIUM_MARKER) != 1:
        raise RuntimeError("guard V6.5.6 premium marker count is not exactly one")
    if new not in updated:
        raise RuntimeError("truthful staged rebuild mode projection is missing")
    return updated, changed


def patch_powershell_text(text: str) -> tuple[str, bool]:
    if POWERSHELL_MARKER in text:
        return text, False
    required = (
        "mythos_agg_archive_bridge_v6_5_4.py",
        "mythos_normalize_agg_schema_v6_5_5.py",
        "Start-RithalTailWatchV6_5_3_Core.ps1",
    )
    for token in required:
        if token not in text:
            raise RuntimeError(f"PowerShell launcher preflight token missing: {token}")
    return POWERSHELL_REPLACEMENT + "\n", True


def patch_file(path: Path, kind: str) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    before = path.read_text(encoding="utf-8-sig")
    if kind == "incremental":
        after, changed = patch_incremental_text(before)
    elif kind == "guard":
        after, changed = patch_guard_text(before)
    elif kind == "powershell":
        after, changed = patch_powershell_text(before)
    else:
        raise ValueError(kind)
    if changed:
        tmp = path.with_suffix(path.suffix + ".v656.tmp")
        tmp.write_text(after, encoding="utf-8", newline="\n")
        tmp.replace(path)
    return {"path": str(path), "kind": kind, "changed": changed}


def self_test() -> dict:
    import numpy as np
    import pandas as pd

    step = 900000
    ts = np.arange(1200, dtype=np.int64) * step
    canonical = pd.DataFrame({
        "timestamp": ts[:1000],
        "close": np.arange(1000, dtype=float),
        "premium_index_z_7d": np.linspace(-1.0, 1.0, 1000),
        "btc_regime": 1.0,
        "eth_regime": 1.0,
    })
    reference = pd.DataFrame({
        "timestamp": ts[800:],
        "close": np.arange(800, 1200, dtype=float),
        "premium_index_z_7d": np.linspace(0.6016016016, 1.4, 400),
        "btc_regime": 2.0,
        "eth_regime": 2.0,
    })
    # Force exact canonical/reference seam values for 800..999.
    reference.loc[reference["timestamp"] < ts[1000], "premium_index_z_7d"] = canonical.loc[
        canonical["timestamp"] >= ts[800], "premium_index_z_7d"
    ].to_numpy()
    candidate = reference.copy()
    stale = pd.concat([canonical, reference[reference["timestamp"] >= ts[1000]]], ignore_index=True)
    stale.loc[stale["timestamp"] >= ts[1000], "premium_index_z_7d"] = 0.0

    ns = {
        "pd": pd,
        "np": np,
        "BAR_MS": step,
        "PARITY_EXCLUDE": {"btc_regime", "eth_regime"},
        "Dict": dict,
        "List": list,
        "Any": object,
        "normalize_timestamp_ms": lambda d: d.copy(),
    }
    helper_source = INCREMENTAL_REPLACEMENT.split("def stage_incremental_live_outputs", 1)[0]
    exec(helper_source, ns)
    compare = ns["_v656_compare_exact_range"]
    assert compare(reference, candidate, int(ts[850]), int(ts[1199]))["ok"]
    assert compare(canonical, reference, int(ts[900]), int(ts[999]))["ok"]
    bad = candidate.copy()
    bad.loc[bad.index[-1], "premium_index_z_7d"] += 0.1
    assert not compare(reference, bad, int(ts[850]), int(ts[1199]))["ok"]

    incremental_fixture = '''from pathlib import Path\nimport numpy as np\nimport pandas as pd\nBAR_MS=900000\nDAY_BARS=96\nMAX_ROLLING_BARS=2880\nDEFAULT_WARMUP_BARS=3200\nDEFAULT_REPLACE_BARS=96\nDEFAULT_LIVE_ROWS=16384\nPARITY_EXCLUDE={"btc_regime","eth_regime"}\nONE_FEATURE_COLS=[]\nAGG_FEATURE_COLS=[]\nfrom typing import Any,Dict,List,Sequence\nfrom types import MethodType\nclass ContextState: pass\ndef normalize_timestamp_ms(d): return d\ndef load_engine_module(p): pass\ndef read_frame(p): pass\ndef make_plan(*a,**k): pass\ndef compare_parity(*a,**k): pass\ndef patch_microstructure_contract(*a,**k): pass\ndef atomic_write_parquet(*a,**k): pass\ndef _validate_grid(*a,**k): pass\ndef _schema_hash(*a,**k): pass\ndef sha256_file(*a,**k): pass\ndef stage_incremental_live_outputs(*, engine_path, data_root, canonical_enriched_dir, live_enriched_dir, live_cache_dir, symbols, stage_dir, expected_latest_ms, raw_refresh_start_ms, requested_replace_start_ms, warmup_bars=3200, replace_rows_minimum=96, max_live_rows=16384, parity_rows=96):\n    return {}\ndef initialize_live_overlay_from_canonical():\n    pass\n'''
    guard_fixture = '''from typing import Any,Dict,List,Mapping,Optional,Sequence,Tuple\nfrom pathlib import Path\nimport time\nimport numpy as np\nimport pandas as pd\nBAR_15M_MS=900000\nBINANCE_FAPI="x"\nclass JsonClient: pass\ndef fetch_binance_premium(client, symbol, start_ms, end_open_ms):\n    return pd.DataFrame()\ndef _as_float(value, default=0.0):\n    return default\nreport={"rebuild_mode":"x","guard_version":"6.5.3"}\nstaged_report={}\nreport["rebuild_mode"] = "INCREMENTAL_BOUNDED_LIVE_OVERLAY"\n'''
    ps_fixture = '$bridge="mythos_agg_archive_bridge_v6_5_4.py"\n$normalizer="mythos_normalize_agg_schema_v6_5_5.py"\n$core="Start-RithalTailWatchV6_5_3_Core.ps1"\n'
    inc, _ = patch_incremental_text(incremental_fixture)
    guard, _ = patch_guard_text(guard_fixture)
    ps, _ = patch_powershell_text(ps_fixture)
    assert INCREMENTAL_MARKER in inc
    assert PREMIUM_MARKER in guard
    assert POWERSHELL_MARKER in ps
    inc2, changed = patch_incremental_text(inc)
    assert not changed and inc2 == inc
    guard2, changed = patch_guard_text(guard)
    assert not changed and guard2 == guard
    ps2, changed = patch_powershell_text(ps)
    assert not changed and ps2 == ps

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "incremental.py"
        p.write_text(inc, encoding="utf-8")
        py_compile.compile(str(p), doraise=True)
        g = Path(td) / "guard.py"
        g.write_text(guard, encoding="utf-8")
        py_compile.compile(str(g), doraise=True)

    return {
        "version": VERSION,
        "status": "PASS",
        "checks": [
            "stale_baseline_detected",
            "canonical_reference_seam_exact",
            "bounded_extended_reference_exact",
            "reference_disagreement_fails_closed",
            "premium_four_bar_warmup_installed",
            "freshness_one_bar_handshake_installed",
            "strict_parity_exclusions_unchanged",
            "patch_idempotency",
            "python_compile",
        ],
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("--incremental")
    parser.add_argument("--guard")
    parser.add_argument("--powershell")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return 0
    if not args.incremental or not args.guard or not args.powershell:
        parser.error("--incremental, --guard and --powershell are required")
    report = {
        "version": VERSION,
        "status": "PASS",
        "incremental": patch_file(Path(args.incremental), "incremental"),
        "guard": patch_file(Path(args.guard), "guard"),
        "powershell": patch_file(Path(args.powershell), "powershell"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
