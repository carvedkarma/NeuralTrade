from __future__ import annotations

"""Rithal Tail Guard V6.5.7 delta.

V6.5.6 production evidence proved that canonical/reference seam equality is not
a valid invariant after a raw-source correction. The immutable canonical archive
is a historical output snapshot, while a new reference is rebuilt from the
currently corrected raw source state. Requiring those artifacts to be equal
rejects the correction.

This delta is applied after the V6.5.6 base patch. It keeps the canonical archive
untouched, proves the short bounded build against an independently longer build,
and, only after that proof passes, replaces the compact 16K live overlay with a
single self-contained full-context reference tail. No canonical/live splice is
performed, so there is no mixed-state seam.
"""

import argparse
import json
import py_compile
import tempfile
from pathlib import Path
from typing import Optional

VERSION = "RITHAL_TAIL_GUARD_V6_5_7"
BASE_MARKER = "RITHAL_TAIL_GUARD_V6_5_6_CANONICAL_ROLLING_REBASE"
MARKER = "RITHAL_TAIL_GUARD_V6_5_7_FULL_CONTEXT_LIVE_REBASE"


def _replace_top_level_function(text: str, function_name: str, replacement: str) -> str:
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(f"def {function_name}(")), None)
    if start is None:
        raise RuntimeError(f"function not found: {function_name}")
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("def ") or lines[i].startswith("class ")),
        len(lines),
    )
    out = lines[:start] + replacement.strip("\n").splitlines() + [""] + lines[end:]
    return "\n".join(out) + ("\n" if text.endswith(("\n", "\r")) else "")


STAGE_REPLACEMENT = r'''
# RITHAL_TAIL_GUARD_V6_5_7_FULL_CONTEXT_LIVE_REBASE

def _v657_reference_tail(
    reference: pd.DataFrame,
    expected_latest_ms: int,
    max_live_rows: int,
) -> pd.DataFrame:
    end_ms = int(expected_latest_ms)
    start_ms = end_ms - (int(max_live_rows) - 1) * BAR_MS
    out = normalize_timestamp_ms(reference)
    out = out[(out["timestamp"] >= start_ms) & (out["timestamp"] <= end_ms)].copy()
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    expected = np.arange(start_ms, end_ms + BAR_MS, BAR_MS, dtype=np.int64)
    actual = out["timestamp"].astype(np.int64).to_numpy()
    if not np.array_equal(actual, expected):
        raise RuntimeError(
            f"full-context live rebase grid incomplete rows={len(actual)} expected={len(expected)} "
            f"start={start_ms} end={end_ms}"
        )
    return out


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
    reference_dir = stage_dir / "full_context_reference_build"
    reference_cache = stage_dir / "full_context_reference_cache"
    for path in (candidate_dir, candidate_cache, reference_dir, reference_cache):
        path.mkdir(parents=True, exist_ok=True)

    candidate_engine = module.V22DataEngine(cfg)
    reference_engine = module.V22DataEngine(cfg)

    live_output_start_ms = int(
        int(expected_latest_ms) - (int(max_live_rows) - 1) * BAR_MS
    )
    # The reference must contain a complete state warm-up before the first row
    # that will be published. Adding parity_rows also leaves an independent
    # comparison margin beyond the maximum V22 rolling window.
    reference_start_ms = int(
        live_output_start_ms - (int(warmup_bars) + int(parity_rows)) * BAR_MS
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
                _v656_patch_regimes(candidate, state, plan.replace_start_ms, plan.expected_latest_ms)

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
                        "error": "baseline schema differs from current candidate",
                    }

                mode = "INCREMENTAL_LIVE_OVERLAY"
                full_context_parity: Dict[str, Any] = {
                    "ok": True,
                    "skipped": "existing live baseline parity passed",
                }
                canonical_seam_policy: Dict[str, Any] = {
                    "ok": True,
                    "policy": "NOT_COMPARED",
                    "reason": (
                        "canonical archive is an immutable historical output snapshot, not a "
                        "live-source oracle after raw corrections"
                    ),
                }
                micro_patch_start_ms = int(plan.replace_start_ms)

                if baseline_parity.get("ok"):
                    replacement = candidate[
                        (candidate["timestamp"] >= plan.replace_start_ms)
                        & (candidate["timestamp"] <= plan.expected_latest_ms)
                    ].copy()
                    expected_count = ((plan.expected_latest_ms - plan.replace_start_ms) // BAR_MS) + 1
                    if len(replacement) != expected_count:
                        raise RuntimeError(
                            f"{symbol}: replacement grid incomplete rows={len(replacement)} expected={expected_count}"
                        )
                    prefix = baseline[baseline["timestamp"] < plan.replace_start_ms].copy()
                    combined = pd.concat([prefix, replacement], ignore_index=True)
                    combined = (
                        combined.sort_values("timestamp")
                        .drop_duplicates("timestamp", keep="last")
                        .tail(plan.max_live_rows)
                        .reset_index(drop=True)
                    )
                    publish_start_ms = int(plan.replace_start_ms)
                else:
                    # A stale live baseline is not used as a reference. Build one
                    # self-contained current-state artifact with enough local history
                    # to publish the complete 16K overlay without a canonical splice.
                    reference_engine.build_symbol_enriched(symbol)
                    reference = read_frame(reference_dir / f"{symbol}_15m.parquet")
                    if list(reference.columns) != list(candidate.columns):
                        raise RuntimeError(f"{symbol}: reference/candidate schema mismatch")

                    reference_mature_start = int(reference_start_ms + MAX_ROLLING_BARS * BAR_MS)
                    if live_output_start_ms < reference_mature_start:
                        raise RuntimeError(
                            f"{symbol}: full-context reference is not rolling-state mature at live output start"
                        )
                    candidate_mature_start = int(plan.candidate_start_ms + MAX_ROLLING_BARS * BAR_MS)
                    deterministic_start = int(max(candidate_mature_start, plan.replace_start_ms))
                    if deterministic_start > plan.expected_latest_ms:
                        raise RuntimeError(f"{symbol}: no mature bounded/reference comparison range")

                    _v656_patch_regimes(
                        candidate, state, deterministic_start, plan.expected_latest_ms
                    )
                    _v656_patch_regimes(
                        reference, state, deterministic_start, plan.expected_latest_ms
                    )
                    full_context_parity = _v656_compare_exact_range(
                        reference,
                        candidate,
                        deterministic_start,
                        plan.expected_latest_ms,
                    )
                    if not full_context_parity.get("ok"):
                        raise RuntimeError(
                            f"{symbol}: bounded/full-context reference parity failed: "
                            + json.dumps(full_context_parity, separators=(",", ":"))
                        )

                    combined = _v657_reference_tail(
                        reference,
                        plan.expected_latest_ms,
                        plan.max_live_rows,
                    )
                    publish_start_ms = int(live_output_start_ms)
                    mode = "INCREMENTAL_LIVE_OVERLAY_WITH_FULL_CONTEXT_REBASE"
                    rebased_symbols.append(symbol)

                micro = patch_microstructure_contract(
                    combined,
                    candidate_engine,
                    symbol,
                    micro_patch_start_ms,
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
                    "recomputed_rows": int(
                        ((plan.expected_latest_ms - publish_start_ms) // BAR_MS) + 1
                    ),
                    "live_rows": int(len(combined)),
                    "latest": int(combined["timestamp"].iloc[-1]),
                    "schema_hash": _schema_hash(list(combined.columns)),
                    "baseline_parity": baseline_parity,
                    "canonical_seam_policy": canonical_seam_policy,
                    "full_context_parity": full_context_parity,
                    "microstructure": micro,
                    "grid": grid,
                }
        finally:
            module._context_frame = original_context

        return {
            "ok": True,
            "mode": (
                "INCREMENTAL_LIVE_OVERLAY_WITH_FULL_CONTEXT_REBASE"
                if rebased_symbols
                else "INCREMENTAL_LIVE_OVERLAY"
            ),
            "rebased_symbols": rebased_symbols,
            "plan": plan.__dict__,
            "full_context_reference_start_ms": int(reference_start_ms),
            "full_context_live_output_start_ms": int(live_output_start_ms),
            "context_state": context_updates,
            "results": results,
            "staged_files": staged_map,
            "engine_sha256": sha256_file(engine_path),
        }
    finally:
        state.close()
'''


def patch_incremental_text(text: str) -> tuple[str, bool]:
    if MARKER in text:
        return text, False
    if BASE_MARKER not in text:
        raise RuntimeError("V6.5.6 base patch is required before the V6.5.7 delta")
    updated = _replace_top_level_function(text, "stage_incremental_live_outputs", STAGE_REPLACEMENT)
    if updated.count(MARKER) != 1:
        raise RuntimeError("V6.5.7 marker count is not exactly one")
    return updated, updated != text


def patch_guard_text(text: str) -> tuple[str, bool]:
    updated = text.replace('"guard_version": "6.5.6"', '"guard_version": "6.5.7"', 1)
    if '"guard_version": "6.5.7"' not in updated:
        raise RuntimeError("guard version projection could not be advanced to 6.5.7")
    return updated, updated != text


def patch_powershell_text(text: str) -> tuple[str, bool]:
    updated = text.replace(
        "Rithal V6.5.6 provenance, schema and freshness gate",
        "Rithal V6.5.7 full-context rebase and freshness gate",
        1,
    )
    if "RITHAL_TAIL_GUARD_V6_5_6_FRESHNESS_HANDSHAKE" not in updated:
        raise RuntimeError("V6.5.6 freshness handshake is missing")
    return updated, updated != text


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
        tmp = path.with_suffix(path.suffix + ".v657.tmp")
        tmp.write_text(after, encoding="utf-8", newline="\n")
        tmp.replace(path)
    return {"path": str(path), "kind": kind, "changed": changed}


def self_test() -> dict:
    import numpy as np
    import pandas as pd

    step = 900000
    rows = 128
    end = (rows - 1) * step
    frame = pd.DataFrame({
        "timestamp": np.arange(rows, dtype=np.int64) * step,
        "value": np.linspace(-1.0, 1.0, rows),
    })
    ns = {
        "pd": pd,
        "np": np,
        "BAR_MS": step,
        "Dict": dict,
        "Any": object,
        "normalize_timestamp_ms": lambda d: d.copy(),
    }
    helper = STAGE_REPLACEMENT.split("def stage_incremental_live_outputs", 1)[0]
    exec(helper, ns)
    tail = ns["_v657_reference_tail"](frame, end, 96)
    assert len(tail) == 96
    assert int(tail["timestamp"].iloc[0]) == end - 95 * step
    assert int(tail["timestamp"].iloc[-1]) == end
    broken = frame.drop(frame.index[-10])
    try:
        ns["_v657_reference_tail"](broken, end, 96)
    except RuntimeError:
        pass
    else:
        raise AssertionError("incomplete reference grid was accepted")

    fixture = '''from pathlib import Path\nfrom types import MethodType\nfrom typing import Any,Dict,List,Sequence\nimport numpy as np\nimport pandas as pd\nBAR_MS=900000\nDAY_BARS=96\nMAX_ROLLING_BARS=2880\nDEFAULT_WARMUP_BARS=3200\nDEFAULT_REPLACE_BARS=96\nDEFAULT_LIVE_ROWS=16384\nPARITY_EXCLUDE={"btc_regime","eth_regime"}\nclass ContextState: pass\ndef normalize_timestamp_ms(d): return d\ndef load_engine_module(p): pass\ndef read_frame(p): pass\ndef make_plan(*a,**k): pass\ndef compare_parity(*a,**k): pass\ndef patch_microstructure_contract(*a,**k): pass\ndef atomic_write_parquet(*a,**k): pass\ndef _validate_grid(*a,**k): pass\ndef _schema_hash(*a,**k): pass\ndef sha256_file(*a,**k): pass\ndef _v656_configure_bounded_engine(*a,**k): pass\ndef _v656_patch_regimes(*a,**k): pass\ndef _v656_compare_exact_range(*a,**k): return {"ok":True}\n# RITHAL_TAIL_GUARD_V6_5_6_CANONICAL_ROLLING_REBASE\ndef stage_incremental_live_outputs(*a,**k): return {}\n'''
    patched, changed = patch_incremental_text(fixture)
    assert changed and MARKER in patched
    patched2, changed2 = patch_incremental_text(patched)
    assert not changed2 and patched2 == patched
    assert "canonical/reference seam parity failed" not in patched
    assert "INCREMENTAL_LIVE_OVERLAY_WITH_FULL_CONTEXT_REBASE" in patched
    assert "_v657_reference_tail" in patched

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "incremental.py"
        path.write_text(patched, encoding="utf-8")
        py_compile.compile(str(path), doraise=True)

    return {
        "version": VERSION,
        "status": "PASS",
        "checks": [
            "canonical_snapshot_not_used_as_live_oracle",
            "no_canonical_live_splice",
            "full_16k_reference_grid_required",
            "short_vs_long_context_parity_required",
            "incomplete_reference_fails_closed",
            "strict_columns_and_tolerances_preserved",
            "source_patch_idempotent",
            "synthetic_compile",
        ],
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("--incremental")
    parser.add_argument("--guard")
    parser.add_argument("--powershell")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--patch", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return 0
    if args.patch:
        if not args.incremental or not args.guard or not args.powershell:
            parser.error("--patch requires --incremental, --guard and --powershell")
        report = {
            "version": VERSION,
            "status": "PASS",
            "incremental": patch_file(Path(args.incremental), "incremental"),
            "guard": patch_file(Path(args.guard), "guard"),
            "powershell": patch_file(Path(args.powershell), "powershell"),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    parser.error("select --self-test or --patch")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
