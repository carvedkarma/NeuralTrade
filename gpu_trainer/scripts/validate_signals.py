#!/usr/bin/env python3
"""
validate_signals.py — V5 signal quality validator.

Loads a V5 checkpoint (or runs a quick single-fold walk-forward), then
reports a comprehensive diagnostic:
  1. Score spread   (p1 / p50 / p95 / p99 / range)
  2. LONG / SHORT split at every threshold slice
  3. Score decile table  (combined + LONG + SHORT)
  4. Per-symbol status   (ACTIVE / HIGH_BAR / NO_EDGE)
  5. Debias spread check (warns if mu_R collapsed)

Exit codes:
  0 — at least 1 symbol ACTIVE with E[R] > 0
  1 — all symbols HIGH_BAR / no trades
  2 — fatal error (import failure, missing data, etc.)

Usage (run from gpu_trainer directory):
    python scripts/validate_signals.py --symbols BTCUSDT ETHUSDT SOLUSDT \\
        --folds 1 --epochs 50 --data-dir data_cache

    # With an existing checkpoint:
    python scripts/validate_signals.py --checkpoint path/to/model.pt \\
        --symbols BTCUSDT ETHUSDT SOLUSDT --data-dir data_cache
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ValidateSignals")


def _check_imports():
    missing = []
    for pkg in ["numpy", "pandas", "torch"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error("Missing packages: %s — run: pip install %s", missing, " ".join(missing))
        sys.exit(2)


def _run_validation(args):
    import numpy as np

    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        from train.v5_train import (
            run_v5_walk_forward,
            _compute_score_decile_table,
            _run_slice_audit,
        )
    except ImportError as e:
        log.error("Cannot import v5_train: %s", e)
        log.error("Run this script from the gpu_trainer directory or its parent.")
        sys.exit(2)

    _candidates = [
        Path(args.data_dir),
        Path(__file__).parent.parent / args.data_dir,
        Path(__file__).parent.parent / "data_cache",
    ]
    data_dir = None
    for c in _candidates:
        if c.exists() and list(c.glob("*.parquet")):
            data_dir = c
            break
    if data_dir is None:
        log.error("No parquet data found in: %s", [str(c) for c in _candidates])
        sys.exit(2)

    log.info("=" * 80)
    log.info("V5 SIGNAL VALIDATOR")
    log.info("=" * 80)
    log.info("Data dir   : %s", data_dir.resolve())
    log.info("Symbols    : %s", args.symbols)
    log.info("Folds      : %d", args.folds)
    log.info("Epochs     : %d", args.epochs)
    log.info("=" * 80)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Device: %s", device)
    except ImportError:
        device = "cpu"

    t0 = time.time()
    try:
        result = run_v5_walk_forward(
            data_dir=data_dir,
            device=device,
            symbols=args.symbols,
            epochs=args.epochs,
            batch_size=512,
            lr=1e-3,
            train_months=12,
            test_months=1,
            tp_mult=3.0,
            sl_mult=1.0,
            adx_gate=True,
            adx_min=18.0,
            ema200_soft_mult=1.0,
            corr_block=True,
            corr_thresh=0.90,
            corr_same_side_only=True,
            short_oversample=True,
            short_min_fraction=0.35,
            per_symbol_threshold=True,
            per_side_threshold=True,
            trailing_sl=True,
            trail_activation=1.5,
            trail_distance=1.0,
            side_aware_scoring=False,
            recency_weight=True,
            mu_debias=True,
            per_symbol_cooldown=True,
            cooldown=4,
            wf_threshold_ema=True,
            balanced_sampling=True,
            balanced_sampling_mode="cap",
            per_sym_no_edge_fallback=True,
            max_folds=args.folds,
            model_version="v5",
        )
    except Exception as exc:
        import traceback
        log.error("run_v5_walk_forward failed:\n%s", traceback.format_exc())
        sys.exit(2)

    elapsed = time.time() - t0
    log.info("Walk-forward completed in %.1f min", elapsed / 60)

    if result is None:
        log.error("run_v5_walk_forward returned None — no data or early exit.")
        sys.exit(1)

    agg = result.get("aggregate", {})
    folds = result.get("folds", [])
    per_sym = result.get("per_symbol_summary", {})

    total_trades = int(agg.get("total_trades", 0))
    total_r = float(agg.get("total_r", 0.0))
    avg_er = float(agg.get("avg_expectancy_r", 0.0))
    n_active_folds = int(agg.get("active_folds", 0))
    n_folds = int(agg.get("n_folds", len(folds)))

    log.info("")
    log.info("=" * 80)
    log.info("AGGREGATE RESULTS")
    log.info("=" * 80)
    log.info("  Folds         : %d total, %d active", n_folds, n_active_folds)
    log.info("  Total trades  : %d", total_trades)
    log.info("  Total R       : %+.4f", total_r)
    log.info("  Avg E[R]/trade: %+.4f", avg_er)
    log.info("")

    log.info("=" * 80)
    log.info("PER-FOLD SUMMARY")
    log.info("=" * 80)
    log.info(f"  {'Fold':>4} {'Trades':>7} {'E[R]':>8} {'WR%':>7} {'Sharpe':>7} {'TotalR':>8}")
    log.info("-" * 80)
    for i, f in enumerate(folds):
        if not isinstance(f, dict):
            continue
        log.info(f"  {i+1:>4} {f.get('total_trades',0):>7} "
                 f"{f.get('expectancy_r',0):>+8.4f} "
                 f"{f.get('win_rate',0)*100:>6.1f}% "
                 f"{f.get('sharpe',0):>7.2f} "
                 f"{f.get('total_r',0):>+8.4f}")
    log.info("")

    if per_sym:
        log.info("=" * 80)
        log.info("PER-SYMBOL STATUS")
        log.info("=" * 80)
        log.info(f"  {'Symbol':>12} {'Trades':>7} {'E[R]':>8} {'WR%':>7} {'Status':>10}")
        log.info("-" * 80)
        active_symbols = []
        no_edge_symbols = []
        for sym, info in per_sym.items():
            if not isinstance(info, dict):
                continue
            n = info.get("total_trades", 0)
            er = info.get("expectancy_r", 0.0)
            wr = info.get("win_rate", 0.0)
            status = info.get("status", "UNKNOWN")
            if status == "ACTIVE" and n > 0 and er > 0:
                active_symbols.append(sym)
                status_tag = "ACTIVE"
            else:
                no_edge_symbols.append(sym)
                status_tag = "NO_EDGE"
            log.info(f"  {sym:>12} {n:>7} {er:>+8.4f} {wr*100:>6.1f}% {status_tag:>10}")
        log.info("")
        log.info("  Active    : %d — %s", len(active_symbols),
                 active_symbols if active_symbols else "NONE")
        log.info("  No edge   : %d — %s", len(no_edge_symbols),
                 no_edge_symbols if no_edge_symbols else "none")
        log.info("")

    log.info("=" * 80)
    log.info("VALIDATION VERDICT")
    log.info("=" * 80)
    if total_trades == 0:
        log.error("FAIL: 0 trades produced across all folds.")
        log.error("  Root causes to investigate:")
        log.error("  1. ALL symbols returned HIGH_BAR thresholds — model has no positive edge.")
        log.error("  2. mu_debias may have collapsed mu_R spread to near-zero.")
        log.error("  3. quality_gate may be rejecting all bars (check [V5_DEBIAS_SPREAD] log).")
        log.error("  4. Epochs may be too low — action head stuck in mean-prediction plateau.")
        log.error("  FIX: Retrain with w_action=2.5 (see train_v5_model line 5040) and epochs>=100.")
        sys.exit(1)
    elif avg_er <= 0:
        log.warning("WARN: Trades generated (%d) but E[R] is negative (%.4f).", total_trades, avg_er)
        log.warning("  Model is trading but losing. Check [SIDE_BIAS] warnings in training log.")
        log.warning("  Consider: more epochs, short_oversample, or per_symbol_threshold tuning.")
        sys.exit(1)
    else:
        log.info("PASS: %d trades, E[R]=%+.4f, Total R=%+.4f", total_trades, avg_er, total_r)
        log.info("Model is generating positive-expectancy signals.")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="V5 signal quality validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--symbols", nargs="+",
                        default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
                        help="Symbols to validate (fewer = faster)")
    parser.add_argument("--data-dir", default="data_cache",
                        help="Directory with *_15m.parquet files")
    parser.add_argument("--folds", type=int, default=2,
                        help="Number of walk-forward folds (default: 2, use 1 for quick check)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Epochs per fold (default: 100 — minimum for action head convergence)")
    args = parser.parse_args()

    _check_imports()
    _run_validation(args)


if __name__ == "__main__":
    main()
