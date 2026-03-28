#!/usr/bin/env python3
"""
validate_signals.py — V5 signal quality validator.

Loads a V5 checkpoint (or runs a quick single-fold walk-forward), then
reports a comprehensive diagnostic:
  1. Score spread   (p1 / p50 / p95 / p99 / range)
  2. LONG / SHORT split at every threshold slice
  3. Score decile table  (combined + LONG + SHORT per fold)
  4. Per-symbol status   (ACTIVE / HIGH_BAR / NO_EDGE)
  5. Debias spread ratio check (warns if mu_R discrimination collapsed)

Exit codes:
  0 — at least 1 symbol ACTIVE AND score_monotonic=True in at least 1 fold
  1 — all symbols HIGH_BAR / no trades / no monotone fold / negative E[R]
  2 — fatal error (import failure, missing data, etc.)

Usage (run from gpu_trainer directory):
    python scripts/validate_signals.py --symbols BTCUSDT ETHUSDT SOLUSDT \\
        --folds 1 --epochs 50 --data-dir data_cache

    # With an existing checkpoint directory:
    python scripts/validate_signals.py --model-path checkpoints/v5_btcusdt \\
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


def _print_score_spread(scores_arr, label="ALL"):
    import numpy as np
    finite = scores_arr[np.isfinite(scores_arr)]
    if len(finite) == 0:
        log.warning("  [%s] No finite scores", label)
        return
    p1, p50, p95, p99 = (
        float(np.percentile(finite, 1)),
        float(np.percentile(finite, 50)),
        float(np.percentile(finite, 95)),
        float(np.percentile(finite, 99)),
    )
    spread = p99 - p1
    log.info("  [%s] Score spread: p1=%.5f  p50=%.5f  p95=%.5f  p99=%.5f  range=%.5f",
             label, p1, p50, p95, p99, spread)
    if spread < 1e-4:
        log.warning("  [%s] WARNING: score spread near-zero (%.2e) — model may be collapsed", label, spread)


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
    if args.model_path:
        log.info("Model path : %s  (checkpoint-eval mode)", args.model_path)
    log.info("=" * 80)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Device: %s", device)
    except ImportError:
        device = "cpu"

    t0 = time.time()
    extra_kwargs = {}
    if args.model_path:
        extra_kwargs["checkpoint_dir"] = args.model_path

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
            # CRITICAL: w_action=2.5 (action head must dominate side learning)
            w_action=2.5,
            **extra_kwargs,
        )
    except Exception:
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
    log.info("PER-FOLD DETAIL (Score Spread + Decile Monotonicity)")
    log.info("=" * 80)
    fold_has_active_monotone = False
    for i, f in enumerate(folds):
        if not isinstance(f, dict):
            continue
        n_tr = f.get("total_trades", 0)
        er = f.get("expectancy_r", 0.0)
        wr = f.get("win_rate", 0.0)
        sharpe = f.get("sharpe", 0.0)
        total_r_fold = f.get("total_r", 0.0)
        mono = f.get("score_monotonic")
        debias_ratio = f.get("debias_spread_ratio")
        mono_str = "PASS" if mono else ("FAIL" if mono is not None else "N/A")
        log.info(f"  Fold {i+1:>2}: trades={n_tr:>5}  E[R]={er:>+8.4f}  "
                 f"WR={wr*100:>5.1f}%  Sharpe={sharpe:>6.2f}  "
                 f"TotalR={total_r_fold:>+8.4f}  decile_mono={mono_str}")
        if debias_ratio is not None:
            flag = "  [OK]" if debias_ratio >= 5.0 else "  [COLLAPSED!]"
            log.info(f"         debias_spread_ratio={debias_ratio:.2f}x{flag}")

        decile_rows = f.get("score_decile_table", [])
        if decile_rows:
            log.info(f"         Score decile table (fold {i+1}):")
            log.info(f"           {'Label':<10} {'ScoreLo':>9} {'ScoreHi':>9} "
                     f"{'N':>5} {'AvgR':>8} {'WR':>7}")
            for row in decile_rows:
                lbl = row.get("label", f"D{row['decile']:02d}")
                log.info(f"           {lbl:<10} {row['score_lo']:>9.4f} {row['score_hi']:>9.4f} "
                         f"{row['n_trades']:>5} {row['avg_r']:>+8.4f} {row['win_rate']:>6.1%}")

        sym_stats = f.get("per_symbol_stats", {})
        if sym_stats:
            log.info(f"         Per-symbol (fold {i+1}):")
            for sn, ss in sym_stats.items():
                log.info(f"           {sn:>12}: trades={ss['trades']:>5}  "
                         f"E[R]={ss['expectancy_r']:>+.4f}  WR={ss['win_rate']:.1%}  "
                         f"Total={ss['total_r']:>+.4f}R")

        active_and_monotone = (
            n_tr > 0 and er > 0 and mono is True
        )
        if active_and_monotone:
            fold_has_active_monotone = True

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
                status_tag = status if status else "NO_EDGE"
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
        log.error("  2. mu_debias may have collapsed mu_R spread (check debias_spread_ratio < 5x).")
        log.error("  3. quality_gate may be rejecting all bars (check [V5_DEBIAS_SPREAD] log).")
        log.error("  4. Epochs may be too low — action head stuck in mean-prediction plateau.")
        log.error("  FIX: Retrain with w_action=2.5 and epochs>=100.")
        sys.exit(1)
    elif not fold_has_active_monotone:
        log.warning("WARN: Trades generated (%d) but no fold has E[R]>0 AND score_monotonic=True.", total_trades)
        log.warning("  Score-to-return monotonicity is required for live edge.")
        log.warning("  Current avg E[R] = %+.4f.", avg_er)
        if avg_er <= 0:
            log.warning("  Additionally, aggregate E[R] is negative — model is trading but losing.")
        log.warning("  Check [SIDE_BIAS] warnings in training log.")
        log.warning("  Consider: more epochs, short_oversample, or per_symbol_threshold tuning.")
        sys.exit(1)
    else:
        log.info("PASS: %d trades  E[R]=%+.4f  Total R=%+.4f", total_trades, avg_er, total_r)
        log.info("At least 1 fold has ACTIVE signals WITH monotone score decile ordering.")
        log.info("Model is generating positive-expectancy, score-ordered signals.")
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
    parser.add_argument("--model-path", default=None,
                        help="Optional: path to checkpoint directory for eval mode. "
                             "When provided, walk-forward is still run but model weights are "
                             "initialized from checkpoint (checkpoint_dir kwarg).")
    args = parser.parse_args()

    _check_imports()
    _run_validation(args)


if __name__ == "__main__":
    main()
