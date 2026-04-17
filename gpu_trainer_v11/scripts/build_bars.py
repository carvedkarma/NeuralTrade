"""Build dollar bars for all symbols.

Two modes:
  --diagnostic  : sweep candidate thresholds on BTCUSDT, write report,
                  print recommended threshold. Does NOT lock anything.
  --build       : read locked threshold from --threshold flag and emit
                  a per-symbol parquet to data_cache_dollar/.

Run --diagnostic first, decide on a threshold, update README.md's
"Locked dollar-bar size" section, then run --build --threshold X.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gpu_trainer_v11.bars.diagnostics import pick_threshold, sweep
from gpu_trainer_v11.bars.dollar_bars import (
    DollarBarConfig,
    bars_per_day,
    build_dollar_bars,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_15M_DIR = REPO_ROOT / "gpu_trainer" / "data_cache"
OUT_DIR = REPO_ROOT / "gpu_trainer_v11" / "data_cache_dollar"
REPORT_DIR = REPO_ROOT / "gpu_trainer_v11" / "reports"

# Coarse logarithmic sweep — refine if needed
DEFAULT_CANDIDATES = [
    1_000_000, 2_000_000, 5_000_000,
    10_000_000, 20_000_000, 50_000_000,
    100_000_000, 200_000_000, 500_000_000,
]


def load_15m(symbol: str) -> pd.DataFrame:
    p = DATA_15M_DIR / f"{symbol}_15m.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def cmd_diagnostic(args: argparse.Namespace) -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_15m("BTCUSDT")
    print(f"BTCUSDT 15m: {len(df):,} bars from "
          f"{pd.to_datetime(df['timestamp'].iloc[0], unit='ms')} to "
          f"{pd.to_datetime(df['timestamp'].iloc[-1], unit='ms')}")
    print(f"Sweeping {len(DEFAULT_CANDIDATES)} thresholds…")
    sweep_df = sweep(df, DEFAULT_CANDIDATES)
    chosen = pick_threshold(sweep_df, target_bars_per_day=args.target_bars_per_day)
    sweep_df["chosen"] = sweep_df["threshold_dollars"] == chosen
    out = REPORT_DIR / "bar_sweep.csv"
    sweep_df.to_csv(out, index=False)
    print(sweep_df.to_string(index=False))
    print(f"\nRecommended threshold: ${chosen:,.0f}")
    print(f"Report written: {out}")
    print("\nNext: update README.md 'Locked dollar-bar size' section,")
    print(f"then run:  python -m gpu_trainer_v11.scripts.build_bars --build --threshold {int(chosen)}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    if args.threshold is None or args.threshold <= 0:
        raise SystemExit("--threshold required and must be positive")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = DollarBarConfig(threshold_dollars=float(args.threshold))

    symbols = args.symbols
    if not symbols:
        symbols = sorted(p.stem.replace("_15m", "") for p in DATA_15M_DIR.glob("*_15m.parquet"))
    print(f"Building dollar bars for {len(symbols)} symbols at ${args.threshold:,}")

    summary = []
    for sym in symbols:
        try:
            df = load_15m(sym)
        except FileNotFoundError:
            print(f"  {sym}: missing 15m parquet, skipping")
            continue
        db = build_dollar_bars(df, cfg)
        out = OUT_DIR / f"{sym}_dollar.parquet"
        db.to_parquet(out, index=False)
        bpd = bars_per_day(db)
        print(f"  {sym}: {len(db):,} bars  ({bpd:.1f}/day) -> {out.name}")
        summary.append({
            "symbol": sym,
            "n_bars": int(len(db)),
            "bars_per_day": float(bpd),
            "first_ts": int(db["timestamp"].iloc[0]) if len(db) else None,
            "last_ts": int(db["timestamp"].iloc[-1]) if len(db) else None,
        })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "dollar_bars_summary.json").write_text(
        json.dumps({"threshold_dollars": float(args.threshold), "symbols": summary}, indent=2)
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=False)

    # also support flags-only style
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--build", action="store_true")
    p.add_argument("--threshold", type=float, default=None,
                   help="Locked dollar threshold (use after --diagnostic)")
    p.add_argument("--target-bars-per-day", type=float, default=96.0)
    p.add_argument("--symbols", nargs="*", default=None,
                   help="Optional subset of symbols (default: all in data_cache/)")
    args = p.parse_args()

    if args.diagnostic and args.build:
        raise SystemExit("Choose --diagnostic or --build, not both")
    if args.diagnostic:
        return cmd_diagnostic(args)
    if args.build:
        return cmd_build(args)
    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
