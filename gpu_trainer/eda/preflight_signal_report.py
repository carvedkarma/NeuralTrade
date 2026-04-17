"""
V10 Phase 1 pre-flight signal report.

For each candidate horizon (4h, 8h, 1d on 15m bars) we report:

  - candidate-bar count (primary_dir != 0 AND triple-barrier valid)
  - meta-label class balance
  - prior-4h-momentum primary rule's UNCONDITIONAL win rate (= mean meta_label)
  - mean / median realized R (gross and net)
  - top-30 features by mutual information with the meta-label

This costs minutes, not GPU-hours, and tells us whether the data has any
chance of carrying signal at the new horizons before we burn cycles
training XGBoost.

Run:
    cd gpu_trainer
    python -m eda.preflight_signal_report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
GPU_TRAINER_DIR = THIS_DIR.parent
if str(GPU_TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(GPU_TRAINER_DIR))

from data.pipeline import FeatureEngineer  # noqa: E402
from labels.meta_label import HORIZONS_BARS, compute_meta_labels  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("preflight")

DATA_CACHE = GPU_TRAINER_DIR / "data_cache"
REPORT_DIR = GPU_TRAINER_DIR / "reports" / "v10_phase1"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _load_btc() -> pd.DataFrame:
    p = DATA_CACHE / "BTCUSDT_15m.parquet"
    if not p.exists():
        raise FileNotFoundError(f"BTC parquet missing: {p}")
    df = pd.read_parquet(p).sort_values("timestamp").reset_index(drop=True)
    log.info("BTC loaded: %d bars  range %s -> %s",
             len(df),
             datetime.utcfromtimestamp(df["timestamp"].iloc[0] / 1000),
             datetime.utcfromtimestamp(df["timestamp"].iloc[-1] / 1000))
    return df


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    eng = FeatureEngineer()
    feats = eng.compute_all_features(df)
    log.info("Features computed: %d columns", feats.shape[1])
    return feats


def _mutual_info(features: pd.DataFrame, y: np.ndarray, mask: np.ndarray,
                 max_rows: int = 50_000, top_k: int = 30) -> list:
    from sklearn.feature_selection import mutual_info_classif

    X = features.loc[mask].to_numpy()
    yy = y[mask]
    finite_rows = np.isfinite(X).all(axis=1)
    X = X[finite_rows]
    yy = yy[finite_rows]
    if len(X) > max_rows:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_rows, replace=False)
        X = X[idx]
        yy = yy[idx]
    if len(X) < 1000:
        log.warning("Only %d finite rows for MI; skipping", len(X))
        return []
    log.info("MI on %d rows x %d features...", X.shape[0], X.shape[1])
    mi = mutual_info_classif(X, yy, discrete_features=False, random_state=42)
    rank = np.argsort(-mi)
    cols = list(features.columns)
    return [
        {"rank": i + 1, "feature": cols[j], "mutual_info": float(mi[j])}
        for i, j in enumerate(rank[:top_k])
    ]


def run_preflight() -> dict:
    df = _load_btc()
    feats = _compute_features(df)
    out = {"generated_at": datetime.utcnow().isoformat(), "symbol": "BTCUSDT",
           "n_bars": int(len(df)), "horizons": {}}
    for hname, hbars in HORIZONS_BARS.items():
        log.info("=== Horizon %s (%d bars) ===", hname, hbars)
        meta = compute_meta_labels(df, horizon_bars=hbars)
        elig = meta["eligible"].to_numpy()
        y = meta["meta_label"].to_numpy()
        R_net = meta["R_net"].to_numpy()
        R_gross = meta["R_gross"].to_numpy()
        primary = meta["primary_dir"].to_numpy()

        n_total = int(len(df))
        n_elig = int(elig.sum())
        wins = int(y[elig].sum())
        losses = n_elig - wins
        long_mask = elig & (primary == 1)
        short_mask = elig & (primary == -1)

        mi = _mutual_info(feats, y, elig)
        horizon_report = {
            "horizon_bars": hbars,
            "n_total_bars": n_total,
            "n_eligible_bars": n_elig,
            "n_long_candidates": int(long_mask.sum()),
            "n_short_candidates": int(short_mask.sum()),
            "primary_rule_unconditional_win_rate": float(wins / n_elig) if n_elig else None,
            "long_unconditional_win_rate": float(y[long_mask].mean()) if long_mask.any() else None,
            "short_unconditional_win_rate": float(y[short_mask].mean()) if short_mask.any() else None,
            "R_gross_mean": float(np.nanmean(R_gross[elig])) if n_elig else None,
            "R_gross_median": float(np.nanmedian(R_gross[elig])) if n_elig else None,
            "R_net_mean": float(np.nanmean(R_net[elig])) if n_elig else None,
            "R_net_median": float(np.nanmedian(R_net[elig])) if n_elig else None,
            "wins": wins,
            "losses": losses,
            "top_features_by_mutual_info": mi,
        }
        out["horizons"][hname] = horizon_report
        log.info(
            "  bars=%d eligible=%d  WR=%.3f  long=%d/%.3f  short=%d/%.3f  R_net_mean=%+.4f",
            n_total, n_elig,
            horizon_report["primary_rule_unconditional_win_rate"] or 0.0,
            horizon_report["n_long_candidates"], horizon_report["long_unconditional_win_rate"] or 0.0,
            horizon_report["n_short_candidates"], horizon_report["short_unconditional_win_rate"] or 0.0,
            horizon_report["R_net_mean"] or 0.0,
        )

    json_path = REPORT_DIR / "preflight_signal_report.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Wrote %s", json_path)

    md_path = REPORT_DIR / "preflight_signal_report.md"
    with open(md_path, "w") as f:
        f.write("# V10 Phase 1 — Pre-flight Signal Report (BTCUSDT)\n\n")
        f.write(f"_Generated {out['generated_at']}_  \n")
        f.write(f"Bars: {out['n_bars']:,}  \n\n")
        for hname, h in out["horizons"].items():
            f.write(f"## Horizon `{hname}` ({h['horizon_bars']} bars)\n\n")
            f.write(f"- Eligible candidate bars: **{h['n_eligible_bars']:,}** "
                    f"(LONG {h['n_long_candidates']:,}, SHORT {h['n_short_candidates']:,})\n")
            f.write(f"- Unconditional primary-rule win rate: "
                    f"**{(h['primary_rule_unconditional_win_rate'] or 0):.3f}**  "
                    f"(LONG {(h['long_unconditional_win_rate'] or 0):.3f}, "
                    f"SHORT {(h['short_unconditional_win_rate'] or 0):.3f})\n")
            f.write(f"- R gross mean: **{(h['R_gross_mean'] or 0):+.4f}**, "
                    f"R net mean (after 6 bps): **{(h['R_net_mean'] or 0):+.4f}**\n\n")
            f.write("### Top 30 features by mutual information vs meta-label\n\n")
            f.write("| Rank | Feature | MI |\n|---:|---|---:|\n")
            for row in h["top_features_by_mutual_info"]:
                f.write(f"| {row['rank']} | `{row['feature']}` | {row['mutual_info']:.5f} |\n")
            f.write("\n")
    log.info("Wrote %s", md_path)
    return out


def main():
    p = argparse.ArgumentParser()
    p.parse_args()
    run_preflight()


if __name__ == "__main__":
    main()
