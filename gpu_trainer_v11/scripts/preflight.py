"""
Pre-flight signal report — pass/fail gate per (rule × horizon).

For each combination of:
    rule    in {A (LONG momentum), B (SHORT mean-reversion)}
    horizon in {16, 32, 96} bars
    symbol  in {BTCUSDT, ETHUSDT, SOLUSDT}    (training pool)

Compute on the FULL history (no train/test split — this is just to see
if there is anything resembling a tradable structure):

    n_signals       = how many bars triggered the primary rule
    base_rate       = fraction of meta_label==1 (raw rule win rate)
    avg_R_net       = mean R_net across signals (slippage included)
    median_R_net    = median R_net
    pf              = profit factor (sum positive R / sum |negative R|)
    shuffle_pf_p95  = 95th percentile of PF under 1000 label shuffles
                      (sanity: real PF should beat its random twin)

Pass criteria PER RULE (across symbols, weighted by n_signals):
    n_signals_total >= 2000
    avg_R_net > 0     after slippage
    pf > shuffle_pf_p95   (the structure is unlikely to be random)

Failing both rules is a hard kill switch — no point training Transformers.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from gpu_trainer_v11.features.compose import compute_features
from gpu_trainer_v11.labels.horizon_conditional import per_bar_barrier_mult
from gpu_trainer_v11.labels.meta_label_v11 import compute_meta_labels_v11
from gpu_trainer_v11.labels.primary_rules import primary_rule

REPO_ROOT = Path(__file__).resolve().parents[2]
DOLLAR_DIR = REPO_ROOT / "gpu_trainer_v11" / "data_cache_dollar"
REPORT_DIR = REPO_ROOT / "gpu_trainer_v11" / "reports"
TRAINING_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
HORIZONS = [16, 32, 96]


def _profit_factor(R: np.ndarray) -> float:
    if R.size == 0:
        return 0.0
    pos = R[R > 0].sum()
    neg = -R[R < 0].sum()
    if neg <= 0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / neg)


def _shuffle_pf_p95(R: np.ndarray, n_iter: int = 500, seed: int = 17) -> float:
    """Sign-shuffle PF distribution. We permute the SIGN of R rather
    than R itself to preserve magnitude distribution; this answers
    "would the same magnitudes random-signed produce this PF?" """
    if R.size == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    mags = np.abs(R)
    pfs = np.zeros(n_iter, dtype=np.float64)
    for i in range(n_iter):
        signs = rng.choice([-1.0, 1.0], size=R.size)
        pfs[i] = _profit_factor(mags * signs)
    pfs = pfs[np.isfinite(pfs)]
    if pfs.size == 0:
        return 0.0
    return float(np.percentile(pfs, 95))


@dataclass
class PreflightCell:
    symbol: str
    rule: str
    horizon: int
    n_signals: int
    base_rate: float
    avg_R_net: float
    median_R_net: float
    pf: float
    shuffle_pf_p95: float


def _load_btc() -> pd.DataFrame | None:
    p = DOLLAR_DIR / "BTCUSDT_dollar.parquet"
    return pd.read_parquet(p) if p.exists() else None


def evaluate_one(
    bars: pd.DataFrame,
    btc_bars: pd.DataFrame | None,
    symbol: str,
    rule: str,
    horizon: int,
) -> PreflightCell:
    bundle = compute_features(bars, btc_bars, symbol)
    primary = primary_rule(bars, bundle.side_data, rule)
    bm = per_bar_barrier_mult(bundle.features.get(
        "atr_pct_bucket", pd.Series(np.ones(len(bars), dtype=np.int8))))
    meta = compute_meta_labels_v11(bars, primary, horizon, bm)
    elig = meta["eligible"].to_numpy()
    R = meta["R_net"].to_numpy()[elig]
    y = meta["meta_label"].to_numpy()[elig]
    if R.size == 0:
        return PreflightCell(symbol, rule, horizon, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return PreflightCell(
        symbol=symbol, rule=rule, horizon=horizon,
        n_signals=int(R.size),
        base_rate=float(y.mean()),
        avg_R_net=float(R.mean()),
        median_R_net=float(np.median(R)),
        pf=_profit_factor(R),
        shuffle_pf_p95=_shuffle_pf_p95(R),
    )


def run() -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    btc = _load_btc()
    cells: list[PreflightCell] = []
    for sym in TRAINING_SYMBOLS:
        p = DOLLAR_DIR / f"{sym}_dollar.parquet"
        if not p.exists():
            print(f"  {sym}: missing dollar bars, skipping")
            continue
        bars = pd.read_parquet(p)
        for rule in ("A", "B"):
            for h in HORIZONS:
                cell = evaluate_one(bars, btc, sym, rule, h)
                cells.append(cell)
                print(f"  {sym} {rule} h={h:>3}  n={cell.n_signals:>6}  "
                      f"base={cell.base_rate:.3f}  avgR={cell.avg_R_net:+.4f}  "
                      f"PF={cell.pf:.2f}  shufPF95={cell.shuffle_pf_p95:.2f}")

    # Aggregate per rule across all training symbols and horizons
    df = pd.DataFrame([asdict(c) for c in cells])
    decisions = {}
    for rule in ("A", "B"):
        sub = df[df["rule"] == rule]
        n_total = int(sub["n_signals"].sum())
        if n_total == 0:
            decisions[rule] = {"pass": False, "reason": "no signals"}
            continue
        # weight by n_signals
        avg_R = float((sub["avg_R_net"] * sub["n_signals"]).sum() / max(n_total, 1))
        # combined PF: pool R values (proxy: weight PF by n_signals)
        pooled_pf = float((sub["pf"] * sub["n_signals"]).sum() / max(n_total, 1))
        pooled_shuffle = float((sub["shuffle_pf_p95"] * sub["n_signals"]).sum() / max(n_total, 1))
        ok = (n_total >= 2000) and (avg_R > 0) and (pooled_pf > pooled_shuffle)
        decisions[rule] = {
            "pass": bool(ok),
            "n_total": n_total,
            "weighted_avg_R": avg_R,
            "weighted_pf": pooled_pf,
            "weighted_shuffle_pf_p95": pooled_shuffle,
        }

    out = {"cells": [asdict(c) for c in cells], "decisions": decisions}
    (REPORT_DIR / "preflight.json").write_text(json.dumps(out, indent=2))

    md_lines = ["# V11 Pre-Flight Report\n",
                "| Symbol | Rule | H | n_signals | base_rate | avg R_net | PF | shuffle PF p95 |",
                "|---|---|---:|---:|---:|---:|---:|---:|"]
    for c in cells:
        md_lines.append(
            f"| {c.symbol} | {c.rule} | {c.horizon} | {c.n_signals} | "
            f"{c.base_rate:.3f} | {c.avg_R_net:+.4f} | {c.pf:.2f} | {c.shuffle_pf_p95:.2f} |"
        )
    md_lines.append("\n## Decisions per rule\n")
    for rule, d in decisions.items():
        md_lines.append(f"- **Rule {rule}**: {'PASS' if d.get('pass') else 'FAIL'}  "
                        f"({json.dumps({k: v for k, v in d.items() if k != 'pass'})})")
    (REPORT_DIR / "preflight.md").write_text("\n".join(md_lines) + "\n")

    print("\n" + "=" * 60)
    for r, d in decisions.items():
        print(f"  Rule {r}: {'PASS' if d.get('pass') else 'FAIL'}  {d}")
    if not any(d.get("pass") for d in decisions.values()):
        print("\n  KILL SWITCH: no rule passed pre-flight. Do NOT proceed to training.")
    return out


def main():
    p = argparse.ArgumentParser()
    p.parse_args()
    run()


if __name__ == "__main__":
    main()
