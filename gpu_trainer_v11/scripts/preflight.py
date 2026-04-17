"""
Pre-flight signal report — pass/fail gate per (rule × horizon × symbol).

For each combination of:
    rule    in {A (LONG momentum), B (SHORT mean-reversion)}
    horizon in {16, 32, 96} bars
    symbol  in {BTCUSDT, ETHUSDT, SOLUSDT}    (training pool)

Compute on the FULL history (no train/test split — this is just to see
if there is anything resembling a tradable structure):

    n_signals     = how many bars triggered the primary rule
    base_rate     = fraction of meta_label==1 (raw rule win rate)
    avg_R_net     = mean R_net across signals (slippage included)
    pf            = profit factor (sum positive R / sum |negative R|)
    shuffle_pf_p95= 95th percentile of PF under sign-shuffle of R
                    (sanity: real PF should beat its random twin)

PASS criterion per RULE (per session-plan T006 spec, locked):
    advance any (rule, horizon) where R_net > 0 on at least one
    training-pool symbol. The shuffle PF is REPORTED as a sanity
    metric only and never used to gate advancement.

This is intentionally permissive — it's a kill-switch for the case
where the rule generates only negative-expectancy signals across the
entire pool. Full PF >= 1.3 / 500 trades hurdles are checked in
walk-forward, not here.
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
    if R.size == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    mags = np.abs(R)
    pfs = np.zeros(n_iter, dtype=np.float64)
    for i in range(n_iter):
        signs = rng.choice([-1.0, 1.0], size=R.size)
        pfs[i] = _profit_factor(mags * signs)
    pfs = pfs[np.isfinite(pfs)]
    return 0.0 if pfs.size == 0 else float(np.percentile(pfs, 95))


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
    cell_pass: bool


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
        return PreflightCell(symbol, rule, horizon, 0, 0.0, 0.0, 0.0, 0.0, 0.0, False)
    pf = _profit_factor(R)
    sh = _shuffle_pf_p95(R)
    avg_R = float(R.mean())
    return PreflightCell(
        symbol=symbol, rule=rule, horizon=horizon,
        n_signals=int(R.size),
        base_rate=float(y.mean()),
        avg_R_net=avg_R,
        median_R_net=float(np.median(R)),
        pf=pf,
        shuffle_pf_p95=sh,
        cell_pass=bool(avg_R > 0),
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
                tag = "PASS" if cell.cell_pass else "fail"
                print(f"  {sym} {rule} h={h:>3}  n={cell.n_signals:>6}  "
                      f"base={cell.base_rate:.3f}  avgR={cell.avg_R_net:+.4f}  "
                      f"PF={cell.pf:.2f}  shufPF95={cell.shuffle_pf_p95:.2f}  [{tag}]")

    df = pd.DataFrame([asdict(c) for c in cells])
    # Decisions are at (rule, horizon) granularity: a (rule, horizon) PASSes
    # iff at least one training-pool symbol shows R_net > 0 for that pair.
    # train_walkforward.py only admits a (rule, horizon) that passed here.
    decisions: dict[str, dict] = {}
    for rule in ("A", "B"):
        sub = df[df["rule"] == rule]
        per_horizon: dict[str, dict] = {}
        for h in HORIZONS:
            cell_sub = sub[sub["horizon"] == h] if not sub.empty else sub
            n_passing = int(cell_sub["cell_pass"].sum()) if not cell_sub.empty else 0
            passing = cell_sub[cell_sub["cell_pass"]] if not cell_sub.empty else cell_sub
            per_horizon[str(h)] = {
                "pass": bool(n_passing > 0),
                "n_passing_symbols": n_passing,
                "passing_symbols": [
                    {"symbol": r["symbol"], "avg_R_net": float(r["avg_R_net"]),
                     "pf": float(r["pf"])}
                    for _, r in passing.iterrows()
                ],
            }
        decisions[rule] = {
            "pass": any(d["pass"] for d in per_horizon.values()),
            "passing_horizons": [int(h) for h, d in per_horizon.items() if d["pass"]],
            "per_horizon": per_horizon,
        }

    out = {"cells": [asdict(c) for c in cells], "decisions": decisions}
    (REPORT_DIR / "preflight.json").write_text(json.dumps(out, indent=2))

    md_lines = ["# V11 Pre-Flight Report\n",
                "Pass criterion per (symbol × rule × horizon) cell: **avg R_net > 0**.\n",
                "Shuffle PF p95 is reported as a sanity metric only (not gated on).\n",
                "Pass criterion per rule: at least one passing cell.\n",
                "| Symbol | Rule | H | n_signals | base_rate | avg R_net | PF | shuffle PF p95 | cell |",
                "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for c in cells:
        md_lines.append(
            f"| {c.symbol} | {c.rule} | {c.horizon} | {c.n_signals} | "
            f"{c.base_rate:.3f} | {c.avg_R_net:+.4f} | {c.pf:.2f} | {c.shuffle_pf_p95:.2f} | "
            f"{'PASS' if c.cell_pass else 'fail'} |"
        )
    md_lines.append("\n## Decisions per (rule, horizon)\n")
    for rule, d in decisions.items():
        md_lines.append(f"- **Rule {rule}**: passing horizons "
                        f"{sorted(d.get('passing_horizons', []))}")
        for h, dh in d["per_horizon"].items():
            md_lines.append(f"  - h={h}: {'PASS' if dh['pass'] else 'FAIL'} "
                            f"({dh['n_passing_symbols']} passing symbols)")
    (REPORT_DIR / "preflight.md").write_text("\n".join(md_lines) + "\n")

    print("\n" + "=" * 60)
    for r, d in decisions.items():
        print(f"  Rule {r}: passing horizons {sorted(d.get('passing_horizons', []))}")
    if not any(d.get("pass") for d in decisions.values()):
        print("\n  KILL SWITCH: no rule passed pre-flight. Do NOT proceed to training.")
    return out


def main():
    p = argparse.ArgumentParser()
    p.parse_args()
    run()


if __name__ == "__main__":
    main()
