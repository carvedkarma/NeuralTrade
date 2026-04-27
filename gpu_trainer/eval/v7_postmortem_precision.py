"""V7 Postmortem + precision-first policy search.

Purpose:
  1) Reconcile the apparent conflict between positive net-bps and negative R.
  2) Quantify which symbols/regimes/time buckets dominate R drag.
  3) Find robust, precision-first policy slices with positive net R.

Data source:
  Uses cached path-level trades produced by:
    python -m gpu_trainer.eval.v7_payoff_fix

Run:
  python -m gpu_trainer.eval.v7_postmortem_precision
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from gpu_trainer.eval.v7_payoff_fix import apply_walk_forward_filter
from gpu_trainer.eval.v7_brilliant import per_symbol_history_gate, WHITELIST


COST_BPS = 8.0
COST_FRAC = COST_BPS / 1e4
CACHE_DIR = Path(".local/cache/v7_payoff_fix")
OUT_JSON = Path(".local/reports/v7_postmortem_precision.json")
OUT_MD = Path(".local/reports/v7_postmortem_precision.md")


@dataclass
class SliceStats:
    n: int
    mean_gross_bps: float
    mean_net_bps: float
    win_rate_net_pct: float
    avg_r_net: float
    median_r_net: float
    mean_cost_over_risk: float
    mean_gross_over_risk: float
    months: int
    win_month_pct: float
    mean_monthly_total_r: float


def _load_cache() -> pd.DataFrame:
    pieces = [pd.read_parquet(p) for p in sorted(CACHE_DIR.glob("*.parquet"))]
    if not pieces:
        raise RuntimeError(
            f"no cache found under {CACHE_DIR}; run v7_payoff_fix first"
        )
    return pd.concat(pieces, ignore_index=True)


def _apply_e2_filters(raw: pd.DataFrame, symbols: Iterable[str]) -> pd.DataFrame:
    df = raw[raw["symbol"].isin(list(symbols))].copy()
    df = df[df["abs_pred_q"] >= 4].copy()  # top ~2%
    hist = per_symbol_history_gate(df, "gross_fixed", COST_FRAC)
    df = df[hist.values].copy()
    keep_parts = []
    for sym in sorted(df["symbol"].unique()):
        s = df[df["symbol"] == sym]
        keep_parts.append(apply_walk_forward_filter(s, "gross_fixed", COST_FRAC))
    keep = pd.concat(keep_parts).sort_index()
    return df[keep.values].copy()


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[out["risk_pct"].notna() & (out["risk_pct"] > 1e-8)].copy()
    out["net"] = out["gross_fixed"] - COST_FRAC
    out["R_net"] = out["net"] / out["risk_pct"]
    out["R_gross"] = out["gross_fixed"] / out["risk_pct"]
    out["risk_bps"] = out["risk_pct"] * 1e4
    out["month"] = (
        pd.to_datetime(out["ts"], unit="ms", utc=True)
        .dt.to_period("M")
        .astype(str)
    )
    return out


def _slice_stats(df: pd.DataFrame) -> SliceStats:
    if df.empty:
        return SliceStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    m = df.groupby("month")["R_net"].sum()
    return SliceStats(
        n=int(len(df)),
        mean_gross_bps=float(df["gross_fixed"].mean() * 1e4),
        mean_net_bps=float(df["net"].mean() * 1e4),
        win_rate_net_pct=float((df["net"] > 0).mean() * 100),
        avg_r_net=float(df["R_net"].mean()),
        median_r_net=float(df["R_net"].median()),
        mean_cost_over_risk=float((COST_FRAC / df["risk_pct"]).mean()),
        mean_gross_over_risk=float((df["gross_fixed"] / df["risk_pct"]).mean()),
        months=int(df["month"].nunique()),
        win_month_pct=float((m > 0).mean() * 100),
        mean_monthly_total_r=float(m.mean()),
    )


def _rank_positive_slices(df: pd.DataFrame) -> pd.DataFrame:
    symbol_sets = [
        ("ADAUSDT",),
        ("XRPUSDT",),
        ("ADAUSDT", "XRPUSDT"),
        ("ADAUSDT", "XRPUSDT", "AVAXUSDT"),
        tuple(sorted(df["symbol"].unique())),
    ]
    risk_floors = [0, 40, 60, 80, 100, 120, 150, 200]
    sessions = [None, "Asia", "EU", "US", "Late"]
    regimes = [None, "WITH", "COUNTER"]

    rows: list[dict] = []
    for syms in symbol_sets:
        base = df[df["symbol"].isin(syms)]
        for rf in risk_floors:
            d = base[base["risk_bps"] >= rf]
            for sess in sessions:
                ds = d if sess is None else d[d["session"] == sess]
                for reg in regimes:
                    x = ds if reg is None else ds[ds["regime"] == reg]
                    if len(x) < 200:
                        continue
                    fold_avg = x.groupby("fold")["R_net"].mean()
                    month_tot = x.groupby("month")["R_net"].sum()
                    rows.append({
                        "symbols": ",".join(syms),
                        "risk_floor_bps": rf,
                        "session": sess or "ALL",
                        "regime": reg or "ALL",
                        "n": int(len(x)),
                        "months": int(x["month"].nunique()),
                        "avg_R_net": float(x["R_net"].mean()),
                        "mean_net_bps": float(x["net"].mean() * 1e4),
                        "win_month_pct": float((month_tot > 0).mean() * 100),
                        "mean_monthly_total_R": float(month_tot.mean()),
                        "fold_min_avg_R": float(fold_avg.min()),
                        "fold_pos_count": int((fold_avg > 0).sum()),
                        "fold_count": int(len(fold_avg)),
                    })
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    robust = res[
        (res["avg_R_net"] > 0)
        & (res["months"] >= 15)
        & (res["n"] >= 300)
    ].copy()
    robust["robust_score"] = (
        robust["avg_R_net"]
        + 0.15 * (robust["fold_pos_count"] / robust["fold_count"])
        + 0.002 * robust["win_month_pct"]
    )
    robust = robust.sort_values(
        ["robust_score", "avg_R_net", "n"], ascending=False
    )
    return robust


def main() -> None:
    raw = _load_cache()
    current = _enrich(_apply_e2_filters(raw, WHITELIST))
    current_stats = _slice_stats(current)

    # Precision candidate selected from robust search:
    #   symbols ADA+XRP, risk floor 100 bps, keep all sessions/regimes.
    p1 = current[
        current["symbol"].isin(["ADAUSDT", "XRPUSDT"])
        & (current["risk_bps"] >= 100)
    ].copy()
    p1_stats = _slice_stats(p1)

    # Precision P2: strongest robust slice from broad search
    # (LONG-only, 4-symbol subset, vol floor 80 bps).
    p2 = current[
        current["symbol"].isin(["ADAUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
        & (current["risk_bps"] >= 80)
        & (current["direction"] > 0)
    ].copy()
    p2_stats = _slice_stats(p2)

    drag_by_symbol = (
        current.groupby("symbol")
        .agg(
            n=("symbol", "size"),
            mean_net_bps=("net", lambda x: float(np.mean(x) * 1e4)),
            avg_R_net=("R_net", "mean"),
            total_R=("R_net", "sum"),
            mean_cost_over_risk=("risk_pct", lambda x: float(np.mean(COST_FRAC / x))),
            win_rate_net_pct=("net", lambda x: float(np.mean(x > 0) * 100)),
        )
        .reset_index()
        .sort_values("avg_R_net")
    )

    robust = _rank_positive_slices(current)
    top_robust = robust.head(25).to_dict("records") if not robust.empty else []

    payload = {
        "cost_bps": COST_BPS,
        "current_e2_stats": asdict(current_stats),
        "precision_p1_stats": asdict(p1_stats),
        "precision_p2_stats": asdict(p2_stats),
        "delta_p1_minus_current": {
            "mean_net_bps": p1_stats.mean_net_bps - current_stats.mean_net_bps,
            "avg_R_net": p1_stats.avg_r_net - current_stats.avg_r_net,
            "win_month_pct": p1_stats.win_month_pct - current_stats.win_month_pct,
        },
        "delta_p2_minus_current": {
            "mean_net_bps": p2_stats.mean_net_bps - current_stats.mean_net_bps,
            "avg_R_net": p2_stats.avg_r_net - current_stats.avg_r_net,
            "win_month_pct": p2_stats.win_month_pct - current_stats.win_month_pct,
        },
        "r_drag_by_symbol": drag_by_symbol.to_dict("records"),
        "top_robust_positive_slices": top_robust,
        "precision_policy_definition": {
            "symbols": ["ADAUSDT", "XRPUSDT"],
            "top_pct": 2.0,
            "hold_minutes": 60,
            "risk_floor_bps": 100,
            "history_gate": True,
            "cell_filter": True,
            "session_filter": "ALL",
            "regime_filter": "ALL",
        },
        "precision_p2_policy_definition": {
            "symbols": ["ADAUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
            "top_pct": 2.0,
            "hold_minutes": 60,
            "direction": "LONG_ONLY",
            "risk_floor_bps": 80,
            "history_gate": True,
            "cell_filter": True,
            "session_filter": "ALL",
            "regime_filter": "ALL",
        }
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    lines = [
        "# V7 Postmortem + Precision Policy",
        "",
        "## Core finding",
        "",
        "The previous mismatch came from optimizing net-bps while cost/risk was still too high for broad-symbol deployment.",
        "For net-R to be positive, gross/risk must exceed cost/risk. Most symbols fail this at 8 bps unless filtered tightly.",
        "",
        "## Current E2 vs Precision P1",
        "",
        f"- Current E2 avg net bps: {current_stats.mean_net_bps:+.2f}",
        f"- Current E2 avg net R: {current_stats.avg_r_net:+.3f}",
        f"- Precision P1 avg net bps: {p1_stats.mean_net_bps:+.2f}",
        f"- Precision P1 avg net R: {p1_stats.avg_r_net:+.3f}",
        f"- Delta avg net R: {p1_stats.avg_r_net - current_stats.avg_r_net:+.3f}",
        f"- Precision P2 avg net bps: {p2_stats.mean_net_bps:+.2f}",
        f"- Precision P2 avg net R: {p2_stats.avg_r_net:+.3f}",
        f"- Precision P2 mean monthly total R: {p2_stats.mean_monthly_total_r:+.2f}",
        "",
        "## Precision P1 definition",
        "",
        "- symbols: ADAUSDT + XRPUSDT",
        "- selectivity: top 2% |pred|",
        "- hold: 60 minutes",
        "- risk floor: vol_16 >= 100 bps",
        "- keep history gate + cell filter",
        "",
        "## Precision P2 definition (runtime best robust)",
        "",
        "- symbols: ADAUSDT + ETHUSDT + SOLUSDT + XRPUSDT",
        "- selectivity: top 2% |pred|",
        "- hold: 60 minutes",
        "- direction: LONG-only",
        "- risk floor: vol_16 >= 80 bps",
        "- keep history gate + cell filter",
        "",
        "## Worst R-drag symbols in current E2",
        "",
        "| symbol | n | avg net bps | avg net R | mean cost/risk | win rate net % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in drag_by_symbol.to_dict("records"):
        lines.append(
            f"| {row['symbol']} | {row['n']:,} | "
            f"{row['mean_net_bps']:+.2f} | {row['avg_R_net']:+.3f} | "
            f"{row['mean_cost_over_risk']:.3f} | {row['win_rate_net_pct']:.1f} |"
        )

    lines += ["", "## Top robust positive-R slices (search)", ""]
    if top_robust:
        lines += [
            "| symbols | risk floor bps | session | regime | n | avg net R | net bps | win-month % |",
            "|---|---:|---|---|---:|---:|---:|---:|",
        ]
        for row in top_robust[:12]:
            lines.append(
                f"| {row['symbols']} | {row['risk_floor_bps']} | "
                f"{row['session']} | {row['regime']} | {row['n']:,} | "
                f"{row['avg_R_net']:+.3f} | {row['mean_net_bps']:+.2f} | "
                f"{row['win_month_pct']:.1f} |"
            )
    else:
        lines.append("_No robust positive-R slice found under current constraints._")

    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
