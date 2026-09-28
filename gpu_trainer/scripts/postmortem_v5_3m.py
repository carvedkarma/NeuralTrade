#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/workspace/gpu_trainer/checkpoints")
REPORT = ROOT / "v5_forward_report.json"
WF_REPORT = ROOT / "v5_walkforward_report.json"
CAND = ROOT / "v5_candidates_allsym_3m_fold1.jsonl"


def _print(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def main() -> None:
    if not REPORT.exists():
        raise SystemExit(f"Missing report: {REPORT}")

    rep = json.loads(REPORT.read_text())

    _print("V5 3-MONTH POSTMORTEM")
    for k in [
        "window_start",
        "window_end",
        "total_trades",
        "total_r",
        "win_rate",
        "expectancy_r",
        "profit_factor",
        "n_long",
        "n_short",
        "low_confidence",
        "gate_mode",
        "gate_cutoff",
        "score_threshold",
        "gate_select_rate",
        "gate_pass_rate",
    ]:
        if k in rep:
            print(f"{k:>20}: {rep[k]}")

    gb = (rep.get("directional_balance") or {}).get("gate_blocks") or {}
    if gb:
        _print("Gate block counts (descending)")
        for k, v in sorted(gb.items(), key=lambda kv: -kv[1]):
            if v:
                print(f"{k:>30}: {v}")

    side_q = rep.get("side_quality") or {}
    if side_q:
        _print("Side quality")
        for k in [
            "long_avg_score",
            "short_avg_score",
            "long_avg_mu_r",
            "short_avg_mu_r",
            "long_head_agree_pct",
            "short_head_agree_pct",
            "short_disagree_pct",
            "short_disagree_expect",
        ]:
            if k in side_q:
                print(f"{k:>30}: {side_q[k]}")

    ps = rep.get("per_symbol_stats") or {}
    if ps:
        _print("Per-symbol trade stats")
        rows = []
        for sym, d in ps.items():
            rows.append(
                {
                    "symbol": sym,
                    "trades": d.get("trades", 0),
                    "win_rate": d.get("win_rate", np.nan),
                    "expectancy_r": d.get("expectancy_r", np.nan),
                    "total_r": d.get("total_r", np.nan),
                }
            )
        df = pd.DataFrame(rows).sort_values("trades", ascending=False)
        print(df.to_string(index=False))

    pq = rep.get("prediction_quality") or {}
    if pq:
        _print("Prediction quality")
        for k in [
            "action_accuracy",
            "mu_r_correlation",
            "mean_predicted_mu_R",
            "mean_actual_R",
            "mean_p_side",
            "mean_p_side_winners",
            "mean_p_side_losers",
        ]:
            if k in pq:
                print(f"{k:>30}: {pq[k]}")

    if CAND.exists() and CAND.stat().st_size > 0:
        dfc = pd.read_json(CAND, lines=True)
        _print("Candidate-level block reasons")
        if "taken" in dfc.columns:
            print(f"candidates_total={len(dfc)} taken={int(dfc['taken'].sum())}")
        if "block_reason" in dfc.columns:
            br = (
                dfc.loc[dfc["taken"] != True, "block_reason"]
                .fillna("")
                .astype(str)
            )
            br = br[br != ""]
            print("\nTop block reasons:")
            print(br.value_counts().head(20).to_string())

            if "oracle_r" in dfc.columns and len(br) > 0:
                tmp = dfc.loc[dfc["taken"] != True, ["block_reason", "oracle_r"]].copy()
                tmp = tmp[tmp["block_reason"].notna() & (tmp["block_reason"].astype(str) != "")]
                tmp["oracle_r"] = pd.to_numeric(tmp["oracle_r"], errors="coerce")
                agg = (
                    tmp.groupby("block_reason")["oracle_r"]
                    .agg(["count", "mean", "sum"])
                    .sort_values("count", ascending=False)
                )
                _print("Blocked-trade oracle R by gate")
                print(agg.head(20).to_string())

        if "symbol" in dfc.columns:
            b = dfc[dfc["taken"] != True].copy()
            if not b.empty and "block_reason" in b.columns:
                b["block_reason"] = b["block_reason"].fillna("").astype(str)
                b = b[b["block_reason"] != ""]
                if not b.empty:
                    piv = (
                        b.pivot_table(
                            index="symbol",
                            columns="block_reason",
                            values="bar_idx",
                            aggfunc="count",
                            fill_value=0,
                        )
                        .assign(total=lambda x: x.sum(axis=1))
                        .sort_values("total", ascending=False)
                    )
                    _print("Symbol x block-reason matrix (top symbols)")
                    print(piv.head(12).to_string())

    if WF_REPORT.exists():
        wf = json.loads(WF_REPORT.read_text())
        _print("WF aggregate")
        print(json.dumps(wf.get("aggregate", {}), indent=2))
        folds = wf.get("folds", [])
        if folds:
            _print("Last fold snapshot")
            print(
                json.dumps(
                    {
                        k: folds[-1].get(k)
                        for k in [
                            "fold",
                            "window_start",
                            "window_end",
                            "total_trades",
                            "total_r",
                            "win_rate",
                            "expectancy_r",
                            "profit_factor",
                            "n_long",
                            "n_short",
                            "gate_mode",
                            "gate_cutoff",
                            "score_threshold",
                            "threshold_ema",
                            "low_confidence",
                        ]
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
