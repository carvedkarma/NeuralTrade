#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.candidate_generator import CandidateConfig, RiskControls
from train.v5_train import V5QualityGateConfig, V5TPDControllerConfig, train_v5_model


CHECKPOINTS = Path("/workspace/gpu_trainer/checkpoints")
DATA_PATH = Path("/workspace/gpu_trainer/data_cache/BTCUSDT_15m.parquet")
TRAIN_END = "2024-11-01"
TEST_START = "2024-11-01"
TEST_END = "2025-03-31"


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _score_variant(total_r: float, trades: int, win_rate: float) -> float:
    trade_pen = 0.0
    if trades < 20:
        trade_pen = float(20 - trades) * 0.25
    return float(total_r + 3.0 * (win_rate - 0.5) - trade_pen)


def _candidate_monthly(candidate_log: Path) -> list[dict[str, Any]]:
    if not candidate_log.exists() or candidate_log.stat().st_size == 0:
        return []
    df = pd.read_json(candidate_log, lines=True)
    if "taken" not in df.columns or "timestamp" not in df.columns:
        return []
    t = df[df["taken"] == True].copy()
    if t.empty:
        return []
    t["month"] = (
        pd.to_datetime(pd.to_numeric(t["timestamp"], errors="coerce"), unit="ms", utc=True)
        .dt.to_period("M")
        .astype(str)
    )
    t["oracle_r"] = pd.to_numeric(t.get("oracle_r", 0.0), errors="coerce").fillna(0.0)
    g = (
        t.groupby("month")
        .agg(trades=("taken", "size"), total_r=("oracle_r", "sum"))
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for row in g.to_dict(orient="records"):
        rows.append(
            {
                "month": str(row["month"]),
                "trades": int(row["trades"]),
                "total_r": float(row["total_r"]),
            }
        )
    return rows


def _candidate_block_reasons(candidate_log: Path, top_n: int = 10) -> dict[str, int]:
    if not candidate_log.exists() or candidate_log.stat().st_size == 0:
        return {}
    df = pd.read_json(candidate_log, lines=True)
    if "taken" not in df.columns or "block_reason" not in df.columns:
        return {}
    b = df[df["taken"] != True]["block_reason"].fillna("").astype(str)
    b = b[b != ""]
    if b.empty:
        return {}
    vc = b.value_counts().head(top_n)
    return {str(k): int(v) for k, v in vc.items()}


def _run_variant(
    *,
    name: str,
    symbols: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    model_version: str,
    phase1_epochs: int,
    ema200_soft_mult: float | None,
    cooldown: int,
    min_p_side: float,
    min_p_short: float,
    edge_topn_per_day: int,
    per_symbol_threshold: bool,
    per_side_threshold: bool,
    dual_specialist: bool,
) -> dict[str, Any]:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    candidate_log = CHECKPOINTS / f"v5_candidates_{name}.jsonl"
    if candidate_log.exists():
        candidate_log.unlink()

    with candidate_log.open("a", encoding="utf-8") as cand_f:
        def candidate_logger(row: dict[str, Any]) -> None:
            cand_f.write(json.dumps(row, default=_json_default) + "\n")

        qual_cfg = V5QualityGateConfig()
        tpd_cfg = V5TPDControllerConfig(min_threshold_floor=0.001)
        result = train_v5_model(
            data_path=DATA_PATH,
            device="cuda" if torch.cuda.is_available() else "cpu",
            epochs=int(max(1, epochs)),
            batch_size=int(max(64, batch_size)),
            lr=float(lr),
            horizon=48,
            tp_mult=3.0,
            sl_mult=1.0,
            symbols=symbols,
            phase1_epochs=int(max(0, phase1_epochs)),
            score_lambda=0.30,
            risk_proxy="mae",
            hold_target=0.30,
            mfe_min=0.05,
            quality_gate_cfg=qual_cfg,
            tpd_ctrl_cfg=tpd_cfg,
            candidate_config=CandidateConfig(enabled=False),
            risk_controls=RiskControls(),
            train_end_date=TRAIN_END,
            test_start_date=TEST_START,
            test_end_date=TEST_END,
            run_forward_test=True,
            freeze_decision=True,
            ema200_regime_gate=ema200_soft_mult is None,
            ema200_soft_mult=ema200_soft_mult,
            adx_gate=True,
            adx_min=14.0,
            adx_exception_top_pct=10.0,
            slippage_base_bps=1.0,
            min_threshold=0.001,
            max_threshold=0.02,
            min_trades=60,
            warmup_skip_bars=48,
            daily_loss_cap=-4.0,
            weekly_loss_cap=-12.0,
            corr_block=True,
            corr_thresh=0.80,
            per_symbol_scaler=True,
            per_symbol_threshold=per_symbol_threshold,
            per_symbol_r_kill=-8.0,
            symbol_embed_dim=8,
            edge_first=True,
            edge_min=0.025,
            edge_pct_floor=65,
            edge_topn_per_day=int(max(1, edge_topn_per_day)),
            regime_side_map={
                "trending_up": "LONG",
                "trending_down": "SHORT",
                "choppy": "BOTH",
            },
            size_floor=0.5,
            adaptive_sizing=True,
            min_size_mult=0.5,
            max_size_mult=1.8,
            kelly_fraction=0.25,
            side_aware_scoring=True,
            recency_weight=True,
            recency_half_life=90,
            short_oversample=True,
            short_min_fraction=0.45,
            per_side_threshold=per_side_threshold,
            min_p_side=float(np.clip(min_p_side, 0.0, 0.99)),
            min_p_short=float(np.clip(min_p_short, 0.0, 0.99)),
            dual_specialist=dual_specialist,
            specialist_mode="none",
            min_mu_r_long=0.0,
            long_disagree_mult=0.3,
            specialist_align_weight=0.5,
            cooldown=int(max(0, cooldown)),
            model_version=model_version,
            candidate_logger=candidate_logger,
        )

    fwd_src = CHECKPOINTS / "v5_forward_report.json"
    wf_src = CHECKPOINTS / "v5_walkforward_report.json"
    fwd_tagged = CHECKPOINTS / f"v5_forward_report_{name}.json"
    wf_tagged = CHECKPOINTS / f"v5_walkforward_report_{name}.json"
    if fwd_src.exists():
        shutil.copy2(fwd_src, fwd_tagged)
    if wf_src.exists():
        shutil.copy2(wf_src, wf_tagged)

    fwd = json.loads(fwd_tagged.read_text()) if fwd_tagged.exists() else {}
    total_r = float(fwd.get("total_r", 0.0))
    trades = int(fwd.get("total_trades", 0))
    win_rate = float(fwd.get("win_rate", 0.0))
    summary = {
        "name": name,
        "model_version": model_version,
        "phase1_epochs": int(phase1_epochs),
        "epochs": int(epochs),
        "ema200_soft_mult": ema200_soft_mult,
        "cooldown": int(cooldown),
        "min_p_side": float(min_p_side),
        "min_p_short": float(min_p_short),
        "per_symbol_threshold": bool(per_symbol_threshold),
        "per_side_threshold": bool(per_side_threshold),
        "dual_specialist": bool(dual_specialist),
        "total_trades": trades,
        "total_r": total_r,
        "win_rate": win_rate,
        "expectancy_r": float(fwd.get("expectancy_r", 0.0)),
        "profit_factor": float(fwd.get("profit_factor", 0.0)),
        "n_long": int(fwd.get("n_long", 0)),
        "n_short": int(fwd.get("n_short", 0)),
        "low_confidence": bool(fwd.get("low_confidence", False)),
        "score": _score_variant(total_r=total_r, trades=trades, win_rate=win_rate),
        "candidate_blocks_top": _candidate_block_reasons(candidate_log),
        "monthly_taken_oracle_r": _candidate_monthly(candidate_log),
        "aggregate": (result or {}).get("aggregate", {}),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize Nov2024-Mar2025 using multi-model sweep.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT", "XRPUSDT"],
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="/workspace/gpu_trainer/checkpoints/v5_nov2024_mar2025_finalizer_summary.json",
    )
    args = parser.parse_args()

    variants = [
        {
            "name": "nov2024_mar2025_v5_baseline",
            "model_version": "v5",
            "phase1_epochs": 1,
            "ema200_soft_mult": None,
            "cooldown": 4,
            "min_p_side": 0.45,
            "min_p_short": 0.0,
            "edge_topn_per_day": 25,
            "per_symbol_threshold": True,
            "per_side_threshold": True,
            "dual_specialist": True,
        },
        {
            "name": "nov2024_mar2025_v5_throughput",
            "model_version": "v5",
            "phase1_epochs": 1,
            "ema200_soft_mult": 0.50,
            "cooldown": 1,
            "min_p_side": 0.38,
            "min_p_short": 0.0,
            "edge_topn_per_day": 35,
            "per_symbol_threshold": False,
            "per_side_threshold": False,
            "dual_specialist": True,
        },
        {
            "name": "nov2024_mar2025_v6_balanced",
            "model_version": "v6",
            "phase1_epochs": 1,
            "ema200_soft_mult": 0.50,
            "cooldown": 1,
            "min_p_side": 0.40,
            "min_p_short": 0.0,
            "edge_topn_per_day": 30,
            "per_symbol_threshold": False,
            "per_side_threshold": False,
            "dual_specialist": False,
        },
    ]

    rows: list[dict[str, Any]] = []
    for variant in variants:
        print(f"\n=== Running variant: {variant['name']} ===")
        row = _run_variant(
            name=variant["name"],
            symbols=list(args.symbols),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            model_version=str(variant["model_version"]),
            phase1_epochs=int(variant["phase1_epochs"]),
            ema200_soft_mult=variant["ema200_soft_mult"],
            cooldown=int(variant["cooldown"]),
            min_p_side=float(variant["min_p_side"]),
            min_p_short=float(variant["min_p_short"]),
            edge_topn_per_day=int(variant["edge_topn_per_day"]),
            per_symbol_threshold=bool(variant["per_symbol_threshold"]),
            per_side_threshold=bool(variant["per_side_threshold"]),
            dual_specialist=bool(variant["dual_specialist"]),
        )
        rows.append(row)
        print(
            f"{row['name']}: trades={row['total_trades']} total_r={row['total_r']:+.4f} "
            f"wr={100.0*row['win_rate']:.1f}% score={row['score']:+.4f}"
        )

    rows.sort(key=lambda x: (x["score"], x["total_r"], x["total_trades"]), reverse=True)
    best = rows[0] if rows else {}
    summary = {
        "window": {"train_end": TRAIN_END, "test_start": TEST_START, "test_end": TEST_END},
        "symbols": list(args.symbols),
        "variants": rows,
        "best": best,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=_json_default))
    print("\n=== Final ranked variants ===")
    print(json.dumps(summary, indent=2, default=_json_default))
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
