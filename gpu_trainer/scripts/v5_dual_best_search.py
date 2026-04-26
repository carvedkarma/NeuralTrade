#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.candidate_generator import CandidateConfig
from train.v5_train import V5QualityGateConfig, V5TPDControllerConfig, run_v5_walk_forward


@dataclass
class SearchConfig:
    name: str
    tp_mult: float
    sl_mult: float
    choppy_mode: str
    min_p_side: float
    min_p_short: float
    long_min_mu_r: float
    long_disagree_mult: float
    specialist_align_weight: float
    edge_topn_per_day: int
    min_threshold: float
    max_threshold: float
    cooldown: int
    short_min_fraction: float


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


def _utc_from_ms(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)


def _active_symbols(data_dir: Path, cutoff_date: str) -> list[str]:
    cutoff = pd.Timestamp(cutoff_date, tz="UTC")
    syms: list[str] = []
    for p in sorted(data_dir.glob("*_15m.parquet")):
        sym = p.stem.replace("_15m", "").upper()
        try:
            df = pd.read_parquet(p, columns=["timestamp"])
        except Exception:
            continue
        if df.empty:
            continue
        end_ts = int(pd.to_numeric(df["timestamp"], errors="coerce").dropna().max())
        if not end_ts:
            continue
        if pd.Timestamp(_utc_from_ms(end_ts)) >= cutoff:
            syms.append(sym)
    return syms


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _monthly_from_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        df = pd.read_json(path, lines=True)
    except Exception:
        return []
    if "taken" not in df.columns:
        return []
    t = df[df["taken"] == True].copy()  # noqa: E712
    if t.empty:
        return []
    if "timestamp" not in t.columns:
        return []
    t["timestamp"] = pd.to_numeric(t["timestamp"], errors="coerce")
    t = t[np.isfinite(t["timestamp"])]
    if t.empty:
        return []
    t["month"] = pd.to_datetime(t["timestamp"], unit="ms", utc=True).dt.to_period("M").astype(str)
    if "oracle_r" in t.columns:
        t["oracle_r"] = pd.to_numeric(t["oracle_r"], errors="coerce")
    else:
        t["oracle_r"] = np.nan
    out = (
        t.groupby("month", as_index=False)
        .agg(
            trades=("taken", "size"),
            total_r=("oracle_r", "sum"),
            win_rate=("oracle_r", lambda s: float((s > 0).mean()) if len(s) else 0.0),
            long_trades=("side", lambda s: int((pd.to_numeric(s, errors="coerce") == 1).sum())),
            short_trades=("side", lambda s: int((pd.to_numeric(s, errors="coerce") == -1).sum())),
        )
        .sort_values("month")
    )
    rows = []
    for _, r in out.iterrows():
        rows.append(
            {
                "month": str(r["month"]),
                "trades": int(r["trades"]),
                "total_r": float(r["total_r"]) if np.isfinite(r["total_r"]) else 0.0,
                "win_rate": float(r["win_rate"]) if np.isfinite(r["win_rate"]) else 0.0,
                "long_trades": int(r["long_trades"]),
                "short_trades": int(r["short_trades"]),
            }
        )
    return rows


def _score_result(
    total_r: float,
    win_rate: float,
    mean_monthly_trades: float,
    long_trades: int,
    short_trades: int,
) -> float:
    # Optimize for profitability first, then precision and cadence near 60-90.
    cadence_pen = 0.0
    if mean_monthly_trades < 60.0:
        cadence_pen = (60.0 - mean_monthly_trades) * 1.5
    elif mean_monthly_trades > 90.0:
        cadence_pen = (mean_monthly_trades - 90.0) * 0.8
    side_pen = 0.0
    if long_trades < 10:
        side_pen += (10 - long_trades) * 1.2
    if short_trades < 10:
        side_pen += (10 - short_trades) * 1.2
    return float(total_r + 140.0 * (win_rate - 0.5) - cadence_pen - side_pen)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search best 3-month dual-specialist V5 config.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--active-cutoff", type=str, default="2026-04-20")
    parser.add_argument("--output-json", type=str, default="checkpoints/v5_dual_best_search.json")
    args = parser.parse_args()

    data_dir = Path("/workspace/gpu_trainer/data_cache")
    symbols = _active_symbols(data_dir, args.active_cutoff)
    if not symbols:
        raise SystemExit("No active symbols discovered.")
    # Keep only majors/most liquid to reduce unstable tails.
    preferred = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT", "XRPUSDT"]
    symbols = [s for s in preferred if s in symbols] or symbols

    quality_cfg = V5QualityGateConfig()
    tpd_cfg = V5TPDControllerConfig(min_threshold_floor=0.001)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoints = Path("/workspace/gpu_trainer/checkpoints")
    checkpoints.mkdir(parents=True, exist_ok=True)

    configs = [
        SearchConfig(
            name="balanced-default",
            tp_mult=2.5,
            sl_mult=1.0,
            choppy_mode="BOTH",
            min_p_side=0.45,
            min_p_short=0.0,
            long_min_mu_r=0.0,
            long_disagree_mult=0.3,
            specialist_align_weight=0.5,
            edge_topn_per_day=25,
            min_threshold=0.001,
            max_threshold=0.02,
            cooldown=0,
            short_min_fraction=0.45,
        ),
        SearchConfig(
            name="choppy-short-strictlong",
            tp_mult=2.2,
            sl_mult=1.0,
            choppy_mode="SHORT",
            min_p_side=0.55,
            min_p_short=0.55,
            long_min_mu_r=0.05,
            long_disagree_mult=0.2,
            specialist_align_weight=0.8,
            edge_topn_per_day=15,
            min_threshold=0.005,
            max_threshold=0.04,
            cooldown=1,
            short_min_fraction=0.50,
        ),
        SearchConfig(
            name="choppy-none-strictlong",
            tp_mult=2.0,
            sl_mult=1.0,
            choppy_mode="NONE",
            min_p_side=0.55,
            min_p_short=0.55,
            long_min_mu_r=0.05,
            long_disagree_mult=0.2,
            specialist_align_weight=0.8,
            edge_topn_per_day=12,
            min_threshold=0.008,
            max_threshold=0.05,
            cooldown=1,
            short_min_fraction=0.50,
        ),
        SearchConfig(
            name="high-precision-soft",
            tp_mult=1.8,
            sl_mult=1.0,
            choppy_mode="SHORT",
            min_p_side=0.58,
            min_p_short=0.55,
            long_min_mu_r=0.04,
            long_disagree_mult=0.2,
            specialist_align_weight=0.7,
            edge_topn_per_day=10,
            min_threshold=0.01,
            max_threshold=0.06,
            cooldown=2,
            short_min_fraction=0.50,
        ),
    ]

    results: list[dict[str, Any]] = []
    for cfg in configs:
        cand_path = checkpoints / f"v5_candidates_{cfg.name}.jsonl"
        if cand_path.exists():
            cand_path.unlink()
        fh = cand_path.open("a", encoding="utf-8")

        def candidate_logger(row: dict[str, Any]) -> None:
            fh.write(json.dumps(row, default=_json_default) + "\n")

        try:
            run_v5_walk_forward(
                data_dir=data_dir,
                device=device,
                symbols=symbols,
                epochs=max(1, int(args.epochs)),
                batch_size=max(64, int(args.batch_size)),
                lr=float(args.lr),
                train_months=12,
                test_months=3,
                horizon=48,
                tp_mult=float(cfg.tp_mult),
                sl_mult=float(cfg.sl_mult),
                quality_gate_cfg=quality_cfg,
                tpd_ctrl_cfg=tpd_cfg,
                candidate_config=CandidateConfig(enabled=False),
                risk_controls=None,
                ema200_regime_gate=False,
                ema200_soft_mult=0.5,
                adx_gate=True,
                adx_min=14.0,
                adx_exception_top_pct=10.0,
                slippage_base_bps=1.0,
                min_threshold=float(cfg.min_threshold),
                max_threshold=float(cfg.max_threshold),
                wf_threshold_ema=True,
                wf_threshold_ema_alpha=0.5,
                min_trades=60,
                warmup_skip_bars=48,
                daily_loss_cap=None,
                weekly_loss_cap=None,
                corr_block=True,
                corr_thresh=0.80,
                per_symbol_scaler=True,
                per_symbol_threshold=True,
                per_symbol_r_kill=None,
                symbol_embed_dim=8,
                edge_first=True,
                edge_min=0.025,
                edge_pct_floor=65,
                edge_topn_per_day=int(cfg.edge_topn_per_day),
                regime_side_map={
                    "trending_up": "LONG",
                    "trending_down": "SHORT",
                    "choppy": str(cfg.choppy_mode).upper(),
                },
                regime_soft=True,
                size_floor=0.5,
                adaptive_sizing=True,
                min_size_mult=0.5,
                max_size_mult=1.8,
                kelly_fraction=0.25,
                side_aware_scoring=True,
                recency_weight=True,
                recency_half_life=90,
                warm_start=False,
                finetune_months=0,
                finetune_epochs=0,
                finetune_lr_mult=0.1,
                temp_scale=False,
                promote_metric="expectancy",
                stage_a_epochs=0,
                cooldown=int(cfg.cooldown),
                max_folds=1,
                candidate_logger=candidate_logger,
                dual_specialist=True,
                min_mu_r_long=float(cfg.long_min_mu_r),
                long_disagree_mult=float(cfg.long_disagree_mult),
                specialist_align_weight=float(cfg.specialist_align_weight),
                short_oversample=True,
                short_min_fraction=float(cfg.short_min_fraction),
                per_side_threshold=True,
                min_p_side=float(cfg.min_p_side),
                min_p_short=float(cfg.min_p_short),
            )
        finally:
            fh.close()

        report = _read_json(checkpoints / "v5_forward_report.json")
        monthly = _monthly_from_candidates(cand_path)
        total_trades = int(report.get("total_trades", 0))
        total_r = float(report.get("total_r", 0.0))
        win_rate = float(report.get("win_rate", 0.0))

        if monthly:
            mean_monthly_trades = float(np.mean([m["trades"] for m in monthly]))
            long_trades = int(np.sum([m["long_trades"] for m in monthly]))
            short_trades = int(np.sum([m["short_trades"] for m in monthly]))
        else:
            mean_monthly_trades = float(total_trades / 3.0)
            long_trades = int(report.get("n_long", 0) or 0)
            short_trades = int(report.get("n_short", 0) or 0)

        row = {
            "config": cfg.name,
            "symbols": symbols,
            "tp_mult": cfg.tp_mult,
            "sl_mult": cfg.sl_mult,
            "choppy_mode": cfg.choppy_mode,
            "min_p_side": cfg.min_p_side,
            "min_p_short": cfg.min_p_short,
            "long_min_mu_r": cfg.long_min_mu_r,
            "long_disagree_mult": cfg.long_disagree_mult,
            "specialist_align_weight": cfg.specialist_align_weight,
            "total_trades": total_trades,
            "total_r": total_r,
            "win_rate": win_rate,
            "expectancy_r": float(report.get("expectancy_r", 0.0)),
            "mean_monthly_trades": mean_monthly_trades,
            "long_trades": long_trades,
            "short_trades": short_trades,
            "monthly": monthly,
        }
        row["score"] = _score_result(
            total_r=row["total_r"],
            win_rate=row["win_rate"],
            mean_monthly_trades=row["mean_monthly_trades"],
            long_trades=row["long_trades"],
            short_trades=row["short_trades"],
        )
        results.append(row)

    results = sorted(results, key=lambda r: r["score"], reverse=True)
    payload = {
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "symbols": symbols,
        "results": results,
        "best": results[0] if results else {},
    }
    out_path = Path("/workspace/gpu_trainer") / Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
