#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.candidate_generator import CandidateConfig
from train.v5_train import V5QualityGateConfig, V5TPDControllerConfig, run_v5_walk_forward


@dataclass
class SweepConfig:
    name: str
    soft_ema: bool
    min_p_side: float
    min_p_short: float
    adx_min: float
    edge_min: float
    edge_pct_floor: int
    edge_topn_per_day: int
    min_threshold: float
    max_threshold: float
    relax_caps: bool
    regime_soft: bool


def _utc_date_from_ms(ts_ms: int) -> datetime:
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
        end_dt = _utc_date_from_ms(end_ts)
        if pd.Timestamp(end_dt) >= cutoff:
            syms.append(sym)
    return syms


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _score_row(row: dict[str, Any]) -> float:
    wr = float(row.get("win_rate", 0.0))
    exp_r = float(row.get("expectancy_r", 0.0))
    total_r = float(row.get("total_r", 0.0))
    trades = float(row.get("total_trades", 0.0))
    low_conf = bool(row.get("low_confidence", False))
    # Precision-first score with minimum throughput guard.
    score = (100.0 * wr) + (30.0 * exp_r) + (0.04 * total_r) + (0.002 * trades)
    if trades < 60:
        score -= 40.0
    if low_conf:
        score -= 25.0
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-specialist V5 precision sweep on 3-month window.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--window-end", type=str, default="2026-04-23")
    parser.add_argument("--active-cutoff", type=str, default="2026-04-20")
    parser.add_argument("--output-json", type=str, default="checkpoints/v5_dual_precision_sweep.json")
    parser.add_argument("--max-configs", type=int, default=2,
                        help="Run first N configs from the preset list (default: 2 for faster turnaround).")
    args = parser.parse_args()

    data_dir = Path("/workspace/gpu_trainer/data_cache")
    symbols = _active_symbols(data_dir, args.active_cutoff)
    if not symbols:
        raise SystemExit("No active symbols discovered for requested cutoff.")

    print(f"[SWEEP] Active symbols ({len(symbols)}): {symbols}")
    print(f"[SWEEP] Window end target: {args.window_end}")

    quality_cfg = V5QualityGateConfig()
    tpd_cfg = V5TPDControllerConfig(min_threshold_floor=0.001)
    regime_side_map = {"trending_up": "LONG", "trending_down": "SHORT", "choppy": "BOTH"}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    configs = [
        SweepConfig(
            name="dual-baseline-strict",
            soft_ema=False,
            min_p_side=0.45,
            min_p_short=0.00,
            adx_min=14.0,
            edge_min=0.025,
            edge_pct_floor=65,
            edge_topn_per_day=25,
            min_threshold=0.001,
            max_threshold=0.02,
            relax_caps=False,
            regime_soft=True,
        ),
        SweepConfig(
            name="dual-precision-softema",
            soft_ema=True,
            min_p_side=0.58,
            min_p_short=0.55,
            adx_min=18.0,
            edge_min=0.040,
            edge_pct_floor=75,
            edge_topn_per_day=12,
            min_threshold=0.010,
            max_threshold=0.060,
            relax_caps=True,
            regime_soft=True,
        ),
        SweepConfig(
            name="dual-ultra-precision",
            soft_ema=True,
            min_p_side=0.62,
            min_p_short=0.60,
            adx_min=20.0,
            edge_min=0.050,
            edge_pct_floor=80,
            edge_topn_per_day=8,
            min_threshold=0.015,
            max_threshold=0.080,
            relax_caps=True,
            regime_soft=False,
        ),
    ]

    checkpoints_dir = Path("/workspace/gpu_trainer/checkpoints")
    results: list[dict[str, Any]] = []

    max_cfg = max(1, int(args.max_configs))
    run_configs = configs[:max_cfg]
    for cfg in run_configs:
        print(f"\n[SWEEP] Running config: {cfg.name}")
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
            tp_mult=3.0,
            sl_mult=1.0,
            quality_gate_cfg=quality_cfg,
            tpd_ctrl_cfg=tpd_cfg,
            candidate_config=CandidateConfig(enabled=False),
            risk_controls=None,
            ema200_regime_gate=not cfg.soft_ema,
            ema200_soft_mult=0.5 if cfg.soft_ema else None,
            adx_gate=True,
            adx_min=float(cfg.adx_min),
            adx_exception_top_pct=10.0,
            slippage_base_bps=1.0,
            min_threshold=float(cfg.min_threshold),
            max_threshold=float(cfg.max_threshold),
            wf_threshold_ema=True,
            wf_threshold_ema_alpha=0.5,
            min_trades=60,
            warmup_skip_bars=48,
            daily_loss_cap=None if cfg.relax_caps else -4.0,
            weekly_loss_cap=None if cfg.relax_caps else -12.0,
            corr_block=True,
            corr_thresh=0.80,
            per_symbol_scaler=True,
            per_symbol_threshold=True,
            per_symbol_r_kill=None if cfg.relax_caps else -8.0,
            symbol_embed_dim=8,
            edge_first=True,
            edge_min=float(cfg.edge_min),
            edge_pct_floor=int(cfg.edge_pct_floor),
            edge_topn_per_day=int(cfg.edge_topn_per_day),
            regime_side_map=regime_side_map,
            regime_soft=bool(cfg.regime_soft),
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
            cooldown=0,
            max_folds=1,
            # Dual specialist + long head-agreement controls.
            dual_specialist=True,
            min_mu_r_long=0.0,
            long_disagree_mult=0.3,
            specialist_align_weight=0.5,
            # Push side balance learning harder for precision.
            short_oversample=True,
            short_min_fraction=0.45,
            per_side_threshold=True,
            min_p_side=float(cfg.min_p_side),
            min_p_short=float(cfg.min_p_short),
        )

        forward_report = _load_json(checkpoints_dir / "v5_forward_report.json")
        wf_report = _load_json(checkpoints_dir / "v5_walkforward_report.json")
        fold = {}
        if isinstance(wf_report, dict):
            folds = wf_report.get("folds", [])
            if folds:
                fold = folds[-1]

        out = {
            "config": cfg.name,
            "symbols": symbols,
            "window_start": forward_report.get("window_start", fold.get("window_start")),
            "window_end": forward_report.get("window_end", fold.get("window_end")),
            "total_trades": int(forward_report.get("total_trades", 0)),
            "total_r": float(forward_report.get("total_r", 0.0)),
            "win_rate": float(forward_report.get("win_rate", 0.0)),
            "expectancy_r": float(forward_report.get("expectancy_r", 0.0)),
            "profit_factor": float(forward_report.get("profit_factor", 0.0)),
            "n_long": int(forward_report.get("n_long", 0)),
            "n_short": int(forward_report.get("n_short", 0)),
            "score_threshold": float(forward_report.get("score_threshold", 0.0)),
            "gate_cutoff": float(forward_report.get("gate_cutoff", 0.0)),
            "gate_mode": str(forward_report.get("gate_mode", "")),
            "low_confidence": bool(forward_report.get("low_confidence", False)),
            "mu_r_correlation": float((forward_report.get("prediction_quality") or {}).get("mu_r_correlation") or 0.0),
            "gate_blocks": (forward_report.get("directional_balance") or {}).get("gate_blocks") or {},
        }
        out["precision_score"] = _score_row(out)
        results.append(out)

        (checkpoints_dir / f"v5_forward_report_{cfg.name}.json").write_text(json.dumps(forward_report, indent=2, default=str))
        (checkpoints_dir / f"v5_walkforward_report_{cfg.name}.json").write_text(json.dumps(wf_report, indent=2, default=str))

        print(
            f"[SWEEP] {cfg.name}: trades={out['total_trades']} win_rate={out['win_rate']:.4f} "
            f"expR={out['expectancy_r']:.4f} totalR={out['total_r']:.4f} "
            f"L/S={out['n_long']}/{out['n_short']} mu_corr={out['mu_r_correlation']:.4f}"
        )

    results_sorted = sorted(results, key=lambda r: r["precision_score"], reverse=True)
    payload = {
        "symbols": symbols,
        "window_end_target": args.window_end,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "max_configs": int(max_cfg),
        "results": results_sorted,
        "best": results_sorted[0] if results_sorted else {},
    }
    out_path = Path("/workspace/gpu_trainer") / Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))

    print("\n[SWEEP] Top by precision score:")
    for r in results_sorted:
        print(
            f"  {r['config']:<24} score={r['precision_score']:.3f} "
            f"WR={r['win_rate']:.4f} expR={r['expectancy_r']:.4f} "
            f"trades={r['total_trades']} L/S={r['n_long']}/{r['n_short']}"
        )
    print(f"[SWEEP] Saved: {out_path}")


if __name__ == "__main__":
    main()
