#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.candidate_generator import CandidateConfig
from train.v5_train import V5QualityGateConfig, V5TPDControllerConfig, run_v5_walk_forward


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


def _run_single(
    *,
    tag: str,
    symbols: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    use_quality_blend: bool,
    quality_blend_min_q: float,
    quality_blend_regimes: list[str],
) -> dict[str, Any]:
    checkpoints = Path("/workspace/gpu_trainer/checkpoints")
    checkpoints.mkdir(parents=True, exist_ok=True)
    candidate_log_path = checkpoints / f"v5_candidates_{tag}.jsonl"
    if candidate_log_path.exists():
        candidate_log_path.unlink()

    quality_blend_weights = {
        "h4_rsi14": -0.104,
        "h1_ema200_pos": -0.103,
        "ema200_pos_15m": -0.070,
        "h4_trend_sign": -0.063,
        "return_100": -0.060,
        "h1_atr_ratio": 0.059,
        "regime_session_sin": -0.059,
        "h1_trend_sign": -0.059,
    }

    with candidate_log_path.open("a", encoding="utf-8") as candidate_log_file:
        def candidate_logger(row: dict[str, Any]) -> None:
            candidate_log_file.write(json.dumps(row, default=_json_default) + "\n")

        result = run_v5_walk_forward(
            data_dir=Path("/workspace/gpu_trainer/data_cache"),
            device="cuda" if torch.cuda.is_available() else "cpu",
            symbols=symbols,
            epochs=max(1, int(epochs)),
            batch_size=max(64, int(batch_size)),
            lr=float(lr),
            train_months=12,
            test_months=3,
            horizon=48,
            tp_mult=3.0,
            sl_mult=1.0,
            quality_gate_cfg=V5QualityGateConfig(),
            tpd_ctrl_cfg=V5TPDControllerConfig(min_threshold_floor=0.001),
            candidate_config=CandidateConfig(enabled=False),
            risk_controls=None,
            ema200_regime_gate=True,
            adx_gate=True,
            adx_min=14.0,
            adx_exception_top_pct=10.0,
            slippage_base_bps=1.0,
            min_threshold=0.001,
            max_threshold=0.02,
            wf_threshold_ema=True,
            wf_threshold_ema_alpha=0.5,
            min_trades=60,
            warmup_skip_bars=48,
            daily_loss_cap=-4.0,
            weekly_loss_cap=-12.0,
            corr_block=True,
            corr_thresh=0.80,
            per_symbol_scaler=True,
            per_symbol_threshold=True,
            per_symbol_r_kill=-8.0,
            symbol_embed_dim=8,
            edge_first=True,
            edge_min=0.025,
            edge_pct_floor=65,
            edge_topn_per_day=25,
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
            warm_start=False,
            finetune_months=0,
            finetune_epochs=0,
            finetune_lr_mult=0.1,
            temp_scale=False,
            promote_metric="expectancy",
            stage_a_epochs=0,
            cooldown=0,
            max_folds=1,
            candidate_logger=candidate_logger,
            dual_specialist=True,
            min_mu_r_long=0.0,
            long_disagree_mult=0.3,
            specialist_align_weight=0.5,
            short_oversample=True,
            short_min_fraction=0.45,
            per_side_threshold=True,
            min_p_side=0.45,
            min_p_short=0.0,
            quality_blend_enabled=bool(use_quality_blend),
            quality_blend_weights=quality_blend_weights if use_quality_blend else None,
            quality_blend_min_quantile=float(np.clip(quality_blend_min_q, 0.0, 0.99)),
            quality_blend_regimes=quality_blend_regimes if use_quality_blend else None,
            quality_blend_block_outside_regimes=False,
            quality_blend_min_bars=300,
        )

    fwd = checkpoints / "v5_forward_report.json"
    wf = checkpoints / "v5_walkforward_report.json"
    tagged_fwd = checkpoints / f"v5_forward_report_{tag}.json"
    tagged_wf = checkpoints / f"v5_walkforward_report_{tag}.json"
    if fwd.exists():
        shutil.copy2(fwd, tagged_fwd)
    if wf.exists():
        shutil.copy2(wf, tagged_wf)

    fold_report = json.loads(tagged_fwd.read_text()) if tagged_fwd.exists() else {}
    return {
        "tag": tag,
        "symbols": symbols,
        "use_quality_blend": bool(use_quality_blend),
        "total_trades": int(fold_report.get("total_trades", 0)),
        "total_r": float(fold_report.get("total_r", 0.0)),
        "win_rate": float(fold_report.get("win_rate", 0.0)),
        "expectancy_r": float(fold_report.get("expectancy_r", 0.0)),
        "quality_blend_blocked": int(fold_report.get("quality_blend_blocked", 0)),
        "quality_blend_pass_rate": float(fold_report.get("quality_blend_pass_rate", 0.0)),
        "quality_blend_cutoff": float(fold_report.get("quality_blend_cutoff", 0.0)),
        "gate_blocks": ((fold_report.get("directional_balance") or {}).get("gate_blocks") or {}),
        "aggregate": (result or {}).get("aggregate", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline vs quality-blend V5 comparison.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT", "XRPUSDT"],
    )
    parser.add_argument("--quality-blend-min-q", type=float, default=0.80)
    parser.add_argument(
        "--quality-blend-regimes",
        type=str,
        default="trending_up,trending_down",
        help="Comma-separated allowed regimes for blend pre-gate.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="/workspace/gpu_trainer/checkpoints/v5_quality_blend_compare_summary.json",
    )
    args = parser.parse_args()

    regimes = [s.strip() for s in str(args.quality_blend_regimes).split(",") if s.strip()]

    baseline = _run_single(
        tag="allsym_3m_baseline_qblend_cmp",
        symbols=list(args.symbols),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_quality_blend=False,
        quality_blend_min_q=args.quality_blend_min_q,
        quality_blend_regimes=regimes,
    )
    blend = _run_single(
        tag="allsym_3m_qblend_cmp",
        symbols=list(args.symbols),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_quality_blend=True,
        quality_blend_min_q=args.quality_blend_min_q,
        quality_blend_regimes=regimes,
    )

    delta = {
        "total_trades": int(blend["total_trades"] - baseline["total_trades"]),
        "total_r": float(blend["total_r"] - baseline["total_r"]),
        "win_rate": float(blend["win_rate"] - baseline["win_rate"]),
        "expectancy_r": float(blend["expectancy_r"] - baseline["expectancy_r"]),
    }
    summary = {"baseline": baseline, "quality_blend": blend, "delta": delta}

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Saved summary: {out_path}")


if __name__ == "__main__":
    main()
