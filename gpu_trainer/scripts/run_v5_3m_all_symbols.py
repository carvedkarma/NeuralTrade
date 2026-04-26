#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3-month V5 walk-forward all-symbol diagnostic.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--output-tag", type=str, default="allsym_3m_fold1")
    parser.add_argument("--soft-ema", action="store_true", default=False,
                        help="Use EMA200 soft multiplier (0.5) instead of hard EMA200 gate.")
    parser.add_argument("--relax-risk-caps", action="store_true", default=False,
                        help="Disable daily/weekly caps and per-symbol kill for cadence probe.")
    parser.add_argument("--choppy-side-mode", type=str, default="BOTH", choices=["BOTH", "LONG", "SHORT", "NONE"],
                        help="Override choppy regime side in regime-side-map.")
    parser.add_argument("--long-min-mu-r", type=float, default=0.0,
                        help="Dual specialist LONG hard gate: block LONG when mu_R below this.")
    parser.add_argument("--long-disagree-mult", type=float, default=0.3,
                        help="Dual specialist LONG disagree multiplier (lower = stricter).")
    parser.add_argument("--specialist-align-weight", type=float, default=0.5,
                        help="LONG specialist alignment loss weight.")
    parser.add_argument("--min-p-side", type=float, default=0.45,
                        help="Minimum side confidence for entry gate.")
    parser.add_argument("--min-p-short", type=float, default=0.0,
                        help="Minimum SHORT confidence for short entries.")
    parser.add_argument("--short-min-fraction", type=float, default=0.45,
                        help="SHORT oversample target fraction.")
    parser.add_argument("--edge-topn-per-day", type=int, default=25,
                        help="Per-symbol daily top-N cap for edge-first mode.")
    parser.add_argument("--cooldown", type=int, default=0,
                        help="Cooldown bars between entries.")
    args = parser.parse_args()

    symbols = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
        "LINKUSDT", "LTCUSDT", "NEARUSDT", "PEPEUSDT", "SUIUSDT", "AAVEUSDT", "ARBUSDT", "DOTUSDT",
        "MATICUSDT", "FILUSDT", "APTUSDT", "OPUSDT",
    ]
    tag = str(args.output_tag).strip() or "allsym_3m_fold1"
    candidate_log_path = Path(f"/workspace/gpu_trainer/checkpoints/v5_candidates_{tag}.jsonl")
    candidate_log_path.parent.mkdir(parents=True, exist_ok=True)
    if candidate_log_path.exists():
        candidate_log_path.unlink()
    candidate_log_file = candidate_log_path.open("a", encoding="utf-8")

    def candidate_logger(row: dict) -> None:
        candidate_log_file.write(json.dumps(row, default=_json_default) + "\n")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        quality_cfg = V5QualityGateConfig()
        tpd_cfg = V5TPDControllerConfig(min_threshold_floor=0.001)
        regime_side_map = {
            "trending_up": "LONG",
            "trending_down": "SHORT",
            "choppy": str(args.choppy_side_mode).upper(),
        }

        run_v5_walk_forward(
            data_dir=Path("/workspace/gpu_trainer/data_cache"),
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
            ema200_regime_gate=not bool(args.soft_ema),
            ema200_soft_mult=0.5 if bool(args.soft_ema) else None,
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
            daily_loss_cap=None if bool(args.relax_risk_caps) else -4.0,
            weekly_loss_cap=None if bool(args.relax_risk_caps) else -12.0,
            corr_block=True,
            corr_thresh=0.80,
            per_symbol_scaler=True,
            per_symbol_threshold=True,
            per_symbol_r_kill=None if bool(args.relax_risk_caps) else -8.0,
            symbol_embed_dim=8,
            edge_first=True,
            edge_min=0.025,
            edge_pct_floor=65,
            edge_topn_per_day=int(max(1, args.edge_topn_per_day)),
            regime_side_map=regime_side_map,
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
            cooldown=int(max(0, args.cooldown)),
            replit_url="https://99f68291-4a03-450a-9815-ebee9435cee2-00-2os5ge21n6uho.spock.replit.dev",
            max_folds=1,
            candidate_logger=candidate_logger,
            dual_specialist=True,
            min_mu_r_long=float(args.long_min_mu_r),
            long_disagree_mult=float(max(0.05, args.long_disagree_mult)),
            specialist_align_weight=float(max(0.0, args.specialist_align_weight)),
            short_oversample=True,
            short_min_fraction=float(np.clip(args.short_min_fraction, 0.20, 0.80)),
            per_side_threshold=True,
            min_p_side=float(np.clip(args.min_p_side, 0.0, 0.99)),
            min_p_short=float(np.clip(args.min_p_short, 0.0, 0.99)),
        )
    finally:
        candidate_log_file.close()


if __name__ == "__main__":
    main()
