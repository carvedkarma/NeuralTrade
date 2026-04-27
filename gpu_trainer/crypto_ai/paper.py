from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .backtest import _long_short_weights, _softmax
from .config import SystemConfig
from .data import DataBundle
from .inference import load_model, normalize_features, predict_range


@dataclass(slots=True)
class PaperTradeReport:
    report_path: Path
    n_steps: int
    cumulative_return: float
    max_drawdown: float
    avg_turnover: float


def paper_trade_replay(
    config: SystemConfig,
    bundle: DataBundle,
    checkpoint_path: Path,
    report_name: str = "paper_replay",
    steps: int = 240,
) -> PaperTradeReport:
    """
    Replay the latest section of data as if running online paper trading.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and config.device != "cpu" else "cpu")
    loaded = load_model(checkpoint_path=checkpoint_path, device=device)
    features_norm = normalize_features(
        bundle.features,
        mean=loaded.normalizer_mean,
        std=loaded.normalizer_std,
    )

    start_idx = max(loaded.val_end_idx, bundle.features.shape[0] - int(max(steps, 1)) - 1)
    end_idx = bundle.features.shape[0] - 1
    idx, logits, pred_returns = predict_range(
        loaded=loaded,
        features_norm=features_norm,
        start_idx=start_idx,
        end_idx=end_idx,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    probs = _softmax(logits, axis=-1)

    costs_per_turn = (config.trading_fee_bps + config.slippage_bps) / 10_000.0
    prev_weights = np.zeros(len(bundle.symbols), dtype=np.float64)
    step_returns = []
    turnover = []
    positions = []

    for i, t in enumerate(idx):
        w = _long_short_weights(
            probs=probs[i],
            pred_ret=pred_returns[i],
            min_signal_strength=config.min_signal_strength,
            top_k=config.risk_top_k,
            max_symbol_weight=config.risk_max_symbol_weight,
            max_gross=config.risk_max_gross_exposure,
        )
        realized = bundle.one_step_returns[t]
        turn = np.abs(w - prev_weights).sum()
        cost = costs_per_turn * turn
        pnl = float(np.dot(w, realized) - cost)
        step_returns.append(pnl)
        turnover.append(float(turn))
        prev_weights = w
        positions.append(
            {
                "timestamp": str(bundle.timestamps[t]),
                "weights": {s: float(v) for s, v in zip(bundle.symbols, w)},
                "pred_returns": {s: float(v) for s, v in zip(bundle.symbols, pred_returns[i])},
            }
        )

    step_returns_arr = np.asarray(step_returns, dtype=np.float64)
    equity = np.cumprod(1.0 + step_returns_arr) if len(step_returns_arr) else np.array([1.0], dtype=np.float64)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity / running_max) - 1.0

    config.ensure_directories()
    report_path = config.reports_dir / f"{report_name}.json"
    payload = {
        "checkpoint_path": str(checkpoint_path),
        "n_steps": int(len(step_returns_arr)),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "cumulative_return": float(equity[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "avg_turnover": float(np.mean(turnover)) if turnover else 0.0,
        "positions_tail": positions[-20:],
    }
    report_path.write_text(json.dumps(payload, indent=2))

    return PaperTradeReport(
        report_path=report_path,
        n_steps=int(len(step_returns_arr)),
        cumulative_return=float(equity[-1] - 1.0),
        max_drawdown=float(drawdown.min()),
        avg_turnover=float(np.mean(turnover)) if turnover else 0.0,
    )
