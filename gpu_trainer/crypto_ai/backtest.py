from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import SystemConfig
from .data import DataBundle
from .inference import load_model, normalize_features, predict_range


@dataclass(slots=True)
class BacktestReport:
    checkpoint_path: Path
    report_path: Path
    start_idx: int
    end_idx: int
    total_return: float
    annualized_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    turnover: float
    n_steps: int


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - x_max)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _annualization_factor(interval: str) -> float:
    if interval.endswith("m"):
        minutes = int(interval[:-1])
        steps_per_day = (24 * 60) / max(minutes, 1)
    elif interval.endswith("h"):
        hours = int(interval[:-1])
        steps_per_day = 24 / max(hours, 1)
    elif interval.endswith("d"):
        steps_per_day = 1.0
    else:
        # Conservative fallback.
        steps_per_day = 24.0
    return steps_per_day * 365.0


def _long_short_weights(
    probs: np.ndarray,
    pred_ret: np.ndarray,
    *,
    min_signal_strength: float,
    top_k: int,
    max_symbol_weight: float,
    max_gross: float,
) -> np.ndarray:
    # probs shape [S, 3], pred_ret shape [S]
    long_strength = probs[:, 2] - probs[:, 0]
    score = 0.65 * long_strength + 0.35 * np.tanh(pred_ret * 25.0)
    weights = np.zeros_like(score, dtype=np.float64)

    if score.size == 0:
        return weights

    # Pick top candidates for long and short.
    top_k = int(max(1, min(top_k, score.size)))
    long_idx = np.argsort(score)[-top_k:]
    short_idx = np.argsort(score)[:top_k]

    long_raw = np.clip(score[long_idx], a_min=min_signal_strength, a_max=None)
    short_raw = np.clip(-score[short_idx], a_min=min_signal_strength, a_max=None)

    if np.any(long_raw > 0):
        long_alloc = long_raw / (long_raw.sum() + 1e-8)
        weights[long_idx] += long_alloc
    if np.any(short_raw > 0):
        short_alloc = short_raw / (short_raw.sum() + 1e-8)
        weights[short_idx] -= short_alloc

    # Cap single-symbol risk.
    weights = np.clip(weights, -max_symbol_weight, max_symbol_weight)

    gross = np.abs(weights).sum()
    if gross > max_gross and gross > 0:
        weights *= max_gross / gross
    return weights


def compute_target_weights(
    probs: np.ndarray,
    pred_ret: np.ndarray,
    config: SystemConfig,
) -> np.ndarray:
    return _long_short_weights(
        probs=probs,
        pred_ret=pred_ret,
        min_signal_strength=config.min_signal_strength,
        top_k=config.risk_top_k,
        max_symbol_weight=config.risk_max_symbol_weight,
        max_gross=config.risk_max_gross_exposure,
    )


def _compute_stats(step_returns: np.ndarray, interval: str) -> tuple[float, float, float, float, float]:
    if len(step_returns) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    equity = np.cumprod(1.0 + step_returns)
    total_return = float(equity[-1] - 1.0) if len(equity) > 0 else 0.0

    ann = _annualization_factor(interval)
    if len(step_returns) > 0:
        mean_r = float(step_returns.mean())
        std_r = float(step_returns.std() + 1e-12)
        sharpe = (mean_r / std_r) * np.sqrt(ann)
        annualized_return = (1.0 + total_return) ** (ann / max(len(step_returns), 1)) - 1.0
    else:
        sharpe = 0.0
        annualized_return = 0.0

    running_max = np.maximum.accumulate(equity) if len(equity) > 0 else np.array([1.0])
    drawdown = (equity / running_max) - 1.0 if len(equity) > 0 else np.array([0.0])
    max_drawdown = float(drawdown.min())

    win_rate = float((step_returns > 0).mean()) if len(step_returns) > 0 else 0.0
    return total_return, annualized_return, sharpe, max_drawdown, win_rate


def backtest_model(
    config: SystemConfig,
    bundle: DataBundle,
    checkpoint_path: Path,
    report_name: str = "backtest",
) -> BacktestReport:
    device = torch.device("cuda" if torch.cuda.is_available() and config.device != "cpu" else "cpu")
    loaded = load_model(checkpoint_path=checkpoint_path, device=device)
    features_norm = normalize_features(
        bundle.features,
        mean=loaded.normalizer_mean,
        std=loaded.normalizer_std,
    )

    start_idx = loaded.val_end_idx
    end_idx = bundle.features.shape[0] - 1
    if end_idx <= start_idx:
        raise RuntimeError(
            f"Backtest range invalid start={start_idx}, end={end_idx}. "
            "Increase history_days or reduce train/val splits."
        )

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

    step_returns = []
    turnover = []
    prev_weights = np.zeros(len(bundle.symbols), dtype=np.float64)

    costs_per_turn = (config.trading_fee_bps + config.slippage_bps) / 10_000.0

    for i, t in enumerate(idx):
        w = _long_short_weights(
            probs=probs[i],
            pred_ret=pred_returns[i],
            min_signal_strength=config.min_signal_strength,
            top_k=config.risk_top_k,
            max_symbol_weight=config.risk_max_symbol_weight,
            max_gross=config.risk_max_gross_exposure,
        )
        realized = bundle.one_step_returns[t]  # one-step future realized returns [S]
        trade_turnover = np.abs(w - prev_weights).sum()
        cost = costs_per_turn * trade_turnover
        pnl = float(np.dot(w, realized) - cost)
        step_returns.append(pnl)
        turnover.append(float(trade_turnover))
        prev_weights = w

    step_returns_arr = np.asarray(step_returns, dtype=np.float64)
    turnover_arr = np.asarray(turnover, dtype=np.float64)
    total_return, annualized_return, sharpe, max_drawdown, win_rate = _compute_stats(
        step_returns_arr,
        config.interval,
    )

    config.ensure_directories()
    report_path = config.reports_dir / f"{report_name}.json"
    payload = {
        "checkpoint_path": str(checkpoint_path),
        "n_steps": int(len(step_returns_arr)),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "start_timestamp": str(bundle.timestamps[start_idx]),
        "end_timestamp": str(bundle.timestamps[end_idx - 1]),
        "metrics": {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "turnover": float(turnover_arr.mean()) if len(turnover_arr) > 0 else 0.0,
        },
        "risk_config": {
            "max_gross_exposure": config.risk_max_gross_exposure,
            "max_symbol_weight": config.risk_max_symbol_weight,
            "top_k": config.risk_top_k,
            "fees_bps": config.trading_fee_bps,
            "slippage_bps": config.slippage_bps,
            "min_signal_strength": config.min_signal_strength,
        },
    }
    report_path.write_text(json.dumps(payload, indent=2))

    return BacktestReport(
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        start_idx=start_idx,
        end_idx=end_idx,
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        turnover=float(turnover_arr.mean()) if len(turnover_arr) > 0 else 0.0,
        n_steps=int(len(step_returns_arr)),
    )
