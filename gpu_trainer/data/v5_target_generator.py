"""
V5 Target Generator: Continuous, Execution-Aware Targets

Computes for each bar (given a horizon):
- ret_h: close-to-close log-return at horizon
- mfe_h: max favorable excursion within horizon (in R-units based on ATR)
- mae_h: max adverse excursion within horizon (in R-units based on ATR)
- vol_h: realized volatility within horizon (std of bar-to-bar returns)

All targets use ONLY future bars within [i+1 .. i+horizon] -- no leakage.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class V5TargetConfig:
    horizon: int = 16
    atr_period: int = 14
    deadzone: float = 0.0005
    mfe_min: float = 0.2


def compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    n = len(df)

    tr = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )

    atr = np.zeros(n, dtype=np.float64)
    if n > period:
        atr[period] = np.mean(tr[1:period + 1])
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


def build_v5_targets(
    df: pd.DataFrame,
    horizon: int = 16,
    atr_period: int = 14,
    deadzone: float = 0.0005,
    mfe_min_r: float = 0.2,
) -> Dict[str, np.ndarray]:
    """Build v5 continuous targets from OHLCV data.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
        horizon: forward-looking window in bars
        atr_period: ATR lookback for R-unit normalization
        deadzone: minimum |ret_h| to classify as directional (for action labels)
        mfe_min_r: minimum MFE in R-units required to classify as non-HOLD

    Returns:
        Dict with keys: ret_h, mfe_h, mae_h, vol_h, action_label, valid_mask, atr
        Each is a numpy array of length len(df).
        valid_mask is True where targets are computable (not tail bars).
    """
    n = len(df)
    closes = df['close'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)

    atr = compute_atr(df, atr_period)

    ret_h = np.full(n, np.nan, dtype=np.float64)
    mfe_h = np.full(n, np.nan, dtype=np.float64)
    mae_h = np.full(n, np.nan, dtype=np.float64)
    vol_h = np.full(n, np.nan, dtype=np.float64)
    action_label = np.full(n, 0, dtype=np.int64)

    for i in range(n - horizon):
        entry_price = closes[i]
        if entry_price <= 0 or atr[i] <= 0:
            continue

        future_closes = closes[i + 1: i + 1 + horizon]
        future_highs = highs[i + 1: i + 1 + horizon]
        future_lows = lows[i + 1: i + 1 + horizon]

        exit_price = future_closes[-1]
        ret = np.log(exit_price / entry_price)
        ret_h[i] = ret

        max_high = np.max(future_highs)
        min_low = np.min(future_lows)
        long_mfe = (max_high - entry_price) / atr[i]
        long_mae = (entry_price - min_low) / atr[i]
        short_mfe = (entry_price - min_low) / atr[i]
        short_mae = (max_high - entry_price) / atr[i]

        if ret >= 0:
            mfe_h[i] = long_mfe
            mae_h[i] = long_mae
        else:
            mfe_h[i] = short_mfe
            mae_h[i] = short_mae

        bar_returns = np.diff(np.log(future_closes))
        if len(bar_returns) > 1:
            vol_h[i] = np.std(bar_returns, ddof=1)
        else:
            vol_h[i] = 0.0

        if abs(ret) < deadzone or mfe_h[i] < mfe_min_r:
            action_label[i] = 0
        elif ret > 0:
            action_label[i] = 1
        else:
            action_label[i] = 2

    valid_mask = (np.isfinite(ret_h) & np.isfinite(mfe_h) & np.isfinite(mae_h) 
                  & np.isfinite(vol_h) & (atr > 0))

    n_valid = int(np.sum(valid_mask))
    n_hold = int(np.sum(action_label[valid_mask] == 0))
    n_long = int(np.sum(action_label[valid_mask] == 1))
    n_short = int(np.sum(action_label[valid_mask] == 2))

    logger.info(f"[V5_TARGETS] horizon={horizon} valid={n_valid}/{n} "
                f"action: HOLD={n_hold} ({n_hold/max(n_valid,1):.1%}) "
                f"LONG={n_long} ({n_long/max(n_valid,1):.1%}) "
                f"SHORT={n_short} ({n_short/max(n_valid,1):.1%})")

    if n_valid > 0:
        ret_valid = ret_h[valid_mask]
        mfe_valid = mfe_h[valid_mask]
        mae_valid = mae_h[valid_mask]
        vol_valid = vol_h[valid_mask]
        logger.info(f"[V5_TARGETS] ret_h: mean={np.mean(ret_valid):.6f} std={np.std(ret_valid):.6f} "
                     f"p5={np.percentile(ret_valid,5):.6f} p95={np.percentile(ret_valid,95):.6f}")
        logger.info(f"[V5_TARGETS] mfe_h: mean={np.mean(mfe_valid):.3f} mae_h: mean={np.mean(mae_valid):.3f} "
                     f"vol_h: mean={np.mean(vol_valid):.6f}")

    assert np.all(np.isfinite(ret_h[valid_mask])), "ret_h contains NaN/Inf in valid region"
    assert np.all(np.isfinite(mfe_h[valid_mask])), "mfe_h contains NaN/Inf in valid region"
    assert np.all(np.isfinite(mae_h[valid_mask])), "mae_h contains NaN/Inf in valid region"
    assert np.all(np.isfinite(vol_h[valid_mask])), "vol_h contains NaN/Inf in valid region"

    return {
        'ret_h': ret_h.astype(np.float32),
        'mfe_h': mfe_h.astype(np.float32),
        'mae_h': mae_h.astype(np.float32),
        'vol_h': vol_h.astype(np.float32),
        'action_label': action_label,
        'valid_mask': valid_mask,
        'atr': atr.astype(np.float32),
    }


def build_barrier_preset_labels(
    df: pd.DataFrame,
    presets: list,
    horizon: int = 16,
    atr_period: int = 14,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build soft barrier selection labels from realized outcomes.

    For each bar, compute realized R under each preset, then produce:
    - oracle_idx: argmax preset (best realized R) -- research only
    - soft_target: softmax(R_preset / temperature) -- for learnable mode

    Returns:
        oracle_idx: (N,) int64 array of best preset index
        soft_target: (N, n_presets) float32 array of soft probabilities
    """
    n = len(df)
    closes = df['close'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    atr = compute_atr(df, atr_period)

    n_presets = len(presets)
    realized_r = np.full((n, n_presets), np.nan, dtype=np.float64)

    for pi, preset in enumerate(presets):
        tp_mult = preset['tp_mult']
        sl_mult = preset['sl_mult']

        for i in range(n - horizon):
            if atr[i] <= 0 or closes[i] <= 0:
                continue

            entry = closes[i]
            tp_dist = atr[i] * tp_mult
            sl_dist = atr[i] * sl_mult

            long_tp = entry + tp_dist
            long_sl = entry - sl_dist
            short_tp = entry - tp_dist
            short_sl = entry + sl_dist

            long_r = np.nan
            short_r = np.nan

            for j in range(i + 1, min(i + 1 + horizon, n)):
                if highs[j] >= long_tp:
                    long_r = tp_mult / sl_mult
                    break
                if lows[j] <= long_sl:
                    long_r = -1.0
                    break
            if np.isnan(long_r):
                long_r = (closes[min(i + horizon, n - 1)] - entry) / (atr[i] * sl_mult)

            for j in range(i + 1, min(i + 1 + horizon, n)):
                if lows[j] <= short_tp:
                    short_r = tp_mult / sl_mult
                    break
                if highs[j] >= short_sl:
                    short_r = -1.0
                    break
            if np.isnan(short_r):
                short_r = (entry - closes[min(i + horizon, n - 1)]) / (atr[i] * sl_mult)

            realized_r[i, pi] = max(long_r, short_r)

    realized_r_clean = np.nan_to_num(realized_r, nan=-999.0)
    oracle_idx = np.argmax(realized_r_clean, axis=1).astype(np.int64)

    shifted = realized_r_clean / max(temperature, 1e-6)
    shifted = shifted - np.max(shifted, axis=1, keepdims=True)
    exp_r = np.exp(shifted)
    soft_target = exp_r / (np.sum(exp_r, axis=1, keepdims=True) + 1e-8)
    soft_target = soft_target.astype(np.float32)

    logger.info(f"[V5_BARRIER_LABELS] {n_presets} presets, oracle distribution: "
                + ", ".join(f"{presets[p]['label']}={np.mean(oracle_idx==p):.1%}" for p in range(n_presets)))

    return oracle_idx, soft_target
