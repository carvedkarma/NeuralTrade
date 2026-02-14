"""
V5.0.1 Target Generator: All Continuous Targets in R-Units

Computes for each bar (given a horizon):
- ret_R: close-to-close return at horizon, in R-units (normalized by ATR)
- mfe_R: max favorable excursion within horizon (in R-units)
- mae_R: max adverse excursion within horizon (in R-units)
- vol_h: realized volatility within horizon (std of bar-to-bar returns)

All targets use ONLY future bars within [i+1 .. i+horizon] -- no leakage.
All continuous targets (ret_R, mfe_R, mae_R) are in R-units for unit consistency.
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
    hold_target: float = 0.30
    mfe_min: float = 0.05


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
    hold_target: float = 0.30,
    mfe_min_r: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Build v5 continuous targets from OHLCV data -- ALL in R-units.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
        horizon: forward-looking window in bars
        atr_period: ATR lookback for R-unit normalization
        hold_target: target fraction of HOLD labels (adaptive deadzone)
        mfe_min_r: minimum MFE in R-units required to classify as non-HOLD

    Returns:
        Dict with keys: ret_R, mfe_R, mae_R, vol_h, action_label, valid_mask, atr
        ret_R, mfe_R, mae_R are ALL in R-units (price_change / ATR).
    """
    n = len(df)
    closes = df['close'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)

    atr = compute_atr(df, atr_period)
    eps = 1e-10

    ret_R = np.full(n, np.nan, dtype=np.float64)
    mfe_R = np.full(n, np.nan, dtype=np.float64)
    mae_R = np.full(n, np.nan, dtype=np.float64)
    vol_h = np.full(n, np.nan, dtype=np.float64)

    for i in range(n - horizon):
        entry_price = closes[i]
        if entry_price <= 0 or atr[i] <= 0:
            continue

        future_closes = closes[i + 1: i + 1 + horizon]
        future_highs = highs[i + 1: i + 1 + horizon]
        future_lows = lows[i + 1: i + 1 + horizon]

        exit_price = future_closes[-1]
        ret_R[i] = (exit_price - entry_price) / (atr[i] + eps)

        max_high = np.max(future_highs)
        min_low = np.min(future_lows)
        long_mfe = (max_high - entry_price) / (atr[i] + eps)
        long_mae = (entry_price - min_low) / (atr[i] + eps)
        short_mfe = (entry_price - min_low) / (atr[i] + eps)
        short_mae = (max_high - entry_price) / (atr[i] + eps)

        if ret_R[i] >= 0:
            mfe_R[i] = long_mfe
            mae_R[i] = long_mae
        else:
            mfe_R[i] = short_mfe
            mae_R[i] = short_mae

        bar_returns = np.diff(np.log(np.maximum(future_closes, eps)))
        if len(bar_returns) > 1:
            vol_h[i] = np.std(bar_returns, ddof=1)
        else:
            vol_h[i] = 0.0

    valid_mask = (np.isfinite(ret_R) & np.isfinite(mfe_R) & np.isfinite(mae_R)
                  & np.isfinite(vol_h) & (atr > 0))

    abs_ret_valid = np.abs(ret_R[valid_mask])
    if len(abs_ret_valid) > 0:
        deadzone_R = float(np.percentile(abs_ret_valid, hold_target * 100))
    else:
        deadzone_R = 0.1
    logger.info(f"[V5_TARGETS] Adaptive deadzone: hold_target={hold_target:.0%} -> deadzone_R={deadzone_R:.4f}")

    action_label = np.full(n, 0, dtype=np.int64)
    for i in range(n):
        if not valid_mask[i]:
            continue
        if np.abs(ret_R[i]) < deadzone_R or mfe_R[i] < mfe_min_r:
            action_label[i] = 0
        elif ret_R[i] > 0:
            action_label[i] = 1
        else:
            action_label[i] = 2

    n_valid = int(np.sum(valid_mask))
    n_hold = int(np.sum(action_label[valid_mask] == 0))
    n_long = int(np.sum(action_label[valid_mask] == 1))
    n_short = int(np.sum(action_label[valid_mask] == 2))

    logger.info(f"[V5_TARGETS] horizon={horizon} valid={n_valid}/{n} "
                f"action: HOLD={n_hold} ({n_hold/max(n_valid,1):.1%}) "
                f"LONG={n_long} ({n_long/max(n_valid,1):.1%}) "
                f"SHORT={n_short} ({n_short/max(n_valid,1):.1%})")

    if n_valid > 0:
        ret_valid = ret_R[valid_mask]
        mfe_valid = mfe_R[valid_mask]
        mae_valid = mae_R[valid_mask]
        vol_valid = vol_h[valid_mask]
        logger.info(f"[V5_TARGETS] ret_R: mean={np.mean(ret_valid):.4f} std={np.std(ret_valid):.4f} "
                     f"p5={np.percentile(ret_valid,5):.4f} p95={np.percentile(ret_valid,95):.4f}")
        logger.info(f"[V5_TARGETS] mfe_R: mean={np.mean(mfe_valid):.3f} mae_R: mean={np.mean(mae_valid):.3f} "
                     f"vol_h: mean={np.mean(vol_valid):.6f}")

    assert np.all(np.isfinite(ret_R[valid_mask])), "ret_R contains NaN/Inf in valid region"
    assert np.all(np.isfinite(mfe_R[valid_mask])), "mfe_R contains NaN/Inf in valid region"
    assert np.all(np.isfinite(mae_R[valid_mask])), "mae_R contains NaN/Inf in valid region"
    assert np.all(np.isfinite(vol_h[valid_mask])), "vol_h contains NaN/Inf in valid region"

    return {
        'ret_R': ret_R.astype(np.float32),
        'mfe_R': mfe_R.astype(np.float32),
        'mae_R': mae_R.astype(np.float32),
        'vol_h': vol_h.astype(np.float32),
        'action_label': action_label,
        'valid_mask': valid_mask,
        'atr': atr.astype(np.float32),
        'deadzone_R': deadzone_R,
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
