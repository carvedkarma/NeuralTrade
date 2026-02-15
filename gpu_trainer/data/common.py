"""
Shared utility functions for data processing.

Single-source-of-truth for ATR computation and other common operations
used across v5_target_generator, candidate_generator, triple_barrier, pipeline.
"""

import numpy as np
import pandas as pd
from typing import Optional


def compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Compute Average True Range (ATR) using Wilder's smoothing.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR lookback period (default 14)

    Returns:
        np.ndarray of ATR values, length = len(df).
        First `period` values are 0 (insufficient data).
    """
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    n = len(df)

    tr = np.zeros(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
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


def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Compute ATR as a percentage of close price.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR lookback period

    Returns:
        np.ndarray of ATR/close ratios.
    """
    atr = compute_atr(df, period)
    closes = df['close'].values.astype(np.float64)
    atr_pct = np.where(closes > 0, atr / closes, 0.0)
    return atr_pct


def generate_v5_sweep_outcomes(
    df: pd.DataFrame,
    horizon: int = 16,
    tp_mult: float = 2.0,
    sl_mult: float = 1.5,
    atr_period: int = 14,
) -> dict:
    """Generate trade outcomes using v5-consistent barrier logic.

    Uses the same ATR computation as v5_target_generator for consistency.
    Simulates bidirectional trades and returns the best-side outcome.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        horizon: forward-looking window in bars
        tp_mult: ATR multiplier for take-profit
        sl_mult: ATR multiplier for stop-loss
        atr_period: ATR lookback period

    Returns:
        dict with:
            realized_r: (N,) float32 array of realized R per bar
            outcome: (N,) object array of outcome types (TP, SL, EXP_WIN, EXP_LOSS)
    """
    n = len(df)
    closes = df['close'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)

    atr = compute_atr(df, atr_period)

    realized_r = np.full(n, np.nan, dtype=np.float64)
    outcomes = np.full(n, "NO_CANDIDATE", dtype=object)

    for i in range(n - horizon):
        if atr[i] <= 0 or closes[i] <= 0:
            continue

        entry = closes[i]
        tp_dist = atr[i] * tp_mult
        sl_dist = atr[i] * sl_mult

        long_r, long_outcome = _simulate_trade(
            highs, lows, closes, i, horizon, n,
            entry, tp_dist, sl_dist, tp_mult, sl_mult, atr[i], side=1
        )
        short_r, short_outcome = _simulate_trade(
            highs, lows, closes, i, horizon, n,
            entry, tp_dist, sl_dist, tp_mult, sl_mult, atr[i], side=-1
        )

        if long_r >= short_r:
            realized_r[i] = long_r
            outcomes[i] = long_outcome
        else:
            realized_r[i] = short_r
            outcomes[i] = short_outcome

    return {
        'realized_r': realized_r.astype(np.float32),
        'outcome': outcomes,
    }


def _simulate_trade(
    highs, lows, closes, i, horizon, n,
    entry, tp_dist, sl_dist, tp_mult, sl_mult, atr_val, side=1
):
    """Simulate a single-direction trade through the barrier.

    Args:
        side: 1 for LONG, -1 for SHORT

    Returns:
        (realized_r, outcome_str)
    """
    if side == 1:
        tp_price = entry + tp_dist
        sl_price = entry - sl_dist
    else:
        tp_price = entry - tp_dist
        sl_price = entry + sl_dist

    for j in range(i + 1, min(i + 1 + horizon, n)):
        if side == 1:
            if highs[j] >= tp_price:
                return tp_mult / sl_mult, "TP"
            if lows[j] <= sl_price:
                return -1.0, "SL"
        else:
            if lows[j] <= tp_price:
                return tp_mult / sl_mult, "TP"
            if highs[j] >= sl_price:
                return -1.0, "SL"

    exit_price = closes[min(i + horizon, n - 1)]
    if side == 1:
        expiry_r = (exit_price - entry) / (atr_val * sl_mult)
    else:
        expiry_r = (entry - exit_price) / (atr_val * sl_mult)

    if expiry_r >= 0:
        return expiry_r, "EXP_WIN"
    else:
        return expiry_r, "EXP_LOSS"
