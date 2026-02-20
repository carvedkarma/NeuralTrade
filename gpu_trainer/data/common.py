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
    """Generate side-conditional trade outcomes using v5-consistent barrier logic.

    Uses the same ATR computation as v5_target_generator for consistency.
    Simulates BOTH LONG and SHORT trades independently per bar.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        horizon: forward-looking window in bars
        tp_mult: ATR multiplier for take-profit
        sl_mult: ATR multiplier for stop-loss
        atr_period: ATR lookback period

    Returns:
        dict with:
            r_long: (N,) float32 array -- realized R if LONG at this bar
            r_short: (N,) float32 array -- realized R if SHORT at this bar
            out_long: (N,) object array -- outcome type if LONG
            out_short: (N,) object array -- outcome type if SHORT
            realized_r: (N,) float32 -- DEPRECATED best-side oracle R (DO NOT USE FOR EVAL)
            outcome: (N,) object -- DEPRECATED best-side oracle outcome (DO NOT USE FOR EVAL)
    """
    n = len(df)
    closes = df['close'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)

    atr = compute_atr(df, atr_period)

    r_long = np.full(n, np.nan, dtype=np.float64)
    r_short = np.full(n, np.nan, dtype=np.float64)
    out_long = np.full(n, "NO_CANDIDATE", dtype=object)
    out_short = np.full(n, "NO_CANDIDATE", dtype=object)
    realized_r_best = np.full(n, np.nan, dtype=np.float64)
    outcomes_best = np.full(n, "NO_CANDIDATE", dtype=object)

    for i in range(n - horizon):
        if atr[i] <= 0 or closes[i] <= 0:
            continue

        entry = closes[i]
        tp_dist = atr[i] * tp_mult
        sl_dist = atr[i] * sl_mult

        long_r_val, long_out_val = _simulate_trade(
            highs, lows, closes, i, horizon, n,
            entry, tp_dist, sl_dist, tp_mult, sl_mult, atr[i], side=1
        )
        short_r_val, short_out_val = _simulate_trade(
            highs, lows, closes, i, horizon, n,
            entry, tp_dist, sl_dist, tp_mult, sl_mult, atr[i], side=-1
        )

        r_long[i] = long_r_val
        r_short[i] = short_r_val
        out_long[i] = long_out_val
        out_short[i] = short_out_val

        if long_r_val >= short_r_val:
            realized_r_best[i] = long_r_val
            outcomes_best[i] = long_out_val
        else:
            realized_r_best[i] = short_r_val
            outcomes_best[i] = short_out_val

    return {
        'r_long': r_long.astype(np.float32),
        'r_short': r_short.astype(np.float32),
        'out_long': out_long,
        'out_short': out_short,
        'realized_r': realized_r_best.astype(np.float32),
        'outcome': outcomes_best,
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
    return expiry_r, "EXP_LOSS"


def _simulate_trade_trailing(
    highs, lows, closes, i, horizon, n,
    entry, tp_dist, sl_dist, tp_mult, sl_mult, atr_val, side=1,
    trail_activation=1.0, trail_distance=1.0, allow_runner=False,
):
    """Simulate a trade with trailing stop-loss.

    The trailing stop works in phases:
      Phase 1 (initial): Fixed SL at entry -/+ sl_dist (same as normal).
      Phase 2 (activated): Once price moves trail_activation * ATR in favor,
                           SL moves to breakeven, then trails at trail_distance * ATR
                           behind the best price seen.
      Phase 3 (runner, optional): If allow_runner=True, after TP level is reached,
                                  the trade stays open with a tight trail (0.5 * trail_distance * ATR)
                                  to capture extended moves.

    Args:
        trail_activation: ATR multiples of favorable move before trailing activates
        trail_distance: ATR multiples behind best price for trailing stop
        allow_runner: if True, don't exit at TP, let it run with tighter trail

    Returns:
        (realized_r, outcome_str)
    """
    if side == 1:
        tp_price = entry + tp_dist
        initial_sl = entry - sl_dist
    else:
        tp_price = entry - tp_dist
        initial_sl = entry + sl_dist

    activation_dist = atr_val * trail_activation
    trail_dist_abs = atr_val * trail_distance

    trailing_active = False
    best_price = entry
    current_sl = initial_sl

    for j in range(i + 1, min(i + 1 + horizon, n)):
        bar_high = highs[j]
        bar_low = lows[j]
        bar_close = closes[j]

        if side == 1:
            if bar_high > best_price:
                best_price = bar_high

            if not allow_runner and bar_high >= tp_price:
                return tp_mult / sl_mult, "TP"

            if allow_runner and bar_high >= tp_price:
                trailing_active = True
                trail_dist_abs = atr_val * trail_distance * 0.5
                new_sl = best_price - trail_dist_abs
                current_sl = max(current_sl, new_sl)

            if not trailing_active:
                if best_price - entry >= activation_dist:
                    trailing_active = True
                    new_sl = max(entry, best_price - trail_dist_abs)
                    current_sl = max(current_sl, new_sl)
            else:
                new_sl = best_price - trail_dist_abs
                current_sl = max(current_sl, new_sl)

            if bar_low <= current_sl:
                realized_r = (current_sl - entry) / (atr_val * sl_mult)
                if trailing_active and realized_r > 0:
                    return realized_r, "TRAIL_WIN"
                elif trailing_active:
                    return max(realized_r, 0.0), "TRAIL_BE"
                else:
                    return -1.0, "SL"

        else:
            if bar_low < best_price:
                best_price = bar_low

            if not allow_runner and bar_low <= tp_price:
                return tp_mult / sl_mult, "TP"

            if allow_runner and bar_low <= tp_price:
                trailing_active = True
                trail_dist_abs = atr_val * trail_distance * 0.5
                new_sl = best_price + trail_dist_abs
                current_sl = min(current_sl, new_sl)

            if not trailing_active:
                if entry - best_price >= activation_dist:
                    trailing_active = True
                    new_sl = min(entry, best_price + trail_dist_abs)
                    current_sl = min(current_sl, new_sl)
            else:
                new_sl = best_price + trail_dist_abs
                current_sl = min(current_sl, new_sl)

            if bar_high >= current_sl:
                realized_r = (entry - current_sl) / (atr_val * sl_mult)
                if trailing_active and realized_r > 0:
                    return realized_r, "TRAIL_WIN"
                elif trailing_active:
                    return max(realized_r, 0.0), "TRAIL_BE"
                else:
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


def generate_v5_sweep_outcomes_trailing(
    df: pd.DataFrame,
    horizon: int = 16,
    tp_mult: float = 2.0,
    sl_mult: float = 1.5,
    atr_period: int = 14,
    trail_activation: float = 1.0,
    trail_distance: float = 1.0,
    allow_runner: bool = False,
) -> dict:
    """Generate side-conditional trade outcomes using trailing stop logic.

    Same interface as generate_v5_sweep_outcomes but uses _simulate_trade_trailing
    for dynamic exits instead of fixed TP/SL barriers.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        horizon: forward-looking window in bars
        tp_mult: ATR multiplier for take-profit level
        sl_mult: ATR multiplier for initial stop-loss
        atr_period: ATR lookback period
        trail_activation: ATR multiples of favorable move before trailing activates
        trail_distance: ATR multiples behind best price for trailing stop
        allow_runner: if True, don't exit at TP, let it run with tighter trail

    Returns:
        dict with r_long, r_short, out_long, out_short, realized_r, outcome
    """
    n = len(df)
    closes = df['close'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)

    atr = compute_atr(df, atr_period)

    r_long = np.full(n, np.nan, dtype=np.float64)
    r_short = np.full(n, np.nan, dtype=np.float64)
    out_long = np.full(n, "NO_CANDIDATE", dtype=object)
    out_short = np.full(n, "NO_CANDIDATE", dtype=object)
    realized_r_best = np.full(n, np.nan, dtype=np.float64)
    outcomes_best = np.full(n, "NO_CANDIDATE", dtype=object)

    for i in range(n - horizon):
        if atr[i] <= 0 or closes[i] <= 0:
            continue

        entry = closes[i]
        tp_dist = atr[i] * tp_mult
        sl_dist = atr[i] * sl_mult

        long_r_val, long_out_val = _simulate_trade_trailing(
            highs, lows, closes, i, horizon, n,
            entry, tp_dist, sl_dist, tp_mult, sl_mult, atr[i], side=1,
            trail_activation=trail_activation, trail_distance=trail_distance,
            allow_runner=allow_runner,
        )
        short_r_val, short_out_val = _simulate_trade_trailing(
            highs, lows, closes, i, horizon, n,
            entry, tp_dist, sl_dist, tp_mult, sl_mult, atr[i], side=-1,
            trail_activation=trail_activation, trail_distance=trail_distance,
            allow_runner=allow_runner,
        )

        r_long[i] = long_r_val
        r_short[i] = short_r_val
        out_long[i] = long_out_val
        out_short[i] = short_out_val

        if long_r_val >= short_r_val:
            realized_r_best[i] = long_r_val
            outcomes_best[i] = long_out_val
        else:
            realized_r_best[i] = short_r_val
            outcomes_best[i] = short_out_val

    return {
        'r_long': r_long.astype(np.float32),
        'r_short': r_short.astype(np.float32),
        'out_long': out_long,
        'out_short': out_short,
        'realized_r': realized_r_best.astype(np.float32),
        'outcome': outcomes_best,
    }
