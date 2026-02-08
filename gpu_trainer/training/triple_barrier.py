"""
Triple Barrier Simulator - Single Source of Truth
==================================================
Used by BOTH labeling and sweep evaluation to ensure parity.

compute_atr_14(df) -> ATR(14) using true range (OHLC)
triple_barrier_outcome_for_index(df, i, side, atr_i, ...) -> (outcome, r)
triple_barrier_batch(df, indices, sides, ...) -> (outcomes[], r_values[])
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_atr_14(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Compute ATR using true range from OHLC data.
    
    Returns numpy array aligned to df rows.
    Matches the ATR method used in labeling (RegressionTargetGenerator._compute_atr).
    """
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    n = len(df)
    
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    
    atr = np.empty(n, dtype=np.float64)
    atr[:] = np.nan
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    for i in range(min(period - 1, n)):
        if np.isnan(atr[i]):
            atr[i] = np.mean(tr[: i + 1]) if i > 0 else tr[0]
    
    return atr


def triple_barrier_outcome_for_index(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    i: int,
    side: int,
    atr_i: float,
    tp_mult: float = 2.0,
    sl_mult: float = 1.5,
    horizon: int = 24,
    r_min_expiry: float = 0.5,
):
    """Simulate a single triple-barrier trade at index i.
    
    Args:
        highs/lows/closes: Price arrays from OHLC
        i: Entry bar index
        side: +1 (LONG) or -1 (SHORT)
        atr_i: ATR value at bar i
        tp_mult/sl_mult: ATR multipliers for TP/SL
        horizon: Max bars to hold
        r_min_expiry: Min R-multiple at expiry to count as win
    
    Returns:
        (outcome, realized_r) where outcome is one of:
        "TP", "SL", "EXP_WIN", "EXP_LOSS"
    """
    n = len(closes)
    entry = closes[i]
    
    a = atr_i
    if np.isnan(a) or a <= 0:
        a = entry * 0.005
    
    tp_dist = tp_mult * a
    sl_dist = sl_mult * a
    
    if side > 0:
        tp_price = entry + tp_dist
        sl_price = entry - sl_dist
    else:
        tp_price = entry - tp_dist
        sl_price = entry + sl_dist
    
    for j in range(1, horizon + 1):
        idx = i + j
        if idx >= n:
            break
        
        if side > 0:
            tp_hit = highs[idx] >= tp_price
            sl_hit = lows[idx] <= sl_price
        else:
            tp_hit = lows[idx] <= tp_price
            sl_hit = highs[idx] >= sl_price
        
        if tp_hit and sl_hit:
            if side > 0:
                tp_excursion = highs[idx] - entry
                sl_excursion = entry - lows[idx]
            else:
                tp_excursion = entry - lows[idx]
                sl_excursion = highs[idx] - entry
            
            if tp_excursion >= sl_excursion:
                return ("TP", tp_mult / sl_mult)
            else:
                return ("SL", -1.0)
        elif tp_hit:
            return ("TP", tp_mult / sl_mult)
        elif sl_hit:
            return ("SL", -1.0)
    
    end_idx = min(i + horizon, n - 1)
    exit_price = closes[end_idx]
    if side > 0:
        pnl = exit_price - entry
    else:
        pnl = entry - exit_price
    
    r_at_expiry = pnl / sl_dist if sl_dist > 0 else 0.0
    
    if r_at_expiry >= r_min_expiry:
        return ("EXP_WIN", r_at_expiry)
    else:
        return ("EXP_LOSS", r_at_expiry)


def triple_barrier_batch(
    df: pd.DataFrame,
    indices: np.ndarray,
    sides: np.ndarray,
    tp_mult: float = 2.0,
    sl_mult: float = 1.5,
    horizon: int = 24,
    r_min_expiry: float = 0.5,
    atr: np.ndarray = None,
):
    """Run triple-barrier simulation for a batch of trade entries.
    
    Args:
        df: OHLCV DataFrame
        indices: Array of entry bar indices
        sides: Array of trade directions (+1 LONG, -1 SHORT)
        tp_mult/sl_mult: ATR multipliers
        horizon: Max bars to hold
        r_min_expiry: Min R at expiry for win
        atr: Pre-computed ATR array (if None, computed internally)
    
    Returns:
        (outcomes, r_values) - arrays of same length as indices
    """
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    
    if atr is None:
        atr = compute_atr_14(df)
    
    n_trades = len(indices)
    outcomes = np.empty(n_trades, dtype=object)
    r_values = np.empty(n_trades, dtype=np.float64)
    
    for k in range(n_trades):
        i = int(indices[k])
        side = int(sides[k])
        atr_i = float(atr[i])
        
        outcome, r = triple_barrier_outcome_for_index(
            highs, lows, closes, i, side, atr_i,
            tp_mult, sl_mult, horizon, r_min_expiry,
        )
        outcomes[k] = outcome
        r_values[k] = r
    
    return outcomes, r_values
