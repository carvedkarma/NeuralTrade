"""
Triple-barrier first-touch labeler (López de Prado, AFML ch. 3).

For every bar i with ATR a_i and close c_i, set
    upper_barrier = c_i + barrier_mult * a_i
    lower_barrier = c_i - barrier_mult * a_i
and look forward bars (i+1 .. i+horizon_bars). The first bar whose
high crosses upper_barrier or whose low crosses lower_barrier
resolves the trade. If neither is touched within the horizon, the
trade times out at horizon_bars.

This module is direction-agnostic: it returns *which* barrier was hit
first and the realized log return at exit. Meta-labeling logic
(primary rule + side-aware win/loss) lives in `meta_label.py`.

Locked Phase-1 defaults (do NOT tune):
    atr_window   = 14
    barrier_mult = 1.5
    horizons     = [16, 32, 96]   # 4h, 8h, 1d on 15m bars
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UPPER = 1
LOWER = -1
TIMEOUT = 0


def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                window: int = 14) -> np.ndarray:
    """Wilder ATR. Returns array same length as input, NaN-filled for warmup."""
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        a = high[i] - low[i]
        b = abs(high[i] - close[i - 1])
        c = abs(low[i] - close[i - 1])
        tr[i] = max(a, b, c)

    atr = np.full(n, np.nan, dtype=np.float64)
    if n <= window:
        return atr
    atr[window] = float(np.mean(tr[1:window + 1]))
    alpha = 1.0 / window
    for i in range(window + 1, n):
        atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
    return atr


def compute_triple_barrier(
    df: pd.DataFrame,
    horizon_bars: int,
    atr_window: int = 14,
    barrier_mult: float = 1.5,
) -> pd.DataFrame:
    """Compute first-touch triple-barrier outcomes for every bar in df.

    df must have columns: timestamp, open, high, low, close.

    Returns DataFrame indexed 0..N-1 with columns:
        timestamp        : entry bar timestamp (ms)
        entry_close      : close at entry bar
        atr              : ATR(atr_window) at entry bar
        upper_price      : barrier price level
        lower_price      : barrier price level
        exit_offset      : bars from entry to exit (1..horizon_bars)
        exit_reason      : +1 upper, -1 lower, 0 timeout
        exit_price       : price at exit (barrier level on touch, close on timeout)
        log_return       : ln(exit_price / entry_close)  (unsigned, side-agnostic)
        valid            : True iff entry has finite ATR and full horizon ahead

    Bars without a complete forward window or without a finite ATR are
    emitted with valid=False and NaN outcome columns.
    """
    if horizon_bars < 1:
        raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")

    n = len(df)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    ts = df["timestamp"].to_numpy()

    atr = _wilder_atr(high, low, close, atr_window)

    exit_offset = np.zeros(n, dtype=np.int32)
    exit_reason = np.zeros(n, dtype=np.int8)
    exit_price = np.full(n, np.nan, dtype=np.float64)
    upper_arr = np.full(n, np.nan, dtype=np.float64)
    lower_arr = np.full(n, np.nan, dtype=np.float64)
    log_ret = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    last_entry = n - horizon_bars - 1
    for i in range(n):
        a_i = atr[i]
        if not np.isfinite(a_i) or i > last_entry:
            continue
        c_i = close[i]
        upper = c_i + barrier_mult * a_i
        lower = c_i - barrier_mult * a_i
        upper_arr[i] = upper
        lower_arr[i] = lower

        end = i + horizon_bars + 1
        hi_win = high[i + 1:end]
        lo_win = low[i + 1:end]

        upper_hits = np.where(hi_win >= upper)[0]
        lower_hits = np.where(lo_win <= lower)[0]
        u_first = upper_hits[0] if upper_hits.size else horizon_bars
        l_first = lower_hits[0] if lower_hits.size else horizon_bars

        if u_first < l_first:
            off = u_first + 1
            reason = UPPER
            exitp = upper
        elif l_first < u_first:
            off = l_first + 1
            reason = LOWER
            exitp = lower
        elif u_first < horizon_bars:
            # tied: both barriers cross in same bar; use bar's open as tiebreaker.
            # Conservative: take whichever side is closer to bar open.
            bar_idx = i + 1 + u_first
            bar_open = df["open"].iat[bar_idx]
            if abs(bar_open - upper) <= abs(bar_open - lower):
                off, reason, exitp = u_first + 1, UPPER, upper
            else:
                off, reason, exitp = l_first + 1, LOWER, lower
        else:
            # neither touched within horizon
            off = horizon_bars
            reason = TIMEOUT
            exitp = close[i + horizon_bars]

        exit_offset[i] = off
        exit_reason[i] = reason
        exit_price[i] = exitp
        log_ret[i] = float(np.log(exitp / c_i))
        valid[i] = True

    return pd.DataFrame({
        "timestamp": ts,
        "entry_close": close,
        "atr": atr,
        "upper_price": upper_arr,
        "lower_price": lower_arr,
        "exit_offset": exit_offset,
        "exit_reason": exit_reason,
        "exit_price": exit_price,
        "log_return": log_ret,
        "valid": valid,
    })
