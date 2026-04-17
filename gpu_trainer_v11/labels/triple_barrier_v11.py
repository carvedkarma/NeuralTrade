"""
Per-bar triple-barrier with variable barrier multiplier.

Identical algorithm to `triple_barrier.compute_triple_barrier` but the
barrier multiplier may vary per bar — used for horizon-conditional
barrier widths (vol-regime scaled).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gpu_trainer_v11.labels.triple_barrier import (
    LOWER, TIMEOUT, UPPER, _wilder_atr,
)


def compute_triple_barrier_per_bar(
    df: pd.DataFrame,
    horizon_bars: int,
    barrier_mult_per_bar: np.ndarray,
    atr_window: int = 14,
) -> pd.DataFrame:
    """First-touch labeler with per-bar barrier multiplier."""
    if horizon_bars < 1:
        raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")
    n = len(df)
    if len(barrier_mult_per_bar) != n:
        raise ValueError("barrier_mult_per_bar must align with df length")

    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    ts = df["timestamp"].to_numpy()
    bm = barrier_mult_per_bar.astype(np.float64)

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
        if not np.isfinite(a_i) or i > last_entry or not np.isfinite(bm[i]):
            continue
        c_i = close[i]
        m = bm[i]
        upper = c_i + m * a_i
        lower = c_i - m * a_i
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
            off, reason, exitp = u_first + 1, UPPER, upper
        elif l_first < u_first:
            off, reason, exitp = l_first + 1, LOWER, lower
        elif u_first < horizon_bars:
            bar_idx = i + 1 + u_first
            bar_open = float(df["open"].iat[bar_idx])
            if abs(bar_open - upper) <= abs(bar_open - lower):
                off, reason, exitp = u_first + 1, UPPER, upper
            else:
                off, reason, exitp = l_first + 1, LOWER, lower
        else:
            off, reason, exitp = horizon_bars, TIMEOUT, close[i + horizon_bars]

        exit_offset[i] = off
        exit_reason[i] = reason
        exit_price[i] = exitp
        log_ret[i] = float(np.log(exitp / c_i))
        valid[i] = True

    return pd.DataFrame({
        "timestamp": ts,
        "entry_close": close,
        "atr": atr,
        "barrier_mult": bm,
        "upper_price": upper_arr,
        "lower_price": lower_arr,
        "exit_offset": exit_offset,
        "exit_reason": exit_reason,
        "exit_price": exit_price,
        "log_return": log_ret,
        "valid": valid,
    })
