"""
V11 specialist-aware meta-labelling.

Inputs:
    bars                : dollar-bar OHLCV frame
    primary_signal      : Series of {-1, 0, +1} from primary_rule(...)
    horizon_bars        : forward-look length
    barrier_mult_per_bar: vol-regime-scaled multiplier (from horizon_conditional)
    slippage_bps        : round-trip slippage in bps (default 6.0)

Output rows align with bars index; only rows where primary_signal != 0
AND triple-barrier valid are `eligible`. For those rows:

    R_gross = primary_signal * (exit_price - entry_close) / (m * atr)
    R_net   = R_gross - slippage_R
    meta_label = 1 iff exit_reason matches primary_signal's profit side

Also returns entry_idx and exit_idx (integer bar indices) for use by
the sample-uniqueness weighting and the walk-forward purge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gpu_trainer_v11.labels.triple_barrier import LOWER, UPPER
from gpu_trainer_v11.labels.triple_barrier_v11 import compute_triple_barrier_per_bar

SLIPPAGE_BPS_DEFAULT = 6.0


def compute_meta_labels_v11(
    bars: pd.DataFrame,
    primary_signal: pd.Series | np.ndarray,
    horizon_bars: int,
    barrier_mult_per_bar: np.ndarray,
    atr_window: int = 14,
    slippage_bps: float = SLIPPAGE_BPS_DEFAULT,
) -> pd.DataFrame:
    n = len(bars)
    primary = (primary_signal.to_numpy() if isinstance(primary_signal, pd.Series)
               else np.asarray(primary_signal)).astype(np.int8)
    if len(primary) != n:
        raise ValueError("primary_signal length mismatch")

    tb = compute_triple_barrier_per_bar(
        bars, horizon_bars=horizon_bars,
        barrier_mult_per_bar=barrier_mult_per_bar,
        atr_window=atr_window,
    )

    atr = tb["atr"].to_numpy()
    bm = tb["barrier_mult"].to_numpy()
    half_width = bm * atr  # price units, varies per bar
    exit_reason = tb["exit_reason"].to_numpy()
    exit_offset = tb["exit_offset"].to_numpy()
    valid = tb["valid"].to_numpy()
    entry_close = tb["entry_close"].to_numpy()
    exit_price = tb["exit_price"].to_numpy()

    eligible = valid & (primary != 0) & np.isfinite(half_width) & (half_width > 0)

    R_gross = np.full(n, np.nan, dtype=np.float64)
    R_gross[eligible] = (primary[eligible] *
                         (exit_price[eligible] - entry_close[eligible]) /
                         half_width[eligible])
    slip_R = (slippage_bps * 1e-4 * entry_close) / np.where(half_width > 0, half_width, np.nan)
    R_net = np.full(n, np.nan, dtype=np.float64)
    R_net[eligible] = R_gross[eligible] - slip_R[eligible]

    meta = np.zeros(n, dtype=np.int8)
    long_win = eligible & (primary == 1) & (exit_reason == UPPER)
    short_win = eligible & (primary == -1) & (exit_reason == LOWER)
    meta[long_win | short_win] = 1

    entry_idx = np.arange(n, dtype=np.int64)
    exit_idx = entry_idx + exit_offset.astype(np.int64)

    return pd.DataFrame({
        "timestamp": tb["timestamp"].values,
        "entry_close": entry_close,
        "atr": atr,
        "barrier_mult": bm,
        "primary_dir": primary,
        "exit_offset": exit_offset,
        "exit_reason": exit_reason,
        "exit_price": exit_price,
        "R_gross": R_gross,
        "R_net": R_net,
        "meta_label": meta,
        "eligible": eligible,
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
    })


HORIZONS_BARS = {"4h": 16, "8h": 32, "1d": 96}
