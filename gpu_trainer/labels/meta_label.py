"""
Meta-labeling layer (López de Prado, AFML ch. 3.6).

Primary rule (no learning involved): trade in the direction of the
sign of the prior `primary_lookback_bars` log return. On a 15-minute
chart with primary_lookback_bars=16 this is "follow the prior 4h
momentum."

Meta-label (the binary target the GBDT predicts): given that primary
direction and the triple-barrier outcome at horizon H, did the trade
hit its profit-side barrier first?
    LONG  meta-label = 1 iff exit_reason == UPPER (+1)
    SHORT meta-label = 1 iff exit_reason == LOWER (-1)

Realized R is computed in 1-R units of the barrier distance using
LINEAR price returns (not log) so that ±1R is exactly symmetric:
    half_width  = barrier_mult * atr                # price units
    R_long      = (exit_price - entry_close) / half_width
    R_short     = (entry_close - exit_price) / half_width
By construction, when the trade's profit-side barrier is touched
R_gross == +1.0 EXACTLY; when the loss-side barrier is touched
R_gross == -1.0 EXACTLY. Slippage is subtracted in the same R-units:
    slippage_R = slippage_bps * 1e-4 * entry_close / half_width
    R_net      = R_gross - slippage_R

Locked Phase-1 defaults (do NOT tune):
    primary_lookback_bars = 16   # 4h on 15m bars
    horizons              = [16, 32, 96]
    barrier_mult          = 1.5
    atr_window            = 14
    slippage_bps          = 6.0  # matches shared_v5_trade_config.slippage_base_bps
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from labels.triple_barrier import compute_triple_barrier, UPPER, LOWER

PRIMARY_LOOKBACK_BARS_DEFAULT = 16
SLIPPAGE_BPS_DEFAULT = 6.0
BARRIER_MULT_DEFAULT = 1.5
ATR_WINDOW_DEFAULT = 14


def compute_primary_direction(close: np.ndarray, lookback_bars: int) -> np.ndarray:
    """Sign of log return over `lookback_bars`. Returns int8 array of {-1, 0, +1}.

    The first `lookback_bars` entries are 0 (insufficient history).
    """
    n = len(close)
    out = np.zeros(n, dtype=np.int8)
    if n <= lookback_bars:
        return out
    prev = close[:-lookback_bars]
    cur = close[lookback_bars:]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(cur / prev)
    out[lookback_bars:] = np.sign(r).astype(np.int8)
    return out


def compute_meta_labels(
    df: pd.DataFrame,
    horizon_bars: int,
    primary_lookback_bars: int = PRIMARY_LOOKBACK_BARS_DEFAULT,
    barrier_mult: float = BARRIER_MULT_DEFAULT,
    atr_window: int = ATR_WINDOW_DEFAULT,
    slippage_bps: float = SLIPPAGE_BPS_DEFAULT,
    triple_barrier_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the meta-label table for one horizon.

    df must have: timestamp, open, high, low, close.

    Returns DataFrame with columns:
        timestamp, entry_close, atr, primary_dir,
        exit_offset, exit_reason, exit_price,
        R_gross, R_net, meta_label, eligible
    Where `eligible` is True iff primary_dir != 0 AND triple-barrier valid.
    Rows that are not eligible carry NaN R values and meta_label=0.
    """
    if triple_barrier_df is None:
        tb = compute_triple_barrier(df, horizon_bars, atr_window, barrier_mult)
    else:
        tb = triple_barrier_df

    close = df["close"].to_numpy(dtype=np.float64)
    primary = compute_primary_direction(close, primary_lookback_bars)

    atr = tb["atr"].to_numpy()
    exit_reason = tb["exit_reason"].to_numpy()
    valid = tb["valid"].to_numpy()
    entry_close = tb["entry_close"].to_numpy()
    exit_price = tb["exit_price"].to_numpy()

    # Symmetric ±1R using linear price returns scaled by half-barrier width.
    half_width = barrier_mult * atr  # in price units
    eligible = valid & (primary != 0) & np.isfinite(half_width) & (half_width > 0)

    R_gross = np.full(len(df), np.nan, dtype=np.float64)
    R_gross[eligible] = (primary[eligible] *
                          (exit_price[eligible] - entry_close[eligible])
                          / half_width[eligible])

    # Round-trip slippage: bps of entry price, expressed in R-units
    # by dividing by the same half-barrier price width.
    slip_R = (slippage_bps * 1e-4 * entry_close) / np.where(half_width > 0, half_width, np.nan)
    R_net = np.full(len(df), np.nan, dtype=np.float64)
    R_net[eligible] = R_gross[eligible] - slip_R[eligible]

    meta = np.zeros(len(df), dtype=np.int8)
    long_win = eligible & (primary == 1) & (exit_reason == UPPER)
    short_win = eligible & (primary == -1) & (exit_reason == LOWER)
    meta[long_win | short_win] = 1

    return pd.DataFrame({
        "timestamp": tb["timestamp"].values,
        "entry_close": entry_close,
        "atr": atr,
        "primary_dir": primary.astype(np.int8),
        "exit_offset": tb["exit_offset"].values,
        "exit_reason": exit_reason,
        "exit_price": tb["exit_price"].values,
        "R_gross": R_gross,
        "R_net": R_net,
        "meta_label": meta,
        "eligible": eligible,
    })


HORIZONS_BARS = {"4h": 16, "8h": 32, "1d": 96}
