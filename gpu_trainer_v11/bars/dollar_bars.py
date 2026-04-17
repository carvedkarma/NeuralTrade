"""
Dollar-bar construction from 15-minute OHLCV.

A dollar bar is emitted whenever cumulative `close × volume` since the
last emission crosses a fixed threshold. Each bar's OHLC is the union
of its constituent 15m bars; its timestamp is the close time of the
final 15m bar that triggered emission.

Construction is strictly causal — bar i is fully formed before bar i+1
begins accumulating, and no future information enters bar i.

Locked design:
    - threshold is a single dollar amount, fixed before training and
      written into `gpu_trainer_v11/README.md`.
    - tie-breaking on the bar that crosses the threshold: include the
      whole 15m bar (over-shoot is small relative to threshold).
    - the partial residual after the final emitted bar is discarded
      (we do not emit a stub trailing bar; it would have a different
      information content than the locked size).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DollarBarConfig:
    threshold_dollars: float
    # Optional cap on the number of 15m bars a single dollar bar may
    # span. Quiet markets can produce pathologically long bars; this
    # protects against the degenerate "1 dollar bar = 1 month" case.
    # Default = 24h on 15m bars = 96.
    max_bars_per_dollar_bar: int = 96


def build_dollar_bars(df_15m: pd.DataFrame, cfg: DollarBarConfig) -> pd.DataFrame:
    """Build dollar bars from a 15m OHLCV dataframe.

    df_15m must have columns: timestamp (ms int), open, high, low, close, volume.
    Rows must be in ascending time order.

    Returns DataFrame with columns:
        timestamp        : close-time of the dollar bar (ms int)
        open             : open of the first constituent 15m bar
        high             : max high of constituents
        low              : min low of constituents
        close            : close of the last constituent 15m bar
        volume           : sum of constituent base-asset volumes
        dollar_volume    : sum of (close × volume) of constituents
        n_15m_bars       : number of 15m bars consumed
        start_timestamp  : open-time of the first constituent (ms int)
    """
    if cfg.threshold_dollars <= 0:
        raise ValueError(f"threshold_dollars must be > 0, got {cfg.threshold_dollars}")
    if cfg.max_bars_per_dollar_bar < 1:
        raise ValueError("max_bars_per_dollar_bar must be >= 1")

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df_15m.columns)
    if missing:
        raise ValueError(f"df_15m missing columns: {missing}")

    ts = df_15m["timestamp"].to_numpy(dtype=np.int64)
    op = df_15m["open"].to_numpy(dtype=np.float64)
    hi = df_15m["high"].to_numpy(dtype=np.float64)
    lo = df_15m["low"].to_numpy(dtype=np.float64)
    cl = df_15m["close"].to_numpy(dtype=np.float64)
    vol = df_15m["volume"].to_numpy(dtype=np.float64)

    n = len(df_15m)
    if n == 0:
        return pd.DataFrame(columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "dollar_volume", "n_15m_bars", "start_timestamp",
        ])

    dollar_per_15m = cl * vol  # close-price proxy; matches threshold semantics

    out_ts, out_open, out_high, out_low, out_close = [], [], [], [], []
    out_vol, out_dol, out_n, out_start_ts = [], [], [], []

    cum_dollar = 0.0
    cum_vol = 0.0
    bar_open = op[0]
    bar_high = hi[0]
    bar_low = lo[0]
    bar_n = 0
    bar_start_ts = ts[0]

    for i in range(n):
        if bar_n == 0:
            bar_open = op[i]
            bar_high = hi[i]
            bar_low = lo[i]
            bar_start_ts = ts[i]
        else:
            if hi[i] > bar_high:
                bar_high = hi[i]
            if lo[i] < bar_low:
                bar_low = lo[i]

        cum_dollar += dollar_per_15m[i]
        cum_vol += vol[i]
        bar_n += 1

        crossed = cum_dollar >= cfg.threshold_dollars
        capped = bar_n >= cfg.max_bars_per_dollar_bar
        if crossed or capped:
            out_ts.append(int(ts[i]))
            out_open.append(float(bar_open))
            out_high.append(float(bar_high))
            out_low.append(float(bar_low))
            out_close.append(float(cl[i]))
            out_vol.append(float(cum_vol))
            out_dol.append(float(cum_dollar))
            out_n.append(int(bar_n))
            out_start_ts.append(int(bar_start_ts))
            cum_dollar = 0.0
            cum_vol = 0.0
            bar_n = 0

    return pd.DataFrame({
        "timestamp": np.array(out_ts, dtype=np.int64),
        "open": np.array(out_open, dtype=np.float64),
        "high": np.array(out_high, dtype=np.float64),
        "low": np.array(out_low, dtype=np.float64),
        "close": np.array(out_close, dtype=np.float64),
        "volume": np.array(out_vol, dtype=np.float64),
        "dollar_volume": np.array(out_dol, dtype=np.float64),
        "n_15m_bars": np.array(out_n, dtype=np.int32),
        "start_timestamp": np.array(out_start_ts, dtype=np.int64),
    })


def bars_per_day(dollar_bars: pd.DataFrame) -> float:
    """Average dollar bars per UTC day across the dataset."""
    if len(dollar_bars) < 2:
        return 0.0
    span_ms = dollar_bars["timestamp"].iloc[-1] - dollar_bars["timestamp"].iloc[0]
    span_days = span_ms / (1000 * 60 * 60 * 24)
    if span_days <= 0:
        return 0.0
    return len(dollar_bars) / span_days
