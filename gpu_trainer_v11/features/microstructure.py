"""
OHLCV-only microstructure proxies.

The 15m parquets have no taker-volume or trade-count columns, so we
reconstruct microstructure indicators from OHLCV alone. These proxies
are well-known and used in the academic literature when high-frequency
trade data is unavailable.

All features are strictly causal — value at bar i uses bars 0..i only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def signed_volume_proxy(df: pd.DataFrame) -> pd.Series:
    """Bar-level signed volume proxy.

    sign = (close - open) / (high - low + eps), clipped to [-1, +1]
    signed_vol = sign * volume

    Captures aggressor direction proxy: bars closing near the high
    after opening near the low are tagged as buyer-aggressive.
    """
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    sign = ((df["close"] - df["open"]) / rng).clip(-1.0, 1.0).fillna(0.0)
    return sign * df["volume"]


def bar_velocity(df: pd.DataFrame, n_15m_col: str = "n_15m_bars") -> pd.Series:
    """Inverse of bar duration: more 15m bars = slower; fewer = faster.

    Standardized to z-score over a rolling 200-bar window.
    """
    if n_15m_col in df.columns:
        speed = 1.0 / df[n_15m_col].clip(lower=1).astype(float)
    else:
        # fallback: compute from timestamp deltas
        dt_min = df["timestamp"].diff().fillna(0) / 60000.0
        speed = 1.0 / dt_min.clip(lower=1.0)
    mean = speed.rolling(200, min_periods=20).mean()
    std = speed.rolling(200, min_periods=20).std().replace(0, np.nan)
    return ((speed - mean) / std).fillna(0.0)


def realized_vol_cone(df: pd.DataFrame) -> pd.DataFrame:
    """Three realized-vol estimators at different lookbacks.

    Returns columns: rv_short (16), rv_med (64), rv_long (256)
    """
    log_ret = np.log(df["close"]).diff()
    out = pd.DataFrame(index=df.index)
    for name, win in [("rv_short", 16), ("rv_med", 64), ("rv_long", 256)]:
        out[name] = (log_ret.pow(2).rolling(win, min_periods=max(4, win // 4)).mean()).pow(0.5)
    return out.fillna(0.0)


def bar_duration_minutes(df: pd.DataFrame) -> pd.Series:
    """Time to form each dollar bar, in minutes.

    On the first bar there is no predecessor — emit 0 (no causal info).
    """
    dt_ms = df["timestamp"].diff().fillna(0)
    return dt_ms / 60000.0


def compute_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all OHLCV-only microstructure features.

    Output: DataFrame indexed like input, columns:
        signed_vol, signed_vol_z, bar_vel, rv_short, rv_med, rv_long, bar_dur_min
    Total: 7 columns (1 extra over locked count of 6 — signed_vol + signed_vol_z
    are kept separately for diagnostic readability; only signed_vol_z is fed
    to the model in `features/compose.py`).
    """
    sv = signed_volume_proxy(df)
    sv_mean = sv.rolling(200, min_periods=20).mean()
    sv_std = sv.rolling(200, min_periods=20).std().replace(0, np.nan)
    sv_z = ((sv - sv_mean) / sv_std).fillna(0.0)

    rv = realized_vol_cone(df)
    out = pd.DataFrame({
        "signed_vol": sv.fillna(0.0),
        "signed_vol_z": sv_z,
        "bar_vel": bar_velocity(df),
        "rv_short": rv["rv_short"],
        "rv_med": rv["rv_med"],
        "rv_long": rv["rv_long"],
        "bar_dur_min": bar_duration_minutes(df),
    }, index=df.index)
    return out
