"""
Regime-context features.

Features:
    vol_of_vol      : rolling std of rv_med (volatility of volatility)
    atr_pct_bucket  : 0/1/2 for low/mid/high ATR percentile within rolling window
    adx_regime      : 0=range / 1=weak-trend / 2=strong-trend  (ADX < 18 / 18-25 / >25)

All strictly causal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=max(2, period // 2)).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = ((up > down) & (up > 0)).astype(float) * up.clip(lower=0)
    minus_dm = ((down > up) & (down > 0)).astype(float) * down.clip(lower=0)
    atr = _atr(df, period).replace(0, np.nan)
    plus_di = 100 * plus_dm.rolling(period, min_periods=max(2, period // 2)).mean() / atr
    minus_di = 100 * minus_dm.rolling(period, min_periods=max(2, period // 2)).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(period, min_periods=max(2, period // 2)).mean()
    return adx.fillna(0.0)


def compute_regime(df: pd.DataFrame, vol_window: int = 200, atr_window: int = 200) -> pd.DataFrame:
    log_ret = np.log(df["close"]).diff()
    rv = log_ret.pow(2).rolling(64, min_periods=8).mean().pow(0.5)
    vol_of_vol = rv.rolling(vol_window, min_periods=20).std().fillna(0.0)

    atr = _atr(df, 14)
    rolling_q33 = atr.rolling(atr_window, min_periods=30).quantile(0.33)
    rolling_q66 = atr.rolling(atr_window, min_periods=30).quantile(0.66)
    bucket = pd.Series(np.where(atr <= rolling_q33, 0,
                                np.where(atr >= rolling_q66, 2, 1)), index=df.index)
    bucket = bucket.where(rolling_q33.notna(), 1).astype(np.int8)

    adx = _adx(df, 14)
    adx_regime = pd.Series(np.where(adx < 18, 0,
                                    np.where(adx < 25, 1, 2)), index=df.index, dtype=np.int8)
    return pd.DataFrame({
        "vol_of_vol": vol_of_vol,
        "atr_pct_bucket": bucket,
        "adx_regime": adx_regime,
        # exposed for downstream label gating
        "atr_14": atr.ffill().fillna(0.0),
        "adx_14": adx,
    }, index=df.index)
