"""
Base technical feature set.

Curated set of standard technical features (returns, MAs, momentum,
volatility, oscillators) computed on dollar bars. Strictly causal:
each feature at bar i uses only bars 0..i.

Total emitted: ~88 features (auto-counted at composition time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=max(2, span // 4)).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    dn = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    roll_dn = dn.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _bb(close: pd.Series, period: int = 20, k: float = 2.0):
    ma = close.rolling(period, min_periods=max(2, period // 4)).mean()
    sd = close.rolling(period, min_periods=max(2, period // 4)).std()
    upper = ma + k * sd
    lower = ma - k * sd
    width = (upper - lower) / ma.replace(0, np.nan)
    return ma, upper, lower, width


def _stoch_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    rsi = _rsi(close, period)
    mn = rsi.rolling(period, min_periods=max(2, period // 4)).min()
    mx = rsi.rolling(period, min_periods=max(2, period // 4)).max()
    rng = (mx - mn).replace(0, np.nan)
    return ((rsi - mn) / rng).fillna(0.5)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = _ema(close, fast) - _ema(close, slow)
    sig_line = _ema(macd_line, signal)
    hist = macd_line - sig_line
    return macd_line, sig_line, hist


def compute_base_technical(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the base technical feature block.

    Returns DataFrame indexed like input.
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    log_close = np.log(close.replace(0, np.nan))
    log_ret = log_close.diff()
    out["log_ret_1"] = log_ret.fillna(0.0)
    for k in (2, 4, 8, 16, 32, 64, 128):
        out[f"log_ret_{k}"] = log_close.diff(k).fillna(0.0)

    # EMAs (price ratio to EMA — stationary form)
    for span in (5, 10, 20, 50, 100, 200):
        ema = _ema(close, span)
        out[f"close_over_ema_{span}"] = (close / ema.replace(0, np.nan) - 1.0).fillna(0.0)

    # EMA crossovers
    out["ema_5_20_diff"] = (_ema(close, 5) / _ema(close, 20).replace(0, np.nan) - 1.0).fillna(0.0)
    out["ema_20_50_diff"] = (_ema(close, 20) / _ema(close, 50).replace(0, np.nan) - 1.0).fillna(0.0)
    out["ema_50_200_diff"] = (_ema(close, 50) / _ema(close, 200).replace(0, np.nan) - 1.0).fillna(0.0)

    # RSI family
    for period in (7, 14, 28):
        out[f"rsi_{period}"] = _rsi(close, period) / 100.0
    out["stoch_rsi_14"] = _stoch_rsi(close, 14)

    # Bollinger
    for period in (20, 50):
        ma, upper, lower, width = _bb(close, period)
        out[f"bb_pct_b_{period}"] = ((close - lower) / (upper - lower).replace(0, np.nan)).fillna(0.5)
        out[f"bb_width_{period}"] = width.fillna(0.0)

    # MACD
    macd_line, sig_line, hist = _macd(close)
    denom = close.replace(0, np.nan)
    out["macd_line_norm"] = (macd_line / denom).fillna(0.0)
    out["macd_signal_norm"] = (sig_line / denom).fillna(0.0)
    out["macd_hist_norm"] = (hist / denom).fillna(0.0)

    # ATR-derived
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    for period in (7, 14, 28):
        atr_p = tr.rolling(period, min_periods=max(2, period // 2)).mean()
        out[f"atr_{period}_pct"] = (atr_p / close.replace(0, np.nan)).fillna(0.0)

    # Range / position-in-range
    bar_range_pct = ((high - low) / close.replace(0, np.nan)).fillna(0.0)
    out["bar_range_pct"] = bar_range_pct
    body_pct = ((close - df["open"]) / close.replace(0, np.nan)).fillna(0.0)
    out["body_pct"] = body_pct
    out["upper_wick_pct"] = ((high - close.where(close >= df["open"], df["open"])) / close.replace(0, np.nan)).fillna(0.0)
    out["lower_wick_pct"] = ((close.where(close <= df["open"], df["open"]) - low) / close.replace(0, np.nan)).fillna(0.0)
    pos_in_range = ((close - low) / (high - low).replace(0, np.nan)).fillna(0.5)
    out["close_pos_in_range"] = pos_in_range

    # Volume features
    log_vol = np.log(vol.replace(0, np.nan))
    out["log_vol"] = log_vol.fillna(0.0)
    for span in (5, 20, 50):
        vol_ma = vol.rolling(span, min_periods=max(2, span // 4)).mean()
        out[f"vol_over_ma_{span}"] = (vol / vol_ma.replace(0, np.nan) - 1.0).fillna(0.0)
    for k in (1, 4, 16):
        out[f"log_vol_diff_{k}"] = log_vol.diff(k).fillna(0.0)

    # Realized-vol features
    for win in (8, 32, 128):
        out[f"rvol_{win}"] = (log_ret.pow(2).rolling(win, min_periods=max(2, win // 4)).mean()).pow(0.5).fillna(0.0)

    # Skew / kurt of returns
    for win in (32, 128):
        out[f"ret_skew_{win}"] = log_ret.rolling(win, min_periods=max(4, win // 4)).skew().fillna(0.0)
        out[f"ret_kurt_{win}"] = log_ret.rolling(win, min_periods=max(4, win // 4)).kurt().fillna(0.0)

    # Drawdown / max-from-rolling-high
    for win in (32, 128):
        roll_max = close.rolling(win, min_periods=2).max()
        roll_min = close.rolling(win, min_periods=2).min()
        out[f"close_over_max_{win}"] = (close / roll_max.replace(0, np.nan) - 1.0).fillna(0.0)
        out[f"close_over_min_{win}"] = (close / roll_min.replace(0, np.nan) - 1.0).fillna(0.0)

    # Z-scores of return
    for win in (32, 128):
        m = log_ret.rolling(win, min_periods=4).mean()
        s = log_ret.rolling(win, min_periods=4).std().replace(0, np.nan)
        out[f"ret_z_{win}"] = ((log_ret - m) / s).fillna(0.0)

    # Higher-order momentum: cumulative log return then z-score
    for win in (16, 64):
        cum = log_ret.rolling(win, min_periods=4).sum()
        m = cum.rolling(win, min_periods=4).mean()
        s = cum.rolling(win, min_periods=4).std().replace(0, np.nan)
        out[f"cumret_z_{win}"] = ((cum - m) / s).fillna(0.0)

    # Auto-correlation of returns at common lags (rolling)
    for lag in (1, 2, 4, 8):
        rolling_ac = log_ret.rolling(64, min_periods=16).corr(log_ret.shift(lag))
        out[f"ret_ac_lag_{lag}"] = rolling_ac.fillna(0.0)

    # Replace inf
    out = out.replace([np.inf, -np.inf], 0.0)
    return out
