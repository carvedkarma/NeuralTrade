"""
Cross-asset features.

For an alt symbol X, computes:
    btc_numeraire_ret      : log(close_X / btc_close_aligned).diff()
    btc_dom_proxy          : rolling z-score of (btc_close - alt_close_normalized)
    btc_alt_corr_64        : rolling 64-bar Pearson corr of log returns
    btc_alt_beta_64        : rolling OLS beta of alt-ret on btc-ret over 64 bars

For BTC itself the function returns zeros (no cross-asset signal vs. self).

Alignment is by timestamp via merge_asof — we take the LATEST btc bar
whose close-time ≤ the alt bar's close-time. This is causal w.r.t. real
trading time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_log_ret(s: pd.Series) -> pd.Series:
    return np.log(s.replace(0, np.nan)).diff()


def compute_cross_asset(
    alt_bars: pd.DataFrame,
    btc_bars: pd.DataFrame | None,
    symbol: str,
) -> pd.DataFrame:
    n = len(alt_bars)
    out = pd.DataFrame(index=alt_bars.index, data={
        "btc_numeraire_ret": np.zeros(n),
        "btc_dom_proxy": np.zeros(n),
        "btc_alt_corr_64": np.zeros(n),
        "btc_alt_beta_64": np.zeros(n),
    })
    if symbol == "BTCUSDT" or btc_bars is None or len(btc_bars) == 0:
        return out

    alt_view = alt_bars[["timestamp", "close"]].copy().rename(columns={"close": "alt_close"})
    btc_view = btc_bars[["timestamp", "close"]].copy().rename(columns={"close": "btc_close"})
    merged = pd.merge_asof(
        alt_view.sort_values("timestamp"),
        btc_view.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    merged = merged.set_index(alt_bars.index)

    btc_close = merged["btc_close"].astype(float)
    alt_close = merged["alt_close"].astype(float)

    btc_num = (alt_close / btc_close.replace(0, np.nan))
    out["btc_numeraire_ret"] = _safe_log_ret(btc_num).fillna(0.0)

    # Normalized BTC-vs-alt level proxy (z-scored over 200 bars)
    proxy_raw = btc_close - alt_close * (btc_close.iloc[0] / alt_close.iloc[0]
                                         if alt_close.iloc[0] else 1.0)
    mean = proxy_raw.rolling(200, min_periods=20).mean()
    std = proxy_raw.rolling(200, min_periods=20).std().replace(0, np.nan)
    out["btc_dom_proxy"] = ((proxy_raw - mean) / std).fillna(0.0)

    btc_ret = _safe_log_ret(btc_close).fillna(0.0)
    alt_ret = _safe_log_ret(alt_close).fillna(0.0)
    win = 64
    out["btc_alt_corr_64"] = btc_ret.rolling(win, min_periods=8).corr(alt_ret).fillna(0.0)
    cov = btc_ret.rolling(win, min_periods=8).cov(alt_ret)
    var = btc_ret.rolling(win, min_periods=8).var().replace(0, np.nan)
    out["btc_alt_beta_64"] = (cov / var).fillna(0.0)
    return out
