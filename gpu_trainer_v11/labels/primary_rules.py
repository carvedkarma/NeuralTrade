"""
The two specialists' primary entry rules.

Each rule emits a Series of {-1, 0, +1} where:
    +1 = enter LONG at this bar's close
    -1 = enter SHORT at this bar's close
     0 = do not enter

Rules are simple, hand-crafted, regime-conditioned. The Transformer's
job is the META layer: given a primary-rule entry, predict whether the
triple-barrier resolves with the rule's intended sign.

Locked rules (do not edit after pre-flight runs):

Specialist A — momentum-after-vol-contraction (LONG only)
    prior_4h_log_return > 0
    AND bb_width_20 < 30th percentile of bb_width_20 over rolling 200 bars
    AND adx_14 > rolling 200-bar median of adx_14

Specialist B — mean-reversion-after-vol-expansion (SHORT only)
    prior_4h_log_return < 0
    AND atr_14 / atr_14.shift(8) > 1.30 (ATR has expanded ≥ 30% in 2h)
    AND bb_width_20 > 70th percentile of bb_width_20 over rolling 200 bars
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

# 4h on dollar-bar timeline ≈ 16 bars (96 bars/day target).
PRIOR_LOOKBACK_BARS = 16
ATR_EXPANSION_LOOKBACK = 8     # ~2h
BB_PCT_WINDOW = 200
ADX_MEDIAN_WINDOW = 200


def _bb_width_20(close: pd.Series) -> pd.Series:
    ma = close.rolling(20, min_periods=5).mean()
    sd = close.rolling(20, min_periods=5).std()
    return ((ma + 2 * sd) - (ma - 2 * sd)) / ma.replace(0, np.nan)


def specialist_A_long(bars: pd.DataFrame, side: pd.DataFrame) -> pd.Series:
    """Momentum-after-vol-contraction LONG rule."""
    log_close = np.log(bars["close"].replace(0, np.nan))
    prior_ret = log_close.diff(PRIOR_LOOKBACK_BARS)

    bb_w = _bb_width_20(bars["close"])
    bb_q30 = bb_w.rolling(BB_PCT_WINDOW, min_periods=30).quantile(0.30)

    adx = side["adx_14"]
    adx_med = adx.rolling(ADX_MEDIAN_WINDOW, min_periods=30).median()

    cond = (prior_ret > 0) & (bb_w < bb_q30) & (adx > adx_med)
    out = pd.Series(np.where(cond, 1, 0), index=bars.index, dtype=np.int8)
    out.iloc[:max(PRIOR_LOOKBACK_BARS, BB_PCT_WINDOW, ADX_MEDIAN_WINDOW)] = 0
    return out


def specialist_B_short(bars: pd.DataFrame, side: pd.DataFrame) -> pd.Series:
    """Mean-reversion-after-vol-expansion SHORT rule."""
    log_close = np.log(bars["close"].replace(0, np.nan))
    prior_ret = log_close.diff(PRIOR_LOOKBACK_BARS)

    atr = side["atr_14"]
    atr_expansion = atr / atr.shift(ATR_EXPANSION_LOOKBACK).replace(0, np.nan)

    bb_w = _bb_width_20(bars["close"])
    bb_q70 = bb_w.rolling(BB_PCT_WINDOW, min_periods=30).quantile(0.70)

    cond = (prior_ret < 0) & (atr_expansion > 1.30) & (bb_w > bb_q70)
    out = pd.Series(np.where(cond, -1, 0), index=bars.index, dtype=np.int8)
    out.iloc[:max(PRIOR_LOOKBACK_BARS, BB_PCT_WINDOW, ATR_EXPANSION_LOOKBACK)] = 0
    return out


def primary_rule(
    bars: pd.DataFrame,
    side_data: pd.DataFrame,
    rule: Literal["A", "B"],
) -> pd.Series:
    if rule == "A":
        return specialist_A_long(bars, side_data)
    if rule == "B":
        return specialist_B_short(bars, side_data)
    raise ValueError(f"unknown rule: {rule}")
