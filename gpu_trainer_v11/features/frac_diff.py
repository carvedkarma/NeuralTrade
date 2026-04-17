"""
Fractional differentiation (Hosking) with auto-d selection.

Reference: López de Prado, Advances in Financial Machine Learning, ch. 5.

Why: log-returns kill the level information; raw prices are non-stationary.
Fractional differentiation finds the smallest d ∈ [0,1] such that the
output series is stationary (ADF test passes), retaining maximum memory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def get_weights(d: float, threshold: float = 1e-4) -> np.ndarray:
    """Generate FFD weights, truncated when |w_k| < threshold.

    w_0 = 1
    w_k = w_{k-1} * (-(d - k + 1) / k)
    """
    w = [1.0]
    k = 1
    while True:
        w_k = w[-1] * (-(d - k + 1) / k)
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
        if k > 10000:
            break
    return np.array(w[::-1], dtype=np.float64)  # newest weight last (matches convolution order)


def frac_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
    """Fixed-window fractional differentiation.

    Uses a fixed window equal to the truncated weight length, so every
    output value is computed from the same number of past observations
    — preserves stationarity of the input distribution.
    """
    w = get_weights(d, threshold)
    win = len(w)
    arr = series.to_numpy(dtype=np.float64)
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < win:
        return pd.Series(out, index=series.index)
    # convolve gives, at position i, sum_{k=0..win-1} w[k] * arr[i-(win-1-k)]
    # which equals sum_{j=0..win-1} w[win-1-j] * arr[i-j] — the standard form
    conv = np.convolve(arr, w, mode="valid")
    out[win - 1:] = conv
    return pd.Series(out, index=series.index)


def find_min_d(series: pd.Series, candidates: list[float] | None = None,
               adf_pvalue: float = 0.01, threshold: float = 1e-4) -> float:
    """Find the smallest d in `candidates` that makes the series stationary.

    If none pass, returns 1.0 (full differencing — log-return-like).
    """
    if candidates is None:
        candidates = [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]
    log_s = np.log(series.dropna().astype(float))
    if len(log_s) < 100:
        return 1.0
    for d in candidates:
        diffed = frac_diff_ffd(log_s, d, threshold).dropna()
        if len(diffed) < 50:
            continue
        try:
            pval = adfuller(diffed, maxlag=1, regression="c", autolag=None)[1]
        except Exception:
            continue
        if pval < adf_pvalue:
            return float(d)
    return 1.0
