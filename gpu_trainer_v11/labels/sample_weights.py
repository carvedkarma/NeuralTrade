"""
Sample-uniqueness weights (López de Prado AFML chapter 4, eq. 4.2).

Each label spans an interval [t_in, t_out] of bars. A sample's weight
is the inverse of the average concurrency it experiences over its
own life — heavily overlapping samples get down-weighted because they
share information.

Algorithm:
    1. For every bar t, count the number of label intervals that contain t.
    2. For each sample i, weight_i = mean(1 / count[t]) for t in [t_in_i, t_out_i].
    3. Normalize so weights average to 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sample_uniqueness_weights(
    t_in_bar_idx: np.ndarray,
    t_out_bar_idx: np.ndarray,
    n_bars: int,
) -> np.ndarray:
    if len(t_in_bar_idx) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(t_in_bar_idx) != len(t_out_bar_idx):
        raise ValueError("t_in and t_out length mismatch")

    coverage = np.zeros(n_bars, dtype=np.int64)
    # Interval is closed on both ends — the entry bar is held; exit bar is when barrier touches.
    for a, b in zip(t_in_bar_idx, t_out_bar_idx):
        a = int(max(0, a))
        b = int(min(n_bars - 1, b))
        if b < a:
            continue
        coverage[a:b + 1] += 1

    inv_cov = np.where(coverage > 0, 1.0 / coverage, 0.0)
    weights = np.zeros(len(t_in_bar_idx), dtype=np.float64)
    for i, (a, b) in enumerate(zip(t_in_bar_idx, t_out_bar_idx)):
        a = int(max(0, a))
        b = int(min(n_bars - 1, b))
        if b < a:
            continue
        seg = inv_cov[a:b + 1]
        weights[i] = float(seg.mean()) if len(seg) > 0 else 0.0

    mean = weights[weights > 0].mean() if (weights > 0).any() else 1.0
    if mean > 0:
        weights = weights / mean
    return weights


def weights_from_label_df(labels_df: pd.DataFrame, n_bars: int) -> np.ndarray:
    """Convenience wrapper over the output of triple_barrier labelling.

    Expects labels_df to carry columns 'entry_idx' and 'exit_idx'
    (integer bar indices into the underlying bar frame).
    """
    if "entry_idx" not in labels_df.columns or "exit_idx" not in labels_df.columns:
        raise ValueError("labels_df must contain entry_idx and exit_idx columns")
    return sample_uniqueness_weights(
        labels_df["entry_idx"].to_numpy(),
        labels_df["exit_idx"].to_numpy(),
        n_bars,
    )
