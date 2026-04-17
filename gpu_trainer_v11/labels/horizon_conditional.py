"""
Horizon-conditional triple-barrier widths, scaled by vol regime.

Per locked contract:
    barrier_mult by atr_pct_bucket:
        bucket 0 (low vol)  -> 1.2
        bucket 1 (mid vol)  -> 1.5
        bucket 2 (high vol) -> 2.0

Returns a per-bar barrier multiplier as a numpy array — used by
triple_barrier.compute_triple_barrier_labels (which already accepts
per-bar pt/sl multipliers via the `pt_sl` argument as fixed scalars,
so for V11 we expand that path slightly with a wrapper).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BUCKET_TO_MULT = {0: 1.2, 1: 1.5, 2: 2.0}


def per_bar_barrier_mult(atr_pct_bucket: pd.Series) -> np.ndarray:
    arr = atr_pct_bucket.fillna(1).astype(int).clip(0, 2).to_numpy()
    return np.vectorize(BUCKET_TO_MULT.get)(arr).astype(np.float64)
