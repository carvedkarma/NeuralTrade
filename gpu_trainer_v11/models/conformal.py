"""
Mondrian split-conformal calibration per regime bucket.

For each vol-regime bucket:
    1. Take calibration probabilities and labels.
    2. Search the smallest probability threshold T such that the bars
       admitted in that bucket satisfy
            cal_n_trades >= MIN_TRADES_PER_BUCKET
            cal_PF       >= MIN_PF
       evaluated on REALIZED R_net of the calibration bars.
    3. Store the threshold per bucket.

At inference, an admitted trade is one whose
    p_hat >= threshold_for_its_regime
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIN_TRADES_PER_BUCKET = 100  # locked by README contract
MIN_PF = 1.4


def _profit_factor(R: np.ndarray) -> float:
    if R.size == 0:
        return 0.0
    pos = R[R > 0].sum(); neg = -R[R < 0].sum()
    if neg <= 0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / neg)


@dataclass
class MondrianCalibrator:
    thresholds: dict = field(default_factory=dict)   # regime -> threshold (None = no admission)
    diag: dict = field(default_factory=dict)         # regime -> {n_admit, pf, mean_R}

    def admit(self, p: np.ndarray, regime: np.ndarray) -> np.ndarray:
        out = np.zeros(len(p), dtype=bool)
        for r, thr in self.thresholds.items():
            if thr is None:
                continue
            mask = (regime == r) & (p >= thr)
            out |= mask
        return out


def fit_mondrian(
    p_cal: np.ndarray, y_cal: np.ndarray, R_cal: np.ndarray, regime_cal: np.ndarray,
    min_trades: int = MIN_TRADES_PER_BUCKET, min_pf: float = MIN_PF,
    candidate_grid: int = 50,
) -> MondrianCalibrator:
    cal = MondrianCalibrator()
    for r in sorted(set(regime_cal.tolist())):
        m = regime_cal == r
        if m.sum() < min_trades:
            cal.thresholds[int(r)] = None
            cal.diag[int(r)] = {"n_bucket": int(m.sum()), "reason": "too few cal bars"}
            continue
        p_b = p_cal[m]; R_b = R_cal[m]
        # Threshold candidates from observed prob quantiles
        qs = np.unique(np.quantile(p_b, np.linspace(0.0, 0.99, candidate_grid)))
        chosen = None
        chosen_diag = None
        for thr in qs:
            sel = p_b >= thr
            if sel.sum() < min_trades:
                continue
            pf = _profit_factor(R_b[sel])
            if pf >= min_pf:
                chosen = float(thr)
                chosen_diag = {"n_admit": int(sel.sum()), "pf": pf,
                               "mean_R": float(R_b[sel].mean())}
                break  # smallest threshold meeting the bar
        cal.thresholds[int(r)] = chosen
        cal.diag[int(r)] = chosen_diag or {"n_admit": 0, "pf": 0.0, "mean_R": 0.0,
                                           "reason": "no threshold met PF bar"}
    return cal
