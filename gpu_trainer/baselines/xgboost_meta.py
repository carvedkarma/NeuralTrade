"""
XGBoost meta-label classifier with isotonic calibration and the
locked Phase-1 threshold-selection rule.

Frozen hyperparameters (do NOT tune):
    n_estimators           = 500
    max_depth              = 6
    learning_rate          = 0.05
    subsample              = 0.8
    colsample_bytree       = 0.8
    min_child_weight       = 3
    reg_alpha              = 0.1
    reg_lambda             = 1.0
    objective              = binary:logistic
    eval_metric            = logloss
    early_stopping_rounds  = 50
    random_state           = 42

Threshold rule (per fold):
  1. Hold out last 20 % of the train window as a calibration set.
  2. Fit isotonic regression on calibration p_meta -> meta_label.
  3. T_fold = smallest calibrated probability on the calibration set
     such that #cal_trades >= 100 AND cal_PF >= 1.4.
  4. If no T satisfies both, fold contributes ZERO trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import xgboost as xgb
except ImportError as _xgb_err:  # pragma: no cover
    raise ImportError(
        "xgboost is required for V10 Phase 1; install with `pip install xgboost>=2.0.0`"
    ) from _xgb_err

from sklearn.isotonic import IsotonicRegression


XGB_PARAMS_LOCKED = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)
EARLY_STOPPING_ROUNDS = 50

CAL_FRACTION = 0.20
CAL_MIN_TRADES = 100
CAL_MIN_PF = 1.4


@dataclass
class FoldThreshold:
    threshold: Optional[float]
    cal_trades: int
    cal_pf: float
    cal_expectancy: float
    reason: str = ""


@dataclass
class FoldModel:
    booster: object
    isotonic: IsotonicRegression
    threshold_info: FoldThreshold
    best_iteration: int
    feature_names: list = field(default_factory=list)


def _profit_factor(R: np.ndarray) -> float:
    pos = R[R > 0].sum()
    neg = -R[R < 0].sum()
    if neg <= 1e-12:
        return float("inf") if pos > 0 else 0.0
    return float(pos / neg)


def _select_threshold(cal_p: np.ndarray, cal_R: np.ndarray) -> FoldThreshold:
    """Smallest calibrated probability T s.t. n_trades>=CAL_MIN_TRADES and PF>=CAL_MIN_PF.

    Iterates over UNIQUE sorted probability values (isotonic produces many
    ties; without grouping, "first index >= sorted_p[k]" is not equivalent
    to "all indices with p >= T"). For each unique T we evaluate the exact
    admitted set { i : cal_p[i] >= T } and check both constraints.

    The smallest T admits the most trades; we walk T upward (fewer trades)
    and return the first T satisfying both. By monotonicity, if no
    candidate T passes, no T does.
    """
    if len(cal_p) == 0:
        return FoldThreshold(None, 0, 0.0, 0.0, "empty calibration set")

    unique_T = np.unique(cal_p)  # ascending
    for T in unique_T:
        sel = cal_p >= T
        admitted = int(sel.sum())
        if admitted < CAL_MIN_TRADES:
            break  # higher T can only admit fewer trades
        Rs = cal_R[sel]
        pos = float(Rs[Rs > 0].sum())
        neg = float(-Rs[Rs < 0].sum())
        if neg <= 1e-12:
            pf = float("inf") if pos > 0 else 0.0
        else:
            pf = pos / neg
        if pf >= CAL_MIN_PF:
            return FoldThreshold(
                threshold=float(T),
                cal_trades=admitted,
                cal_pf=float(pf),
                cal_expectancy=float(Rs.mean()),
                reason="ok",
            )
    return FoldThreshold(None, 0, 0.0, 0.0,
                         "no T satisfies both n>=100 and PF>=1.4")


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    R_train: np.ndarray,
    feature_names: list,
) -> FoldModel:
    """Train XGBoost on the train fold using the locked hyperparams.

    Internally splits off the last CAL_FRACTION of the training data
    (preserving temporal order — caller must pass rows in time order)
    as the validation/calibration set.
    """
    n = len(X_train)
    if n < 500:
        raise ValueError(f"train fold too small: {n} rows")

    cut = int(n * (1.0 - CAL_FRACTION))
    X_tr, X_cal = X_train[:cut], X_train[cut:]
    y_tr, y_cal = y_train[:cut], y_train[cut:]
    R_cal = R_train[cut:]

    pos_w = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())

    booster = xgb.XGBClassifier(
        scale_pos_weight=float(pos_w),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **XGB_PARAMS_LOCKED,
    )
    booster.fit(
        X_tr, y_tr,
        eval_set=[(X_cal, y_cal)],
        verbose=False,
    )
    best_iter = int(getattr(booster, "best_iteration", booster.n_estimators - 1) or 0)

    cal_raw = booster.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(cal_raw, y_cal)
    cal_calibrated = iso.predict(cal_raw)

    thr = _select_threshold(cal_calibrated, R_cal)

    return FoldModel(
        booster=booster,
        isotonic=iso,
        threshold_info=thr,
        best_iteration=best_iter,
        feature_names=list(feature_names),
    )


def predict_calibrated(model: FoldModel, X: np.ndarray) -> np.ndarray:
    raw = model.booster.predict_proba(X)[:, 1]
    return model.isotonic.predict(raw)
