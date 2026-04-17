"""
Adversarial-validation drift detector.

Concatenate train and test feature matrices; label train=0, test=1; train
a small classifier (gradient-boosted trees on a sample) and report ROC AUC
on a held-out split. AUC near 0.5 means no drift; >> 0.5 means the model
can tell train and test apart by features alone — a red flag for the
walk-forward conclusion.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def adversarial_auc(
    X_train: np.ndarray,
    X_test: np.ndarray,
    sample_max: int = 20000,
    seed: int = 17,
) -> float:
    if len(X_train) == 0 or len(X_test) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    n_tr = min(sample_max, len(X_train))
    n_te = min(sample_max, len(X_test))
    Xtr = X_train[rng.choice(len(X_train), n_tr, replace=False)]
    Xte = X_test[rng.choice(len(X_test), n_te, replace=False)]
    X = np.vstack([Xtr, Xte])
    y = np.concatenate([np.zeros(n_tr), np.ones(n_te)])

    try:
        import xgboost as xgb
    except Exception:
        return float("nan")

    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, n_jobs=2,
        eval_metric="auc", random_state=seed, verbosity=0,
    )
    clf.fit(Xa, ya)
    p = clf.predict_proba(Xb)[:, 1]
    try:
        return float(roc_auc_score(yb, p))
    except Exception:
        return float("nan")
