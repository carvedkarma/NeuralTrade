"""
Transfer-entropy proxy for causal feature selection.

We approximate transfer entropy with discretized lagged mutual
information:
    TE(X -> Y) ≈ I(Y_t ; X_{t-1} | Y_{t-1})

For meta-label selection we treat Y as the binary meta-label (a single
discrete RV with no auto-history that matters at scale 1), so the
condition on Y_{t-1} is dropped and we score:
    score(feature) = I(meta_label_t ; feature_{t-1})

Both meta_label and the feature are discretized to `n_bins` equal-
frequency buckets on the train portion only. The score is a single
non-negative float per feature; we keep the top-K.

Strict causality: only train data is used to compute the ranking, and
the lag ensures the feature observation precedes the label.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score


def _discretize(x: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.quantile(x[np.isfinite(x)], np.linspace(0, 1, n_bins + 1))
    qs[0] = -np.inf
    qs[-1] = np.inf
    qs = np.unique(qs)
    if len(qs) <= 2:
        return np.zeros_like(x, dtype=np.int32)
    return np.searchsorted(qs[1:-1], x).astype(np.int32)


@dataclass
class FeatureRanking:
    feature_names: list[str]
    scores: np.ndarray           # one MI score per feature
    keep_mask: np.ndarray        # boolean mask of length n_features


def rank_features(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_bins: int = 8,
    lag_bars: int = 1,
) -> pd.DataFrame:
    """Rank features by lagged MI to the binary meta-label.

    Returns a DataFrame with columns: feature, mi_score, rank.
    """
    if len(X_train) <= lag_bars + 10:
        raise ValueError("not enough train rows to compute MI")
    y_disc = y_train[lag_bars:].astype(np.int32)

    rows = []
    for col in X_train.columns:
        x = X_train[col].to_numpy()
        x_lagged = x[:-lag_bars] if lag_bars > 0 else x
        if len(x_lagged) != len(y_disc):
            continue
        finite = np.isfinite(x_lagged)
        if finite.sum() < 50:
            rows.append((col, 0.0))
            continue
        x_d = _discretize(x_lagged[finite], n_bins)
        y_d = y_disc[finite]
        try:
            score = float(mutual_info_score(y_d, x_d))
        except Exception:
            score = 0.0
        rows.append((col, score))

    df = pd.DataFrame(rows, columns=["feature", "mi_score"]).sort_values(
        "mi_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def select_top_k(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    k: int = 64,
    n_bins: int = 8,
    lag_bars: int = 1,
) -> FeatureRanking:
    ranking_df = rank_features(X_train, y_train, n_bins=n_bins, lag_bars=lag_bars)
    keep = set(ranking_df.head(k)["feature"].tolist())
    keep_mask = np.array([c in keep for c in X_train.columns], dtype=bool)
    return FeatureRanking(
        feature_names=list(X_train.columns),
        scores=ranking_df.set_index("feature").reindex(X_train.columns)["mi_score"].to_numpy(),
        keep_mask=keep_mask,
    )
