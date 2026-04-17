"""
Per-symbol diversification probe.

Score the (already trained, frozen) ensemble of bagged finetunes on
the post-2024 windows of every symbol NOT in the training pool. Apply
the same per-fold conformal thresholds. Reports per-symbol PF, trades,
expectancy.

This is read-only — no retraining, no threshold re-search.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from gpu_trainer_v11.features.compose import compute_features
from gpu_trainer_v11.labels.horizon_conditional import per_bar_barrier_mult
from gpu_trainer_v11.labels.meta_label_v11 import compute_meta_labels_v11
from gpu_trainer_v11.labels.primary_rules import primary_rule
from gpu_trainer_v11.models.causal_transformer import CausalTransformer
from gpu_trainer_v11.models.conformal import MondrianCalibrator
from gpu_trainer_v11.models.datasets import build_sequences
from gpu_trainer_v11.models.finetune import predict_proba_bagged

REPO_ROOT = Path(__file__).resolve().parents[2]
DOLLAR_DIR = REPO_ROOT / "gpu_trainer_v11" / "data_cache_dollar"


@dataclass
class SymbolProbeRow:
    symbol: str
    n_eligible: int
    n_trades: int
    pf: float
    expectancy_R: float
    win_rate: float


def probe_symbols(
    symbols: list[str],
    rule: str,
    horizon_bars: int,
    feature_cols: list[str],          # FULL bundle column order (dim must match trunk)
    keep_mask: np.ndarray,            # per-fold transfer-entropy top-K input gate
    bagged_models: list[CausalTransformer],
    conformal: MondrianCalibrator,
    btc_bars: pd.DataFrame | None,
    seq_len: int = 128,
    start_ts_ms: int | None = None,   # e.g. 2024-01-01
) -> list[SymbolProbeRow]:
    """Apply the trained ensemble + conformal threshold to OOS symbols
    UNDER THE EXACT SAME INPUT REGIME used in training/test:
      - same feature columns in the same order (so dim matches the trunk)
      - same per-fold zero-mask on non-selected columns (transfer-entropy gate)
    Without the gate, OOS scores are not comparable to fold scores and the
    contract's beat-check is not auditable."""
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if keep_mask.shape[0] != len(feature_cols):
        raise ValueError(
            f"keep_mask len {keep_mask.shape[0]} != feature_cols len {len(feature_cols)}; "
            f"diversification gate is unaligned with training input.")
    feat_gate = keep_mask.astype(np.float32)[None, :]
    rows: list[SymbolProbeRow] = []
    for sym in symbols:
        p = DOLLAR_DIR / f"{sym}_dollar.parquet"
        if not p.exists():
            continue
        bars = pd.read_parquet(p)
        if start_ts_ms is not None:
            bars = bars[bars["timestamp"] >= start_ts_ms].reset_index(drop=True)
        if len(bars) < seq_len + 200:
            rows.append(SymbolProbeRow(sym, 0, 0, 0.0, 0.0, 0.0))
            continue
        bundle = compute_features(bars, btc_bars, sym)
        # Project onto SAME feature columns the model trained on; missing -> 0
        for col in feature_cols:
            if col not in bundle.features.columns:
                bundle.features[col] = 0.0
        feats_full = bundle.features[feature_cols]
        bm = per_bar_barrier_mult(bundle.features["atr_pct_bucket"])
        primary = primary_rule(bars, bundle.side_data, rule)
        meta = compute_meta_labels_v11(bars, primary, horizon_bars, bm)
        regime = bundle.features["atr_pct_bucket"].to_numpy(dtype=np.int8)
        feat_mat = feats_full.to_numpy(dtype=np.float32) * feat_gate

        n_elig = int(meta["eligible"].sum())
        if n_elig == 0:
            rows.append(SymbolProbeRow(sym, 0, 0, 0.0, 0.0, 0.0))
            continue
        weights = np.ones(n_elig, dtype=np.float32)
        seqs = build_sequences(feat_mat, meta, weights, regime, seq_len=seq_len)
        if len(seqs.X) == 0:
            rows.append(SymbolProbeRow(sym, n_elig, 0, 0.0, 0.0, 0.0))
            continue
        p_test = predict_proba_bagged(bagged_models, seqs.X)
        admit = conformal.admit(p_test, seqs.regime)
        R = seqs.R[admit]
        n_tr = int(admit.sum())
        if n_tr == 0:
            rows.append(SymbolProbeRow(sym, n_elig, 0, 0.0, 0.0, 0.0))
            continue
        pos = R[R > 0].sum(); neg = -R[R < 0].sum()
        pf = float("inf") if neg <= 0 and pos > 0 else (pos / neg if neg > 0 else 0.0)
        rows.append(SymbolProbeRow(
            symbol=sym, n_eligible=n_elig, n_trades=n_tr,
            pf=float(pf), expectancy_R=float(R.mean()),
            win_rate=float((R > 0).mean()),
        ))
    return rows
