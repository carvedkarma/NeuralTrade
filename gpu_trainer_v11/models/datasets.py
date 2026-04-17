"""
Sequence dataset assembly for the V11 Transformer.

Given a feature matrix X [N, F] aligned to bars and a meta-label table,
build (sequence, label, weight) triples where each sequence is the last
`seq_len` bars ending at (and including) the entry bar of an eligible
label.

Strict causality: sequence ending at bar i contains bars i-seq_len+1 ... i;
the label is the meta-label produced by the triple-barrier resolution
of an entry at bar i, looking forward only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class V11Sequences:
    X: np.ndarray         # [N_eligible, seq_len, F]   float32
    y: np.ndarray         # [N_eligible]                int8
    w: np.ndarray         # [N_eligible]                float32
    R: np.ndarray         # [N_eligible]                float32
    entry_idx: np.ndarray # [N_eligible]                int64
    exit_idx: np.ndarray  # [N_eligible]                int64
    regime: np.ndarray    # [N_eligible]                int8 (0/1/2 ATR-pct bucket)


def build_sequences(
    feature_matrix: np.ndarray,         # [N, F] float32
    meta_df,                            # pandas DataFrame with eligible / meta_label / R_net / entry_idx / exit_idx
    sample_weights: np.ndarray,         # [N_eligible_in_meta] aligned to eligible rows
    regime_per_bar: np.ndarray,         # [N] int regimes
    seq_len: int = 128,
) -> V11Sequences:
    eligible = meta_df["eligible"].to_numpy()
    n = feature_matrix.shape[0]
    F = feature_matrix.shape[1]
    elig_rows = np.where(eligible)[0]
    # Drop rows whose context window is incomplete
    keep = elig_rows[elig_rows >= (seq_len - 1)]

    if keep.size == 0:
        return V11Sequences(
            X=np.zeros((0, seq_len, F), dtype=np.float32),
            y=np.zeros(0, dtype=np.int8), w=np.zeros(0, dtype=np.float32),
            R=np.zeros(0, dtype=np.float32),
            entry_idx=np.zeros(0, dtype=np.int64),
            exit_idx=np.zeros(0, dtype=np.int64),
            regime=np.zeros(0, dtype=np.int8),
        )

    X_seq = np.zeros((keep.size, seq_len, F), dtype=np.float32)
    for i, idx in enumerate(keep):
        X_seq[i] = feature_matrix[idx - seq_len + 1: idx + 1]

    # sample_weights is aligned to eligible rows in meta_df order; we built a different keep,
    # so map back: eligible_idx_in_meta -> kept index
    elig_to_pos = {int(v): i for i, v in enumerate(elig_rows)}
    pos_for_keep = np.array([elig_to_pos[int(k)] for k in keep], dtype=np.int64)
    w_seq = sample_weights[pos_for_keep].astype(np.float32) if sample_weights.size else np.ones(keep.size, dtype=np.float32)

    y = meta_df["meta_label"].to_numpy()[keep].astype(np.int8)
    R = meta_df["R_net"].to_numpy()[keep].astype(np.float32)
    exit_idx = meta_df["exit_idx"].to_numpy()[keep].astype(np.int64)
    regime = regime_per_bar[keep].astype(np.int8)

    return V11Sequences(X=X_seq, y=y, w=w_seq, R=R,
                        entry_idx=keep.astype(np.int64), exit_idx=exit_idx,
                        regime=regime)
