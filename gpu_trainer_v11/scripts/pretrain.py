"""
Pooled masked-feature pretraining of the V11 trunk.

Per locked contract (README §"Training (frozen)"):
    Pretrain symbols = pooled BTC + ETH + SOL,
                       fixed window 2021-02-01 → 2023-02-01 only.
    Mask ratio       = 0.15
    Epochs           = 20
    Batch            = 128
    LR               = 3e-4 cosine, AdamW (β=0.9/0.95, wd=0.01)

The pretrain window is FIXED and chosen so that it lies fully inside
or before every walk-forward TRAIN window (the first WF train starts at
2021-02 and the first WF test starts at 2023-02). It therefore never
leaks into any walk-forward TEST set.

Outputs:
    reports/pretrained_trunk.pt     — trunk state_dict + cfg + feature columns
    reports/pretrain_meta.json      — symbols pooled, window, n_sequences

run_walk_forward loads this checkpoint and starts every fold's bagged
finetune from the pooled trunk weights.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gpu_trainer_v11.features.compose import compute_features
from gpu_trainer_v11.models.causal_transformer import V11ModelConfig
from gpu_trainer_v11.models.pretrain import pretrain

REPO_ROOT = Path(__file__).resolve().parents[2]
DOLLAR_DIR = REPO_ROOT / "gpu_trainer_v11" / "data_cache_dollar"
REPORT_DIR = REPO_ROOT / "gpu_trainer_v11" / "reports"

PRETRAIN_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
WIN_START = pd.Timestamp("2021-02-01", tz="UTC").value // 10**6
WIN_END = pd.Timestamp("2023-02-01", tz="UTC").value // 10**6
SEQ_LEN = 128
LOCKED_EPOCHS = 20
LOCKED_BATCH = 128
LOCKED_MASK_RATIO = 0.15
LOCKED_LR = 3e-4
STRIDE = 4   # subsample sequences within window


def _load_btc() -> pd.DataFrame | None:
    p = DOLLAR_DIR / "BTCUSDT_dollar.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _build_pool() -> tuple[np.ndarray, list[str]]:
    """Returns stacked sequences (N, seq_len, n_features) and the locked
    feature column list. Feature columns are computed independently per
    symbol; all symbols use the same column ordering by construction
    (compute_features is deterministic and column-stable)."""
    btc = _load_btc()
    seqs: list[np.ndarray] = []
    cols_ref: list[str] | None = None
    for sym in PRETRAIN_SYMBOLS:
        p = DOLLAR_DIR / f"{sym}_dollar.parquet"
        if not p.exists():
            print(f"  {sym}: missing dollar bars, skipping")
            continue
        bars = pd.read_parquet(p)
        bars = bars[(bars["timestamp"] >= WIN_START) & (bars["timestamp"] < WIN_END)] \
                  .reset_index(drop=True)
        if len(bars) < SEQ_LEN + 50:
            print(f"  {sym}: too few bars in window, skipping")
            continue
        bundle = compute_features(bars, btc, sym)
        feats = bundle.features
        if cols_ref is None:
            cols_ref = list(feats.columns)
        feats = feats[cols_ref]                       # enforce column order
        feat_mat = feats.to_numpy(dtype=np.float32)
        idx = np.arange(SEQ_LEN - 1, len(feat_mat), STRIDE)
        sym_seqs = np.stack([feat_mat[i - SEQ_LEN + 1: i + 1] for i in idx])
        print(f"  {sym}: window {len(bars):>6} bars -> {len(sym_seqs):>5} sequences")
        seqs.append(sym_seqs)
    if not seqs or cols_ref is None:
        raise SystemExit("FATAL: no pretrain data assembled.")
    X = np.concatenate(seqs, axis=0)
    print(f"  pool: {len(X)} sequences  ({X.shape[1]} bars × {X.shape[2]} features)")
    return X, cols_ref


def main():
    ap = argparse.ArgumentParser(description="Pooled masked-feature pretrain. Hyperparameters LOCKED.")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", type=Path, default=REPORT_DIR / "pretrained_trunk.pt")
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    X, feature_cols = _build_pool()
    cfg = V11ModelConfig(n_features=X.shape[2], seq_len=SEQ_LEN)
    print(f"  pretrain: epochs={LOCKED_EPOCHS}  batch={LOCKED_BATCH}  "
          f"mask_ratio={LOCKED_MASK_RATIO}  lr={LOCKED_LR}")
    trunk = pretrain(cfg, X, epochs=LOCKED_EPOCHS, batch=LOCKED_BATCH,
                     mask_ratio=LOCKED_MASK_RATIO, base_lr=LOCKED_LR, seed=args.seed)
    torch.save({
        "trunk_state_dict": trunk.state_dict(),
        "cfg": {"n_features": cfg.n_features, "seq_len": cfg.seq_len},
        "feature_cols": feature_cols,
        "pool": PRETRAIN_SYMBOLS,
        "window_ms": [WIN_START, WIN_END],
        "n_sequences": int(len(X)),
        "saved_at": datetime.utcnow().isoformat(),
    }, args.out)
    (REPORT_DIR / "pretrain_meta.json").write_text(json.dumps({
        "pool": PRETRAIN_SYMBOLS,
        "window": ["2021-02-01", "2023-02-01"],
        "n_sequences": int(len(X)),
        "n_features": int(X.shape[2]),
        "checkpoint": str(args.out),
    }, indent=2))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
