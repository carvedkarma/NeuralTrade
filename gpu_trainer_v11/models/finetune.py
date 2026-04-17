"""
Bagged supervised finetuning of the V11 trunk.

Per locked contract:
    bagging   = N=5 independent finetune runs per specialist
    epochs    = 30
    batch     = 64
    lr        = 1e-4 cosine; head lr 5e-4
    loss      = BCEWithLogits, sample-uniqueness-weighted
    sampler   = WeightedRandomSampler so positives & negatives appear roughly balanced per batch
    early stop= validation logloss, patience 5
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from gpu_trainer_v11.models.causal_transformer import CausalTransformer, V11ModelConfig


def _logloss(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y.float(), reduction="none")
    if w is None:
        return bce.mean()
    w = w / w.mean().clamp(min=1e-8)
    return (bce * w).mean()


def _balanced_sampler(y: np.ndarray, w: np.ndarray) -> WeightedRandomSampler:
    pos_mask = y == 1
    n_pos = max(1, pos_mask.sum())
    n_neg = max(1, (~pos_mask).sum())
    sw = np.where(pos_mask, 1.0 / n_pos, 1.0 / n_neg) * w
    sw = sw / sw.sum()
    return WeightedRandomSampler(weights=torch.from_numpy(sw.astype(np.float64)),
                                 num_samples=len(sw), replacement=True)


def finetune_one(
    pretrained: CausalTransformer,
    X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray, w_val: np.ndarray,
    epochs: int = 30,
    batch: int = 64,
    base_lr: float = 1e-4,
    head_lr: float = 5e-4,
    patience: int = 5,
    seed: int = 17,
    device: str | None = None,
) -> CausalTransformer:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = copy.deepcopy(pretrained).to(device)
    head_params = list(model.cls_head.parameters())
    head_ids = {id(p) for p in head_params}
    trunk_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([
        {"params": trunk_params, "lr": base_lr},
        {"params": head_params, "lr": head_lr},
    ], betas=(0.9, 0.95), weight_decay=0.01)

    sampler = _balanced_sampler(y_train, w_train)
    train_ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
        torch.from_numpy(w_train.astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)),
        torch.from_numpy(y_val.astype(np.float32)),
        torch.from_numpy(w_val.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch * 2, shuffle=False, drop_last=False)

    total_steps = max(1, epochs * len(train_loader))
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    step = 0
    for ep in range(epochs):
        model.train()
        for xb, yb, wb in train_loader:
            xb = xb.to(device); yb = yb.to(device); wb = wb.to(device)
            logits = model.forward_classify(xb)
            loss = _logloss(logits, yb, wb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            progress = step / total_steps
            for g, base in zip(opt.param_groups, [base_lr, head_lr]):
                g["lr"] = base * 0.5 * (1.0 + math.cos(math.pi * progress))
            opt.step()
            step += 1

        model.eval()
        with torch.no_grad():
            tot_loss = 0.0; tot_n = 0
            for xb, yb, wb in val_loader:
                xb = xb.to(device); yb = yb.to(device); wb = wb.to(device)
                logits = model.forward_classify(xb)
                l = _logloss(logits, yb, wb)
                tot_loss += float(l.item()) * xb.size(0)
                tot_n += xb.size(0)
            val_loss = tot_loss / max(tot_n, 1)
        print(f"  ft ep {ep+1}/{epochs}  val_logloss={val_loss:.5f}  best={best_val:.5f}  bad={bad}")
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  ft early stop at epoch {ep+1}")
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def finetune_bagged(
    pretrained: CausalTransformer,
    X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray, w_val: np.ndarray,
    n_bag: int = 5,
    base_seed: int = 17,
    **kwargs,
) -> list[CausalTransformer]:
    """Train N independent finetuned models. Each sees a bootstrap-resampled
    train set; the val set is shared so early stopping is comparable."""
    rng = np.random.default_rng(base_seed)
    n = len(X_train)
    models = []
    for i in range(n_bag):
        idx = rng.integers(0, n, size=n)
        Xt = X_train[idx]; yt = y_train[idx]; wt = w_train[idx]
        print(f"-- bag {i+1}/{n_bag} (seed={base_seed + i}) --")
        m = finetune_one(pretrained, Xt, yt, wt, X_val, y_val, w_val,
                         seed=base_seed + i, **kwargs)
        models.append(m)
    return models


@torch.no_grad()
def predict_proba_bagged(models: list[CausalTransformer],
                         X: np.ndarray,
                         batch: int = 256,
                         device: str | None = None) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(X) == 0:
        return np.zeros(0, dtype=np.float64)
    avg = np.zeros(len(X), dtype=np.float64)
    for m in models:
        m.eval(); m.to(device)
        out = []
        for s in range(0, len(X), batch):
            xb = torch.from_numpy(X[s:s + batch].astype(np.float32)).to(device)
            logits = m.forward_classify(xb)
            out.append(torch.sigmoid(logits).cpu().numpy())
        avg += np.concatenate(out, axis=0)
    return avg / len(models)
