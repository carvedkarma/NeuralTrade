"""
Self-supervised pretraining for the V11 trunk.

Per locked contract:
    mask_ratio  = 0.15  (per-bar feature mask — the ENTIRE bar's feature
                        vector is replaced by a learnable [MASK] embedding
                        with probability 0.15)
    epochs      = 20
    batch       = 128
    lr          = 3e-4 with cosine decay
    optimizer   = AdamW (β=0.9/0.95, wd=0.01)

Loss: per-feature MSE at masked positions only.

The trunk's classification head is unused during pretraining; only the
reconstruction head is trained alongside the trunk.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gpu_trainer_v11.models.causal_transformer import CausalTransformer, V11ModelConfig


def _make_loader(X: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    t = torch.from_numpy(X.astype(np.float32))
    return DataLoader(TensorDataset(t), batch_size=batch, shuffle=shuffle, drop_last=False)


def pretrain(
    cfg: V11ModelConfig,
    X_seq: np.ndarray,             # [N, T, F] unsupervised sequences (eligible OR all bars)
    epochs: int = 20,
    batch: int = 128,
    base_lr: float = 3e-4,
    mask_ratio: float = 0.15,
    device: str | None = None,
    seed: int = 17,
) -> CausalTransformer:
    """Returns a pretrained CausalTransformer in eval-ready state."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    model = CausalTransformer(cfg).to(device)
    mask_token = nn.Parameter(torch.zeros(cfg.n_features, device=device))
    nn.init.normal_(mask_token, std=0.02)

    params = list(model.parameters()) + [mask_token]
    opt = torch.optim.AdamW(params, lr=base_lr, betas=(0.9, 0.95), weight_decay=0.01)

    loader = _make_loader(X_seq, batch, shuffle=True)
    total_steps = max(1, epochs * len(loader))
    step = 0

    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        ep_n = 0
        for (xb,) in loader:
            xb = xb.to(device)              # [B, T, F]
            B, T, F = xb.shape
            mask = torch.rand(B, T, device=device) < mask_ratio   # [B, T]
            xb_in = torch.where(mask.unsqueeze(-1), mask_token, xb)
            recon = model.forward_pretrain(xb_in)                  # [B, T, F]
            target = xb
            err = (recon - target) ** 2
            err = err.mean(dim=-1)                                  # [B, T]
            denom = mask.sum().clamp(min=1.0)
            loss = (err * mask).sum() / denom

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            # cosine LR
            progress = step / total_steps
            lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.step()
            step += 1
            ep_loss += float(loss.item()) * B
            ep_n += B
        print(f"  pretrain ep {ep+1}/{epochs}  loss={ep_loss/max(ep_n,1):.5f}")
    model.eval()
    return model
