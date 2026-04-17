"""
Causal Transformer encoder for V11.

Locked architecture (do NOT change after pre-flight):
    n_layers   = 4
    d_model    = 128
    n_heads    = 4
    d_ff       = 256
    seq_len    = 128
    pos enc    = sinusoidal
    causal mask= strict lower-triangular
    pretraining head : Linear(d_model, n_features) at masked positions
    classification head: Linear(d_model, 1) at the LAST position only

Inputs are real-valued feature vectors per bar; the input projection is
a learned Linear(n_features, d_model). No tokenization.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class V11ModelConfig:
    n_features: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    seq_len: int = 128
    dropout: float = 0.1


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.pe[:, : x.size(1)]


def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
    # True = "do not attend" (PyTorch convention for src_mask boolean)
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


class CausalTransformer(nn.Module):
    def __init__(self, cfg: V11ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.n_features, cfg.d_model)
        self.pe = SinusoidalPositionalEncoding(cfg.d_model, max_len=cfg.seq_len * 4)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.recon_head = nn.Linear(cfg.d_model, cfg.n_features)  # pretrain
        self.cls_head = nn.Linear(cfg.d_model, 1)                 # finetune

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder; return per-position hidden states."""
        h = self.input_proj(x)
        h = self.pe(h)
        mask = _causal_mask(h.size(1), h.device)
        h = self.encoder(h, mask=mask, is_causal=False)  # we pass an explicit mask
        return self.norm(h)

    def forward_pretrain(self, x: torch.Tensor) -> torch.Tensor:
        """Return reconstructions at every position. Loss is computed only at masked positions externally."""
        h = self.trunk(x)
        return self.recon_head(h)

    def forward_classify(self, x: torch.Tensor) -> torch.Tensor:
        """Return logit at the LAST position only.  Output: [B]"""
        h = self.trunk(x)
        return self.cls_head(h[:, -1, :]).squeeze(-1)
