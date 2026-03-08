"""
V5 Forecaster: Separating Market Forecasting from Decision Layer

Architecture:
- Shared trunk (ResidualBlock-based MLP, same as EnhancedMultiHeadMLP)
- ret_dist_head: predicts Normal(mu, sigma) for ret_h distribution
- mfe_head: regression for max favorable excursion (Huber)
- mae_head: regression for max adverse excursion (Huber)
- action_head: 3-class logits {HOLD=0, LONG=1, SHORT=2}
- barrier_head: N-class logits for preset selection (optional)
- regime_head: 3-class {chop, trend, highvol} (optional)

All outputs produced in one forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
import math


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3, use_layer_norm: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if use_layer_norm else nn.Identity()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + self.skip(x)


@dataclass
class V5ForecasterConfig:
    input_dim: int = 63
    hidden_dims: list = None
    dropout: float = 0.3
    use_layer_norm: bool = True
    use_residual: bool = True
    n_barrier_presets: int = 0
    enable_regime_head: bool = False
    n_symbols: int = 1
    symbol_embed_dim: int = 8

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128, 64]


class V5Forecaster(nn.Module):
    """V5 multi-head forecaster separating market state from decision."""

    def __init__(self, config: V5ForecasterConfig):
        super().__init__()
        self.config = config
        self.name = "V5Forecaster"

        self.input_dim = config.input_dim
        self.created_at = datetime.now().isoformat()
        self.training_history = []
        self.best_val_loss = float('inf')
        self.epochs_trained = 0

        if config.n_symbols > 1:
            self.symbol_embedding = nn.Embedding(config.n_symbols, config.symbol_embed_dim)
            trunk_input_dim = config.input_dim + config.symbol_embed_dim
        else:
            self.symbol_embedding = None
            trunk_input_dim = config.input_dim

        trunk_layers = []
        prev_dim = trunk_input_dim
        for hidden_dim in config.hidden_dims:
            if config.use_residual:
                trunk_layers.append(ResidualBlock(
                    prev_dim, hidden_dim,
                    dropout=config.dropout,
                    use_layer_norm=config.use_layer_norm
                ))
            else:
                trunk_layers.append(nn.Linear(prev_dim, hidden_dim))
                if config.use_layer_norm:
                    trunk_layers.append(nn.LayerNorm(hidden_dim))
                trunk_layers.append(nn.GELU())
                trunk_layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim

        self.trunk = nn.Sequential(*trunk_layers)
        self.trunk_dim = prev_dim

        self.ret_dist_head = nn.Sequential(
            nn.Linear(self.trunk_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

        self.mfe_head = nn.Sequential(
            nn.Linear(self.trunk_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

        self.mae_head = nn.Sequential(
            nn.Linear(self.trunk_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

        self.action_head = nn.Sequential(
            nn.Linear(self.trunk_dim, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(48, 3)
        )

        if config.n_barrier_presets > 1:
            self.barrier_head = nn.Sequential(
                nn.Linear(self.trunk_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(32, config.n_barrier_presets)
            )
        else:
            self.barrier_head = None

        if config.enable_regime_head:
            self.regime_head = nn.Sequential(
                nn.Linear(self.trunk_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(32, 3)
            )
        else:
            self.regime_head = None

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, symbol_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if x.dim() == 3:
            x = x[:, -1, :]

        if self.symbol_embedding is not None and symbol_ids is not None:
            sym_emb = self.symbol_embedding(symbol_ids)
            x = torch.cat([x, sym_emb], dim=-1)

        features = self.trunk(x)

        ret_raw = self.ret_dist_head(features)
        ret_mu = torch.clamp(ret_raw[:, 0:1], -10.0, 10.0)
        ret_log_sigma = torch.clamp(ret_raw[:, 1:2], -8.0, 2.0)
        ret_sigma = torch.exp(ret_log_sigma)

        mfe_pred = self.mfe_head(features)
        mfe_pred = torch.clamp(mfe_pred, 0.0, 20.0)

        mae_pred = self.mae_head(features)
        mae_pred = torch.clamp(mae_pred, 0.0, 20.0)

        action_logits = self.action_head(features)
        action_logits = torch.clamp(action_logits, -10.0, 10.0)

        result = {
            'ret_mu': ret_mu,
            'ret_log_sigma': ret_log_sigma,
            'ret_sigma': ret_sigma,
            'mfe': mfe_pred,
            'mae': mae_pred,
            'action_logits': action_logits,
        }

        if self.barrier_head is not None:
            barrier_logits = self.barrier_head(features)
            barrier_logits = torch.clamp(barrier_logits, -10.0, 10.0)
            result['barrier_logits'] = barrier_logits

        if self.regime_head is not None:
            regime_logits = self.regime_head(features)
            regime_logits = torch.clamp(regime_logits, -10.0, 10.0)
            result['regime_logits'] = regime_logits

        return result

    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
