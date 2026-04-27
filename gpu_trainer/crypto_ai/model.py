from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        length = x.size(1)
        return x + self.pe[:length].unsqueeze(0)


class MultiSymbolTransformer(nn.Module):
    """Transformer for multi-symbol sequence classification and regression."""

    def __init__(
        self,
        num_symbols: int,
        num_features: int,
        d_model: int = 192,
        nhead_time: int = 6,
        nhead_symbol: int = 4,
        num_time_layers: int = 3,
        num_symbol_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_symbols = num_symbols
        self.num_features = num_features
        self.d_model = d_model

        self.feature_proj = nn.Linear(num_features, d_model)
        self.symbol_embedding = nn.Parameter(torch.zeros(1, num_symbols, d_model))
        nn.init.normal_(self.symbol_embedding, std=0.02)

        self.time_pos_encoding = PositionalEncoding(d_model=d_model)

        time_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead_time,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.time_encoder = nn.TransformerEncoder(time_encoder_layer, num_layers=num_time_layers)

        symbol_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead_symbol,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.symbol_encoder = nn.TransformerEncoder(symbol_encoder_layer, num_layers=num_symbol_layers)

        self.norm = nn.LayerNorm(d_model)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        self.return_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, symbols, features]
        Returns:
            logits: [batch, symbols, 3]
            horizon_returns: [batch, symbols]
        """
        batch, seq_len, symbols, _ = x.shape
        if symbols != self.num_symbols:
            raise ValueError(f"Expected {self.num_symbols} symbols, got {symbols}")

        # Encode each symbol over time independently.
        h = self.feature_proj(x)  # [B, L, S, D]
        h = h.permute(0, 2, 1, 3).contiguous()  # [B, S, L, D]
        h = h.view(batch * symbols, seq_len, self.d_model)
        h = self.time_pos_encoding(h)
        h = self.time_encoder(h)

        # Use the last token as the time-context summary per symbol.
        h_last = h[:, -1, :].view(batch, symbols, self.d_model)
        h_last = h_last + self.symbol_embedding
        h_last = self.symbol_encoder(h_last)
        h_last = self.norm(h_last)

        logits = self.cls_head(h_last)
        horizon_returns = self.return_head(h_last).squeeze(-1)
        return logits, horizon_returns

