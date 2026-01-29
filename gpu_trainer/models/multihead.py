"""
Multi-Head Model Architecture for Institutional Trading

Full trading output vector:
1. Classification head → direction_score [-1, 1]
2. Regression head → expected return (μ) and volatility (σ)
3. Quantile head → uncertainty quantiles (q10, q25, q50, q75, q90)
4. Trading head → entry_offset, sl_distance, tp_distance
5. Candle head → future candle deltas (Δclose, Δhigh, Δlow) for N steps

This enables complete trade planning from neural network output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, NamedTuple, List
from dataclasses import dataclass, field
from .base import BaseModel, PositionalEncoding, AttentionBlock


@dataclass
class MultiHeadOutput:
    """Output from multi-head model."""
    # Classification head: direction probabilities
    class_logits: torch.Tensor  # [batch, 3] for SHORT/HOLD/LONG
    
    # Regression head: expected return
    mu: torch.Tensor  # [batch, 1] expected forward return
    
    # Quantile head: return distribution
    quantiles: torch.Tensor  # [batch, 5] for q10, q25, q50, q75, q90
    
    # Optional: uncertainty estimate (std of predictions)
    sigma: Optional[torch.Tensor] = None  # [batch, 1]
    
    # Trading head outputs
    entry_offset: Optional[torch.Tensor] = None  # [batch, 1] entry price offset
    sl_distance: Optional[torch.Tensor] = None   # [batch, 1] stop loss distance
    tp_distance: Optional[torch.Tensor] = None   # [batch, 1] take profit distance
    
    # Candle prediction head outputs
    candle_deltas: Optional[torch.Tensor] = None  # [batch, n_steps, 3] for Δclose, Δhigh, Δlow


class QuantileHead(nn.Module):
    """
    Quantile regression head using monotonic constraint.
    
    Outputs q10, q25, q50, q75, q90 with enforced ordering.
    Uses delta parameterization: each quantile = prev + softplus(delta)
    """
    
    QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        
        self.n_quantiles = len(self.QUANTILES)
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # q50 (median) prediction - can be any value
        self.median_head = nn.Linear(hidden_dim // 2, 1)
        
        # Lower deltas (q50 - q25, q25 - q10) - must be positive
        self.lower_deltas = nn.Linear(hidden_dim // 2, 2)
        
        # Upper deltas (q75 - q50, q90 - q75) - must be positive
        self.upper_deltas = nn.Linear(hidden_dim // 2, 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns quantiles [batch, 5] in order: q10, q25, q50, q75, q90
        """
        h = self.shared(x)
        
        # Median (q50) - unrestricted
        q50 = self.median_head(h)  # [batch, 1]
        
        # Lower deltas - use softplus to ensure positive
        lower_d = F.softplus(self.lower_deltas(h))  # [batch, 2]
        d_25_10 = lower_d[:, 0:1]  # q25 - q10
        d_50_25 = lower_d[:, 1:2]  # q50 - q25
        
        # Upper deltas - use softplus to ensure positive
        upper_d = F.softplus(self.upper_deltas(h))  # [batch, 2]
        d_75_50 = upper_d[:, 0:1]  # q75 - q50
        d_90_75 = upper_d[:, 1:2]  # q90 - q75
        
        # Build quantiles with monotonic ordering
        q25 = q50 - d_50_25
        q10 = q25 - d_25_10
        q75 = q50 + d_75_50
        q90 = q75 + d_90_75
        
        # Stack in order: q10, q25, q50, q75, q90
        quantiles = torch.cat([q10, q25, q50, q75, q90], dim=-1)
        
        return quantiles


class RegressionHead(nn.Module):
    """
    Regression head for expected return (μ) and uncertainty (σ).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Expected return (can be negative or positive)
        self.mu_head = nn.Linear(hidden_dim // 2, 1)
        
        # Uncertainty (must be positive)
        self.sigma_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, sigma) tensors."""
        h = self.shared(x)
        
        mu = self.mu_head(h)
        sigma = F.softplus(self.sigma_head(h)) + 1e-6  # Ensure positive
        
        return mu, sigma


class ClassificationHead(nn.Module):
    """
    Classification head for direction (SHORT/HOLD/LONG).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits [batch, num_classes]."""
        return self.classifier(x)


class TradingHead(nn.Module):
    """
    Trading head for entry/SL/TP distance prediction.
    
    Outputs:
    - entry_offset: Price offset from current (can be small positive/negative)
    - sl_distance: Stop loss distance (always positive, applied directionally)
    - tp_distance: Take profit distance (always positive, applied directionally)
    
    Distances are normalized by current volatility during training.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Entry offset (small, can be + or -)
        self.entry_head = nn.Linear(hidden_dim // 2, 1)
        
        # SL distance (must be positive)
        self.sl_head = nn.Linear(hidden_dim // 2, 1)
        
        # TP distance (must be positive)
        self.tp_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (entry_offset, sl_distance, tp_distance).
        
        Distances are positive, offset can be small +/-.
        """
        h = self.shared(x)
        
        # Entry offset: small value, allow +/-
        entry_offset = torch.tanh(self.entry_head(h)) * 0.01  # Max 1% offset
        
        # SL distance: positive, typically 0.5% - 5%
        sl_distance = F.softplus(self.sl_head(h)) * 0.01 + 0.003  # Min 0.3%, scale to typical
        
        # TP distance: positive, typically 1% - 10%
        tp_distance = F.softplus(self.tp_head(h)) * 0.02 + 0.005  # Min 0.5%, scale to typical
        
        return entry_offset, sl_distance, tp_distance


class CandlePredictionHead(nn.Module):
    """
    Predicts future candle deltas (NOT raw prices).
    
    For each future step, predicts:
    - Δclose: Close price change from current
    - Δhigh: High price change from current  
    - Δlow: Low price change from current
    
    Uses percentage changes for scale invariance.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 n_future_steps: int = 5, dropout: float = 0.1):
        super().__init__()
        
        self.n_steps = n_future_steps
        self.n_outputs_per_step = 3  # Δclose, Δhigh, Δlow
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Separate heads for each future step (more capacity)
        self.step_heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.n_outputs_per_step)
            for _ in range(n_future_steps)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns candle deltas [batch, n_steps, 3].
        
        Output[:, i, 0] = Δclose for step i
        Output[:, i, 1] = Δhigh for step i
        Output[:, i, 2] = Δlow for step i
        
        Values are percentage changes (e.g., 0.01 = 1% move).
        """
        h = self.shared(x)
        
        step_outputs = []
        for step_head in self.step_heads:
            step_pred = step_head(h)  # [batch, 3]
            step_outputs.append(step_pred)
            
        # Stack: [batch, n_steps, 3]
        candle_deltas = torch.stack(step_outputs, dim=1)
        
        # Scale to reasonable range (typically -10% to +10%)
        candle_deltas = torch.tanh(candle_deltas) * 0.10
        
        return candle_deltas


class MultiHeadTransformer(BaseModel):
    """
    Transformer with multi-head output for institutional trading.
    
    Outputs:
    - Classification: direction probabilities
    - Regression: expected return (μ) and uncertainty (σ)
    - Quantiles: q10, q25, q50, q75, q90 for SL/TP derivation
    - Trading: entry_offset, sl_distance, tp_distance
    - Candles: future candle deltas (Δclose, Δhigh, Δlow)
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 200,
        num_classes: int = 3,
        num_quantiles: int = 5,
        n_future_candles: int = 5
    ):
        super().__init__("multihead_transformer", input_dim, num_classes)
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_quantiles = num_quantiles
        self.n_future_candles = n_future_candles
        
        # Shared encoder backbone
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Multi-head outputs (6 heads total)
        self.class_head = ClassificationHead(d_model, d_model // 2, num_classes, dropout)
        self.regression_head = RegressionHead(d_model, d_model // 2, dropout)
        self.quantile_head = QuantileHead(d_model, d_model // 2, dropout)
        self.trading_head = TradingHead(d_model, d_model // 2, dropout)
        self.candle_head = CandlePredictionHead(d_model, d_model // 2, n_future_candles, dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Shared encoder - returns pooled representation."""
        x = self.input_projection(x)
        
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        
        for block in self.attention_blocks:
            x = block(x, mask)
            
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        
        return x
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass - returns class logits for backward compatibility.
        Use forward_multihead() for full multi-head output.
        """
        features = self.encode(x, mask)
        return self.class_head(features)
    
    def forward_multihead(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> MultiHeadOutput:
        """
        Full multi-head forward pass.
        
        Returns MultiHeadOutput with all heads:
        - class_logits: [batch, 3]
        - mu: [batch, 1]
        - sigma: [batch, 1]
        - quantiles: [batch, 5]
        - entry_offset, sl_distance, tp_distance: [batch, 1] each
        - candle_deltas: [batch, n_steps, 3]
        """
        features = self.encode(x, mask)
        
        class_logits = self.class_head(features)
        mu, sigma = self.regression_head(features)
        quantiles = self.quantile_head(features)
        entry_offset, sl_distance, tp_distance = self.trading_head(features)
        candle_deltas = self.candle_head(features)
        
        return MultiHeadOutput(
            class_logits=class_logits,
            mu=mu,
            quantiles=quantiles,
            sigma=sigma,
            entry_offset=entry_offset,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            candle_deltas=candle_deltas
        )
    
    def predict_with_quantiles(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convenience method for inference.
        
        Returns dict with full trading output:
        - probabilities: softmax of class logits
        - direction: argmax of probabilities
        - mu: expected return
        - sigma: uncertainty
        - quantiles: {q10, q25, q50, q75, q90}
        - trading: {entry_offset, sl_distance, tp_distance}
        - candle_deltas: future candle predictions
        """
        self.eval()
        with torch.no_grad():
            output = self.forward_multihead(x)
            
            probs = F.softmax(output.class_logits, dim=-1)
            direction = torch.argmax(probs, dim=-1)
            
            return {
                'probabilities': probs,
                'direction': direction,
                'mu': output.mu,
                'sigma': output.sigma,
                'q10': output.quantiles[:, 0:1],
                'q25': output.quantiles[:, 1:2],
                'q50': output.quantiles[:, 2:3],
                'q75': output.quantiles[:, 3:4],
                'q90': output.quantiles[:, 4:5],
                'entry_offset': output.entry_offset,
                'sl_distance': output.sl_distance,
                'tp_distance': output.tp_distance,
                'candle_deltas': output.candle_deltas,
            }


class MultiHeadLSTM(BaseModel):
    """
    Bidirectional LSTM with multi-head output.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 3,
        num_quantiles: int = 5,
        n_future_candles: int = 5
    ):
        super().__init__("multihead_lstm", input_dim, num_classes)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_quantiles = num_quantiles
        self.n_future_candles = n_future_candles
        
        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output dimension is 2x hidden due to bidirectional
        encoder_dim = hidden_dim * 2
        
        # Multi-head outputs (6 heads total)
        self.class_head = ClassificationHead(encoder_dim, encoder_dim // 2, num_classes, dropout)
        self.regression_head = RegressionHead(encoder_dim, encoder_dim // 2, dropout)
        self.quantile_head = QuantileHead(encoder_dim, encoder_dim // 2, dropout)
        self.trading_head = TradingHead(encoder_dim, encoder_dim // 2, dropout)
        self.candle_head = CandlePredictionHead(encoder_dim, encoder_dim // 2, n_future_candles, dropout)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the final hidden state from LSTM."""
        output, (h_n, c_n) = self.lstm(x)
        
        # Concatenate forward and backward final hidden states
        h_forward = h_n[-2, :, :]  # Last layer forward
        h_backward = h_n[-1, :, :]  # Last layer backward
        features = torch.cat([h_forward, h_backward], dim=-1)
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class logits for backward compatibility."""
        features = self.encode(x)
        return self.class_head(features)
    
    def forward_multihead(self, x: torch.Tensor) -> MultiHeadOutput:
        """Full multi-head forward pass with all trading outputs."""
        features = self.encode(x)
        
        class_logits = self.class_head(features)
        mu, sigma = self.regression_head(features)
        quantiles = self.quantile_head(features)
        entry_offset, sl_distance, tp_distance = self.trading_head(features)
        candle_deltas = self.candle_head(features)
        
        return MultiHeadOutput(
            class_logits=class_logits,
            mu=mu,
            quantiles=quantiles,
            sigma=sigma,
            entry_offset=entry_offset,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            candle_deltas=candle_deltas
        )


class MultiHeadCNN(BaseModel):
    """
    1D CNN (ResNet-style) with multi-head output.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_channels: int = 128,
        num_blocks: int = 4,
        dropout: float = 0.2,
        num_classes: int = 3,
        num_quantiles: int = 5,
        n_future_candles: int = 5
    ):
        super().__init__("multihead_cnn", input_dim, num_classes)
        
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.num_quantiles = num_quantiles
        self.n_future_candles = n_future_candles
        
        # Initial projection
        self.input_conv = nn.Conv1d(input_dim, hidden_channels, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm1d(hidden_channels)
        
        # Residual blocks with increasing channels
        self.blocks = nn.ModuleList()
        in_ch = hidden_channels
        for i in range(num_blocks):
            out_ch = hidden_channels * (2 ** min(i, 2))  # Cap at 4x
            self.blocks.append(self._make_block(in_ch, out_ch, dropout))
            in_ch = out_ch
            
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Encoder output dimension
        encoder_dim = in_ch
        
        # Multi-head outputs (6 heads total)
        self.class_head = ClassificationHead(encoder_dim, encoder_dim // 2, num_classes, dropout)
        self.regression_head = RegressionHead(encoder_dim, encoder_dim // 2, dropout)
        self.quantile_head = QuantileHead(encoder_dim, encoder_dim // 2, dropout)
        self.trading_head = TradingHead(encoder_dim, encoder_dim // 2, dropout)
        self.candle_head = CandlePredictionHead(encoder_dim, encoder_dim // 2, n_future_candles, dropout)
        
    def _make_block(self, in_ch: int, out_ch: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [batch, seq_len, features]
        Output: [batch, encoder_dim]
        """
        # Transpose for Conv1d: [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.gelu(x)
        
        for block in self.blocks:
            x = block(x) + F.interpolate(x, size=x.size(-1)) if x.size(1) == block[0].out_channels else block(x)
            
        x = self.global_pool(x).squeeze(-1)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class logits for backward compatibility."""
        features = self.encode(x)
        return self.class_head(features)
    
    def forward_multihead(self, x: torch.Tensor) -> MultiHeadOutput:
        """Full multi-head forward pass with all trading outputs."""
        features = self.encode(x)
        
        class_logits = self.class_head(features)
        mu, sigma = self.regression_head(features)
        quantiles = self.quantile_head(features)
        entry_offset, sl_distance, tp_distance = self.trading_head(features)
        candle_deltas = self.candle_head(features)
        
        return MultiHeadOutput(
            class_logits=class_logits,
            mu=mu,
            quantiles=quantiles,
            sigma=sigma,
            entry_offset=entry_offset,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            candle_deltas=candle_deltas
        )


class MultiHeadGNN(BaseModel):
    """
    Graph Neural Network with multi-head output for cross-asset trading.
    Based on CrossAssetGNN architecture with temporal encoding + graph attention.
    Handles 3D input (batch, seq, features) for training compatibility.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_assets: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_classes: int = 3,
        num_quantiles: int = 5,
        n_future_candles: int = 5
    ):
        super().__init__("multihead_gnn", input_dim, num_classes)
        
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        self.num_quantiles = num_quantiles
        self.n_future_candles = n_future_candles
        
        # Temporal encoder (from CrossAssetGNN)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.temporal_lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Node encoder for graph structure (from CrossAssetGNN)
        features_per_asset = max(input_dim // num_assets, 1)
        self.node_encoder = nn.Sequential(
            nn.Linear(features_per_asset, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph attention layers (simplified for 3D input)
        self.graph_attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.graph_attention_layers.append(
                nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            )
            
        # Edge predictor for adjacency (from CrossAssetGNN)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Multi-head outputs
        self.class_head = ClassificationHead(hidden_dim, num_classes)
        self.regression_head = RegressionHead(hidden_dim)
        self.quantile_head = QuantileHead(hidden_dim, num_quantiles)
        self.trading_head = TradingHead(hidden_dim)
        self.candle_head = CandlePredictionHead(hidden_dim, n_future_steps=n_future_candles)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using temporal + attention encoding."""
        # x: (batch, seq, features)
        batch_size, seq_len, _ = x.shape
        
        # Temporal encoding
        h = self.temporal_encoder(x)
        h, _ = self.temporal_lstm(h)
        
        # Apply graph attention layers (self-attention across sequence)
        for attn_layer in self.graph_attention_layers:
            h_attn, _ = attn_layer(h, h, h)
            h = h + h_attn  # Residual connection
        
        # Take last hidden state
        h = h[:, -1, :]  # (batch, hidden)
        
        return h
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class logits for backward compatibility."""
        features = self.encode(x)
        return self.class_head(features)
    
    def forward_multihead(self, x: torch.Tensor) -> MultiHeadOutput:
        """Full multi-head forward pass with all trading outputs."""
        features = self.encode(x)
        
        class_logits = self.class_head(features)
        mu, sigma = self.regression_head(features)
        quantiles = self.quantile_head(features)
        entry_offset, sl_distance, tp_distance = self.trading_head(features)
        candle_deltas = self.candle_head(features)
        
        return MultiHeadOutput(
            class_logits=class_logits,
            mu=mu,
            quantiles=quantiles,
            sigma=sigma,
            entry_offset=entry_offset,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            candle_deltas=candle_deltas
        )


class MultiHeadVAE(BaseModel):
    """
    Variational Autoencoder with multi-head output for regime detection.
    Based on MarketVAE architecture with encoder/decoder + latent space.
    Uses latent representations for multi-head predictions.
    """
    
    def __init__(
        self,
        input_dim: int,
        sequence_length: int = 25,
        latent_dim: int = 64,
        hidden_dims: list = None,
        dropout: float = 0.2,
        num_classes: int = 3,
        num_quantiles: int = 5,
        n_future_candles: int = 5
    ):
        super().__init__("multihead_vae", input_dim, num_classes)
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.num_quantiles = num_quantiles
        self.n_future_candles = n_future_candles
        
        # Encoder (from MarketVAE)
        encoder_layers = []
        in_features = input_dim * sequence_length
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            in_features = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space (from MarketVAE)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder (from MarketVAE) - for reconstruction loss if needed
        decoder_layers = []
        in_features = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            in_features = hidden_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim * sequence_length))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Multi-head outputs (from latent space)
        self.class_head = ClassificationHead(latent_dim, num_classes)
        self.regression_head = RegressionHead(latent_dim)
        self.quantile_head = QuantileHead(latent_dim, num_quantiles)
        self.trading_head = TradingHead(latent_dim)
        self.candle_head = CandlePredictionHead(latent_dim, n_future_steps=n_future_candles)
    
    def encode_to_latent(self, x: torch.Tensor) -> tuple:
        """Encode input to latent distribution parameters (mu, log_var)."""
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from latent distribution."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        x_recon = self.decoder(z)
        x_recon = x_recon.view(-1, self.sequence_length, self.input_dim)
        return x_recon
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent features from input using reparameterization."""
        mu, log_var = self.encode_to_latent(x)
        z = self.reparameterize(mu, log_var)
        return z
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class logits for backward compatibility."""
        features = self.encode(x)
        return self.class_head(features)
    
    def forward_multihead(self, x: torch.Tensor) -> MultiHeadOutput:
        """Full multi-head forward pass with all trading outputs."""
        features = self.encode(x)
        
        class_logits = self.class_head(features)
        mu, sigma = self.regression_head(features)
        quantiles = self.quantile_head(features)
        entry_offset, sl_distance, tp_distance = self.trading_head(features)
        candle_deltas = self.candle_head(features)
        
        return MultiHeadOutput(
            class_logits=class_logits,
            mu=mu,
            quantiles=quantiles,
            sigma=sigma,
            entry_offset=entry_offset,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            candle_deltas=candle_deltas
        )
    
    def forward_with_reconstruction(self, x: torch.Tensor) -> tuple:
        """Forward pass returning both multi-head outputs and reconstruction for VAE loss."""
        mu_latent, log_var = self.encode_to_latent(x)
        z = self.reparameterize(mu_latent, log_var)
        x_recon = self.decode(z)
        
        # Multi-head outputs from latent
        class_logits = self.class_head(z)
        mu, sigma = self.regression_head(z)
        quantiles = self.quantile_head(z)
        entry_offset, sl_distance, tp_distance = self.trading_head(z)
        candle_deltas = self.candle_head(z)
        
        multihead_output = MultiHeadOutput(
            class_logits=class_logits,
            mu=mu,
            quantiles=quantiles,
            sigma=sigma,
            entry_offset=entry_offset,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            candle_deltas=candle_deltas
        )
        
        return multihead_output, x_recon, mu_latent, log_var


# Factory function to get multi-head model
def get_multihead_model(
    architecture: str,
    input_dim: int,
    **kwargs
) -> BaseModel:
    """
    Factory function to create multi-head models.
    
    Args:
        architecture: One of 'transformer', 'lstm', 'cnn', 'gnn', 'vae'
        input_dim: Number of input features
        **kwargs: Architecture-specific arguments
        
    Returns:
        Multi-head model instance
    """
    models = {
        'transformer': MultiHeadTransformer,
        'lstm': MultiHeadLSTM,
        'cnn': MultiHeadCNN,
        'gnn': MultiHeadGNN,
        'vae': MultiHeadVAE,
    }
    
    if architecture not in models:
        raise ValueError(f"Unknown architecture: {architecture}. Available: {list(models.keys())}")
    
    return models[architecture](input_dim=input_dim, **kwargs)
