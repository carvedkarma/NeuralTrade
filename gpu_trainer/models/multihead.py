"""
Multi-Head Model Architecture for Institutional Trading

Instead of classification-only training, these models output:
1. Classification head → direction (LONG/HOLD/SHORT)
2. Regression head → expected return (μ)
3. Quantile head → uncertainty quantiles (q10, q25, q50, q75, q90)

SL/TP are derived mathematically from quantiles, NOT learned directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, NamedTuple
from dataclasses import dataclass
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


class MultiHeadTransformer(BaseModel):
    """
    Transformer with multi-head output for institutional trading.
    
    Outputs:
    - Classification: direction probabilities
    - Regression: expected return (μ) and uncertainty (σ)
    - Quantiles: q10, q25, q50, q75, q90 for SL/TP derivation
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
        num_quantiles: int = 5
    ):
        # Store num_quantiles for checkpoint compatibility
        super().__init__("multihead_transformer", input_dim, num_classes)
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_quantiles = num_quantiles
        
        # Shared encoder backbone
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Multi-head outputs
        self.class_head = ClassificationHead(d_model, d_model // 2, num_classes, dropout)
        self.regression_head = RegressionHead(d_model, d_model // 2, dropout)
        self.quantile_head = QuantileHead(d_model, d_model // 2, dropout)
        
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
        """
        features = self.encode(x, mask)
        
        class_logits = self.class_head(features)
        mu, sigma = self.regression_head(features)
        quantiles = self.quantile_head(features)
        
        return MultiHeadOutput(
            class_logits=class_logits,
            mu=mu,
            quantiles=quantiles,
            sigma=sigma
        )
    
    def predict_with_quantiles(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convenience method for inference.
        
        Returns dict with:
        - probabilities: softmax of class logits
        - direction: argmax of probabilities
        - mu: expected return
        - sigma: uncertainty
        - quantiles: {q10, q25, q50, q75, q90}
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
        num_quantiles: int = 5
    ):
        super().__init__("multihead_lstm", input_dim, num_classes)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_quantiles = num_quantiles
        
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
        
        # Multi-head outputs
        self.class_head = ClassificationHead(encoder_dim, encoder_dim // 2, num_classes, dropout)
        self.regression_head = RegressionHead(encoder_dim, encoder_dim // 2, dropout)
        self.quantile_head = QuantileHead(encoder_dim, encoder_dim // 2, dropout)
        
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
        """Full multi-head forward pass."""
        features = self.encode(x)
        
        class_logits = self.class_head(features)
        mu, sigma = self.regression_head(features)
        quantiles = self.quantile_head(features)
        
        return MultiHeadOutput(
            class_logits=class_logits,
            mu=mu,
            quantiles=quantiles,
            sigma=sigma
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
        num_quantiles: int = 5
    ):
        super().__init__("multihead_cnn", input_dim, num_classes)
        
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.num_quantiles = num_quantiles
        
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
        
        # Multi-head outputs
        self.class_head = ClassificationHead(encoder_dim, encoder_dim // 2, num_classes, dropout)
        self.regression_head = RegressionHead(encoder_dim, encoder_dim // 2, dropout)
        self.quantile_head = QuantileHead(encoder_dim, encoder_dim // 2, dropout)
        
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
        """Full multi-head forward pass."""
        features = self.encode(x)
        
        class_logits = self.class_head(features)
        mu, sigma = self.regression_head(features)
        quantiles = self.quantile_head(features)
        
        return MultiHeadOutput(
            class_logits=class_logits,
            mu=mu,
            quantiles=quantiles,
            sigma=sigma
        )


# Factory function to get multi-head model
def get_multihead_model(
    architecture: str,
    input_dim: int,
    **kwargs
) -> BaseModel:
    """
    Factory function to create multi-head models.
    
    Args:
        architecture: One of 'transformer', 'lstm', 'cnn'
        input_dim: Number of input features
        **kwargs: Architecture-specific arguments
        
    Returns:
        Multi-head model instance
    """
    models = {
        'transformer': MultiHeadTransformer,
        'lstm': MultiHeadLSTM,
        'cnn': MultiHeadCNN,
    }
    
    if architecture not in models:
        raise ValueError(f"Unknown architecture: {architecture}. Available: {list(models.keys())}")
    
    return models[architecture](input_dim=input_dim, **kwargs)
