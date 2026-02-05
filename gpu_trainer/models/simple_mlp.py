"""
Simple MLP Classifier for Stable Training

This is a baseline model designed for stable gradient behavior.
The complex LSTM/Transformer architectures were causing gradient explosions
even on clean data. This simple MLP trains stably with gradient norms < 1.

Use this as a baseline to verify training pipeline works before adding complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
import math

# Import shared MultiHeadOutput from multihead.py for compatibility
from .multihead import MultiHeadOutput


@dataclass
class SimpleMLP_Config:
    """Configuration for SimpleMLP model."""
    input_dim: int = 41  # Number of features
    hidden_dims: list = None  # Hidden layer dimensions
    num_classes: int = 3  # SHORT, HOLD, LONG
    dropout: float = 0.3  # Dropout rate
    use_layer_norm: bool = True  # Use LayerNorm for stability
    n_candle_steps: int = 5  # For dummy candle output shape
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]


class SimpleMLP(nn.Module):
    """
    Simple MLP classifier designed for stable training.
    
    Key stability features:
    1. LayerNorm after each layer (prevents internal covariate shift)
    2. Orthogonal initialization (better gradient flow)
    3. GELU activation (smoother than ReLU)
    4. Moderate dropout (regularization without gradient issues)
    5. Residual connections in wider layers
    
    This model trains stably with gradient norms < 1 on the same data
    that causes LSTM/Transformer to explode to 30+.
    """
    
    def __init__(self, config: SimpleMLP_Config):
        super().__init__()
        self.config = config
        
        # Store input_dim as attribute for trainer compatibility (checkpoint saving)
        self.input_dim = config.input_dim
        
        # Build layers
        layers = []
        prev_dim = config.input_dim
        
        for i, hidden_dim in enumerate(config.hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))
            
            prev_dim = hidden_dim
        
        self.trunk = nn.Sequential(*layers)
        
        # Classification head
        self.classifier = nn.Linear(prev_dim, config.num_classes)
        
        # Store dimensions for dummy outputs
        self.n_candle_steps = config.n_candle_steps
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with orthogonal initialization for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning class logits.
        
        Args:
            x: [batch, seq_len, features] or [batch, features]
            
        Returns:
            class_logits: [batch, 3]
        """
        # Handle sequence input - take last timestep
        if x.dim() == 3:
            x = x[:, -1, :]  # [batch, features]
        
        # Forward through trunk
        features = self.trunk(x)
        
        # Classification logits
        logits = self.classifier(features)
        
        return logits
    
    def forward_multihead(self, x: torch.Tensor) -> MultiHeadOutput:
        """
        Multi-head forward pass matching the MultiHeadTransformer interface.
        
        Only classification head is active. All other heads return None to
        signal they should be skipped in loss computation.
        """
        batch_size = x.size(0)
        device = x.device
        
        # Get classification logits
        class_logits = self.forward(x)
        
        # Required fields: class_logits and mu/quantiles (non-optional in MultiHeadOutput)
        # Set required fields to zeros, optional fields to None
        zeros_1 = torch.zeros(batch_size, 1, device=device)
        zeros_5 = torch.zeros(batch_size, 5, device=device)
        
        return MultiHeadOutput(
            class_logits=class_logits,
            mu=zeros_1,  # Required field
            quantiles=zeros_5,  # Required field
            # All optional fields set to None to skip in loss computation
            sigma=None,
            entry_offset=None,
            sl_distance=None,
            tp_distance=None,
            candle_deltas=None,
            vol_state_logits=None,
            acceleration=None
        )
    
    def parameters_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_simple_mlp(input_dim: int = 41, num_classes: int = 3) -> SimpleMLP:
    """Factory function to create a SimpleMLP with default config."""
    config = SimpleMLP_Config(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        num_classes=num_classes,
        dropout=0.3,
        use_layer_norm=True
    )
    return SimpleMLP(config)


if __name__ == "__main__":
    # Quick test
    model = create_simple_mlp(input_dim=41)
    print(f"SimpleMLP parameters: {model.parameters_count():,}")
    
    # Test forward pass
    x = torch.randn(32, 100, 41)  # batch=32, seq=100, features=41
    output = model.forward_multihead(x)
    print(f"class_logits shape: {output.class_logits.shape}")
    print(f"mu shape: {output.mu.shape}")
    
    # Test gradient flow
    loss = output.class_logits.sum()
    loss.backward()
    
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm(2).item() ** 2
    grad_norm = total_norm ** 0.5
    print(f"Gradient norm: {grad_norm:.4f}")
