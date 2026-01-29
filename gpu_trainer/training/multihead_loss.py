"""
Multi-Head Loss Functions for Institutional Trading

Combined loss = L_class + λ₁·L_regression + λ₂·L_quantile

Where:
- L_class: CrossEntropyLoss for direction classification
- L_regression: MSE for expected return (μ)
- L_quantile: Pinball loss for quantile regression (q10, q25, q50, q75, q90)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MultiHeadLossConfig:
    """Configuration for multi-head loss weights."""
    
    # Loss weights
    lambda_class: float = 1.0       # Weight for classification loss
    lambda_mu: float = 0.5          # Weight for regression (μ) loss
    lambda_sigma: float = 0.2       # Weight for uncertainty (σ) loss  
    lambda_quantile: float = 0.5    # Weight for quantile loss
    lambda_trading: float = 0.3     # Weight for trading (entry/SL/TP) loss
    lambda_candle: float = 0.3      # Weight for candle prediction loss
    
    # Classification options
    class_weights: Optional[torch.Tensor] = None  # For imbalanced classes
    label_smoothing: float = 0.1    # Smoothing for classification
    
    # Regression options
    mu_huber_delta: float = 0.02    # Delta for Huber loss (robust to outliers)
    trading_huber_delta: float = 0.01  # Delta for trading distances
    candle_huber_delta: float = 0.02   # Delta for candle deltas
    
    # Quantile options
    quantiles: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    
    # Future candle options
    n_future_candles: int = 5


class PinballLoss(nn.Module):
    """
    Pinball (quantile) loss for quantile regression.
    
    For quantile q and error e = y - y_hat:
    L_q(e) = q * max(e, 0) + (1-q) * max(-e, 0)
           = max(q*e, (q-1)*e)
    """
    
    def __init__(self, quantiles: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)):
        super().__init__()
        self.quantiles = quantiles
        self.register_buffer('q_tensor', torch.tensor(quantiles))
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute pinball loss.
        
        Args:
            predictions: [batch, n_quantiles] predicted quantile values
            targets: [batch, 1] or [batch] actual values
            
        Returns:
            Scalar loss value
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
            
        # Broadcast targets to match quantiles: [batch, n_quantiles]
        targets = targets.expand_as(predictions)
        
        # Compute errors: e = y - y_hat
        errors = targets - predictions  # [batch, n_quantiles]
        
        # Get quantiles on same device
        q = self.q_tensor.to(predictions.device)
        
        # Pinball loss: max(q*e, (q-1)*e)
        loss = torch.max(q * errors, (q - 1) * errors)
        
        return loss.mean()


class GaussianNLLLoss(nn.Module):
    """
    Negative log-likelihood loss for Gaussian distribution.
    
    Encourages model to predict both mean (μ) and variance (σ²).
    NLL = 0.5 * (log(σ²) + (y - μ)² / σ²)
    """
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        
    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian NLL loss.
        
        Args:
            mu: [batch, 1] predicted mean
            sigma: [batch, 1] predicted std (must be positive)
            targets: [batch, 1] or [batch] actual values
            
        Returns:
            Scalar loss value
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
            
        # Ensure sigma is positive
        sigma = sigma.clamp(min=self.eps)
        variance = sigma ** 2
        
        # NLL = 0.5 * (log(σ²) + (y - μ)² / σ²)
        nll = 0.5 * (torch.log(variance) + (targets - mu) ** 2 / variance)
        
        return nll.mean()


class MultiHeadLoss(nn.Module):
    """
    Combined loss for multi-head trading model.
    
    Total Loss = λ_class * L_class + λ_mu * L_mu + λ_sigma * L_sigma + 
                 λ_quantile * L_quantile + λ_trading * L_trading + λ_candle * L_candle
    
    Where:
    - L_class: CrossEntropy with optional label smoothing
    - L_mu: Huber loss for expected return
    - L_sigma: Gaussian NLL for uncertainty calibration
    - L_quantile: Pinball loss for quantile regression
    - L_trading: Huber loss for entry_offset, sl_distance, tp_distance
    - L_candle: Huber loss for future candle deltas
    """
    
    def __init__(self, config: Optional[MultiHeadLossConfig] = None):
        super().__init__()
        
        self.config = config or MultiHeadLossConfig()
        
        # Classification loss
        self.class_loss = nn.CrossEntropyLoss(
            weight=self.config.class_weights,
            label_smoothing=self.config.label_smoothing
        )
        
        # Regression loss (Huber for robustness)
        self.mu_loss = nn.HuberLoss(delta=self.config.mu_huber_delta)
        
        # Uncertainty loss
        self.sigma_loss = GaussianNLLLoss()
        
        # Quantile loss
        self.quantile_loss = PinballLoss(self.config.quantiles)
        
        # Trading loss (entry/SL/TP)
        self.trading_loss = nn.HuberLoss(delta=self.config.trading_huber_delta)
        
        # Candle prediction loss
        self.candle_loss = nn.HuberLoss(delta=self.config.candle_huber_delta)
        
    def forward(
        self,
        class_logits: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        quantiles: torch.Tensor,
        class_targets: torch.Tensor,
        return_targets: torch.Tensor,
        entry_offset: Optional[torch.Tensor] = None,
        sl_distance: Optional[torch.Tensor] = None,
        tp_distance: Optional[torch.Tensor] = None,
        candle_deltas: Optional[torch.Tensor] = None,
        trading_targets: Optional[Dict[str, torch.Tensor]] = None,
        candle_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            class_logits: [batch, 3] classification logits
            mu: [batch, 1] predicted expected return
            sigma: [batch, 1] predicted uncertainty
            quantiles: [batch, 5] predicted quantiles
            class_targets: [batch] class labels (0=SHORT, 1=HOLD, 2=LONG)
            return_targets: [batch] or [batch, 1] actual forward returns
            entry_offset: [batch, 1] predicted entry offset (optional)
            sl_distance: [batch, 1] predicted SL distance (optional)
            tp_distance: [batch, 1] predicted TP distance (optional)
            candle_deltas: [batch, n_steps, 3] predicted candle deltas (optional)
            trading_targets: Dict with 'entry_offset', 'sl_distance', 'tp_distance' targets
            candle_targets: [batch, n_steps, 3] actual candle deltas
            
        Returns:
            Dict with 'total' loss and individual components
        """
        # Classification loss
        l_class = self.class_loss(class_logits, class_targets)
        
        # Regression loss (expected return)
        if return_targets.dim() == 1:
            return_targets = return_targets.unsqueeze(-1)
        l_mu = self.mu_loss(mu, return_targets)
        
        # Uncertainty calibration loss
        l_sigma = self.sigma_loss(mu, sigma, return_targets)
        
        # Quantile loss
        l_quantile = self.quantile_loss(quantiles, return_targets)
        
        # Trading loss (if targets provided)
        l_trading = torch.tensor(0.0, device=class_logits.device)
        if trading_targets is not None and entry_offset is not None:
            l_entry = self.trading_loss(entry_offset, trading_targets['entry_offset'])
            l_sl = self.trading_loss(sl_distance, trading_targets['sl_distance'])
            l_tp = self.trading_loss(tp_distance, trading_targets['tp_distance'])
            l_trading = (l_entry + l_sl + l_tp) / 3.0
        
        # Candle prediction loss (if targets provided)
        l_candle = torch.tensor(0.0, device=class_logits.device)
        if candle_targets is not None and candle_deltas is not None:
            l_candle = self.candle_loss(candle_deltas, candle_targets)
        
        # Combined loss
        total = (
            self.config.lambda_class * l_class +
            self.config.lambda_mu * l_mu +
            self.config.lambda_sigma * l_sigma +
            self.config.lambda_quantile * l_quantile +
            self.config.lambda_trading * l_trading +
            self.config.lambda_candle * l_candle
        )
        
        return {
            'total': total,
            'class': l_class,
            'mu': l_mu,
            'sigma': l_sigma,
            'quantile': l_quantile,
            'trading': l_trading,
            'candle': l_candle
        }


class QuantileCalibrationLoss(nn.Module):
    """
    Additional loss term for quantile calibration.
    
    Ensures that the coverage of each quantile matches its target probability.
    E.g., q10 should be below actual 10% of the time.
    """
    
    def __init__(self, quantiles: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)):
        super().__init__()
        self.quantiles = quantiles
        self.register_buffer('q_tensor', torch.tensor(quantiles))
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute calibration loss.
        
        Penalizes deviation from expected coverage.
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
            
        targets = targets.expand_as(predictions)
        q = self.q_tensor.to(predictions.device)
        
        # Compute actual coverage (how often target < prediction)
        coverage = (targets < predictions).float().mean(dim=0)  # [n_quantiles]
        
        # Should match target quantiles
        calibration_error = (coverage - q) ** 2
        
        return calibration_error.mean()


def create_multihead_loss(
    class_weights: Optional[torch.Tensor] = None,
    lambda_class: float = 1.0,
    lambda_mu: float = 0.5,
    lambda_quantile: float = 0.5
) -> MultiHeadLoss:
    """
    Factory function to create multi-head loss with custom weights.
    """
    config = MultiHeadLossConfig(
        lambda_class=lambda_class,
        lambda_mu=lambda_mu,
        lambda_quantile=lambda_quantile,
        class_weights=class_weights
    )
    return MultiHeadLoss(config)
