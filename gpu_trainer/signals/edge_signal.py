"""
Edge-Based Signal Generation

Institutional-grade signal generation using:
    edge = (μ - cost) / σ
    confidence = edge value
    enter_trade = confidence > threshold

This approach naturally produces "strong signals only" because:
- Weak signals have low μ (expected return)
- Uncertain signals have high σ (volatility)
- Both result in low edge

Output format includes:
- action: LONG, SHORT, or NO_TRADE
- confidence: Edge-based confidence score
- expected_move: μ prediction
- uncertainty: σ prediction
- cost_estimate: Transaction cost estimate
- suggested_order_type: maker vs taker
- risk_adjusted_size: Position size recommendation
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradingCosts:
    """Transaction costs for edge calculation."""
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004
    base_slippage: float = 0.0001
    vol_slippage_mult: float = 0.5
    avg_funding_8h: float = 0.0001
    
    def round_trip_cost(self, volatility: float, is_taker: bool = True, 
                        hold_hours: float = 4) -> float:
        """Calculate total round-trip cost."""
        fee = self.taker_fee if is_taker else self.maker_fee
        slippage = self.base_slippage + self.vol_slippage_mult * volatility
        funding = self.avg_funding_8h * (hold_hours / 8)
        return (fee * 2) + (slippage * 2) + abs(funding)


@dataclass
class EdgeSignal:
    """
    Complete trading signal output.
    
    All the information needed to make a trading decision.
    """
    timestamp: int
    
    action: str  # "LONG", "SHORT", "NO_TRADE"
    confidence: float  # Edge-based confidence (edge / σ)
    
    expected_move: float  # μ (expected return)
    uncertainty: float  # σ (volatility/uncertainty)
    edge: float  # μ - cost
    
    cost_estimate: float
    suggested_order_type: str  # "MAKER" or "TAKER"
    urgency: str  # "LOW", "MEDIUM", "HIGH"
    
    position_size_pct: float  # Recommended position size as % of capital
    stop_loss_pct: float  # Suggested stop loss distance
    take_profit_pct: float  # Suggested take profit distance
    
    regime: str  # Current market regime
    expert_weights: Dict[str, float]  # MoE expert weights if available
    
    p10_return: Optional[float] = None
    p50_return: Optional[float] = None
    p90_return: Optional[float] = None
    
    reasons: Optional[list] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response."""
        return asdict(self)


class EdgeSignalGenerator:
    """
    Generates trading signals from model predictions.
    
    Uses edge-based confidence scoring:
    1. Get μ (expected return) and σ (uncertainty) from model
    2. Calculate cost based on current volatility
    3. Compute edge = μ - cost
    4. Compute confidence = edge / σ
    5. Generate signal only if confidence exceeds threshold
    """
    
    def __init__(self,
                 min_confidence: float = 0.5,
                 min_edge_pct: float = 0.001,
                 max_position_pct: float = 0.1,
                 costs: Optional[TradingCosts] = None):
        self.min_confidence = min_confidence
        self.min_edge_pct = min_edge_pct
        self.max_position_pct = max_position_pct
        self.costs = costs or TradingCosts()
        
    def calculate_edge_metrics(self,
                                mu: float,
                                sigma: float,
                                current_volatility: float) -> Tuple[float, float, float, bool]:
        """
        Calculate edge metrics.
        
        Returns:
            edge: Expected profit after costs
            confidence: Risk-adjusted edge
            cost: Transaction cost
            is_taker: Whether to use taker order
        """
        cost_maker = self.costs.round_trip_cost(current_volatility, is_taker=False)
        cost_taker = self.costs.round_trip_cost(current_volatility, is_taker=True)
        
        edge_maker = abs(mu) - cost_maker
        edge_taker = abs(mu) - cost_taker
        
        urgency_threshold = 1.5  # Edge ratio threshold for taker
        
        if edge_taker > edge_maker * urgency_threshold and edge_taker > 0:
            is_taker = True
            cost = cost_taker
            edge = edge_taker
        else:
            is_taker = False
            cost = cost_maker
            edge = edge_maker
        
        confidence = edge / max(sigma, 0.001) if sigma > 0 else 0
        
        return edge, confidence, cost, is_taker
    
    def calculate_position_size(self,
                                 edge: float,
                                 sigma: float,
                                 capital: float = 1.0) -> float:
        """
        Calculate position size using bounded Kelly criterion.
        
        Kelly fraction = edge / σ²
        Bounded to prevent over-betting
        """
        if sigma <= 0 or edge <= 0:
            return 0
        
        kelly = edge / (sigma ** 2)
        
        half_kelly = kelly * 0.5
        
        position_pct = min(half_kelly, self.max_position_pct)
        
        position_pct = max(0, position_pct)
        
        return position_pct
    
    def calculate_stops(self,
                        mu: float,
                        sigma: float,
                        direction: int) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels.
        
        Uses ATR-based approach with μ/σ information.
        """
        stop_mult = 2.0
        stop_loss = sigma * stop_mult
        
        risk = stop_loss
        reward = abs(mu)
        rr_ratio = reward / max(risk, 0.001)
        
        if rr_ratio < 1.5:
            take_profit = risk * 1.5
        else:
            take_profit = abs(mu) * 1.2
        
        return stop_loss, take_profit
    
    def determine_urgency(self,
                          edge: float,
                          sigma: float,
                          mu: float) -> str:
        """Determine signal urgency based on edge characteristics."""
        confidence = edge / max(sigma, 0.001)
        
        if confidence > 2.0 and abs(mu) > 0.01:
            return "HIGH"
        elif confidence > 1.0:
            return "MEDIUM"
        else:
            return "LOW"
    
    def generate_signal(self,
                        mu: float,
                        sigma: float,
                        current_price: float,
                        current_volatility: float,
                        timestamp: int,
                        regime: str = "UNKNOWN",
                        expert_weights: Optional[Dict[str, float]] = None,
                        quantiles: Optional[Tuple[float, float, float]] = None) -> EdgeSignal:
        """
        Generate a complete trading signal.
        
        Args:
            mu: Predicted expected return
            sigma: Predicted uncertainty
            current_price: Current asset price
            current_volatility: Current realized volatility
            timestamp: Signal timestamp
            regime: Current market regime
            expert_weights: MoE expert weights
            quantiles: (p10, p50, p90) return predictions
            
        Returns:
            EdgeSignal with complete trading recommendation
        """
        edge, confidence, cost, is_taker = self.calculate_edge_metrics(
            mu, sigma, current_volatility
        )
        
        should_trade = (
            confidence >= self.min_confidence and
            edge >= self.min_edge_pct
        )
        
        if should_trade:
            action = "LONG" if mu > 0 else "SHORT"
            direction = 1 if mu > 0 else -1
        else:
            action = "NO_TRADE"
            direction = 0
        
        position_size = self.calculate_position_size(edge, sigma) if should_trade else 0
        
        if should_trade:
            stop_loss, take_profit = self.calculate_stops(mu, sigma, direction)
        else:
            stop_loss, take_profit = 0, 0
        
        urgency = self.determine_urgency(edge, sigma, mu) if should_trade else "LOW"
        
        reasons = []
        if should_trade:
            reasons.append(f"Edge: {edge*100:.3f}% after costs")
            reasons.append(f"Confidence: {confidence:.2f}σ")
            reasons.append(f"Expected move: {mu*100:.3f}%")
            if regime != "UNKNOWN":
                reasons.append(f"Regime: {regime}")
        else:
            if confidence < self.min_confidence:
                reasons.append(f"Low confidence: {confidence:.2f} < {self.min_confidence}")
            if edge < self.min_edge_pct:
                reasons.append(f"Insufficient edge: {edge*100:.3f}% < {self.min_edge_pct*100:.3f}%")
        
        signal = EdgeSignal(
            timestamp=timestamp,
            action=action,
            confidence=confidence,
            expected_move=mu,
            uncertainty=sigma,
            edge=edge,
            cost_estimate=cost,
            suggested_order_type="TAKER" if is_taker else "MAKER",
            urgency=urgency,
            position_size_pct=position_size,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            regime=regime,
            expert_weights=expert_weights or {},
            p10_return=quantiles[0] if quantiles else None,
            p50_return=quantiles[1] if quantiles else None,
            p90_return=quantiles[2] if quantiles else None,
            reasons=reasons
        )
        
        return signal


class ModelSignalInterface:
    """
    Interface between ML models and signal generation.
    
    Handles:
    - Getting predictions from models
    - Converting to edge signals
    - Aggregating multiple model predictions
    """
    
    def __init__(self,
                 signal_generator: EdgeSignalGenerator = None,
                 device: str = "cuda"):
        self.signal_generator = signal_generator or EdgeSignalGenerator()
        self.device = device
        
    def generate_from_model(self,
                            model: torch.nn.Module,
                            features: torch.Tensor,
                            current_price: float,
                            current_volatility: float,
                            timestamp: int,
                            regime: str = "UNKNOWN") -> EdgeSignal:
        """
        Generate signal from a single model.
        """
        model.eval()
        
        with torch.no_grad():
            features = features.to(self.device)
            
            output = model(features)
            
            if isinstance(output, tuple):
                mu, sigma = output[0], output[1]
                if len(output) > 2:
                    gate_weights = output[2]
                else:
                    gate_weights = None
            else:
                mu = output[:, 0]
                sigma = torch.abs(output[:, 1]) + 0.001
                gate_weights = None
            
            mu_val = mu.cpu().numpy().item() if mu.dim() > 0 else mu.cpu().item()
            sigma_val = sigma.cpu().numpy().item() if sigma.dim() > 0 else sigma.cpu().item()
            
            expert_weights = None
            if gate_weights is not None:
                weights = gate_weights.cpu().numpy().flatten()
                expert_names = ["trend", "mean_reversion", "volatility", "chaos"]
                expert_weights = {
                    expert_names[i]: float(weights[i])
                    for i in range(min(len(weights), len(expert_names)))
                }
        
        signal = self.signal_generator.generate_signal(
            mu=mu_val,
            sigma=sigma_val,
            current_price=current_price,
            current_volatility=current_volatility,
            timestamp=timestamp,
            regime=regime,
            expert_weights=expert_weights
        )
        
        return signal
    
    def generate_from_ensemble(self,
                                models: list,
                                features: torch.Tensor,
                                current_price: float,
                                current_volatility: float,
                                timestamp: int) -> EdgeSignal:
        """
        Generate signal from an ensemble of models.
        
        Combines predictions using uncertainty-weighted averaging.
        """
        all_mus = []
        all_sigmas = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                features = features.to(self.device)
                output = model(features)
                
                if isinstance(output, tuple):
                    mu, sigma = output[0], output[1]
                else:
                    mu = output[:, 0]
                    sigma = torch.abs(output[:, 1]) + 0.001
                
                all_mus.append(mu.cpu().item())
                all_sigmas.append(sigma.cpu().item())
        
        sigmas = np.array(all_sigmas)
        precisions = 1 / (sigmas ** 2 + 1e-8)
        weights = precisions / precisions.sum()
        
        combined_mu = np.sum(np.array(all_mus) * weights)
        
        combined_sigma = np.sqrt(np.sum(sigmas ** 2 * weights))
        
        signal = self.signal_generator.generate_signal(
            mu=float(combined_mu),
            sigma=float(combined_sigma),
            current_price=current_price,
            current_volatility=current_volatility,
            timestamp=timestamp,
            regime="ENSEMBLE"
        )
        
        return signal


def create_signal_generator(
    min_confidence: float = 0.5,
    min_edge_pct: float = 0.001,
    max_position_pct: float = 0.1
) -> EdgeSignalGenerator:
    """Factory function to create signal generator."""
    costs = TradingCosts(
        maker_fee=0.0002,
        taker_fee=0.0004,
        base_slippage=0.0001,
        vol_slippage_mult=0.5,
        avg_funding_8h=0.0001
    )
    
    return EdgeSignalGenerator(
        min_confidence=min_confidence,
        min_edge_pct=min_edge_pct,
        max_position_pct=max_position_pct,
        costs=costs
    )
