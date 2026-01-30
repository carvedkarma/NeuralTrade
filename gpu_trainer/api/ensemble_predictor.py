"""
Professional Ensemble Predictor for Trading Signals

Design principles:
1. Direction models (Transformer, TFT, LSTM, CNN) vote on direction
2. VAE acts as regime gate (trend/range/chop detection)
3. GNN acts as risk filter (risk-on/off detection)
4. Weights based on walk-forward trading metrics, NOT accuracy
5. Calibrated probabilities via temperature scaling
6. Confidence margin = p_top1 - p_top2 (not raw max)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    CHOPPY = "CHOPPY"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    UNKNOWN = "UNKNOWN"

class RiskRegime(Enum):
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    NEUTRAL = "NEUTRAL"
    CORRELATION_SHOCK = "CORRELATION_SHOCK"
    UNKNOWN = "UNKNOWN"

@dataclass
class ModelWeight:
    """Walk-forward trading metric weights for a model."""
    model_name: str
    expectancy: float  # Expected profit per trade
    precision_on_trade: float  # Precision when model decides to trade (not HOLD)
    profit_factor: float  # Gross profit / gross loss
    f1_directional: float  # F1 on LONG/SHORT only
    sharpe: float  # Walk-forward Sharpe ratio
    calibration_temp: float = 1.0  # Temperature for probability calibration
    
    @property
    def composite_weight(self) -> float:
        """Compute composite weight from trading metrics."""
        weights = {
            'expectancy': 0.3,
            'precision': 0.25,
            'profit_factor': 0.2,
            'f1': 0.15,
            'sharpe': 0.1
        }
        
        score = (
            weights['expectancy'] * max(0, self.expectancy) +
            weights['precision'] * self.precision_on_trade +
            weights['profit_factor'] * min(2.0, self.profit_factor) / 2.0 +
            weights['f1'] * self.f1_directional +
            weights['sharpe'] * max(0, min(3.0, self.sharpe)) / 3.0
        )
        return max(0.01, score)  # Minimum weight

@dataclass
class EnsembleSignal:
    """Output signal from ensemble predictor."""
    action: str  # LONG, SHORT, HOLD, NO_TRADE
    confidence: float
    confidence_margin: float  # p_top1 - p_top2
    edge: float
    
    # Regime information
    market_regime: str
    risk_regime: str
    regime_confidence: float
    
    # Model agreement
    agreement_pct: float
    weighted_agreement: float
    disagreement_score: float
    
    # Position sizing adjustments
    position_size_pct: float
    regime_adjusted_size: float
    
    # Thresholds used
    confidence_threshold_used: float
    regime_adjustment: str
    
    # Per-model breakdown
    model_votes: Dict[str, Dict[str, Any]]
    reasons: List[str]
    
    # Probabilities
    ensemble_probs: Dict[str, float]

class EnsemblePredictor:
    """
    Professional ensemble predictor with:
    - Direction model voting
    - VAE regime gating
    - GNN risk filtering
    - Walk-forward metric weighting
    - Temperature-scaled calibration
    """
    
    DIRECTION_MODELS = ['transformer', 'tft', 'lstm', 'cnn', 'bidirectional', 'stacked', 'conv_lstm', 'resnet', 'inception', 'wavenet']
    REGIME_MODELS = ['vae', 'market_vae', 'conditional_vae', 'cvae']
    RISK_MODELS = ['gnn', 'cross_asset', 'temporal_gnn', 'crossasset']
    
    def __init__(self, model_instances: Dict[str, torch.nn.Module], device: str = "cuda"):
        self.device = device
        self.model_instances = model_instances
        
        # Classify models by role
        self.direction_models = {}
        self.regime_models = {}
        self.risk_models = {}
        
        for name, model in model_instances.items():
            name_lower = name.lower()
            if any(dm in name_lower for dm in self.REGIME_MODELS):
                self.regime_models[name] = model
            elif any(rm in name_lower for rm in self.RISK_MODELS):
                self.risk_models[name] = model
            else:
                self.direction_models[name] = model
        
        # Default model weights (will be updated from walk-forward metrics)
        self.model_weights = self._load_or_create_weights()
        
        # Base thresholds
        self.base_confidence_threshold = 0.15
        self.base_margin_threshold = 0.10
        self.majority_weight_threshold = 0.55
        
        logger.info(f"EnsemblePredictor initialized:")
        logger.info(f"  Direction models: {list(self.direction_models.keys())}")
        logger.info(f"  Regime models: {list(self.regime_models.keys())}")
        logger.info(f"  Risk models: {list(self.risk_models.keys())}")
    
    def _load_or_create_weights(self) -> Dict[str, ModelWeight]:
        """Load walk-forward metric weights or create defaults."""
        weights_path = Path(__file__).parent.parent / "checkpoints" / "model_weights.json"
        
        if weights_path.exists():
            try:
                with open(weights_path) as f:
                    data = json.load(f)
                return {
                    name: ModelWeight(**w) for name, w in data.items()
                }
            except Exception as e:
                logger.warning(f"Failed to load model weights: {e}")
        
        # Create default weights for all models
        defaults = {}
        for name in self.model_instances:
            defaults[name] = ModelWeight(
                model_name=name,
                expectancy=0.001,  # 0.1% expected per trade
                precision_on_trade=0.55,  # 55% precision on trades
                profit_factor=1.2,  # 1.2:1 profit factor
                f1_directional=0.45,  # 45% F1 on LONG/SHORT
                sharpe=0.5,  # 0.5 Sharpe ratio
                calibration_temp=1.0
            )
        return defaults
    
    def save_weights(self, weights: Dict[str, ModelWeight]):
        """Save model weights from walk-forward evaluation."""
        weights_path = Path(__file__).parent.parent / "checkpoints" / "model_weights.json"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {name: {
            'model_name': w.model_name,
            'expectancy': w.expectancy,
            'precision_on_trade': w.precision_on_trade,
            'profit_factor': w.profit_factor,
            'f1_directional': w.f1_directional,
            'sharpe': w.sharpe,
            'calibration_temp': w.calibration_temp
        } for name, w in weights.items()}
        
        with open(weights_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.model_weights = weights
        logger.info(f"Saved model weights to {weights_path}")
    
    def _calibrate_probs(self, probs: np.ndarray, model_name: str) -> np.ndarray:
        """Apply temperature scaling for probability calibration."""
        temp = self.model_weights.get(model_name, ModelWeight(
            model_name=model_name,
            expectancy=0, precision_on_trade=0.5,
            profit_factor=1.0, f1_directional=0.4, sharpe=0
        )).calibration_temp
        
        if temp == 1.0:
            return probs
        
        # Apply temperature scaling
        logits = np.log(probs + 1e-8)
        scaled_logits = logits / temp
        calibrated = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))
        return calibrated
    
    def _get_model_prediction(self, model: torch.nn.Module, features: torch.Tensor, model_name: str) -> Dict:
        """Get prediction from a single model."""
        try:
            model.eval()
            with torch.no_grad():
                output = model(features)
                probs = F.softmax(output, dim=-1).cpu().numpy()[0]
                
                # Calibrate probabilities
                probs = self._calibrate_probs(probs, model_name)
                
                # Direction and confidence margin
                action_idx = int(np.argmax(probs))
                sorted_probs = np.sort(probs)[::-1]
                confidence = float(probs[action_idx])
                confidence_margin = float(sorted_probs[0] - sorted_probs[1])
                
                action_map = {0: "SHORT", 1: "HOLD", 2: "LONG"}
                
                return {
                    "model": model_name,
                    "probs": probs.tolist(),
                    "action_idx": action_idx,
                    "action": action_map[action_idx],
                    "confidence": confidence,
                    "confidence_margin": confidence_margin,
                    "p_long": float(probs[2]),
                    "p_short": float(probs[0]),
                    "p_hold": float(probs[1])
                }
        except Exception as e:
            logger.error(f"Prediction error for {model_name}: {e}")
            return None
    
    def _detect_regime_from_vae(self, model: torch.nn.Module, features: torch.Tensor, model_name: str) -> Tuple[MarketRegime, float]:
        """Use VAE latent space for regime detection."""
        try:
            model.eval()
            with torch.no_grad():
                # Get latent representation
                # === FIX: Handle VAE encode correctly ===
                # MultiHeadVAE.encode() returns single z tensor, not (mu, log_var)
                # Use encode_to_latent() if available for (mu, log_var) tuple
                if hasattr(model, 'get_latent'):
                    z = model.get_latent(features)
                elif hasattr(model, 'encode_to_latent'):
                    # Use encode_to_latent() which returns (mu, log_var)
                    mu, log_var = model.encode_to_latent(features)
                    z = mu  # Use mu for regime stability
                elif hasattr(model, 'encode'):
                    # encode() may return single tensor (z) or tuple (mu, log_var)
                    result = model.encode(features)
                    if isinstance(result, tuple) and len(result) == 2:
                        mu, _ = result
                        z = mu
                    else:
                        z = result  # Single tensor returned
                else:
                    # Fallback: use forward pass
                    output = model(features)
                    probs = F.softmax(output, dim=-1).cpu().numpy()[0]
                    
                    # High HOLD probability suggests ranging/choppy market
                    p_hold = probs[1]
                    p_directional = probs[0] + probs[2]
                    
                    if p_hold > 0.6:
                        return MarketRegime.CHOPPY, float(p_hold)
                    elif p_directional > 0.7:
                        return MarketRegime.TRENDING, float(p_directional)
                    else:
                        return MarketRegime.RANGING, 0.5
                
                # Analyze latent dimensions for regime
                z_np = z.cpu().numpy()[0]
                z_var = np.var(z_np)
                z_mean_abs = np.mean(np.abs(z_np))
                
                # High variance in latent = volatile/transitioning
                # Low variance + low mean = stable/ranging
                # Low variance + high mean = trending
                if z_var > 1.5:
                    return MarketRegime.HIGH_VOLATILITY, min(0.9, z_var / 3.0)
                elif z_mean_abs > 1.0:
                    return MarketRegime.TRENDING, min(0.9, z_mean_abs / 2.0)
                elif z_var < 0.5:
                    return MarketRegime.CHOPPY, 1.0 - z_var
                else:
                    return MarketRegime.RANGING, 0.5
                    
        except Exception as e:
            logger.error(f"Regime detection error for {model_name}: {e}")
            return MarketRegime.UNKNOWN, 0.0
    
    def _detect_risk_from_gnn(self, model: torch.nn.Module, features: torch.Tensor, model_name: str) -> Tuple[RiskRegime, float]:
        """Use GNN for cross-asset risk regime detection."""
        try:
            model.eval()
            with torch.no_grad():
                # Get prediction
                output = model(features)
                probs = F.softmax(output, dim=-1).cpu().numpy()[0]
                
                # Get asset relations if available
                if hasattr(model, 'get_asset_relations'):
                    relations = model.get_asset_relations(features)
                    relations_np = relations.cpu().numpy()[0]
                    
                    # High correlation = potential shock/risk-off
                    off_diag = relations_np[~np.eye(relations_np.shape[0], dtype=bool)]
                    avg_correlation = float(np.mean(np.abs(off_diag)))
                    
                    if avg_correlation > 0.8:
                        return RiskRegime.CORRELATION_SHOCK, avg_correlation
                    elif avg_correlation > 0.6:
                        return RiskRegime.RISK_OFF, avg_correlation
                    elif avg_correlation < 0.3:
                        return RiskRegime.RISK_ON, 1.0 - avg_correlation
                    else:
                        return RiskRegime.NEUTRAL, 0.5
                
                # Fallback: use directional signal
                p_short = probs[0]
                p_long = probs[2]
                
                if p_short > 0.6:
                    return RiskRegime.RISK_OFF, float(p_short)
                elif p_long > 0.6:
                    return RiskRegime.RISK_ON, float(p_long)
                else:
                    return RiskRegime.NEUTRAL, 0.5
                    
        except Exception as e:
            logger.error(f"Risk detection error for {model_name}: {e}")
            return RiskRegime.UNKNOWN, 0.0
    
    def _compute_weighted_consensus(self, predictions: List[Dict]) -> Tuple[str, float, float, float]:
        """Compute weighted consensus from direction model predictions."""
        if not predictions:
            return "HOLD", 0.0, 0.0, 1.0
        
        # Get weights for each model
        weight_sum = 0
        weighted_votes = {"LONG": 0.0, "SHORT": 0.0, "HOLD": 0.0}
        weighted_probs = np.zeros(3)
        
        for pred in predictions:
            model_name = pred["model"]
            weight = self.model_weights.get(model_name, ModelWeight(
                model_name=model_name,
                expectancy=0, precision_on_trade=0.5,
                profit_factor=1.0, f1_directional=0.4, sharpe=0
            )).composite_weight
            
            weighted_votes[pred["action"]] += weight
            weighted_probs += weight * np.array(pred["probs"])
            weight_sum += weight
        
        # Normalize
        if weight_sum > 0:
            weighted_probs /= weight_sum
            for action in weighted_votes:
                weighted_votes[action] /= weight_sum
        
        # Determine consensus action
        consensus_action = max(weighted_votes, key=weighted_votes.get)
        weighted_agreement = weighted_votes[consensus_action]
        
        # Compute disagreement score (entropy of vote distribution)
        vote_probs = np.array(list(weighted_votes.values()))
        vote_probs = vote_probs / (vote_probs.sum() + 1e-8)
        disagreement = -np.sum(vote_probs * np.log(vote_probs + 1e-8)) / np.log(3)
        
        # Average confidence margin
        avg_margin = float(np.mean([p["confidence_margin"] for p in predictions]))
        
        return consensus_action, weighted_agreement, avg_margin, disagreement
    
    def predict(self, features: np.ndarray) -> EnsembleSignal:
        """
        Make ensemble prediction with regime gating and risk filtering.
        
        Steps:
        1. Get predictions from all direction models
        2. Compute weighted consensus
        3. Detect market regime from VAE (adjust thresholds)
        4. Detect risk regime from GNN (adjust position size)
        5. Apply gating logic
        6. Return final signal
        """
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Step 1: Get direction model predictions
        direction_predictions = []
        for name, model in self.direction_models.items():
            pred = self._get_model_prediction(model, x, name)
            if pred:
                direction_predictions.append(pred)
        
        # Step 2: Weighted consensus
        consensus_action, weighted_agreement, avg_margin, disagreement = \
            self._compute_weighted_consensus(direction_predictions)
        
        # Step 3: Regime detection from VAE
        market_regime = MarketRegime.UNKNOWN
        regime_confidence = 0.0
        for name, model in self.regime_models.items():
            regime, conf = self._detect_regime_from_vae(model, x, name)
            if conf > regime_confidence:
                market_regime = regime
                regime_confidence = conf
        
        # Step 4: Risk detection from GNN
        risk_regime = RiskRegime.UNKNOWN
        risk_confidence = 0.0
        for name, model in self.risk_models.items():
            risk, conf = self._detect_risk_from_gnn(model, x, name)
            if conf > risk_confidence:
                risk_regime = risk
                risk_confidence = conf
        
        # Step 5: Apply regime gating
        confidence_threshold = self.base_confidence_threshold
        margin_threshold = self.base_margin_threshold
        position_multiplier = 1.0
        regime_adjustment = "NONE"
        reasons = []
        
        # VAE regime gating
        if market_regime == MarketRegime.CHOPPY:
            confidence_threshold *= 1.5
            margin_threshold *= 1.5
            position_multiplier *= 0.5
            regime_adjustment = "RAISED_THRESHOLDS (choppy market)"
            reasons.append(f"VAE detects choppy market (conf={regime_confidence:.2f}) - raised thresholds")
        elif market_regime == MarketRegime.HIGH_VOLATILITY:
            position_multiplier *= 0.7
            regime_adjustment = "REDUCED_SIZE (high volatility)"
            reasons.append(f"VAE detects high volatility - reduced position size")
        elif market_regime == MarketRegime.TRENDING:
            position_multiplier *= 1.1
            reasons.append(f"VAE detects trending market - favorable conditions")
        
        # GNN risk gating
        if risk_regime == RiskRegime.CORRELATION_SHOCK:
            confidence_threshold *= 2.0
            position_multiplier *= 0.3
            reasons.append(f"GNN detects correlation shock - extreme caution")
        elif risk_regime == RiskRegime.RISK_OFF:
            if consensus_action == "LONG":
                confidence_threshold *= 1.3
            position_multiplier *= 0.7
            reasons.append(f"GNN detects risk-off regime - reduced exposure")
        elif risk_regime == RiskRegime.RISK_ON:
            if consensus_action == "SHORT":
                confidence_threshold *= 1.2
            reasons.append(f"GNN detects risk-on regime")
        
        # Step 6: Final decision
        # Compute ensemble probabilities
        if direction_predictions:
            ensemble_probs = np.mean([p["probs"] for p in direction_predictions], axis=0)
        else:
            ensemble_probs = np.array([0.2, 0.6, 0.2])
        
        confidence = float(ensemble_probs.max())
        sorted_probs = np.sort(ensemble_probs)[::-1]
        final_margin = float(sorted_probs[0] - sorted_probs[1])
        
        # Check thresholds
        passes_confidence = avg_margin >= confidence_threshold
        passes_agreement = weighted_agreement >= self.majority_weight_threshold
        passes_margin = final_margin >= margin_threshold
        
        final_action = consensus_action
        if consensus_action in ["LONG", "SHORT"]:
            if not (passes_confidence and passes_agreement and passes_margin):
                final_action = "HOLD"
                reasons.append(f"Gated: conf={passes_confidence}, agree={passes_agreement}, margin={passes_margin}")
        
        # Compute edge
        p_long = float(ensemble_probs[2])
        p_short = float(ensemble_probs[0])
        mu = (p_long - p_short) * 0.01
        cost = 0.001  # ~0.1% round-trip
        edge = abs(mu) - cost
        
        # Position sizing
        base_position = 0.05  # 5% base
        if final_action in ["LONG", "SHORT"]:
            position_size = base_position * confidence * position_multiplier
            position_size = max(0.01, min(0.10, position_size))  # 1-10% range
        else:
            position_size = 0.0
        
        regime_adjusted_size = position_size * position_multiplier
        
        # Model votes breakdown
        model_votes = {}
        for pred in direction_predictions:
            weight = self.model_weights.get(pred["model"], ModelWeight(
                model_name=pred["model"],
                expectancy=0, precision_on_trade=0.5,
                profit_factor=1.0, f1_directional=0.4, sharpe=0
            ))
            model_votes[pred["model"]] = {
                "action": pred["action"],
                "confidence": pred["confidence"],
                "confidence_margin": pred["confidence_margin"],
                "weight": weight.composite_weight,
                "probs": {
                    "SHORT": pred["probs"][0],
                    "HOLD": pred["probs"][1],
                    "LONG": pred["probs"][2]
                }
            }
        
        # Count agreement
        if direction_predictions:
            agreement_count = sum(1 for p in direction_predictions if p["action"] == consensus_action)
            agreement_pct = agreement_count / len(direction_predictions)
        else:
            agreement_pct = 0.0
        
        reasons.append(f"Consensus: {consensus_action} ({agreement_pct*100:.0f}% models, {weighted_agreement*100:.0f}% weight)")
        
        return EnsembleSignal(
            action=final_action,
            confidence=confidence,
            confidence_margin=final_margin,
            edge=edge,
            market_regime=market_regime.value,
            risk_regime=risk_regime.value,
            regime_confidence=regime_confidence,
            agreement_pct=agreement_pct,
            weighted_agreement=weighted_agreement,
            disagreement_score=disagreement,
            position_size_pct=position_size,
            regime_adjusted_size=regime_adjusted_size,
            confidence_threshold_used=confidence_threshold,
            regime_adjustment=regime_adjustment,
            model_votes=model_votes,
            reasons=reasons,
            ensemble_probs={
                "SHORT": float(ensemble_probs[0]),
                "HOLD": float(ensemble_probs[1]),
                "LONG": float(ensemble_probs[2])
            }
        )
