"""
Multi-Head Trainer for Institutional Trading Models

Trains models with:
1. Classification head (direction)
2. Regression head (expected return μ, uncertainty σ)
3. Quantile head (q10, q25, q50, q75, q90)

Uses combined loss with configurable weights.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, SequentialSampler, RandomSampler
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import time
import logging
from torch.utils.tensorboard import SummaryWriter

try:
    from .multihead_loss import MultiHeadLoss, MultiHeadLossConfig
    from ..models.multihead import MultiHeadOutput
    from ..utils.trade_gating import (
        TradeGateConfig, compute_trade_gate, apply_cooldown, 
        log_gate_statistics, GateFailure, DEFAULT_GATE_CONFIG
    )
    from .walk_forward import save_walk_forward_weights
except ImportError:
    # Fallback for direct script execution
    from training.multihead_loss import MultiHeadLoss, MultiHeadLossConfig
    from models.multihead import MultiHeadOutput
    from utils.trade_gating import (
        TradeGateConfig, compute_trade_gate, apply_cooldown,
        log_gate_statistics, GateFailure, DEFAULT_GATE_CONFIG
    )
    from training.walk_forward import save_walk_forward_weights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiHeadDataset(Dataset):
    """
    Dataset for multi-head training.
    
    Each sample contains:
    - features: [seq_len, input_dim] input sequence
    - class_label: int (0=SHORT, 1=HOLD, 2=LONG)
    - forward_return: float (actual return for regression/quantile targets)
    - trading_targets: [3] tensor (entry_offset, sl_distance, tp_distance)
    - candle_targets: [n_future, 3] tensor (close, high, low deltas)
    - regime_id: int (0=BULL, 1=BEAR, 2=HIGH_VOL, 3=LOW_VOL_CHOP) - for training balance only
    - vol_state: int (0=contraction, 1=neutral, 2=expansion) - Flow Forecast target
    - acceleration: float (momentum change) - Flow Forecast target
    """
    
    def __init__(
        self,
        features: np.ndarray,
        class_labels: np.ndarray,
        forward_returns: np.ndarray,
        entry_offset: Optional[np.ndarray] = None,
        sl_distance: Optional[np.ndarray] = None,
        tp_distance: Optional[np.ndarray] = None,
        candle_targets: Optional[np.ndarray] = None,
        regime_ids: Optional[np.ndarray] = None,
        vol_state: Optional[np.ndarray] = None,
        acceleration: Optional[np.ndarray] = None,
        n_future_candles: int = 5,
        sequence_length: int = 100
    ):
        self.features = features.astype(np.float32)
        self.class_labels = class_labels.astype(np.int64)
        self.forward_returns = forward_returns.astype(np.float32)
        self.sequence_length = sequence_length
        self.n_future_candles = n_future_candles
        
        # Trading targets (entry/SL/TP) - optional for backward compatibility
        # Handle both numpy arrays and scalars
        def to_float_array(val):
            if val is None:
                return None
            if isinstance(val, np.ndarray):
                return val.astype(np.float32)
            # Scalar - convert to array of same length as features
            return np.full(len(features), float(val), dtype=np.float32)
        
        self.entry_offset = to_float_array(entry_offset)
        self.sl_distance = to_float_array(sl_distance)
        self.tp_distance = to_float_array(tp_distance)
        
        # Candle prediction targets - optional for backward compatibility
        self.candle_targets = candle_targets.astype(np.float32) if candle_targets is not None else None
        
        # Regime IDs for balanced training (0=BULL, 1=BEAR, 2=HIGH_VOL, 3=LOW_VOL_CHOP)
        self.regime_ids = regime_ids.astype(np.int64) if regime_ids is not None else None
        
        # Flow Forecast targets - optional for backward compatibility
        self.vol_state = vol_state.astype(np.int64) if vol_state is not None else None
        self.acceleration = to_float_array(acceleration)
        
        # Create sequences
        self.valid_indices = list(range(sequence_length, len(features)))
        
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        actual_idx = self.valid_indices[idx]
        
        # Get sequence ending at actual_idx
        start_idx = actual_idx - self.sequence_length
        seq = self.features[start_idx:actual_idx]
        
        # Trading targets: [entry_offset, sl_distance, tp_distance]
        # Check all three are available before accessing
        if self.entry_offset is not None and self.sl_distance is not None and self.tp_distance is not None:
            trading = np.array([
                self.entry_offset[actual_idx],
                self.sl_distance[actual_idx],
                self.tp_distance[actual_idx]
            ], dtype=np.float32)
        else:
            trading = np.zeros(3, dtype=np.float32)
        
        # Candle targets: reshape to [n_future, 3]
        if self.candle_targets is not None:
            c = self.candle_targets[actual_idx]
            # Columns are: close_1, high_1, low_1, close_2, high_2, low_2, ...
            # Reshape to [n_future, 3] where 3 = (close, high, low)
            c = c.reshape(self.n_future_candles, 3)
        else:
            c = np.zeros((self.n_future_candles, 3), dtype=np.float32)
        
        # Flow Forecast targets
        vol_state = self.vol_state[actual_idx] if self.vol_state is not None else 1  # default neutral
        accel = self.acceleration[actual_idx] if self.acceleration is not None else 0.0
        
        return (
            torch.from_numpy(seq),
            torch.tensor(self.class_labels[actual_idx]),
            torch.tensor(self.forward_returns[actual_idx]),
            torch.from_numpy(trading),
            torch.from_numpy(c),
            torch.tensor(vol_state, dtype=torch.long),
            torch.tensor(accel, dtype=torch.float32)
        )
    
    def get_regime_ids_for_valid_indices(self) -> np.ndarray:
        """Get regime IDs only for valid indices (for creating balanced sampler)."""
        if self.regime_ids is None:
            # Default to LOW_VOL_CHOP (3) if no regime labels
            return np.full(len(self.valid_indices), 3, dtype=np.int64)
        return self.regime_ids[self.valid_indices]
    
    def get_regime_distribution(self) -> Dict[str, float]:
        """Get regime distribution for logging."""
        regime_ids = self.get_regime_ids_for_valid_indices()
        total = len(regime_ids)
        
        REGIME_NAMES = {0: "BULL", 1: "BEAR", 2: "HIGH_VOL", 3: "LOW_VOL_CHOP"}
        distribution = {}
        
        for regime_id, name in REGIME_NAMES.items():
            count = np.sum(regime_ids == regime_id)
            distribution[name] = count / total if total > 0 else 0.0
        
        return distribution


def create_regime_balanced_loader(
    dataset: 'MultiHeadDataset',
    batch_size: int = 64,
    num_workers: int = 4,
    target_balance: float = 0.25
) -> DataLoader:
    """
    Create a DataLoader with regime-balanced sampling.
    
    Uses WeightedRandomSampler to ensure ~25% of each regime in training batches.
    This prevents the model from overfitting to dominant market regimes.
    
    Args:
        dataset: MultiHeadDataset with regime_ids
        batch_size: Batch size
        num_workers: Number of data loader workers
        target_balance: Target proportion per regime (default 0.25 for 4 regimes)
        
    Returns:
        DataLoader with balanced sampling
    """
    regime_ids = dataset.get_regime_ids_for_valid_indices()
    total = len(regime_ids)
    
    # Calculate inverse frequency weights
    weights = np.ones(total, dtype=np.float64)
    
    for regime_id in range(4):  # 4 regimes
        count = np.sum(regime_ids == regime_id)
        if count > 0:
            actual_proportion = count / total
            regime_weight = target_balance / actual_proportion
            regime_weight = np.clip(regime_weight, 0.25, 4.0)
            mask = regime_ids == regime_id
            weights[mask] = regime_weight
    
    # Normalize weights
    weights = weights * (total / weights.sum())
    
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=total,
        replacement=True
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


class TrainingHealthMonitor:
    """
    Real-time Training Health Monitor for detecting training issues.
    
    Monitors for:
    1. Loss divergence (loss increasing over rolling window)
    2. Accuracy collapse (accuracy dropping significantly)
    3. Class distribution skew (model predicting only one class)
    4. Gradient explosion (large gradient norms)
    5. NaN/Inf values in loss or gradients
    
    Sends alerts to GUI via callback when issues detected.
    
    2024 Research: Early detection of training issues saves compute and prevents bad models.
    """
    
    def __init__(
        self, 
        window_size: int = 10,
        loss_divergence_threshold: float = 0.5,
        accuracy_drop_threshold: float = 0.15,
        class_skew_threshold: float = 0.85,
        gradient_explosion_threshold: float = 10.0,
        alert_callback: Optional[Callable[[str, str, Dict], None]] = None
    ):
        """
        Args:
            window_size: Rolling window for trend detection
            loss_divergence_threshold: Alert if loss increases by this fraction
            accuracy_drop_threshold: Alert if accuracy drops by this absolute amount
            class_skew_threshold: Alert if one class > this fraction of predictions
            gradient_explosion_threshold: Alert if gradient norm exceeds this
            alert_callback: Function(alert_type, message, details) to call on alerts
        """
        self.window_size = window_size
        self.loss_divergence_threshold = loss_divergence_threshold
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.class_skew_threshold = class_skew_threshold
        self.gradient_explosion_threshold = gradient_explosion_threshold
        self.alert_callback = alert_callback
        
        # History tracking
        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = []
        self.class_distribution_history: List[Dict[int, float]] = []
        self.gradient_norm_history: List[float] = []
        
        # Alert state (prevent spam)
        self.alerts_sent: Dict[str, int] = {}
        self.alert_cooldown = 5  # epochs between same alert type
        
        # Best values for comparison
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        
    def update(
        self,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        class_predictions: Optional[np.ndarray] = None,
        gradient_norm: Optional[float] = None
    ) -> List[Dict]:
        """
        Update monitor with epoch metrics and check for issues.
        
        Returns list of alert dictionaries if issues detected.
        """
        alerts = []
        
        # Check for NaN/Inf
        if np.isnan(train_loss) or np.isinf(train_loss):
            alerts.append(self._create_alert(
                epoch, "CRITICAL", "nan_loss",
                "Training loss is NaN/Inf! Training has diverged.",
                {"loss": train_loss}
            ))
        
        # Update histories
        self.loss_history.append(train_loss)
        self.accuracy_history.append(train_accuracy)
        
        # Track best values
        if train_loss < self.best_loss:
            self.best_loss = train_loss
        if train_accuracy > self.best_accuracy:
            self.best_accuracy = train_accuracy
        
        # Check loss divergence (loss increasing trend)
        if len(self.loss_history) >= self.window_size:
            recent_losses = self.loss_history[-self.window_size:]
            early_avg = np.mean(recent_losses[:self.window_size//2])
            late_avg = np.mean(recent_losses[self.window_size//2:])
            
            if early_avg > 0 and (late_avg - early_avg) / early_avg > self.loss_divergence_threshold:
                alerts.append(self._create_alert(
                    epoch, "WARNING", "loss_divergence",
                    f"Loss increasing: {early_avg:.4f} → {late_avg:.4f} (+{((late_avg-early_avg)/early_avg)*100:.1f}%)",
                    {"early_avg": early_avg, "late_avg": late_avg}
                ))
        
        # Check accuracy collapse
        if len(self.accuracy_history) >= self.window_size:
            peak_accuracy = max(self.accuracy_history[:-self.window_size//2]) if len(self.accuracy_history) > self.window_size else self.best_accuracy
            recent_accuracy = np.mean(self.accuracy_history[-self.window_size//2:])
            
            if peak_accuracy - recent_accuracy > self.accuracy_drop_threshold:
                alerts.append(self._create_alert(
                    epoch, "WARNING", "accuracy_drop",
                    f"Accuracy dropped: {peak_accuracy*100:.1f}% → {recent_accuracy*100:.1f}%",
                    {"peak": peak_accuracy, "current": recent_accuracy}
                ))
        
        # Check class distribution skew
        if class_predictions is not None:
            unique, counts = np.unique(class_predictions, return_counts=True)
            total = len(class_predictions)
            distribution = {int(u): c/total for u, c in zip(unique, counts)}
            self.class_distribution_history.append(distribution)
            
            max_class_ratio = max(distribution.values()) if distribution else 0
            if max_class_ratio > self.class_skew_threshold:
                majority_class = max(distribution, key=distribution.get)
                class_names = {0: "SHORT", 1: "HOLD", 2: "LONG"}
                alerts.append(self._create_alert(
                    epoch, "WARNING", "class_skew",
                    f"Model predicting mostly {class_names.get(majority_class, majority_class)}: {max_class_ratio*100:.1f}%",
                    {"distribution": distribution}
                ))
        
        # Check gradient explosion
        if gradient_norm is not None:
            self.gradient_norm_history.append(gradient_norm)
            if gradient_norm > self.gradient_explosion_threshold:
                alerts.append(self._create_alert(
                    epoch, "WARNING", "gradient_explosion",
                    f"Large gradient norm: {gradient_norm:.2f} (threshold: {self.gradient_explosion_threshold})",
                    {"gradient_norm": gradient_norm}
                ))
        
        # Send alerts via callback
        for alert in alerts:
            if self.alert_callback and self._should_send_alert(epoch, alert['type']):
                self.alert_callback(alert['severity'], alert['message'], alert)
        
        return alerts
    
    def _create_alert(self, epoch: int, severity: str, alert_type: str, message: str, details: Dict) -> Dict:
        """Create alert dictionary."""
        return {
            "epoch": epoch,
            "severity": severity,  # CRITICAL, WARNING, INFO
            "type": alert_type,
            "message": message,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
    
    def _should_send_alert(self, epoch: int, alert_type: str) -> bool:
        """Check if we should send this alert (cooldown logic)."""
        last_sent = self.alerts_sent.get(alert_type, -999)
        if epoch - last_sent >= self.alert_cooldown:
            self.alerts_sent[alert_type] = epoch
            return True
        return False
    
    def get_health_summary(self) -> Dict:
        """Get overall health summary."""
        issues = []
        
        if len(self.loss_history) >= 3:
            recent_loss = np.mean(self.loss_history[-3:])
            if recent_loss > self.best_loss * 1.5:
                issues.append("loss_elevated")
        
        if len(self.accuracy_history) >= 3:
            recent_acc = np.mean(self.accuracy_history[-3:])
            if recent_acc < self.best_accuracy * 0.8:
                issues.append("accuracy_degraded")
        
        if len(self.class_distribution_history) >= 1:
            recent_dist = self.class_distribution_history[-1]
            if max(recent_dist.values()) > self.class_skew_threshold:
                issues.append("class_imbalanced")
        
        return {
            "status": "HEALTHY" if not issues else "ISSUES_DETECTED",
            "issues": issues,
            "best_loss": self.best_loss,
            "best_accuracy": self.best_accuracy,
            "current_loss": self.loss_history[-1] if self.loss_history else None,
            "current_accuracy": self.accuracy_history[-1] if self.accuracy_history else None
        }


class MultiHeadTrainer:
    """
    Trainer for multi-head models.
    
    Handles:
    - Combined loss optimization
    - Class weighting for imbalanced data
    - Per-head metric tracking
    - Walk-forward validation
    - Training health monitoring with real-time alerts
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: str = "cuda",
        loss_config: Optional[MultiHeadLossConfig] = None,
        class_weights: Optional[torch.Tensor] = None,
        gui_mode: bool = False,
        feature_scaler = None,
        feature_columns: Optional[List[str]] = None,
        training_mode: str = "stf",
        horizon_periods: int = 16,
        health_alert_callback: Optional[Callable[[str, str, Dict], None]] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.gui_mode = gui_mode
        
        # Store training config for checkpoint saving
        self.feature_scaler = feature_scaler  # sklearn StandardScaler, not AMP GradScaler
        self.feature_columns = feature_columns
        self.training_mode = training_mode
        self.horizon_periods = horizon_periods
        
        # Setup loss
        if loss_config is None:
            loss_config = MultiHeadLossConfig(class_weights=class_weights)
        self.criterion = MultiHeadLoss(loss_config).to(device)
        
        # TRAINING HEALTH MONITOR - 2024 Best Practice
        # Raised gradient_explosion_threshold from 10.0 to 20.0 - with gradient clipping
        # at 1.0, pre-clip norms up to 15-20 are normal and clipping handles them
        self.health_monitor = TrainingHealthMonitor(
            window_size=10,
            loss_divergence_threshold=0.5,
            accuracy_drop_threshold=0.15,
            class_skew_threshold=0.85,
            gradient_explosion_threshold=20.0,  # Was 10.0, raised to reduce false alarms
            alert_callback=health_alert_callback
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Scheduler - reduced max_lr multiplier from 10x to 3x to prevent gradient explosion
        # Original: 10x caused gradient norms of 30-40 which destabilized training
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.training.learning_rate * 3,  # Was 10x, reduced to 3x
            epochs=config.training.epochs,
            steps_per_epoch=len(train_loader)
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0
        self.epoch_callback: Optional[Callable] = None
        
        # Logging
        log_dir = Path(config.training.log_dir) / "multihead"
        self.writer = SummaryWriter(log_dir)
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with multi-head outputs (all 8 heads)."""
        self.model.train()
        
        total_losses = {
            'total': 0.0, 'class': 0.0, 'mu': 0.0, 
            'sigma': 0.0, 'quantile': 0.0, 'trading': 0.0, 'candle': 0.0,
            'vol_state': 0.0, 'acceleration': 0.0, 'confidence_penalty': 0.0
        }
        correct = 0
        total = 0
        
        # Track gradient norms and predictions for health monitoring
        gradient_norms = []
        all_predictions = []
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            # Handle both 5-item (legacy) and 7-item (with flow forecast) batches
            if len(batch_data) == 7:
                features, class_labels, returns, trading, candle_tgt, vol_state_tgt, accel_tgt = batch_data
            else:
                # Legacy 5-item format
                features, class_labels, returns, trading, candle_tgt = batch_data
                vol_state_tgt = None
                accel_tgt = None
            
            features = features.to(self.device)
            class_labels = class_labels.to(self.device)
            returns = returns.to(self.device)
            trading = trading.to(self.device)  # [batch, 3]
            candle_tgt = candle_tgt.to(self.device)  # [batch, n_future, 3]
            
            # Flow Forecast targets
            if vol_state_tgt is not None:
                vol_state_tgt = vol_state_tgt.to(self.device)
            if accel_tgt is not None:
                accel_tgt = accel_tgt.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Multi-head forward pass
            output = self.model.forward_multihead(features)
            
            # Build trading_targets dict
            trading_targets = {
                "entry_offset": trading[:, 0:1],
                "sl_distance": trading[:, 1:2],
                "tp_distance": trading[:, 2:3],
            }
            
            # Compute combined loss (all 8 heads)
            losses = self.criterion(
                class_logits=output.class_logits,
                mu=output.mu,
                sigma=output.sigma,
                quantiles=output.quantiles,
                class_targets=class_labels,
                return_targets=returns,
                entry_offset=output.entry_offset,
                sl_distance=output.sl_distance,
                tp_distance=output.tp_distance,
                candle_deltas=output.candle_deltas,
                trading_targets=trading_targets,
                candle_targets=candle_tgt,
                vol_state_logits=output.vol_state_logits,
                vol_state_targets=vol_state_tgt,
                acceleration_pred=output.acceleration,
                acceleration_targets=accel_tgt
            )
            
            loss = losses['total']
            loss.backward()
            
            # Track gradient norm BEFORE clipping (for health monitoring)
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5
            gradient_norms.append(grad_norm)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Track losses
            for key in total_losses:
                if key in losses:
                    total_losses[key] += losses[key].item()
            
            # Track accuracy and predictions for health monitoring
            preds = output.class_logits.argmax(dim=-1)
            correct += (preds == class_labels).sum().item()
            total += len(class_labels)
            all_predictions.extend(preds.cpu().numpy().tolist())
            
            self.global_step += 1
            
        # Average losses
        n_batches = len(self.train_loader)
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        avg_losses['accuracy'] = correct / total
        
        # Compute average gradient norm for this epoch
        avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0.0
        avg_losses['gradient_norm'] = avg_grad_norm
        
        # HEALTH MONITORING: Check for training issues
        alerts = self.health_monitor.update(
            epoch=epoch,
            train_loss=avg_losses['total'],
            train_accuracy=avg_losses['accuracy'],
            class_predictions=np.array(all_predictions) if all_predictions else None,
            gradient_norm=avg_grad_norm
        )
        
        # Log any alerts
        for alert in alerts:
            if alert['severity'] == 'CRITICAL':
                logger.error(f"[HEALTH CRITICAL] {alert['message']}")
            else:
                logger.warning(f"[HEALTH WARNING] {alert['message']}")
        
        return avg_losses
    
    def validate(self, epoch: int = 0) -> Dict[str, float]:
        """Validate with all heads (8 heads).
        
        Args:
            epoch: Current training epoch (used for monitoring sweep interval)
        """
        self.model.eval()
        
        total_losses = {
            'total': 0.0, 'class': 0.0, 'mu': 0.0,
            'sigma': 0.0, 'quantile': 0.0, 'trading': 0.0, 'candle': 0.0,
            'vol_state': 0.0, 'acceleration': 0.0
        }
        correct = 0
        total = 0
        
        # Per-class tracking
        class_correct = {0: 0, 1: 0, 2: 0}
        class_total = {0: 0, 1: 0, 2: 0}
        
        # Quantile calibration tracking
        quantile_below = torch.zeros(5)  # How often target < predicted quantile
        quantile_count = 0
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                # Handle both 5-item (legacy) and 7-item (with flow forecast) batches
                if len(batch_data) == 7:
                    features, class_labels, returns, trading, candle_tgt, vol_state_tgt, accel_tgt = batch_data
                else:
                    features, class_labels, returns, trading, candle_tgt = batch_data
                    vol_state_tgt = None
                    accel_tgt = None
                
                features = features.to(self.device)
                class_labels = class_labels.to(self.device)
                returns = returns.to(self.device)
                trading = trading.to(self.device)
                candle_tgt = candle_tgt.to(self.device)
                
                # Flow Forecast targets
                if vol_state_tgt is not None:
                    vol_state_tgt = vol_state_tgt.to(self.device)
                if accel_tgt is not None:
                    accel_tgt = accel_tgt.to(self.device)
                
                output = self.model.forward_multihead(features)
                
                # Build trading_targets dict
                trading_targets = {
                    "entry_offset": trading[:, 0:1],
                    "sl_distance": trading[:, 1:2],
                    "tp_distance": trading[:, 2:3],
                }
                
                losses = self.criterion(
                    class_logits=output.class_logits,
                    mu=output.mu,
                    sigma=output.sigma,
                    quantiles=output.quantiles,
                    class_targets=class_labels,
                    return_targets=returns,
                    entry_offset=output.entry_offset,
                    sl_distance=output.sl_distance,
                    tp_distance=output.tp_distance,
                    candle_deltas=output.candle_deltas,
                    trading_targets=trading_targets,
                    candle_targets=candle_tgt,
                    vol_state_logits=output.vol_state_logits,
                    vol_state_targets=vol_state_tgt,
                    acceleration_pred=output.acceleration,
                    acceleration_targets=accel_tgt
                )
                
                for key in total_losses:
                    if key in losses:
                        total_losses[key] += losses[key].item()
                
                # Classification accuracy
                preds = output.class_logits.argmax(dim=-1)
                correct += (preds == class_labels).sum().item()
                total += len(class_labels)
                
                # Per-class accuracy
                for c in [0, 1, 2]:
                    mask = class_labels == c
                    class_correct[c] += (preds[mask] == c).sum().item()
                    class_total[c] += mask.sum().item()
                
                # Quantile calibration
                returns_expanded = returns.unsqueeze(-1).expand_as(output.quantiles)
                quantile_below += (returns_expanded < output.quantiles).float().sum(dim=0).cpu()
                quantile_count += len(returns)
        
        n_batches = len(self.val_loader)
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        avg_losses['accuracy'] = correct / total
        
        # Per-class metrics
        for c, name in [(0, 'short'), (1, 'hold'), (2, 'long')]:
            if class_total[c] > 0:
                avg_losses[f'acc_{name}'] = class_correct[c] / class_total[c]
            else:
                avg_losses[f'acc_{name}'] = 0.0
        
        # Quantile calibration (should be ~[0.1, 0.25, 0.5, 0.75, 0.9])
        if quantile_count > 0:
            calibration = quantile_below / quantile_count
            avg_losses['q10_cal'] = calibration[0].item()
            avg_losses['q25_cal'] = calibration[1].item()
            avg_losses['q50_cal'] = calibration[2].item()
            avg_losses['q75_cal'] = calibration[3].item()
            avg_losses['q90_cal'] = calibration[4].item()
        
        # Compute trading-aware metrics (monitoring only, runs every N epochs)
        trading_metrics = self._compute_trading_metrics(epoch=epoch)
        avg_losses.update(trading_metrics)
        
        # Compute per-regime metrics (if regime_ids available)
        regime_metrics = self._compute_regime_metrics()
        if regime_metrics:
            avg_losses['regime_metrics'] = regime_metrics
        
        return avg_losses
    
    def _compute_trading_metrics(self, epoch: int = 0) -> Dict[str, float]:
        """
        MONITORING SWEEP: Compute trading metrics for training observability.
        
        NOTE: This is for MONITORING ONLY during training.
        - Does NOT save or freeze any policy
        - Does NOT alter training behavior
        - Runs every MONITORING_EPOCH_INTERVAL epochs to save time
        
        For the actual frozen execution policy, use the dedicated
        post-training PolicySelector class after training completes.
        
        Returns:
            Dictionary of trading metrics (informational only)
        """
        # Only run monitoring sweep every N epochs to save time
        MONITORING_EPOCH_INTERVAL = 5
        if epoch > 0 and epoch % MONITORING_EPOCH_INTERVAL != 0:
            return {
                'expectancy': 0.0, 'hit_rate': 0.0, 'profit_factor': 0.0,
                'sharpe': 0.0, 'max_drawdown': 0.0, 'num_trades': 0,
                'avg_win': 0.0, 'avg_loss': 0.0, 'win_loss_ratio': 0.0,
                'risk_adjusted_score': 0.0,
                'min_confidence': 0.0, 'spread_multiplier': 0.0, 'cooldown': 0,
                '_skipped': True
            }
        # Trading policy parameters
        FIXED_COST = 0.0009  # 0.09% round-trip cost
        SPREAD_MULTIPLIER = 3.0  # K: require spread >= K * cost
        COOLDOWN = 8  # Bars to wait after a trade (horizon/2)
        
        # Confidence thresholds to sweep - matched to actual distribution
        # (confidence max ~0.43, mean ~0.16, so old [0.3-1.1] was too high)
        CONFIDENCE_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
        
        # Collect all model outputs
        all_predictions = []
        all_returns = []
        all_mus = []
        all_sigmas = []
        all_q10 = []
        all_q25 = []
        all_q75 = []
        all_q90 = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Handle both old (5 values) and new (7 values with vol_state, accel) formats
                features = batch[0].to(self.device)
                returns = batch[2]
                output = self.model.forward_multihead(features)
                
                probs = torch.softmax(output.class_logits, dim=-1)
                preds = probs.argmax(dim=-1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_returns.extend(returns.cpu().numpy())
                all_mus.extend(output.mu.squeeze().cpu().numpy())
                all_sigmas.extend(output.sigma.squeeze().cpu().numpy())
                
                # Extract quantiles if available
                if output.quantiles is not None:
                    quantiles = output.quantiles.cpu().numpy()
                    all_q10.extend(quantiles[:, 0])  # q10
                    all_q25.extend(quantiles[:, 1])  # q25
                    all_q75.extend(quantiles[:, 3])  # q75
                    all_q90.extend(quantiles[:, 4])  # q90
        
        preds = np.array(all_predictions)
        returns = np.array(all_returns)
        mus = np.array(all_mus)
        raw_sigmas = np.array(all_sigmas)
        
        # === DIAGNOSTIC: Class prediction distribution ===
        n_total = len(preds)
        n_short = (preds == 0).sum()  # SHORT
        n_hold = (preds == 1).sum()   # HOLD
        n_long = (preds == 2).sum()   # LONG
        logger.info("=" * 70)
        logger.info("PREDICTION DISTRIBUTION (epoch %d):", epoch)
        logger.info("  SHORT (0): %5d / %d (%.1f%%)", n_short, n_total, 100*n_short/n_total if n_total > 0 else 0)
        logger.info("  HOLD  (1): %5d / %d (%.1f%%)", n_hold, n_total, 100*n_hold/n_total if n_total > 0 else 0)
        logger.info("  LONG  (2): %5d / %d (%.1f%%)", n_long, n_total, 100*n_long/n_total if n_total > 0 else 0)
        if n_short + n_long == 0:
            logger.warning(">>> MODEL PREDICTS 100%% HOLD - NO TRADES POSSIBLE <<<")
        elif (n_short + n_long) / n_total < 0.05:
            logger.warning(">>> MODEL PREDICTS %.1f%% DIRECTIONAL - VERY FEW TRADES <<<", 
                          100*(n_short + n_long)/n_total)
        logger.info("=" * 70)
        
        # === CRITICAL FIX: Convert log_sigma to sigma ===
        # Model outputs log_sigma when use_log_sigma=True (default)
        # sigma = exp(log_sigma)
        # If log_sigma is negative (typical), exp() gives values in (0, 1)
        if hasattr(self.model, 'use_log_sigma') and self.model.use_log_sigma:
            logger.info("CONF_DEBUG | Converting log_sigma to sigma (exp)")
            sigmas = np.exp(np.clip(raw_sigmas, -10, 10))  # Clip to prevent overflow
            logger.info("CONF_DEBUG | log_sigma range: [%.4f, %.4f], sigma range: [%.6f, %.6f]",
                       raw_sigmas.min(), raw_sigmas.max(), sigmas.min(), sigmas.max())
        else:
            sigmas = raw_sigmas
        
        # Handle quantiles (use mu-based fallback if not available)
        if len(all_q10) > 0:
            q10 = np.array(all_q10)
            q25 = np.array(all_q25)
            q75 = np.array(all_q75)
            q90 = np.array(all_q90)
        else:
            # Fallback: approximate quantiles from mu and sigma
            q10 = mus - 1.28 * sigmas
            q25 = mus - 0.67 * sigmas
            q75 = mus + 0.67 * sigmas
            q90 = mus + 1.28 * sigmas
        
        # Compute spread and confidence for all samples
        spread = q75 - q25  # Distribution width
        confidence = np.abs(mus) / np.maximum(sigmas, 1e-6)  # |mu| / sigma
        
        # === CONF_DEBUG: Log confidence distribution ===
        logger.info("CONF_DEBUG | Confidence stats: min=%.4f, max=%.4f, mean=%.4f, median=%.4f",
                   confidence.min(), confidence.max(), confidence.mean(), np.median(confidence))
        logger.info("CONF_DEBUG | Sigma stats: min=%.6f, max=%.6f, mean=%.6f", 
                   sigmas.min(), sigmas.max(), sigmas.mean())
        logger.info("CONF_DEBUG | Mu stats: min=%.6f, max=%.6f, mean=%.6f",
                   mus.min(), mus.max(), mus.mean())
        logger.info("CONF_DEBUG | Spread stats: min=%.6f, max=%.6f, mean=%.6f",
                   spread.min(), spread.max(), spread.mean())
        
        # Base trade signals (LONG=2, SHORT=0)
        long_signal = preds == 2
        short_signal = preds == 0
        directional_signal = long_signal | short_signal
        
        logger.info("CONF_DEBUG | Base directional signals: %d / %d samples (%.1f%%)",
                   directional_signal.sum(), len(directional_signal), 
                   100 * directional_signal.sum() / len(directional_signal))
        
        # === NEW: Minimum predicted-move filter ===
        # If abs(mu) < MIN_MOVE_FACTOR × sigma → no trade (insufficient edge)
        # NOTE: mu values are typically much smaller than sigma (mu~0.001, sigma~0.01)
        # Use a low factor (0.10) to filter only the weakest predictions
        MIN_MOVE_FACTOR = 0.10
        move_gate = np.abs(mus) >= (MIN_MOVE_FACTOR * sigmas)
        n_move_pass = move_gate.sum()
        logger.info("MOVE_GATE | abs(mu) >= %.2f×sigma: %d / %d pass (%.1f%%)",
                   MIN_MOVE_FACTOR, n_move_pass, len(move_gate), 
                   100 * n_move_pass / len(move_gate) if len(move_gate) > 0 else 0)
        logger.info("MOVE_GATE | mu_range=[%.6f, %.6f], sigma_range=[%.6f, %.6f], threshold=%.6f",
                   mus.min(), mus.max(), sigmas.min(), sigmas.max(), 
                   MIN_MOVE_FACTOR * sigmas.mean())
        
        # === SWEEP CONFIDENCE THRESHOLDS TO FIND BEST POLICY ===
        MIN_TRADES = 30  # Minimum trades for policy eligibility
        
        best_metrics = None
        best_score = float('-inf')
        best_threshold = 0.5
        prev_trade_count = float('inf')  # For monotonicity check
        
        sweep_results = []
        
        for min_conf in CONFIDENCE_THRESHOLDS:
            # === GATE ORDER: spread -> confidence -> direction -> cooldown -> trade ===
            # Gate 1: Spread gate - sufficient price movement opportunity
            spread_gate = spread >= (SPREAD_MULTIPLIER * FIXED_COST)
            # Gate 2: Confidence gate - sufficient signal strength
            conf_gate = confidence >= min_conf
            # Gate 3: Direction gate - model predicts LONG or SHORT (not HOLD)
            # (directional_signal already computed above)
            
            # === CONF_DEBUG: Per-threshold logging ===
            n_spread_pass = spread_gate.sum()
            n_conf_pass = conf_gate.sum()
            n_directional = directional_signal.sum()
            n_dir_and_spread = (directional_signal & spread_gate).sum()
            n_dir_and_conf = (directional_signal & conf_gate).sum()
            n_move_and_dir = (move_gate & directional_signal).sum()
            
            logger.info("CONF_DEBUG | threshold=%.2f | spread_pass=%d, conf_pass=%d, move_pass=%d, "
                       "dir=%d, dir&spread=%d, dir&conf=%d, dir&move=%d",
                       min_conf, n_spread_pass, n_conf_pass, n_move_pass,
                       n_directional, n_dir_and_spread, n_dir_and_conf, n_move_and_dir)
            
            # Combined gates (order: spread -> confidence -> direction -> move_filter)
            # move_gate: abs(mu) >= 0.5 × sigma (minimum predicted move)
            trade_allowed = spread_gate & conf_gate & directional_signal & move_gate
            n_trade_allowed = trade_allowed.sum()
            
            # Gate 4: Cooldown - prevent overtrading
            final_trades = self._apply_cooldown(trade_allowed, COOLDOWN)
            n_final = final_trades.sum()
            
            # Validate monotonicity: trades should decrease as threshold increases
            if n_final > prev_trade_count:
                logger.warning("MONOTONICITY VIOLATION: threshold=%.2f has %d trades > prev %d",
                             min_conf, n_final, prev_trade_count)
            prev_trade_count = n_final
            
            logger.info("CONF_DEBUG | threshold=%.2f | trade_allowed=%d, after_cooldown=%d | %s",
                       min_conf, n_trade_allowed, n_final,
                       "PASS" if n_final > 0 else "FAIL (no trades)")
            
            # Compute PnL with ATR-based asymmetric SL/TP
            # SL = 1.5 × sigma, TP = 2.2 × sigma, min RR >= 1.5
            metrics = self._compute_pnl_with_atr_exits(
                final_trades, long_signal, short_signal, returns, 
                mus, sigmas, FIXED_COST
            )
            metrics['min_confidence'] = min_conf
            metrics['spread_multiplier'] = SPREAD_MULTIPLIER
            metrics['cooldown'] = COOLDOWN
            
            # Compute risk-adjusted score:
            # score = expectancy - 0.5*max_drawdown (or -0.25*abs(avg_loss) if no DD)
            max_dd = metrics.get('max_drawdown', 0.0)
            avg_loss = metrics.get('avg_loss', 0.0)
            if max_dd > 0:
                risk_penalty = 0.5 * max_dd
            else:
                risk_penalty = 0.25 * abs(avg_loss)
            
            risk_adjusted_score = metrics['expectancy'] - risk_penalty
            metrics['risk_adjusted_score'] = risk_adjusted_score
            
            sweep_results.append(metrics)
            
            # Track best - require MIN_TRADES and use risk-adjusted score
            if metrics['num_trades'] >= MIN_TRADES and risk_adjusted_score > best_score:
                best_score = risk_adjusted_score
                best_metrics = metrics
                best_threshold = min_conf
        
        # Log MONITORING sweep report (informational only - not for live trading)
        logger.info("=" * 60)
        logger.info("MONITORING SWEEP (epoch %d) - spread_K=%.1f, cooldown=%d, min_trades=%d", 
                   epoch, SPREAD_MULTIPLIER, COOLDOWN, MIN_TRADES)
        logger.info("NOTE: This is for MONITORING ONLY. Use PolicySelector for frozen live policy.")
        logger.info("-" * 60)
        for m in sweep_results:
            eligible = m['num_trades'] >= MIN_TRADES
            is_best = m['min_confidence'] == best_threshold and eligible and best_score > float('-inf')
            status = "★ MONITORING BEST" if is_best else ("" if eligible else "(ineligible)")
            logger.info(
                f"conf>={m['min_confidence']:.2f}: Trades={m['num_trades']:4d}, "
                f"Exp={m['expectancy']:+.4f}, Score={m['risk_adjusted_score']:+.4f}, "
                f"Hit={m['hit_rate']:.1%}, MaxDD={m['max_drawdown']:.4f}, "
                f"Sharpe={m['sharpe']:+.2f} {status}"
            )
        logger.info("=" * 60)
        
        # Return best metrics for monitoring (NOT saved as policy)
        if best_metrics is None:
            best_metrics = sweep_results[-1] if sweep_results else {
                'expectancy': 0.0, 'hit_rate': 0.0, 'profit_factor': 0.0,
                'sharpe': 0.0, 'max_drawdown': 0.0, 'num_trades': 0,
                'avg_win': 0.0, 'avg_loss': 0.0, 'win_loss_ratio': 0.0,
                'risk_adjusted_score': 0.0,
                'min_confidence': 0.5, 'spread_multiplier': 3.0, 'cooldown': 8
            }
        
        logger.info(f"MONITORING: Best observed - Score: {best_metrics.get('risk_adjusted_score', 0):.4f}, "
                   f"Exp: {best_metrics['expectancy']:.4f}, "
                   f"Trades: {best_metrics['num_trades']}")
        
        return best_metrics
    
    def _apply_cooldown(self, trade_signals: np.ndarray, cooldown: int) -> np.ndarray:
        """
        Apply cooldown to prevent signal spam.
        After taking a trade, no new trades for `cooldown` candles.
        
        Args:
            trade_signals: Boolean array of trade signals
            cooldown: Number of bars to wait after a trade
            
        Returns:
            Filtered trade signals with cooldown applied
        """
        result = np.zeros_like(trade_signals, dtype=bool)
        last_trade_idx = -cooldown - 1  # Start with no cooldown active
        
        for i in range(len(trade_signals)):
            if trade_signals[i] and (i - last_trade_idx) > cooldown:
                result[i] = True
                last_trade_idx = i
        
        return result
    
    def _compute_pnl_with_quantile_exits(
        self, 
        final_trades: np.ndarray,
        long_signal: np.ndarray,
        short_signal: np.ndarray,
        returns: np.ndarray,
        q10: np.ndarray,
        q25: np.ndarray,
        q75: np.ndarray,
        q90: np.ndarray,
        cost: float
    ) -> Dict[str, float]:
        """
        Compute PnL using asymmetric SL/TP derived from quantiles.
        
        For LONG trades:
            - SL distance from q10 (downside risk)
            - TP from q75 or q90 (upside potential)
        For SHORT trades:
            - SL distance from q90 (upside risk)
            - TP from q10 or q25 (downside potential)
        
        This ensures proper risk:reward asymmetry.
        """
        metrics = {}
        
        # Get trades
        long_trades = final_trades & long_signal
        short_trades = final_trades & short_signal
        trade_mask = long_trades | short_trades
        
        num_trades = trade_mask.sum()
        if num_trades == 0:
            return {
                'expectancy': 0.0, 'hit_rate': 0.0, 'profit_factor': 0.0,
                'sharpe': 0.0, 'max_drawdown': 0.0, 'num_trades': 0,
                'avg_win': 0.0, 'avg_loss': 0.0, 'win_loss_ratio': 0.0
            }
        
        # Compute PnL with asymmetric exits
        # For simulation, we still use actual returns but cap based on quantiles
        trade_pnl = np.zeros(len(returns))
        
        for i in range(len(returns)):
            if long_trades[i]:
                # LONG trade: profit if price goes up
                actual_ret = returns[i]
                sl_level = q10[i]  # Stop at q10
                tp_level = q75[i]  # Take profit at q75
                
                # Simulate exit: hit SL if return goes below q10, hit TP if above q75
                if actual_ret <= sl_level:
                    pnl = sl_level - cost  # Stopped out
                elif actual_ret >= tp_level:
                    pnl = tp_level - cost  # Take profit hit
                else:
                    pnl = actual_ret - cost  # Normal exit
                trade_pnl[i] = pnl
                
            elif short_trades[i]:
                # SHORT trade: profit if price goes down
                actual_ret = returns[i]
                sl_level = q90[i]  # Stop at q90 (price going up = bad)
                tp_level = q25[i]  # Take profit at q25 (price going down = good)
                
                # For short: we profit when price goes down (negative return)
                # SL triggers if return > q90, TP if return < q25
                if actual_ret >= sl_level:
                    pnl = -sl_level - cost  # Stopped out
                elif actual_ret <= tp_level:
                    pnl = -tp_level - cost  # Take profit hit
                else:
                    pnl = -actual_ret - cost  # Normal exit
                trade_pnl[i] = pnl
        
        # Filter to actual trades
        trade_returns = trade_pnl[trade_mask]
        num_trades = len(trade_returns)
        
        # Expectancy
        metrics['expectancy'] = float(np.mean(trade_returns)) if num_trades > 0 else 0.0
        
        # Hit rate
        wins = (trade_returns > 0).sum()
        metrics['hit_rate'] = float(wins / num_trades) if num_trades > 0 else 0.0
        
        # Profit factor
        gross_profits = trade_returns[trade_returns > 0].sum()
        gross_losses = abs(trade_returns[trade_returns < 0].sum())
        metrics['profit_factor'] = float(gross_profits / gross_losses) if gross_losses > 0 else 0.0
        
        # Sharpe ratio
        if num_trades > 1 and np.std(trade_returns) > 0:
            annual_factor = np.sqrt(2190)  # ~6 trades/day
            sharpe = (np.mean(trade_returns) / np.std(trade_returns)) * annual_factor
            metrics['sharpe'] = float(sharpe)
        else:
            metrics['sharpe'] = 0.0
        
        # Max drawdown
        cumulative = np.cumsum(trade_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        metrics['max_drawdown'] = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
        
        # Number of trades
        metrics['num_trades'] = int(num_trades)
        
        # Average win / loss
        if wins > 0:
            metrics['avg_win'] = float(np.mean(trade_returns[trade_returns > 0]))
        else:
            metrics['avg_win'] = 0.0
        
        losses_count = (trade_returns < 0).sum()
        if losses_count > 0:
            metrics['avg_loss'] = float(np.mean(trade_returns[trade_returns < 0]))
        else:
            metrics['avg_loss'] = 0.0
        
        # Win/loss ratio
        if metrics['avg_loss'] != 0:
            metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss'])
        else:
            metrics['win_loss_ratio'] = 0.0
        
        return metrics
    
    def _compute_pnl_with_atr_exits(
        self, 
        final_trades: np.ndarray,
        long_signal: np.ndarray,
        short_signal: np.ndarray,
        returns: np.ndarray,
        mus: np.ndarray,
        sigmas: np.ndarray,
        cost: float
    ) -> Dict[str, float]:
        """
        Compute PnL using ATR-based asymmetric SL/TP.
        
        Uses sigma (predicted volatility) as ATR proxy:
        - SL = 1.5 × sigma (stop loss distance)
        - TP = 2.2 × sigma (take profit distance)
        - Enforces minimum RR >= 1.5
        
        This provides proper risk:reward asymmetry based on volatility.
        """
        # ATR-based exit parameters
        SL_ATR_MULT = 1.5   # Stop loss = 1.5 × ATR (sigma)
        TP_ATR_MULT = 2.2   # Take profit = 2.2 × ATR (sigma)
        MIN_RR = 1.5        # Minimum risk:reward ratio
        
        metrics = {}
        
        # Get trades
        long_trades = final_trades & long_signal
        short_trades = final_trades & short_signal
        trade_mask = long_trades | short_trades
        
        num_trades = trade_mask.sum()
        if num_trades == 0:
            return {
                'expectancy': 0.0, 'hit_rate': 0.0, 'profit_factor': 0.0,
                'sharpe': 0.0, 'max_drawdown': 0.0, 'num_trades': 0,
                'avg_win': 0.0, 'avg_loss': 0.0, 'win_loss_ratio': 0.0
            }
        
        # Compute PnL with ATR-based asymmetric exits
        trade_pnl = np.zeros(len(returns))
        
        for i in range(len(returns)):
            if not (long_trades[i] or short_trades[i]):
                continue
                
            # Use sigma as ATR proxy (represents volatility/uncertainty)
            atr = max(sigmas[i], 1e-6)  # Prevent division by zero
            
            # Compute SL and TP distances
            sl_distance = SL_ATR_MULT * atr
            tp_distance = TP_ATR_MULT * atr
            
            # Enforce minimum RR ratio
            if tp_distance < MIN_RR * sl_distance:
                tp_distance = MIN_RR * sl_distance
            
            actual_ret = returns[i]
            
            if long_trades[i]:
                # LONG trade: profit if price goes up
                # SL triggers if return goes below -sl_distance
                # TP triggers if return goes above +tp_distance
                if actual_ret <= -sl_distance:
                    pnl = -sl_distance - cost  # Stopped out (loss)
                elif actual_ret >= tp_distance:
                    pnl = tp_distance - cost   # Take profit hit (win)
                else:
                    pnl = actual_ret - cost    # Normal exit
                trade_pnl[i] = pnl
                
            elif short_trades[i]:
                # SHORT trade: profit if price goes down
                # SL triggers if return goes above +sl_distance (price up = bad)
                # TP triggers if return goes below -tp_distance (price down = good)
                if actual_ret >= sl_distance:
                    pnl = -sl_distance - cost  # Stopped out (loss)
                elif actual_ret <= -tp_distance:
                    pnl = tp_distance - cost   # Take profit hit (win)
                else:
                    pnl = -actual_ret - cost   # Normal exit (profit when price down)
                trade_pnl[i] = pnl
        
        # Filter to actual trades
        trade_returns = trade_pnl[trade_mask]
        num_trades = len(trade_returns)
        
        # Expectancy
        metrics['expectancy'] = float(np.mean(trade_returns)) if num_trades > 0 else 0.0
        
        # Hit rate
        wins = (trade_returns > 0).sum()
        metrics['hit_rate'] = float(wins / num_trades) if num_trades > 0 else 0.0
        
        # Profit factor
        gross_profits = trade_returns[trade_returns > 0].sum()
        gross_losses = abs(trade_returns[trade_returns < 0].sum())
        metrics['profit_factor'] = float(gross_profits / gross_losses) if gross_losses > 0 else 0.0
        
        # Sharpe ratio
        if num_trades > 1 and np.std(trade_returns) > 0:
            annual_factor = np.sqrt(2190)  # ~6 trades/day
            sharpe = (np.mean(trade_returns) / np.std(trade_returns)) * annual_factor
            metrics['sharpe'] = float(sharpe)
        else:
            metrics['sharpe'] = 0.0
        
        # Max drawdown
        cumulative = np.cumsum(trade_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        metrics['max_drawdown'] = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
        
        # Number of trades
        metrics['num_trades'] = int(num_trades)
        
        # Average win / loss
        if wins > 0:
            metrics['avg_win'] = float(np.mean(trade_returns[trade_returns > 0]))
        else:
            metrics['avg_win'] = 0.0
        
        losses_count = (trade_returns < 0).sum()
        if losses_count > 0:
            metrics['avg_loss'] = float(np.mean(trade_returns[trade_returns < 0]))
        else:
            metrics['avg_loss'] = 0.0
        
        # Win/loss ratio
        if metrics['avg_loss'] != 0:
            metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss'])
        else:
            metrics['win_loss_ratio'] = 0.0
        
        return metrics
    
    def _compute_regime_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute per-regime trading metrics for validation.
        
        Tracks performance separately for:
        - BULL (0): Trending up markets
        - BEAR (1): Trending down markets  
        - HIGH_VOL (2): High volatility periods
        - LOW_VOL_CHOP (3): Low volatility ranging
        
        IMPORTANT: This method assumes val_loader iterates in sequential order
        (shuffle=False, no random sampler). This is enforced by the training
        pipeline which only uses balanced sampling for train_loader, not val_loader.
        
        Returns:
            Dictionary mapping regime name to metrics dict
        """
        REGIME_NAMES = {0: "BULL", 1: "BEAR", 2: "HIGH_VOL", 3: "LOW_VOL_CHOP"}
        
        # Check if val_loader dataset has regime_ids
        val_dataset = self.val_loader.dataset
        if not hasattr(val_dataset, 'regime_ids') or val_dataset.regime_ids is None:
            return {}  # No regime data available
        
        # CRITICAL: Verify val_loader uses sequential ordering (not shuffled/random sampler)
        # Regime ID alignment depends on deterministic sequential iteration
        sampler = self.val_loader.sampler
        
        # STRICT CHECK: Require SequentialSampler for regime ID alignment
        # Any non-sequential sampler will cause misalignment between predictions and regime IDs
        if not isinstance(sampler, SequentialSampler):
            sampler_name = type(sampler).__name__
            logger.info(f"val_loader uses {sampler_name} (not SequentialSampler) - skipping per-regime metrics for alignment safety")
            return {}
        
        # Pre-compute full regime IDs array for all valid indices (sequential order)
        all_regime_ids_precomputed = val_dataset.get_regime_ids_for_valid_indices()
        expected_samples = len(val_dataset)
        
        all_predictions = []
        all_returns = []
        total_processed = 0
        
        cost = 0.001  # 0.1% round-trip cost
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Handle both 5-item (legacy) and 7-item (with flow forecast) batches
                features = batch[0].to(self.device)
                returns = batch[2]  # returns is always at index 2
                output = self.model.forward_multihead(features)
                
                probs = torch.softmax(output.class_logits, dim=-1)
                preds = probs.argmax(dim=-1)
                
                batch_size = len(returns)
                all_predictions.extend(preds.cpu().numpy())
                all_returns.extend(returns.cpu().numpy())
                total_processed += batch_size
        
        preds = np.array(all_predictions)
        returns = np.array(all_returns)
        
        # STRICT ALIGNMENT: Processed samples must match available regime IDs
        if total_processed != len(all_regime_ids_precomputed):
            # Allow for drop_last=True which drops incomplete final batch
            batch_size = self.val_loader.batch_size or 1
            tolerance = batch_size  # At most one batch can be dropped
            difference = abs(total_processed - len(all_regime_ids_precomputed))
            
            if difference > tolerance:
                logger.error(f"Regime ID mismatch: processed {total_processed} samples but "
                            f"{len(all_regime_ids_precomputed)} regime IDs available (diff={difference})")
                return {}
            else:
                logger.info(f"Minor sample count difference ({difference}) likely from drop_last, proceeding")
        
        # Use precomputed regime IDs, sliced to match number of processed samples
        regime_ids = all_regime_ids_precomputed[:len(preds)]
        
        # Final strict length check
        if len(preds) != len(regime_ids):
            logger.error(f"Length mismatch after slicing: {len(preds)} predictions vs {len(regime_ids)} regime IDs")
            return {}
        
        regime_metrics = {}
        
        for regime_id, regime_name in REGIME_NAMES.items():
            regime_mask = regime_ids == regime_id
            regime_count = regime_mask.sum()
            
            if regime_count == 0:
                regime_metrics[regime_name] = {
                    'samples': 0, 'trades': 0, 'expectancy': 0.0, 'hit_rate': 0.0
                }
                continue
            
            regime_preds = preds[regime_mask]
            regime_returns = returns[regime_mask]
            
            # Filter for directional predictions
            long_mask = regime_preds == 2
            short_mask = regime_preds == 0
            trade_mask = long_mask | short_mask
            
            # Compute PnL for each trade
            trade_pnl = np.zeros(len(regime_returns))
            trade_pnl[long_mask] = regime_returns[long_mask] - cost
            trade_pnl[short_mask] = -regime_returns[short_mask] - cost
            
            trade_returns = trade_pnl[trade_mask]
            num_trades = len(trade_returns)
            
            if num_trades == 0:
                regime_metrics[regime_name] = {
                    'samples': int(regime_count),
                    'trades': 0,
                    'expectancy': 0.0,
                    'hit_rate': 0.0
                }
                continue
            
            expectancy = float(np.mean(trade_returns))
            wins = (trade_returns > 0).sum()
            hit_rate = float(wins / num_trades) if num_trades > 0 else 0.0
            
            regime_metrics[regime_name] = {
                'samples': int(regime_count),
                'trades': int(num_trades),
                'expectancy': expectancy,
                'hit_rate': hit_rate
            }
        
        # Log per-regime performance
        logger.info("Per-Regime Validation Metrics:")
        for regime_name, metrics in regime_metrics.items():
            if metrics['trades'] > 0:
                logger.info(f"  {regime_name}: {metrics['samples']} samples, {metrics['trades']} trades, "
                           f"Exp={metrics['expectancy']:.4f}, HitRate={metrics['hit_rate']:.2%}")
            else:
                logger.info(f"  {regime_name}: {metrics['samples']} samples, 0 trades")
        
        return regime_metrics
    
    def train(
        self,
        num_epochs: Optional[int] = None,
        early_stopping_patience: int = 30,
        min_epochs: int = 40,
        save_best: bool = True,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Full training loop with min_epochs protection.
        
        Args:
            num_epochs: Total epochs to train
            early_stopping_patience: Epochs without improvement before stopping (default 30)
            min_epochs: Minimum epochs before early stopping can trigger (default 40)
            save_best: Whether to save best checkpoint
            checkpoint_path: Path to save checkpoint
        
        Returns history of metrics per epoch.
        
        IMPORTANT: Early stopping uses val_loss ONLY (not monitoring sweep expectancy).
        PolicySelector handles policy selection post-training.
        """
        epochs = num_epochs or self.config.training.epochs
        
        logger.info(f"[TRAINING CONFIG] epochs={epochs}, min_epochs={min_epochs}, patience={early_stopping_patience}")
        logger.info(f"[TRAINING CONFIG] Early stopping uses val_loss only - PolicySelector handles policy post-training")
        
        # === DIAGNOSTIC: Log training label distribution at start ===
        # PHASE 2: Also compute focal alpha and set prior biases
        try:
            all_labels = []
            for batch in self.train_loader:
                labels = batch[1]  # labels are second element
                all_labels.extend(labels.cpu().numpy())
            all_labels = np.array(all_labels)
            n_total = len(all_labels)
            n_short = (all_labels == 0).sum()
            n_hold = (all_labels == 1).sum()
            n_long = (all_labels == 2).sum()
            logger.info("=" * 70)
            logger.info("TRAINING LABEL DISTRIBUTION:")
            logger.info("  SHORT (0): %5d / %d (%.1f%%)", n_short, n_total, 100*n_short/n_total if n_total > 0 else 0)
            logger.info("  HOLD  (1): %5d / %d (%.1f%%)", n_hold, n_total, 100*n_hold/n_total if n_total > 0 else 0)
            logger.info("  LONG  (2): %5d / %d (%.1f%%)", n_long, n_total, 100*n_long/n_total if n_total > 0 else 0)
            if n_hold / n_total > 0.90:
                logger.warning(">>> TRAINING DATA IS %.1f%% HOLD - MODEL WILL LEARN TO PREDICT HOLD <<<", 
                              100*n_hold/n_total)
                logger.warning(">>> CONSIDER: Use pure_directional or regime label mode to balance labels <<<")
            logger.info("=" * 70)
            
            # === PHASE 2: Compute class priors and focal alpha ===
            if n_total > 0:
                class_priors = torch.tensor([
                    n_short / n_total,
                    n_hold / n_total, 
                    n_long / n_total
                ], dtype=torch.float32)
                
                # Focal Loss alpha = inverse frequency (higher weight for rare classes)
                # Normalize so they sum to num_classes (3.0)
                inv_freq = 1.0 / (class_priors + 1e-6)
                focal_alpha = inv_freq / inv_freq.sum() * 3.0
                focal_alpha = torch.clamp(focal_alpha, max=10.0)  # Cap to prevent instability
                
                logger.info("PHASE 2 - FOCAL LOSS CONFIGURATION:")
                logger.info("  Class priors: SHORT=%.3f, HOLD=%.3f, LONG=%.3f", 
                           class_priors[0], class_priors[1], class_priors[2])
                logger.info("  Focal alpha:  SHORT=%.3f, HOLD=%.3f, LONG=%.3f",
                           focal_alpha[0], focal_alpha[1], focal_alpha[2])
                
                # Update the FocalLoss with computed alpha if using focal loss
                # Use buffer-safe set_alpha method to avoid device/state issues
                if hasattr(self.criterion, 'class_loss') and hasattr(self.criterion.class_loss, 'set_alpha'):
                    self.criterion.class_loss.set_alpha(focal_alpha.to(self.device))
                    logger.info("  -> Updated FocalLoss alpha weights (buffer-safe)")
                
                # Set prior biases in classification head if model supports it
                if hasattr(self.model, 'class_head') and hasattr(self.model.class_head, 'set_class_priors'):
                    self.model.class_head.set_class_priors(class_priors.to(self.device))
                    logger.info("  -> Initialized classification head with prior biases")
                
                logger.info("=" * 70)
                
        except Exception as e:
            logger.warning(f"Could not compute label distribution: {e}")
        
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'class_loss': [], 'mu_loss': [],
            'quantile_loss': []
        }
        
        # Track best trading metrics for model_weights.json save
        best_trading_metrics = None
        best_trading_score = float('-inf')
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch=epoch)
            
            # Log metrics
            history['train_loss'].append(train_metrics['total'])
            history['val_loss'].append(val_metrics['total'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['class_loss'].append(val_metrics['class'])
            history['mu_loss'].append(val_metrics['mu'])
            history['quantile_loss'].append(val_metrics['quantile'])
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_metrics['total'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['total'], epoch)
            self.writer.add_scalar('Loss/class', val_metrics['class'], epoch)
            self.writer.add_scalar('Loss/mu', val_metrics['mu'], epoch)
            self.writer.add_scalar('Loss/quantile', val_metrics['quantile'], epoch)
            self.writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            
            # Quantile calibration
            for q in ['q10', 'q25', 'q50', 'q75', 'q90']:
                if f'{q}_cal' in val_metrics:
                    self.writer.add_scalar(f'Calibration/{q}', val_metrics[f'{q}_cal'], epoch)
            
            # Trading metrics
            if 'expectancy' in val_metrics:
                self.writer.add_scalar('Trading/expectancy', val_metrics['expectancy'], epoch)
                self.writer.add_scalar('Trading/hit_rate', val_metrics['hit_rate'], epoch)
                self.writer.add_scalar('Trading/sharpe', val_metrics['sharpe'], epoch)
                self.writer.add_scalar('Trading/profit_factor', val_metrics['profit_factor'], epoch)
                self.writer.add_scalar('Trading/max_drawdown', val_metrics['max_drawdown'], epoch)
                self.writer.add_scalar('Trading/num_trades', val_metrics['num_trades'], epoch)
                
                # Track best trading metrics for model_weights.json
                # Always track, but prefer runs with more trades
                current_score = val_metrics.get('risk_adjusted_score', val_metrics['expectancy'])
                num_trades = val_metrics.get('num_trades', 0)
                
                # Update best if: (a) more trades OR (b) same/more trades with better score
                should_update = False
                if best_trading_metrics is None:
                    should_update = True  # First observation
                elif num_trades > best_trading_metrics.get('num_trades', 0):
                    should_update = True  # More trades = better sample
                elif num_trades == best_trading_metrics.get('num_trades', 0) and current_score > best_trading_score:
                    should_update = True  # Same trades, better score
                
                if should_update:
                    best_trading_score = current_score
                    best_trading_metrics = val_metrics.copy()
                    best_trading_metrics['best_epoch'] = epoch + 1
                    logger.info(f"[BEST TRADING] Updated at epoch {epoch+1}: score={current_score:.4f}, trades={num_trades}")
            
            # Progress callback
            if self.epoch_callback:
                self.epoch_callback(epoch, epochs, train_metrics, val_metrics)
            
            # Logging
            if not self.gui_mode:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train: {train_metrics['total']:.4f} (acc: {train_metrics['accuracy']:.3f}) - "
                    f"Val: {val_metrics['total']:.4f} (acc: {val_metrics['accuracy']:.3f})"
                )
            
            # Early stopping check - uses val_loss ONLY (not monitoring sweep expectancy)
            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self.patience_counter = 0
                
                if save_best and checkpoint_path:
                    self._save_checkpoint(checkpoint_path, val_metrics)
                    logger.info(f"[CHECKPOINT] Saved best model at epoch {epoch+1} with val_loss={val_metrics['total']:.4f}")
            else:
                self.patience_counter += 1
            
            # CRITICAL: Early stopping ONLY after min_epochs reached
            # This ensures multihead quantiles/vol_state/accel have enough epochs to converge
            if epoch + 1 >= min_epochs and self.patience_counter >= early_stopping_patience:
                logger.info(f"[EARLY STOPPING] Triggered at epoch {epoch+1} (min_epochs={min_epochs} reached, no improvement for {early_stopping_patience} epochs)")
                break
            elif epoch + 1 < min_epochs and self.patience_counter >= early_stopping_patience:
                # Log but DO NOT break - keep training until min_epochs
                logger.info(f"[MIN_EPOCHS PROTECTION] Epoch {epoch+1}/{epochs} - patience exhausted but min_epochs={min_epochs} not reached, continuing...")
        
        self.writer.close()
        
        # === SAVE WALK-FORWARD WEIGHTS TO model_weights.json ===
        if best_trading_metrics is not None:
            try:
                total_trades = best_trading_metrics.get('num_trades', 0)
                
                # Convert to format expected by save_walk_forward_weights
                wf_summary = {
                    'total_trades': total_trades,
                    'overall_win_rate': best_trading_metrics.get('hit_rate', 0.5),
                    'overall_expectancy': best_trading_metrics.get('expectancy', 0.0),
                    'overall_profit_factor': best_trading_metrics.get('profit_factor', 1.0),
                    'overall_sharpe': best_trading_metrics.get('sharpe', 0.0),
                    'worst_drawdown': best_trading_metrics.get('max_drawdown', 0.0),
                    'n_folds': 1,  # Single validation split (not true walk-forward)
                    'avg_trades_per_fold': total_trades,
                    'best_epoch': best_trading_metrics.get('best_epoch', epochs),
                    'source': 'validation_sweep'  # Mark as val-derived, not full walk-forward
                }
                
                # Determine model name from model attribute or default
                model_name = getattr(self.model, 'name', 'unknown').lower()
                
                # Save to checkpoints directory
                weights_dir = str(Path(__file__).parent.parent / "checkpoints")
                
                logger.info("=" * 70)
                logger.info("SAVING WALK-FORWARD WEIGHTS TO model_weights.json")
                logger.info(f"  Model: {model_name}")
                logger.info(f"  Trades: {total_trades}")
                logger.info(f"  Expectancy: {wf_summary['overall_expectancy']:.4f}")
                logger.info(f"  Win Rate: {wf_summary['overall_win_rate']:.2%}")
                logger.info(f"  Sharpe: {wf_summary['overall_sharpe']:.2f}")
                
                if total_trades < 30:
                    logger.warning(f"  ⚠️ LOW TRADE COUNT ({total_trades} < 30) - metrics may be unreliable")
                    
                logger.info("=" * 70)
                
                save_walk_forward_weights(model_name, wf_summary, weights_dir, force_save=True)
                logger.info(f"[SUCCESS] Saved weights to {weights_dir}/model_weights.json")
                
            except Exception as e:
                logger.error(f"[ERROR] Failed to save walk-forward weights: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.warning("=" * 70)
            logger.warning("NO WALK-FORWARD WEIGHTS SAVED - No eligible trading metrics found")
            logger.warning("  Requires: num_trades >= 30 from monitoring sweeps")
            logger.warning("  Check: MIN_MOVE_FACTOR, confidence thresholds, spread filter")
            logger.warning("=" * 70)
        
        return history
    
    def _save_checkpoint(self, path: str, metrics: Dict[str, float]):
        """Save model checkpoint with metrics, scaler, and feature config."""
        # Get FeatureEngineer version for tracking
        try:
            from data.pipeline import FeatureEngineer
            fe_version = FeatureEngineer.VERSION
        except:
            fe_version = "unknown"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model.name,
            'input_dim': self.model.input_dim,
            'model_type': 'multihead',
            'feature_engineer_version': fe_version,
            'training_mode': self.training_mode,  # Use actual training mode
            'horizon_periods': self.horizon_periods,  # Use actual horizon
        }
        
        # Include sklearn feature scaler if available (not AMP GradScaler)
        if self.feature_scaler is not None:
            try:
                # Save sklearn StandardScaler parameters
                checkpoint['scaler_mean'] = self.feature_scaler.mean_.tolist()
                checkpoint['scaler_scale'] = self.feature_scaler.scale_.tolist()
                checkpoint['scaler_var'] = self.feature_scaler.var_.tolist() if hasattr(self.feature_scaler, 'var_') else None
                checkpoint['scaler_n_features'] = self.feature_scaler.n_features_in_ if hasattr(self.feature_scaler, 'n_features_in_') else None
            except Exception as e:
                logger.warning(f"Could not save scaler state: {e}")
        
        # Include feature columns for validation at inference
        if self.feature_columns is not None:
            checkpoint['feature_columns'] = self.feature_columns
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path} (FE: {fe_version}, mode: {self.training_mode}, horizon: {self.horizon_periods})")


def create_multihead_dataloaders(
    features: np.ndarray,
    class_labels: np.ndarray,
    forward_returns: np.ndarray,
    sequence_length: int = 100,
    batch_size: int = 64,
    val_split: float = 0.2,
    purge_gap: int = 50
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val dataloaders with proper time-series split.
    
    Uses chronological split with purge gap to prevent lookahead.
    
    Args:
        features: [n_samples, input_dim]
        class_labels: [n_samples]
        forward_returns: [n_samples]
        sequence_length: Sequence length for model
        batch_size: Batch size
        val_split: Fraction for validation
        purge_gap: Gap between train and val to prevent leakage
    """
    n_samples = len(features)
    
    # Chronological split
    split_idx = int(n_samples * (1 - val_split)) - purge_gap
    
    # Train: [0, split_idx)
    train_features = features[:split_idx]
    train_labels = class_labels[:split_idx]
    train_returns = forward_returns[:split_idx]
    
    # Val: [split_idx + purge_gap, end)
    val_start = split_idx + purge_gap
    val_features = features[val_start:]
    val_labels = class_labels[val_start:]
    val_returns = forward_returns[val_start:]
    
    # Create datasets
    train_dataset = MultiHeadDataset(
        train_features, train_labels, train_returns, sequence_length
    )
    val_dataset = MultiHeadDataset(
        val_features, val_labels, val_returns, sequence_length
    )
    
    # Create dataloaders (no shuffle for train to preserve temporal order)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    logger.info(f"Created dataloaders: train={len(train_dataset)}, val={len(val_dataset)}")
    
    return train_loader, val_loader
