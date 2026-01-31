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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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
except ImportError:
    # Fallback for direct script execution
    from training.multihead_loss import MultiHeadLoss, MultiHeadLossConfig
    from models.multihead import MultiHeadOutput

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
        
        return (
            torch.from_numpy(seq),
            torch.tensor(self.class_labels[actual_idx]),
            torch.tensor(self.forward_returns[actual_idx]),
            torch.from_numpy(trading),
            torch.from_numpy(c)
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


class MultiHeadTrainer:
    """
    Trainer for multi-head models.
    
    Handles:
    - Combined loss optimization
    - Class weighting for imbalanced data
    - Per-head metric tracking
    - Walk-forward validation
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
        gui_mode: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.gui_mode = gui_mode
        
        # Setup loss
        if loss_config is None:
            loss_config = MultiHeadLossConfig(class_weights=class_weights)
        self.criterion = MultiHeadLoss(loss_config).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.training.learning_rate * 10,
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
        """Train one epoch with multi-head outputs (all 6 heads)."""
        self.model.train()
        
        total_losses = {
            'total': 0.0, 'class': 0.0, 'mu': 0.0, 
            'sigma': 0.0, 'quantile': 0.0, 'trading': 0.0, 'candle': 0.0
        }
        correct = 0
        total = 0
        
        for batch_idx, (features, class_labels, returns, trading, candle_tgt) in enumerate(self.train_loader):
            features = features.to(self.device)
            class_labels = class_labels.to(self.device)
            returns = returns.to(self.device)
            trading = trading.to(self.device)  # [batch, 3]
            candle_tgt = candle_tgt.to(self.device)  # [batch, n_future, 3]
            
            self.optimizer.zero_grad()
            
            # Multi-head forward pass
            output = self.model.forward_multihead(features)
            
            # Build trading_targets dict
            trading_targets = {
                "entry_offset": trading[:, 0:1],
                "sl_distance": trading[:, 1:2],
                "tp_distance": trading[:, 2:3],
            }
            
            # Compute combined loss (all 6 heads)
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
                candle_targets=candle_tgt
            )
            
            loss = losses['total']
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Track losses
            for key in total_losses:
                if key in losses:
                    total_losses[key] += losses[key].item()
            
            # Track accuracy
            preds = output.class_logits.argmax(dim=-1)
            correct += (preds == class_labels).sum().item()
            total += len(class_labels)
            
            self.global_step += 1
            
        # Average losses
        n_batches = len(self.train_loader)
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        avg_losses['accuracy'] = correct / total
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate with all heads (6 heads)."""
        self.model.eval()
        
        total_losses = {
            'total': 0.0, 'class': 0.0, 'mu': 0.0,
            'sigma': 0.0, 'quantile': 0.0, 'trading': 0.0, 'candle': 0.0
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
            for features, class_labels, returns, trading, candle_tgt in self.val_loader:
                features = features.to(self.device)
                class_labels = class_labels.to(self.device)
                returns = returns.to(self.device)
                trading = trading.to(self.device)
                candle_tgt = candle_tgt.to(self.device)
                
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
                    candle_targets=candle_tgt
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
        
        # Compute trading-aware metrics
        trading_metrics = self._compute_trading_metrics()
        avg_losses.update(trading_metrics)
        
        return avg_losses
    
    def _compute_trading_metrics(self) -> Dict[str, float]:
        """
        Compute trading-aware evaluation metrics.
        
        These metrics are what actually matter for trading performance:
        - Expectancy: Average profit per trade (R-multiple)
        - Hit rate: Percentage of winning trades
        - Cost-adjusted Sharpe: Risk-adjusted returns after costs
        - Max drawdown: Largest peak-to-trough decline
        - Profit factor: Gross profits / gross losses
        
        Returns:
            Dictionary of trading metrics
        """
        all_predictions = []
        all_returns = []
        all_mus = []
        
        cost = 0.001  # 0.1% round-trip cost
        
        with torch.no_grad():
            for features, class_labels, returns, trading, candle_tgt in self.val_loader:
                features = features.to(self.device)
                output = self.model.forward_multihead(features)
                
                probs = torch.softmax(output.class_logits, dim=-1)
                preds = probs.argmax(dim=-1)
                confidence = probs.max(dim=-1).values
                
                all_predictions.extend(preds.cpu().numpy())
                all_returns.extend(returns.cpu().numpy())
                all_mus.extend(output.mu.squeeze().cpu().numpy())
        
        preds = np.array(all_predictions)
        returns = np.array(all_returns)
        mus = np.array(all_mus)
        
        metrics = {}
        
        # Filter for directional predictions (LONG=2, SHORT=0)
        long_mask = preds == 2
        short_mask = preds == 0
        trade_mask = long_mask | short_mask
        
        if trade_mask.sum() == 0:
            logger.warning("No directional trades in validation set")
            return {
                'expectancy': 0.0, 'hit_rate': 0.0, 'profit_factor': 0.0,
                'sharpe': 0.0, 'max_drawdown': 0.0, 'num_trades': 0
            }
        
        # Compute PnL for each trade
        trade_pnl = np.zeros(len(returns))
        trade_pnl[long_mask] = returns[long_mask] - cost  # LONG: profit if price goes up
        trade_pnl[short_mask] = -returns[short_mask] - cost  # SHORT: profit if price goes down
        
        # Filter to only actual trades
        trade_returns = trade_pnl[trade_mask]
        num_trades = len(trade_returns)
        
        # Expectancy (average R per trade)
        metrics['expectancy'] = float(np.mean(trade_returns)) if num_trades > 0 else 0.0
        
        # Hit rate (percentage of winning trades)
        wins = (trade_returns > 0).sum()
        metrics['hit_rate'] = float(wins / num_trades) if num_trades > 0 else 0.0
        
        # Profit factor (gross profits / gross losses)
        gross_profits = trade_returns[trade_returns > 0].sum()
        gross_losses = abs(trade_returns[trade_returns < 0].sum())
        metrics['profit_factor'] = float(gross_profits / gross_losses) if gross_losses > 0 else 0.0
        
        # Cost-adjusted Sharpe ratio (annualized)
        if num_trades > 1 and np.std(trade_returns) > 0:
            # Assuming each trade is ~4h, so ~6 trades/day = ~2190 trades/year
            annual_factor = np.sqrt(2190)
            sharpe = (np.mean(trade_returns) / np.std(trade_returns)) * annual_factor
            metrics['sharpe'] = float(sharpe)
        else:
            metrics['sharpe'] = 0.0
        
        # Max drawdown (simple cumulative PnL version)
        cumulative = np.cumsum(trade_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        metrics['max_drawdown'] = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
        
        # Number of trades
        metrics['num_trades'] = int(num_trades)
        
        # Average win / average loss
        if wins > 0:
            avg_win = np.mean(trade_returns[trade_returns > 0])
            metrics['avg_win'] = float(avg_win)
        else:
            metrics['avg_win'] = 0.0
        
        losses_count = (trade_returns < 0).sum()
        if losses_count > 0:
            avg_loss = np.mean(trade_returns[trade_returns < 0])
            metrics['avg_loss'] = float(avg_loss)
        else:
            metrics['avg_loss'] = 0.0
        
        # Win/loss ratio (R:R)
        if metrics['avg_loss'] != 0:
            metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss'])
        else:
            metrics['win_loss_ratio'] = 0.0
        
        logger.info(f"Trading Metrics - Expectancy: {metrics['expectancy']:.4f}, "
                   f"Hit Rate: {metrics['hit_rate']:.2%}, "
                   f"Sharpe: {metrics['sharpe']:.2f}, "
                   f"Trades: {metrics['num_trades']}")
        
        return metrics
    
    def train(
        self,
        num_epochs: Optional[int] = None,
        early_stopping_patience: int = 10,
        save_best: bool = True,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Returns history of metrics per epoch.
        """
        epochs = num_epochs or self.config.training.epochs
        
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'class_loss': [], 'mu_loss': [],
            'quantile_loss': []
        }
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            
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
            
            # Early stopping check
            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self.patience_counter = 0
                
                if save_best and checkpoint_path:
                    self._save_checkpoint(checkpoint_path, val_metrics)
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        self.writer.close()
        return history
    
    def _save_checkpoint(self, path: str, metrics: Dict[str, float]):
        """Save model checkpoint with metrics."""
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
            'model_type': 'multihead'
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")


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
