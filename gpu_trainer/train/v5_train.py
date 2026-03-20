"""
V5.0.1 Training Pipeline: Forecaster with Quality Gating + TPD Controller

Changes from v5.0:
- All targets in R-units (ret_R, mfe_R, mae_R) -- no more log-return/R-unit mixing
- Adaptive deadzone targeting ~30% HOLD rate (--v5-hold-target)
- Class-balanced action CE loss (inverse frequency weighting)
- Score formula uses R-units consistently: edge = p_dir * (mu_R / (mae_R + eps))
- Candidate warmup: disable candidates for first N epochs
- (B) Uncertainty/Quality Gating: sigma, mae, mu_R, p_trade gates
- (B) Lightweight calibration: 10-bin ECE on p_trade vs observed win-rate
- (C) Trade Frequency Controller: adaptive threshold to hit target TPD
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from collections import defaultdict, Counter
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from dataclasses import dataclass, field

log = logging.getLogger("QuickStart")

V5_FEATURE_VERSION = "v5.0.1_forecaster"

_active_pusher = None
_active_fold_num = 0
_active_total_folds = 1


def _compute_ema(close_arr, period=200):
    """Compute EMA using only past data (no leakage). Returns array same length as input."""
    alpha = 2.0 / (period + 1)
    ema = np.empty_like(close_arr, dtype=np.float64)
    ema[0] = close_arr[0]
    for i in range(1, len(close_arr)):
        ema[i] = alpha * close_arr[i] + (1 - alpha) * ema[i - 1]
    return ema


def _compute_adx(high, low, close, period=14):
    """Compute ADX indicator. Returns array same length as input (NaN-filled for warmup).

    Uses Wilder's smoothing method (EMA with alpha=1/period).
    Epsilon guards prevent divide-by-zero warnings in low-volatility bars.
    """
    _EPS = 1e-10
    n = len(close)
    adx = np.full(n, np.nan)
    if n < period * 3:
        return adx

    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        hl = high[i] - low[i]
        hpc = abs(high[i] - close[i - 1])
        lpc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hpc, lpc)

        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    alpha = 1.0 / period
    atr = np.zeros(n)
    plus_di_smooth = np.zeros(n)
    minus_di_smooth = np.zeros(n)

    atr[period] = np.mean(tr[1:period + 1])
    plus_di_smooth[period] = np.mean(plus_dm[1:period + 1])
    minus_di_smooth[period] = np.mean(minus_dm[1:period + 1])

    for i in range(period + 1, n):
        atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
        plus_di_smooth[i] = plus_di_smooth[i - 1] * (1 - alpha) + plus_dm[i] * alpha
        minus_di_smooth[i] = minus_di_smooth[i - 1] * (1 - alpha) + minus_dm[i] * alpha

    plus_di = np.where(atr > _EPS, 100 * plus_di_smooth / atr, 0)
    minus_di = np.where(atr > _EPS, 100 * minus_di_smooth / atr, 0)
    di_sum = plus_di + minus_di
    dx = np.where(di_sum > _EPS, 100 * np.abs(plus_di - minus_di) / di_sum, 0)

    adx_start = period * 2
    if adx_start < n:
        adx[adx_start] = np.mean(dx[period:adx_start + 1])
        for i in range(adx_start + 1, n):
            adx[i] = adx[i - 1] * (1 - alpha) + dx[i] * alpha

    return adx


@dataclass
class V5ForwardTestConfig:
    """Config for frozen decision layer in forward test."""
    score_threshold: float = 0.0
    score_lambda: float = 0.5
    mae_cap: float = 2.0
    risk_proxy: str = 'mae'
    tp_mult: float = 2.0
    sl_mult: float = 1.5
    horizon: int = 16
    cooldown: int = 4
    quality_gate_cfg: Optional['V5QualityGateConfig'] = None
    side_mode: str = 'action_head'
    rr_weight: float = 0.0
    weekly_loss_cap: Optional[float] = None
    warmup_skip_bars: int = 0
    corr_block: bool = False
    corr_window_days: int = 30
    corr_thresh: float = 0.90
    corr_same_side_only: bool = True
    corr_log_matrix: bool = True
    corr_max_block: int = 5
    symbols_list: Optional[list] = None
    adaptive_sizing: bool = False
    kelly_fraction: float = 0.25
    max_size_mult: float = 2.5
    min_size_mult: float = 0.25
    regime_scaling: bool = False
    regime_bull_mult: float = 1.5
    regime_bear_mult: float = 0.5
    regime_lookback: int = 20
    daily_loss_cap: Optional[float] = None
    trailing_equity_stop: Optional[float] = None
    per_symbol_daily_r_budget: Optional[float] = None
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None
    min_threshold_pct: Optional[float] = None
    max_trades_per_day: Optional[int] = None
    trailing_sl: bool = False
    trail_activation: float = 1.5
    trail_distance: float = 1.0
    allow_runner: bool = False
    conviction_sizing: bool = False
    conviction_tier_top_pct: float = 5.0
    conviction_tier_top_mult: float = 2.5
    conviction_tier_high_pct: float = 20.0
    conviction_tier_high_mult: float = 1.5
    conviction_confidence_threshold: float = 0.65
    conviction_confidence_boost: float = 1.3
    temperature: float = 1.0
    adx_gate: bool = False
    adx_period: int = 14
    adx_min: float = 18.0
    adx_exception_top_pct: float = 10.0
    ultra_conviction: bool = False
    ultra_risk_cap: float = 0.05
    ultra_score_pct: float = 0.95
    ultra_adx_min: float = 25.0
    ultra_edge_min: float = 0.03
    ultra_dd_max: float = 0.10
    ultra_max_per_day: int = 1
    ultra_mult: float = 3.0
    ddt_enable: bool = False
    ddt_lookback_trades: int = 60
    ddt_bad_rollr: float = 6.0
    ddt_thr_k: float = 0.60
    ddt_thr_min: float = 0.08
    ddt_thr_max: float = 0.25
    ddt_size_k: float = 0.70
    ddt_min_size_mult: float = 0.25
    ddt_alpha_down: float = 0.30
    ddt_alpha_up: float = 0.05
    ddt_warmup_trades: int = 20
    multi_regime: bool = False
    regime_adx_trending: float = 25.0
    regime_adx_choppy: float = 20.0
    regime_atr_high_vol: float = 1.3
    regime_atr_low_vol: float = 0.7
    regime_atr_window: int = 96
    regime_ema_slope_window: int = 10
    regime_ema_buffer: float = 0.005
    edge_first: bool = False
    edge_min: float = 0.03
    edge_pct_floor: int = 70
    edge_topn_per_day: int = 4
    regime_side_map: Optional[dict] = None
    regime_soft: bool = True
    regime_disagree_mult: float = 0.3
    regime_none_mult: float = 0.2
    per_symbol_soft_kill: bool = True
    edge_topn_soft: bool = True
    edge_topn_decay: float = 0.7
    size_floor: float = 0.0
    calibration_monitor: bool = False
    calibration_warn_ece: float = 0.10
    calibration_block_ece: float = 0.15
    feature_psi: bool = False
    head_disagreement_gate: bool = False
    sigma_discount: bool = False
    min_p_side: float = 0.0
    min_p_short: float = 0.0
    side_aware_scoring: bool = False
    per_symbol_cooldown: bool = True
    slippage_base_bps: float = 0.0
    slippage_impact_mult: float = 0.0
    ood_gate: bool = False
    ood_sigma_mult: float = 1.5
    ood_size_reduction: float = 0.5
    mu_debias: bool = True
    mu_debias_alpha: float = 0.01
    min_trades: int = 20
    per_symbol_r_kill: Optional[float] = None
    per_symbol_thresholds: Optional[dict] = None
    soft_gate_floor: bool = True
    weekly_cap_dynamic: bool = False
    weekly_cap_scale: float = 2.0
    quality_gate_enabled: bool = False
    quality_gate_window: int = 50
    quality_gate_min_accuracy: float = 0.30
    quality_gate_min_wr: float = 0.35
    quality_gate_severe_accuracy: float = 0.20
    direction_balance_cap: bool = False
    direction_balance_threshold: float = 0.75
    direction_balance_severe: float = 0.85
    ema200_soft_mult: Optional[float] = None
    per_side_threshold: bool = False


def compute_feature_importance_report(
    model, device, val_feat, val_ret_R, val_valid, val_sym_ids,
    feature_names, checkpoint_dir, n_repeats=5,
    corr_threshold=0.85,
):
    """Compute permutation importance and pairwise Spearman correlation for all features.

    Permutation importance: for each feature, shuffle it n_repeats times,
    measure the drop in prediction quality (mean |mu_R| for valid bars).
    Higher importance = larger drop when shuffled.

    Correlation: compute pairwise Spearman rank correlation on training features,
    flag pairs with |correlation| > corr_threshold.

    Returns dict with importance ranking and correlation flags.
    Saves report to checkpoint_dir/v5_feature_report.json.
    """
    from scipy.stats import spearmanr

    model.eval()
    n_features = val_feat.shape[1]
    n_samples = len(val_feat)

    valid_mask = val_valid.astype(bool)
    n_valid = int(np.sum(valid_mask))

    if n_valid < 100:
        log.warning("[V5_FEAT_REPORT] Only %d valid bars, skipping feature report", n_valid)
        return {}

    feat_tensor = torch.tensor(val_feat, dtype=torch.float32)
    sym_tensor = torch.tensor(val_sym_ids, dtype=torch.long) if val_sym_ids is not None else None

    with torch.no_grad():
        batch_size = 2048
        all_mu = []
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            feat_batch = feat_tensor[start:end].to(device)
            sym_batch = sym_tensor[start:end].to(device) if sym_tensor is not None else None
            outputs = model(feat_batch, symbol_ids=sym_batch)
            all_mu.append(outputs['ret_mu'].cpu().numpy().squeeze(-1))
        baseline_mu = np.concatenate(all_mu, axis=0)

    baseline_valid_mu = baseline_mu[valid_mask]
    baseline_ret_valid = val_ret_R[valid_mask]
    baseline_mse = float(np.mean((baseline_valid_mu - baseline_ret_valid) ** 2))

    log.info("[V5_FEAT_REPORT] Computing permutation importance for %d features "
             "(%d valid bars, %d repeats)...", n_features, n_valid, n_repeats)

    importance_scores = np.zeros(n_features)

    for fi in range(n_features):
        drop_sum = 0.0
        for rep in range(n_repeats):
            shuffled_feat = val_feat.copy()
            rng = np.random.RandomState(42 + fi * n_repeats + rep)
            shuffled_feat[:, fi] = rng.permutation(shuffled_feat[:, fi])

            shuffled_tensor = torch.tensor(shuffled_feat, dtype=torch.float32)
            all_mu_shuf = []
            with torch.no_grad():
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    feat_batch = shuffled_tensor[start:end].to(device)
                    sym_batch = sym_tensor[start:end].to(device) if sym_tensor is not None else None
                    outputs = model(feat_batch, symbol_ids=sym_batch)
                    all_mu_shuf.append(outputs['ret_mu'].cpu().numpy().squeeze(-1))
            shuf_mu = np.concatenate(all_mu_shuf, axis=0)

            shuf_mse = float(np.mean((shuf_mu[valid_mask] - baseline_ret_valid) ** 2))
            drop_sum += (shuf_mse - baseline_mse)

        importance_scores[fi] = drop_sum / n_repeats

        if (fi + 1) % 10 == 0 or fi == n_features - 1:
            log.info("[V5_FEAT_REPORT] Permutation importance: %d/%d features done",
                     fi + 1, n_features)

    sorted_indices = np.argsort(-importance_scores)
    importance_ranking = []
    for rank, idx in enumerate(sorted_indices):
        name = feature_names[idx] if feature_names and idx < len(feature_names) else f"feature_{idx}"
        importance_ranking.append({
            'rank': rank + 1,
            'feature': name,
            'importance': float(importance_scores[idx]),
            'feature_index': int(idx),
        })

    log.info("[V5_FEAT_REPORT] Top 10 features by permutation importance:")
    for entry in importance_ranking[:10]:
        log.info("  #%d  %s  importance=%.6f", entry['rank'], entry['feature'], entry['importance'])

    log.info("[V5_FEAT_REPORT] Bottom 10 features (least important):")
    for entry in importance_ranking[-10:]:
        log.info("  #%d  %s  importance=%.6f", entry['rank'], entry['feature'], entry['importance'])

    log.info("[V5_FEAT_REPORT] Computing pairwise Spearman correlation on %d features "
             "(%d samples)...", n_features, n_samples)

    subsample_n = min(n_samples, 50000)
    if subsample_n < n_samples:
        rng_sub = np.random.RandomState(123)
        sub_idx = rng_sub.choice(n_samples, subsample_n, replace=False)
        feat_sub = val_feat[sub_idx]
    else:
        feat_sub = val_feat

    corr_matrix, _ = spearmanr(feat_sub, axis=0)
    if corr_matrix.ndim == 0:
        corr_matrix = np.array([[corr_matrix]])

    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    high_corr_pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            c = float(corr_matrix[i, j])
            if abs(c) > corr_threshold:
                name_i = feature_names[i] if feature_names and i < len(feature_names) else f"feature_{i}"
                name_j = feature_names[j] if feature_names and j < len(feature_names) else f"feature_{j}"
                high_corr_pairs.append({
                    'feature_a': name_i,
                    'feature_b': name_j,
                    'correlation': c,
                    'abs_correlation': abs(c),
                    'index_a': i,
                    'index_b': j,
                })

    high_corr_pairs.sort(key=lambda x: -x['abs_correlation'])

    log.info("[V5_FEAT_REPORT] Found %d feature pairs with |correlation| > %.2f:",
             len(high_corr_pairs), corr_threshold)
    for pair in high_corr_pairs[:20]:
        log.info("  %s <-> %s  corr=%.4f",
                 pair['feature_a'], pair['feature_b'], pair['correlation'])

    report = {
        'baseline_mse': baseline_mse,
        'n_features': n_features,
        'n_valid_bars': n_valid,
        'n_repeats': n_repeats,
        'corr_threshold': corr_threshold,
        'importance_ranking': importance_ranking,
        'high_correlation_pairs': high_corr_pairs,
        'n_high_corr_pairs': len(high_corr_pairs),
        'generated_at': datetime.now().isoformat(),
    }

    report_path = Path(checkpoint_dir) / "v5_feature_report.json"
    import json
    def _serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return str(obj)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=_serialize)

    log.info("[V5_FEAT_REPORT] Feature report saved to %s", report_path)
    log.info("[V5_FEAT_REPORT] Summary: %d features, %d high-correlation pairs, "
             "top feature=%s (importance=%.6f)",
             n_features, len(high_corr_pairs),
             importance_ranking[0]['feature'] if importance_ranking else "?",
             importance_ranking[0]['importance'] if importance_ranking else 0.0)

    return report


def _parse_date_to_ms(date_str: str) -> int:
    """Parse YYYY-MM-DD to millisecond timestamp."""
    from datetime import datetime as dt, timezone
    d = dt.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(d.timestamp() * 1000)


def _compute_time_split(sym_df, train_end_date=None, test_start_date=None, test_end_date=None,
                        purge_bars: int = 0):
    """Compute train/test indices for a single symbol based on timestamp dates.

    Returns (train_indices, test_indices) as numpy arrays.
    If no dates provided, falls back to 80/20 percentage split.

    purge_bars: number of bars to exclude between train end and test start
        to prevent label leakage from forward-looking targets. Should be
        set to the prediction horizon (e.g. 24 bars for 6h on 15m data).
    """
    timestamps = sym_df['timestamp'].values
    n = len(sym_df)

    if train_end_date is None and test_start_date is None:
        split_idx = int(n * 0.8)
        train_end_idx = max(0, split_idx - purge_bars)
        test_start_idx = split_idx
        if purge_bars > 0:
            log.info(f"[V5_PURGE] Percentage split: purge gap of {purge_bars} bars "
                     f"(train ends at {train_end_idx}, test starts at {test_start_idx})")
        return np.arange(train_end_idx), np.arange(test_start_idx, n)

    train_end_ms = _parse_date_to_ms(train_end_date) if train_end_date else None
    test_start_ms = _parse_date_to_ms(test_start_date) if test_start_date else train_end_ms
    test_end_ms = _parse_date_to_ms(test_end_date) if test_end_date else None

    if train_end_ms is not None:
        train_mask = timestamps < train_end_ms
    else:
        train_mask = np.ones(n, dtype=bool)

    if purge_bars > 0 and train_end_ms is not None:
        train_indices_raw = np.where(train_mask)[0]
        if len(train_indices_raw) > purge_bars:
            purge_start = len(train_indices_raw) - purge_bars
            train_mask[train_indices_raw[purge_start:]] = False
            log.info(f"[V5_PURGE] Removed last {purge_bars} bars from train set "
                     f"(label horizon overlap protection)")

    test_mask = np.ones(n, dtype=bool)
    if test_start_ms is not None:
        test_mask &= timestamps >= test_start_ms
    if test_end_ms is not None:
        test_mask &= timestamps <= test_end_ms

    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    if len(train_indices) == 0:
        log.warning(f"[V5] Time-based split: EMPTY train set! Check date range.")
    if len(test_indices) == 0:
        log.warning(f"[V5] Time-based split: EMPTY test set! Check date range.")

    return train_indices, test_indices


@dataclass
class V5QualityGateConfig:
    sigma_max: float = 1.0
    mae_max: float = 1.0
    mu_R_min: float = 0.05
    p_trade_min: float = 0.40
    enable_calib: bool = False


@dataclass
class V5TPDControllerConfig:
    target_tpd: float = 6.5
    tpd_tol: float = 1.5
    thr_warmup_epochs: int = 3
    thr_step_mult: float = 0.10
    score_threshold: Optional[float] = None
    score_lambda: float = 0.5
    mae_cap: float = 2.0
    side_mode: str = 'action_head'
    rr_weight: float = 0.0
    min_threshold_floor: float = 0.02


class V5Dataset(Dataset):
    def __init__(self, features, ret_R, mfe_R, mae_R, vol_h, action_labels,
                 valid_mask, symbol_ids=None, barrier_labels=None, barrier_soft=None,
                 sample_weights=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.ret_R = torch.tensor(ret_R, dtype=torch.float32)
        self.mfe_R = torch.tensor(mfe_R, dtype=torch.float32)
        self.mae_R = torch.tensor(mae_R, dtype=torch.float32)
        self.vol_h = torch.tensor(vol_h, dtype=torch.float32)
        self.action_labels = torch.tensor(action_labels, dtype=torch.long)
        self.valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
        self.symbol_ids = torch.tensor(symbol_ids, dtype=torch.long) if symbol_ids is not None else None
        self.barrier_labels = torch.tensor(barrier_labels, dtype=torch.long) if barrier_labels is not None else None
        self.barrier_soft = torch.tensor(barrier_soft, dtype=torch.float32) if barrier_soft is not None else None
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32) if sample_weights is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = {
            'features': self.features[idx],
            'ret_R': self.ret_R[idx],
            'mfe_R': self.mfe_R[idx],
            'mae_R': self.mae_R[idx],
            'vol_h': self.vol_h[idx],
            'action_label': self.action_labels[idx],
            'valid': self.valid_mask[idx],
        }
        if self.symbol_ids is not None:
            item['symbol_id'] = self.symbol_ids[idx]
        if self.barrier_labels is not None:
            item['barrier_label'] = self.barrier_labels[idx]
        if self.barrier_soft is not None:
            item['barrier_soft'] = self.barrier_soft[idx]
        if self.sample_weights is not None:
            item['sample_weight'] = self.sample_weights[idx]
        return item


class V6SequenceDataset(Dataset):
    """Sequence dataset for V6Forecaster — builds sliding windows per symbol.

    Each sample returns (seq_len, n_features) window plus targets for the last bar.
    Windows never cross symbol boundaries. Zero-padded at the start of each symbol.
    """

    def __init__(self, features_per_symbol, ret_R_per_symbol, mfe_R_per_symbol,
                 mae_R_per_symbol, vol_h_per_symbol, action_per_symbol,
                 valid_per_symbol, symbol_ids_per_symbol,
                 barrier_oracle_per_symbol=None, barrier_soft_per_symbol=None,
                 seq_len=16, sample_weights_per_symbol=None):
        self.seq_len = seq_len
        self.index_map = []
        self.features_list = []
        self.ret_R_list = []
        self.mfe_R_list = []
        self.mae_R_list = []
        self.vol_h_list = []
        self.action_list = []
        self.valid_list = []
        self.sym_id_list = []
        self.barrier_oracle_list = []
        self.barrier_soft_list = []
        self.sample_weights = []

        n_features = features_per_symbol[0].shape[1] if len(features_per_symbol) > 0 and len(features_per_symbol[0]) > 0 else 85

        for si in range(len(features_per_symbol)):
            n_bars = len(features_per_symbol[si])
            if n_bars == 0:
                continue
            self.features_list.append(torch.tensor(features_per_symbol[si], dtype=torch.float32))
            self.ret_R_list.append(torch.tensor(ret_R_per_symbol[si], dtype=torch.float32))
            self.mfe_R_list.append(torch.tensor(mfe_R_per_symbol[si], dtype=torch.float32))
            self.mae_R_list.append(torch.tensor(mae_R_per_symbol[si], dtype=torch.float32))
            self.vol_h_list.append(torch.tensor(vol_h_per_symbol[si], dtype=torch.float32))
            self.action_list.append(torch.tensor(action_per_symbol[si], dtype=torch.long))
            self.valid_list.append(torch.tensor(valid_per_symbol[si], dtype=torch.bool))
            self.sym_id_list.append(torch.tensor(symbol_ids_per_symbol[si], dtype=torch.long))

            if barrier_oracle_per_symbol is not None:
                self.barrier_oracle_list.append(torch.tensor(barrier_oracle_per_symbol[si], dtype=torch.long))
            if barrier_soft_per_symbol is not None:
                self.barrier_soft_list.append(torch.tensor(barrier_soft_per_symbol[si], dtype=torch.float32))

            sym_idx_in_list = len(self.features_list) - 1
            for bar_idx in range(n_bars):
                self.index_map.append((sym_idx_in_list, bar_idx))

            if sample_weights_per_symbol is not None and si < len(sample_weights_per_symbol):
                w = sample_weights_per_symbol[si]
                if hasattr(w, '__len__'):
                    self.sample_weights.extend(w)
                else:
                    self.sample_weights.extend([w] * n_bars)

        self.n_features = n_features
        self.has_barriers = len(self.barrier_oracle_list) > 0
        self.has_barrier_soft = len(self.barrier_soft_list) > 0
        self.has_weights = len(self.sample_weights) > 0
        if self.has_weights:
            self.sample_weights = torch.tensor(self.sample_weights, dtype=torch.float32)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        sym_idx, bar_idx = self.index_map[idx]
        features = self.features_list[sym_idx]
        n_bars = features.shape[0]

        start = max(0, bar_idx - self.seq_len + 1)
        end = bar_idx + 1
        window = features[start:end]

        if window.shape[0] < self.seq_len:
            pad_size = self.seq_len - window.shape[0]
            padding = torch.zeros(pad_size, self.n_features, dtype=torch.float32)
            window = torch.cat([padding, window], dim=0)

        next_bar_features = torch.zeros(self.n_features, dtype=torch.float32)
        if bar_idx + 1 < n_bars:
            next_bar_features = features[bar_idx + 1]

        item = {
            'features': window,
            'ret_R': self.ret_R_list[sym_idx][bar_idx],
            'mfe_R': self.mfe_R_list[sym_idx][bar_idx],
            'mae_R': self.mae_R_list[sym_idx][bar_idx],
            'vol_h': self.vol_h_list[sym_idx][bar_idx],
            'action_label': self.action_list[sym_idx][bar_idx],
            'valid': self.valid_list[sym_idx][bar_idx],
            'symbol_id': self.sym_id_list[sym_idx][bar_idx],
            'next_bar_features': next_bar_features,
            'has_next_bar': torch.tensor(bar_idx + 1 < n_bars, dtype=torch.bool),
        }

        if self.has_barriers:
            item['barrier_label'] = self.barrier_oracle_list[sym_idx][bar_idx]
        if self.has_barrier_soft:
            item['barrier_soft'] = self.barrier_soft_list[sym_idx][bar_idx]
        if self.has_weights:
            item['sample_weight'] = self.sample_weights[idx]

        return item


def compute_v5_loss(outputs, batch, w_ret=1.0, w_mfe=0.25, w_mae=0.25,
                    w_action=2.0, w_barrier=0.25, w_regime=0.1,
                    barrier_mode='fixed', action_weights=None, epoch=0,
                    sample_weights=None, mae_asym_weight=1.0):
    """Compute v5 composite loss with class-balanced action CE.

    Uses clamped Gaussian NLL to prevent log(sigma) term from dominating.
    Three-stage schedule: epochs 0-5 action-heavy, 6-15 balanced, 16+ full.

    If sample_weights is provided, computes per-sample losses and applies
    inverse-frequency weighting: loss = (per_sample_loss * weights).sum() / weights.sum()
    """
    valid = batch['valid']
    if valid.sum() == 0:
        return torch.tensor(0.0, device=outputs['ret_mu'].device, requires_grad=True), {}

    sw = None
    if sample_weights is not None:
        sw = sample_weights[valid]

    ret_mu = outputs['ret_mu'][valid].squeeze(-1)
    ret_log_sigma = outputs['ret_log_sigma'][valid].squeeze(-1)
    ret_true = batch['ret_R'][valid]

    sigma = torch.exp(ret_log_sigma).clamp(min=0.01, max=5.0)
    squared_error = ((ret_true - ret_mu) / (sigma + 1e-8)) ** 2
    log_term = torch.log(sigma + 1e-8)
    nll = log_term + 0.5 * squared_error
    if sw is not None:
        L_ret = (nll * sw).sum() / sw.sum()
    else:
        L_ret = nll.mean()

    mfe_pred = outputs['mfe'][valid].squeeze(-1)
    mfe_true = batch['mfe_R'][valid]
    if sw is not None:
        mfe_err = F.smooth_l1_loss(mfe_pred, mfe_true, reduction='none')
        L_mfe = (mfe_err * sw).sum() / sw.sum()
    else:
        L_mfe = F.smooth_l1_loss(mfe_pred, mfe_true)

    mae_pred = outputs['mae'][valid].squeeze(-1)
    mae_true = batch['mae_R'][valid]
    mae_err = F.smooth_l1_loss(mae_pred, mae_true, reduction='none')
    if mae_asym_weight > 1.0:
        underest_mask = (mae_pred < mae_true).float()
        asym_mult = 1.0 + underest_mask * (mae_asym_weight - 1.0)
        mae_err = mae_err * asym_mult
    if sw is not None:
        L_mae = (mae_err * sw).sum() / sw.sum()
    else:
        L_mae = mae_err.mean()

    action_logits = outputs['action_logits'][valid]
    action_true = batch['action_label'][valid]
    if sw is not None:
        action_per_sample = F.cross_entropy(action_logits, action_true,
                                            weight=action_weights, reduction='none')
        L_action = (action_per_sample * sw).sum() / sw.sum()
    elif action_weights is not None:
        L_action = F.cross_entropy(action_logits, action_true, weight=action_weights)
    else:
        L_action = F.cross_entropy(action_logits, action_true)

    LONG_IDX, SHORT_IDX = 1, 2
    SIDE_BAL_W = 0.30
    action_probs = F.softmax(action_logits, dim=-1)
    eps = 1e-8

    ret_true_valid = batch['ret_R'][valid]
    bull_mask = ret_true_valid > 0.20
    bear_mask = ret_true_valid < -0.20
    chop_mask = ~bull_mask & ~bear_mask

    n_bull = int(bull_mask.sum().item())
    n_bear = int(bear_mask.sum().item())
    n_chop = int(chop_mask.sum().item())

    group_kl_list = []
    for mask, p_long_target in [(bull_mask, 0.60), (bear_mask, 0.40), (chop_mask, 0.50)]:
        if mask.sum() == 0:
            continue
        p_long_g = action_probs[mask, LONG_IDX].mean()
        p_short_g = action_probs[mask, SHORT_IDX].mean()
        pred_g = torch.stack([p_long_g, p_short_g])
        pred_g = pred_g / (pred_g.sum() + eps)
        p_short_target = 1.0 - p_long_target
        target_g = torch.tensor([p_long_target, p_short_target],
                                 dtype=pred_g.dtype, device=pred_g.device)
        kl_g = F.kl_div((pred_g + eps).log(), target_g.detach(), reduction="batchmean")
        group_kl_list.append(kl_g)

    if group_kl_list:
        L_side_balance = torch.stack(group_kl_list).mean()
    else:
        L_side_balance = torch.tensor(0.0, device=action_logits.device)

    if epoch == 0 or epoch % 10 == 0:
        log.info(f"[V5_SIDE_BAL] regime_conditional=True SIDE_BAL_W={SIDE_BAL_W} "
                 f"bull={n_bull} bear={n_bear} chop={n_chop} L_side_bal={float(L_side_balance.item()):.4f}")

    L_action = L_action + SIDE_BAL_W * L_side_balance

    losses = {
        'L_ret': L_ret.item(),
        'L_mfe': L_mfe.item(),
        'L_mae': L_mae.item(),
        'L_action': L_action.item(),
        'L_side_balance': float(L_side_balance.item()) if torch.is_tensor(L_side_balance) else 0.0,
    }

    if epoch <= 5:
        eff_w_ret = w_ret * 0.3
        eff_w_mfe = w_mfe * 0.3
        eff_w_mae = w_mae * 0.3
        eff_w_action = w_action * 2.0
    elif epoch <= 15:
        eff_w_ret = w_ret * 0.7
        eff_w_mfe = w_mfe * 0.7
        eff_w_mae = w_mae * 0.7
        eff_w_action = w_action * 1.0
    else:
        eff_w_ret = w_ret
        eff_w_mfe = w_mfe
        eff_w_mae = w_mae
        eff_w_action = w_action

    total = eff_w_ret * L_ret + eff_w_mfe * L_mfe + eff_w_mae * L_mae + eff_w_action * L_action

    if 'barrier_logits' in outputs and barrier_mode != 'fixed':
        barrier_logits = outputs['barrier_logits'][valid]
        if barrier_mode == 'oracle' and 'barrier_label' in batch:
            barrier_true = batch['barrier_label'][valid]
            L_barrier = F.cross_entropy(barrier_logits, barrier_true)
        elif barrier_mode == 'learnable' and 'barrier_soft' in batch:
            log_probs = F.log_softmax(barrier_logits, dim=-1)
            soft = batch['barrier_soft'][valid]
            L_barrier = -(soft * log_probs).sum(dim=-1).mean()
        else:
            L_barrier = torch.tensor(0.0, device=total.device)
        total = total + w_barrier * L_barrier
        losses['L_barrier'] = L_barrier.item()

    if 'regime_logits' in outputs and 'regime_label' in batch:
        regime_logits = outputs['regime_logits'][valid]
        regime_true = batch['regime_label'][valid]
        L_regime = F.cross_entropy(regime_logits, regime_true)
        total = total + w_regime * L_regime
        losses['L_regime'] = L_regime.item()
    elif 'regime_logits' in outputs:
        losses['L_regime'] = 0.0

    losses['total'] = total.item()
    return total, losses


def compute_v6_loss(outputs, batch, w_ret=1.0, w_mfe=0.25, w_mae=0.25,
                    w_action=2.0, w_barrier=0.25, w_regime=0.1,
                    w_moe_balance=0.05, w_aux=0.1, w_confidence=0.15,
                    barrier_mode='fixed', action_weights=None, epoch=0,
                    sample_weights=None, mae_asym_weight=1.0):
    """Compute V6 composite loss: all V5 components + MoE balance + aux + confidence.

    Additional V6 loss components:
      - MoE load balancing: prevents expert collapse (weight: 0.01)
      - Auxiliary self-supervised: next-bar feature prediction MSE (weight: 0.1)
      - Confidence calibration: BCE on predicted vs actual correctness (weight: 0.15)
    """
    v5_loss, losses = compute_v5_loss(
        outputs, batch, w_ret=w_ret, w_mfe=w_mfe, w_mae=w_mae,
        w_action=w_action, w_barrier=w_barrier, w_regime=w_regime,
        barrier_mode=barrier_mode, action_weights=action_weights,
        epoch=epoch, sample_weights=sample_weights,
        mae_asym_weight=mae_asym_weight,
    )

    total = v5_loss

    if 'moe_balance_loss' in outputs:
        L_moe = outputs['moe_balance_loss']
        total = total + w_moe_balance * L_moe
        losses['L_moe_balance'] = L_moe.item()

    if 'aux_next_bar' in outputs and 'next_bar_features' in batch:
        valid = batch['valid']
        has_next = batch.get('has_next_bar', None)
        if has_next is not None:
            aux_mask = valid & has_next
        else:
            aux_mask = valid

        if aux_mask.sum() > 0:
            aux_pred = outputs['aux_next_bar'][aux_mask]
            aux_true = batch['next_bar_features'][aux_mask]
            per_sample_aux = F.mse_loss(aux_pred, aux_true, reduction='none').mean(dim=-1)
            if sample_weights is not None:
                sw = sample_weights[aux_mask]
                L_aux = (per_sample_aux * sw).sum() / sw.sum()
            else:
                L_aux = per_sample_aux.mean()
            total = total + w_aux * L_aux
            losses['L_aux'] = L_aux.item()
        else:
            losses['L_aux'] = 0.0

    if 'confidence' in outputs:
        valid = batch['valid']
        if valid.sum() > 0:
            conf_pred = outputs['confidence'][valid].squeeze(-1)
            action_logits = outputs['action_logits'][valid]
            action_true = batch['action_label'][valid]
            predicted_action = action_logits.argmax(dim=-1)
            correct = (predicted_action == action_true).float()
            per_sample_conf = F.binary_cross_entropy(conf_pred, correct.detach(), reduction='none')
            if sample_weights is not None:
                sw = sample_weights[valid]
                L_conf = (per_sample_conf * sw).sum() / sw.sum()
            else:
                L_conf = per_sample_conf.mean()
            total = total + w_confidence * L_conf
            losses['L_confidence'] = L_conf.item()
        else:
            losses['L_confidence'] = 0.0

    losses['total'] = total.item()
    return total, losses


def _extract_v5_arrays(concat_outputs, temperature=1.0):
    """Extract numpy arrays from concatenated model outputs for scoring/gating."""
    mu_R = concat_outputs['ret_mu'].numpy().squeeze(-1)
    mae_pred = concat_outputs['mae'].numpy().squeeze(-1)
    mfe_pred = concat_outputs['mfe'].numpy().squeeze(-1)
    action_logits = concat_outputs['action_logits'].numpy()

    sigma = None
    if 'ret_sigma' in concat_outputs:
        sigma = concat_outputs['ret_sigma'].numpy().squeeze(-1)
    elif 'ret_log_sigma' in concat_outputs:
        sigma = np.exp(concat_outputs['ret_log_sigma'].numpy().squeeze(-1))

    scaled_logits = action_logits / max(temperature, 0.1)
    action_probs = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    action_probs = action_probs / (action_probs.sum(axis=1, keepdims=True) + 1e-8)
    p_hold = action_probs[:, 0]
    p_long = action_probs[:, 1]
    p_short = action_probs[:, 2]
    p_trade = np.maximum(p_long, p_short)

    return {
        'mu_R': mu_R, 'sigma': sigma, 'mae': mae_pred, 'mfe': mfe_pred,
        'p_hold': p_hold, 'p_long': p_long, 'p_short': p_short, 'p_trade': p_trade,
        'action_logits': action_logits,
    }


def v5_quality_mask(arrays, cfg: V5QualityGateConfig, epoch: int = 999,
                    ref_arrays=None):
    """Apply data-adaptive quality gates with warmup bypass and minimum pass rate.

    Three safeguards prevent the multiplicative filtering problem:
    1. Warmup bypass: epochs 0-4 skip quality gates entirely (all pass)
    2. Lenient percentiles: p90 sigma/mae, p25 |mu_R|, p40 p_trade
       Each gate independently passes ~60-90% so intersection ≈ 25-40%
    3. Minimum pass rate: if combined pass rate < 10%, progressively relax
       all thresholds until at least 10% pass

    ref_arrays: if provided, percentile thresholds are computed from these
        (training-set arrays) instead of the test-set arrays, preventing
        lookahead bias in the forward test.

    Returns: boolean mask, diagnostics dict
    """
    ref = ref_arrays if ref_arrays is not None else arrays
    n = len(arrays['mu_R'])
    min_pass_rate = 0.10

    if epoch < 5:
        log.info("[V5_QUAL_DIAG] epoch=%d WARMUP: bypassing quality gates (all %d bars pass)", epoch, n)
        return np.ones(n, dtype=bool), {
            'total': n, 'final': n, 'warmup_bypass': True,
            'passed_sigma': n, 'passed_mae': n, 'passed_mu': n, 'passed_ptrade': n,
        }

    sigma_pass = np.ones(n, dtype=bool)
    adaptive_sigma = cfg.sigma_max
    if arrays['sigma'] is not None:
        ref_sigma = ref['sigma'] if ref.get('sigma') is not None else arrays['sigma']
        finite_sigma = ref_sigma[np.isfinite(ref_sigma)]
        if len(finite_sigma) > 100:
            adaptive_sigma = min(cfg.sigma_max, float(np.percentile(finite_sigma, 90)))
        sigma_pass = np.isfinite(arrays['sigma']) & (arrays['sigma'] <= adaptive_sigma)

    ref_mae = ref['mae'] if ref is not None else arrays['mae']
    finite_mae = ref_mae[np.isfinite(ref_mae)]
    adaptive_mae = cfg.mae_max
    if len(finite_mae) > 100:
        adaptive_mae = min(cfg.mae_max, float(np.percentile(finite_mae, 90)))
    mae_pass = np.isfinite(arrays['mae']) & (arrays['mae'] <= adaptive_mae)

    mu_R = arrays['mu_R']
    ref_mu = ref['mu_R'] if ref is not None else mu_R
    abs_mu_ref = np.abs(ref_mu[np.isfinite(ref_mu)])
    adaptive_mu_min = cfg.mu_R_min
    if len(abs_mu_ref) > 100:
        adaptive_mu_min = max(cfg.mu_R_min * 0.01, float(np.percentile(abs_mu_ref, 25)))
    edge_pass = np.isfinite(mu_R) & (np.abs(mu_R) >= adaptive_mu_min)

    pt = arrays['p_trade']
    ref_pt = ref['p_trade'] if ref is not None else pt
    adaptive_ptrade = cfg.p_trade_min
    if len(ref_pt) > 100:
        adaptive_ptrade = max(cfg.p_trade_min * 0.3, float(np.percentile(ref_pt, 40)))
    ptrade_pass = pt >= adaptive_ptrade

    final_mask = sigma_pass & mae_pass & edge_pass & ptrade_pass
    n_passed = int(np.sum(final_mask))

    if n_passed < n * min_pass_rate and n > 100:
        log.info("[V5_QUAL_DIAG] pass_rate=%.1f%% < %.0f%%, relaxing gates...",
                 100.0 * n_passed / n, 100.0 * min_pass_rate)
        for relax_step in range(5):
            relax_factor = 1.0 + 0.2 * (relax_step + 1)
            relaxed_sigma = adaptive_sigma * relax_factor
            relaxed_mae = adaptive_mae * relax_factor
            relaxed_mu = adaptive_mu_min / relax_factor
            relaxed_ptrade = adaptive_ptrade / relax_factor

            r_sigma = np.ones(n, dtype=bool)
            if arrays['sigma'] is not None:
                r_sigma = np.isfinite(arrays['sigma']) & (arrays['sigma'] <= relaxed_sigma)
            r_mae = np.isfinite(arrays['mae']) & (arrays['mae'] <= relaxed_mae)
            r_edge = np.isfinite(mu_R) & (np.abs(mu_R) >= relaxed_mu)
            r_ptrade = pt >= relaxed_ptrade

            relaxed_mask = r_sigma & r_mae & r_edge & r_ptrade
            n_relaxed = int(np.sum(relaxed_mask))

            if n_relaxed >= n * min_pass_rate:
                final_mask = relaxed_mask
                n_passed = n_relaxed
                adaptive_sigma = relaxed_sigma
                adaptive_mae = relaxed_mae
                adaptive_mu_min = relaxed_mu
                adaptive_ptrade = relaxed_ptrade
                log.info("[V5_QUAL_DIAG] relaxed at step %d: pass_rate=%.1f%% "
                         "sigma<=%.3f mae<=%.3f |mu|>=%.4f ptrade>=%.3f",
                         relax_step + 1, 100.0 * n_passed / n,
                         adaptive_sigma, adaptive_mae, adaptive_mu_min, adaptive_ptrade)
                break
        else:
            log.info("[V5_QUAL_DIAG] max relaxation reached, using finiteness-only gate")
            final_mask = np.isfinite(mu_R) & np.isfinite(arrays['mae'])
            if arrays['sigma'] is not None:
                final_mask &= np.isfinite(arrays['sigma'])
            n_passed = int(np.sum(final_mask))

    diag = {
        'total': n,
        'passed_sigma': int(np.sum(sigma_pass)),
        'passed_mae': int(np.sum(mae_pass)),
        'passed_mu': int(np.sum(edge_pass)),
        'passed_ptrade': int(np.sum(ptrade_pass)),
        'final': n_passed,
        'adaptive_sigma': adaptive_sigma,
        'adaptive_mae': adaptive_mae,
        'adaptive_mu_min': adaptive_mu_min,
        'adaptive_ptrade': adaptive_ptrade,
    }

    log.info("[V5_QUAL_DIAG] total=%d passed_sigma=%d passed_mae=%d "
             "passed_mu=%d passed_ptrade=%d final=%d (%.1f%%) | "
             "adaptive: sigma<=%.3f mae<=%.3f |mu|>=%.4f ptrade>=%.3f",
             diag['total'], diag['passed_sigma'], diag['passed_mae'],
             diag['passed_mu'], diag['passed_ptrade'], diag['final'],
             100.0 * diag['final'] / max(n, 1),
             adaptive_sigma, adaptive_mae, adaptive_mu_min, adaptive_ptrade)

    return final_mask, diag


def compute_v5_calibration(p_trade, realized_r, n_bins=10):
    """Compute 10-bin ECE for p_trade vs observed win-rate.

    Args:
        p_trade: predicted max(p_long, p_short) array
        realized_r: realized R from trades
        n_bins: number of calibration bins

    Returns:
        ece: float, bin_details: list of dicts
    """
    finite_mask = np.isfinite(p_trade) & np.isfinite(realized_r)
    p = p_trade[finite_mask]
    r = realized_r[finite_mask]
    observed_win = (r > 0).astype(float)

    if len(p) < 20:
        log.info("[V5_CALIB] insufficient data (%d bars), skipping", len(p))
        return 0.0, []

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_details = []

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = (p >= lo) & (p < hi) if b < n_bins - 1 else (p >= lo) & (p <= hi)
        n_bin = int(np.sum(in_bin))
        if n_bin == 0:
            bin_details.append({'n': 0, 'p': 0.0, 'obs': 0.0})
            continue
        avg_p = float(np.mean(p[in_bin]))
        avg_obs = float(np.mean(observed_win[in_bin]))
        ece += (n_bin / len(p)) * abs(avg_p - avg_obs)
        bin_details.append({'n': n_bin, 'p': round(avg_p, 3), 'obs': round(avg_obs, 3)})

    bin_str = " ".join(f"b{i}:(n={bd['n']},p={bd['p']:.3f},obs={bd['obs']:.3f})" for i, bd in enumerate(bin_details))
    log.info("[V5_CALIB] ece=%.4f %s", ece, bin_str)

    return ece, bin_details


def fit_temperature_scaling(logits, labels, n_classes=3, lr=0.01, max_iter=200):
    """Fit temperature scaling on validation action logits.

    Args:
        logits: numpy array of shape (N, n_classes) - raw action logits
        labels: numpy array of shape (N,) - true action labels
        n_classes: number of classes
        lr: learning rate for optimization
        max_iter: maximum iterations

    Returns:
        temperature: float, optimal temperature
        ece_before: float, ECE before scaling
        ece_after: float, ECE after scaling
    """
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    probs_before = torch.softmax(logits_t, dim=-1).numpy()
    preds_before = np.argmax(probs_before, axis=-1)
    confs_before = np.max(probs_before, axis=-1)
    correct_before = (preds_before == labels).astype(float)
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_before = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = (confs_before >= lo) & (confs_before < hi) if b < n_bins - 1 else (confs_before >= lo) & (confs_before <= hi)
        n_bin = int(np.sum(in_bin))
        if n_bin == 0:
            continue
        ece_before += (n_bin / len(confs_before)) * abs(np.mean(confs_before[in_bin]) - np.mean(correct_before[in_bin]))

    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled = logits_t / temperature.clamp(min=0.1)
        loss = F.cross_entropy(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temp_val = float(temperature.clamp(min=0.1).item())

    probs_after = torch.softmax(logits_t / temp_val, dim=-1).numpy()
    preds_after = np.argmax(probs_after, axis=-1)
    confs_after = np.max(probs_after, axis=-1)
    correct_after = (preds_after == labels).astype(float)
    ece_after = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = (confs_after >= lo) & (confs_after < hi) if b < n_bins - 1 else (confs_after >= lo) & (confs_after <= hi)
        n_bin = int(np.sum(in_bin))
        if n_bin == 0:
            continue
        ece_after += (n_bin / len(confs_after)) * abs(np.mean(confs_after[in_bin]) - np.mean(correct_after[in_bin]))

    log.info(f"[V5_TEMP_SCALE] temperature={temp_val:.4f} ECE: before={ece_before:.4f} → after={ece_after:.4f}")
    return temp_val, ece_before, ece_after


def compute_v5_scores(outputs_or_arrays, horizon_bars=16, score_lambda=0.5,
                      risk_proxy='mae', mae_cap=2.0, _arrays=None,
                      side_mode='action_head', rr_weight=0.0,
                      min_mu_r_score=0.03, slippage_bps=0.0,
                      sigma_discount=False, min_p_side=0.0,
                      min_p_short=0.0, side_aware_scoring=False):
    """Compute execution-aware v5 scores -- all in R-units.

    Two side-selection modes:
      side_mode='mu_sign' (LEGACY, DEPRECATED):
        edge_long  = p_long  * mu_R / risk       (positive only when mu_R > 0)
        edge_short = p_short * (-mu_R) / risk     (positive only when mu_R < 0)
        BUG: when mu_R > 0, edge_short is always negative -> NEVER picks SHORT

      side_mode='action_head' (DEFAULT, v5.0.6 fix):
        abs_mu = |mu_R|                            (magnitude only, direction-agnostic)
        edge_long  = p_long  * abs_mu / risk       (driven by learned p_long)
        edge_short = p_short * abs_mu / risk       (driven by learned p_short)
        Direction comes from action_head probabilities (trained on directional labels),
        not from mu_R sign. SHORTs fire when p_short > p_long.

    Penalty:
      side_mode='action_head':
        Conviction-based, scaled by magnitude:
        penalty = lambda * (1 - p_side) * mu_over_risk
        Scales proportionally with edge magnitude so penalty never dominates
        when mu_R is small (e.g. after debiasing). Score is positive when
        p_side > lambda/(1+lambda). Independent of mu_R sign.
        LONG and SHORT with equal p_side get equal penalty.

      side_mode='mu_sign' (LEGACY):
        When chosen side conflicts with mu_R sign, apply lambda * |mu_R| / risk penalty.
        LONG chosen but mu_R < 0 -> penalize. SHORT chosen but mu_R > 0 -> penalize.
        This discourages but does NOT block counter-mu_R trades.

    Risk/Reward bonus (rr_weight > 0):
        rr_ratio = mfe_pred / (mae_pred + eps)
        score += rr_weight * rr_ratio * abs_mu / risk
        Rewards setups with favorable excursion profiles (high MFE, low MAE).

    min_mu_r_score: minimum |mu_R| to generate a positive score. Trades with
        predicted |mu_R| below this floor get score = -inf to prevent taking
        trades with negligible expected move (even if mae is also tiny).
    """
    if _arrays is not None:
        mu_R = _arrays['mu_R']
        mae_pred = _arrays['mae']
        mfe_pred = _arrays['mfe']
        p_long = _arrays['p_long']
        p_short = _arrays['p_short']
    else:
        mu_R = outputs_or_arrays['ret_mu'].detach().cpu().numpy().squeeze(-1)
        mae_pred = outputs_or_arrays['mae'].detach().cpu().numpy().squeeze(-1)
        mfe_pred = outputs_or_arrays['mfe'].detach().cpu().numpy().squeeze(-1)
        action_logits = outputs_or_arrays['action_logits'].detach().cpu().numpy()
        action_probs = np.exp(action_logits - np.max(action_logits, axis=1, keepdims=True))
        action_probs = action_probs / (action_probs.sum(axis=1, keepdims=True) + 1e-8)
        p_long = action_probs[:, 1]
        p_short = action_probs[:, 2]

    risk = np.maximum(mae_pred, 0.25)

    if slippage_bps > 0:
        slippage_r = slippage_bps / 10000.0 / np.maximum(risk, 1e-6)
        mu_R_adj = mu_R - np.sign(mu_R) * slippage_r
        log.debug(f"[V5_SLIP] Deducting {slippage_bps:.1f} bps slippage from mu_R "
                  f"(avg deduction: {float(np.mean(slippage_r)):.4f} R)")
    else:
        mu_R_adj = mu_R

    abs_mu = np.abs(mu_R_adj)
    mu_over_risk = np.divide(abs_mu, risk, out=np.zeros_like(mu_R_adj), where=risk > 0)

    if side_mode == 'action_head' and side_aware_scoring:
        # Use abs_mu (magnitude) for both directions so the action head
        # (p_long / p_short) drives direction-selection, not mu_R sign.
        # Apply a 0.5x soft multiplier when mu_R sign disagrees with the
        # chosen direction — penalises but does NOT zero out counter-mu_R
        # signals.  The previous hard-zero approach (mu_short = max(-mu_R,0))
        # caused edge_short = 0 whenever mu_R > 0, which prevented ALL short
        # trades in models trained primarily on bull-market data.
        MU_DISAGREE_MULT = 0.5
        edge_long = p_long * mu_over_risk
        edge_short = p_short * mu_over_risk
        # Scale down longs when mu_R disagrees (mu_R < 0)
        long_mu_agrees = (mu_R_adj >= 0)
        edge_long = np.where(long_mu_agrees, edge_long, edge_long * MU_DISAGREE_MULT)
        # Scale down shorts when mu_R disagrees (mu_R > 0)
        short_mu_agrees = (mu_R_adj <= 0)
        edge_short = np.where(short_mu_agrees, edge_short, edge_short * MU_DISAGREE_MULT)
    elif side_mode == 'action_head':
        edge_long = p_long * mu_over_risk
        edge_short = p_short * mu_over_risk
    else:
        edge_long = p_long * np.divide(mu_R, risk, out=np.zeros_like(mu_R), where=risk > 0)
        edge_short = p_short * np.divide(-mu_R, risk, out=np.zeros_like(mu_R), where=risk > 0)

    best_edge = np.maximum(edge_long, edge_short)
    sides = np.where(edge_long >= edge_short, 1, -1)

    if side_mode == 'action_head':
        p_side = np.where(sides == 1, p_long, p_short)
        directional_penalty = (1.0 - p_side) * mu_over_risk
        penalty = score_lambda * directional_penalty
    else:
        penalty_long = np.maximum(0.0, -mu_R_adj)
        penalty_short = np.maximum(0.0, mu_R_adj)
        penalty_long_scaled = np.divide(penalty_long, risk, out=np.zeros_like(mu_R_adj), where=risk > 0)
        penalty_short_scaled = np.divide(penalty_short, risk, out=np.zeros_like(mu_R_adj), where=risk > 0)
        directional_penalty = np.where(sides == 1, penalty_long_scaled, penalty_short_scaled)
        penalty = score_lambda * directional_penalty

    scores = best_edge - penalty

    if rr_weight > 0:
        rr_ratio = np.divide(mfe_pred, risk, out=np.ones_like(mfe_pred), where=risk > 0)
        rr_bonus = rr_weight * rr_ratio * mu_over_risk
        scores = scores + rr_bonus

    n_sigma_discounted = 0
    if sigma_discount and _arrays is not None and _arrays.get('sigma') is not None:
        sigma = _arrays['sigma']
        sharpness = 1.0 / (1.0 + np.maximum(sigma, 0.0))
        scores = scores * sharpness
        n_sigma_discounted = int(np.sum(np.isfinite(scores) & (sharpness < 0.667)))

    n_pside_killed = 0
    if min_p_side > 0 and side_mode == 'action_head':
        p_side = np.where(sides == 1, p_long, p_short)
        low_conviction = p_side < min_p_side
        n_pside_killed = int(np.sum(low_conviction & np.isfinite(scores)))
        scores[low_conviction] = -np.inf

    n_pshort_killed = 0
    if min_p_short > 0 and side_mode == 'action_head':
        short_low_conv = (sides == -1) & (p_short < min_p_short)
        n_pshort_killed = int(np.sum(short_low_conv & np.isfinite(scores)))
        scores[short_low_conv] = -np.inf

    n_suppressed = 0
    if min_mu_r_score > 0:
        tiny_mu_mask = abs_mu < min_mu_r_score
        n_suppressed = int(np.sum(tiny_mu_mask))
        scores[tiny_mu_mask] = -np.inf

    n_long_sides = int(np.sum(sides == 1))
    n_short_sides = int(np.sum(sides == -1))

    finite_mask = np.isfinite(scores)
    finite_scores = scores[finite_mask]
    if len(finite_scores) > 0:
        s_mean = float(np.mean(finite_scores))
        s_std = float(np.std(finite_scores))
        s_p50 = float(np.percentile(finite_scores, 50))
        s_p90 = float(np.percentile(finite_scores, 90))
        s_pct_pos = float(np.mean(finite_scores > 0) * 100)
    else:
        s_mean = s_std = s_p50 = s_p90 = 0.0
        s_pct_pos = 0.0

    return scores, sides, {
        'mu_R_mean': float(np.nanmean(mu_R)),
        'mu_R_std': float(np.nanstd(mu_R)),
        'mae_R_mean': float(np.nanmean(mae_pred)),
        'mfe_R_mean': float(np.nanmean(mfe_pred)),
        'p_long_mean': float(np.nanmean(p_long)),
        'p_short_mean': float(np.nanmean(p_short)),
        'edge_long_mean': float(np.nanmean(edge_long)),
        'edge_short_mean': float(np.nanmean(edge_short)),
        'edge_L': edge_long,
        'edge_S': edge_short,
        'penalty_mean': float(np.nanmean(penalty)),
        'score_mean': s_mean,
        'score_std': s_std,
        'score_p50': s_p50,
        'score_p90': s_p90,
        'score_pct_positive': s_pct_pos,
        'n_finite_scores': int(np.sum(finite_mask)),
        'n_suppressed': n_suppressed,
        'side_mode': side_mode,
        'rr_weight': rr_weight,
        'min_mu_r_score': min_mu_r_score,
        'n_mu_suppressed': n_suppressed,
        'n_sigma_discounted': n_sigma_discounted,
        'n_pside_killed': n_pside_killed,
        'n_pshort_killed': n_pshort_killed,
        'side_aware_scoring': side_aware_scoring,
        'n_long_all': n_long_sides,
        'n_short_all': n_short_sides,
        'long_pct_all': float(100 * n_long_sides / max(n_long_sides + n_short_sides, 1)),
    }


def _tpd_controller_step(scores, quality_mask, candidate_mask,
                         current_threshold, epoch, val_bars,
                         tpd_cfg: V5TPDControllerConfig,
                         cooldown=4):
    """Adaptive threshold controller to hit target trades/day.

    Falls back to all-finite-scores when quality/candidate filtering
    yields too few eligible bars, ensuring the controller never gets stuck.

    Returns: new_threshold, n_trades, tpd, action_str
    """
    combined_mask = quality_mask.copy()
    if candidate_mask is not None:
        combined_mask &= candidate_mask

    eligible_scores = scores[combined_mask]
    finite_mask = np.isfinite(eligible_scores)
    eligible_finite = eligible_scores[finite_mask]

    val_days = val_bars / 96.0

    if len(eligible_finite) < 50:
        all_finite = scores[np.isfinite(scores)]
        if len(all_finite) >= 50:
            log.info("[V5_TPD_CTRL] Only %d eligible after gating, falling back to "
                     "all %d finite scores", len(eligible_finite), len(all_finite))
            eligible_finite = all_finite
            combined_mask = np.isfinite(scores)
        else:
            log.warning("[V5_TPD_CTRL] SKIP: only %d total finite scores", len(all_finite))
            return current_threshold, 0, 0.0, "SKIP"

    score_std = float(np.std(eligible_finite))
    sp5 = float(np.percentile(eligible_finite, 5))
    sp99 = float(np.percentile(eligible_finite, 99))

    if current_threshold is None:
        current_threshold = float(np.percentile(eligible_finite, 75))
        log.info("[V5_TPD_CTRL] Initializing threshold to p75=%.4f (from %d scores, "
                 "p5=%.4f p50=%.4f p95=%.4f p99=%.4f)",
                 current_threshold, len(eligible_finite),
                 sp5,
                 float(np.percentile(eligible_finite, 50)),
                 float(np.percentile(eligible_finite, 95)),
                 sp99)

    selected_indices = np.where(combined_mask)[0]
    above_thr = scores[selected_indices] >= current_threshold
    sel_above = selected_indices[above_thr]

    chronological_sel = sel_above[np.argsort(sel_above)]
    taken = []
    last_bar = -cooldown - 1
    for idx in chronological_sel:
        if idx - last_bar >= cooldown:
            taken.append(idx)
            last_bar = idx
    n_trades = len(taken)
    tpd = n_trades / max(val_days, 1e-6)

    target_tpd = tpd_cfg.target_tpd
    tol = tpd_cfg.tpd_tol
    error = tpd - target_tpd

    if epoch <= tpd_cfg.thr_warmup_epochs:
        action = "WARMUP"
        new_threshold = current_threshold
    elif abs(error) <= tol:
        action = "HOLD"
        new_threshold = current_threshold
    else:
        step = tpd_cfg.thr_step_mult * max(score_std, 1e-4) * max(1.0, abs(error))
        if error > tol:
            new_threshold = current_threshold + step
            action = "UP"
        else:
            new_threshold = current_threshold - step
            action = "DOWN"
        new_threshold = float(np.clip(new_threshold, sp5, sp99))

    floor = tpd_cfg.min_threshold_floor
    if new_threshold < floor:
        log.info("[V5_TPD_CTRL] TPD floor clamp engaged: attempted=%.4f floor=%.4f → clamped",
                 new_threshold, floor)
        new_threshold = floor

    log.info("[V5_TPD_CTRL] epoch=%d tpd=%.1f target=%.1f±%.1f thr=%.4f->%.4f "
             "step_mult=%.2f score_std=%.4f action=%s trades=%d eligible=%d "
             "clamp=[%.4f,%.4f] floor=%.4f",
             epoch, tpd, target_tpd, tol,
             current_threshold, new_threshold,
             tpd_cfg.thr_step_mult, score_std, action, n_trades,
             len(eligible_finite), sp5, sp99, floor)

    return new_threshold, n_trades, tpd, action


def _run_v5_sweep(scores, sides, precomputed_outcomes, precomputed_r,
                  val_bars, epoch, tp_mult, sl_mult,
                  target_tpd=6.5, target_tpd_tol=1.5, min_trades=30,
                  candidate_mask=None, risk_controls=None,
                  symbol_ids=None, horizon_bars=16,
                  quality_mask=None, score_threshold=None,
                  r_long=None, r_short=None, out_long=None, out_short=None,
                  close_prices=None, ema200_regime_gate=False,
                  timestamps=None, weekly_loss_cap=None, cooldown=4):
    """Score-based sweep for v5 model.

    If side-conditional arrays (r_long, r_short, out_long, out_short) are provided,
    uses predicted side to select outcome. Otherwise falls back to precomputed_r/outcomes
    (DEPRECATED oracle best-side).

    If score_threshold is provided, uses threshold-based selection (TPD controller).
    Otherwise falls back to percentile-based sweep.

    Capital protection:
    - ema200_regime_gate: hard-blocks LONG when close < EMA200, SHORT when close > EMA200
    - weekly_loss_cap: stops trading for remainder of week when cumulative weekly R drops below cap
    """
    from data.candidate_generator import apply_risk_controls, RiskControls

    COOLDOWN = cooldown

    ema200 = None
    if ema200_regime_gate and close_prices is not None:
        ema200 = _compute_ema(close_prices, 200)
        log.info("[V5_SWEEP] EMA200 regime gate ENABLED")

    use_side_conditional = (r_long is not None and r_short is not None
                           and out_long is not None and out_short is not None)

    _VALID_OUTCOMES = ["TP", "SL", "EXP_WIN", "EXP_LOSS", "TRAIL_WIN", "TRAIL_BE"]

    if use_side_conditional:
        side_r = np.where(sides == 1, r_long, r_short).astype(float)
        side_out = np.where(sides == 1, out_long, out_short)
        safe_outcomes = np.where(
            np.isin(side_out, _VALID_OUTCOMES),
            side_out, "NO_CANDIDATE"
        )
        safe_r = np.where(np.isnan(side_r), 0.0, side_r)
    else:
        safe_outcomes = np.where(
            np.isin(precomputed_outcomes, _VALID_OUTCOMES),
            precomputed_outcomes, "NO_CANDIDATE"
        )
        safe_r = precomputed_r.copy().astype(float)
        safe_r = np.where(np.isnan(safe_r), 0.0, safe_r)

    scores_work = scores.copy()
    if quality_mask is not None:
        scores_work[~quality_mask] = -np.inf
    if candidate_mask is not None:
        scores_work[~candidate_mask] = -np.inf

    tpd_lo = target_tpd - target_tpd_tol
    tpd_hi = target_tpd + target_tpd_tol
    val_days = val_bars / 96.0

    sweep_results = []
    best_in_freq_score = float('-inf')
    best_in_freq_label = ""
    best_any_score = float('-inf')
    best_any_pct = 0.0
    best_any_label = ""

    TOP_PCTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    if score_threshold is not None:
        thresholds_to_sweep = [score_threshold]
        labels_for_thresholds = ["tpd_ctrl"]
    else:
        thresholds_to_sweep = []
        labels_for_thresholds = []

    for pct in TOP_PCTS:
        finite_scores = scores_work[np.isfinite(scores_work)]
        if len(finite_scores) == 0:
            continue
        threshold = np.percentile(finite_scores, (1 - pct) * 100)
        thresholds_to_sweep.append(threshold)
        labels_for_thresholds.append(f"top{int(pct*100)}%")

    week_boundaries = None
    if weekly_loss_cap is not None and timestamps is not None:
        from datetime import timedelta
        trade_dates = np.array([datetime.utcfromtimestamp(ts / 1000) for ts in timestamps])
        week_ids = np.zeros(len(timestamps), dtype=np.int64)
        first_date = trade_dates[0]
        monday = first_date - timedelta(days=first_date.weekday())
        for i in range(len(trade_dates)):
            week_ids[i] = (trade_dates[i] - monday).days // 7
        week_boundaries = week_ids

    for thr, label in zip(thresholds_to_sweep, labels_for_thresholds):
        selected = scores_work >= thr
        sel_indices = np.where(selected)[0]
        if len(sel_indices) == 0:
            continue

        chronological_idx = sel_indices[np.argsort(sel_indices)]
        taken = []
        last_bar = -COOLDOWN - 1
        ema_blocked = 0
        weekly_blocked = 0
        current_week_r = 0.0
        current_week_id = -1
        week_killed = False
        for idx in chronological_idx:
            if idx - last_bar < COOLDOWN:
                continue
            if ema200 is not None:
                side_val = sides[idx]
                close_val = close_prices[idx]
                ema_val = ema200[idx]
                if side_val == 1 and close_val < ema_val:
                    ema_blocked += 1
                    continue
                if side_val == -1 and close_val > ema_val:
                    ema_blocked += 1
                    continue
            if weekly_loss_cap is not None and week_boundaries is not None:
                wk = week_boundaries[idx]
                if wk != current_week_id:
                    current_week_id = wk
                    current_week_r = 0.0
                    week_killed = False
                if week_killed:
                    weekly_blocked += 1
                    continue
            taken.append(idx)
            last_bar = idx
            if weekly_loss_cap is not None and week_boundaries is not None:
                trade_r = safe_r[idx]
                if not np.isnan(trade_r):
                    current_week_r += trade_r
                if current_week_r <= weekly_loss_cap:
                    week_killed = True
                    log.debug("[V5_SWEEP_GATE] weekly_cap hit: week=%d cumR=%.2f cap=%.2f",
                              current_week_id, current_week_r, weekly_loss_cap)

        n_above_thr = len(sel_indices)
        n_after_cooldown = len(taken)
        if ema_blocked > 0 and label == "tpd_ctrl":
            log.info(f"[V5_SWEEP_GATE] EMA200 blocked {ema_blocked} trades in {label}")
        if weekly_blocked > 0 and label == "tpd_ctrl":
            log.info(f"[V5_SWEEP_GATE] Weekly cap blocked {weekly_blocked} trades in {label}")

        if len(taken) < 5:
            log.debug("[V5_SWEEP_DIAG] %s: above_thr=%d after_cooldown=%d (<5, skipped)",
                      label, n_above_thr, n_after_cooldown)
            continue

        taken = np.array(taken)
        t_outcomes = safe_outcomes[taken]
        t_r = safe_r[taken]

        valid_trades = np.isin(t_outcomes, _VALID_OUTCOMES)
        n_valid_outcome = int(valid_trades.sum())
        if n_valid_outcome < 5:
            log.debug("[V5_SWEEP_DIAG] %s: above_thr=%d after_cooldown=%d valid_outcomes=%d (<5, skipped)",
                      label, n_above_thr, n_after_cooldown, n_valid_outcome)
            continue

        t_r_valid = t_r[valid_trades]
        n_trades = len(t_r_valid)
        wins = t_r_valid[t_r_valid > 0]
        losses = t_r_valid[t_r_valid <= 0]
        winrate = len(wins) / max(n_trades, 1)
        expect = np.mean(t_r_valid)
        median_r = np.median(t_r_valid)
        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        std_r = np.std(t_r_valid) if n_trades > 1 else 1.0
        trades_per_year = (n_trades / max(val_days, 1e-6)) * 252
        sharpe = expect / max(std_r, 1e-6) * np.sqrt(max(trades_per_year, 1))

        total_win = np.sum(wins)
        total_loss = abs(np.sum(losses))
        pf = min(total_win / max(total_loss, 1e-6), 999.99)

        n_tp = np.sum(t_outcomes[valid_trades] == "TP")
        n_sl = np.sum(t_outcomes[valid_trades] == "SL")
        n_exp = np.sum(np.isin(t_outcomes[valid_trades], ["EXP_WIN", "EXP_LOSS"]))
        pct_tp = n_tp / max(n_trades, 1)
        pct_sl = n_sl / max(n_trades, 1)
        pct_exp = n_exp / max(n_trades, 1)

        tpd = n_trades / max(val_days, 1e-6)

        pct_val = 0.0
        if label.startswith("top"):
            try:
                pct_val = int(label.replace("top", "").replace("%", "")) / 100.0
            except ValueError:
                pass

        equity = np.cumsum(t_r_valid)
        running_max = np.maximum.accumulate(equity)
        drawdowns = equity - running_max
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
        total_r = float(np.sum(t_r_valid))

        m = {
            'label': label, 'pct': pct_val, 'trades': n_trades,
            'expect': expect, 'winrate': winrate, 'sharpe': sharpe, 'pf': pf,
            'avg_win_r': avg_win, 'avg_loss_r': avg_loss, 'median_r': median_r,
            'pct_tp': pct_tp, 'pct_sl': pct_sl, 'pct_exp': pct_exp,
            'trades_per_day': tpd, 'threshold': thr,
            'max_dd': max_dd, 'total_r': total_r,
        }
        sweep_results.append(m)

        composite = expect * min(sharpe, 10.0)
        if tpd_lo <= tpd <= tpd_hi and n_trades >= min_trades:
            if composite > best_in_freq_score:
                best_in_freq_score = composite
                best_in_freq_label = label
        if composite > best_any_score:
            best_any_score = composite
            best_any_pct = pct_val
            best_any_label = label

    if best_in_freq_label:
        best_label = best_in_freq_label
        best_score_val = best_in_freq_score
        best_pct = next(m['pct'] for m in sweep_results if m['label'] == best_in_freq_label)
    elif best_any_label:
        best_label = best_any_label
        best_score_val = best_any_score
        best_pct = best_any_pct
    else:
        best_label = ""
        best_score_val = float('-inf')
        best_pct = 0.0

    score_arr = np.array(scores)
    combined_eligible = np.ones(len(score_arr), dtype=bool)
    if quality_mask is not None:
        combined_eligible &= quality_mask
    if candidate_mask is not None:
        combined_eligible &= candidate_mask
    scores_eligible = score_arr[combined_eligible]
    scores_finite = scores_eligible[np.isfinite(scores_eligible)]
    log.info("-" * 120)
    log.info("[V5_SCORE_DIAG] total=%d eligible=%d finite=%d nan_or_inf=%d",
             len(score_arr), len(scores_eligible), len(scores_finite),
             len(scores_eligible) - len(scores_finite))
    if len(scores_finite) > 0:
        sp50 = float(np.percentile(scores_finite, 50))
        sp75 = float(np.percentile(scores_finite, 75))
        sp90 = float(np.percentile(scores_finite, 90))
        sp95 = float(np.percentile(scores_finite, 95))
        sp99 = float(np.percentile(scores_finite, 99))
        log.info("v5 score percentiles (val): p50=%.4f p75=%.4f p90=%.4f p95=%.4f p99=%.4f",
                 sp50, sp75, sp90, sp95, sp99)
    else:
        log.info("v5 score percentiles: EMPTY (no finite candidate scores)")

    n_valid_outcomes_total = np.sum(np.isin(safe_outcomes, _VALID_OUTCOMES))
    n_no_cand = np.sum(safe_outcomes == "NO_CANDIDATE")
    log.info("[V5_SWEEP_PIPELINE] precomputed_outcomes: valid=%d no_candidate=%d total=%d",
             n_valid_outcomes_total, n_no_cand, len(safe_outcomes))
    log.info("V5 SWEEP (epoch %d) | cooldown=%d | TP=%.1fx SL=%.1fx ATR | val_days=%.1f | target=%.1f±%.1f tpd",
             epoch, COOLDOWN, tp_mult, sl_mult, val_days, target_tpd, target_tpd_tol)
    log.info("%-10s %5s %8s %6s %6s %5s | %6s %6s %6s | %4s %4s %4s | %5s",
             "Select", "Trds", "Expect", "WR", "Shrpe", "PF",
             "WinR", "LosR", "MedR", "%TP", "%SL", "%EX", "T/Day")
    log.info("-" * 120)
    for m in sweep_results:
        marker = ""
        if m['label'] == best_label and m['trades'] >= min_trades and best_score_val > float('-inf'):
            marker = " <<< BEST"
        log.info("%-10s %5d %+8.3f %5.1f%% %6.2f %5.2f | %+6.3f %+6.3f %+6.3f | %3.0f%% %3.0f%% %3.0f%% | %5.1f%s",
                 m['label'], m['trades'], m['expect'], m['winrate']*100, m['sharpe'], m['pf'],
                 m['avg_win_r'], m['avg_loss_r'], m['median_r'],
                 m['pct_tp']*100, m['pct_sl']*100, m['pct_exp']*100, m['trades_per_day'], marker)
    log.info("-" * 120)

    best_row = next((m for m in sweep_results if m['label'] == best_label), None)
    best_pf = best_row['pf'] if best_row else 0.0
    best_max_dd = best_row['max_dd'] if best_row else 0.0
    best_tpd = best_row['trades_per_day'] if best_row else 0.0
    best_threshold = best_row['threshold'] if best_row else 0.0

    return sweep_results, best_label, best_score_val, best_pct, best_pf, best_max_dd, best_tpd, best_threshold


def _run_per_symbol_sweep(scores, sides, precomputed_outcomes, precomputed_r,
                          val_bars, symbol_ids, symbols_list, global_threshold,
                          r_long=None, r_short=None, out_long=None, out_short=None,
                          quality_mask=None, candidate_mask=None,
                          min_trades_per_symbol=10, cooldown=4):
    """Per-symbol threshold sweep: find optimal threshold per symbol.

    For each symbol, runs a mini-sweep on its bars only.
    Returns dict mapping symbol_id -> threshold (or global_threshold*3 for NO EDGE symbols).
    """
    COOLDOWN = cooldown
    _VALID_OUTCOMES = ["TP", "SL", "EXP_WIN", "EXP_LOSS", "TRAIL_WIN", "TRAIL_BE"]

    use_side_conditional = (r_long is not None and r_short is not None
                           and out_long is not None and out_short is not None)

    if use_side_conditional:
        side_r = np.where(sides == 1, r_long, r_short).astype(float)
        side_out = np.where(sides == 1, out_long, out_short)
        safe_outcomes = np.where(np.isin(side_out, _VALID_OUTCOMES), side_out, "NO_CANDIDATE")
        safe_r = np.where(np.isnan(side_r), 0.0, side_r)
    else:
        safe_outcomes = np.where(np.isin(precomputed_outcomes, _VALID_OUTCOMES),
                                 precomputed_outcomes, "NO_CANDIDATE")
        safe_r = precomputed_r.copy().astype(float)
        safe_r = np.where(np.isnan(safe_r), 0.0, safe_r)

    scores_work = scores.copy()
    if quality_mask is not None:
        scores_work[~quality_mask] = -np.inf
    if candidate_mask is not None:
        scores_work[~candidate_mask] = -np.inf

    unique_sym_ids = np.unique(symbol_ids)
    per_sym_thresholds = {}
    no_edge_symbols = []

    log.info("=" * 100)
    log.info("  PER-SYMBOL THRESHOLD SWEEP")
    log.info("=" * 100)
    log.info(f"{'Symbol':>12} {'Bars':>6} {'BestThr':>10} {'Trades':>7} {'E[R]':>10} "
             f"{'WR':>7} {'PF':>7} {'TotalR':>10} {'Status':>10}")
    log.info("-" * 100)

    for sym_id in unique_sym_ids:
        sym_mask = symbol_ids == sym_id
        sym_name = symbols_list[sym_id] if symbols_list and sym_id < len(symbols_list) else f"sym_{sym_id}"
        sym_scores = scores_work[sym_mask]
        sym_safe_r = safe_r[sym_mask]
        sym_safe_outcomes = safe_outcomes[sym_mask]
        sym_sides = sides[sym_mask]
        n_sym_bars = int(np.sum(sym_mask))

        sym_finite = sym_scores[np.isfinite(sym_scores)]
        if len(sym_finite) < min_trades_per_symbol:
            high_bar = global_threshold * 3.0
            per_sym_thresholds[int(sym_id)] = high_bar
            no_edge_symbols.append(sym_name)
            log.info(f"{sym_name:>12} {n_sym_bars:>6} {high_bar:>10.4f} {0:>7} {'-':>10} "
                     f"{'-':>7} {'-':>7} {'-':>10} {'HIGH_BAR':>10}")
            continue

        thresholds_to_try = [global_threshold]
        for pct in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            thr = float(np.percentile(sym_finite, (1 - pct) * 100))
            thresholds_to_try.append(thr)

        best_sym_composite = float('-inf')
        best_sym_threshold = global_threshold
        best_sym_metrics = None

        for thr in thresholds_to_try:
            selected = sym_scores >= thr
            sel_idx = np.where(selected)[0]
            if len(sel_idx) == 0:
                continue

            taken = []
            last_bar = -COOLDOWN - 1
            for idx in sel_idx:
                if idx - last_bar < COOLDOWN:
                    continue
                taken.append(idx)
                last_bar = idx

            if len(taken) < max(5, min_trades_per_symbol // 2):
                continue

            taken = np.array(taken)
            t_outcomes = sym_safe_outcomes[taken]
            t_r = sym_safe_r[taken]
            valid_trades = np.isin(t_outcomes, _VALID_OUTCOMES)
            n_valid = int(valid_trades.sum())
            if n_valid < 5:
                continue

            t_r_valid = t_r[valid_trades]
            n_trades = len(t_r_valid)
            wins = t_r_valid[t_r_valid > 0]
            losses = t_r_valid[t_r_valid <= 0]
            expect = float(np.mean(t_r_valid))
            winrate = len(wins) / max(n_trades, 1)
            std_r = float(np.std(t_r_valid)) if n_trades > 1 else 1.0
            sym_days = n_sym_bars / 96.0
            trades_per_year = (n_trades / max(sym_days, 1e-6)) * 252
            sharpe = expect / max(std_r, 1e-6) * np.sqrt(max(trades_per_year, 1))
            total_win = float(np.sum(wins))
            total_loss = abs(float(np.sum(losses)))
            pf = min(total_win / max(total_loss, 1e-6), 999.99)
            total_r = float(np.sum(t_r_valid))

            composite = expect * min(sharpe, 10.0)
            if composite > best_sym_composite and n_trades >= min_trades_per_symbol:
                best_sym_composite = composite
                best_sym_threshold = thr
                best_sym_metrics = {
                    'trades': n_trades, 'expect': expect, 'winrate': winrate,
                    'sharpe': sharpe, 'pf': pf, 'total_r': total_r,
                }

        if best_sym_metrics is not None and best_sym_metrics['expect'] > 0:
            per_sym_thresholds[int(sym_id)] = best_sym_threshold
            m = best_sym_metrics
            log.info(f"{sym_name:>12} {n_sym_bars:>6} {best_sym_threshold:>10.4f} {m['trades']:>7} "
                     f"{m['expect']:>+10.4f} {m['winrate']:>6.1%} {m['pf']:>7.2f} "
                     f"{m['total_r']:>+10.4f} {'ACTIVE':>10}")
        else:
            high_bar = global_threshold * 3.0
            per_sym_thresholds[int(sym_id)] = high_bar
            no_edge_symbols.append(sym_name)
            if best_sym_metrics is not None:
                m = best_sym_metrics
                log.info(f"{sym_name:>12} {n_sym_bars:>6} {high_bar:>10.4f} {m['trades']:>7} "
                         f"{m['expect']:>+10.4f} {m['winrate']:>6.1%} {m['pf']:>7.2f} "
                         f"{m['total_r']:>+10.4f} {'HIGH_BAR':>10}")
            else:
                log.info(f"{sym_name:>12} {n_sym_bars:>6} {high_bar:>10.4f} {0:>7} {'-':>10} "
                         f"{'-':>7} {'-':>7} {'-':>10} {'HIGH_BAR':>10}")

    log.info("-" * 100)
    active = [s for s in symbols_list if s not in no_edge_symbols] if symbols_list else []
    log.info(f"  Active symbols: {len(active)}/{len(unique_sym_ids)} | "
             f"NO EDGE: {no_edge_symbols if no_edge_symbols else 'none'}")
    log.info("=" * 100)

    return per_sym_thresholds, no_edge_symbols


def _build_train_ref_arrays(model, device, train_feat, train_sym_ids,
                            config, total_train,
                            use_v6=False, v6_seq_len=16):
    """Run inference on training data to build reference arrays for quality gate.

    These training-set statistics prevent lookahead bias in the forward test
    by anchoring percentile thresholds to in-sample distributions only.
    """
    log.info(f"[V5_REF] Building training reference arrays ({total_train} bars) for quality gate...")
    model.eval()

    if use_v6:
        unique_syms = np.unique(train_sym_ids)
        feat_per_sym = []
        ret_per_sym = []
        mfe_per_sym = []
        mae_per_sym = []
        vol_per_sym = []
        act_per_sym = []
        valid_per_sym = []
        symid_per_sym = []
        for s in unique_syms:
            mask = train_sym_ids == s
            n_s = int(mask.sum())
            feat_per_sym.append(train_feat[mask])
            ret_per_sym.append(np.zeros(n_s, dtype=np.float32))
            mfe_per_sym.append(np.zeros(n_s, dtype=np.float32))
            mae_per_sym.append(np.zeros(n_s, dtype=np.float32))
            vol_per_sym.append(np.zeros(n_s, dtype=np.float32))
            act_per_sym.append(np.zeros(n_s, dtype=np.int64))
            valid_per_sym.append(np.ones(n_s, dtype=np.bool_))
            symid_per_sym.append(train_sym_ids[mask])
        train_ds = V6SequenceDataset(
            features_per_symbol=feat_per_sym,
            ret_R_per_symbol=ret_per_sym,
            mfe_R_per_symbol=mfe_per_sym,
            mae_R_per_symbol=mae_per_sym,
            vol_h_per_symbol=vol_per_sym,
            action_per_symbol=act_per_sym,
            valid_per_symbol=valid_per_sym,
            symbol_ids_per_symbol=symid_per_sym,
            seq_len=v6_seq_len,
        )
        log.info(f"[V6_REF] Using V6SequenceDataset for ref arrays: {len(train_ds)} samples, seq_len={v6_seq_len}")
    else:
        train_valid = np.ones(total_train, dtype=np.float32)
        train_ds = V5Dataset(
            train_feat,
            np.zeros(total_train, dtype=np.float32),
            np.zeros(total_train, dtype=np.float32),
            np.zeros(total_train, dtype=np.float32),
            np.zeros(total_train, dtype=np.float32),
            np.zeros(total_train, dtype=np.int64),
            train_valid,
            train_sym_ids,
            np.zeros(total_train, dtype=np.int64),
            np.zeros((total_train, 1), dtype=np.float32),
        )
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=False)

    all_outputs = {
        'ret_mu': [], 'ret_log_sigma': [], 'ret_sigma': [],
        'mae': [], 'mfe': [], 'action_logits': []
    }

    with torch.no_grad():
        for batch in train_loader:
            feat = batch['features'].to(device)
            sym_id = batch.get('symbol_id')
            if sym_id is not None:
                sym_id = sym_id.to(device)
            outputs = model(feat, symbol_ids=sym_id)
            for k in all_outputs:
                if k in outputs:
                    all_outputs[k].append(outputs[k].detach().cpu())

    concat_outputs = {}
    for k in all_outputs:
        if all_outputs[k]:
            concat_outputs[k] = torch.cat(all_outputs[k], dim=0)

    ref_arrays = _extract_v5_arrays(concat_outputs, temperature=config.temperature)

    train_scores, _, _ = compute_v5_scores(
        None, horizon_bars=config.horizon,
        score_lambda=config.score_lambda,
        risk_proxy=config.risk_proxy,
        mae_cap=config.mae_cap,
        _arrays=ref_arrays,
        side_mode=config.side_mode,
        rr_weight=config.rr_weight,
        slippage_bps=config.slippage_base_bps,
        sigma_discount=config.sigma_discount,
        min_p_side=config.min_p_side,
        min_p_short=config.min_p_short,
        side_aware_scoring=config.side_aware_scoring,
    )
    ref_arrays['_train_scores'] = train_scores

    log.info(f"[V5_REF] Training reference arrays built: mu_R mean={float(np.mean(ref_arrays['mu_R'])):.4f}, "
             f"mae mean={float(np.mean(ref_arrays['mae'])):.4f}, "
             f"score mean={float(np.nanmean(train_scores[np.isfinite(train_scores)])):.4f}")

    return ref_arrays


def _compute_side_distribution(sides_array, indices=None):
    """Compute LONG/SHORT/HOLD counts and percentages for given indices.

    Args:
        sides_array: full array of side values (1=LONG, -1=SHORT, 0=HOLD)
        indices: subset of indices to analyze (None = use all)

    Returns:
        dict with counts and percentages for each side
    """
    if indices is not None and len(indices) > 0:
        s = sides_array[indices]
    elif indices is not None:
        s = np.array([], dtype=sides_array.dtype)
    else:
        s = sides_array

    n = len(s)
    n_long = int(np.sum(s == 1))
    n_short = int(np.sum(s == -1))
    n_hold = int(np.sum(s == 0))

    long_short_total = n_long + n_short
    long_pct = 100.0 * n_long / max(long_short_total, 1)
    short_pct = 100.0 * n_short / max(long_short_total, 1)
    hold_pct = 100.0 * n_hold / max(n, 1)

    return {
        'total': n,
        'long': n_long,
        'short': n_short,
        'hold': n_hold,
        'long_pct': long_pct,
        'short_pct': short_pct,
        'hold_pct': hold_pct,
    }


def _print_directional_balance_diagnostics(stage_distributions, gate_blocks):
    """Print stage-by-stage side distribution table and imbalance warnings.

    Args:
        stage_distributions: dict of stage_name -> side distribution dict
        gate_blocks: Counter with keys like "ema200", "multi_regime", "quality", etc.
    """
    log.info("")
    log.info("=" * 80)
    log.info("  DIRECTIONAL BALANCE DIAGNOSTICS")
    log.info("=" * 80)
    log.info(f"  {'Stage':<20} {'Total':>8} {'LONG':>8} {'LONG%':>8} {'SHORT':>8} {'SHORT%':>8} {'HOLD':>8} {'HOLD%':>8}")
    log.info("  " + "-" * 76)

    for stage_name, dist in stage_distributions.items():
        log.info(f"  {stage_name:<20} {dist['total']:>8} "
                 f"{dist['long']:>8} {dist['long_pct']:>7.1f}% "
                 f"{dist['short']:>8} {dist['short_pct']:>7.1f}% "
                 f"{dist['hold']:>8} {dist['hold_pct']:>7.1f}%")

    log.info("")
    log.info("  Gate Blocks:")
    if gate_blocks:
        for gate_name, count in gate_blocks.most_common():
            log.info(f"    {gate_name:<25} {count:>6} blocked")
    else:
        log.info("    (none)")

    for stage_name, dist in stage_distributions.items():
        long_short_total = dist['long'] + dist['short']
        if long_short_total > 0:
            max_pct = max(dist['long_pct'], dist['short_pct'])
            if max_pct > 80.0:
                dominant = "LONG" if dist['long_pct'] > dist['short_pct'] else "SHORT"
                log.warning(f"[V5_BALANCE_WARN] Stage '{stage_name}': {dominant} dominance "
                            f"at {max_pct:.1f}% (>{80}%% threshold). "
                            f"LONG={dist['long']}, SHORT={dist['short']}")

    log.info("=" * 80)


def run_v5_forward_test(
    model, device,
    test_features, test_outcomes, test_realized_r,
    test_sym_ids, test_cand_mask, test_valid,
    test_bars, config: V5ForwardTestConfig,
    test_start_date=None, test_end_date=None,
    test_timestamps=None,
    r_long=None, r_short=None, out_long=None, out_short=None,
    close_prices=None, ema200_regime_gate=False,
    high_prices=None, low_prices=None,
    train_ref_arrays=None,
    use_v6=False, v6_seq_len=16,
):
    """Run forward test with completely frozen decision layer.

    No TPD adaptation, no calibration tuning, no percentile sweep.
    Single pass: compute outputs → scores → quality gate → fixed threshold → cooldown → trades.
    """
    from data.common import generate_v5_sweep_outcomes

    model.eval()

    if use_v6:
        unique_syms = np.unique(test_sym_ids)
        feat_per_sym = []
        ret_per_sym = []
        mfe_per_sym = []
        mae_per_sym = []
        vol_per_sym = []
        act_per_sym = []
        valid_per_sym = []
        symid_per_sym = []
        for s in unique_syms:
            mask = test_sym_ids == s
            n_s = int(mask.sum())
            feat_per_sym.append(test_features[mask])
            ret_per_sym.append(np.zeros(n_s, dtype=np.float32))
            mfe_per_sym.append(np.zeros(n_s, dtype=np.float32))
            mae_per_sym.append(np.zeros(n_s, dtype=np.float32))
            vol_per_sym.append(np.zeros(n_s, dtype=np.float32))
            act_per_sym.append(np.zeros(n_s, dtype=np.int64))
            valid_per_sym.append(np.array(test_valid[mask], dtype=np.bool_))
            symid_per_sym.append(test_sym_ids[mask])
        test_ds = V6SequenceDataset(
            features_per_symbol=feat_per_sym,
            ret_R_per_symbol=ret_per_sym,
            mfe_R_per_symbol=mfe_per_sym,
            mae_R_per_symbol=mae_per_sym,
            vol_h_per_symbol=vol_per_sym,
            action_per_symbol=act_per_sym,
            valid_per_symbol=valid_per_sym,
            symbol_ids_per_symbol=symid_per_sym,
            seq_len=v6_seq_len,
        )
        log.info(f"[V6_FWD] Using V6SequenceDataset for forward test: {len(test_ds)} samples, seq_len={v6_seq_len}")
    else:
        test_ds = V5Dataset(
            test_features,
            np.zeros(len(test_features), dtype=np.float32),
            np.zeros(len(test_features), dtype=np.float32),
            np.zeros(len(test_features), dtype=np.float32),
            np.zeros(len(test_features), dtype=np.float32),
            np.zeros(len(test_features), dtype=np.int64),
            test_valid,
            test_sym_ids,
            np.zeros(len(test_features), dtype=np.int64),
            np.zeros((len(test_features), 1), dtype=np.float32),
        )
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    all_outputs = {
        'ret_mu': [], 'ret_log_sigma': [], 'ret_sigma': [],
        'mae': [], 'mfe': [], 'action_logits': []
    }

    with torch.no_grad():
        for batch in test_loader:
            feat = batch['features'].to(device)
            sym_id = batch.get('symbol_id')
            if sym_id is not None:
                sym_id = sym_id.to(device)
            outputs = model(feat, symbol_ids=sym_id)
            for k in all_outputs:
                if k in outputs:
                    all_outputs[k].append(outputs[k].detach().cpu())

    concat_outputs = {}
    for k in all_outputs:
        if all_outputs[k]:
            concat_outputs[k] = torch.cat(all_outputs[k], dim=0)

    arrays = _extract_v5_arrays(concat_outputs, temperature=config.temperature)
    if config.temperature != 1.0:
        log.info(f"[V5_FWD] Applied temperature={config.temperature:.4f} to action logits")

    if config.mu_debias and test_sym_ids is not None:
        mu_R_raw = arrays['mu_R'].copy()
        alpha = config.mu_debias_alpha
        unique_syms = np.unique(test_sym_ids)
        ema_by_sym = {int(s): 0.0 for s in unique_syms}
        for i in range(len(mu_R_raw)):
            sym_id = int(test_sym_ids[i])
            mu_val = float(mu_R_raw[i])
            if not np.isnan(mu_val):
                ema_by_sym[sym_id] = (1.0 - alpha) * ema_by_sym[sym_id] + alpha * mu_val
            arrays['mu_R'][i] = mu_R_raw[i] - ema_by_sym[sym_id]
        for sym_id in unique_syms:
            sym_mask = test_sym_ids == sym_id
            sym_name = sym_id if not hasattr(config, 'symbols_list') or config.symbols_list is None else (
                config.symbols_list[int(sym_id)] if int(sym_id) < len(config.symbols_list) else str(sym_id)
            )
            log.info(f"[V5_MU_DEBIAS] {sym_name}: final_ema={ema_by_sym[int(sym_id)]:+.6f} "
                     f"raw_mean={float(np.nanmean(mu_R_raw[sym_mask])):+.6f} "
                     f"debiased_mean={float(np.nanmean(arrays['mu_R'][sym_mask])):+.6f}")
        overall_raw = float(np.nanmean(mu_R_raw))
        overall_deb = float(np.nanmean(arrays['mu_R']))
        log.info(f"[V5_MU_DEBIAS] Overall: raw_mean={overall_raw:+.6f} debiased_mean={overall_deb:+.6f} "
                 f"alpha={alpha}")
    elif config.mu_debias:
        mu_R_raw = arrays['mu_R'].copy()
        alpha = config.mu_debias_alpha
        ema_val = 0.0
        for i in range(len(mu_R_raw)):
            mu_val = float(mu_R_raw[i])
            if not np.isnan(mu_val):
                ema_val = (1.0 - alpha) * ema_val + alpha * mu_val
            arrays['mu_R'][i] = mu_R_raw[i] - ema_val
        log.info(f"[V5_MU_DEBIAS] Single-symbol mode: final_ema={ema_val:+.6f} "
                 f"raw_mean={float(np.nanmean(mu_R_raw)):+.6f} "
                 f"debiased_mean={float(np.nanmean(arrays['mu_R'])):+.6f}")

    qg_cfg = config.quality_gate_cfg or V5QualityGateConfig()
    quality_mask, qual_diag = v5_quality_mask(arrays, qg_cfg, epoch=999,
                                               ref_arrays=train_ref_arrays)

    scores, sides, score_diag = compute_v5_scores(
        None, horizon_bars=config.horizon,
        score_lambda=config.score_lambda,
        risk_proxy=config.risk_proxy,
        mae_cap=config.mae_cap,
        _arrays=arrays,
        side_mode=config.side_mode,
        rr_weight=config.rr_weight,
        slippage_bps=config.slippage_base_bps,
        sigma_discount=config.sigma_discount,
        min_p_side=config.min_p_side,
        min_p_short=config.min_p_short,
        side_aware_scoring=config.side_aware_scoring,
    )

    if 'edge_L' in score_diag:
        arrays['edge_L'] = score_diag['edge_L']
    if 'edge_S' in score_diag:
        arrays['edge_S'] = score_diag['edge_S']

    log.info(f"[V5_FWD] side_mode={config.side_mode} rr_weight={config.rr_weight}")
    log.info(f"[V5_FWD] Score stats: mean={score_diag['score_mean']:.4f} "
             f"p50={score_diag['score_p50']:.4f} p90={score_diag['score_p90']:.4f} "
             f"%pos={score_diag['score_pct_positive']:.1f}%")
    log.info(f"[V5_FWD] Score side analysis: p_long_mean={score_diag['p_long_mean']:.4f} "
             f"p_short_mean={score_diag['p_short_mean']:.4f} "
             f"edge_long_mean={score_diag['edge_long_mean']:.4f} "
             f"edge_short_mean={score_diag['edge_short_mean']:.4f}")
    mu_R_post = arrays['mu_R']
    abs_mu_post = np.abs(mu_R_post)
    finite_scores_diag = scores[np.isfinite(scores)]
    p_side_arr = np.where(sides == 1, arrays['p_long'], arrays['p_short'])
    log.info(f"[V5_FWD] Score components: edge_mean={score_diag['edge_long_mean']:.4f} "
             f"penalty_mean={score_diag['penalty_mean']:.4f} net_score_mean={score_diag['score_mean']:.4f}")
    log.info(f"[V5_FWD] mu_R post-debias: mean={float(np.nanmean(mu_R_post)):+.4f} "
             f"std={float(np.nanstd(mu_R_post)):.4f} |mu_R|_mean={float(np.nanmean(abs_mu_post)):.4f}")
    log.info(f"[V5_FWD] p_side: mean={float(np.nanmean(p_side_arr)):.4f} "
             f"p10={float(np.nanpercentile(p_side_arr, 10)):.4f} "
             f"p50={float(np.nanpercentile(p_side_arr, 50)):.4f} "
             f"p90={float(np.nanpercentile(p_side_arr, 90)):.4f}")
    all_long = int(np.sum(sides == 1))
    all_short = int(np.sum(sides == -1))
    log.info(f"[V5_SIDE_DIAG] ALL bars: long={all_long} short={all_short} "
             f"long_pct={100*all_long/max(all_long+all_short,1):.1f}% "
             f"(if >95%% one-sided, this is MODEL BIAS not a bug)")
    log.info(f"[V5_FWD] Quality gate: {qual_diag.get('passed_pct', 0):.1f}% pass "
             f"({qual_diag.get('final', 0)}/{qual_diag.get('total', 0)})")
    hard_floor = config.min_threshold if config.min_threshold is not None else 0.02
    effective_threshold = max(hard_floor, config.score_threshold)
    if config.score_threshold < hard_floor:
        log.info(f"[V5_FWD] Hard floor engaged: threshold {config.score_threshold:.4f} < floor {hard_floor:.4f} → clamped to {effective_threshold:.4f}")
    if config.max_threshold is not None and effective_threshold > config.max_threshold:
        log.info(f"[V5_FWD] Ceiling cap engaged: threshold {effective_threshold:.4f} > cap {config.max_threshold:.4f} → clamped to {config.max_threshold:.4f}")
        effective_threshold = config.max_threshold
    pct_floor = None
    if config.min_threshold_pct is not None:
        pct_scores = scores.copy()
        pct_scores[np.isnan(pct_scores)] = -np.inf
        if quality_mask is not None:
            pct_scores[~quality_mask] = -np.inf
        if test_cand_mask is not None:
            pct_scores[~test_cand_mask.astype(bool)] = -np.inf
        finite_scores = pct_scores[np.isfinite(pct_scores)]
        if len(finite_scores) > 0:
            pct_floor = float(np.percentile(finite_scores, config.min_threshold_pct))
            log.info(f"[V5_FWD] Percentile floor: p{config.min_threshold_pct:.0f} of valid scores = {pct_floor:.4f} "
                     f"(from {len(finite_scores)} valid bars)")
        else:
            log.warning("[V5_FWD] No valid scores for percentile calculation — percentile floor disabled")

    if config.min_threshold is not None and effective_threshold < config.min_threshold:
        log.info(f"[V5_FWD] Fixed floor: {effective_threshold:.4f} < {config.min_threshold:.4f} → clamped")
        effective_threshold = config.min_threshold
    if pct_floor is not None and effective_threshold < pct_floor:
        log.info(f"[V5_FWD] Percentile floor: {effective_threshold:.4f} < p{config.min_threshold_pct:.0f}={pct_floor:.4f} → clamped")
        effective_threshold = pct_floor
    if config.max_threshold is not None and effective_threshold > config.max_threshold:
        log.info(f"[V5_FWD] Final ceiling cap: {effective_threshold:.4f} > cap {config.max_threshold:.4f} → clamped (ceiling always wins)")
        effective_threshold = config.max_threshold

    log.info(f"[V5_FWD] Effective threshold={effective_threshold:.4f} "
             f"(calibrated={config.score_threshold:.4f}, "
             f"fixed_floor={config.min_threshold}, "
             f"ceiling_cap={config.max_threshold}, "
             f"pct_floor={pct_floor}) cooldown={config.cooldown}")
    if config.max_trades_per_day is not None:
        if test_timestamps is not None:
            log.info(f"[V5_FWD] Max trades/day cap ENABLED: {config.max_trades_per_day}")
        else:
            log.warning("[V5_FWD] Max trades/day cap set but test_timestamps is None — cap will be INACTIVE")

    valid_bool = test_valid.astype(bool) if not isinstance(test_valid, np.ndarray) else test_valid.astype(bool)

    scores_work = scores.copy()
    scores_work[np.isnan(scores_work)] = -np.inf
    scores_work[~quality_mask] = -np.inf
    scores_work[~valid_bool] = -np.inf
    if test_cand_mask is not None:
        scores_work[~test_cand_mask.astype(bool)] = -np.inf

    finite_work = scores_work[np.isfinite(scores_work)]
    if len(finite_work) > 0:
        pct_above = float(np.mean(finite_work >= effective_threshold) * 100)
        log.info(f"[V5_FWD] Threshold check: {len(finite_work)} finite scores, "
                 f"{pct_above:.1f}% above threshold={effective_threshold:.4f} "
                 f"(score p90={float(np.percentile(finite_work, 90)):.4f} "
                 f"p99={float(np.percentile(finite_work, 99)):.4f} "
                 f"max={float(np.max(finite_work)):.4f})")
    else:
        log.warning("[V5_FWD] No finite scores after quality/validity masking")

    ddt = None
    ddt_base_threshold = effective_threshold
    if config.ddt_enable:
        from train.drawdown_throttle import DrawdownAdaptiveThrottle, DDTConfig
        ddt_cfg = DDTConfig(
            enabled=True,
            lookback_trades=config.ddt_lookback_trades,
            bad_rollr=config.ddt_bad_rollr,
            thr_k=config.ddt_thr_k,
            thr_min=config.ddt_thr_min,
            thr_max=config.ddt_thr_max,
            size_k=config.ddt_size_k,
            min_size_mult=config.ddt_min_size_mult,
            alpha_down=config.ddt_alpha_down,
            alpha_up=config.ddt_alpha_up,
            warmup_trades=config.ddt_warmup_trades,
        )
        ddt = DrawdownAdaptiveThrottle(ddt_cfg)
        log.info(f"[V5_DDT] Drawdown-Adaptive Throttle ENABLED: "
                 f"lookback={config.ddt_lookback_trades} bad_rollr={config.ddt_bad_rollr:.1f} "
                 f"thr_k={config.ddt_thr_k:.2f} thr_range=[{config.ddt_thr_min:.2f},{config.ddt_thr_max:.2f}] "
                 f"size_k={config.ddt_size_k:.2f} min_size={config.ddt_min_size_mult:.2f} "
                 f"alpha_down={config.ddt_alpha_down:.2f} alpha_up={config.ddt_alpha_up:.2f} "
                 f"warmup={config.ddt_warmup_trades}")
        ddt_base_threshold = config.ddt_thr_min
        log.info(f"[V5_DDT] Using relaxed pre-filter threshold={ddt_base_threshold:.4f} "
                 f"(DDT will dynamically adjust in loop)")

    if config.per_symbol_thresholds and test_sym_ids is not None:
        hard_floor = config.min_threshold if config.min_threshold is not None else 0.02
        per_bar_threshold = np.full(len(scores_work), ddt_base_threshold, dtype=np.float64)
        _first_val = next(iter(config.per_symbol_thresholds.values()), None)
        _is_per_side_format = isinstance(_first_val, dict)
        if _is_per_side_format and config.per_side_threshold:
            for sym_id_key, side_thrs in config.per_symbol_thresholds.items():
                sym_mask = test_sym_ids == int(sym_id_key)
                long_thr = side_thrs.get('long', ddt_base_threshold)
                short_thr = side_thrs.get('short', ddt_base_threshold)
                for thr_val, side_val in [(long_thr, 1), (short_thr, -1)]:
                    clamped = thr_val
                    if np.isfinite(clamped):
                        clamped = max(clamped, hard_floor)
                        if config.max_threshold is not None:
                            clamped = min(clamped, config.max_threshold)
                    side_bar_mask = sym_mask & (sides == side_val)
                    per_bar_threshold[side_bar_mask] = clamped
            n_inf_thr = int(np.sum(np.isinf(per_bar_threshold) & (per_bar_threshold > 0)))
            log.info(f"[V5_FWD] Per-symbol per-side thresholds active: "
                     f"{len(config.per_symbol_thresholds)} symbols configured, "
                     f"{n_inf_thr} bars have inf threshold (NO EDGE), "
                     f"floor={hard_floor:.4f} ceiling={config.max_threshold}")
        else:
            for sym_id_key, sym_thr in config.per_symbol_thresholds.items():
                clamped_thr = sym_thr if not isinstance(sym_thr, dict) else ddt_base_threshold
                if np.isfinite(clamped_thr):
                    clamped_thr = max(clamped_thr, hard_floor)
                    if config.max_threshold is not None:
                        clamped_thr = min(clamped_thr, config.max_threshold)
                sym_mask = test_sym_ids == int(sym_id_key)
                per_bar_threshold[sym_mask] = clamped_thr
            n_inf_thr = int(np.sum(np.isinf(per_bar_threshold) & (per_bar_threshold > 0)))
            log.info(f"[V5_FWD] Per-symbol thresholds active: "
                     f"{len(config.per_symbol_thresholds)} symbols configured, "
                     f"{n_inf_thr} bars have inf threshold (NO EDGE symbols), "
                     f"floor={hard_floor:.4f} ceiling={config.max_threshold}")
        selected = scores_work >= per_bar_threshold
    else:
        selected = scores_work >= ddt_base_threshold
    sel_indices = np.where(selected)[0]

    edge_first_blocked = 0
    edge_bar_values = None
    if config.edge_first:
        edge_L = arrays.get('edge_L', None)
        edge_S = arrays.get('edge_S', None)
        if edge_L is not None and edge_S is not None:
            edge_bar_values = np.maximum(edge_L, edge_S)
        else:
            mu_R = arrays['mu_R']
            risk = np.maximum(arrays.get('mae', arrays.get('sigma', np.ones_like(mu_R))), 0.25)
            edge_bar_values = np.abs(mu_R) / risk

        ef_pct_threshold = 0.0
        if config.edge_pct_floor > 0:
            ef_finite = edge_bar_values[np.isfinite(edge_bar_values)]
            if len(ef_finite) > 0:
                ef_pct_threshold = float(np.percentile(ef_finite, config.edge_pct_floor))
        ef_threshold = max(config.edge_min, ef_pct_threshold)

        ef_pass_mask = np.isfinite(edge_bar_values) & (edge_bar_values >= ef_threshold)
        n_before = len(sel_indices)
        sel_indices = sel_indices[ef_pass_mask[sel_indices]]
        edge_first_blocked = n_before - len(sel_indices)
        log.info(f"[V5_EDGE_FIRST] ENABLED: edge_min={config.edge_min:.4f} "
                 f"pct_floor=p{config.edge_pct_floor}={ef_pct_threshold:.4f} "
                 f"effective_threshold={ef_threshold:.4f} "
                 f"topn_per_day={config.edge_topn_per_day} "
                 f"passed={len(sel_indices)}/{n_before} blocked={edge_first_blocked}")

    regime_side_blocked = 0

    ema200 = None
    if ema200_regime_gate and close_prices is not None:
        if config.multi_regime and config.regime_side_map:
            log.info("[V5_GATE] EMA200 hard gate skipped — multi-regime active")
        else:
            ema200 = _compute_ema(close_prices, 200)
            log.info("[V5_FWD] EMA200 regime gate ENABLED")

    week_boundaries = None
    if config.weekly_loss_cap is not None and test_timestamps is not None:
        from datetime import timedelta
        trade_dates = np.array([datetime.utcfromtimestamp(ts / 1000) for ts in test_timestamps])
        week_ids = np.zeros(len(test_timestamps), dtype=np.int64)
        first_date = trade_dates[0]
        monday = first_date - timedelta(days=first_date.weekday())
        for i in range(len(trade_dates)):
            week_ids[i] = (trade_dates[i] - monday).days // 7
        week_boundaries = week_ids
        log.info(f"[V5_FWD] Weekly loss cap ENABLED: cap={config.weekly_loss_cap:.1f}R")

    if config.warmup_skip_bars > 0:
        log.info(f"[V5_FWD] Warmup skip ENABLED: first {config.warmup_skip_bars} bars blocked")

    position_sizer = None
    regime_scaler = None
    daily_tracker = None
    equity_stop = None
    size_multipliers = {}

    if config.adaptive_sizing:
        from train.v5_position_sizer import AdaptivePositionSizer, AdaptiveSizingConfig
        sizer_cfg = AdaptiveSizingConfig(
            enabled=True,
            kelly_fraction=config.kelly_fraction,
            max_size_mult=config.max_size_mult,
            min_size_mult=config.min_size_mult,
        )
        position_sizer = AdaptivePositionSizer(sizer_cfg)
        log.info(f"[V5_FWD] Adaptive sizing ENABLED: kelly_f={config.kelly_fraction} "
                 f"range=[{config.min_size_mult}, {config.max_size_mult}]")

    if config.regime_scaling:
        from train.v5_position_sizer import RegimeScaler, RegimeScalingConfig
        regime_cfg = RegimeScalingConfig(
            enabled=True,
            bull_mult=config.regime_bull_mult,
            bear_mult=config.regime_bear_mult,
            lookback_trades=config.regime_lookback,
        )
        regime_scaler = RegimeScaler(regime_cfg)
        log.info(f"[V5_FWD] Regime scaling ENABLED: bull={config.regime_bull_mult} "
                 f"bear={config.regime_bear_mult} lookback={config.regime_lookback}")

    if config.daily_loss_cap is not None or config.per_symbol_daily_r_budget is not None:
        from train.v5_position_sizer import DailyLossTracker, LossManagementConfig
        loss_cfg = LossManagementConfig(
            daily_loss_cap=config.daily_loss_cap,
            per_symbol_daily_r_budget=config.per_symbol_daily_r_budget,
        )
        daily_tracker = DailyLossTracker(loss_cfg)
        if config.daily_loss_cap is not None:
            log.info(f"[V5_FWD] Daily loss cap ENABLED: {config.daily_loss_cap}R")
        if config.per_symbol_daily_r_budget is not None:
            log.info(f"[V5_FWD] Per-symbol daily R budget ENABLED: {config.per_symbol_daily_r_budget}R")

    if config.per_symbol_r_kill is not None:
        log.info(f"[V5_FWD] Per-symbol cumulative R kill switch ENABLED: floor={config.per_symbol_r_kill}R")

    if config.soft_gate_floor and config.size_floor > 0:
        log.info(f"[V5_FWD] Soft gate floor ENABLED: soft gates clamped to min={config.size_floor}")
    if config.weekly_cap_dynamic:
        log.info(f"[V5_FWD] Dynamic weekly cap ENABLED: scale={config.weekly_cap_scale}")
    if config.quality_gate_enabled:
        if config.quality_gate_window < 1:
            log.warning("[V5_FWD] quality_gate_window < 1, clamping to 1")
            config.quality_gate_window = 1
        log.info(f"[V5_FWD] Rolling quality gate ENABLED: window={config.quality_gate_window} "
                 f"min_accuracy={config.quality_gate_min_accuracy} min_wr={config.quality_gate_min_wr}")
    if config.direction_balance_cap:
        log.info(f"[V5_FWD] Direction balance cap ENABLED: threshold={config.direction_balance_threshold} "
                 f"severe={config.direction_balance_severe}")

    if config.trailing_equity_stop is not None:
        from train.v5_position_sizer import TrailingEquityStop
        equity_stop = TrailingEquityStop(config.trailing_equity_stop)
        log.info(f"[V5_FWD] Trailing equity stop ENABLED: {config.trailing_equity_stop}R")

    conviction_sizer = None
    if config.conviction_sizing:
        from train.v5_position_sizer import ConvictionSizer, ConvictionSizingConfig
        conv_cfg = ConvictionSizingConfig(
            enabled=True,
            tier_top_pct=config.conviction_tier_top_pct,
            tier_top_mult=config.conviction_tier_top_mult,
            tier_high_pct=config.conviction_tier_high_pct,
            tier_high_mult=config.conviction_tier_high_mult,
            confidence_boost_threshold=config.conviction_confidence_threshold,
            confidence_boost_mult=config.conviction_confidence_boost,
        )
        conviction_sizer = ConvictionSizer(conv_cfg)
        log.info(f"[V5_FWD] Conviction sizing ENABLED: top{config.conviction_tier_top_pct}%→"
                 f"{config.conviction_tier_top_mult}x, high{config.conviction_tier_high_pct}%→"
                 f"{config.conviction_tier_high_mult}x, conf_thresh={config.conviction_confidence_threshold}")

    ultra_sizer = None
    if config.ultra_conviction:
        from train.v5_position_sizer import UltraConvictionSizer, UltraConvictionConfig
        ultra_cfg = UltraConvictionConfig(
            enabled=True,
            risk_cap=config.ultra_risk_cap,
            score_pct=config.ultra_score_pct,
            adx_min=config.ultra_adx_min,
            edge_min=config.ultra_edge_min,
            dd_max=config.ultra_dd_max,
            max_per_day=config.ultra_max_per_day,
            mult=config.ultra_mult,
        )
        ultra_sizer = UltraConvictionSizer(ultra_cfg)
        _usp_display = config.ultra_score_pct * 100 if config.ultra_score_pct <= 1.0 else config.ultra_score_pct
        log.info(f"[V5_FWD] Ultra-Conviction ENABLED: risk_cap={config.ultra_risk_cap:.2f} "
                 f"score_pct=p{_usp_display:.0f} adx_min={config.ultra_adx_min:.1f} "
                 f"edge_min={config.ultra_edge_min:.3f} dd_max={config.ultra_dd_max:.2f} "
                 f"max/day={config.ultra_max_per_day} mult={config.ultra_mult:.1f}")

    corr_tracker = None
    corr_blocker = None
    sym_id_to_name = {}
    if config.symbols_list is not None and test_sym_ids is not None:
        for si, s in enumerate(config.symbols_list):
            sym_id_to_name[si] = s
    if (config.corr_block and config.symbols_list is not None
            and len(config.symbols_list) >= 2 and test_sym_ids is not None):
        from train.v5_correlation import RollingDailyCorr, CorrBlocker, CorrConfig
        sym_names = list(config.symbols_list)
        for si, s in enumerate(sym_names):
            sym_id_to_name[si] = s
        corr_cfg = CorrConfig(
            enabled=True,
            window_days=config.corr_window_days,
            threshold=config.corr_thresh,
            same_side_only=config.corr_same_side_only,
            log_matrix=config.corr_log_matrix,
            max_block=getattr(config, 'corr_max_block', 5),
        )
        corr_tracker = RollingDailyCorr(sym_names, window_days=config.corr_window_days)
        corr_blocker = CorrBlocker(corr_tracker, corr_cfg)
        log.info(f"[V5_CORR] Correlation blocker ENABLED: thresh={config.corr_thresh:.2f} "
                 f"window={config.corr_window_days}d same_side={config.corr_same_side_only}")

    chronological_idx = sel_indices[np.argsort(sel_indices)]
    taken = []
    ema_blocked = 0
    warmup_blocked = 0
    weekly_blocked = 0
    corr_blocked = 0
    ddt_blocked = 0
    cooldown_blocked = 0
    head_disagree_blocked = 0
    current_week_r = 0.0
    current_week_id = -1
    week_killed = False
    weekly_r_history = []
    dynamic_weekly_cap = config.weekly_loss_cap
    last_bar = -config.cooldown - 1
    per_sym_last_bar = defaultdict(lambda: -config.cooldown - 1)
    daily_blocked = 0
    equity_blocked = 0
    tpd_blocked = 0
    tpd_current_date = ""
    tpd_current_count = 0
    edge_topn_blocked = 0
    ef_topn_current_date = {}
    ef_topn_current_count = defaultdict(int)

    gate_blocks = Counter()
    gate_blocked_r = defaultdict(list)
    post_ema200_indices = []
    post_regime_indices = []

    soft_gate_sizing = {}

    quality_gate_recent_correct = []
    quality_gate_recent_wins = []
    quality_gate_blocked = 0
    direction_balance_recent = []
    direction_balance_reductions = 0

    sym_cumulative_r = defaultdict(float)
    killed_symbols = set()
    per_sym_kill_blocked = 0

    open_positions: dict = {}
    trade_spans: dict = defaultdict(list)

    use_side_conditional_for_cap = True

    adx_values = None
    adx_blocked = 0
    if config.adx_gate and high_prices is not None and low_prices is not None and close_prices is not None:
        adx_values = _compute_adx(high_prices, low_prices, close_prices, period=config.adx_period)
        valid_adx = adx_values[~np.isnan(adx_values)]
        if len(valid_adx) > 0:
            log.info(f"[V5_FWD] ADX gate ENABLED: min={config.adx_min:.1f} period={config.adx_period} "
                     f"exception_top_pct={config.adx_exception_top_pct:.1f}% "
                     f"ADX stats: mean={np.mean(valid_adx):.1f} p25={np.percentile(valid_adx, 25):.1f} "
                     f"p50={np.percentile(valid_adx, 50):.1f} p75={np.percentile(valid_adx, 75):.1f}")
        else:
            log.warning("[V5_FWD] ADX gate enabled but no valid ADX values computed")
            adx_values = None
    elif config.adx_gate:
        log.warning("[V5_FWD] ADX gate enabled but high/low/close prices not available — gate INACTIVE")

    adx_exception_threshold = None
    if adx_values is not None and config.adx_exception_top_pct > 0:
        if train_ref_arrays is not None and '_train_scores' in train_ref_arrays:
            ref_scores_for_pct = train_ref_arrays['_train_scores']
            ref_scores_for_pct = ref_scores_for_pct[np.isfinite(ref_scores_for_pct)]
            log.info("[V5_FWD] ADX exception: using TRAINING-set score distribution (no lookahead)")
        else:
            ref_scores_for_pct = scores_work[np.isfinite(scores_work)]
        if len(ref_scores_for_pct) > 0:
            adx_exception_threshold = float(np.percentile(ref_scores_for_pct, 100 - config.adx_exception_top_pct))
            log.info(f"[V5_FWD] ADX exception: top {config.adx_exception_top_pct}% scores "
                     f"(threshold={adx_exception_threshold:.4f}) bypass ADX gate")

    ema200_for_regime = None
    atr_for_regime = None
    needs_ema_atr = (regime_scaler is not None or config.multi_regime
                     or config.ultra_conviction)
    if needs_ema_atr and close_prices is not None:
        ema200_for_regime = _compute_ema(close_prices, 200)
        diffs = np.abs(np.diff(close_prices, prepend=close_prices[0]))
        atr_period = 14
        atr_for_regime = np.full_like(diffs, np.nan)
        for i in range(atr_period, len(diffs)):
            atr_for_regime[i] = np.mean(diffs[i - atr_period:i])

    multi_regime_classifier = None
    atr_rolling_for_regime = None
    bar_regimes = None
    if config.multi_regime and close_prices is not None:
        from train.v5_position_sizer import MultiRegimeClassifier, MultiRegimeConfig
        mr_cfg = MultiRegimeConfig(
            enabled=True,
            adx_trending_threshold=config.regime_adx_trending,
            adx_choppy_threshold=config.regime_adx_choppy,
            atr_high_vol_ratio=config.regime_atr_high_vol,
            atr_low_vol_ratio=config.regime_atr_low_vol,
            atr_rolling_window=config.regime_atr_window,
            ema_slope_window=config.regime_ema_slope_window,
            ema_price_buffer=config.regime_ema_buffer,
        )
        multi_regime_classifier = MultiRegimeClassifier(mr_cfg)
        if atr_for_regime is not None:
            atr_rolling_for_regime = np.full_like(atr_for_regime, np.nan)
            window = config.regime_atr_window
            min_valid = max(window // 2, 20)
            for i in range(14, len(atr_for_regime)):
                lookback = min(i, window)
                valid_slice = atr_for_regime[max(0, i - lookback):i]
                valid_vals = valid_slice[~np.isnan(valid_slice)]
                if len(valid_vals) >= min_valid:
                    atr_rolling_for_regime[i] = float(np.mean(valid_vals))

        n_bars = len(close_prices)
        bar_regimes = np.array(["unknown"] * n_bars, dtype=object)
        slope_lookback = config.regime_ema_slope_window
        for i in range(n_bars):
            mr_adx = float(adx_values[i]) if adx_values is not None and i < len(adx_values) else float('nan')
            mr_atr_cur = float(atr_for_regime[i]) if atr_for_regime is not None and i < len(atr_for_regime) else float('nan')
            mr_atr_roll = float(atr_rolling_for_regime[i]) if atr_rolling_for_regime is not None and i < len(atr_rolling_for_regime) else float('nan')
            mr_ema = float(ema200_for_regime[i]) if ema200_for_regime is not None and i < len(ema200_for_regime) else float('nan')
            mr_ema_prev = float(ema200_for_regime[max(0, i - slope_lookback)]) if ema200_for_regime is not None else float('nan')
            bar_regimes[i] = multi_regime_classifier.classify(
                adx_val=mr_adx, atr_current=mr_atr_cur,
                atr_rolling=mr_atr_roll, close_price=float(close_prices[i]),
                ema200_val=mr_ema, ema200_prev=mr_ema_prev,
            )

        mrd = multi_regime_classifier.get_diagnostics()
        log.info(f"[V5_FWD] Multi-Regime Classifier ENABLED (sizing-only, not a gate): "
                 f"adx_trend={config.regime_adx_trending:.1f} "
                 f"adx_chop={config.regime_adx_choppy:.1f} "
                 f"atr_hi={config.regime_atr_high_vol:.2f} "
                 f"atr_lo={config.regime_atr_low_vol:.2f} "
                 f"atr_window={config.regime_atr_window}")
        log.info(f"[V5_REGIME] Pre-computed regime labels for {mrd['total_classified']}/{n_bars} bars: "
                 f"counts={mrd['regime_counts']} pct={mrd['regime_pct']}")

    def _oracle_r(i):
        """Get oracle R for bar i (side-conditional)."""
        if r_long is not None and r_short is not None:
            s = sides[i]
            v = float(r_long[i]) if s == 1 else float(r_short[i])
            return v if not np.isnan(v) else 0.0
        if test_realized_r is not None:
            v = float(test_realized_r[i])
            return v if not np.isnan(v) else 0.0
        return 0.0

    for idx in chronological_idx:
        if config.warmup_skip_bars > 0 and idx < config.warmup_skip_bars:
            warmup_blocked += 1
            gate_blocks["warmup"] += 1
            gate_blocked_r["warmup"].append(_oracle_r(idx))
            continue
        if getattr(config, 'per_symbol_cooldown', True) and test_sym_ids is not None:
            bar_sym_id = int(test_sym_ids[idx])
            if idx - per_sym_last_bar[bar_sym_id] < config.cooldown:
                cooldown_blocked += 1
                gate_blocks["cooldown"] += 1
                gate_blocked_r["cooldown"].append(_oracle_r(idx))
                continue
        elif idx - last_bar < config.cooldown:
            cooldown_blocked += 1
            gate_blocks["cooldown"] += 1
            gate_blocked_r["cooldown"].append(_oracle_r(idx))
            continue

        if config.per_symbol_r_kill is not None and test_sym_ids is not None:
            sym_name_kill = sym_id_to_name.get(int(test_sym_ids[idx]), None)
            if sym_name_kill and sym_name_kill in killed_symbols:
                if config.per_symbol_soft_kill:
                    cum_r = sym_cumulative_r[sym_name_kill]
                    kill_floor = config.per_symbol_r_kill
                    half_floor = kill_floor * 0.5
                    if cum_r <= kill_floor:
                        mult = 0.1
                    elif cum_r <= half_floor:
                        frac = (cum_r - kill_floor) / (half_floor - kill_floor)
                        mult = 0.1 + frac * 0.65
                    else:
                        frac = min(1.0, (cum_r - half_floor) / abs(half_floor)) if half_floor != 0 else 1.0
                        mult = 0.75 + frac * 0.25
                    soft_gate_sizing[idx] = soft_gate_sizing.get(idx, 1.0) * mult
                    gate_blocks["per_symbol_kill_soft"] += 1
                else:
                    per_sym_kill_blocked += 1
                    gate_blocks["per_symbol_kill"] += 1
                    gate_blocked_r["per_symbol_kill"].append(_oracle_r(idx))
                    continue

        if adx_values is not None:
            adx_val = adx_values[idx]
            if not np.isnan(adx_val):
                is_exception = (adx_exception_threshold is not None and scores_work[idx] >= adx_exception_threshold)
                if adx_val < config.adx_min and not is_exception:
                    adx_blocked += 1
                    gate_blocks["adx"] += 1
                    gate_blocked_r["adx"].append(_oracle_r(idx))
                    continue

        expired = [k for k, v in open_positions.items() if v['expiry'] <= idx]
        for k in expired:
            pos = open_positions[k]
            if corr_tracker is not None:
                entry_idx = pos['entry_bar']
                if use_side_conditional_for_cap:
                    tr = float(r_long[entry_idx]) if pos['side'] == 1 else float(r_short[entry_idx])
                else:
                    tr = float(test_realized_r[entry_idx]) if test_realized_r is not None else 0.0
                if not np.isnan(tr) and test_timestamps is not None:
                    date_str = datetime.utcfromtimestamp(
                        test_timestamps[entry_idx] / 1000).strftime('%Y-%m-%d')
                    corr_tracker.record_trade(pos['symbol'], date_str, tr)
            if corr_blocker is not None and pos.get('symbol'):
                corr_blocker.on_position_closed(pos['symbol'])
            del open_positions[k]

        if ema200 is not None:
            side_val = sides[idx]
            close_val = close_prices[idx]
            ema_val = ema200[idx]
            _ema200_against = (side_val == 1 and close_val < ema_val) or (side_val == -1 and close_val > ema_val)
            if _ema200_against:
                _side_str = "LONG" if side_val == 1 else "SHORT"
                if config.ema200_soft_mult is not None:
                    log.debug("[V5_GATE] ema200_soft side=%s close=%.2f ema200=%.2f mult=%.2f",
                              _side_str, close_val, ema_val, config.ema200_soft_mult)
                    soft_gate_sizing[idx] = soft_gate_sizing.get(idx, 1.0) * config.ema200_soft_mult
                    gate_blocks["ema200_soft"] += 1
                else:
                    log.debug("[V5_GATE] blocked_by=ema200 side=%s close=%.2f ema200=%.2f",
                              _side_str, close_val, ema_val)
                    ema_blocked += 1
                    gate_blocks["ema200"] += 1
                    gate_blocked_r["ema200"].append(_oracle_r(idx))
                    continue

        post_ema200_indices.append(idx)

        bar_regime = "unknown"
        if bar_regimes is not None and idx < len(bar_regimes):
            bar_regime = str(bar_regimes[idx])
        elif ema200_for_regime is not None and close_prices is not None and idx < len(close_prices):
            if close_prices[idx] > ema200_for_regime[idx] * 1.01:
                bar_regime = "trending_up"
            elif close_prices[idx] < ema200_for_regime[idx] * 0.99:
                bar_regime = "trending_down"
            else:
                bar_regime = "choppy"

        if config.regime_side_map is not None:
            allowed = config.regime_side_map.get(bar_regime, "BOTH")
            side_val = sides[idx]
            is_ultra_override = False
            if config.ultra_conviction and allowed == "NONE":
                is_ultra_override = True
            if not is_ultra_override:
                regime_disagrees = False
                if allowed == "NONE":
                    regime_disagrees = True
                elif allowed == "LONG" and side_val != 1:
                    regime_disagrees = True
                elif allowed == "SHORT" and side_val != -1:
                    regime_disagrees = True
                if regime_disagrees:
                    if config.regime_soft:
                        mult = config.regime_none_mult if allowed == "NONE" else config.regime_disagree_mult
                        soft_gate_sizing[idx] = soft_gate_sizing.get(idx, 1.0) * mult
                        gate_blocks["multi_regime_soft"] += 1
                    else:
                        regime_side_blocked += 1
                        gate_blocks["multi_regime"] += 1
                        gate_blocked_r["multi_regime"].append(_oracle_r(idx))
                        continue

        post_regime_indices.append(idx)

        if config.edge_first and config.edge_topn_per_day > 0 and test_timestamps is not None:
            bar_date = datetime.utcfromtimestamp(
                test_timestamps[idx] / 1000).strftime('%Y-%m-%d')
            topn_sym_id = int(test_sym_ids[idx]) if test_sym_ids is not None else 0
            if ef_topn_current_date.get(topn_sym_id) != bar_date:
                ef_topn_current_date[topn_sym_id] = bar_date
                ef_topn_current_count[topn_sym_id] = 0
            sym_count = ef_topn_current_count[topn_sym_id]
            if sym_count >= config.edge_topn_per_day:
                if config.edge_topn_soft:
                    excess = sym_count - config.edge_topn_per_day
                    decay = config.edge_topn_decay ** (excess + 1)
                    decay = max(decay, 0.15)
                    penalized_score = scores_work[idx] * decay
                    if penalized_score < effective_threshold:
                        edge_topn_blocked += 1
                        gate_blocks["edge_topn"] += 1
                        gate_blocked_r["edge_topn"].append(_oracle_r(idx))
                        continue
                    soft_gate_sizing[idx] = soft_gate_sizing.get(idx, 1.0) * decay
                    gate_blocks["edge_topn_soft"] += 1
                else:
                    edge_topn_blocked += 1
                    gate_blocks["edge_topn"] += 1
                    gate_blocked_r["edge_topn"].append(_oracle_r(idx))
                    continue

        if config.weekly_loss_cap is not None and week_boundaries is not None:
            wk = week_boundaries[idx]
            if wk != current_week_id:
                if current_week_id >= 0:
                    weekly_r_history.append(current_week_r)
                    if config.weekly_cap_dynamic and len(weekly_r_history) >= 4:
                        rolling_4w = sum(weekly_r_history[-4:]) / 4.0
                        if rolling_4w > 0:
                            dynamic_weekly_cap = config.weekly_loss_cap * config.weekly_cap_scale
                        else:
                            dynamic_weekly_cap = config.weekly_loss_cap * 0.5
                        log.debug("[V5_GATE] dynamic weekly_cap: base=%.1f effective=%.1f rolling_4w_avg=%.2f",
                                  config.weekly_loss_cap, dynamic_weekly_cap, rolling_4w)
                current_week_id = wk
                current_week_r = 0.0
                week_killed = False
            if week_killed:
                weekly_blocked += 1
                gate_blocks["weekly_cap"] += 1
                gate_blocked_r["weekly_cap"].append(_oracle_r(idx))
                continue

        if corr_blocker is not None and test_sym_ids is not None:
            sym_name = sym_id_to_name.get(int(test_sym_ids[idx]), None)
            side_val = int(sides[idx])
            open_by_sym = {v['symbol']: v['side'] for k, v in open_positions.items()
                           if v['symbol'] != sym_name}
            if sym_name and corr_blocker.should_block(sym_name, side_val, open_by_sym):
                corr_blocked += 1
                gate_blocks["correlation"] += 1
                gate_blocked_r["correlation"].append(_oracle_r(idx))
                continue

        if daily_tracker is not None and test_timestamps is not None:
            date_str = datetime.utcfromtimestamp(
                test_timestamps[idx] / 1000).strftime('%Y-%m-%d')
            daily_tracker.new_bar(date_str)
            trade_sym = None
            if test_sym_ids is not None and sym_id_to_name:
                trade_sym = sym_id_to_name.get(int(test_sym_ids[idx]), None)
            if daily_tracker.should_block(symbol=trade_sym):
                daily_blocked += 1
                gate_blocks["daily_loss"] += 1
                gate_blocked_r["daily_loss"].append(_oracle_r(idx))
                continue

        if equity_stop is not None and equity_stop.should_block():
            equity_blocked += 1
            gate_blocks["equity_stop"] += 1
            gate_blocked_r["equity_stop"].append(_oracle_r(idx))
            continue

        if config.max_trades_per_day is not None and test_timestamps is not None:
            bar_date = datetime.utcfromtimestamp(
                test_timestamps[idx] / 1000).strftime('%Y-%m-%d')
            if bar_date != tpd_current_date:
                tpd_current_date = bar_date
                tpd_current_count = 0
            if tpd_current_count >= config.max_trades_per_day:
                tpd_blocked += 1
                gate_blocks["max_tpd"] += 1
                gate_blocked_r["max_tpd"].append(_oracle_r(idx))
                continue

        if ddt is not None:
            ddt_thr = ddt.effective_threshold(effective_threshold)
            if scores_work[idx] < ddt_thr:
                ddt_blocked += 1
                gate_blocks["ddt"] += 1
                gate_blocked_r["ddt"].append(_oracle_r(idx))
                ddt.record_block()
                continue

        if config.head_disagreement_gate:
            disagreements = 0
            side_val = sides[idx]
            mu_val = float(arrays['mu_R'][idx])
            if side_val == 1 and mu_val < -0.01:
                disagreements += 1
            elif side_val == -1 and mu_val > 0.01:
                disagreements += 1
            if arrays.get('sigma') is not None:
                sigma_val = float(arrays['sigma'][idx])
                mae_val = float(arrays['mae'][idx])
                if sigma_val > mae_val * 2.0:
                    disagreements += 1
            p_trade_val = float(arrays['p_trade'][idx])
            if p_trade_val < 0.35:
                disagreements += 1
            if disagreements >= 2:
                head_disagree_blocked += 1
                gate_blocks["head_disagreement"] += 1
                gate_blocked_r["head_disagreement"].append(_oracle_r(idx))
                continue

        taken.append(idx)
        last_bar = idx
        if test_sym_ids is not None:
            per_sym_last_bar[int(test_sym_ids[idx])] = idx

        if config.max_trades_per_day is not None:
            tpd_current_count += 1

        if config.edge_first and config.edge_topn_per_day > 0:
            topn_inc_sym = int(test_sym_ids[idx]) if test_sym_ids is not None else 0
            ef_topn_current_count[topn_inc_sym] += 1

        soft_gate_mult_raw = soft_gate_sizing.get(idx, 1.0)
        if config.soft_gate_floor and config.size_floor > 0 and soft_gate_mult_raw < config.size_floor:
            soft_gate_mult = config.size_floor
        else:
            soft_gate_mult = soft_gate_mult_raw

        pos_sizer_mults = []
        if position_sizer is not None:
            p_win = float(arrays['p_long'][idx]) if sides[idx] == 1 else float(arrays['p_short'][idx])
            pos_sizer_mults.append(position_sizer.compute_size_multiplier(
                score=float(scores[idx]),
                p_win=p_win,
                mu_r=float(arrays['mu_R'][idx]),
                mfe=float(arrays['mfe'][idx]),
                mae=float(arrays['mae'][idx]),
            ))

        if regime_scaler is not None:
            regime_mult = regime_scaler.compute_regime_multiplier(
                idx=idx, side=int(sides[idx]),
                close_prices=close_prices,
                ema200=ema200_for_regime,
                atr_values=atr_for_regime,
            )
            pos_sizer_mults.append(regime_mult)

        if conviction_sizer is not None:
            p_dir = float(arrays['p_long'][idx]) if sides[idx] == 1 else float(arrays['p_short'][idx])
            conv_mult = conviction_sizer.compute_size_multiplier(
                score=float(scores[idx]),
                p_directional=p_dir,
                side=int(sides[idx]),
            )
            pos_sizer_mults.append(conv_mult)

        if pos_sizer_mults:
            trade_size_mult = soft_gate_mult * max(pos_sizer_mults)
        else:
            trade_size_mult = soft_gate_mult

        if ultra_sizer is not None:
            ultra_sizer.record_score(float(scores[idx]))
            trade_sym_ultra = ""
            if test_sym_ids is not None and sym_id_to_name:
                trade_sym_ultra = sym_id_to_name.get(int(test_sym_ids[idx]), "")
            ultra_date = ""
            if test_timestamps is not None:
                ultra_date = datetime.utcfromtimestamp(
                    test_timestamps[idx] / 1000).strftime('%Y-%m-%d')
            ultra_adx = float(adx_values[idx]) if adx_values is not None and idx < len(adx_values) else 0.0
            edge_l_val = float(arrays.get('edge_L', np.zeros(1))[min(idx, len(arrays.get('edge_L', np.zeros(1)))-1)]) if 'edge_L' in arrays else 0.0
            edge_s_val = float(arrays.get('edge_S', np.zeros(1))[min(idx, len(arrays.get('edge_S', np.zeros(1)))-1)]) if 'edge_S' in arrays else 0.0
            is_ultra = ultra_sizer.evaluate(
                symbol=trade_sym_ultra, side=int(sides[idx]),
                score=float(scores[idx]),
                edge_l=edge_l_val, edge_s=edge_s_val,
                adx_val=ultra_adx, regime=bar_regime,
                date_str=ultra_date, capital_blocked=False,
            )
            if is_ultra:
                atr_val_here = float(atr_for_regime[idx]) if atr_for_regime is not None and idx < len(atr_for_regime) else 0.0
                stop_dist_pct = (config.sl_mult * atr_val_here / float(close_prices[idx])) if close_prices is not None and atr_val_here > 0 else 0.01
                trade_size_mult = ultra_sizer.apply_ultra_sizing(trade_size_mult, stop_dist_pct)

        if ddt is not None:
            ddt_size = ddt.size_multiplier()
            trade_size_mult *= ddt_size

        if config.size_floor > 0 and trade_size_mult < config.size_floor:
            sf_allow = True
            if ddt is not None:
                ddt_diag_now = ddt.diagnostics()
                if ddt_diag_now['rolling_sum_r'] < 0:
                    sf_allow = False
                if ddt_diag_now['throttle_level'] > 0.5:
                    sf_allow = False
            if sf_allow:
                trade_size_mult = config.size_floor

        if config.quality_gate_enabled and len(quality_gate_recent_correct) >= config.quality_gate_window:
            rolling_acc = sum(quality_gate_recent_correct) / len(quality_gate_recent_correct)
            rolling_wr = sum(quality_gate_recent_wins) / len(quality_gate_recent_wins)
            if rolling_acc < config.quality_gate_severe_accuracy:
                quality_mult = 0.1
                trade_size_mult *= quality_mult
                quality_gate_blocked += 1
            elif rolling_acc < config.quality_gate_min_accuracy and rolling_wr < config.quality_gate_min_wr:
                quality_mult = 0.25
                trade_size_mult *= quality_mult
                quality_gate_blocked += 1

        if config.direction_balance_cap:
            direction_balance_recent.append(int(sides[idx]))
            if len(direction_balance_recent) > 100:
                direction_balance_recent.pop(0)
            if len(direction_balance_recent) >= 20:
                n_long = sum(1 for s in direction_balance_recent if s == 1)
                long_pct = n_long / len(direction_balance_recent)
                dominant_is_long = long_pct > 0.5
                dominant_pct = long_pct if dominant_is_long else (1.0 - long_pct)
                is_dominant_side = (dominant_is_long and sides[idx] == 1) or (not dominant_is_long and sides[idx] == -1)
                if is_dominant_side:
                    if dominant_pct >= config.direction_balance_severe:
                        trade_size_mult *= 0.25
                        direction_balance_reductions += 1
                    elif dominant_pct >= config.direction_balance_threshold:
                        trade_size_mult *= 0.5
                        direction_balance_reductions += 1

        size_multipliers[idx] = trade_size_mult

        if corr_tracker is not None and test_sym_ids is not None:
            sym_name = sym_id_to_name.get(int(test_sym_ids[idx]), None)
            if sym_name:
                pos_key = f"{sym_name}_{idx}"
                open_positions[pos_key] = {
                    'symbol': sym_name,
                    'side': int(sides[idx]),
                    'entry_bar': idx,
                    'expiry': idx + config.horizon,
                }
                trade_spans[sym_name].append((idx, idx + config.horizon))

        needs_post_r = (config.weekly_loss_cap is not None
                        or daily_tracker is not None
                        or equity_stop is not None
                        or regime_scaler is not None
                        or ultra_sizer is not None
                        or ddt is not None
                        or config.per_symbol_r_kill is not None
                        or config.quality_gate_enabled)
        if needs_post_r:
            if use_side_conditional_for_cap:
                post_trade_r = float(r_long[idx]) if sides[idx] == 1 else float(r_short[idx])
            else:
                post_trade_r = float(test_realized_r[idx]) if test_realized_r is not None else 0.0
            post_r_valid = not np.isnan(post_trade_r)

            if config.weekly_loss_cap is not None and week_boundaries is not None:
                if post_r_valid:
                    current_week_r += post_trade_r
                effective_wcap = dynamic_weekly_cap if config.weekly_cap_dynamic else config.weekly_loss_cap
                if current_week_r <= effective_wcap:
                    week_killed = True
                    log.info("[V5_GATE] weekly_cap hit: week=%d cumR=%.2f cap=%.2f",
                             current_week_id, current_week_r, effective_wcap)

            if daily_tracker is not None and post_r_valid:
                trade_sym = None
                if test_sym_ids is not None and sym_id_to_name:
                    trade_sym = sym_id_to_name.get(int(test_sym_ids[idx]), None)
                daily_tracker.record_trade(post_trade_r * size_multipliers.get(idx, 1.0), symbol=trade_sym)

            if equity_stop is not None and post_r_valid:
                equity_stop.update(post_trade_r * size_multipliers.get(idx, 1.0))

            if regime_scaler is not None and post_r_valid:
                regime_scaler.record_trade_result(post_trade_r)

            if ultra_sizer is not None and post_r_valid:
                ultra_sizer.update_equity(post_trade_r * size_multipliers.get(idx, 1.0))

            if ddt is not None and post_r_valid:
                sized_r = post_trade_r * size_multipliers.get(idx, 1.0)
                ddt.update_on_trade_close(sized_r)
                ddt.log_trade_close(
                    realized_r=sized_r,
                    thr_used=ddt.effective_threshold(effective_threshold),
                    size_mult=ddt.size_multiplier(),
                )

            if config.per_symbol_r_kill is not None and post_r_valid and test_sym_ids is not None:
                kill_sym_name = sym_id_to_name.get(int(test_sym_ids[idx]), None)
                if kill_sym_name:
                    sym_cumulative_r[kill_sym_name] += post_trade_r
                    if sym_cumulative_r[kill_sym_name] <= config.per_symbol_r_kill:
                        killed_symbols.add(kill_sym_name)
                        log.warning(f"[V5_GATE] Symbol {kill_sym_name} killed at cumR={sym_cumulative_r[kill_sym_name]:.2f}R (floor={config.per_symbol_r_kill}R)")

            if config.quality_gate_enabled and post_r_valid:
                side_val = int(sides[idx])
                raw_long_r = float(r_long[idx]) if r_long is not None else post_trade_r
                price_went_up = not np.isnan(raw_long_r) and raw_long_r > 0
                is_correct = (side_val == 1 and price_went_up) or (side_val == -1 and not price_went_up)
                is_win = post_trade_r > 0
                quality_gate_recent_correct.append(1.0 if is_correct else 0.0)
                quality_gate_recent_wins.append(1.0 if is_win else 0.0)
                if len(quality_gate_recent_correct) > config.quality_gate_window:
                    quality_gate_recent_correct.pop(0)
                    quality_gate_recent_wins.pop(0)

    if ema200 is not None:
        log.info(f"[V5_GATE] EMA200 blocked {ema_blocked} trades")
    if warmup_blocked > 0:
        log.info(f"[V5_GATE] Warmup blocked {warmup_blocked} trades (first {config.warmup_skip_bars} bars)")
    if weekly_blocked > 0:
        log.info(f"[V5_GATE] Weekly cap blocked {weekly_blocked} trades")
    if corr_blocked > 0:
        log.info(f"[V5_GATE] Correlation blocked {corr_blocked} trades")
    if daily_blocked > 0:
        log.info(f"[V5_GATE] Daily loss cap blocked {daily_blocked} trades")
    if equity_blocked > 0:
        log.info(f"[V5_GATE] Trailing equity stop blocked {equity_blocked} trades")
    if tpd_blocked > 0:
        log.info(f"[V5_GATE] Max trades/day cap blocked {tpd_blocked} trades")
    if adx_blocked > 0:
        log.info(f"[V5_GATE] ADX regime gate blocked {adx_blocked} trades (min={config.adx_min:.1f})")
    if edge_first_blocked > 0:
        log.info(f"[V5_GATE] Edge-first blocked {edge_first_blocked} candidates (pre-loop)")
    if regime_side_blocked > 0:
        log.info(f"[V5_GATE] Regime side map blocked {regime_side_blocked} trades")
    if edge_topn_blocked > 0:
        log.info(f"[V5_GATE] Edge top-N/day blocked {edge_topn_blocked} trades")
    if quality_gate_blocked > 0:
        final_acc = sum(quality_gate_recent_correct) / len(quality_gate_recent_correct) if quality_gate_recent_correct else 0
        final_wr = sum(quality_gate_recent_wins) / len(quality_gate_recent_wins) if quality_gate_recent_wins else 0
        log.info(f"[V5_QUALITY] Rolling quality gate reduced sizing on {quality_gate_blocked} trades "
                 f"(final rolling_acc={final_acc:.1%} rolling_wr={final_wr:.1%})")
    if direction_balance_reductions > 0:
        log.info(f"[V5_BALANCE] Direction balance cap reduced sizing on {direction_balance_reductions} trades")
    if config.weekly_cap_dynamic and weekly_r_history:
        log.info(f"[V5_GATE] Dynamic weekly cap: final_effective={dynamic_weekly_cap:.1f} "
                 f"base={config.weekly_loss_cap:.1f} weeks_tracked={len(weekly_r_history)}")
    if head_disagree_blocked > 0:
        log.info(f"[V5_GATE] Head disagreement blocked {head_disagree_blocked} trades")
    if per_sym_kill_blocked > 0:
        log.info(f"[V5_GATE] Per-symbol R kill blocked {per_sym_kill_blocked} trades "
                 f"(killed_symbols={sorted(killed_symbols)}, floor={config.per_symbol_r_kill}R)")
        for ks in sorted(killed_symbols):
            log.info(f"  {ks}: final cumR={sym_cumulative_r[ks]:.2f}R")
    if ddt_blocked > 0:
        log.info(f"[V5_DDT] Throttle blocked {ddt_blocked} trades")
    if ddt is not None:
        ddt_diag = ddt.diagnostics()
        log.info(f"[V5_DDT] Summary: throttle_now={ddt_diag['throttle_level']:.3f} "
                 f"rolling_sum_r={ddt_diag['rolling_sum_r']:.2f} "
                 f"total_trades={ddt_diag['total_trades_seen']} "
                 f"max_throttle_seen={ddt_diag['max_throttle_seen']:.3f}")
    if ultra_sizer is not None:
        ud = ultra_sizer.get_diagnostics()
        log.info(f"[V5_ULTRA] Summary: applied={ud['ultra_applied']} skipped={ud['ultra_skipped']} "
                 f"risk_cap={ud['risk_cap']:.2f} skip_reasons={ud['skip_reasons']}")
    n_total_bars = len(scores) if scores is not None else 0
    n_candidates = len(chronological_idx)
    n_taken = len(taken)
    log.info(f"[V5_DROPOFF] bars_total={n_total_bars} | candidates={n_candidates} | "
             f"warmup={warmup_blocked} cooldown={cooldown_blocked} adx={adx_blocked} "
             f"ema={ema_blocked} regime_side={regime_side_blocked} "
             f"edge_topn={edge_topn_blocked} weekly={weekly_blocked} corr={corr_blocked} "
             f"daily={daily_blocked} equity={equity_blocked} tpd={tpd_blocked} "
             f"head_disagree={head_disagree_blocked} per_sym_kill={per_sym_kill_blocked} "
             f"ddt={ddt_blocked} edge_first_pre={edge_first_blocked} → trades_taken={n_taken}")

    if gate_blocked_r:
        log.info("=" * 80)
        log.info("  GATE IMPACT ANALYSIS (oracle R of blocked trades)")
        log.info("=" * 80)
        log.info(f"  {'Gate':<25s} {'Blocked':>8s} {'E[R]':>10s} {'TotalR':>10s} {'WinR%':>8s}")
        log.info(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
        for gate_name, r_list in sorted(gate_blocked_r.items(), key=lambda x: -len(x[1])):
            if len(r_list) == 0:
                continue
            arr = np.array(r_list)
            mean_r = float(np.mean(arr))
            total_r = float(np.sum(arr))
            win_pct = float(np.mean(arr > 0) * 100)
            verdict = "DESTROYING VALUE" if mean_r > 0.02 else ("PROTECTING" if mean_r < -0.05 else "neutral")
            log.info(f"  {gate_name:<25s} {len(r_list):>8d} {mean_r:>+10.4f} {total_r:>+10.2f} {win_pct:>7.1f}%  ← {verdict}")
        log.info("=" * 80)

    soft_counts = {k: v for k, v in gate_blocks.items() if k.endswith("_soft")}
    if soft_counts:
        log.info("[V5_SOFT_GATES] Soft gate pass-throughs (reduced size, not blocked):")
        for name, count in sorted(soft_counts.items(), key=lambda x: -x[1]):
            log.info(f"  {name}: {count} trades passed with reduced size")

    if config.edge_first and edge_bar_values is not None and len(taken) > 0:
        taken_arr = np.array(taken)
        taken_edges = edge_bar_values[taken_arr]
        taken_finite = taken_edges[np.isfinite(taken_edges)]
        if len(taken_finite) > 0:
            log.info(f"[V5_EDGE_REPORT] Edge of taken trades: "
                     f"mean={np.mean(taken_finite):.4f} "
                     f"p50={np.percentile(taken_finite, 50):.4f} "
                     f"p75={np.percentile(taken_finite, 75):.4f} "
                     f"p90={np.percentile(taken_finite, 90):.4f}")
    if config.regime_side_map is not None:
        log.info(f"[V5_REGIME_SIDE] regime_side_map_enabled=1 "
                 f"skipped_by_side_map={regime_side_blocked}")

    n_eligible = len(sel_indices)
    n_hold_all = int(np.sum(sides[sel_indices] == 0)) if len(sel_indices) > 0 else 0
    n_long_all = int(np.sum(sides[sel_indices] == 1)) if len(sel_indices) > 0 else 0
    n_short_all = int(np.sum(sides[sel_indices] == -1)) if len(sel_indices) > 0 else 0
    unique_sides_eligible = set(np.unique(sides[sel_indices]).tolist()) if len(sel_indices) > 0 else set()
    log.info(f"[V5_SIDE_DIAG] eligible={n_eligible} "
             f"long={n_long_all} short={n_short_all} hold={n_hold_all} "
             f"unique_sides={unique_sides_eligible}")

    use_side_conditional = (r_long is not None and r_short is not None
                            and out_long is not None and out_short is not None)

    if not use_side_conditional:
        raise ValueError(
            "[V5_FWD] FATAL: r_long/r_short/out_long/out_short are REQUIRED. "
            "Oracle best-side fallback has been removed to prevent data leakage. "
            "Pass side-conditional arrays from generate_v5_sweep_outcomes()."
        )

    side_r = np.where(sides == 1, r_long, r_short).astype(float)
    side_out = np.where(sides == 1, out_long, out_short)
    _valid_outcomes = ["TP", "SL", "EXP_WIN", "EXP_LOSS", "TRAIL_WIN", "TRAIL_BE"]
    safe_outcomes = np.where(
        np.isin(side_out, _valid_outcomes),
        side_out, "NO_CANDIDATE"
    )
    safe_r = np.where(np.isnan(side_r), 0.0, side_r)
    log.info("[V5_FWD] Using side-conditional outcomes (predicted side selects LONG/SHORT R)")

    low_confidence = False
    if 0 < len(taken) < config.min_trades:
        log.warning(f"[V5_FWD] LOW CONFIDENCE: only {len(taken)} trades (minimum {config.min_trades}). "
                    f"Keeping trades but marking fold as low-confidence (threshold EMA will not blend).")
        low_confidence = True

    if len(taken) == 0:
        n_finite = int(np.sum(np.isfinite(scores_work))) if scores_work is not None else 0
        n_candidates = len(chronological_idx)
        n_high_bar_symbols = 0
        if config.per_symbol_thresholds:
            n_high_bar_symbols = sum(1 for v in config.per_symbol_thresholds.values()
                                     if not np.isfinite(v) or v > effective_threshold * 2)
        total_blocked = sum(gate_blocks.values())
        log.warning("[V5_FWD] No trades taken in forward test!")
        log.info("=" * 80)
        log.info("  DEAD FOLD DIAGNOSTICS — why 0 trades?")
        log.info("=" * 80)
        log.info(f"  Total test bars:          {len(scores) if scores is not None else 0}")
        log.info(f"  Finite scores (post QG):  {n_finite}")
        if n_finite > 0:
            finite_vals = scores_work[np.isfinite(scores_work)]
            log.info(f"  Score distribution:       p50={float(np.percentile(finite_vals, 50)):.4f} "
                     f"p90={float(np.percentile(finite_vals, 90)):.4f} "
                     f"max={float(np.max(finite_vals)):.4f}")
        log.info(f"  Effective threshold:      {effective_threshold:.4f}")
        log.info(f"  Candidates (above thr):   {n_candidates}")
        if config.per_symbol_thresholds:
            log.info(f"  Per-symbol thr active:    {len(config.per_symbol_thresholds)} symbols, "
                     f"{n_high_bar_symbols} have HIGH_BAR threshold")
        if total_blocked > 0:
            log.info(f"  Total gate blocks:        {total_blocked}")
            for gname, gcount in sorted(gate_blocks.items(), key=lambda x: -x[1]):
                if gcount > 0:
                    log.info(f"    {gname:<25s} {gcount:>6d} blocked")
        if n_finite == 0:
            log.info("  → CAUSE: No finite scores. Model produced no usable predictions for this window.")
        elif n_candidates == 0:
            log.info(f"  → CAUSE: Threshold too high ({effective_threshold:.4f}) — no scores passed it.")
        elif total_blocked >= n_candidates:
            log.info(f"  → CAUSE: All {n_candidates} candidates were blocked by gates.")
        log.info("=" * 80)
        from collections import OrderedDict
        stage_distributions_empty = OrderedDict()
        stage_distributions_empty['pre'] = _compute_side_distribution(sides, chronological_idx)
        stage_distributions_empty['post_ema200'] = _compute_side_distribution(
            sides, np.array(post_ema200_indices, dtype=np.intp) if post_ema200_indices else np.array([], dtype=np.intp))
        stage_distributions_empty['post_regime'] = _compute_side_distribution(
            sides, np.array(post_regime_indices, dtype=np.intp) if post_regime_indices else np.array([], dtype=np.intp))
        stage_distributions_empty['final'] = _compute_side_distribution(sides, np.array([], dtype=np.intp))
        _print_directional_balance_diagnostics(stage_distributions_empty, gate_blocks)
        report = _build_empty_report(test_start_date, test_end_date, config)
        report['directional_balance'] = {
            'stage_distributions': {k: dict(v) for k, v in stage_distributions_empty.items()},
            'gate_blocks': dict(gate_blocks),
        }
        _print_forward_report(report)
        return report

    taken = np.array(taken)
    t_outcomes = safe_outcomes[taken]
    t_r_unsized = safe_r[taken].copy()
    t_sides = sides[taken]

    t_size_mults = np.array([size_multipliers.get(idx, 1.0) for idx in taken])
    t_r = t_r_unsized * t_size_mults

    has_sizing = position_sizer is not None or regime_scaler is not None or conviction_sizer is not None or ultra_sizer is not None
    if has_sizing:
        log.info(f"[V5_SIZE] Applied sizing to {len(taken)} trades: "
                 f"unsized_totalR={np.sum(t_r_unsized):.2f} → sized_totalR={np.sum(t_r):.2f} "
                 f"avg_mult={np.mean(t_size_mults):.3f} "
                 f"min_mult={np.min(t_size_mults):.3f} max_mult={np.max(t_size_mults):.3f}")

    n_taken_long = int(np.sum(t_sides == 1))
    n_taken_short = int(np.sum(t_sides == -1))
    log.info(f"[V5_SIDE_DIAG] taken={len(taken)} long={n_taken_long} short={n_taken_short} "
             f"unique_sides_taken={set(np.unique(t_sides).tolist())}")

    if n_taken_short == 0 and len(taken) > 10:
        log.warning("WARNING: SHORT trades = 0 in forward test. Check side encoding or model bias.")
    if n_taken_long == 0 and len(taken) > 10:
        log.warning("WARNING: LONG trades = 0 in forward test. Check side encoding or model bias.")

    valid_trades = np.isin(t_outcomes, ["TP", "SL", "EXP_WIN", "EXP_LOSS", "TRAIL_WIN", "TRAIL_BE"])

    log.info(f"[V5_FWD] Selected {len(sel_indices)} bars above threshold, "
             f"{len(taken)} after cooldown, {valid_trades.sum()} with valid outcomes")

    t_r_valid = t_r[valid_trades]
    t_r_unsized_valid = t_r_unsized[valid_trades]
    t_outcomes_valid = t_outcomes[valid_trades]
    t_sides_valid = t_sides[valid_trades]
    taken_valid = taken[valid_trades]

    t_timestamps = None
    if test_timestamps is not None:
        t_timestamps = test_timestamps[taken_valid]

    report = _compute_forward_metrics(
        t_r_valid, t_outcomes_valid, t_sides_valid,
        test_bars, config, test_start_date, test_end_date,
        trade_timestamps=t_timestamps,
    )
    report['low_confidence'] = low_confidence

    side_quality = {}
    if len(taken_valid) > 0 and arrays is not None:
        t_scores_valid = scores[taken_valid]
        t_mu_r_valid = arrays['mu_R'][taken_valid]
        t_p_long_valid = arrays['p_long'][taken_valid]
        t_p_short_valid = arrays['p_short'][taken_valid]
        t_p_side_valid = np.where(t_sides_valid == 1, t_p_long_valid, t_p_short_valid)

        long_mask_v = t_sides_valid == 1
        short_mask_v = t_sides_valid == -1
        n_long_v = int(np.sum(long_mask_v))
        n_short_v = int(np.sum(short_mask_v))

        if n_long_v > 0:
            side_quality['long_avg_score'] = float(np.mean(t_scores_valid[long_mask_v]))
            side_quality['long_avg_p_side'] = float(np.mean(t_p_side_valid[long_mask_v]))
            side_quality['long_avg_mu_r'] = float(np.mean(t_mu_r_valid[long_mask_v]))
            side_quality['long_head_agree_pct'] = float(100 * np.mean(t_mu_r_valid[long_mask_v] > 0))

        if n_short_v > 0:
            side_quality['short_avg_score'] = float(np.mean(t_scores_valid[short_mask_v]))
            side_quality['short_avg_p_side'] = float(np.mean(t_p_side_valid[short_mask_v]))
            side_quality['short_avg_mu_r'] = float(np.mean(t_mu_r_valid[short_mask_v]))
            side_quality['short_head_agree_pct'] = float(100 * np.mean(t_mu_r_valid[short_mask_v] < 0))
            short_disagree = t_mu_r_valid[short_mask_v] > 0
            n_disagree = int(np.sum(short_disagree))
            side_quality['short_disagree_trades'] = n_disagree
            side_quality['short_disagree_pct'] = float(100 * n_disagree / max(n_short_v, 1))
            if n_disagree > 0:
                disagree_r = t_r_valid[short_mask_v][short_disagree]
                side_quality['short_disagree_expect'] = float(np.mean(disagree_r))

    report['side_quality'] = side_quality

    report['ddt_diagnostics'] = ddt.diagnostics() if ddt is not None else None
    report['ddt_blocked'] = ddt_blocked if ddt is not None else 0

    if test_sym_ids is not None and len(taken_valid) > 0:
        taken_sym_ids = test_sym_ids[taken_valid]
        sym_id_map = {i: s for i, s in enumerate(config.symbols_list)} if config.symbols_list else {}
        if not sym_id_map and sym_id_to_name:
            sym_id_map = sym_id_to_name
        per_sym_stats = {}
        for si_u in np.unique(taken_sym_ids):
            si_u = int(si_u)
            sym_name = sym_id_map.get(si_u, f"sym_{si_u}")
            sym_mask = taken_sym_ids == si_u
            sym_r = t_r_valid[sym_mask]
            sym_n = len(sym_r)
            sym_wr = float(np.sum(sym_r > 0)) / max(sym_n, 1)
            sym_total_r = float(np.sum(sym_r))
            sym_expect = float(np.mean(sym_r)) if sym_n > 0 else 0.0
            per_sym_stats[sym_name] = {
                'trades': sym_n, 'win_rate': round(sym_wr, 3),
                'expectancy_r': round(sym_expect, 4), 'total_r': round(sym_total_r, 4),
            }
        report['per_symbol_stats'] = per_sym_stats
        log.info("[V5_FWD] Per-symbol breakdown:")
        for sn, ss in per_sym_stats.items():
            log.info(f"  {sn}: {ss['trades']} trades | WR {ss['win_rate']:.1%} | "
                     f"E[R]={ss['expectancy_r']:+.4f} | Total={ss['total_r']:+.4f}R")

    if config.per_symbol_r_kill is not None:
        report['per_symbol_r_kill'] = {
            'floor': config.per_symbol_r_kill,
            'killed_symbols': sorted(killed_symbols),
            'blocked_trades': per_sym_kill_blocked,
            'cumulative_r': {k: round(v, 4) for k, v in sym_cumulative_r.items()},
        }

    if len(taken_valid) > 0:
        pred_quality = {}
        t_mu_R = arrays['mu_R'][taken_valid]
        t_actual_r = t_r_valid
        t_pred_mae = arrays['mae'][taken_valid]
        t_pred_mfe = arrays['mfe'][taken_valid]
        t_pred_sigma = arrays['sigma'][taken_valid] if arrays['sigma'] is not None else None
        t_p_long = arrays['p_long'][taken_valid]
        t_p_short = arrays['p_short'][taken_valid]
        t_pred_sides = t_sides_valid

        correct_side = ((t_pred_sides == 1) & (t_actual_r > 0)) | ((t_pred_sides == -1) & (t_actual_r < 0))
        action_accuracy = float(np.mean(correct_side))
        pred_quality['action_accuracy'] = round(action_accuracy, 4)

        finite_mask = np.isfinite(t_mu_R) & np.isfinite(t_actual_r)
        if np.sum(finite_mask) > 5:
            mu_r_corr = float(np.corrcoef(t_mu_R[finite_mask], t_actual_r[finite_mask])[0, 1])
            pred_quality['mu_r_correlation'] = round(mu_r_corr, 4)
        else:
            pred_quality['mu_r_correlation'] = None

        pred_quality['mean_predicted_mae'] = round(float(np.nanmean(t_pred_mae)), 4)
        pred_quality['mean_predicted_mfe'] = round(float(np.nanmean(t_pred_mfe)), 4)
        pred_quality['mean_predicted_mu_R'] = round(float(np.nanmean(t_mu_R)), 4)
        pred_quality['mean_actual_R'] = round(float(np.mean(t_actual_r)), 4)

        if t_pred_sigma is not None and np.sum(np.isfinite(t_pred_sigma)) > 0:
            sigma_finite = np.isfinite(t_pred_sigma) & np.isfinite(t_mu_R) & np.isfinite(t_actual_r)
            if np.sum(sigma_finite) > 5:
                residuals = np.abs(t_actual_r[sigma_finite] - t_mu_R[sigma_finite])
                within_1sigma = float(np.mean(residuals <= t_pred_sigma[sigma_finite]))
                within_2sigma = float(np.mean(residuals <= 2 * t_pred_sigma[sigma_finite]))
                pred_quality['sigma_1std_coverage'] = round(within_1sigma, 4)
                pred_quality['sigma_2std_coverage'] = round(within_2sigma, 4)
                pred_quality['mean_predicted_sigma'] = round(float(np.mean(t_pred_sigma[sigma_finite])), 4)

        p_side_for_taken = np.where(t_pred_sides == 1, t_p_long, t_p_short)
        pred_quality['mean_p_side'] = round(float(np.mean(p_side_for_taken)), 4)
        winners_mask = t_actual_r > 0
        losers_mask = t_actual_r < 0
        if np.sum(winners_mask) > 0:
            pred_quality['mean_p_side_winners'] = round(float(np.mean(p_side_for_taken[winners_mask])), 4)
        if np.sum(losers_mask) > 0:
            pred_quality['mean_p_side_losers'] = round(float(np.mean(p_side_for_taken[losers_mask])), 4)

        report['prediction_quality'] = pred_quality
        log.info(f"[V5_PRED_QUALITY] Action accuracy: {action_accuracy:.1%} "
                 f"| mu_R↔actual_R corr: {pred_quality.get('mu_r_correlation', 'N/A')} "
                 f"| mean mu_R: {pred_quality['mean_predicted_mu_R']:+.4f} vs actual: {pred_quality['mean_actual_R']:+.4f}")
        log.info(f"[V5_PRED_QUALITY] MAE pred: {pred_quality['mean_predicted_mae']:.4f} "
                 f"| MFE pred: {pred_quality['mean_predicted_mfe']:.4f} "
                 f"| mean p_side: {pred_quality['mean_p_side']:.4f}")
        if 'sigma_1std_coverage' in pred_quality:
            log.info(f"[V5_PRED_QUALITY] Sigma calibration: {pred_quality['sigma_1std_coverage']:.1%} within ±1σ "
                     f"(expect ~68%) | {pred_quality['sigma_2std_coverage']:.1%} within ±2σ (expect ~95%) "
                     f"| mean σ: {pred_quality['mean_predicted_sigma']:.4f}")
        if 'mean_p_side_winners' in pred_quality:
            log.info(f"[V5_PRED_QUALITY] Conviction: winners p_side={pred_quality.get('mean_p_side_winners', 'N/A'):.4f} "
                     f"vs losers p_side={pred_quality.get('mean_p_side_losers', 'N/A'):.4f}")

    if corr_tracker is not None:
        for k, pos in open_positions.items():
            entry_idx = pos['entry_bar']
            if use_side_conditional_for_cap:
                tr = float(r_long[entry_idx]) if pos['side'] == 1 else float(r_short[entry_idx])
            else:
                tr = float(test_realized_r[entry_idx]) if test_realized_r is not None else 0.0
            if not np.isnan(tr) and test_timestamps is not None:
                date_str = datetime.utcfromtimestamp(
                    test_timestamps[entry_idx] / 1000).strftime('%Y-%m-%d')
                corr_tracker.record_trade(pos['symbol'], date_str, tr)
        open_positions.clear()

        from train.v5_correlation import compute_overlap_ratio
        overlap = compute_overlap_ratio(dict(trade_spans), test_bars)
        report['_corr_tracker'] = corr_tracker
        report['_corr_blocker'] = corr_blocker
        report['_trade_spans'] = dict(trade_spans)
        report['_overlap_ratio'] = overlap
        report['corr_blocked_trades'] = corr_blocked

    if position_sizer or regime_scaler or daily_tracker or equity_stop or conviction_sizer or ultra_sizer:
        from train.v5_position_sizer import build_sizing_diagnostics
        sizing_diag = build_sizing_diagnostics(
            sizer=position_sizer, regime=regime_scaler,
            daily_tracker=daily_tracker, equity_stop=equity_stop,
            sized_r=t_r_valid if has_sizing else None,
            unsized_r=t_r_unsized_valid if has_sizing else None,
            conviction=conviction_sizer,
            ultra=ultra_sizer,
        )
        report['sizing_diagnostics'] = sizing_diag
        report['daily_blocked_trades'] = daily_blocked
        report['equity_blocked_trades'] = equity_blocked

        if has_sizing:
            log.info(f"[V5_SIZE] Sizing comparison: "
                     f"unsized={np.sum(t_r_unsized_valid):.2f}R → sized={np.sum(t_r_valid):.2f}R "
                     f"(impact: {np.sum(t_r_valid) - np.sum(t_r_unsized_valid):+.2f}R)")
        if daily_tracker:
            dt_diag = daily_tracker.get_diagnostics()
            log.info(f"[V5_GATE] Daily tracker: {dt_diag['days_killed']} days killed, "
                     f"{dt_diag['trades_blocked_daily_cap']} trades blocked (daily), "
                     f"{dt_diag['trades_blocked_symbol_cap']} trades blocked (per-symbol)")
        if equity_stop:
            es_diag = equity_stop.get_diagnostics()
            log.info(f"[V5_GATE] Equity stop: {es_diag['stop_triggers']} triggers, "
                     f"{es_diag['trades_blocked_equity_stop']} blocked, "
                     f"maxDD={es_diag['max_drawdown_r']:.2f}R")

    from collections import OrderedDict
    stage_distributions = OrderedDict()
    stage_distributions['pre'] = _compute_side_distribution(sides, chronological_idx)
    stage_distributions['post_ema200'] = _compute_side_distribution(
        sides, np.array(post_ema200_indices, dtype=np.intp) if post_ema200_indices else np.array([], dtype=np.intp))
    stage_distributions['post_regime'] = _compute_side_distribution(
        sides, np.array(post_regime_indices, dtype=np.intp) if post_regime_indices else np.array([], dtype=np.intp))
    stage_distributions['final'] = _compute_side_distribution(
        sides, taken if isinstance(taken, np.ndarray) else np.array(taken, dtype=np.intp) if len(taken) > 0 else np.array([], dtype=np.intp))

    _print_directional_balance_diagnostics(stage_distributions, gate_blocks)

    report['directional_balance'] = {
        'stage_distributions': {k: dict(v) for k, v in stage_distributions.items()},
        'gate_blocks': dict(gate_blocks),
    }

    _print_forward_report(report)
    return report


def _build_empty_report(test_start_date, test_end_date, config):
    return {
        'window_start': test_start_date,
        'window_end': test_end_date,
        'total_trades': 0,
        'trades_per_day': 0.0,
        'win_rate': 0.0,
        'expectancy_r': 0.0,
        'profit_factor': 0.0,
        'max_drawdown_r': 0.0,
        'avg_win_r': 0.0,
        'avg_loss_r': 0.0,
        'pct_tp': 0.0,
        'pct_sl': 0.0,
        'pct_exp': 0.0,
        'sharpe': 0.0,
        'total_r': 0.0,
        'score_threshold': config.score_threshold,
        'tp_mult': config.tp_mult,
        'sl_mult': config.sl_mult,
        'horizon': config.horizon,
        'cooldown': config.cooldown,
        'ddt_diagnostics': None,
        'ddt_blocked': 0,
        'low_confidence': False,
    }


def _compute_forward_metrics(t_r, t_outcomes, t_sides, test_bars, config, start_date, end_date,
                              trade_timestamps=None):
    n = len(t_r)
    val_days = test_bars / 96.0

    wins = t_r[t_r > 0]
    losses = t_r[t_r <= 0]

    winrate = len(wins) / max(n, 1)
    expect = float(np.mean(t_r)) if n > 0 else 0.0
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    total_win = float(np.sum(wins))
    total_loss = float(abs(np.sum(losses)))
    pf = min(total_win / max(total_loss, 1e-6), 999.99)

    if trade_timestamps is not None and n > 0:
        trade_days = np.array([datetime.utcfromtimestamp(ts / 1000).strftime('%Y-%m-%d')
                               for ts in trade_timestamps])
        unique_days = np.unique(trade_days)
        daily_pnl = np.array([float(np.sum(t_r[trade_days == d])) for d in unique_days])
        n_trading_days = len(unique_days)
        daily_mean = float(np.mean(daily_pnl))
        daily_std = float(np.std(daily_pnl, ddof=1)) if n_trading_days > 1 else 1.0
        sharpe = daily_mean / max(daily_std, 1e-6) * np.sqrt(252)
    else:
        std_r = float(np.std(t_r)) if n > 1 else 1.0
        trades_per_year = (n / max(val_days, 1e-6)) * 252
        sharpe = float(expect / max(std_r, 1e-6) * np.sqrt(max(trades_per_year, 1)))

    n_tp = int(np.sum(t_outcomes == "TP"))
    n_sl = int(np.sum(t_outcomes == "SL"))
    n_exp = int(np.sum(np.isin(t_outcomes, ["EXP_WIN", "EXP_LOSS"])))
    n_trail_win = int(np.sum(t_outcomes == "TRAIL_WIN"))
    n_trail_be = int(np.sum(t_outcomes == "TRAIL_BE"))

    equity_curve = np.cumsum(t_r)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve - running_max
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    tpd = n / max(val_days, 1e-6)

    long_mask = t_sides == 1
    short_mask = t_sides == -1
    n_long = int(np.sum(long_mask))
    n_short = int(np.sum(short_mask))

    long_r = t_r[long_mask]
    short_r = t_r[short_mask]

    direction_stats = {
        'long_trades': n_long,
        'long_win_rate': float(np.sum(long_r > 0) / max(n_long, 1)),
        'long_expectancy_r': float(np.mean(long_r)) if n_long > 0 else 0.0,
        'long_total_r': float(np.sum(long_r)),
        'short_trades': n_short,
        'short_win_rate': float(np.sum(short_r > 0) / max(n_short, 1)),
        'short_expectancy_r': float(np.mean(short_r)) if n_short > 0 else 0.0,
        'short_total_r': float(np.sum(short_r)),
    }

    weekly_stats = []
    if trade_timestamps is not None and n > 0:
        from datetime import timedelta
        trade_dates = np.array([datetime.utcfromtimestamp(ts / 1000) for ts in trade_timestamps])
        first_date = trade_dates.min()
        monday = first_date - timedelta(days=first_date.weekday())
        week_num = 0
        current_start = monday
        while current_start < trade_dates.max() + timedelta(days=1):
            current_end = current_start + timedelta(days=7)
            week_mask = (trade_dates >= current_start) & (trade_dates < current_end)
            w_r = t_r[week_mask]
            if len(w_r) > 0:
                week_num += 1
                week_label = current_start.strftime('%m/%d')
                weekly_stats.append({
                    'week': week_num,
                    'week_start': week_label,
                    'trades': len(w_r),
                    'expectancy': float(np.mean(w_r)),
                    'total_r': float(np.sum(w_r)),
                    'win_rate': float(np.sum(w_r > 0) / len(w_r)),
                })
            current_start = current_end
    elif n > 0:
        bars_per_week = 96 * 7
        n_weeks = max(1, int(np.ceil(test_bars / bars_per_week)))
        week_size = max(1, n // n_weeks) if n_weeks > 0 else n
        for w in range(n_weeks):
            w_start = w * week_size
            w_end = min((w + 1) * week_size, n)
            if w_start >= n:
                break
            w_r = t_r[w_start:w_end]
            weekly_stats.append({
                'week': w + 1,
                'trades': len(w_r),
                'expectancy': float(np.mean(w_r)) if len(w_r) > 0 else 0.0,
                'total_r': float(np.sum(w_r)),
            })

    sortino = 0.0
    if n > 1:
        downside_r = t_r[t_r < 0]
        downside_dev = float(np.sqrt(np.mean(downside_r ** 2))) if len(downside_r) > 0 else 1e-6
        if trade_timestamps is not None and n > 0 and n_trading_days > 1:
            sortino = daily_mean / max(
                float(np.sqrt(np.mean(np.minimum(daily_pnl, 0) ** 2))) if len(daily_pnl) > 0 else 1e-6,
                1e-6
            ) * np.sqrt(252)
        else:
            trades_per_year_s = (n / max(val_days, 1e-6)) * 252
            sortino = float(expect / max(downside_dev, 1e-6) * np.sqrt(max(trades_per_year_s, 1)))

    t_stat = 0.0
    p_value = 1.0
    if n > 2:
        from scipy import stats as sp_stats
        t_result = sp_stats.ttest_1samp(t_r, 0.0)
        t_stat = float(t_result.statistic) if np.isfinite(t_result.statistic) else 0.0
        p_value = float(t_result.pvalue) if np.isfinite(t_result.pvalue) else 1.0

    ci_lower = 0.0
    ci_upper = 0.0
    if n > 5:
        rng = np.random.RandomState(42)
        boot_means = np.array([
            float(np.mean(rng.choice(t_r, size=n, replace=True)))
            for _ in range(1000)
        ])
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))

    if sharpe > 5 and pf > 3 and winrate > 0.75:
        log.warning("[V5_FWD_SANITY] Metrics unusually high: Sharpe=%.1f PF=%.1f WR=%.1f%%. "
                    "Check for leakage or oracle-side contamination.",
                    sharpe, pf, winrate * 100)

    return {
        'window_start': start_date,
        'window_end': end_date,
        'total_trades': n,
        'trades_per_day': float(tpd),
        'n_long': n_long,
        'n_short': n_short,
        'win_rate': float(winrate),
        'expectancy_r': float(expect),
        'profit_factor': float(pf),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'ci_95_lower': float(ci_lower),
        'ci_95_upper': float(ci_upper),
        'max_drawdown_r': float(max_dd),
        'avg_win_r': float(avg_win),
        'avg_loss_r': float(avg_loss),
        'total_r': float(np.sum(t_r)),
        'pct_tp': float(n_tp / max(n, 1)),
        'pct_sl': float(n_sl / max(n, 1)),
        'pct_exp': float(n_exp / max(n, 1)),
        'n_trail_win': n_trail_win,
        'n_trail_be': n_trail_be,
        'pct_trail': float((n_trail_win + n_trail_be) / max(n, 1)),
        'score_threshold': config.score_threshold,
        'tp_mult': config.tp_mult,
        'sl_mult': config.sl_mult,
        'horizon': config.horizon,
        'cooldown': config.cooldown,
        'equity_final_r': float(equity_curve[-1]) if len(equity_curve) > 0 else 0.0,
        'direction_stats': direction_stats,
        'weekly_stats': weekly_stats,
    }


def _print_forward_report(report):
    log.info("")
    log.info("=" * 80)
    log.info("  FORWARD TEST REPORT")
    log.info("=" * 80)
    log.info(f"  Window:         {report['window_start']} → {report['window_end']}")
    log.info(f"  Threshold:      {report['score_threshold']:.4f}")
    log.info(f"  TP/SL/Horizon:  {report['tp_mult']:.1f}x / {report['sl_mult']:.1f}x ATR / {report['horizon']} bars")
    log.info(f"  Cooldown:       {report['cooldown']} bars")
    log.info("-" * 80)
    log.info(f"  Total Trades:   {report['total_trades']}")
    log.info(f"  Trades/Day:     {report['trades_per_day']:.2f}")
    log.info(f"  Long/Short:     {report.get('n_long', 0)}/{report.get('n_short', 0)}")
    log.info(f"  Win Rate:       {report['win_rate']:.1%}")
    log.info(f"  Expectancy:     {report['expectancy_r']:+.4f} R")
    log.info(f"  Profit Factor:  {report['profit_factor']:.2f}")
    log.info(f"  Sharpe (daily): {report['sharpe']:.2f}")
    if 'sortino' in report:
        log.info(f"  Sortino:        {report['sortino']:.2f}")
    if 't_stat' in report:
        sig_marker = "***" if report.get('p_value', 1) < 0.01 else "**" if report.get('p_value', 1) < 0.05 else "*" if report.get('p_value', 1) < 0.10 else "ns"
        log.info(f"  t-stat:         {report['t_stat']:.3f}  p={report['p_value']:.4f} [{sig_marker}]")
    if 'ci_95_lower' in report:
        log.info(f"  95%% CI (E[R]):  [{report['ci_95_lower']:+.4f}, {report['ci_95_upper']:+.4f}]")
    log.info(f"  Max Drawdown:   {report['max_drawdown_r']:.4f} R")
    log.info(f"  Avg Win R:      {report['avg_win_r']:+.4f}")
    log.info(f"  Avg Loss R:     {report['avg_loss_r']:+.4f}")
    log.info(f"  Total R:        {report['total_r']:+.4f}")
    pct_trail = report.get('pct_trail', 0)
    if pct_trail > 0:
        log.info(f"  %%TP/%%SL/%%EX/%%TR: {report['pct_tp']:.0%} / {report['pct_sl']:.0%} / {report['pct_exp']:.0%} / {pct_trail:.0%}")
        log.info(f"  Trail Wins/BE:  {report.get('n_trail_win', 0)} / {report.get('n_trail_be', 0)}")
    else:
        log.info(f"  %%TP/%%SL/%%EX:    {report['pct_tp']:.0%} / {report['pct_sl']:.0%} / {report['pct_exp']:.0%}")
    log.info(f"  Equity Final:   {report.get('equity_final_r', 0):+.4f} R")
    ddt_diag = report.get('ddt_diagnostics')
    if ddt_diag is not None:
        log.info(f"  DDT Blocked:    {report.get('ddt_blocked', 0)} trades")
        log.info(f"  DDT Throttle:   {ddt_diag['throttle_level']:.3f} "
                 f"(peak={ddt_diag['max_throttle_seen']:.3f})")
        log.info(f"  DDT Roll R:     {ddt_diag['rolling_sum_r']:+.2f} "
                 f"(lookback={ddt_diag['lookback_trades']})")
    log.info("-" * 80)
    ds = report.get('direction_stats', {})
    if ds:
        log.info("  Direction Breakdown:")
        log.info(f"    LONG:  {ds.get('long_trades',0)} trades | WR {ds.get('long_win_rate',0):.1%} | "
                 f"Expect {ds.get('long_expectancy_r',0):+.4f} R | Total {ds.get('long_total_r',0):+.4f} R")
        log.info(f"    SHORT: {ds.get('short_trades',0)} trades | WR {ds.get('short_win_rate',0):.1%} | "
                 f"Expect {ds.get('short_expectancy_r',0):+.4f} R | Total {ds.get('short_total_r',0):+.4f} R")
    sq = report.get('side_quality', {})
    if sq:
        log.info("-" * 80)
        log.info("  Side Quality Diagnostics:")
        log.info(f"    LONG  avg_score={sq.get('long_avg_score',0):+.4f}  "
                 f"avg_p_side={sq.get('long_avg_p_side',0):.4f}  "
                 f"avg_mu_R={sq.get('long_avg_mu_r',0):+.4f}  "
                 f"head_agree={sq.get('long_head_agree_pct',0):.0f}%")
        log.info(f"    SHORT avg_score={sq.get('short_avg_score',0):+.4f}  "
                 f"avg_p_side={sq.get('short_avg_p_side',0):.4f}  "
                 f"avg_mu_R={sq.get('short_avg_mu_r',0):+.4f}  "
                 f"head_agree={sq.get('short_head_agree_pct',0):.0f}%")
        if sq.get('short_disagree_trades', 0) > 0:
            log.info(f"    SHORT head-disagree trades (mu_R>0): {sq['short_disagree_trades']} "
                     f"({sq.get('short_disagree_pct',0):.0f}%) → "
                     f"expect={sq.get('short_disagree_expect',0):+.4f} R")
    log.info("-" * 80)
    if report.get('weekly_stats'):
        log.info("  Weekly Breakdown:")
        has_week_start = 'week_start' in report['weekly_stats'][0]
        if has_week_start:
            log.info(f"  {'Week':>6} {'Start':>8} {'Trades':>8} {'WR':>7} {'Expect':>10} {'Total R':>10}")
            for ws in report['weekly_stats']:
                log.info(f"  {ws['week']:>6} {ws.get('week_start',''):>8} {ws['trades']:>8} "
                         f"{ws.get('win_rate',0):>6.1%} {ws['expectancy']:>+10.4f} {ws['total_r']:>+10.4f}")
        else:
            log.info(f"  {'Week':>6} {'Trades':>8} {'Expect':>10} {'Total R':>10}")
            for ws in report['weekly_stats']:
                log.info(f"  {ws['week']:>6} {ws['trades']:>8} {ws['expectancy']:>+10.4f} {ws['total_r']:>+10.4f}")
    log.info("=" * 80)


def run_v5_walk_forward(
    data_dir, device, symbols, epochs, batch_size, lr,
    train_months=12, test_months=1,
    horizon=16, tp_mult=2.0, sl_mult=1.5,
    score_lambda=0.5, risk_proxy='mae',
    quality_gate_cfg=None, tpd_ctrl_cfg=None,
    candidate_config=None, risk_controls=None,
    hold_target=0.30, mfe_min=0.05,
    w_ret=1.0, w_mfe=0.25, w_mae=0.25, w_action=2.0,
    w_barrier=0.25, w_regime=0.1,
    warmup_epochs=5, min_lr=None,
    barrier_mode='fixed', barrier_presets=None,
    use_regime_head=False, cand_warmup_epochs=3,
    ema200_regime_gate=False, weekly_loss_cap=None, warmup_skip_bars=0,
    corr_block=False, corr_window_days=30, corr_thresh=0.70,
    corr_same_side_only=True, corr_log_matrix=True, corr_max_block=5,
    adaptive_sizing=False, kelly_fraction=0.25, max_size_mult=2.5, min_size_mult=0.25,
    regime_scaling=False, regime_bull_mult=1.5, regime_bear_mult=0.5, regime_lookback=20,
    daily_loss_cap=None, trailing_equity_stop=None, per_symbol_daily_r_budget=None,
    min_threshold=None, max_threshold=None, min_threshold_pct=None, max_trades_per_day=None,
    trailing_sl=False, trail_activation=1.5, trail_distance=1.0, allow_runner=False,
    conviction_sizing=False, conviction_tier_top_pct=5.0, conviction_tier_top_mult=2.5,
    conviction_tier_high_pct=20.0, conviction_tier_high_mult=1.5,
    conviction_confidence_threshold=0.65, conviction_confidence_boost=1.3,
    adx_gate=False, adx_period=14, adx_min=18.0, adx_exception_top_pct=10.0,
    temp_scale=False, promote_metric='expectancy', stage_a_epochs=0,
    balanced_sampling=True, balanced_sampling_mode='cap', per_symbol_scaler=False, symbol_embed_dim=8,
    ultra_conviction=False, ultra_risk_cap=0.05, ultra_score_pct=0.95,
    ultra_adx_min=25.0, ultra_edge_min=0.03, ultra_dd_max=0.10,
    ultra_max_per_day=1, ultra_mult=3.0,
    ddt_enable=False, ddt_lookback_trades=60, ddt_bad_rollr=6.0,
    ddt_thr_k=0.60, ddt_thr_min=0.08, ddt_thr_max=0.25,
    ddt_size_k=0.70, ddt_min_size_mult=0.25,
    ddt_alpha_down=0.30, ddt_alpha_up=0.05, ddt_warmup_trades=20,
    multi_regime=False, regime_adx_trending=25.0, regime_adx_choppy=20.0,
    regime_atr_high_vol=1.3, regime_atr_low_vol=0.7,
    regime_atr_window=96, regime_ema_slope_window=10, regime_ema_buffer=0.005,
    edge_first=False, edge_min=0.03, edge_pct_floor=70, edge_topn_per_day=4,
    regime_side_map=None,
    regime_soft=True, regime_disagree_mult=0.3, regime_none_mult=0.2,
    per_symbol_soft_kill=True,
    edge_topn_soft=True, edge_topn_decay=0.7,
    size_floor=0.0,
    soft_gate_floor=True,
    weekly_cap_dynamic=False, weekly_cap_scale=2.0,
    quality_gate_enabled=False, quality_gate_window=50,
    direction_balance_cap=False, direction_balance_threshold=0.75,
    recency_weight=False, recency_half_life=90,
    finetune_months=0, finetune_epochs=5, finetune_lr_mult=0.1,
    warm_start=False, warm_start_lr_mult=0.3,
    head_disagreement_gate=False, slippage_base_bps=0.0,
    sigma_discount=False, min_p_side=0.0, min_p_short=0.0,
    side_aware_scoring=False, mae_asym_weight=1.0,
    per_symbol_cooldown=True,
    cooldown=4,
    min_trades=20,
    wf_threshold_ema=True, wf_threshold_ema_alpha=0.5,
    wf_threshold_decay=0.5,
    mu_debias=True, mu_debias_alpha=0.01,
    per_symbol_r_kill=None,
    per_symbol_threshold=False,
    short_oversample=False,
    short_min_fraction=0.35,
    ema200_soft_mult=None,
    per_side_threshold=False,
    replit_url=None,
    model_version='v5',
    v6_seq_len=16,
    v6_conv_channels=128,
    v6_n_conv_layers=3,
    v6_attn_heads=4,
    v6_attn_layers=2,
    v6_n_experts=4,
    v6_expert_top_k=2,
    v6_feature_mask_ratio=0.15,
    v6_aux_weight=0.1,
    v6_confidence_weight=0.15,
    v6_moe_balance_weight=0.05,
):
    """Walk-forward analysis: rolling train/test windows."""
    try:
        from dateutil.relativedelta import relativedelta
    except ImportError:
        log.error("[V5_WF] python-dateutil not installed. Install with: pip install python-dateutil")
        return

    from train.training_push import TrainingProgressPusher
    pusher = TrainingProgressPusher(replit_url=replit_url)

    first_ts = None
    last_ts = None
    for sym in symbols:
        parquet_path = data_dir / f"{sym}_15m.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            sym_first = df['timestamp'].min()
            sym_last = df['timestamp'].max()
            if first_ts is None or sym_first < first_ts:
                first_ts = sym_first
            if last_ts is None or sym_last > last_ts:
                last_ts = sym_last

    if first_ts is None:
        log.error("[V5_WF] No data found for any symbol")
        return

    data_start = datetime.utcfromtimestamp(first_ts / 1000)
    data_end = datetime.utcfromtimestamp(last_ts / 1000)
    log.info(f"[V5_WF] Data range: {data_start.strftime('%Y-%m-%d')} → {data_end.strftime('%Y-%m-%d')}")

    first_test_start = data_start + relativedelta(months=train_months)
    if first_test_start >= data_end:
        log.error(f"[V5_WF] Not enough data for {train_months}m train + {test_months}m test")
        return

    folds = []
    fold_num = 0
    current_test_start = first_test_start

    while current_test_start < data_end:
        fold_num += 1
        train_end = current_test_start
        test_end = current_test_start + relativedelta(months=test_months)
        if test_end > data_end:
            test_end = data_end

        train_start = train_end - relativedelta(months=train_months)
        if train_start < data_start:
            train_start = data_start

        folds.append({
            'fold': fold_num,
            'train_start': train_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'test_start': current_test_start.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
        })
        current_test_start = test_end

    log.info(f"[V5_WF] Generated {len(folds)} folds (train={train_months}m, test={test_months}m)")
    for f in folds:
        log.info(f"  Fold {f['fold']}: train {f['train_start']}→{f['train_end']} | test {f['test_start']}→{f['test_end']}")

    try:
        import torch
        try:
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CPU"
        except Exception:
            gpu_name = "CPU (CUDA unavailable)"
    except Exception:
        gpu_name = "Unknown"

    pusher.session_start(
        session_type="walk_forward",
        total_folds=len(folds),
        total_epochs=epochs,
        symbols=symbols,
        config={
            "lr": lr, "batch_size": batch_size, "epochs": epochs,
            "horizon": horizon, "tp_mult": tp_mult, "sl_mult": sl_mult,
            "score_lambda": score_lambda, "risk_proxy": risk_proxy,
            "balanced_sampling": balanced_sampling, "per_symbol_scaler": per_symbol_scaler,
        },
        gpu_name=gpu_name,
        train_months=train_months,
        test_months=test_months,
    )

    all_reports = []
    data_path = data_dir / f"{symbols[0]}_15m.parquet"
    threshold_ema = None
    blended_threshold = None
    wf_threshold_decay = max(0.01, min(1.0, wf_threshold_decay))
    prev_fold_state_dict = None

    for fold in folds:
        log.info(f"\n{'='*80}")
        log.info(f"  WALK-FORWARD FOLD {fold['fold']}/{len(folds)}")
        log.info(f"{'='*80}")

        pusher.fold_start(
            fold_num=fold['fold'],
            train_start=fold['train_start'],
            train_end=fold['train_end'],
            test_start=fold['test_start'],
            test_end=fold['test_end'],
        )

        global _active_pusher, _active_fold_num, _active_total_folds
        _active_pusher = pusher
        _active_fold_num = fold['fold']
        _active_total_folds = len(folds)

        train_v5_model(
            data_path, device, epochs, batch_size, lr,
            warmup_epochs=warmup_epochs, min_lr=min_lr,
            tp_mult=tp_mult, sl_mult=sl_mult, horizon=horizon,
            symbols=symbols,
            w_ret=w_ret, w_mfe=w_mfe, w_mae=w_mae, w_action=w_action,
            w_barrier=w_barrier, w_regime=w_regime,
            score_lambda=score_lambda, risk_proxy=risk_proxy,
            hold_target=hold_target, mfe_min=mfe_min,
            barrier_mode=barrier_mode, barrier_presets=barrier_presets,
            use_regime_head=use_regime_head,
            candidate_config=candidate_config, risk_controls=risk_controls,
            cand_warmup_epochs=cand_warmup_epochs,
            quality_gate_cfg=quality_gate_cfg, tpd_ctrl_cfg=tpd_ctrl_cfg,
            train_end_date=fold['train_end'],
            test_start_date=fold['test_start'],
            test_end_date=fold['test_end'],
            run_forward_test=True,
            freeze_decision=True,
            ema200_regime_gate=ema200_regime_gate,
            weekly_loss_cap=weekly_loss_cap,
            warmup_skip_bars=warmup_skip_bars,
            corr_block=corr_block,
            corr_window_days=corr_window_days,
            corr_thresh=corr_thresh,
            corr_same_side_only=corr_same_side_only,
            corr_log_matrix=corr_log_matrix,
            corr_max_block=corr_max_block,
            adaptive_sizing=adaptive_sizing,
            kelly_fraction=kelly_fraction,
            max_size_mult=max_size_mult,
            min_size_mult=min_size_mult,
            regime_scaling=regime_scaling,
            regime_bull_mult=regime_bull_mult,
            regime_bear_mult=regime_bear_mult,
            regime_lookback=regime_lookback,
            daily_loss_cap=daily_loss_cap,
            trailing_equity_stop=trailing_equity_stop,
            per_symbol_daily_r_budget=per_symbol_daily_r_budget,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            min_threshold_pct=min_threshold_pct,
            max_trades_per_day=max_trades_per_day,
            trailing_sl=trailing_sl,
            trail_activation=trail_activation,
            trail_distance=trail_distance,
            allow_runner=allow_runner,
            conviction_sizing=conviction_sizing,
            conviction_tier_top_pct=conviction_tier_top_pct,
            conviction_tier_top_mult=conviction_tier_top_mult,
            conviction_tier_high_pct=conviction_tier_high_pct,
            conviction_tier_high_mult=conviction_tier_high_mult,
            conviction_confidence_threshold=conviction_confidence_threshold,
            conviction_confidence_boost=conviction_confidence_boost,
            adx_gate=adx_gate,
            adx_period=adx_period,
            adx_min=adx_min,
            adx_exception_top_pct=adx_exception_top_pct,
            temp_scale=temp_scale,
            promote_metric=promote_metric,
            stage_a_epochs=stage_a_epochs,
            balanced_sampling=balanced_sampling,
            balanced_sampling_mode=balanced_sampling_mode,
            symbol_embed_dim=symbol_embed_dim,
            per_symbol_scaler=per_symbol_scaler,
            ultra_conviction=ultra_conviction,
            ultra_risk_cap=ultra_risk_cap,
            ultra_score_pct=ultra_score_pct,
            ultra_adx_min=ultra_adx_min,
            ultra_edge_min=ultra_edge_min,
            ultra_dd_max=ultra_dd_max,
            ultra_max_per_day=ultra_max_per_day,
            ultra_mult=ultra_mult,
            ddt_enable=ddt_enable,
            ddt_lookback_trades=ddt_lookback_trades,
            ddt_bad_rollr=ddt_bad_rollr,
            ddt_thr_k=ddt_thr_k,
            ddt_thr_min=ddt_thr_min,
            ddt_thr_max=ddt_thr_max,
            ddt_size_k=ddt_size_k,
            ddt_min_size_mult=ddt_min_size_mult,
            ddt_alpha_down=ddt_alpha_down,
            ddt_alpha_up=ddt_alpha_up,
            ddt_warmup_trades=ddt_warmup_trades,
            multi_regime=multi_regime,
            regime_adx_trending=regime_adx_trending,
            regime_adx_choppy=regime_adx_choppy,
            regime_atr_high_vol=regime_atr_high_vol,
            regime_atr_low_vol=regime_atr_low_vol,
            regime_atr_window=regime_atr_window,
            regime_ema_slope_window=regime_ema_slope_window,
            regime_ema_buffer=regime_ema_buffer,
            edge_first=edge_first,
            edge_min=edge_min,
            edge_pct_floor=edge_pct_floor,
            edge_topn_per_day=edge_topn_per_day,
            regime_side_map=regime_side_map,
            regime_soft=regime_soft,
            regime_disagree_mult=regime_disagree_mult,
            regime_none_mult=regime_none_mult,
            per_symbol_soft_kill=per_symbol_soft_kill,
            edge_topn_soft=edge_topn_soft,
            edge_topn_decay=edge_topn_decay,
            size_floor=size_floor,
            soft_gate_floor=soft_gate_floor,
            weekly_cap_dynamic=weekly_cap_dynamic,
            weekly_cap_scale=weekly_cap_scale,
            quality_gate_enabled=quality_gate_enabled,
            quality_gate_window=quality_gate_window,
            direction_balance_cap=direction_balance_cap,
            direction_balance_threshold=direction_balance_threshold,
            recency_weight=recency_weight,
            recency_half_life=recency_half_life,
            finetune_months=finetune_months,
            finetune_epochs=finetune_epochs,
            finetune_lr_mult=finetune_lr_mult,
            warm_start_state_dict=prev_fold_state_dict if warm_start else None,
            head_disagreement_gate=head_disagreement_gate,
            slippage_base_bps=slippage_base_bps,
            sigma_discount=sigma_discount,
            min_p_side=min_p_side,
            min_p_short=min_p_short,
            side_aware_scoring=side_aware_scoring,
            mae_asym_weight=mae_asym_weight,
            per_symbol_cooldown=per_symbol_cooldown,
            cooldown=cooldown,
            min_trades=min_trades,
            mu_debias=mu_debias,
            mu_debias_alpha=mu_debias_alpha,
            wf_threshold_override=blended_threshold if threshold_ema is not None and wf_threshold_ema else None,
            fold_id=fold['fold'],
            per_symbol_r_kill=per_symbol_r_kill,
            per_symbol_threshold=per_symbol_threshold,
            short_oversample=short_oversample,
            short_min_fraction=short_min_fraction,
            ema200_soft_mult=ema200_soft_mult,
            per_side_threshold=per_side_threshold,
            model_version=model_version,
            v6_seq_len=v6_seq_len,
            v6_conv_channels=v6_conv_channels,
            v6_n_conv_layers=v6_n_conv_layers,
            v6_attn_heads=v6_attn_heads,
            v6_attn_layers=v6_attn_layers,
            v6_n_experts=v6_n_experts,
            v6_expert_top_k=v6_expert_top_k,
            v6_feature_mask_ratio=v6_feature_mask_ratio,
            v6_aux_weight=v6_aux_weight,
            v6_confidence_weight=v6_confidence_weight,
            v6_moe_balance_weight=v6_moe_balance_weight,
        )

        report_path = Path("checkpoints") / "v5_forward_report.json"
        if report_path.exists():
            import json
            with open(report_path) as f:
                fold_report = json.load(f)
            fold_report['fold'] = fold['fold']

            fold_threshold = fold_report.get('score_threshold', None)
            fold_total_trades = fold_report.get('total_trades', 0)
            fold_low_conf = fold_report.get('low_confidence', False)
            fold_total_r = fold_report.get('total_r', 0)

            if warm_start:
                import torch as _torch
                if fold_total_r > 0 and fold_total_trades > 0:
                    best_ckpt = Path("checkpoints") / "best_v5_expectancy.pt"
                    if not best_ckpt.exists():
                        best_ckpt = Path("checkpoints") / "best_v5_loss.pt"
                    if best_ckpt.exists():
                        try:
                            ckpt = _torch.load(best_ckpt, map_location='cpu', weights_only=False)
                            prev_fold_state_dict = ckpt['model_state_dict']
                            log.info(f"[V5_WF] Fold {fold['fold']} was profitable ({fold_total_r:+.2f}R) "
                                     f"— warm-start enabled for next fold")
                        except Exception as e:
                            log.warning(f"[V5_WF] Failed to load fold {fold['fold']} checkpoint: {e} — resetting to random init")
                            prev_fold_state_dict = None
                    else:
                        prev_fold_state_dict = None
                else:
                    prev_fold_state_dict = None
                    if fold_total_trades == 0:
                        log.info(f"[V5_WF] Fold {fold['fold']} was DEAD (0 trades) "
                                 f"— skipping warm-start, next fold uses random init")
                    else:
                        log.info(f"[V5_WF] Fold {fold['fold']} was negative ({fold_total_r:+.2f}R) "
                                 f"— skipping warm-start, next fold uses random init")

            if fold_total_trades == 0 and threshold_ema is not None:
                min_threshold = 0.01
                old_ema = threshold_ema
                threshold_ema = max(min_threshold, threshold_ema * wf_threshold_decay)
                log.info(f"[V5_WF_THR] Fold {fold['fold']}: DEAD FOLD (0 trades) — "
                         f"decaying threshold_ema {old_ema:.4f} × {wf_threshold_decay} → {threshold_ema:.4f}")
                fold_report['threshold_ema'] = threshold_ema
            elif fold_low_conf:
                log.info(f"[V5_WF_THR] Fold {fold['fold']}: LOW_CONF ({fold_total_trades} trades) — "
                         f"skipping EMA blend, keeping threshold_ema={threshold_ema:.4f}" if threshold_ema is not None else
                         f"[V5_WF_THR] Fold {fold['fold']}: LOW_CONF ({fold_total_trades} trades) — "
                         f"no prior EMA, using sweep={fold_threshold}")
                if threshold_ema is None and fold_threshold is not None:
                    threshold_ema = fold_threshold
                fold_report['threshold_ema'] = threshold_ema
            elif fold_threshold is not None:
                if wf_threshold_ema and threshold_ema is not None:
                    blended_threshold = wf_threshold_ema_alpha * fold_threshold + (1 - wf_threshold_ema_alpha) * threshold_ema
                    log.info(f"[V5_WF_THR] Fold {fold['fold']}: sweep={fold_threshold:.4f} "
                             f"prev_ema={threshold_ema:.4f} → blended={blended_threshold:.4f}")
                    threshold_ema = blended_threshold
                else:
                    threshold_ema = fold_threshold
                    blended_threshold = fold_threshold
                    log.info(f"[V5_WF_THR] Fold {fold['fold']}: initial threshold={fold_threshold:.4f} (no prior EMA)")
                fold_report['threshold_ema'] = threshold_ema

            pusher.fold_end(
                fold_num=fold['fold'],
                completed_folds=len(all_reports),
                report=fold_report,
            )

            all_reports.append(fold_report)
        else:
            if warm_start:
                prev_fold_state_dict = None
                log.info(f"[V5_WF] Fold {fold['fold']}: NO REPORT — skipping warm-start, next fold uses random init")
            if threshold_ema is not None:
                min_threshold = 0.01
                old_ema = threshold_ema
                threshold_ema = max(min_threshold, threshold_ema * wf_threshold_decay)
                log.info(f"[V5_WF_THR] Fold {fold['fold']}: NO REPORT FILE — "
                         f"decaying threshold_ema {old_ema:.4f} × {wf_threshold_decay} → {threshold_ema:.4f}")
            pusher.fold_end(
                fold_num=fold['fold'],
                completed_folds=len(all_reports),
                report={"total_trades": 0, "total_r": 0},
            )

    if all_reports:
        log.info("\n" + "=" * 120)
        log.info("  WALK-FORWARD SUMMARY")
        log.info("=" * 120)
        header = (f"{'Fold':>6} {'Window':>25} {'Trades':>8} {'L/S':>10} {'WR':>7} "
                  f"{'E[R]':>10} {'PF':>7} {'Sharpe':>8} {'MaxDD':>10} "
                  f"{'TotalR':>10} {'Threshold':>10} {'Status':>10}")
        log.info(header)
        log.info("-" * 120)

        total_trades = 0
        total_r = 0.0
        all_expectancies = []
        active_folds = 0
        total_long = 0
        total_short = 0

        for r in all_reports:
            window = f"{r.get('window_start','?')}→{r.get('window_end','?')}"
            n_trades = r['total_trades']
            thr = r.get('threshold_ema', r.get('score_threshold', 0.0))
            thr_str = f"{thr:.4f}" if thr else "-"

            if n_trades == 0:
                status = "DEAD"
                log.info(f"{r['fold']:>6} {window:>25} {n_trades:>8} {'-/-':>10} {'-':>7} "
                         f"{'-':>10} {'-':>7} {'-':>8} {'-':>10} "
                         f"{'+0.0000':>10} {thr_str:>10} {status:>10}")
            else:
                n_long = r.get('n_long', 0)
                n_short = r.get('n_short', 0)
                ls_str = f"{n_long}/{n_short}"
                is_low_conf = r.get('low_confidence', False)
                status = "LOW_CONF" if is_low_conf else "ACTIVE"
                active_folds += 1
                total_long += n_long
                total_short += n_short
                log.info(f"{r['fold']:>6} {window:>25} {n_trades:>8} {ls_str:>10} "
                         f"{r['win_rate']:>6.1%} {r['expectancy_r']:>+10.4f} {r['profit_factor']:>7.2f} "
                         f"{r['sharpe']:>8.2f} {r['max_drawdown_r']:>10.4f} "
                         f"{r['total_r']:>+10.4f} {thr_str:>10} {status:>10}")

            total_trades += n_trades
            total_r += r['total_r']
            if n_trades > 0:
                all_expectancies.append(r['expectancy_r'])

        log.info("-" * 120)
        avg_expect = float(np.mean(all_expectancies)) if all_expectancies else 0.0
        ls_total = f"{total_long}/{total_short}"
        log.info(f"{'TOTAL':>6} {'':>25} {total_trades:>8} {ls_total:>10} {'':>7} "
                 f"{avg_expect:>+10.4f} {'':>7} {'':>8} {'':>10} "
                 f"{total_r:>+10.4f} {'':>10} {'':>10}")
        log.info(f"  Active Folds: {active_folds}/{len(all_reports)} | "
                 f"Threshold EMA: {threshold_ema:.4f}" if threshold_ema else
                 f"  Active Folds: {active_folds}/{len(all_reports)}")
        log.info("=" * 120)

        wf_per_symbol = {}
        for r in all_reports:
            pss = r.get('per_symbol_stats', {})
            for sym_name, sym_stats in pss.items():
                if sym_name not in wf_per_symbol:
                    wf_per_symbol[sym_name] = {'trades': 0, 'wins': 0, 'total_r': 0.0, 'folds': 0, 'rs': []}
                wf_per_symbol[sym_name]['trades'] += sym_stats['trades']
                wf_per_symbol[sym_name]['wins'] += int(sym_stats['win_rate'] * sym_stats['trades'])
                wf_per_symbol[sym_name]['total_r'] += sym_stats['total_r']
                wf_per_symbol[sym_name]['folds'] += 1
                wf_per_symbol[sym_name]['rs'].append(sym_stats['total_r'])

        if wf_per_symbol:
            log.info("\n" + "=" * 100)
            log.info("  WALK-FORWARD PER-SYMBOL SUMMARY")
            log.info("=" * 100)
            log.info(f"{'Symbol':>12} {'Folds':>7} {'Trades':>8} {'WR':>7} {'E[R]':>10} "
                     f"{'TotalR':>10} {'Avg/Fold':>10} {'Edge':>6}")
            log.info("-" * 100)

            sorted_syms = sorted(wf_per_symbol.items(), key=lambda x: x[1]['total_r'], reverse=True)
            wf_per_symbol_report = {}
            for sym_name, sd in sorted_syms:
                wr = sd['wins'] / max(sd['trades'], 1)
                expect = sd['total_r'] / max(sd['trades'], 1)
                avg_per_fold = sd['total_r'] / max(sd['folds'], 1)
                edge = "YES" if sd['total_r'] > 0 and expect > 0 else "NO"
                log.info(f"{sym_name:>12} {sd['folds']:>7} {sd['trades']:>8} {wr:>6.1%} "
                         f"{expect:>+10.4f} {sd['total_r']:>+10.4f} {avg_per_fold:>+10.4f} {edge:>6}")
                wf_per_symbol_report[sym_name] = {
                    'folds': sd['folds'], 'trades': sd['trades'],
                    'win_rate': round(wr, 3), 'expectancy_r': round(expect, 4),
                    'total_r': round(sd['total_r'], 4), 'avg_per_fold_r': round(avg_per_fold, 4),
                    'edge': edge, 'per_fold_r': [round(x, 4) for x in sd['rs']],
                }

            log.info("-" * 100)
            edge_syms = [s for s, d in sorted_syms if d['total_r'] > 0]
            no_edge_syms = [s for s, d in sorted_syms if d['total_r'] <= 0]
            log.info(f"  Edge symbols ({len(edge_syms)}): {edge_syms}")
            log.info(f"  No-edge symbols ({len(no_edge_syms)}): {no_edge_syms}")
            log.info("=" * 100)

        agg_report = {
            'total_trades': total_trades,
            'total_r': round(total_r, 4),
            'avg_expectancy_r': round(avg_expect, 4),
            'n_folds': len(all_reports),
            'active_folds': active_folds,
            'final_threshold_ema': threshold_ema,
            'per_symbol': wf_per_symbol_report if wf_per_symbol else {},
        }

        pusher.session_end(
            status="completed",
            completed_folds=len(all_reports),
            aggregate_metrics=agg_report,
        )

        agg_path = Path("checkpoints") / "v5_walkforward_report.json"
        import json
        with open(agg_path, 'w') as f:
            json.dump({
                'folds': all_reports,
                'aggregate': agg_report,
                'per_symbol_summary': wf_per_symbol_report if wf_per_symbol else {},
            }, f, indent=2, default=str)
        log.info(f"[V5_WF] Walk-forward report saved to {agg_path}")


def train_v5_model(
    data_path, device, epochs, batch_size, lr,
    checkpoint_interval=25, warmup_epochs=5, min_lr=None,
    tp_mult=2.0, sl_mult=1.5, horizon=16,
    symbols=None,
    w_ret=1.0, w_mfe=0.25, w_mae=0.25, w_action=2.0,
    w_barrier=0.25, w_regime=0.1,
    score_lambda=0.5, risk_proxy='mae',
    target_tpd=6.5, target_tpd_tol=1.5,
    hold_target=0.30, mfe_min=0.05,
    barrier_mode='fixed',
    barrier_presets=None,
    use_regime_head=False,
    candidate_config=None,
    risk_controls=None,
    q_min_tp=0.3, r_min_expiry_strict=1.0,
    auto_balance_enter_labels=True,
    target_enter_rate=0.18,
    target_enter_rate_min=0.12,
    target_enter_rate_max=0.25,
    balance_search_steps=30,
    cand_warmup_epochs=3,
    quality_gate_cfg=None,
    tpd_ctrl_cfg=None,
    train_end_date=None,
    test_start_date=None,
    test_end_date=None,
    run_forward_test=False,
    freeze_decision=True,
    run_diagnostics=False,
    ema200_regime_gate=False,
    weekly_loss_cap=None,
    warmup_skip_bars=0,
    corr_block=False,
    corr_window_days=30,
    corr_thresh=0.70,
    corr_same_side_only=True,
    corr_log_matrix=True,
    corr_max_block=5,
    adaptive_sizing=False,
    kelly_fraction=0.25,
    max_size_mult=2.5,
    min_size_mult=0.25,
    regime_scaling=False,
    regime_bull_mult=1.5,
    regime_bear_mult=0.5,
    regime_lookback=20,
    daily_loss_cap=None,
    trailing_equity_stop=None,
    per_symbol_daily_r_budget=None,
    min_threshold=None,
    max_threshold=None,
    min_threshold_pct=None,
    max_trades_per_day=None,
    trailing_sl=False, trail_activation=1.5, trail_distance=1.0, allow_runner=False,
    conviction_sizing=False, conviction_tier_top_pct=5.0, conviction_tier_top_mult=2.5,
    conviction_tier_high_pct=20.0, conviction_tier_high_mult=1.5,
    conviction_confidence_threshold=0.65, conviction_confidence_boost=1.3,
    promote_metric='expectancy',
    adx_gate=False, adx_period=14, adx_min=18.0, adx_exception_top_pct=10.0,
    temp_scale=False,
    stage_a_epochs=0, stage_a_w_action_mult=2.0, stage_a_w_regime_mult=1.5, stage_a_w_reg_mult=0.5,
    symbol_embed_dim=8,
    balanced_sampling=True, balanced_sampling_mode='cap', per_symbol_scaler=False,
    ultra_conviction=False, ultra_risk_cap=0.05, ultra_score_pct=0.95,
    ultra_adx_min=25.0, ultra_edge_min=0.03, ultra_dd_max=0.10,
    ultra_max_per_day=1, ultra_mult=3.0,
    ddt_enable=False, ddt_lookback_trades=60, ddt_bad_rollr=6.0,
    ddt_thr_k=0.60, ddt_thr_min=0.08, ddt_thr_max=0.25,
    ddt_size_k=0.70, ddt_min_size_mult=0.25,
    ddt_alpha_down=0.30, ddt_alpha_up=0.05, ddt_warmup_trades=20,
    multi_regime=False, regime_adx_trending=25.0, regime_adx_choppy=20.0,
    regime_atr_high_vol=1.3, regime_atr_low_vol=0.7,
    regime_atr_window=96, regime_ema_slope_window=10, regime_ema_buffer=0.005,
    edge_first=False, edge_min=0.03, edge_pct_floor=70, edge_topn_per_day=4,
    regime_side_map=None,
    regime_soft=True, regime_disagree_mult=0.3, regime_none_mult=0.2,
    per_symbol_soft_kill=True,
    edge_topn_soft=True, edge_topn_decay=0.7,
    size_floor=0.0,
    soft_gate_floor=True,
    weekly_cap_dynamic=False, weekly_cap_scale=2.0,
    quality_gate_enabled=False, quality_gate_window=50,
    direction_balance_cap=False, direction_balance_threshold=0.75,
    recency_weight=False, recency_half_life=90,
    finetune_months=0, finetune_epochs=5, finetune_lr_mult=0.1,
    warm_start_state_dict=None,
    head_disagreement_gate=False, slippage_base_bps=0.0,
    sigma_discount=False, min_p_side=0.0, min_p_short=0.0,
    side_aware_scoring=False, mae_asym_weight=1.0,
    per_symbol_cooldown=True,
    cooldown=4,
    mu_debias=True, mu_debias_alpha=0.01,
    min_trades=20,
    wf_threshold_override=None,
    fold_id=0,
    feature_report=False,
    per_symbol_r_kill=None,
    per_symbol_threshold=False,
    short_oversample=False,
    short_min_fraction=0.35,
    ema200_soft_mult=None,
    per_side_threshold=False,
    model_version='v5',
    v6_seq_len=16,
    v6_conv_channels=128,
    v6_n_conv_layers=3,
    v6_attn_heads=4,
    v6_attn_layers=2,
    v6_n_experts=4,
    v6_expert_top_k=2,
    v6_feature_mask_ratio=0.15,
    v6_aux_weight=0.1,
    v6_confidence_weight=0.15,
    v6_moe_balance_weight=0.05,
):
    """V5/V6 Forecaster training pipeline with quality gating + TPD controller."""
    from config import config as app_config
    from data.candidate_generator import (
        CandidateConfig, generate_candidate_mask,
        apply_risk_controls, RiskControls,
        BARRIER_PRESETS, PresetConfig,
    )
    from data.v5_target_generator import build_v5_targets, build_barrier_preset_labels
    from models.v5_forecaster import V5Forecaster, V5ForecasterConfig

    use_v6 = (model_version == 'v6')
    if use_v6:
        from models.v6_forecaster import V6Forecaster, V6ForecasterConfig
        log.info("[V6] V6Forecaster architecture enabled — Temporal-MoE-Attention model")

    if candidate_config is None:
        candidate_config = CandidateConfig(enabled=False)
    if risk_controls is None:
        risk_controls = RiskControls()
    if quality_gate_cfg is None:
        quality_gate_cfg = V5QualityGateConfig()
    if tpd_ctrl_cfg is None:
        tpd_ctrl_cfg = V5TPDControllerConfig(
            target_tpd=target_tpd, tpd_tol=target_tpd_tol,
            score_lambda=score_lambda,
        )
    else:
        target_tpd = tpd_ctrl_cfg.target_tpd
        target_tpd_tol = tpd_ctrl_cfg.tpd_tol

    presets = []
    if barrier_presets:
        for name in barrier_presets:
            presets.append(BARRIER_PRESETS.get(name, BARRIER_PRESETS['standard']))
    if not presets:
        presets = [{'tp_mult': tp_mult, 'sl_mult': sl_mult, 'label': 'default'}]

    if min_lr is None:
        min_lr = lr * 0.01

    vtag = "V6" if use_v6 else "V5"
    ctag = f"[{vtag}_CONFIG]"

    log.info("=" * 60)
    if use_v6:
        log.info("  V6.0 TEMPORAL-MoE-ATTENTION FORECASTER - TRAINING")
    else:
        log.info("  V5.0.1 FORECASTER - TRAINING")
    log.info("=" * 60)
    log.info(f"Version: {V5_FEATURE_VERSION} (model: {vtag})")
    if use_v6:
        log.info(f"{ctag} seq_len={v6_seq_len} conv_channels={v6_conv_channels} n_conv_layers={v6_n_conv_layers}")
        log.info(f"{ctag} attn_heads={v6_attn_heads} attn_layers={v6_attn_layers}")
        log.info(f"{ctag} n_experts={v6_n_experts} expert_top_k={v6_expert_top_k}")
        log.info(f"{ctag} feature_mask_ratio={v6_feature_mask_ratio}")
        log.info(f"{ctag} loss weights: aux={v6_aux_weight} confidence={v6_confidence_weight} moe_balance={v6_moe_balance_weight}")
    log.info(f"{ctag} w_ret={w_ret} w_mfe={w_mfe} w_mae={w_mae} w_action={w_action}")
    log.info(f"{ctag} w_barrier={w_barrier} w_regime={w_regime}")
    log.info(f"{ctag} score_lambda={score_lambda} risk_proxy={risk_proxy}")
    log.info(f"{ctag} hold_target={hold_target} mfe_min={mfe_min}")
    log.info(f"{ctag} barrier_mode={barrier_mode} presets={[p.get('label','?') for p in presets]}")
    log.info(f"{ctag} target_tpd={target_tpd} tpd_tol={target_tpd_tol}")
    log.info(f"{ctag} candidates={candidate_config.enabled} regime_head={use_regime_head}")
    log.info(f"{ctag} cand_warmup_epochs={cand_warmup_epochs}")
    log.info(f"{ctag} horizon={horizon} epochs={epochs} batch={batch_size} lr={lr}")
    log.info(f"{ctag} ALL targets in R-units (price_change / ATR)")
    log.info(f"{ctag} balanced_sampling={balanced_sampling} balanced_sampling_mode={balanced_sampling_mode} per_symbol_scaler={per_symbol_scaler}")
    log.info(f"[{vtag}_QUAL_CONFIG] sigma_max={quality_gate_cfg.sigma_max} mae_max={quality_gate_cfg.mae_max} "
             f"mu_R_min={quality_gate_cfg.mu_R_min} p_trade_min={quality_gate_cfg.p_trade_min} "
             f"enable_calib={quality_gate_cfg.enable_calib}")
    log.info(f"[{vtag}_TPD_CONFIG] target={tpd_ctrl_cfg.target_tpd}±{tpd_ctrl_cfg.tpd_tol} "
             f"warmup={tpd_ctrl_cfg.thr_warmup_epochs} step_mult={tpd_ctrl_cfg.thr_step_mult} "
             f"mae_cap={tpd_ctrl_cfg.mae_cap} init_thr={tpd_ctrl_cfg.score_threshold}")

    if barrier_mode == 'oracle':
        log.warning("[V5] barrier_mode=oracle: WARNING hindsight leakage, research only!")

    from data.pipeline import FeatureEngineer
    data_dir = Path("data_cache")

    if symbols is None or len(symbols) == 0:
        symbols = ["BTCUSDT"]

    train_features = []
    train_ret_R_list = []
    train_mfe_R_list = []
    train_mae_R_list = []
    train_vol_h_list = []
    train_action_list = []
    train_valid_list = []
    train_sym_ids_list = []
    train_cand_mask_list = []
    train_outcomes_list = []
    train_realized_r_list = []
    train_barrier_oracle_list = []
    train_barrier_soft_list = []
    train_timestamps_list = []

    val_features = []
    val_ret_R_list = []
    val_mfe_R_list = []
    val_mae_R_list = []
    val_vol_h_list = []
    val_action_list = []
    val_valid_list = []
    val_sym_ids_list = []
    val_cand_mask_list = []
    val_outcomes_list = []
    val_realized_r_list = []
    val_r_long_list = []
    val_r_short_list = []
    val_out_long_list = []
    val_out_short_list = []
    val_barrier_oracle_list = []
    val_barrier_soft_list = []
    val_timestamps_list = []
    val_close_list = []
    val_high_list = []
    val_low_list = []

    features_df_columns = None

    from data.common import generate_v5_sweep_outcomes
    if trailing_sl:
        from data.common import generate_v5_sweep_outcomes_trailing

    for si, sym in enumerate(symbols):
        parquet_path = data_dir / f"{sym}_15m.parquet"
        if not parquet_path.exists():
            log.error(f"[V5] Data file not found: {parquet_path}")
            sys.exit(1)

        log.info(f"[V5] Loading {sym} from {parquet_path}")
        sym_df = pd.read_parquet(parquet_path)
        sym_df = sym_df.sort_values('timestamp').reset_index(drop=True)
        log.info(f"[V5] {sym}: {len(sym_df)} bars loaded")

        fe = FeatureEngineer()
        sym_features_df = fe.compute_all_features(sym_df)

        max_lookback = 50
        warmup_mask = np.zeros(len(sym_features_df), dtype=bool)
        warmup_mask[:max_lookback] = True
        sym_features_df = sym_features_df.ffill().bfill()
        sym_features_df = sym_features_df.fillna(0)

        if features_df_columns is None:
            features_df_columns = list(sym_features_df.columns)
        else:
            sym_features_df = sym_features_df.reindex(columns=features_df_columns, fill_value=0)

        sym_cand_mask = None
        if candidate_config.enabled:
            sym_cand_mask, _ = generate_candidate_mask(
                sym_df, candidate_config, symbol=sym
            )

        if trailing_sl:
            sweep_result = generate_v5_sweep_outcomes_trailing(
                sym_df, horizon=horizon, tp_mult=tp_mult,
                sl_mult=sl_mult, atr_period=14,
                trail_activation=trail_activation,
                trail_distance=trail_distance,
                allow_runner=allow_runner,
            )
            if si == 0:
                log.info(f"[V5] Trailing SL ENABLED: activation={trail_activation}x ATR, "
                         f"distance={trail_distance}x ATR, runner={allow_runner}")
        else:
            sweep_result = generate_v5_sweep_outcomes(
                sym_df, horizon=horizon, tp_mult=tp_mult,
                sl_mult=sl_mult, atr_period=14,
            )

        v5_targets = build_v5_targets(
            sym_df, horizon=horizon, atr_period=14,
            hold_target=hold_target, mfe_min_r=mfe_min,
            barrier_outcomes=sweep_result,
        )

        v5_targets['valid_mask'][:max_lookback] = False
        sym_realized_r = sweep_result['realized_r']
        sym_outcomes = sweep_result['outcome']
        sym_r_long = sweep_result['r_long']
        sym_r_short = sweep_result['r_short']
        sym_out_long = sweep_result['out_long']
        sym_out_short = sweep_result['out_short']

        sym_realized_r[:max_lookback] = np.nan
        sym_outcomes[:max_lookback] = "NO_CANDIDATE"
        sym_r_long[:max_lookback] = np.nan
        sym_r_short[:max_lookback] = np.nan
        sym_out_long[:max_lookback] = "NO_CANDIDATE"
        sym_out_short[:max_lookback] = "NO_CANDIDATE"
        if sym_cand_mask is not None:
            sym_cand_mask[:max_lookback] = False

        barrier_oracle = np.zeros(len(sym_df), dtype=np.int64)
        barrier_soft = np.zeros((len(sym_df), len(presets)), dtype=np.float32)
        if len(presets) > 1 and barrier_mode in ('oracle', 'learnable'):
            barrier_oracle, barrier_soft = build_barrier_preset_labels(
                sym_df, presets, horizon=horizon, temperature=1.0
            )

        n = len(sym_features_df)
        sym_id_arr = np.full(n, si, dtype=np.int64)
        cand_arr = sym_cand_mask if sym_cand_mask is not None else np.ones(n, dtype=bool)

        train_idx, test_idx = _compute_time_split(
            sym_df, train_end_date=train_end_date,
            test_start_date=test_start_date, test_end_date=test_end_date,
            purge_bars=horizon,
        )

        if train_end_date:
            train_ts = sym_df['timestamp'].values
            train_date_min = datetime.utcfromtimestamp(train_ts[train_idx[0]] / 1000).strftime('%Y-%m-%d') if len(train_idx) > 0 else "?"
            train_date_max = datetime.utcfromtimestamp(train_ts[train_idx[-1]] / 1000).strftime('%Y-%m-%d') if len(train_idx) > 0 else "?"
            test_date_min = datetime.utcfromtimestamp(train_ts[test_idx[0]] / 1000).strftime('%Y-%m-%d') if len(test_idx) > 0 else "?"
            test_date_max = datetime.utcfromtimestamp(train_ts[test_idx[-1]] / 1000).strftime('%Y-%m-%d') if len(test_idx) > 0 else "?"
            log.info(f"[V5] {sym}: TIME-BASED split train={len(train_idx)} ({train_date_min}→{train_date_max}) "
                     f"val/test={len(test_idx)} ({test_date_min}→{test_date_max})")
        else:
            log.info(f"[V5] {sym}: per-symbol split at bar {len(train_idx)}/{n} "
                     f"(train={len(train_idx)}, val={len(test_idx)})")

        feat_arr = sym_features_df.values.astype(np.float32)
        ret_arr = v5_targets['ret_R'][:n]
        vol_arr = v5_targets['vol_h'][:n]
        act_arr = v5_targets['action_label'][:n]
        val_arr = v5_targets['valid_mask'][:n]

        mfe_long = v5_targets['mfe_R_long'][:n] if 'mfe_R_long' in v5_targets else v5_targets['mfe_R'][:n]
        mae_long = v5_targets['mae_R_long'][:n] if 'mae_R_long' in v5_targets else v5_targets['mae_R'][:n]
        mfe_short = v5_targets['mfe_R_short'][:n] if 'mfe_R_short' in v5_targets else v5_targets['mfe_R'][:n]
        mae_short = v5_targets['mae_R_short'][:n] if 'mae_R_short' in v5_targets else v5_targets['mae_R'][:n]
        mfe_arr = np.where(act_arr == 2, mfe_short, mfe_long)
        mae_arr = np.where(act_arr == 2, mae_short, mae_long)

        train_features.append(feat_arr[train_idx])
        train_ret_R_list.append(ret_arr[train_idx])
        train_mfe_R_list.append(mfe_arr[train_idx])
        train_mae_R_list.append(mae_arr[train_idx])
        train_vol_h_list.append(vol_arr[train_idx])
        train_action_list.append(act_arr[train_idx])
        train_valid_list.append(val_arr[train_idx])
        train_sym_ids_list.append(sym_id_arr[train_idx])
        train_cand_mask_list.append(cand_arr[train_idx])
        train_outcomes_list.append(sym_outcomes[train_idx])
        train_realized_r_list.append(sym_realized_r[train_idx])
        train_barrier_oracle_list.append(barrier_oracle[train_idx])
        train_barrier_soft_list.append(barrier_soft[train_idx])
        train_timestamps_list.append(sym_df['timestamp'].values[train_idx])

        val_features.append(feat_arr[test_idx])
        val_ret_R_list.append(ret_arr[test_idx])
        val_mfe_R_list.append(mfe_arr[test_idx])
        val_mae_R_list.append(mae_arr[test_idx])
        val_vol_h_list.append(vol_arr[test_idx])
        val_action_list.append(act_arr[test_idx])
        val_valid_list.append(val_arr[test_idx])
        val_sym_ids_list.append(sym_id_arr[test_idx])
        val_cand_mask_list.append(cand_arr[test_idx])
        val_outcomes_list.append(sym_outcomes[test_idx])
        val_realized_r_list.append(sym_realized_r[test_idx])
        val_r_long_list.append(sym_r_long[test_idx])
        val_r_short_list.append(sym_r_short[test_idx])
        val_out_long_list.append(sym_out_long[test_idx])
        val_out_short_list.append(sym_out_short[test_idx])
        val_barrier_oracle_list.append(barrier_oracle[test_idx])
        val_barrier_soft_list.append(barrier_soft[test_idx])
        val_timestamps_list.append(sym_df['timestamp'].values[test_idx])
        val_close_list.append(sym_df['close'].values[test_idx])
        val_high_list.append(sym_df['high'].values[test_idx])
        val_low_list.append(sym_df['low'].values[test_idx])

    per_sym_train_counts = [len(arr) for arr in train_features]
    per_sym_val_counts = [len(arr) for arr in val_features]
    for si_log, sym_log in enumerate(symbols):
        log.info(f"[V5_BALANCE] {sym_log}: train={per_sym_train_counts[si_log]} val={per_sym_val_counts[si_log]}")

    effective_mode = 'none'
    if balanced_sampling and len(symbols) > 1 and len(train_features) > 1:
        effective_mode = balanced_sampling_mode if balanced_sampling_mode in ('cap', 'weighted') else 'cap'

    per_symbol_sample_weights = None

    if effective_mode == 'weighted':
        nonzero_counts = [c for c in per_sym_train_counts if c > 0]
        zero_syms = [symbols[i] for i, c in enumerate(per_sym_train_counts) if c == 0]
        if zero_syms:
            log.info(f"[V6_BALANCE] Symbols with 0 train samples: {zero_syms}")
        if nonzero_counts:
            max_count = max(nonzero_counts)
            per_symbol_sample_weights = []
            weight_log_parts = []
            for i in range(len(train_features)):
                count_i = per_sym_train_counts[i]
                if count_i > 0:
                    w_i = max_count / count_i
                else:
                    w_i = 1.0
                per_symbol_sample_weights.append(np.full(count_i, w_i, dtype=np.float32))
                sym_name = symbols[i] if i < len(symbols) else f"sym_{i}"
                weight_log_parts.append(f"{sym_name}: {w_i:.2f}x")
            log.info(f"[V6_BALANCE] Weighted mode — keeping ALL data, inverse-frequency weights: "
                     f"{', '.join(weight_log_parts)}")
        else:
            log.warning(f"[V6_BALANCE] ALL symbols have 0 train samples — skipping balance step")

    elif effective_mode == 'cap':
        nonzero_counts = [c for c in per_sym_train_counts if c > 0]
        zero_syms = [symbols[i] for i, c in enumerate(per_sym_train_counts) if c == 0]
        if zero_syms:
            log.info(f"[V5_BALANCE] Symbols with 0 train samples in this fold (not yet listed): {zero_syms}")
        if not nonzero_counts:
            log.warning(f"[V5_BALANCE] ALL symbols have 0 train samples — skipping balance step")
            min_train = 0
            max_train = 0
        else:
            min_train = min(nonzero_counts)
            max_train = max(nonzero_counts)
        if min_train > 0 and max_train > min_train * 1.05:
            log.info(f"[V5_BALANCE] Capping per-symbol train samples to min={min_train} "
                     f"(was max={max_train}, ratio={max_train/min_train:.2f}x)")
            for i in range(len(train_features)):
                if len(train_features[i]) > min_train:
                    train_features[i] = train_features[i][:min_train]
                    train_ret_R_list[i] = train_ret_R_list[i][:min_train]
                    train_mfe_R_list[i] = train_mfe_R_list[i][:min_train]
                    train_mae_R_list[i] = train_mae_R_list[i][:min_train]
                    train_vol_h_list[i] = train_vol_h_list[i][:min_train]
                    train_action_list[i] = train_action_list[i][:min_train]
                    train_valid_list[i] = train_valid_list[i][:min_train]
                    train_sym_ids_list[i] = train_sym_ids_list[i][:min_train]
                    train_cand_mask_list[i] = train_cand_mask_list[i][:min_train]
                    train_outcomes_list[i] = train_outcomes_list[i][:min_train]
                    train_realized_r_list[i] = train_realized_r_list[i][:min_train]
                    train_barrier_oracle_list[i] = train_barrier_oracle_list[i][:min_train]
                    train_barrier_soft_list[i] = train_barrier_soft_list[i][:min_train]
                    if i < len(train_timestamps_list):
                        train_timestamps_list[i] = train_timestamps_list[i][:min_train]
            balanced_counts = [len(arr) for arr in train_features]
            log.info(f"[V5_BALANCE] After balancing: {dict(zip(symbols, balanced_counts))}")
        else:
            log.info(f"[V5_BALANCE] Symbol sizes within 5% — no capping needed")
    else:
        if len(symbols) > 1:
            log.info(f"[V5_BALANCE] Balanced sampling DISABLED — using all data as-is")

    from sklearn.preprocessing import RobustScaler
    per_symbol_scalers = {}

    if per_symbol_scaler and len(symbols) > 1:
        log.info("[V5_SCALER] Fitting per-symbol RobustScalers...")
        for si, sym_name in enumerate(symbols):
            if len(train_features[si]) == 0:
                log.warning(f"[V5_SCALER] {sym_name}: SKIPPED — 0 train samples in this fold")
                per_symbol_scalers[sym_name] = None
                continue
            if len(val_features[si]) == 0:
                log.warning(f"[V5_SCALER] {sym_name}: SKIPPED — 0 val samples in this fold (train has {len(train_features[si])})")
                sym_scaler = RobustScaler()
                train_features[si] = sym_scaler.fit_transform(train_features[si]).astype(np.float32)
                per_symbol_scalers[sym_name] = sym_scaler
                continue
            sym_scaler = RobustScaler()
            train_features[si] = sym_scaler.fit_transform(train_features[si]).astype(np.float32)
            val_features[si] = sym_scaler.transform(val_features[si]).astype(np.float32)
            per_symbol_scalers[sym_name] = sym_scaler
            tr_mean = np.mean(train_features[si], axis=0)
            tr_std = np.std(train_features[si], axis=0)
            log.info(f"[{vtag}_SCALER] {sym_name}: fitted on {len(train_features[si])} train bars, "
                     f"applied to {len(val_features[si])} val bars | "
                     f"post-scale mean=[{tr_mean.min():.3f}, {tr_mean.max():.3f}] "
                     f"std=[{tr_std.min():.3f}, {tr_std.max():.3f}]")

        non_empty_train = [f for f in train_features if len(f) > 0]
        non_empty_val = [f for f in val_features if len(f) > 0]
        train_feat = np.concatenate(non_empty_train, axis=0) if non_empty_train else np.empty((0, train_features[0].shape[1] if train_features else 85), dtype=np.float32)
        val_feat = np.concatenate(non_empty_val, axis=0) if non_empty_val else np.empty((0, train_feat.shape[1]), dtype=np.float32)

        scaler = RobustScaler()
        scaler.center_ = np.zeros(train_feat.shape[1])
        scaler.scale_ = np.ones(train_feat.shape[1])
        log.info(f"[{vtag}_SCALER] Per-symbol scaling complete. Identity global scaler set for checkpoint compat.")

        import joblib
        scalers_path = Path("checkpoints") / "per_symbol_scalers.joblib"
        scalers_path.parent.mkdir(exist_ok=True)
        joblib.dump(per_symbol_scalers, scalers_path)
        log.info(f"[{vtag}_SCALER] Saved {len(per_symbol_scalers)} per-symbol scalers to {scalers_path}")
    else:
        if per_symbol_scaler and len(symbols) <= 1:
            log.info(f"[{vtag}_SCALER] --per-symbol-scaler enabled but only 1 symbol — using global scaler")

        train_feat = np.concatenate(train_features, axis=0)
        val_feat = np.concatenate(val_features, axis=0)

        scaler = RobustScaler()
        train_feat = scaler.fit_transform(train_feat).astype(np.float32)
        val_feat = scaler.transform(val_feat).astype(np.float32)
        log.info(f"[{vtag}] RobustScaler fitted on {len(train_feat)} train bars, applied to {len(val_feat)} val bars")

    train_regime_trend_raw = None
    if features_df_columns is not None and 'regime_trend' in features_df_columns:
        rt_idx = features_df_columns.index('regime_trend')
        train_regime_trend_raw = train_feat[:, rt_idx].copy()

    def _concat_lists(lst):
        return np.concatenate(lst, axis=0)

    train_ret_R = _concat_lists(train_ret_R_list)
    train_mfe_R = _concat_lists(train_mfe_R_list)
    train_mae_R = _concat_lists(train_mae_R_list)
    train_vol_h = _concat_lists(train_vol_h_list)
    train_action = _concat_lists(train_action_list)
    train_valid = _concat_lists(train_valid_list)
    train_sym_ids = _concat_lists(train_sym_ids_list)
    train_cand_mask = _concat_lists(train_cand_mask_list)
    train_outcomes = _concat_lists(train_outcomes_list)
    train_realized_r = _concat_lists(train_realized_r_list)
    train_barrier_oracle = _concat_lists(train_barrier_oracle_list)
    train_barrier_soft = np.concatenate(train_barrier_soft_list, axis=0)
    train_timestamps = _concat_lists(train_timestamps_list) if train_timestamps_list else np.array([], dtype=np.float64)

    val_ret_R = _concat_lists(val_ret_R_list)
    val_mfe_R = _concat_lists(val_mfe_R_list)
    val_mae_R = _concat_lists(val_mae_R_list)
    val_vol_h = _concat_lists(val_vol_h_list)
    val_action_arr = _concat_lists(val_action_list)
    val_valid = _concat_lists(val_valid_list)
    val_sym_ids_arr = _concat_lists(val_sym_ids_list)
    val_cand_mask_arr = _concat_lists(val_cand_mask_list)
    val_outcomes_arr = _concat_lists(val_outcomes_list)
    val_realized_r_arr = _concat_lists(val_realized_r_list)
    val_r_long_arr = _concat_lists(val_r_long_list)
    val_r_short_arr = _concat_lists(val_r_short_list)
    val_out_long_arr = _concat_lists(val_out_long_list)
    val_out_short_arr = _concat_lists(val_out_short_list)
    val_barrier_oracle = _concat_lists(val_barrier_oracle_list)
    val_barrier_soft = np.concatenate(val_barrier_soft_list, axis=0)
    val_timestamps_arr = np.concatenate(val_timestamps_list, axis=0)
    val_close_arr = np.concatenate(val_close_list, axis=0)
    val_high_arr = np.concatenate(val_high_list, axis=0)
    val_low_arr = np.concatenate(val_low_list, axis=0)

    for label, arr, vmask in [
        ('train_ret_R', train_ret_R, train_valid),
        ('train_mfe_R', train_mfe_R, train_valid),
        ('train_mae_R', train_mae_R, train_valid),
        ('val_ret_R', val_ret_R, val_valid),
        ('val_mfe_R', val_mfe_R, val_valid),
        ('val_mae_R', val_mae_R, val_valid),
    ]:
        nan_in_valid = np.sum(np.isnan(arr[vmask])) if np.any(vmask) else 0
        if nan_in_valid > 0:
            log.warning(f"[{vtag}_NAN_AUDIT] %s has %d NaNs in %d valid bars (%.1f%%)",
                        label, nan_in_valid, int(np.sum(vmask)),
                        100.0 * nan_in_valid / max(int(np.sum(vmask)), 1))
    nan_feats_train = np.sum(np.isnan(train_feat))
    nan_feats_val = np.sum(np.isnan(val_feat))
    if nan_feats_train > 0 or nan_feats_val > 0:
        log.warning(f"[{vtag}_NAN_AUDIT] Features NaN: train=%d val=%d", nan_feats_train, nan_feats_val)

    train_ret_R = np.nan_to_num(train_ret_R, nan=0.0)
    train_mfe_R = np.nan_to_num(train_mfe_R, nan=0.0)
    train_mae_R = np.nan_to_num(train_mae_R, nan=0.0)
    train_vol_h = np.nan_to_num(train_vol_h, nan=0.0)
    val_ret_R = np.nan_to_num(val_ret_R, nan=0.0)
    val_mfe_R = np.nan_to_num(val_mfe_R, nan=0.0)
    val_mae_R = np.nan_to_num(val_mae_R, nan=0.0)
    val_vol_h = np.nan_to_num(val_vol_h, nan=0.0)

    total_train = len(train_feat)
    total_val = len(val_feat)
    total_bars = total_train + total_val
    input_dim = train_feat.shape[1]
    log.info(f"[{vtag}] Total bars: {total_bars} (train={total_train}, val={total_val}) | "
             f"Features: {input_dim} | Symbols: {len(symbols)}")

    if len(symbols) > 1:
        for si_log, sym_log in enumerate(symbols):
            sym_train_n = int(np.sum(train_sym_ids == si_log))
            sym_val_n = int(np.sum(val_sym_ids_arr == si_log))
            log.info(f"[{vtag}_SYM_DIST] {sym_log} (id={si_log}): train={sym_train_n} val={sym_val_n}")

    valid_train_action = train_action[train_valid]
    n_hold = int(np.sum(valid_train_action == 0))
    n_long = int(np.sum(valid_train_action == 1))
    n_short = int(np.sum(valid_train_action == 2))
    n_total_act = max(n_hold + n_long + n_short, 1)

    log.info(f"[{vtag}_ACTION_DIST] TRAIN: HOLD={n_hold} ({n_hold/n_total_act:.1%}) "
             f"LONG={n_long} ({n_long/n_total_act:.1%}) SHORT={n_short} ({n_short/n_total_act:.1%})")
    long_pct = n_long / max(n_long + n_short, 1) * 100
    short_pct = n_short / max(n_long + n_short, 1) * 100
    log.info(f"[{vtag}_SIDE_BALANCE] Training label bias: LONG={long_pct:.1f}% SHORT={short_pct:.1f}% | "
             f"Using balanced 50/50 KL target (not training distribution)")

    action_class_weights = np.ones(3, dtype=np.float32)
    if n_hold > 0 and n_long > 0 and n_short > 0:
        counts = np.array([n_hold, n_long, n_short], dtype=np.float64)
        inv_freq = n_total_act / (3.0 * counts)
        inv_freq = np.clip(inv_freq, 0.5, 3.0)
        action_class_weights = inv_freq.astype(np.float32)
    log.info(f"[{vtag}_ACTION_DIST] Class weights: HOLD={action_class_weights[0]:.3f} "
             f"LONG={action_class_weights[1]:.3f} SHORT={action_class_weights[2]:.3f}")

    if short_oversample and n_long > 0:
        import math
        min_frac = float(short_min_fraction)
        target_short = int(math.ceil(n_long * min_frac / max(1.0 - min_frac, 1e-8)))
        if n_short < target_short:
            extra_needed = target_short - n_short
            short_indices_all = np.where(train_valid & (train_action == 2))[0]
            if len(short_indices_all) > 0:
                rng = np.random.default_rng(seed=42)
                oversample_idx = rng.choice(short_indices_all, size=extra_needed, replace=True)
                train_feat = np.concatenate([train_feat, train_feat[oversample_idx]], axis=0)
                train_ret_R = np.concatenate([train_ret_R, train_ret_R[oversample_idx]])
                train_mfe_R = np.concatenate([train_mfe_R, train_mfe_R[oversample_idx]])
                train_mae_R = np.concatenate([train_mae_R, train_mae_R[oversample_idx]])
                train_vol_h = np.concatenate([train_vol_h, train_vol_h[oversample_idx]])
                train_action = np.concatenate([train_action, train_action[oversample_idx]])
                train_valid = np.concatenate([train_valid, train_valid[oversample_idx]])
                train_sym_ids = np.concatenate([train_sym_ids, train_sym_ids[oversample_idx]])
                train_barrier_oracle = np.concatenate([train_barrier_oracle, train_barrier_oracle[oversample_idx]])
                train_barrier_soft = np.concatenate([train_barrier_soft, train_barrier_soft[oversample_idx]], axis=0)
                if len(train_timestamps) > 0 and len(train_timestamps) == len(train_feat) - extra_needed:
                    train_timestamps = np.concatenate([train_timestamps, train_timestamps[oversample_idx]])
                before_n = n_short
                n_short = int(np.sum(train_valid & (train_action == 2)))
                n_total_act = max(int(np.sum(train_valid & (train_action == 0))) + n_long + n_short, 1)
                new_frac = n_short / max(n_long + n_short, 1)
                log.info(f"[V5_SHORT_OS] Oversampled {extra_needed} SHORT: {before_n} → {n_short} "
                         f"({new_frac:.1%} of LONG+SHORT) target_frac={min_frac:.0%} target_count={target_short}")
                action_class_weights = np.ones(3, dtype=np.float32)
                n_hold_new = int(np.sum(train_valid & (train_action == 0)))
                if n_hold_new > 0 and n_long > 0 and n_short > 0:
                    counts_new = np.array([n_hold_new, n_long, n_short], dtype=np.float64)
                    inv_freq_new = n_total_act / (3.0 * counts_new)
                    inv_freq_new = np.clip(inv_freq_new, 0.5, 3.0)
                    action_class_weights = inv_freq_new.astype(np.float32)
                    log.info(f"[V5_SHORT_OS] Updated class weights after oversample: "
                             f"HOLD={action_class_weights[0]:.3f} LONG={action_class_weights[1]:.3f} "
                             f"SHORT={action_class_weights[2]:.3f}")
        else:
            log.info(f"[V5_SHORT_OS] n_short={n_short} already >= target={target_short} "
                     f"(min_frac={min_frac:.0%}) — no oversampling needed")

    action_weights_tensor = torch.tensor(action_class_weights, dtype=torch.float32).to(device)

    valid_val_action = val_action_arr[val_valid]
    vn_hold = int(np.sum(valid_val_action == 0))
    vn_long = int(np.sum(valid_val_action == 1))
    vn_short = int(np.sum(valid_val_action == 2))
    vn_total = max(vn_hold + vn_long + vn_short, 1)
    log.info(f"[{vtag}_ACTION_DIST] VAL: HOLD={vn_hold} ({vn_hold/vn_total:.1%}) "
             f"LONG={vn_long} ({vn_long/vn_total:.1%}) SHORT={vn_short} ({vn_short/vn_total:.1%})")

    train_ret_valid = train_ret_R[train_valid]
    if len(train_ret_valid) > 0:
        log.info(f"[{vtag}_DATA_DIAG] ret_R train: mean={np.mean(train_ret_valid):.4f} "
                 f"std={np.std(train_ret_valid):.4f} p5={np.percentile(train_ret_valid,5):.4f} "
                 f"p95={np.percentile(train_ret_valid,95):.4f}")

    concat_sample_weights = None
    if per_symbol_sample_weights is not None:
        concat_sample_weights = np.concatenate(per_symbol_sample_weights, axis=0)
        log.info(f"[V6_BALANCE] Concatenated sample_weights: len={len(concat_sample_weights)} "
                 f"min={concat_sample_weights.min():.2f} max={concat_sample_weights.max():.2f}")

    if recency_weight and recency_half_life <= 0:
        log.warning(f"[V5_RECENCY] Invalid recency_half_life={recency_half_life} — must be > 0, disabling")
        recency_weight = False

    if recency_weight and len(train_timestamps) > 0:
        ts_max = train_timestamps.max()
        ts_min = train_timestamps.min()
        if ts_max > ts_min:
            half_life_ms = recency_half_life * 24 * 3600 * 1000
            decay_rate = np.log(2) / half_life_ms
            recency_weights = np.exp(decay_rate * (train_timestamps - ts_max))
            recency_weights = recency_weights / recency_weights.mean()
            log.info(f"[V5_RECENCY] Recency weighting enabled: half_life={recency_half_life}d "
                     f"weights min={recency_weights.min():.3f} max={recency_weights.max():.3f} "
                     f"mean={recency_weights.mean():.3f} median={np.median(recency_weights):.3f}")
            if concat_sample_weights is not None:
                recency_weights = recency_weights * concat_sample_weights
                log.info(f"[V5_RECENCY] Combined with existing weights: "
                         f"min={recency_weights.min():.3f} max={recency_weights.max():.3f}")
            concat_sample_weights = recency_weights.astype(np.float32)
        else:
            log.warning("[V5_RECENCY] All timestamps identical — skipping recency weighting")

    n_barrier = len(presets) if len(presets) > 1 and barrier_mode != 'fixed' else 0
    n_syms = len(symbols) if len(symbols) > 1 else 1

    if use_v6:
        train_sw_per_sym = None
        if per_symbol_sample_weights is not None:
            train_sw_per_sym = per_symbol_sample_weights

        train_ds = V6SequenceDataset(
            features_per_symbol=train_features,
            ret_R_per_symbol=train_ret_R_list,
            mfe_R_per_symbol=train_mfe_R_list,
            mae_R_per_symbol=train_mae_R_list,
            vol_h_per_symbol=train_vol_h_list,
            action_per_symbol=train_action_list,
            valid_per_symbol=train_valid_list,
            symbol_ids_per_symbol=train_sym_ids_list,
            barrier_oracle_per_symbol=train_barrier_oracle_list,
            barrier_soft_per_symbol=train_barrier_soft_list,
            seq_len=v6_seq_len,
            sample_weights_per_symbol=train_sw_per_sym,
        )
        val_ds = V6SequenceDataset(
            features_per_symbol=val_features,
            ret_R_per_symbol=val_ret_R_list,
            mfe_R_per_symbol=val_mfe_R_list,
            mae_R_per_symbol=val_mae_R_list,
            vol_h_per_symbol=val_vol_h_list,
            action_per_symbol=val_action_list,
            valid_per_symbol=val_valid_list,
            symbol_ids_per_symbol=val_sym_ids_list,
            barrier_oracle_per_symbol=val_barrier_oracle_list,
            barrier_soft_per_symbol=val_barrier_soft_list,
            seq_len=v6_seq_len,
        )
        log.info(f"[V6] V6SequenceDataset created: train={len(train_ds)} val={len(val_ds)} seq_len={v6_seq_len}")
    else:
        train_ds = V5Dataset(
            train_feat, train_ret_R, train_mfe_R,
            train_mae_R, train_vol_h, train_action,
            train_valid, train_sym_ids,
            train_barrier_oracle, train_barrier_soft,
            sample_weights=concat_sample_weights,
        )
        val_ds = V5Dataset(
            val_feat, val_ret_R, val_mfe_R,
            val_mae_R, val_vol_h, val_action_arr,
            val_valid, val_sym_ids_arr,
            val_barrier_oracle, val_barrier_soft,
        )

    regime_sample_weights = None
    if train_regime_trend_raw is not None and not use_v6:
        regime_trend_vals = train_regime_trend_raw
        regime_sample_weights = np.ones(len(train_feat), dtype=np.float64)
        trending_mask = np.abs(regime_trend_vals) > 0.5
        choppy_mask = np.abs(regime_trend_vals) < 0.2
        regime_sample_weights[trending_mask] = 1.3
        regime_sample_weights[choppy_mask] = 0.7
        n_trending = int(np.sum(trending_mask))
        n_choppy = int(np.sum(choppy_mask))
        n_normal = len(train_feat) - n_trending - n_choppy
        log.info(f"[V5_REGIME_WEIGHT] Regime sample weighting: "
                 f"trending(1.3x)={n_trending} choppy(0.7x)={n_choppy} normal(1.0x)={n_normal}")

    if regime_sample_weights is not None:
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=regime_sample_weights,
            num_samples=len(regime_sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
        log.info("[V5_REGIME_WEIGHT] Using WeightedRandomSampler for regime-conditional training")
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    if use_v6:
        v6_config = V6ForecasterConfig(
            input_dim=input_dim,
            seq_len=v6_seq_len,
            conv_channels=v6_conv_channels,
            n_conv_layers=v6_n_conv_layers,
            n_attn_layers=v6_attn_layers,
            n_attn_heads=v6_attn_heads,
            n_experts=v6_n_experts,
            expert_top_k=v6_expert_top_k,
            dropout=0.15,
            n_symbols=n_syms,
            symbol_embed_dim=symbol_embed_dim,
            feature_mask_ratio=v6_feature_mask_ratio,
            enable_aux_head=True,
            enable_confidence_head=True,
            n_barrier_presets=n_barrier,
            enable_regime_head=use_regime_head,
        )
        model = V6Forecaster(v6_config).to(device)
        model_config = v6_config
        log.info(f"[V6] V6Forecaster: {model.parameters_count():,} params | "
                 f"{v6_n_experts} experts (top-{v6_expert_top_k}) | "
                 f"seq_len={v6_seq_len} | conv={v6_conv_channels} | "
                 f"attn={v6_attn_layers}x{v6_attn_heads}h | mask={v6_feature_mask_ratio}")
    else:
        model_config = V5ForecasterConfig(
            input_dim=input_dim,
            hidden_dims=[512, 256, 128, 64],
            dropout=0.3,
            use_layer_norm=True,
            use_residual=True,
            n_barrier_presets=n_barrier,
            enable_regime_head=use_regime_head,
            n_symbols=n_syms,
            symbol_embed_dim=symbol_embed_dim,
        )
        model = V5Forecaster(model_config).to(device)
        log.info(f"[{vtag}] Model parameters: {model.parameters_count():,}")

    if warm_start_state_dict is not None:
        try:
            model.load_state_dict(warm_start_state_dict, strict=False)
            log.info(f"[{vtag}_WARM_START] Loaded previous fold model weights as initialization")
        except Exception as e:
            log.warning(f"[{vtag}_WARM_START] Failed to load previous weights: {e} — using random init")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs])

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    best_val_loss = float('inf')
    best_expectancy = float('-inf')
    best_expectancy_pct = 0.0
    best_promote_pf = float('-inf')
    best_promote_max_dd = 0.0
    best_sweep_row = None
    patience = 0
    max_patience = 25
    promote_patience = 0
    max_promote_patience = 25
    log.info(f"{ctag} promote_metric={promote_metric}")

    val_cand_mask = val_cand_mask_arr
    val_outcomes = val_outcomes_arr
    val_realized_r = val_realized_r_arr
    val_sym_ids = val_sym_ids_arr
    val_bars = total_val

    current_score_threshold = tpd_ctrl_cfg.score_threshold

    if use_v6:
        ckpt_model_config = {
            'input_dim': input_dim,
            'model_version': 'v6',
            'seq_len': v6_seq_len,
            'conv_channels': v6_conv_channels,
            'n_conv_layers': v6_n_conv_layers,
            'n_attn_layers': v6_attn_layers,
            'n_attn_heads': v6_attn_heads,
            'attn_ff_dim': model_config.attn_ff_dim,
            'n_experts': v6_n_experts,
            'expert_top_k': v6_expert_top_k,
            'expert_hidden_dims': model_config.expert_hidden_dims,
            'trunk_output_dim': model_config.trunk_output_dim,
            'dropout': model_config.dropout,
            'n_barrier_presets': model_config.n_barrier_presets,
            'enable_regime_head': model_config.enable_regime_head,
            'n_symbols': model_config.n_symbols,
            'symbol_embed_dim': model_config.symbol_embed_dim,
            'feature_mask_ratio': v6_feature_mask_ratio,
            'enable_aux_head': True,
            'enable_confidence_head': True,
        }
    else:
        ckpt_model_config = {
            'input_dim': input_dim,
            'hidden_dims': model_config.hidden_dims,
            'dropout': model_config.dropout,
            'n_barrier_presets': model_config.n_barrier_presets,
            'enable_regime_head': model_config.enable_regime_head,
            'n_symbols': model_config.n_symbols,
            'symbol_embed_dim': model_config.symbol_embed_dim,
        }
    ckpt_train_config = {
        'w_ret': w_ret, 'w_mfe': w_mfe, 'w_mae': w_mae,
        'w_action': w_action, 'w_barrier': w_barrier, 'w_regime': w_regime,
        'score_lambda': score_lambda, 'risk_proxy': risk_proxy,
        'hold_target': hold_target, 'mfe_min': mfe_min,
        'barrier_mode': barrier_mode,
        'target_tpd': target_tpd, 'target_tpd_tol': target_tpd_tol,
        'action_class_weights': action_class_weights.tolist(),
        'v5_config': {
            'candidate_engine': candidate_config.enabled,
            'barrier_presets': [p.get('label', 'default') for p in presets],
            'risk_controls': {
                'daily_loss_limit_r': risk_controls.daily_loss_limit_r,
                'max_concurrent_trades': risk_controls.max_concurrent_trades,
                'max_symbol_exposure': risk_controls.max_symbol_exposure,
            },
            'quality_gates': {
                'sigma_max': quality_gate_cfg.sigma_max,
                'mae_max': quality_gate_cfg.mae_max,
                'mu_R_min': quality_gate_cfg.mu_R_min,
                'p_trade_min': quality_gate_cfg.p_trade_min,
                'enable_calib': quality_gate_cfg.enable_calib,
            },
            'tpd_controller': {
                'target_tpd': tpd_ctrl_cfg.target_tpd,
                'tpd_tol': tpd_ctrl_cfg.tpd_tol,
                'thr_warmup_epochs': tpd_ctrl_cfg.thr_warmup_epochs,
                'thr_step_mult': tpd_ctrl_cfg.thr_step_mult,
                'mae_cap': tpd_ctrl_cfg.mae_cap,
                'min_threshold_floor': tpd_ctrl_cfg.min_threshold_floor,
            },
        },
    }

    if stage_a_epochs > 0:
        log.info(f"[{vtag}_STAGED] 3-phase training enabled: "
                 f"Phase A (epochs 1-{stage_a_epochs}): w_action×{stage_a_w_action_mult}, "
                 f"w_regime×{stage_a_w_regime_mult}, w_reg×{stage_a_w_reg_mult} | "
                 f"Phase B (epochs {stage_a_epochs+1}-{epochs}): normal weights")

    moe_collapse_counter = 0
    moe_base_weight = v6_moe_balance_weight
    moe_reinit_count = 0
    moe_last_reinit_epoch = -999
    MOE_MAX_REINITS = 3
    MOE_REINIT_COOLDOWN = 10

    for epoch in range(1, epochs + 1):
        use_candidates_this_epoch = candidate_config.enabled and epoch > cand_warmup_epochs
        if candidate_config.enabled and epoch == cand_warmup_epochs + 1:
            log.info(f"[{vtag}] Candidate warmup complete (epoch {epoch}), enabling candidate mask for sweep")

        in_stage_a = stage_a_epochs > 0 and epoch <= stage_a_epochs
        epoch_w_action = w_action * stage_a_w_action_mult if in_stage_a else w_action
        epoch_w_regime = w_regime * stage_a_w_regime_mult if in_stage_a else w_regime
        epoch_w_ret = w_ret * stage_a_w_reg_mult if in_stage_a else w_ret
        epoch_w_mfe = w_mfe * stage_a_w_reg_mult if in_stage_a else w_mfe
        epoch_w_mae = w_mae * stage_a_w_reg_mult if in_stage_a else w_mae

        if in_stage_a and epoch == 1:
            log.info(f"[{vtag}_STAGED] Phase A active: w_action={epoch_w_action:.2f} "
                     f"w_regime={epoch_w_regime:.2f} w_ret={epoch_w_ret:.2f}")
        if stage_a_epochs > 0 and epoch == stage_a_epochs + 1:
            log.info(f"[{vtag}_STAGED] Phase B starts: normal weights restored "
                     f"w_action={w_action:.2f} w_regime={w_regime:.2f} w_ret={w_ret:.2f}")

        model.train()
        train_losses = []
        loss_breakdown = {}

        for batch in train_loader:
            feat = batch['features'].to(device)
            sym_id = batch.get('symbol_id')
            if sym_id is not None:
                sym_id = sym_id.to(device)

            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

            outputs = model(feat, symbol_ids=sym_id)

            batch_sw = batch_gpu.get('sample_weight')

            if use_v6:
                loss, ld = compute_v6_loss(
                    outputs, batch_gpu,
                    w_ret=epoch_w_ret, w_mfe=epoch_w_mfe, w_mae=epoch_w_mae,
                    w_action=epoch_w_action, w_barrier=w_barrier, w_regime=epoch_w_regime,
                    w_moe_balance=v6_moe_balance_weight,
                    w_aux=v6_aux_weight, w_confidence=v6_confidence_weight,
                    barrier_mode=barrier_mode,
                    action_weights=action_weights_tensor,
                    epoch=epoch,
                    sample_weights=batch_sw,
                    mae_asym_weight=mae_asym_weight,
                )
            else:
                loss, ld = compute_v5_loss(
                    outputs, batch_gpu,
                    w_ret=epoch_w_ret, w_mfe=epoch_w_mfe, w_mae=epoch_w_mae,
                    w_action=epoch_w_action, w_barrier=w_barrier, w_regime=epoch_w_regime,
                    barrier_mode=barrier_mode,
                    action_weights=action_weights_tensor,
                    epoch=epoch,
                    sample_weights=batch_sw,
                    mae_asym_weight=mae_asym_weight,
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            for k, v in ld.items():
                loss_breakdown.setdefault(k, []).append(v)

        scheduler.step()
        avg_train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        all_val_outputs = {
            'ret_mu': [], 'ret_log_sigma': [], 'ret_sigma': [],
            'mae': [], 'mfe': [], 'action_logits': []
        }

        with torch.no_grad():
            for batch in val_loader:
                feat = batch['features'].to(device)
                sym_id = batch.get('symbol_id')
                if sym_id is not None:
                    sym_id = sym_id.to(device)

                batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}

                outputs = model(feat, symbol_ids=sym_id)
                if use_v6:
                    vloss, _ = compute_v6_loss(
                        outputs, batch_gpu,
                        w_ret=w_ret, w_mfe=w_mfe, w_mae=w_mae,
                        w_action=w_action, w_barrier=w_barrier, w_regime=w_regime,
                        w_moe_balance=v6_moe_balance_weight,
                        w_aux=v6_aux_weight, w_confidence=v6_confidence_weight,
                        barrier_mode=barrier_mode,
                        action_weights=action_weights_tensor,
                        epoch=epoch,
                        mae_asym_weight=mae_asym_weight,
                    )
                else:
                    vloss, _ = compute_v5_loss(
                        outputs, batch_gpu,
                        w_ret=w_ret, w_mfe=w_mfe, w_mae=w_mae,
                        w_action=w_action, w_barrier=w_barrier, w_regime=w_regime,
                        barrier_mode=barrier_mode,
                        action_weights=action_weights_tensor,
                        epoch=epoch,
                        mae_asym_weight=mae_asym_weight,
                    )
                val_losses.append(vloss.item())

                for k in all_val_outputs:
                    if k in outputs:
                        all_val_outputs[k].append(outputs[k].detach().cpu())

        avg_val_loss = np.mean(val_losses)

        lb_str = " | ".join(f"{k}={np.mean(v):.4f}" for k, v in loss_breakdown.items() if v)
        current_lr = optimizer.param_groups[0]['lr']

        action_logits_cat = torch.cat(all_val_outputs['action_logits'], dim=0).numpy()
        action_preds = np.argmax(action_logits_cat, axis=1)
        n_pred = min(len(action_preds), len(val_action_arr))
        action_acc = np.mean(action_preds[:n_pred] == val_action_arr[:n_pred])

        pred_hold = np.sum(action_preds[:n_pred] == 0)
        pred_long = np.sum(action_preds[:n_pred] == 1)
        pred_short = np.sum(action_preds[:n_pred] == 2)

        tag = "[V6]" if use_v6 else "[V5]"
        log.info(f"{tag} Epoch {epoch:03d}/{epochs} | train={avg_train_loss:.4f} val={avg_val_loss:.4f} "
                 f"lr={current_lr:.2e} act_acc={action_acc:.3f} "
                 f"pred[H/L/S]={pred_hold}/{pred_long}/{pred_short} | {lb_str}")

        if use_v6 and hasattr(model, 'get_expert_usage'):
            expert_usage = model.get_expert_usage()
            if expert_usage:
                usage_vals = list(expert_usage.values())
                usage_str = " ".join(f"{k}={v:.1f}%" for k, v in expert_usage.items())
                gate_entropy = -sum((v/100) * math.log(v/100 + 1e-10) for v in usage_vals) / math.log(len(usage_vals))
                log.info(f"[V6_MoE] Expert usage: {usage_str} | entropy={gate_entropy:.3f} (1.0=perfect balance)")

                any_collapsed = any(v < 5.0 for v in usage_vals)
                if any_collapsed:
                    moe_collapse_counter += 1
                    if moe_collapse_counter >= 3:
                        v6_moe_balance_weight = moe_base_weight * 20.0
                        if hasattr(model, 'moe'):
                            model.moe._entropy_bonus = True
                        log.warning(f"[V6_MoE] COLLAPSE DETECTED for {moe_collapse_counter} consecutive epochs — "
                                    f"boosting w_moe_balance to {v6_moe_balance_weight:.3f} (20x base) + entropy bonus ON")

                    can_reinit = (moe_reinit_count < MOE_MAX_REINITS and
                                  (epoch - moe_last_reinit_epoch) >= MOE_REINIT_COOLDOWN)
                    if moe_collapse_counter >= 5 and can_reinit and hasattr(model, 'moe'):
                        dead_indices = [i for i, v in enumerate(usage_vals) if v < 5.0]
                        if dead_indices:
                            model.moe.reinit_dead_experts(dead_indices)
                            reinit_params = set()
                            for di in dead_indices:
                                for p in model.moe.experts[di].parameters():
                                    reinit_params.add(id(p))
                            reinit_params.add(id(model.moe.gate.weight))
                            reinit_params.add(id(model.moe.gate_noise.weight))
                            if model.moe.gate.bias is not None:
                                reinit_params.add(id(model.moe.gate.bias))
                            for group in optimizer.param_groups:
                                for gp in group['params']:
                                    if id(gp) in reinit_params and gp in optimizer.state:
                                        for sk, sv in optimizer.state[gp].items():
                                            if isinstance(sv, torch.Tensor):
                                                optimizer.state[gp][sk] = torch.zeros_like(sv)
                            moe_reinit_count += 1
                            moe_last_reinit_epoch = epoch
                            moe_collapse_counter = 0
                            log.warning(f"[V6_MoE] REINIT #{moe_reinit_count}/{MOE_MAX_REINITS}: "
                                        f"dead experts {dead_indices} — full MLP+gate cloned from strongest alive expert + noise + optimizer state reset")
                else:
                    if moe_collapse_counter >= 3:
                        v6_moe_balance_weight = moe_base_weight
                        if hasattr(model, 'moe'):
                            model.moe._entropy_bonus = False
                        log.info(f"[V6_MoE] Collapse recovered — restoring w_moe_balance to {moe_base_weight:.3f}, entropy bonus OFF")
                    moe_collapse_counter = 0

        do_sweep = (epoch % 5 == 0) or (epoch == epochs) or (epoch <= 3)
        if do_sweep:
            concat_outputs = {}
            for k in all_val_outputs:
                if all_val_outputs[k]:
                    concat_outputs[k] = torch.cat(all_val_outputs[k], dim=0)

            arrays = _extract_v5_arrays(concat_outputs)

            quality_mask, qual_diag = v5_quality_mask(arrays, quality_gate_cfg, epoch=epoch)

            scores, sides, score_diag = compute_v5_scores(
                None, horizon_bars=horizon,
                score_lambda=tpd_ctrl_cfg.score_lambda,
                risk_proxy=risk_proxy,
                mae_cap=tpd_ctrl_cfg.mae_cap,
                _arrays=arrays,
                side_mode=tpd_ctrl_cfg.side_mode,
                rr_weight=tpd_ctrl_cfg.rr_weight,
                sigma_discount=sigma_discount,
                min_p_side=min_p_side,
                min_p_short=min_p_short,
                side_aware_scoring=side_aware_scoring,
                slippage_bps=slippage_base_bps,
            )

            log.info(f"[{vtag}_SCORE_DIAG] mu_R: mean={score_diag['mu_R_mean']:.4f} std={score_diag['mu_R_std']:.4f} | "
                     f"mae_R: mean={score_diag['mae_R_mean']:.3f} mfe_R: mean={score_diag['mfe_R_mean']:.3f} | "
                     f"p_long={score_diag['p_long_mean']:.3f} p_short={score_diag['p_short_mean']:.3f} | "
                     f"edge_L={score_diag['edge_long_mean']:.4f} edge_S={score_diag['edge_short_mean']:.4f} "
                     f"penalty={score_diag['penalty_mean']:.4f} | "
                     f"score: mean={score_diag['score_mean']:.4f} p50={score_diag['score_p50']:.4f} "
                     f"p90={score_diag['score_p90']:.4f} %pos={score_diag['score_pct_positive']:.1f}%")

            mu_arr = arrays['mu_R']
            sig_arr = arrays['sigma'] if arrays['sigma'] is not None else np.zeros_like(mu_arr)
            mae_arr_diag = arrays['mae']
            pt_arr = arrays['p_trade']
            log.info(f"[{vtag}_OUTPUT_DIST] mu_R: p5={np.percentile(mu_arr,5):.4f} p50={np.percentile(mu_arr,50):.4f} "
                     f"p95={np.percentile(mu_arr,95):.4f} | "
                     f"sigma: p5={np.percentile(sig_arr,5):.4f} p50={np.percentile(sig_arr,50):.4f} "
                     f"p95={np.percentile(sig_arr,95):.4f} | "
                     f"mae: p5={np.percentile(mae_arr_diag,5):.3f} p50={np.percentile(mae_arr_diag,50):.3f} "
                     f"p95={np.percentile(mae_arr_diag,95):.3f} | "
                     f"p_trade: p5={np.percentile(pt_arr,5):.3f} p50={np.percentile(pt_arr,50):.3f} "
                     f"p95={np.percentile(pt_arr,95):.3f}")

            sweep_cand_mask = val_cand_mask if use_candidates_this_epoch else None

            current_score_threshold, tpd_trades, tpd_val, tpd_action = _tpd_controller_step(
                scores, quality_mask, sweep_cand_mask,
                current_score_threshold, epoch, val_bars,
                tpd_ctrl_cfg, cooldown=cooldown,
            )

            (sweep_results, sweep_label, sweep_expect, sweep_pct,
             sweep_pf, sweep_max_dd, sweep_tpd, sweep_threshold) = _run_v5_sweep(
                scores, sides, val_outcomes, val_realized_r,
                val_bars, epoch, tp_mult, sl_mult,
                target_tpd=target_tpd, target_tpd_tol=target_tpd_tol,
                candidate_mask=sweep_cand_mask,
                risk_controls=risk_controls,
                symbol_ids=val_sym_ids,
                horizon_bars=horizon,
                quality_mask=quality_mask,
                score_threshold=current_score_threshold,
                r_long=val_r_long_arr, r_short=val_r_short_arr,
                out_long=val_out_long_arr, out_short=val_out_short_arr,
                close_prices=val_close_arr,
                ema200_regime_gate=ema200_regime_gate,
                timestamps=val_timestamps_arr,
                weekly_loss_cap=weekly_loss_cap,
                cooldown=cooldown,
            )

            log.info(f"[{vtag}_EPOCH_TRADING] epoch={epoch:03d} | expect={sweep_expect:+.4f} PF={sweep_pf:.2f} "
                     f"maxDD={sweep_max_dd:.2f} T/day={sweep_tpd:.1f} thr={sweep_threshold:.4f} "
                     f"best_at={sweep_label}")

            if _active_pusher:
                lb_dict = {k: float(np.mean(v)) for k, v in loss_breakdown.items() if v}
                _active_pusher.epoch_update(
                    fold_num=_active_fold_num, epoch=epoch, total_epochs=epochs,
                    train_loss=float(avg_train_loss), val_loss=float(avg_val_loss),
                    loss_breakdown=lb_dict, action_accuracy=float(action_acc),
                    learning_rate=float(current_lr), total_folds=_active_total_folds,
                    sweep_metrics={
                        "expectancy": float(sweep_expect),
                        "profit_factor": float(sweep_pf),
                        "max_drawdown": float(sweep_max_dd),
                        "trades_per_day": float(sweep_tpd),
                        "threshold": float(sweep_threshold),
                        "win_rate": float(sweep_pct),
                        "score_diag": {k: float(v) for k, v in score_diag.items() if isinstance(v, (int, float))},
                    },
                )

            best_per_sym_thresholds = None
            if per_symbol_threshold and symbols and len(symbols) > 1:
                if per_side_threshold:
                    log.info("[V5_PER_SIDE_THR] Running per-side threshold sweep (LONG then SHORT)...")
                    long_mask_sweep = sides == 1
                    short_mask_sweep = sides == -1
                    scores_long_only = scores.copy()
                    scores_long_only[~long_mask_sweep] = -np.inf
                    scores_short_only = scores.copy()
                    scores_short_only[~short_mask_sweep] = -np.inf
                    _MIN_SIDE_TRADES = 5
                    log.info("[V5_PER_SIDE_THR] === LONG threshold sweep ===")
                    per_sym_long, _ = _run_per_symbol_sweep(
                        scores_long_only, sides, val_outcomes, val_realized_r,
                        val_bars, val_sym_ids, symbols, sweep_threshold,
                        r_long=val_r_long_arr, r_short=val_r_short_arr,
                        out_long=val_out_long_arr, out_short=val_out_short_arr,
                        quality_mask=quality_mask, candidate_mask=sweep_cand_mask,
                        min_trades_per_symbol=_MIN_SIDE_TRADES,
                        cooldown=cooldown,
                    )
                    log.info("[V5_PER_SIDE_THR] === SHORT threshold sweep ===")
                    per_sym_short, _ = _run_per_symbol_sweep(
                        scores_short_only, sides, val_outcomes, val_realized_r,
                        val_bars, val_sym_ids, symbols, sweep_threshold,
                        r_long=val_r_long_arr, r_short=val_r_short_arr,
                        out_long=val_out_long_arr, out_short=val_out_short_arr,
                        quality_mask=quality_mask, candidate_mask=sweep_cand_mask,
                        min_trades_per_symbol=_MIN_SIDE_TRADES,
                        cooldown=cooldown,
                    )
                    best_per_sym_thresholds = {}
                    all_sym_ids_sweep = set(per_sym_long.keys()) | set(per_sym_short.keys())
                    for _sid in all_sym_ids_sweep:
                        best_per_sym_thresholds[_sid] = {
                            'long': per_sym_long.get(_sid, sweep_threshold * 3.0),
                            'short': per_sym_short.get(_sid, sweep_threshold * 3.0),
                        }
                    no_edge_syms = []
                    log.info(f"[V5_PER_SIDE_THR] Per-side thresholds computed for {len(best_per_sym_thresholds)} symbols")
                    for _sid in sorted(best_per_sym_thresholds.keys()):
                        _sym = symbols[_sid] if _sid < len(symbols) else f"sym_{_sid}"
                        _lt = best_per_sym_thresholds[_sid]['long']
                        _st = best_per_sym_thresholds[_sid]['short']
                        log.info(f"  {_sym:>12}: LONG={_lt:.4f} SHORT={_st:.4f}")
                else:
                    best_per_sym_thresholds, no_edge_syms = _run_per_symbol_sweep(
                        scores, sides, val_outcomes, val_realized_r,
                        val_bars, val_sym_ids, symbols, sweep_threshold,
                        r_long=val_r_long_arr, r_short=val_r_short_arr,
                        out_long=val_out_long_arr, out_short=val_out_short_arr,
                        quality_mask=quality_mask, candidate_mask=sweep_cand_mask,
                        min_trades_per_symbol=max(5, min_trades // 2),
                        cooldown=cooldown,
                    )

            if quality_gate_cfg.enable_calib:
                val_p_trade = arrays['p_trade'][:len(val_realized_r)]
                compute_v5_calibration(val_p_trade, val_realized_r)

            ckpt_v5_config = ckpt_train_config['v5_config']
            ckpt_v5_config['tpd_controller']['current_threshold'] = current_score_threshold

            promote_better = False
            if promote_metric == 'expectancy':
                promote_better = sweep_expect > best_expectancy
            elif promote_metric == 'pf':
                promote_better = sweep_pf > best_promote_pf
            else:
                promote_better = avg_val_loss < best_val_loss

            if promote_better and sweep_expect > -999:
                best_expectancy = sweep_expect
                best_expectancy_pct = sweep_pct
                best_promote_pf = sweep_pf
                best_promote_max_dd = sweep_max_dd
                best_sweep_row = next(
                    (m for m in sweep_results if m['label'] == sweep_label), None
                )
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_config': ckpt_model_config,
                    'train_config': ckpt_train_config,
                    'feature_columns': features_df_columns,
                    'n_features': input_dim,
                    'feature_version': V5_FEATURE_VERSION,
                    'model_type': 'v6_forecaster' if use_v6 else 'v5_forecaster',
                    'scaler_center': scaler.center_,
                    'scaler_scale': scaler.scale_,
                    'barrier_config': {
                        'tp_mult': tp_mult, 'sl_mult': sl_mult,
                        'horizon': horizon,
                        'presets': [p.get('label', 'default') for p in presets],
                    },
                    'best_expectancy': best_expectancy,
                    'best_expectancy_pct': best_expectancy_pct,
                    'best_pf': best_promote_pf,
                    'best_max_dd': best_promote_max_dd,
                    'promote_metric': promote_metric,
                    'symbol_map': {s: i for i, s in enumerate(symbols)},
                    'trained_at': datetime.now().isoformat(),
                    'per_symbol_thresholds': best_per_sym_thresholds,
                    'per_symbol_scalers': {
                        sym: {'center_': s.center_.tolist(), 'scale_': s.scale_.tolist()}
                        for sym, s in per_symbol_scalers.items() if s is not None
                    } if per_symbol_scalers else None,
                }, checkpoint_dir / "best_v5_expectancy.pt")
                log.info(f"[{vtag}_CKPT] New best ({promote_metric}): expect={best_expectancy:.4f} "
                         f"PF={best_promote_pf:.2f} maxDD={best_promote_max_dd:.2f} at {sweep_label}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            ckpt_loss_data = {
                'model_state_dict': model.state_dict(),
                'model_config': ckpt_model_config,
                'train_config': ckpt_train_config,
                'feature_columns': features_df_columns,
                'n_features': input_dim,
                'feature_version': V5_FEATURE_VERSION,
                'model_type': 'v6_forecaster' if use_v6 else 'v5_forecaster',
                'scaler_center': scaler.center_,
                'scaler_scale': scaler.scale_,
                'symbol_map': {s: i for i, s in enumerate(symbols)},
                'trained_at': datetime.now().isoformat(),
            }
            if per_symbol_scalers:
                ckpt_loss_data['per_symbol_scalers'] = {
                    sym: {'center_': s.center_.tolist(), 'scale_': s.scale_.tolist()}
                    for sym, s in per_symbol_scalers.items() if s is not None
                }
            torch.save(ckpt_loss_data, checkpoint_dir / "best_v5_loss.pt")
            log.info(f"[{vtag}_CKPT] New best val_loss={best_val_loss:.4f}")
        else:
            patience += 1

        if promote_metric == 'val_loss':
            if patience >= max_patience:
                log.info(f"[{vtag}] Early stopping at epoch {epoch} (val_loss patience={max_patience})")
                break
        else:
            if do_sweep and not promote_better:
                promote_patience += 1
            elif do_sweep and promote_better:
                promote_patience = 0
            if promote_patience >= max_promote_patience:
                log.info(f"[{vtag}] Early stopping at epoch {epoch} ({promote_metric} patience={max_promote_patience})")
                break

    log.info("=" * 60)
    log.info(f"[{vtag}] Training complete. Best expectancy={best_expectancy:.4f} best_loss={best_val_loss:.4f}")
    log.info(f"[{vtag}] Final score_threshold={current_score_threshold}")
    log.info("=" * 60)

    if finetune_months > 0 and len(train_timestamps) > 0 and not use_v6:
        best_ckpt_ft = checkpoint_dir / "best_v5_expectancy.pt"
        if not best_ckpt_ft.exists():
            best_ckpt_ft = checkpoint_dir / "best_v5_loss.pt"
        if best_ckpt_ft.exists():
            ft_ckpt = torch.load(best_ckpt_ft, map_location=device, weights_only=False)
            model.load_state_dict(ft_ckpt['model_state_dict'])
            ts_max_ft = train_timestamps.max()
            ft_cutoff_ms = ts_max_ft - finetune_months * 30.44 * 24 * 3600 * 1000
            ft_mask = train_timestamps >= ft_cutoff_ms
            ft_count = int(ft_mask.sum())
            if ft_count >= 100:
                ft_lr = lr * finetune_lr_mult
                log.info(f"[{vtag}_FINETUNE] Fine-tuning on last {finetune_months} months: "
                         f"{ft_count}/{len(train_timestamps)} samples, LR={ft_lr:.2e}, epochs={finetune_epochs}")
                ft_ds = V5Dataset(
                    train_feat[ft_mask], train_ret_R[ft_mask], train_mfe_R[ft_mask],
                    train_mae_R[ft_mask], train_vol_h[ft_mask], train_action[ft_mask],
                    train_valid[ft_mask], train_sym_ids[ft_mask],
                    train_barrier_oracle[ft_mask], train_barrier_soft[ft_mask],
                )
                ft_loader = DataLoader(ft_ds, batch_size=batch_size, shuffle=True, drop_last=False)
                ft_optimizer = torch.optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=1e-4)
                for ft_ep in range(1, finetune_epochs + 1):
                    model.train()
                    ft_losses = []
                    for batch in ft_loader:
                        feat = batch['features'].to(device)
                        sym_id = batch.get('symbol_id')
                        if sym_id is not None:
                            sym_id = sym_id.to(device)
                        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                     for k, v in batch.items()}
                        outputs = model(feat, symbol_ids=sym_id)
                        loss, _ = compute_v5_loss(
                            outputs, batch_gpu,
                            w_ret=w_ret, w_mfe=w_mfe, w_mae=w_mae,
                            w_action=w_action, w_barrier=w_barrier, w_regime=w_regime,
                            barrier_mode=barrier_mode,
                            action_weights=action_weights_tensor,
                            epoch=epochs + ft_ep,
                            mae_asym_weight=mae_asym_weight,
                        )
                        ft_optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        ft_optimizer.step()
                        ft_losses.append(loss.item())
                    avg_ft_loss = np.mean(ft_losses)
                    log.info(f"[{vtag}_FINETUNE] Epoch {ft_ep}/{finetune_epochs} loss={avg_ft_loss:.4f}")
                ft_ckpt['model_state_dict'] = model.state_dict()
                torch.save(ft_ckpt, best_ckpt_ft)
                log.info(f"[{vtag}_FINETUNE] Saved fine-tuned model to {best_ckpt_ft}")
            else:
                log.warning(f"[{vtag}_FINETUNE] Only {ft_count} samples in last {finetune_months} months — skipping (need >=100)")
        else:
            log.warning(f"[{vtag}_FINETUNE] No checkpoint found for fine-tuning — skipping")

    fitted_temperature = 1.0
    if temp_scale:
        log.info(f"[{vtag}_TEMP_SCALE] Fitting temperature scaling on validation set...")
        best_ckpt_for_temp = checkpoint_dir / "best_v5_expectancy.pt"
        if not best_ckpt_for_temp.exists():
            best_ckpt_for_temp = checkpoint_dir / "best_v5_loss.pt"
        if best_ckpt_for_temp.exists():
            temp_ckpt = torch.load(best_ckpt_for_temp, map_location=device, weights_only=False)
            model.load_state_dict(temp_ckpt['model_state_dict'])
            model.eval()
            all_logits = []
            with torch.no_grad():
                for batch in val_loader:
                    feat = batch['features'].to(device)
                    sym_id = batch.get('symbol_id')
                    if sym_id is not None:
                        sym_id = sym_id.to(device)
                    outputs = model(feat, symbol_ids=sym_id)
                    all_logits.append(outputs['action_logits'].cpu())
            logits_cat = torch.cat(all_logits, dim=0).numpy()
            n_valid = min(len(logits_cat), len(val_action_arr))
            fitted_temperature, ece_before, ece_after = fit_temperature_scaling(
                logits_cat[:n_valid], val_action_arr[:n_valid]
            )
            temp_ckpt['temperature'] = fitted_temperature
            temp_ckpt['ece_before_temp'] = ece_before
            temp_ckpt['ece_after_temp'] = ece_after
            torch.save(temp_ckpt, best_ckpt_for_temp)
            log.info(f"[{vtag}_TEMP_SCALE] Saved temperature={fitted_temperature:.4f} to checkpoint")

    fwd_report = None

    if run_forward_test and (test_start_date or train_end_date):
        log.info("=" * 60)
        log.info("  V5 FORWARD TEST (frozen decision layer)")
        log.info("=" * 60)

        best_ckpt_path = checkpoint_dir / "best_v5_expectancy.pt"
        if not best_ckpt_path.exists():
            best_ckpt_path = checkpoint_dir / "best_v5_loss.pt"

        if best_ckpt_path.exists():
            ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            log.info(f"[V5_FWD] Loaded best checkpoint from {best_ckpt_path}")

            ckpt_threshold = current_score_threshold
            if 'train_config' in ckpt:
                tc = ckpt['train_config']
                v5c = tc.get('v5_config', {})
                tpd_c = v5c.get('tpd_controller', {})
                if 'current_threshold' in tpd_c and tpd_c['current_threshold'] is not None:
                    ckpt_threshold = tpd_c['current_threshold']
            if ckpt_threshold is None:
                ckpt_threshold = 0.02
            ckpt_floor = min_threshold if min_threshold is not None else 0.02
            if ckpt_threshold < ckpt_floor:
                log.info(f"[V5_FWD] Clamping calibrated threshold {ckpt_threshold:.4f} → floor {ckpt_floor:.4f}")
                ckpt_threshold = ckpt_floor

            if wf_threshold_override is not None:
                log.info(f"[V5_FWD] Walk-forward threshold override: sweep={ckpt_threshold:.4f} → blended={wf_threshold_override:.4f}")
                ckpt_threshold = wf_threshold_override

            ckpt_temperature = ckpt.get('temperature', fitted_temperature)
            if ckpt_temperature != 1.0:
                log.info(f"[V5_FWD] Using temperature={ckpt_temperature:.4f} from checkpoint")

            ckpt_per_sym_thr = None
            if per_symbol_threshold:
                ckpt_per_sym_thr = ckpt.get('per_symbol_thresholds', None)
                if ckpt_per_sym_thr:
                    log.info(f"[V5_FWD] Per-symbol thresholds loaded from checkpoint: "
                             f"{len(ckpt_per_sym_thr)} symbols")
                    _first_ckpt_val = next(iter(ckpt_per_sym_thr.values()), None)
                    _is_per_side = isinstance(_first_ckpt_val, dict)
                    for sym_id_k, sym_thr_v in sorted(ckpt_per_sym_thr.items(), key=lambda x: int(x[0])):
                        sym_name_k = symbols[int(sym_id_k)] if symbols and int(sym_id_k) < len(symbols) else f"sym_{sym_id_k}"
                        if _is_per_side:
                            lt = sym_thr_v.get('long', float('inf'))
                            st = sym_thr_v.get('short', float('inf'))
                            log.info(f"  {sym_name_k}: LONG={lt:.4f} SHORT={st:.4f}")
                        else:
                            thr_str = f"{sym_thr_v:.4f}" if np.isfinite(sym_thr_v) else "inf (NO EDGE)"
                            log.info(f"  {sym_name_k}: threshold={thr_str}")
                else:
                    log.warning("[V5_FWD] Per-symbol threshold enabled but not found in checkpoint — using global threshold")

            fwd_config = V5ForwardTestConfig(
                score_threshold=ckpt_threshold,
                score_lambda=tpd_ctrl_cfg.score_lambda,
                mae_cap=tpd_ctrl_cfg.mae_cap,
                risk_proxy=risk_proxy,
                tp_mult=tp_mult,
                sl_mult=sl_mult,
                horizon=horizon,
                cooldown=cooldown,
                quality_gate_cfg=quality_gate_cfg,
                side_mode=tpd_ctrl_cfg.side_mode,
                rr_weight=tpd_ctrl_cfg.rr_weight,
                weekly_loss_cap=weekly_loss_cap,
                warmup_skip_bars=warmup_skip_bars,
                corr_block=corr_block,
                corr_window_days=corr_window_days,
                corr_thresh=corr_thresh,
                corr_same_side_only=corr_same_side_only,
                corr_log_matrix=corr_log_matrix,
                corr_max_block=corr_max_block,
                symbols_list=symbols if symbols else None,
                adaptive_sizing=adaptive_sizing,
                kelly_fraction=kelly_fraction,
                max_size_mult=max_size_mult,
                min_size_mult=min_size_mult,
                regime_scaling=regime_scaling,
                regime_bull_mult=regime_bull_mult,
                regime_bear_mult=regime_bear_mult,
                regime_lookback=regime_lookback,
                daily_loss_cap=daily_loss_cap,
                trailing_equity_stop=trailing_equity_stop,
                per_symbol_daily_r_budget=per_symbol_daily_r_budget,
                min_threshold=min_threshold,
                max_threshold=max_threshold,
                min_threshold_pct=min_threshold_pct,
                max_trades_per_day=max_trades_per_day,
                trailing_sl=trailing_sl,
                trail_activation=trail_activation,
                trail_distance=trail_distance,
                allow_runner=allow_runner,
                conviction_sizing=conviction_sizing,
                conviction_tier_top_pct=conviction_tier_top_pct,
                conviction_tier_top_mult=conviction_tier_top_mult,
                conviction_tier_high_pct=conviction_tier_high_pct,
                conviction_tier_high_mult=conviction_tier_high_mult,
                conviction_confidence_threshold=conviction_confidence_threshold,
                conviction_confidence_boost=conviction_confidence_boost,
                temperature=ckpt_temperature,
                adx_gate=adx_gate,
                adx_period=adx_period,
                adx_min=adx_min,
                adx_exception_top_pct=adx_exception_top_pct,
                ultra_conviction=ultra_conviction,
                ultra_risk_cap=ultra_risk_cap,
                ultra_score_pct=ultra_score_pct,
                ultra_adx_min=ultra_adx_min,
                ultra_edge_min=ultra_edge_min,
                ultra_dd_max=ultra_dd_max,
                ultra_max_per_day=ultra_max_per_day,
                ultra_mult=ultra_mult,
                ddt_enable=ddt_enable,
                ddt_lookback_trades=ddt_lookback_trades,
                ddt_bad_rollr=ddt_bad_rollr,
                ddt_thr_k=ddt_thr_k,
                ddt_thr_min=ddt_thr_min,
                ddt_thr_max=ddt_thr_max,
                ddt_size_k=ddt_size_k,
                ddt_min_size_mult=ddt_min_size_mult,
                ddt_alpha_down=ddt_alpha_down,
                ddt_alpha_up=ddt_alpha_up,
                ddt_warmup_trades=ddt_warmup_trades,
                multi_regime=multi_regime,
                regime_adx_trending=regime_adx_trending,
                regime_adx_choppy=regime_adx_choppy,
                regime_atr_high_vol=regime_atr_high_vol,
                regime_atr_low_vol=regime_atr_low_vol,
                regime_atr_window=regime_atr_window,
                regime_ema_slope_window=regime_ema_slope_window,
                regime_ema_buffer=regime_ema_buffer,
                edge_first=edge_first,
                edge_min=edge_min,
                edge_pct_floor=edge_pct_floor,
                edge_topn_per_day=edge_topn_per_day,
                regime_side_map=regime_side_map,
                regime_soft=regime_soft,
                regime_disagree_mult=regime_disagree_mult,
                regime_none_mult=regime_none_mult,
                per_symbol_soft_kill=per_symbol_soft_kill,
                edge_topn_soft=edge_topn_soft,
                edge_topn_decay=edge_topn_decay,
                size_floor=size_floor,
                soft_gate_floor=soft_gate_floor,
                weekly_cap_dynamic=weekly_cap_dynamic,
                weekly_cap_scale=weekly_cap_scale,
                quality_gate_enabled=quality_gate_enabled,
                quality_gate_window=quality_gate_window,
                direction_balance_cap=direction_balance_cap,
                direction_balance_threshold=direction_balance_threshold,
                head_disagreement_gate=head_disagreement_gate,
                sigma_discount=sigma_discount,
                min_p_side=min_p_side,
                min_p_short=min_p_short,
                side_aware_scoring=side_aware_scoring,
                per_symbol_cooldown=per_symbol_cooldown,
                slippage_base_bps=slippage_base_bps,
                mu_debias=mu_debias,
                mu_debias_alpha=mu_debias_alpha,
                min_trades=min_trades,
                per_symbol_r_kill=per_symbol_r_kill,
                per_symbol_thresholds=ckpt_per_sym_thr,
                ema200_soft_mult=ema200_soft_mult,
                per_side_threshold=per_side_threshold,
            )

            train_ref_arrays = _build_train_ref_arrays(
                model, device, train_feat, train_sym_ids,
                fwd_config, total_train,
                use_v6=use_v6, v6_seq_len=v6_seq_len,
            )

            fwd_report = run_v5_forward_test(
                model=model,
                device=device,
                test_features=val_feat,
                test_outcomes=val_outcomes_arr,
                test_realized_r=val_realized_r_arr,
                test_sym_ids=val_sym_ids_arr,
                test_cand_mask=val_cand_mask_arr,
                test_valid=val_valid,
                test_bars=total_val,
                config=fwd_config,
                test_start_date=test_start_date or train_end_date,
                test_end_date=test_end_date,
                test_timestamps=val_timestamps_arr,
                r_long=val_r_long_arr,
                r_short=val_r_short_arr,
                out_long=val_out_long_arr,
                out_short=val_out_short_arr,
                close_prices=val_close_arr,
                ema200_regime_gate=ema200_regime_gate,
                high_prices=val_high_arr,
                low_prices=val_low_arr,
                train_ref_arrays=train_ref_arrays,
                use_v6=use_v6, v6_seq_len=v6_seq_len,
            )

            report_path = checkpoint_dir / "v5_forward_report.json"
            import json
            serializable_report = {k: v for k, v in fwd_report.items()
                                   if not k.startswith('_')}
            with open(report_path, 'w') as f:
                json.dump(serializable_report, f, indent=2, default=str)
            log.info(f"[V5_FWD] Report saved to {report_path}")

            if '_corr_tracker' in fwd_report:
                from train.v5_correlation import (
                    build_fold_corr_report, log_corr_report, save_corr_report,
                    compute_overlap_ratio
                )
                corr_report = build_fold_corr_report(
                    fold_id=fold_id,
                    window_train=train_end_date or "?",
                    window_test=f"{test_start_date or '?'}→{test_end_date or '?'}",
                    corr_tracker=fwd_report['_corr_tracker'],
                    overlap_ratio=fwd_report.get('_overlap_ratio', 0.0),
                    blocker=fwd_report.get('_corr_blocker'),
                )
                if corr_log_matrix:
                    log_corr_report(corr_report, fold_id=fold_id)
                save_corr_report(corr_report, fold_id=fold_id,
                                 output_dir=str(checkpoint_dir))
        else:
            log.warning("[V5_FWD] No checkpoint found, skipping forward test")

    if run_diagnostics:
        from train.v5_diagnostics import run_all_diagnostics

        sweep_metrics = None
        if best_sweep_row is not None:
            sweep_metrics = {
                'win_rate': best_sweep_row.get('winrate', 0),
                'expectancy_r': best_sweep_row.get('expect', 0),
                'profit_factor': best_sweep_row.get('pf', 0),
                'sharpe': best_sweep_row.get('sharpe', 0),
            }

        n_model_trades = fwd_report.get('total_trades', 100) if fwd_report else 100

        diag_results = run_all_diagnostics(
            model=model, device=device,
            val_loader=val_loader, val_action_arr=val_action_arr,
            val_valid=val_valid,
            val_feat=val_feat, val_ret_R=val_ret_R,
            feature_names=features_df_columns,
            val_outcomes=val_outcomes_arr,
            val_realized_r=val_realized_r_arr,
            val_timestamps=val_timestamps_arr,
            test_bars=total_val, horizon=horizon,
            sweep_metrics=sweep_metrics,
            forward_report=fwd_report,
            n_trades_model=n_model_trades,
        )

        diag_path = checkpoint_dir / "v5_diagnostics_report.json"
        import json
        def _make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return str(obj)

        with open(diag_path, 'w') as f:
            json.dump(diag_results, f, indent=2, default=_make_serializable)
        log.info(f"[V5_DIAG] Full diagnostics report saved to {diag_path}")

    if feature_report:
        log.info("=" * 60)
        log.info("  V5 FEATURE IMPORTANCE REPORT")
        log.info("=" * 60)

        best_ckpt_for_report = checkpoint_dir / "best_v5_expectancy.pt"
        if not best_ckpt_for_report.exists():
            best_ckpt_for_report = checkpoint_dir / "best_v5_loss.pt"
        if best_ckpt_for_report.exists():
            report_ckpt = torch.load(best_ckpt_for_report, map_location=device, weights_only=False)
            model.load_state_dict(report_ckpt['model_state_dict'])
            log.info(f"[V5_FEAT_REPORT] Loaded best checkpoint for feature report")

        compute_feature_importance_report(
            model=model,
            device=device,
            val_feat=val_feat,
            val_ret_R=val_ret_R,
            val_valid=val_valid,
            val_sym_ids=val_sym_ids_arr,
            feature_names=features_df_columns,
            checkpoint_dir=checkpoint_dir,
        )
