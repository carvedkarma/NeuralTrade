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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from dataclasses import dataclass, field

log = logging.getLogger("QuickStart")

V5_FEATURE_VERSION = "v5.0.1_forecaster"


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


class V5Dataset(Dataset):
    def __init__(self, features, ret_R, mfe_R, mae_R, vol_h, action_labels,
                 valid_mask, symbol_ids=None, barrier_labels=None, barrier_soft=None):
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
        return item


def compute_v5_loss(outputs, batch, w_ret=1.0, w_mfe=0.25, w_mae=0.25,
                    w_action=2.0, w_barrier=0.25, w_regime=0.1,
                    barrier_mode='fixed', action_weights=None, epoch=0):
    """Compute v5 composite loss with class-balanced action CE.

    Uses clamped Gaussian NLL to prevent log(sigma) term from dominating.
    Three-stage schedule: epochs 0-5 action-heavy, 6-15 balanced, 16+ full.
    """
    valid = batch['valid']
    if valid.sum() == 0:
        return torch.tensor(0.0, device=outputs['ret_mu'].device, requires_grad=True), {}

    ret_mu = outputs['ret_mu'][valid].squeeze(-1)
    ret_log_sigma = outputs['ret_log_sigma'][valid].squeeze(-1)
    ret_true = batch['ret_R'][valid]

    sigma = torch.exp(ret_log_sigma).clamp(min=0.01, max=5.0)
    squared_error = ((ret_true - ret_mu) / (sigma + 1e-8)) ** 2
    log_term = torch.log(sigma + 1e-8)
    nll = log_term + 0.5 * squared_error
    L_ret = nll.mean()

    huber = nn.SmoothL1Loss()
    mfe_pred = outputs['mfe'][valid].squeeze(-1)
    mfe_true = batch['mfe_R'][valid]
    L_mfe = huber(mfe_pred, mfe_true)

    mae_pred = outputs['mae'][valid].squeeze(-1)
    mae_true = batch['mae_R'][valid]
    L_mae = huber(mae_pred, mae_true)

    action_logits = outputs['action_logits'][valid]
    action_true = batch['action_label'][valid]
    if action_weights is not None:
        L_action = F.cross_entropy(action_logits, action_true, weight=action_weights)
    else:
        L_action = F.cross_entropy(action_logits, action_true)

    losses = {
        'L_ret': L_ret.item(),
        'L_mfe': L_mfe.item(),
        'L_mae': L_mae.item(),
        'L_action': L_action.item(),
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


def _extract_v5_arrays(concat_outputs):
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

    action_probs = np.exp(action_logits - np.max(action_logits, axis=1, keepdims=True))
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


def v5_quality_mask(arrays, cfg: V5QualityGateConfig, epoch: int = 999):
    """Apply data-adaptive quality gates with warmup bypass and minimum pass rate.

    Three safeguards prevent the multiplicative filtering problem:
    1. Warmup bypass: epochs 0-4 skip quality gates entirely (all pass)
    2. Lenient percentiles: p90 sigma/mae, p25 |mu_R|, p40 p_trade
       Each gate independently passes ~60-90% so intersection ≈ 25-40%
    3. Minimum pass rate: if combined pass rate < 10%, progressively relax
       all thresholds until at least 10% pass

    Returns: boolean mask, diagnostics dict
    """
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
        finite_sigma = arrays['sigma'][np.isfinite(arrays['sigma'])]
        if len(finite_sigma) > 100:
            adaptive_sigma = min(cfg.sigma_max, float(np.percentile(finite_sigma, 90)))
        sigma_pass = np.isfinite(arrays['sigma']) & (arrays['sigma'] <= adaptive_sigma)

    finite_mae = arrays['mae'][np.isfinite(arrays['mae'])]
    adaptive_mae = cfg.mae_max
    if len(finite_mae) > 100:
        adaptive_mae = min(cfg.mae_max, float(np.percentile(finite_mae, 90)))
    mae_pass = np.isfinite(arrays['mae']) & (arrays['mae'] <= adaptive_mae)

    mu_R = arrays['mu_R']
    abs_mu = np.abs(mu_R[np.isfinite(mu_R)])
    adaptive_mu_min = cfg.mu_R_min
    if len(abs_mu) > 100:
        adaptive_mu_min = max(cfg.mu_R_min * 0.01, float(np.percentile(abs_mu, 25)))
    edge_pass = np.isfinite(mu_R) & (np.abs(mu_R) >= adaptive_mu_min)

    pt = arrays['p_trade']
    adaptive_ptrade = cfg.p_trade_min
    if len(pt) > 100:
        adaptive_ptrade = max(cfg.p_trade_min * 0.3, float(np.percentile(pt, 40)))
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


def compute_v5_scores(outputs_or_arrays, horizon_bars=16, score_lambda=0.5,
                      risk_proxy='mae', mae_cap=2.0, _arrays=None):
    """Compute execution-aware v5 scores -- all in R-units.

    Score = max(edge_long, edge_short) - penalty
    edge_long  = p_long  * mu_R / (mae_R + eps)
    edge_short = p_short * (-mu_R) / (mae_R + eps)
    penalty    = lambda * max(0, -mu_R)   [penalize only wrong-side bets]

    The penalty only activates when mu_R is negative for LONGs or positive
    for SHORTs -- i.e., when the predicted direction opposes the chosen side.
    This keeps penalty on the same scale as edge (~0.001-0.01 R-units)
    instead of raw MAE (~0.9 R-units) which created a 2000x mismatch.
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

    risk = np.maximum(mae_pred, 1e-3)

    edge_long = p_long * np.divide(mu_R, risk, out=np.zeros_like(mu_R), where=risk > 0)
    edge_short = p_short * np.divide(-mu_R, risk, out=np.zeros_like(mu_R), where=risk > 0)

    best_edge = np.maximum(edge_long, edge_short)
    sides = np.where(edge_long >= edge_short, 1, -1)

    penalty_long = np.maximum(0.0, -mu_R)
    penalty_short = np.maximum(0.0, mu_R)
    directional_penalty = np.where(sides == 1, penalty_long, penalty_short)
    penalty = score_lambda * directional_penalty

    scores = best_edge - penalty

    return scores, sides, {
        'mu_R_mean': float(np.nanmean(mu_R)),
        'mu_R_std': float(np.nanstd(mu_R)),
        'mae_R_mean': float(np.nanmean(mae_pred)),
        'mfe_R_mean': float(np.nanmean(mfe_pred)),
        'p_long_mean': float(np.nanmean(p_long)),
        'p_short_mean': float(np.nanmean(p_short)),
        'edge_long_mean': float(np.nanmean(edge_long)),
        'edge_short_mean': float(np.nanmean(edge_short)),
        'penalty_mean': float(np.nanmean(penalty)),
        'score_mean': float(np.nanmean(scores)),
        'score_std': float(np.nanstd(scores)),
        'score_p50': float(np.nanpercentile(scores, 50)),
        'score_p90': float(np.nanpercentile(scores, 90)),
        'score_pct_positive': float(np.nanmean(scores > 0) * 100),
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

    log.info("[V5_TPD_CTRL] epoch=%d tpd=%.1f target=%.1f±%.1f thr=%.4f->%.4f "
             "step_mult=%.2f score_std=%.4f action=%s trades=%d eligible=%d "
             "clamp=[%.4f,%.4f]",
             epoch, tpd, target_tpd, tol,
             current_threshold, new_threshold,
             tpd_cfg.thr_step_mult, score_std, action, n_trades,
             len(eligible_finite), sp5, sp99)

    return new_threshold, n_trades, tpd, action


def _run_v5_sweep(scores, sides, precomputed_outcomes, precomputed_r,
                  val_bars, epoch, tp_mult, sl_mult,
                  target_tpd=6.5, target_tpd_tol=1.5, min_trades=30,
                  candidate_mask=None, risk_controls=None,
                  symbol_ids=None, horizon_bars=16,
                  quality_mask=None, score_threshold=None):
    """Score-based sweep for v5 model.

    If score_threshold is provided, uses threshold-based selection (TPD controller).
    Otherwise falls back to percentile-based sweep.
    """
    from data.candidate_generator import apply_risk_controls, RiskControls

    COOLDOWN = 4

    safe_outcomes = np.where(
        np.isin(precomputed_outcomes, ["TP", "SL", "EXP_WIN", "EXP_LOSS"]),
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

    for thr, label in zip(thresholds_to_sweep, labels_for_thresholds):
        selected = scores_work >= thr
        sel_indices = np.where(selected)[0]
        if len(sel_indices) == 0:
            continue

        chronological_idx = sel_indices[np.argsort(sel_indices)]
        taken = []
        last_bar = -COOLDOWN - 1
        for idx in chronological_idx:
            if idx - last_bar >= COOLDOWN:
                taken.append(idx)
                last_bar = idx

        n_above_thr = len(sel_indices)
        n_after_cooldown = len(taken)

        if len(taken) < 5:
            log.debug("[V5_SWEEP_DIAG] %s: above_thr=%d after_cooldown=%d (<5, skipped)",
                      label, n_above_thr, n_after_cooldown)
            continue

        taken = np.array(taken)
        t_outcomes = safe_outcomes[taken]
        t_r = safe_r[taken]

        valid_trades = np.isin(t_outcomes, ["TP", "SL", "EXP_WIN", "EXP_LOSS"])
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
        sharpe = expect / max(std_r, 1e-6) * np.sqrt(252 * 96)

        total_win = np.sum(wins)
        total_loss = abs(np.sum(losses))
        pf = total_win / max(total_loss, 1e-6)

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

        m = {
            'label': label, 'pct': pct_val, 'trades': n_trades,
            'expect': expect, 'winrate': winrate, 'sharpe': sharpe, 'pf': pf,
            'avg_win_r': avg_win, 'avg_loss_r': avg_loss, 'median_r': median_r,
            'pct_tp': pct_tp, 'pct_sl': pct_sl, 'pct_exp': pct_exp,
            'trades_per_day': tpd, 'threshold': thr,
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

    n_valid_outcomes_total = np.sum(np.isin(safe_outcomes, ["TP", "SL", "EXP_WIN", "EXP_LOSS"]))
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

    return sweep_results, best_label, best_score_val, best_pct


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
):
    """V5.0.1 Forecaster training pipeline with quality gating + TPD controller."""
    from config import config as app_config
    from data.candidate_generator import (
        CandidateConfig, generate_candidate_mask,
        apply_risk_controls, RiskControls,
        BARRIER_PRESETS, PresetConfig,
    )
    from data.v5_target_generator import build_v5_targets, build_barrier_preset_labels
    from models.v5_forecaster import V5Forecaster, V5ForecasterConfig

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

    presets = []
    if barrier_presets:
        for name in barrier_presets:
            presets.append(BARRIER_PRESETS.get(name, BARRIER_PRESETS['standard']))
    if not presets:
        presets = [{'tp_mult': tp_mult, 'sl_mult': sl_mult, 'label': 'default'}]

    if min_lr is None:
        min_lr = lr * 0.01

    log.info("=" * 60)
    log.info("  V5.0.1 FORECASTER - TRAINING")
    log.info("=" * 60)
    log.info(f"Version: {V5_FEATURE_VERSION}")
    log.info(f"[V5_CONFIG] w_ret={w_ret} w_mfe={w_mfe} w_mae={w_mae} w_action={w_action}")
    log.info(f"[V5_CONFIG] w_barrier={w_barrier} w_regime={w_regime}")
    log.info(f"[V5_CONFIG] score_lambda={score_lambda} risk_proxy={risk_proxy}")
    log.info(f"[V5_CONFIG] hold_target={hold_target} mfe_min={mfe_min}")
    log.info(f"[V5_CONFIG] barrier_mode={barrier_mode} presets={[p.get('label','?') for p in presets]}")
    log.info(f"[V5_CONFIG] target_tpd={target_tpd} tpd_tol={target_tpd_tol}")
    log.info(f"[V5_CONFIG] candidates={candidate_config.enabled} regime_head={use_regime_head}")
    log.info(f"[V5_CONFIG] cand_warmup_epochs={cand_warmup_epochs}")
    log.info(f"[V5_CONFIG] horizon={horizon} epochs={epochs} batch={batch_size} lr={lr}")
    log.info(f"[V5_CONFIG] ALL targets in R-units (price_change / ATR)")
    log.info(f"[V5_QUAL_CONFIG] sigma_max={quality_gate_cfg.sigma_max} mae_max={quality_gate_cfg.mae_max} "
             f"mu_R_min={quality_gate_cfg.mu_R_min} p_trade_min={quality_gate_cfg.p_trade_min} "
             f"enable_calib={quality_gate_cfg.enable_calib}")
    log.info(f"[V5_TPD_CONFIG] target={tpd_ctrl_cfg.target_tpd}±{tpd_ctrl_cfg.tpd_tol} "
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
    val_barrier_oracle_list = []
    val_barrier_soft_list = []

    features_df_columns = None

    from data.common import generate_v5_sweep_outcomes

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

        v5_targets = build_v5_targets(
            sym_df, horizon=horizon, atr_period=14,
            hold_target=hold_target, mfe_min_r=mfe_min
        )

        v5_targets['valid_mask'][:max_lookback] = False

        sym_cand_mask = None
        if candidate_config.enabled:
            sym_cand_mask, _ = generate_candidate_mask(
                sym_df, candidate_config, symbol=sym
            )

        sweep_result = generate_v5_sweep_outcomes(
            sym_df, horizon=horizon, tp_mult=tp_mult,
            sl_mult=sl_mult, atr_period=14,
        )
        sym_realized_r = sweep_result['realized_r']
        sym_outcomes = sweep_result['outcome']

        sym_realized_r[:max_lookback] = np.nan
        sym_outcomes[:max_lookback] = "NO_CANDIDATE"
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

        sym_split = int(n * 0.8)
        log.info(f"[V5] {sym}: per-symbol split at bar {sym_split}/{n} "
                 f"(train={sym_split}, val={n - sym_split})")

        feat_arr = sym_features_df.values.astype(np.float32)
        ret_arr = v5_targets['ret_R'][:n]
        mfe_arr = v5_targets['mfe_R'][:n]
        mae_arr = v5_targets['mae_R'][:n]
        vol_arr = v5_targets['vol_h'][:n]
        act_arr = v5_targets['action_label'][:n]
        val_arr = v5_targets['valid_mask'][:n]

        train_features.append(feat_arr[:sym_split])
        train_ret_R_list.append(ret_arr[:sym_split])
        train_mfe_R_list.append(mfe_arr[:sym_split])
        train_mae_R_list.append(mae_arr[:sym_split])
        train_vol_h_list.append(vol_arr[:sym_split])
        train_action_list.append(act_arr[:sym_split])
        train_valid_list.append(val_arr[:sym_split])
        train_sym_ids_list.append(sym_id_arr[:sym_split])
        train_cand_mask_list.append(cand_arr[:sym_split])
        train_outcomes_list.append(sym_outcomes[:sym_split])
        train_realized_r_list.append(sym_realized_r[:sym_split])
        train_barrier_oracle_list.append(barrier_oracle[:sym_split])
        train_barrier_soft_list.append(barrier_soft[:sym_split])

        val_features.append(feat_arr[sym_split:])
        val_ret_R_list.append(ret_arr[sym_split:])
        val_mfe_R_list.append(mfe_arr[sym_split:])
        val_mae_R_list.append(mae_arr[sym_split:])
        val_vol_h_list.append(vol_arr[sym_split:])
        val_action_list.append(act_arr[sym_split:])
        val_valid_list.append(val_arr[sym_split:])
        val_sym_ids_list.append(sym_id_arr[sym_split:])
        val_cand_mask_list.append(cand_arr[sym_split:])
        val_outcomes_list.append(sym_outcomes[sym_split:])
        val_realized_r_list.append(sym_realized_r[sym_split:])
        val_barrier_oracle_list.append(barrier_oracle[sym_split:])
        val_barrier_soft_list.append(barrier_soft[sym_split:])

    train_feat = np.concatenate(train_features, axis=0)
    val_feat = np.concatenate(val_features, axis=0)

    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    train_feat = scaler.fit_transform(train_feat).astype(np.float32)
    val_feat = scaler.transform(val_feat).astype(np.float32)
    log.info(f"[V5] RobustScaler fitted on {len(train_feat)} train bars, applied to {len(val_feat)} val bars")

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
    val_barrier_oracle = _concat_lists(val_barrier_oracle_list)
    val_barrier_soft = np.concatenate(val_barrier_soft_list, axis=0)

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
            log.warning("[V5_NAN_AUDIT] %s has %d NaNs in %d valid bars (%.1f%%)",
                        label, nan_in_valid, int(np.sum(vmask)),
                        100.0 * nan_in_valid / max(int(np.sum(vmask)), 1))
    nan_feats_train = np.sum(np.isnan(train_feat))
    nan_feats_val = np.sum(np.isnan(val_feat))
    if nan_feats_train > 0 or nan_feats_val > 0:
        log.warning("[V5_NAN_AUDIT] Features NaN: train=%d val=%d", nan_feats_train, nan_feats_val)

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
    log.info(f"[V5] Total bars: {total_bars} (train={total_train}, val={total_val}) | "
             f"Features: {input_dim} | Symbols: {len(symbols)}")

    valid_train_action = train_action[train_valid]
    n_hold = int(np.sum(valid_train_action == 0))
    n_long = int(np.sum(valid_train_action == 1))
    n_short = int(np.sum(valid_train_action == 2))
    n_total_act = max(n_hold + n_long + n_short, 1)

    log.info(f"[V5_ACTION_DIST] TRAIN: HOLD={n_hold} ({n_hold/n_total_act:.1%}) "
             f"LONG={n_long} ({n_long/n_total_act:.1%}) SHORT={n_short} ({n_short/n_total_act:.1%})")

    action_class_weights = np.ones(3, dtype=np.float32)
    if n_hold > 0 and n_long > 0 and n_short > 0:
        counts = np.array([n_hold, n_long, n_short], dtype=np.float64)
        inv_freq = n_total_act / (3.0 * counts)
        inv_freq = np.clip(inv_freq, 0.5, 3.0)
        action_class_weights = inv_freq.astype(np.float32)
    log.info(f"[V5_ACTION_DIST] Class weights: HOLD={action_class_weights[0]:.3f} "
             f"LONG={action_class_weights[1]:.3f} SHORT={action_class_weights[2]:.3f}")

    action_weights_tensor = torch.tensor(action_class_weights, dtype=torch.float32).to(device)

    valid_val_action = val_action_arr[val_valid]
    vn_hold = int(np.sum(valid_val_action == 0))
    vn_long = int(np.sum(valid_val_action == 1))
    vn_short = int(np.sum(valid_val_action == 2))
    vn_total = max(vn_hold + vn_long + vn_short, 1)
    log.info(f"[V5_ACTION_DIST] VAL: HOLD={vn_hold} ({vn_hold/vn_total:.1%}) "
             f"LONG={vn_long} ({vn_long/vn_total:.1%}) SHORT={vn_short} ({vn_short/vn_total:.1%})")

    train_ret_valid = train_ret_R[train_valid]
    if len(train_ret_valid) > 0:
        log.info(f"[V5_DATA_DIAG] ret_R train: mean={np.mean(train_ret_valid):.4f} "
                 f"std={np.std(train_ret_valid):.4f} p5={np.percentile(train_ret_valid,5):.4f} "
                 f"p95={np.percentile(train_ret_valid,95):.4f}")

    train_ds = V5Dataset(
        train_feat, train_ret_R, train_mfe_R,
        train_mae_R, train_vol_h, train_action,
        train_valid, train_sym_ids,
        train_barrier_oracle, train_barrier_soft,
    )
    val_ds = V5Dataset(
        val_feat, val_ret_R, val_mfe_R,
        val_mae_R, val_vol_h, val_action_arr,
        val_valid, val_sym_ids_arr,
        val_barrier_oracle, val_barrier_soft,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    n_barrier = len(presets) if len(presets) > 1 and barrier_mode != 'fixed' else 0
    model_config = V5ForecasterConfig(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128, 64],
        dropout=0.3,
        use_layer_norm=True,
        use_residual=True,
        n_barrier_presets=n_barrier,
        enable_regime_head=use_regime_head,
        n_symbols=len(symbols) if len(symbols) > 1 else 1,
        symbol_embed_dim=4,
    )
    model = V5Forecaster(model_config).to(device)
    log.info(f"[V5] Model parameters: {model.parameters_count():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs])

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    best_val_loss = float('inf')
    best_expectancy = float('-inf')
    best_expectancy_pct = 0.0
    patience = 0
    max_patience = 25

    val_cand_mask = val_cand_mask_arr
    val_outcomes = val_outcomes_arr
    val_realized_r = val_realized_r_arr
    val_sym_ids = val_sym_ids_arr
    val_bars = total_val

    current_score_threshold = tpd_ctrl_cfg.score_threshold

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
            },
        },
    }

    for epoch in range(1, epochs + 1):
        use_candidates_this_epoch = candidate_config.enabled and epoch > cand_warmup_epochs
        if candidate_config.enabled and epoch == cand_warmup_epochs + 1:
            log.info(f"[V5] Candidate warmup complete (epoch {epoch}), enabling candidate mask for sweep")

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
            loss, ld = compute_v5_loss(
                outputs, batch_gpu,
                w_ret=w_ret, w_mfe=w_mfe, w_mae=w_mae,
                w_action=w_action, w_barrier=w_barrier, w_regime=w_regime,
                barrier_mode=barrier_mode,
                action_weights=action_weights_tensor,
                epoch=epoch,
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
                vloss, _ = compute_v5_loss(
                    outputs, batch_gpu,
                    w_ret=w_ret, w_mfe=w_mfe, w_mae=w_mae,
                    w_action=w_action, w_barrier=w_barrier, w_regime=w_regime,
                    barrier_mode=barrier_mode,
                    action_weights=action_weights_tensor,
                    epoch=epoch,
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

        log.info(f"[V5] Epoch {epoch:03d}/{epochs} | train={avg_train_loss:.4f} val={avg_val_loss:.4f} "
                 f"lr={current_lr:.2e} act_acc={action_acc:.3f} "
                 f"pred[H/L/S]={pred_hold}/{pred_long}/{pred_short} | {lb_str}")

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
            )

            log.info(f"[V5_SCORE_DIAG] mu_R: mean={score_diag['mu_R_mean']:.4f} std={score_diag['mu_R_std']:.4f} | "
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
            log.info(f"[V5_OUTPUT_DIST] mu_R: p5={np.percentile(mu_arr,5):.4f} p50={np.percentile(mu_arr,50):.4f} "
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
                tpd_ctrl_cfg,
            )

            sweep_results, sweep_label, sweep_expect, sweep_pct = _run_v5_sweep(
                scores, sides, val_outcomes, val_realized_r,
                val_bars, epoch, tp_mult, sl_mult,
                target_tpd=target_tpd, target_tpd_tol=target_tpd_tol,
                candidate_mask=sweep_cand_mask,
                risk_controls=risk_controls,
                symbol_ids=val_sym_ids,
                horizon_bars=horizon,
                quality_mask=quality_mask,
                score_threshold=current_score_threshold,
            )

            if quality_gate_cfg.enable_calib:
                val_p_trade = arrays['p_trade'][:len(val_realized_r)]
                compute_v5_calibration(val_p_trade, val_realized_r)

            ckpt_v5_config = ckpt_train_config['v5_config']
            ckpt_v5_config['tpd_controller']['current_threshold'] = current_score_threshold

            if sweep_expect > best_expectancy:
                best_expectancy = sweep_expect
                best_expectancy_pct = sweep_pct
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_config': ckpt_model_config,
                    'train_config': ckpt_train_config,
                    'feature_columns': features_df_columns,
                    'n_features': input_dim,
                    'feature_version': V5_FEATURE_VERSION,
                    'model_type': 'v5_forecaster',
                    'scaler_center': scaler.center_,
                    'scaler_scale': scaler.scale_,
                    'barrier_config': {
                        'tp_mult': tp_mult, 'sl_mult': sl_mult,
                        'horizon': horizon,
                        'presets': [p.get('label', 'default') for p in presets],
                    },
                    'best_expectancy': best_expectancy,
                    'best_expectancy_pct': best_expectancy_pct,
                    'trained_at': datetime.now().isoformat(),
                }, checkpoint_dir / "best_v5_expectancy.pt")
                log.info(f"[V5_CKPT] New best expectancy={best_expectancy:.4f} at {sweep_label}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': ckpt_model_config,
                'train_config': ckpt_train_config,
                'feature_columns': features_df_columns,
                'n_features': input_dim,
                'feature_version': V5_FEATURE_VERSION,
                'model_type': 'v5_forecaster',
                'scaler_center': scaler.center_,
                'scaler_scale': scaler.scale_,
                'trained_at': datetime.now().isoformat(),
            }, checkpoint_dir / "best_v5_loss.pt")
            log.info(f"[V5_CKPT] New best val_loss={best_val_loss:.4f}")
        else:
            patience += 1
            if patience >= max_patience:
                log.info(f"[V5] Early stopping at epoch {epoch} (patience={max_patience})")
                break

    log.info("=" * 60)
    log.info(f"[V5] Training complete. Best expectancy={best_expectancy:.4f} best_loss={best_val_loss:.4f}")
    log.info(f"[V5] Final score_threshold={current_score_threshold}")
    log.info("=" * 60)
