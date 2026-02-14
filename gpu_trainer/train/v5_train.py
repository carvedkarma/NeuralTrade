"""
V5 Training Pipeline: Separating Market Forecasting from Decision Layer

train_v5_model() implements:
- V5Forecaster model with trunk + heads
- Composite loss (ret_nll + mfe/mae Huber + action CE + barrier CE + regime CE)
- V5 scoring: execution-aware score from predicted distributions
- Validation sweep with candidate mask and risk controls
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

log = logging.getLogger("QuickStart")

V5_FEATURE_VERSION = "v5.0_forecaster"


class V5Dataset(Dataset):
    def __init__(self, features, ret_h, mfe_h, mae_h, vol_h, action_labels,
                 valid_mask, symbol_ids=None, barrier_labels=None, barrier_soft=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.ret_h = torch.tensor(ret_h, dtype=torch.float32)
        self.mfe_h = torch.tensor(mfe_h, dtype=torch.float32)
        self.mae_h = torch.tensor(mae_h, dtype=torch.float32)
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
            'ret_h': self.ret_h[idx],
            'mfe_h': self.mfe_h[idx],
            'mae_h': self.mae_h[idx],
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
                    w_action=0.5, w_barrier=0.25, w_regime=0.1,
                    barrier_mode='fixed'):
    """Compute v5 composite loss."""
    valid = batch['valid']
    if valid.sum() == 0:
        return torch.tensor(0.0, device=outputs['ret_mu'].device, requires_grad=True), {}

    ret_mu = outputs['ret_mu'][valid].squeeze(-1)
    ret_log_sigma = outputs['ret_log_sigma'][valid].squeeze(-1)
    ret_true = batch['ret_h'][valid]

    sigma = torch.exp(ret_log_sigma)
    nll = 0.5 * torch.log(2 * torch.pi * sigma ** 2 + 1e-8) + \
          0.5 * ((ret_true - ret_mu) / (sigma + 1e-8)) ** 2
    L_ret = nll.mean()

    huber = nn.SmoothL1Loss()
    mfe_pred = outputs['mfe'][valid].squeeze(-1)
    mfe_true = batch['mfe_h'][valid]
    L_mfe = huber(mfe_pred, mfe_true)

    mae_pred = outputs['mae'][valid].squeeze(-1)
    mae_true = batch['mae_h'][valid]
    L_mae = huber(mae_pred, mae_true)

    action_logits = outputs['action_logits'][valid]
    action_true = batch['action_label'][valid]
    L_action = F.cross_entropy(action_logits, action_true)

    losses = {
        'L_ret': L_ret.item(),
        'L_mfe': L_mfe.item(),
        'L_mae': L_mae.item(),
        'L_action': L_action.item(),
    }

    total = w_ret * L_ret + w_mfe * L_mfe + w_mae * L_mae + w_action * L_action

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


def compute_v5_scores(outputs, horizon_bars=16, score_lambda=0.5, risk_proxy='mae'):
    """Compute execution-aware v5 scores for ranking."""
    ret_mu = outputs['ret_mu'].detach().cpu().numpy().squeeze(-1)
    mae_pred = outputs['mae'].detach().cpu().numpy().squeeze(-1)
    action_logits = outputs['action_logits'].detach().cpu().numpy()

    action_probs = np.exp(action_logits - np.max(action_logits, axis=1, keepdims=True))
    action_probs = action_probs / (action_probs.sum(axis=1, keepdims=True) + 1e-8)
    p_long = action_probs[:, 1]
    p_short = action_probs[:, 2]

    if risk_proxy == 'mae':
        risk = np.maximum(mae_pred, 1e-6)
    else:
        ret_log_sigma = outputs['ret_log_sigma'].detach().cpu().numpy().squeeze(-1)
        risk = np.maximum(np.exp(ret_log_sigma), 1e-6)

    downside_penalty = score_lambda * np.maximum(0, mae_pred)

    e_long = p_long * (ret_mu / risk) - downside_penalty
    e_short = p_short * (-ret_mu / risk) - downside_penalty
    scores = np.maximum(e_long, e_short)
    scores = scores / max(horizon_bars, 1)

    sides = np.where(e_long >= e_short, 1, -1)

    return scores, sides


def _run_v5_sweep(scores, sides, precomputed_outcomes, precomputed_r,
                  val_bars, epoch, tp_mult, sl_mult,
                  target_tpd=6.5, target_tpd_tol=1.5, min_trades=30,
                  candidate_mask=None, risk_controls=None,
                  symbol_ids=None, horizon_bars=16):
    """Score-based sweep for v5 model."""
    from data.candidate_generator import apply_risk_controls, RiskControls

    TOP_PCTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    COOLDOWN = 4

    safe_outcomes = np.where(
        np.isin(precomputed_outcomes, ["TP", "SL", "EXP_WIN", "EXP_LOSS"]),
        precomputed_outcomes, "NO_CANDIDATE"
    )
    safe_r = np.where(np.isnan(precomputed_r.astype(float)), 0.0, precomputed_r.astype(float))

    if candidate_mask is not None:
        non_cand = ~candidate_mask
        scores_work = scores.copy()
        scores_work[non_cand] = -np.inf
    else:
        scores_work = scores.copy()

    tpd_lo = target_tpd - target_tpd_tol
    tpd_hi = target_tpd + target_tpd_tol
    val_days = val_bars / 96.0

    sweep_results = []
    best_in_freq_score = float('-inf')
    best_in_freq_label = ""
    best_any_score = float('-inf')
    best_any_pct = 0.0
    best_any_label = ""

    for pct in TOP_PCTS:
        finite_scores = scores_work[np.isfinite(scores_work)]
        if len(finite_scores) == 0:
            continue
        threshold = np.percentile(finite_scores, (1 - pct) * 100)
        selected = scores_work >= threshold
        sel_indices = np.where(selected)[0]
        if len(sel_indices) == 0:
            continue

        sorted_idx = sel_indices[np.argsort(-scores_work[sel_indices])]
        taken = []
        last_bar = -COOLDOWN - 1
        for idx in sorted_idx:
            if idx - last_bar >= COOLDOWN:
                taken.append(idx)
                last_bar = idx

        if len(taken) < 5:
            continue

        taken = np.array(taken)
        t_outcomes = safe_outcomes[taken]
        t_r = safe_r[taken]

        valid_trades = np.isin(t_outcomes, ["TP", "SL", "EXP_WIN", "EXP_LOSS"])
        if valid_trades.sum() < 5:
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

        label = f"top{int(pct*100)}%"
        m = {
            'label': label, 'pct': pct, 'trades': n_trades,
            'expect': expect, 'winrate': winrate, 'sharpe': sharpe, 'pf': pf,
            'avg_win_r': avg_win, 'avg_loss_r': avg_loss, 'median_r': median_r,
            'pct_tp': pct_tp, 'pct_sl': pct_sl, 'pct_exp': pct_exp,
            'trades_per_day': tpd,
        }
        sweep_results.append(m)

        composite = expect * min(sharpe, 10.0)
        if tpd_lo <= tpd <= tpd_hi and n_trades >= min_trades:
            if composite > best_in_freq_score:
                best_in_freq_score = composite
                best_in_freq_label = label
        if composite > best_any_score:
            best_any_score = composite
            best_any_pct = pct
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
    if candidate_mask is not None:
        scores_eligible = score_arr[candidate_mask]
    else:
        scores_eligible = score_arr
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

    log.info("V5 SWEEP (epoch %d) | cooldown=%d | TP=%.1fx SL=%.1fx ATR | val_days=%.1f | target=%.1f±%.1f tpd",
             epoch, COOLDOWN, tp_mult, sl_mult, val_days, target_tpd, target_tpd_tol)
    log.info("%-8s %5s %8s %6s %6s %5s | %6s %6s %6s | %4s %4s %4s | %5s",
             "Select", "Trds", "Expect", "WR", "Shrpe", "PF",
             "WinR", "LosR", "MedR", "%TP", "%SL", "%EX", "T/Day")
    log.info("-" * 120)
    for m in sweep_results:
        marker = ""
        if m['label'] == best_label and m['trades'] >= min_trades and best_score_val > float('-inf'):
            marker = " <<< BEST"
        log.info("%-8s %5d %+8.3f %5.1f%% %6.2f %5.2f | %+6.3f %+6.3f %+6.3f | %3.0f%% %3.0f%% %3.0f%% | %5.1f%s",
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
    w_ret=1.0, w_mfe=0.25, w_mae=0.25, w_action=0.5,
    w_barrier=0.25, w_regime=0.1,
    score_lambda=0.5, risk_proxy='mae',
    target_tpd=6.5, target_tpd_tol=1.5,
    deadzone=0.0005, mfe_min=0.2,
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
):
    """V5 Forecaster training pipeline."""
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

    presets = []
    if barrier_presets:
        for name in barrier_presets:
            presets.append(BARRIER_PRESETS.get(name, BARRIER_PRESETS['standard']))
    if not presets:
        presets = [{'tp_mult': tp_mult, 'sl_mult': sl_mult, 'label': 'default'}]

    if min_lr is None:
        min_lr = lr * 0.01

    log.info("=" * 60)
    log.info("  V5.0 FORECASTER - TRAINING")
    log.info("=" * 60)
    log.info(f"Version: {V5_FEATURE_VERSION}")
    log.info(f"[V5_CONFIG] w_ret={w_ret} w_mfe={w_mfe} w_mae={w_mae} w_action={w_action}")
    log.info(f"[V5_CONFIG] w_barrier={w_barrier} w_regime={w_regime}")
    log.info(f"[V5_CONFIG] score_lambda={score_lambda} risk_proxy={risk_proxy}")
    log.info(f"[V5_CONFIG] deadzone={deadzone} mfe_min={mfe_min}")
    log.info(f"[V5_CONFIG] barrier_mode={barrier_mode} presets={[p.get('label','?') for p in presets]}")
    log.info(f"[V5_CONFIG] target_tpd={target_tpd} tpd_tol={target_tpd_tol}")
    log.info(f"[V5_CONFIG] candidates={candidate_config.enabled} regime_head={use_regime_head}")
    log.info(f"[V5_CONFIG] horizon={horizon} epochs={epochs} batch={batch_size} lr={lr}")

    if barrier_mode == 'oracle':
        log.warning("[V5] barrier_mode=oracle: WARNING hindsight leakage, research only!")

    from data.pipeline import FeatureEngineer
    data_dir = Path("data_cache")

    if symbols is None or len(symbols) == 0:
        symbols = ["BTCUSDT"]

    all_features = []
    all_ret_h = []
    all_mfe_h = []
    all_mae_h = []
    all_vol_h = []
    all_action = []
    all_valid = []
    all_sym_ids = []
    all_cand_mask = []
    all_outcomes = []
    all_realized_r = []
    all_barrier_oracle = []
    all_barrier_soft = []
    features_df_columns = None

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
        sym_features_df = fe.compute_features(sym_df)
        sym_features_df = sym_features_df.fillna(0)

        if features_df_columns is None:
            features_df_columns = list(sym_features_df.columns)
        else:
            sym_features_df = sym_features_df.reindex(columns=features_df_columns, fill_value=0)

        v5_targets = build_v5_targets(
            sym_df, horizon=horizon, atr_period=14,
            deadzone=deadzone, mfe_min_r=mfe_min
        )

        sym_cand_mask = None
        if candidate_config.enabled:
            sym_cand_mask, _ = generate_candidate_mask(
                sym_df, candidate_config, symbol=sym
            )

        from data.regression_targets import generate_v47_quality_targets
        htf_cols = [c for c in sym_features_df.columns if c.startswith('h1_') or c.startswith('h4_')]
        htf_features_df = sym_features_df[htf_cols].copy()
        label_df = generate_v47_quality_targets(
            sym_df, htf_features_df,
            horizon_periods=horizon,
            tp_atr_mult=tp_mult, sl_atr_mult=sl_mult,
            q_min_tp=q_min_tp,
            r_min_expiry_strict=r_min_expiry_strict,
            soft_label_temp=1.0,
            auto_balance=auto_balance_enter_labels,
            target_enter_rate=target_enter_rate,
            target_enter_rate_min=target_enter_rate_min,
            target_enter_rate_max=target_enter_rate_max,
            balance_search_steps=balance_search_steps,
        )
        sym_realized_r = label_df['realized_r'].values.astype(np.float32)
        sym_outcomes = label_df['outcome'].values

        barrier_oracle = np.zeros(len(sym_df), dtype=np.int64)
        barrier_soft = np.zeros((len(sym_df), len(presets)), dtype=np.float32)
        if len(presets) > 1 and barrier_mode in ('oracle', 'learnable'):
            barrier_oracle, barrier_soft = build_barrier_preset_labels(
                sym_df, presets, horizon=horizon, temperature=1.0
            )

        n = len(sym_features_df)
        sym_id_arr = np.full(n, si, dtype=np.int64)
        cand_arr = sym_cand_mask if sym_cand_mask is not None else np.ones(n, dtype=bool)

        all_features.append(sym_features_df.values.astype(np.float32))
        all_ret_h.append(v5_targets['ret_h'][:n])
        all_mfe_h.append(v5_targets['mfe_h'][:n])
        all_mae_h.append(v5_targets['mae_h'][:n])
        all_vol_h.append(v5_targets['vol_h'][:n])
        all_action.append(v5_targets['action_label'][:n])
        all_valid.append(v5_targets['valid_mask'][:n])
        all_sym_ids.append(sym_id_arr)
        all_cand_mask.append(cand_arr[:n])
        all_outcomes.append(sym_outcomes[:n])
        all_realized_r.append(sym_realized_r[:n])
        all_barrier_oracle.append(barrier_oracle[:n])
        all_barrier_soft.append(barrier_soft[:n])

    features_all = np.concatenate(all_features, axis=0)
    ret_h_all = np.concatenate(all_ret_h, axis=0)
    mfe_h_all = np.concatenate(all_mfe_h, axis=0)
    mae_h_all = np.concatenate(all_mae_h, axis=0)
    vol_h_all = np.concatenate(all_vol_h, axis=0)
    action_all = np.concatenate(all_action, axis=0)
    valid_all = np.concatenate(all_valid, axis=0)
    sym_ids_all = np.concatenate(all_sym_ids, axis=0)
    cand_mask_all = np.concatenate(all_cand_mask, axis=0)
    outcomes_all = np.concatenate(all_outcomes, axis=0)
    realized_r_all = np.concatenate(all_realized_r, axis=0)
    barrier_oracle_all = np.concatenate(all_barrier_oracle, axis=0)
    barrier_soft_all = np.concatenate(all_barrier_soft, axis=0)

    total_bars = len(features_all)
    input_dim = features_all.shape[1]
    log.info(f"[V5] Total bars: {total_bars} | Features: {input_dim} | Symbols: {len(symbols)}")

    split_idx = int(total_bars * 0.8)
    train_idx = np.arange(split_idx)
    val_idx = np.arange(split_idx, total_bars)

    ret_h_all = np.nan_to_num(ret_h_all, nan=0.0)
    mfe_h_all = np.nan_to_num(mfe_h_all, nan=0.0)
    mae_h_all = np.nan_to_num(mae_h_all, nan=0.0)
    vol_h_all = np.nan_to_num(vol_h_all, nan=0.0)

    train_ds = V5Dataset(
        features_all[train_idx], ret_h_all[train_idx], mfe_h_all[train_idx],
        mae_h_all[train_idx], vol_h_all[train_idx], action_all[train_idx],
        valid_all[train_idx], sym_ids_all[train_idx],
        barrier_oracle_all[train_idx], barrier_soft_all[train_idx],
    )
    val_ds = V5Dataset(
        features_all[val_idx], ret_h_all[val_idx], mfe_h_all[val_idx],
        mae_h_all[val_idx], vol_h_all[val_idx], action_all[val_idx],
        valid_all[val_idx], sym_ids_all[val_idx],
        barrier_oracle_all[val_idx], barrier_soft_all[val_idx],
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

    val_cand_mask = cand_mask_all[val_idx]
    val_outcomes = outcomes_all[val_idx]
    val_realized_r = realized_r_all[val_idx]
    val_sym_ids = sym_ids_all[val_idx]
    val_bars = len(val_idx)

    for epoch in range(1, epochs + 1):
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
            'ret_mu': [], 'ret_log_sigma': [], 'mae': [], 'mfe': [], 'action_logits': []
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
                )
                val_losses.append(vloss.item())

                for k in all_val_outputs:
                    if k in outputs:
                        all_val_outputs[k].append(outputs[k].detach().cpu())

        avg_val_loss = np.mean(val_losses)

        lb_str = " | ".join(f"{k}={np.mean(v):.4f}" for k, v in loss_breakdown.items() if v)
        current_lr = optimizer.param_groups[0]['lr']

        action_all_val = action_all[val_idx]
        action_logits_cat = torch.cat(all_val_outputs['action_logits'], dim=0).numpy()
        action_preds = np.argmax(action_logits_cat, axis=1)
        action_acc = np.mean(action_preds[:len(action_all_val)] == action_all_val[:len(action_preds)])

        log.info(f"[V5] Epoch {epoch:03d}/{epochs} | train={avg_train_loss:.4f} val={avg_val_loss:.4f} "
                 f"lr={current_lr:.2e} act_acc={action_acc:.3f} | {lb_str}")

        do_sweep = (epoch % 5 == 0) or (epoch == epochs) or (epoch <= 3)
        if do_sweep:
            concat_outputs = {}
            for k in all_val_outputs:
                if all_val_outputs[k]:
                    concat_outputs[k] = torch.cat(all_val_outputs[k], dim=0)

            scores, sides = compute_v5_scores(
                concat_outputs, horizon_bars=horizon,
                score_lambda=score_lambda, risk_proxy=risk_proxy
            )

            sweep_results, sweep_label, sweep_expect, sweep_pct = _run_v5_sweep(
                scores, sides, val_outcomes, val_realized_r,
                val_bars, epoch, tp_mult, sl_mult,
                target_tpd=target_tpd, target_tpd_tol=target_tpd_tol,
                candidate_mask=val_cand_mask if candidate_config.enabled else None,
                risk_controls=risk_controls,
                symbol_ids=val_sym_ids,
                horizon_bars=horizon,
            )

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
                'deadzone': deadzone, 'mfe_min': mfe_min,
                'barrier_mode': barrier_mode,
                'target_tpd': target_tpd, 'target_tpd_tol': target_tpd_tol,
                'v5_config': {
                    'candidate_engine': candidate_config.enabled,
                    'barrier_presets': [p.get('label', 'default') for p in presets],
                    'risk_controls': {
                        'daily_loss_limit_r': risk_controls.daily_loss_limit_r,
                        'max_concurrent_trades': risk_controls.max_concurrent_trades,
                        'max_symbol_exposure': risk_controls.max_symbol_exposure,
                    },
                },
            }

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
                'model_config': ckpt_model_config if do_sweep else {'input_dim': input_dim},
                'train_config': ckpt_train_config if do_sweep else {},
                'feature_columns': features_df_columns,
                'n_features': input_dim,
                'feature_version': V5_FEATURE_VERSION,
                'model_type': 'v5_forecaster',
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
    log.info("=" * 60)
