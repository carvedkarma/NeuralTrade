#!/usr/bin/env python3
"""
BTC Futures GPU Trainer - Quick Start (v3.1.0 ENTER QUALITY)
=============================================================
One-script setup: Downloads data from your Replit dashboard,
trains the ENTER QUALITY model on your GPU, and pushes predictions back.

The model predicts WHETHER to enter a trend-following trade (binary ENTER=0/1),
not WHICH direction. Direction comes from HTF (1H/4H) trend alignment.

Usage:
    python quick_start.py --url https://YOUR-APP.replit.app

That's it. Everything else is automatic.
"""

import argparse
import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("QuickStart")

FEATURE_VERSION = "v3.1.0_enter_quality_stf47_htf10"


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            log.info(f"GPU: {name} ({mem:.1f} GB)")
            return "cuda"
        else:
            log.warning("No GPU found - training will be slow on CPU")
            return "cpu"
    except ImportError:
        log.error("PyTorch not installed! Run: pip install -r requirements.txt")
        sys.exit(1)


def download_data(replit_url: str, data_dir: Path, force_fresh: bool = False):
    import requests

    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "BTCUSDT_15m.csv"
    parquet_path = data_dir / "BTCUSDT_15m.parquet"

    if parquet_path.exists() and not force_fresh:
        import pandas as pd
        existing = pd.read_parquet(parquet_path)
        log.info(f"Found existing data: {len(existing)} candles")
        resp = input("Re-download fresh data? (y/N): ").strip().lower()
        if resp != 'y':
            return parquet_path

    url = f"{replit_url.rstrip('/')}/api/data/export-csv?symbol=BTCUSDT&timeframe=15m"
    log.info(f"Downloading BTC 15m data from dashboard...")
    log.info(f"  URL: {url}")

    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        log.error(f"Cannot connect to {replit_url}")
        log.error("Make sure your Replit dashboard is running!")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        log.error(f"Server returned error: {e}")
        sys.exit(1)

    with open(csv_path, 'wb') as f:
        total = 0
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            total += len(chunk)
    log.info(f"  Downloaded {total / 1024:.0f} KB")

    import pandas as pd
    df = pd.read_csv(csv_path)
    log.info(f"  Loaded {len(df)} candles")

    if len(df) < 1000:
        log.error(f"Only {len(df)} candles - need at least 1,000 for training")
        log.error("Go to your dashboard's Neural Network tab and download more historical data first")
        sys.exit(1)

    date_min = datetime.fromtimestamp(df['timestamp'].min() / 1000).strftime('%Y-%m-%d')
    date_max = datetime.fromtimestamp(df['timestamp'].max() / 1000).strftime('%Y-%m-%d')
    log.info(f"  Date range: {date_min} to {date_max}")

    df.to_parquet(parquet_path, index=False)
    log.info(f"  Saved to {parquet_path}")

    csv_path.unlink(missing_ok=True)
    return parquet_path


def train_enter_model(data_path: Path, device: str, epochs: int, batch_size: int, lr: float,
                      checkpoint_interval: int = 25, warmup_epochs: int = 5, min_lr: float = None,
                      tp_mult: float = 2.0, sl_mult: float = 1.5, horizon: int = 24, slope_eps: float = 0.05):
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    from config import config

    log.info("=" * 60)
    log.info("  ENTER QUALITY MODEL - TRAINING")
    log.info("=" * 60)
    log.info(f"Version: {FEATURE_VERSION}")

    df = pd.read_parquet(data_path)
    log.info(f"Loaded {len(df)} candles")

    from data.pipeline import FeatureEngineer
    engineer = FeatureEngineer()
    features_df = engineer.compute_all_features(df)
    features_df = features_df.fillna(0)
    log.info(f"Computed {len(features_df.columns)} features ({engineer.STF_FEATURE_COUNT} STF + {engineer.HTF_FEATURE_COUNT} HTF)")

    htf_cols = [c for c in features_df.columns if c.startswith('h1_') or c.startswith('h4_')]
    htf_features_df = features_df[htf_cols].copy()
    log.info(f"HTF features for labeling: {htf_cols}")

    from data.regression_targets import generate_enter_quality_targets
    label_df = generate_enter_quality_targets(
        df, htf_features_df,
        horizon_periods=horizon,
        tp_atr_mult=tp_mult, sl_atr_mult=sl_mult,
        slope_eps=slope_eps,
    )

    enter_labels = label_df['enter_label'].values.astype(np.float32)
    side_hints = label_df['side_hint'].values.astype(np.int64)

    forward_returns = ((df['close'].shift(-horizon) - df['close']) / df['close']).fillna(0).values.astype(np.float32)

    sequence_length = config.data.sequence_length
    valid_start = sequence_length
    features_np = features_df.values[valid_start:].astype(np.float32)
    enter_np = enter_labels[valid_start:].astype(np.float32)
    side_np = side_hints[valid_start:].astype(np.int64)
    returns_np = forward_returns[valid_start:].astype(np.float32)

    n_total = len(features_np)
    purge_gap = horizon + sequence_length
    val_samples = max(int(n_total * 0.1), purge_gap)
    train_samples = n_total - purge_gap - val_samples

    if train_samples < sequence_length * 3:
        log.error(f"Not enough data for training: {train_samples} samples")
        sys.exit(1)

    train_end = train_samples
    val_start_idx = train_end + purge_gap
    val_end = val_start_idx + val_samples

    log.info(f"Data split: train={train_samples}, purge={purge_gap}, val={val_samples}")

    train_features_raw = features_np[:train_end]
    train_enter = enter_np[:train_end]
    train_side = side_np[:train_end]
    train_returns = returns_np[:train_end]

    val_features_raw = features_np[val_start_idx:val_end]
    val_enter = enter_np[val_start_idx:val_end]
    val_side = side_np[val_start_idx:val_end]
    val_returns = returns_np[val_start_idx:val_end]

    train_features_df_scaled = pd.DataFrame(train_features_raw, columns=features_df.columns)
    engineer.fit_scalers(train_features_df_scaled)
    clip_range = 5.0
    train_scaled = engineer.transform_and_clip(train_features_df_scaled, clip_range=clip_range).values.astype(np.float32)
    val_features_df_scaled = pd.DataFrame(val_features_raw, columns=features_df.columns)
    val_scaled = engineer.transform_and_clip(val_features_df_scaled, clip_range=clip_range).values.astype(np.float32)

    def clean_enter(features, enter, side, returns, name):
        features = np.where(np.isinf(features), np.nan, features)
        returns = np.where(np.isinf(returns), np.nan, returns)
        mask = np.isnan(features).any(axis=1) | np.isnan(returns)
        valid = ~mask
        dropped = mask.sum()
        if dropped > 0:
            log.info(f"  {name}: dropped {dropped} NaN rows")
        return features[valid], enter[valid], side[valid], returns[valid]

    train_scaled, train_enter, train_side, train_returns = clean_enter(train_scaled, train_enter, train_side, train_returns, "Train")
    val_scaled, val_enter, val_side, val_returns = clean_enter(val_scaled, val_enter, val_side, val_returns, "Val")

    pos_count = train_enter.sum()
    neg_count = len(train_enter) - pos_count
    pos_weight = neg_count / max(pos_count, 1)
    pos_weight = min(pos_weight, 10.0)
    log.info(f"ENTER label distribution: ENTER=1: {int(pos_count)} ({100*pos_count/len(train_enter):.1f}%), ENTER=0: {int(neg_count)} ({100*neg_count/len(train_enter):.1f}%)")
    log.info(f"BCE pos_weight: {pos_weight:.2f}")

    class EnterDataset(Dataset):
        def __init__(self, features, enter_labels, side_hints, returns, seq_len):
            self.features = features.astype(np.float32)
            self.enter_labels = enter_labels.astype(np.float32)
            self.side_hints = side_hints.astype(np.int64)
            self.returns = returns.astype(np.float32)
            self.seq_len = seq_len
            self.valid_indices = list(range(seq_len, len(features)))

        def __len__(self):
            return len(self.valid_indices)

        def __getitem__(self, idx):
            actual_idx = self.valid_indices[idx]
            start = actual_idx - self.seq_len
            seq = self.features[start:actual_idx]
            return (
                torch.from_numpy(seq),
                torch.tensor(self.enter_labels[actual_idx], dtype=torch.float32),
                torch.tensor(self.side_hints[actual_idx], dtype=torch.long),
                torch.tensor(self.returns[actual_idx], dtype=torch.float32),
            )

    train_dataset = EnterDataset(train_scaled, train_enter, train_side, train_returns, sequence_length)
    val_dataset = EnterDataset(val_scaled, val_enter, val_side, val_returns, sequence_length)

    log.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    input_dim = features_np.shape[1]

    from models.simple_mlp import EnhancedMultiHeadMLP, EnhancedMultiHeadMLP_Config
    mlp_config = EnhancedMultiHeadMLP_Config(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128, 64],
        num_classes=3,
        dropout=0.3,
        use_layer_norm=True,
        use_residual=True,
        enable_quantile_head=False,
        enable_vol_state_head=False,
        enable_mu_head=False,
        enable_sigma_head=False,
        enable_enter_head=True,
    )
    model = EnhancedMultiHeadMLP(mlp_config)
    model.name = "EnterQualityMLP"
    model.to(device)
    log.info(f"Model: EnterQualityMLP ({model.parameters_count():,} parameters)")
    log.info(f"Architecture: [512, 256, 128, 64] with residual connections")
    log.info(f"Active head: enter_head (binary) | All other heads DISABLED")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    effective_min_lr = min_lr if min_lr is not None else lr * 0.05
    warmup_sched = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=effective_min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    for pg in optimizer.param_groups:
        pg['lr'] = lr * 1e-3

    log.info(f"Training for {epochs} epochs (lr={lr}, batch={batch_size})")
    log.info(f"LR schedule: {warmup_epochs}-epoch warmup -> cosine annealing to {effective_min_lr:.2e}")
    log.info(f"Early stopping: patience=50, min_epochs=40")
    log.info(f"Barriers: TP={tp_mult}x ATR, SL={sl_mult}x ATR, horizon={horizon} bars")
    log.info("-" * 60)

    best_val_loss = float('inf')
    best_val_prauc = 0.0
    patience = 0
    max_patience = 50
    min_epochs = 40
    history = {'train_loss': [], 'val_loss': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_prauc': []}

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            features_batch, enter_batch, side_batch, returns_batch = batch
            features_batch = features_batch.to(device)
            enter_batch = enter_batch.to(device)

            optimizer.zero_grad()
            output = model.forward_multihead(features_batch)
            enter_logits = output.enter_logits.squeeze(-1)
            loss = criterion(enter_logits, enter_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(n_batches, 1)

        model.eval()
        val_loss_total = 0
        val_n = 0
        all_probs = []
        all_targets = []
        all_sides = []
        all_val_returns = []

        with torch.no_grad():
            for batch in val_loader:
                features_batch, enter_batch, side_batch, returns_batch = batch
                features_batch = features_batch.to(device)
                enter_batch = enter_batch.to(device)

                output = model.forward_multihead(features_batch)
                enter_logits = output.enter_logits.squeeze(-1)
                v_loss = criterion(enter_logits, enter_batch)
                val_loss_total += v_loss.item()
                val_n += 1

                probs = torch.sigmoid(enter_logits).cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(enter_batch.cpu().numpy())
                all_sides.extend(side_batch.numpy())
                all_val_returns.extend(returns_batch.numpy())

        avg_val_loss = val_loss_total / max(val_n, 1)

        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        all_sides = np.array(all_sides)
        all_val_returns = np.array(all_val_returns)

        threshold = 0.5
        preds = (all_probs >= threshold).astype(int)
        tp = ((preds == 1) & (all_targets == 1)).sum()
        fp = ((preds == 1) & (all_targets == 0)).sum()
        fn = ((preds == 0) & (all_targets == 1)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        pos_rate = all_targets.mean()

        try:
            from sklearn.metrics import average_precision_score
            prauc = average_precision_score(all_targets, all_probs) if all_targets.sum() > 0 else 0.0
        except ImportError:
            prauc = 0.0

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)
        history['val_prauc'].append(prauc)

        current_lr = optimizer.param_groups[0]['lr']

        log.info(
            f"Epoch {epoch+1}/{epochs} | Loss T:{avg_train_loss:.4f} V:{avg_val_loss:.4f} | "
            f"P:{precision:.1%} R:{recall:.1%} F1:{f1:.1%} | PR-AUC:{prauc:.3f} | "
            f"Pos:{pos_rate:.1%} | Pred1:{preds.mean():.1%} | LR:{current_lr:.2e}"
        )

        if prauc > best_val_prauc:
            best_val_prauc = prauc
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_dim': input_dim,
                    'hidden_dims': [512, 256, 128, 64],
                    'num_classes': 3,
                    'dropout': 0.3,
                    'use_layer_norm': True,
                    'use_residual': True,
                    'enable_enter_head': True,
                    'enable_quantile_head': False,
                    'enable_vol_state_head': False,
                    'enable_mu_head': False,
                    'enable_sigma_head': False,
                },
                'feature_columns': list(features_df.columns),
                'n_features': input_dim,
                'feature_version': FEATURE_VERSION,
                'model_type': 'enter_quality',
                'barrier_config': {'tp_mult': tp_mult, 'sl_mult': sl_mult, 'horizon': horizon, 'slope_eps': slope_eps},
                'best_prauc': best_val_prauc,
                'trained_at': datetime.now().isoformat(),
            }, checkpoint_dir / "best_enter_prauc.pt")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_dim': input_dim,
                    'hidden_dims': [512, 256, 128, 64],
                    'num_classes': 3,
                    'dropout': 0.3,
                    'use_layer_norm': True,
                    'use_residual': True,
                    'enable_enter_head': True,
                    'enable_quantile_head': False,
                    'enable_vol_state_head': False,
                    'enable_mu_head': False,
                    'enable_sigma_head': False,
                },
                'feature_columns': list(features_df.columns),
                'n_features': input_dim,
                'feature_version': FEATURE_VERSION,
                'model_type': 'enter_quality',
                'barrier_config': {'tp_mult': tp_mult, 'sl_mult': sl_mult, 'horizon': horizon, 'slope_eps': slope_eps},
                'best_val_loss': best_val_loss,
                'trained_at': datetime.now().isoformat(),
            }, checkpoint_dir / "best_enter_loss.pt")
        else:
            patience += 1

        if epoch + 1 >= min_epochs and patience >= max_patience:
            log.info(f"Early stopping at epoch {epoch+1} (patience={max_patience})")
            break

        MONITORING_INTERVAL = 5
        if (epoch + 1) % MONITORING_INTERVAL == 0:
            _run_enter_trading_sweep(all_probs, all_targets, all_sides, all_val_returns, epoch + 1, tp_mult, sl_mult)

        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0 and (epoch + 1) < epochs:
            log.info("=" * 60)
            log.info(f"  CHECKPOINT @ Epoch {epoch+1}/{epochs}")
            log.info("=" * 60)
            log.info(f"  Val Loss: {avg_val_loss:.4f} | Best: {best_val_loss:.4f}")
            log.info(f"  PR-AUC: {prauc:.3f} | Best: {best_val_prauc:.3f}")
            log.info(f"  Patience: {patience}/{max_patience}")
            log.info(f"  P:{precision:.1%} R:{recall:.1%} F1:{f1:.1%}")
            try:
                resp = input("Continue training? (Y/n): ").strip().lower()
                if resp == 'n':
                    log.info("User stopped training at checkpoint")
                    break
            except EOFError:
                pass

    scaler_path = checkpoint_dir / "scaler.joblib"
    engineer.save_scalers(str(scaler_path))
    log.info(f"Scaler saved to {scaler_path}")

    best_ckpt = checkpoint_dir / "best_enter_prauc.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        log.info(f"Loaded best PR-AUC checkpoint (PR-AUC={best_val_prauc:.3f})")

    return model, engineer, list(features_df.columns), history


def _run_enter_trading_sweep(probs, targets, sides, returns, epoch, tp_mult, sl_mult):
    import numpy as np
    THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    COOLDOWN = 4
    FIXED_COST = 0.0009
    MIN_TRADES = 30

    rolling_window = 20
    data_atr = np.full_like(returns, max(np.std(returns), 1e-6))
    for i in range(rolling_window, len(returns)):
        data_atr[i] = max(np.std(returns[i - rolling_window:i]), 1e-6)

    best_metrics = None
    best_score = float('-inf')
    best_thresh = 0.5
    sweep_results = []

    for thresh in THRESHOLDS:
        trade_signal = (probs >= thresh) & (sides != 0)

        final_trades = np.zeros_like(trade_signal, dtype=bool)
        last_trade = -COOLDOWN - 1
        for i in range(len(trade_signal)):
            if trade_signal[i] and (i - last_trade) > COOLDOWN:
                final_trades[i] = True
                last_trade = i

        trade_pnl = []
        for i in range(len(returns)):
            if not final_trades[i]:
                continue
            atr = max(data_atr[i], 1e-6)
            sl_dist = sl_mult * atr
            tp_dist = tp_mult * atr
            actual_ret = returns[i]
            side = sides[i]

            if side > 0:
                if actual_ret <= -sl_dist:
                    pnl = -sl_dist - FIXED_COST
                elif actual_ret >= tp_dist:
                    pnl = tp_dist - FIXED_COST
                else:
                    pnl = actual_ret - FIXED_COST
            else:
                if actual_ret >= sl_dist:
                    pnl = -sl_dist - FIXED_COST
                elif actual_ret <= -tp_dist:
                    pnl = tp_dist - FIXED_COST
                else:
                    pnl = -actual_ret - FIXED_COST
            trade_pnl.append(pnl)

        trade_pnl = np.array(trade_pnl)
        n_trades = len(trade_pnl)
        if n_trades == 0:
            sweep_results.append({'thresh': thresh, 'trades': 0, 'expect': 0, 'winrate': 0, 'sharpe': 0, 'pf': 0})
            continue

        expect = float(np.mean(trade_pnl))
        wins = (trade_pnl > 0).sum()
        winrate = wins / n_trades
        gross_profit = trade_pnl[trade_pnl > 0].sum()
        gross_loss = abs(trade_pnl[trade_pnl < 0].sum())
        pf = float(gross_profit / gross_loss) if gross_loss > 0 else 0.0
        sharpe = float(np.mean(trade_pnl) / np.std(trade_pnl) * np.sqrt(252 * 96)) if np.std(trade_pnl) > 0 and n_trades > 1 else 0.0

        m = {'thresh': thresh, 'trades': n_trades, 'expect': expect, 'winrate': winrate, 'sharpe': sharpe, 'pf': pf}
        sweep_results.append(m)

        if n_trades >= MIN_TRADES and expect > best_score:
            best_score = expect
            best_metrics = m
            best_thresh = thresh

    log.info("-" * 70)
    log.info("ENTER TRADING SWEEP (epoch %d) | cooldown=%d bars | p_enter threshold", epoch, COOLDOWN)
    log.info("%-8s %6s %8s %7s %7s %8s", "Thresh", "Trades", "Expect", "WinRate", "Sharpe", "PF")
    log.info("-" * 70)
    for m in sweep_results:
        marker = " << BEST" if m['thresh'] == best_thresh and m['trades'] >= MIN_TRADES and best_score > float('-inf') else ""
        log.info("%-8.0f%% %5d  %+.4f  %5.1f%%  %+5.2f   %5.2f%s",
                 m['thresh'] * 100, m['trades'], m['expect'], m['winrate'] * 100, m['sharpe'], m['pf'], marker)
    log.info("-" * 70)


def make_enter_prediction(model, engineer, feature_columns, data_path, device):
    import torch
    import numpy as np
    import pandas as pd

    log.info("Generating ENTER QUALITY prediction from latest data...")

    df = pd.read_parquet(data_path)
    from data.pipeline import FeatureEngineer
    feat_engineer = FeatureEngineer()

    if len(feature_columns) != FeatureEngineer.TOTAL_FEATURE_COUNT:
        raise RuntimeError(
            f"FATAL: feature_columns has {len(feature_columns)} cols, expected {FeatureEngineer.TOTAL_FEATURE_COUNT}. "
            f"Checkpoint mismatch - retrain the model."
        )

    features_df = feat_engineer.compute_all_features(df)
    features_df = features_df.fillna(0)

    missing = set(feature_columns) - set(features_df.columns)
    extra = set(features_df.columns) - set(feature_columns)
    if missing or extra:
        log.error(f"FATAL: Feature column mismatch!")
        if missing:
            log.error(f"  Missing: {sorted(missing)}")
        if extra:
            log.error(f"  Extra: {sorted(extra)}")
        raise RuntimeError(f"Feature column mismatch: {len(missing)} missing, {len(extra)} extra. Retrain.")

    features_df = features_df.reindex(columns=feature_columns, fill_value=0)

    last_features = features_df.iloc[-1:].copy()
    last_scaled = engineer.transform_and_clip(
        pd.DataFrame(last_features.values, columns=feature_columns),
        clip_range=5.0
    ).values.astype(np.float32)
    last_scaled = np.where(np.isinf(last_scaled), 0, last_scaled)
    last_scaled = np.where(np.isnan(last_scaled), 0, last_scaled)

    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(last_scaled).to(device)
        output = model.forward_multihead(x)

    p_enter = float(torch.sigmoid(output.enter_logits).cpu().item())

    last_row = features_df.iloc[-1]
    h1_trend = last_row.get('h1_trend_sign', 0)
    h4_trend = last_row.get('h4_trend_sign', 0)
    h1_slope = last_row.get('h1_sma20_slope', 0)
    h1_range_pos = last_row.get('h1_range_pos', 0.5)

    trend_aligned = (h1_trend == h4_trend) and (h1_trend != 0)
    slope_ok = abs(h1_slope) > 0.05
    range_ok = True
    if h1_trend > 0 and h1_range_pos < 0.2:
        range_ok = False
    if h1_trend < 0 and h1_range_pos > 0.8:
        range_ok = False

    if h1_trend > 0:
        side = "LONG"
    elif h1_trend < 0:
        side = "SHORT"
    else:
        side = "NEUTRAL"

    current_price = float(df.iloc[-1]['close'])

    atr_window = min(20, len(df) - 1)
    if atr_window < 2:
        atr = current_price * 0.005
    else:
        highs = df.iloc[-atr_window:]['high'].values
        lows = df.iloc[-atr_window:]['low'].values
        true_ranges = []
        for i in range(1, len(highs)):
            prev_close = float(df.iloc[-atr_window + i - 1]['close'])
            tr = max(float(highs[i]) - float(lows[i]),
                     abs(float(highs[i]) - prev_close),
                     abs(float(lows[i]) - prev_close))
            true_ranges.append(tr)
        atr = float(np.mean(true_ranges))

    enter_threshold = 0.55
    should_trade = p_enter >= enter_threshold and trend_aligned and slope_ok and range_ok

    if should_trade:
        action = side
    else:
        action = "HOLD"

    if action == "LONG":
        sl_price = current_price - 1.5 * atr
        tp_price = current_price + 2.0 * atr
    elif action == "SHORT":
        sl_price = current_price + 1.5 * atr
        tp_price = current_price - 2.0 * atr
    else:
        sl_price = current_price - 1.0 * atr
        tp_price = current_price + 1.0 * atr

    sl_pct = abs(current_price - sl_price) / current_price
    tp_pct = abs(tp_price - current_price) / current_price
    rr = tp_pct / sl_pct if sl_pct > 0 else 1.0

    ACCOUNT_RISK_PER_TRADE = 0.02
    if sl_pct > 0:
        position_size = ACCOUNT_RISK_PER_TRADE / sl_pct * 100
    else:
        position_size = 1.0
    if p_enter > 0.75:
        position_size *= 1.25
    elif p_enter < 0.55:
        position_size *= 0.5
    position_size = min(max(position_size, 0.5), 5.0)

    confidence = p_enter
    edge = p_enter - 0.5

    prediction = {
        "action": action,
        "confidence": round(confidence, 4),
        "direction_probs": {"SHORT": round(1.0 if side == "SHORT" else 0.0, 4),
                            "HOLD": round(1.0 if action == "HOLD" else 0.0, 4),
                            "LONG": round(1.0 if side == "LONG" else 0.0, 4)},
        "quantiles": {},
        "vol_state": "neutral",
        "vol_state_probs": {"contraction": 0.33, "neutral": 0.34, "expansion": 0.33},
        "expected_return": round(edge, 6),
        "uncertainty": round(1.0 - p_enter, 6),
        "edge": round(edge, 4),
        "entry_price": round(current_price, 2),
        "stop_loss_price": round(sl_price, 2),
        "take_profit_price": round(tp_price, 2),
        "stop_loss_pct": round(sl_pct, 4),
        "take_profit_pct": round(tp_pct, 4),
        "risk_reward_ratio": round(rr, 2),
        "position_size_pct": round(position_size, 1),
        "current_price": round(current_price, 2),
        "model_name": "enter_quality_v3.1",
        "is_multihead": True,
        "urgency": "high" if p_enter > 0.7 and should_trade else ("medium" if should_trade else "low"),
        "suggested_order_type": "limit",
        "reasons": [],
    }

    reasons = []
    if should_trade:
        reasons.append(f"ENTER signal: p_enter={p_enter:.1%}")
        reasons.append(f"HTF trend: {side} (1H={h1_trend:+.0f}, 4H={h4_trend:+.0f})")
        if abs(h1_slope) > 0.1:
            reasons.append(f"Strong trend slope ({h1_slope:.2f})")
    else:
        if not trend_aligned:
            reasons.append("HTF trends not aligned")
        if not slope_ok:
            reasons.append(f"Weak slope ({h1_slope:.2f})")
        if not range_ok:
            reasons.append(f"Range position against trend ({h1_range_pos:.2f})")
        if p_enter < enter_threshold:
            reasons.append(f"p_enter {p_enter:.1%} < {enter_threshold:.0%} threshold")

    prediction["reasons"] = reasons if reasons else ["No signal"]

    return prediction


def push_prediction(replit_url: str, prediction: dict):
    import requests

    url = f"{replit_url.rstrip('/')}/api/gpu/push-prediction"
    log.info(f"Pushing prediction to dashboard...")
    log.info(f"  Action: {prediction['action']} | Confidence: {prediction['confidence']:.1%}")
    log.info(f"  Price: ${prediction['current_price']:,.2f}")
    log.info(f"  Entry: ${prediction['entry_price']:,.2f} | SL: ${prediction['stop_loss_price']:,.2f} | TP: ${prediction['take_profit_price']:,.2f}")

    try:
        resp = requests.post(url, json=prediction, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        log.info(f"  Pushed successfully! (id={result.get('id', '?')})")
        return True
    except Exception as e:
        log.error(f"  Failed to push: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="BTC Futures GPU Trainer - ENTER QUALITY Model (v3.1.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_start.py --url https://your-app.replit.app
  python quick_start.py --url https://your-app.replit.app --epochs 300
  python quick_start.py --url https://your-app.replit.app --predict-only
        """
    )
    parser.add_argument("--url", required=True, help="Your Replit dashboard URL")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs (default: 300)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="LR warmup epochs (default: 5)")
    parser.add_argument("--min-lr", type=float, default=None, help="Min LR for cosine annealing")
    parser.add_argument("--predict-only", action="store_true", help="Skip training, predict from saved model")
    parser.add_argument("--no-push", action="store_true", help="Train but don't push prediction")
    parser.add_argument("--checkpoint-interval", type=int, default=25, help="Pause every N epochs (0=no pausing)")
    parser.add_argument("--tp-mult", type=float, default=2.0, help="TP ATR multiplier (default: 2.0)")
    parser.add_argument("--sl-mult", type=float, default=1.5, help="SL ATR multiplier (default: 1.5)")
    parser.add_argument("--horizon", type=int, default=24, help="Horizon bars (default: 24)")
    parser.add_argument("--slope-eps", type=float, default=0.05, help="Min slope for trend gate (default: 0.05)")

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  BTC FUTURES - ENTER QUALITY MODEL v3.1.0")
    print("=" * 60)
    print()

    device = check_gpu()
    data_dir = Path("data_cache")

    if not args.predict_only:
        data_path = download_data(args.url, data_dir)

        model, engineer, feature_columns, history = train_enter_model(
            data_path, device, args.epochs, args.batch_size, args.lr,
            checkpoint_interval=args.checkpoint_interval,
            warmup_epochs=args.warmup_epochs, min_lr=args.min_lr,
            tp_mult=args.tp_mult, sl_mult=args.sl_mult,
            horizon=args.horizon, slope_eps=args.slope_eps,
        )

        print()
        log.info("=" * 60)
        log.info("  TRAINING COMPLETE")
        log.info("=" * 60)

        if history['val_loss']:
            best_loss = min(history['val_loss'])
            best_prauc = max(history['val_prauc']) if history['val_prauc'] else 0
            log.info(f"  Best val loss: {best_loss:.4f}")
            log.info(f"  Best PR-AUC: {best_prauc:.3f}")
            log.info(f"  Epochs trained: {len(history['val_loss'])}")
    else:
        import torch
        checkpoint_path = Path("checkpoints/best_enter_prauc.pt")
        if not checkpoint_path.exists():
            checkpoint_path = Path("checkpoints/best_enter_loss.pt")
        if not checkpoint_path.exists():
            log.error("No trained ENTER model found! Run without --predict-only first.")
            sys.exit(1)

        log.info("Downloading fresh data for prediction...")
        data_path = download_data(args.url, data_dir, force_fresh=True)

        log.info("Loading saved model...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        saved_version = checkpoint.get('feature_version', 'unknown')
        if saved_version != FEATURE_VERSION:
            log.error(f"FATAL: Feature version mismatch! Model: '{saved_version}', current: '{FEATURE_VERSION}'")
            sys.exit(1)
        log.info(f"Feature version: {saved_version} (matches)")

        feature_columns = checkpoint.get('feature_columns', [])
        if not feature_columns:
            log.error("FATAL: No feature_columns in checkpoint - retrain.")
            sys.exit(1)

        from models.simple_mlp import EnhancedMultiHeadMLP, EnhancedMultiHeadMLP_Config
        cfg = checkpoint.get('model_config', {})
        mlp_config = EnhancedMultiHeadMLP_Config(
            input_dim=cfg.get('input_dim', 57),
            hidden_dims=cfg.get('hidden_dims', [512, 256, 128, 64]),
            num_classes=3,
            dropout=0.3,
            use_layer_norm=True,
            use_residual=True,
            enable_enter_head=True,
            enable_quantile_head=False,
            enable_vol_state_head=False,
            enable_mu_head=False,
            enable_sigma_head=False,
        )
        model = EnhancedMultiHeadMLP(mlp_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        from data.pipeline import FeatureEngineer
        engineer = FeatureEngineer()
        scaler_path = Path("checkpoints/scaler.joblib")
        if scaler_path.exists():
            engineer.load_scalers(str(scaler_path))
        else:
            log.warning("No saved scaler found - prediction quality may be reduced")

    if not args.no_push:
        prediction = make_enter_prediction(model, engineer, feature_columns, data_path, device)

        print()
        log.info("=" * 60)
        log.info(f"  SIGNAL: {prediction['action']} | p_enter: {prediction['confidence']:.1%}")
        log.info("=" * 60)
        log.info(f"  Price: ${prediction['current_price']:,.2f}")
        log.info(f"  Entry: ${prediction['entry_price']:,.2f}")
        log.info(f"  SL:    ${prediction['stop_loss_price']:,.2f} ({prediction['stop_loss_pct']:.2%})")
        log.info(f"  TP:    ${prediction['take_profit_price']:,.2f} ({prediction['take_profit_pct']:.2%})")
        log.info(f"  R:R = {prediction['risk_reward_ratio']:.1f} | Position: {prediction['position_size_pct']:.1f}%")
        if prediction.get('reasons'):
            log.info(f"  Reasons: {', '.join(prediction['reasons'])}")
        log.info("=" * 60)

        is_hold = prediction['action'] == "HOLD"
        if is_hold:
            log.info("Signal: HOLD - pushing to dashboard (no trade)")
        else:
            log.info(f"Signal: {prediction['action']} PASSED - pushing to dashboard")
        push_prediction(args.url, prediction)
    else:
        log.info("Skipping prediction push (--no-push)")

    print()
    log.info("Done! Check your dashboard to see the prediction.")
    print()


if __name__ == "__main__":
    main()
