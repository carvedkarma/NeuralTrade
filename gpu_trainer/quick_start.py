#!/usr/bin/env python3
"""
BTC Futures GPU Trainer - Quick Start
=====================================
One-script setup: Downloads data from your Replit dashboard,
trains the model on your GPU, and pushes predictions back.

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


def download_data(replit_url: str, data_dir: Path):
    import requests

    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "BTCUSDT_15m.csv"
    parquet_path = data_dir / "BTCUSDT_15m.parquet"

    if parquet_path.exists():
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


def train_model(data_path: Path, device: str, epochs: int, batch_size: int, lr: float):
    import torch
    import numpy as np
    import pandas as pd
    from config import config

    log.info("=" * 60)
    log.info("  STARTING TRAINING")
    log.info("=" * 60)

    df = pd.read_parquet(data_path)
    log.info(f"Loaded {len(df)} candles")

    from data.pipeline import FeatureEngineer, create_labels
    engineer = FeatureEngineer()
    features_df = engineer.compute_technical_features(df)
    features_df = features_df.fillna(0)
    log.info(f"Computed {len(features_df.columns)} features")

    horizon = 16
    from data.regression_targets import generate_multihead_targets
    targets_df = generate_multihead_targets(
        df, horizon_periods=horizon, n_future_candles=5,
        use_pure_directional=True, directional_threshold=0.0020
    )

    labels = targets_df['class_label'].values.astype(np.int64)
    forward_returns = targets_df['forward_return'].values.astype(np.float32)

    entry_offset = targets_df['entry_offset'].values.astype(np.float32)
    sl_distance = targets_df['sl_distance'].values.astype(np.float32)
    tp_distance = targets_df['tp_distance'].values.astype(np.float32)

    candle_cols = []
    for i in range(1, 6):
        candle_cols.extend([
            f"candle_delta_close_{i}",
            f"candle_delta_high_{i}",
            f"candle_delta_low_{i}"
        ])
    candle_targets = targets_df[candle_cols].values.astype(np.float32)

    sequence_length = config.data.sequence_length
    valid_start = sequence_length
    features_np = features_df.values[valid_start:].astype(np.float32)
    labels_np = labels[valid_start:].astype(np.int64)
    forward_returns_np = forward_returns[valid_start:].astype(np.float32)
    entry_offset_np = entry_offset[valid_start:].astype(np.float32)
    sl_distance_np = sl_distance[valid_start:].astype(np.float32)
    tp_distance_np = tp_distance[valid_start:].astype(np.float32)
    candle_targets_np = candle_targets[valid_start:].astype(np.float32)

    n_total = len(features_np)
    purge_gap = horizon + sequence_length
    val_samples = max(int(n_total * 0.1), purge_gap)
    train_samples = n_total - purge_gap - val_samples

    if train_samples < sequence_length * 3:
        log.error(f"Not enough data for training: {train_samples} samples")
        sys.exit(1)

    train_end = train_samples
    val_start = train_end + purge_gap
    val_end = val_start + val_samples

    log.info(f"Data split: train={train_samples}, purge={purge_gap}, val={val_samples}")

    train_features_raw = features_np[:train_end]
    train_labels = labels_np[:train_end]
    val_features_raw = features_np[val_start:val_end]
    val_labels = labels_np[val_start:val_end]

    train_returns = forward_returns_np[:train_end]
    val_returns = forward_returns_np[val_start:val_end]
    train_entry = entry_offset_np[:train_end]
    train_sl = sl_distance_np[:train_end]
    train_tp = tp_distance_np[:train_end]
    val_entry = entry_offset_np[val_start:val_end]
    val_sl = sl_distance_np[val_start:val_end]
    val_tp = tp_distance_np[val_start:val_end]
    train_candles = candle_targets_np[:train_end]
    val_candles = candle_targets_np[val_start:val_end]

    train_features_df = pd.DataFrame(train_features_raw, columns=features_df.columns)
    engineer.fit_scalers(train_features_df)
    clip_range = 5.0
    train_scaled = engineer.transform_and_clip(train_features_df, clip_range=clip_range).values.astype(np.float32)
    val_features_df = pd.DataFrame(val_features_raw, columns=features_df.columns)
    val_scaled = engineer.transform_and_clip(val_features_df, clip_range=clip_range).values.astype(np.float32)

    def clean(features, labels, returns, entry, sl, tp, candles, name):
        features = np.where(np.isinf(features), np.nan, features)
        returns = np.where(np.isinf(returns), np.nan, returns)
        entry = np.where(np.isinf(entry), np.nan, entry)
        sl = np.where(np.isinf(sl), np.nan, sl)
        tp = np.where(np.isinf(tp), np.nan, tp)
        candles = np.where(np.isinf(candles), np.nan, candles)

        mask = np.isnan(features).any(axis=1) | np.isnan(returns)
        mask = mask | np.isnan(entry) | np.isnan(sl) | np.isnan(tp)
        mask = mask | np.isnan(candles).any(axis=1)
        valid = ~mask
        dropped = mask.sum()
        if dropped > 0:
            log.info(f"  {name}: dropped {dropped} NaN rows")
        return (features[valid], labels[valid], returns[valid],
                entry[valid], sl[valid], tp[valid], candles[valid])

    train_scaled, train_labels, train_returns, train_entry, train_sl, train_tp, train_candles = \
        clean(train_scaled, train_labels, train_returns, train_entry, train_sl, train_tp, train_candles, "Train")
    val_scaled, val_labels, val_returns, val_entry, val_sl, val_tp, val_candles = \
        clean(val_scaled, val_labels, val_returns, val_entry, val_sl, val_tp, val_candles, "Val")

    from training.multihead_trainer import MultiHeadTrainer, MultiHeadDataset, MultiHeadLossConfig
    from torch.utils.data import DataLoader

    train_dataset = MultiHeadDataset(
        train_scaled, train_labels, train_returns,
        entry_offset=train_entry, sl_distance=train_sl, tp_distance=train_tp,
        candle_targets=train_candles, regime_ids=None, n_future_candles=5,
        sequence_length=sequence_length
    )
    val_dataset = MultiHeadDataset(
        val_scaled, val_labels, val_returns,
        entry_offset=val_entry, sl_distance=val_sl, tp_distance=val_tp,
        candle_targets=val_candles, regime_ids=None, n_future_candles=5,
        sequence_length=sequence_length
    )

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
        enable_quantile_head=True,
        enable_vol_state_head=True,
        enable_mu_head=True,
        enable_sigma_head=True
    )
    model = EnhancedMultiHeadMLP(mlp_config)
    model.name = "EnhancedMultiHeadMLP"
    model.count_parameters = model.parameters_count
    log.info(f"Model: EnhancedMultiHeadMLP ({model.parameters_count():,} parameters)")
    log.info(f"Architecture: [512, 256, 128, 64] with residual connections")
    log.info(f"Heads: Classification + Quantile + VolState + Mu + Sigma (all 5)")

    unique_labels, label_counts = np.unique(train_labels, return_counts=True)
    total = len(train_labels)
    MAX_CLASS_WEIGHT = 10.0
    class_weights_list = []
    for c in range(3):
        if c in unique_labels:
            idx = np.where(unique_labels == c)[0][0]
            w = min(total / (3 * label_counts[idx]), MAX_CLASS_WEIGHT)
        else:
            w = 1.0
        class_weights_list.append(w)
    class_weights = torch.FloatTensor(class_weights_list)
    log.info(f"Class weights: {class_weights.numpy()}")

    loss_config = MultiHeadLossConfig(
        class_weights=class_weights,
        use_focal_loss=True,
        focal_gamma=2.0,
        lambda_quantile=0.3,
        lambda_mu=0.3,
        lambda_sigma=0.2,
        lambda_vol_state=0.2,
        head_enabled_quantile=True,
        head_enabled_vol_state=True,
        head_enabled_mu=True,
        head_enabled_sigma=True,
    )

    config.training.epochs = epochs
    config.training.learning_rate = lr

    trainer = MultiHeadTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        loss_config=loss_config,
    )

    log.info(f"Training for {epochs} epochs (lr={lr}, batch={batch_size})...")
    log.info("-" * 60)
    history = trainer.train(num_epochs=epochs)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / "best_enhanced_mlp.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': input_dim,
            'hidden_dims': [512, 256, 128, 64],
            'num_classes': 3,
            'dropout': 0.3,
            'use_layer_norm': True,
            'use_residual': True,
            'enable_quantile_head': True,
            'enable_vol_state_head': True,
            'enable_mu_head': True,
            'enable_sigma_head': True,
        },
        'feature_columns': list(features_df.columns),
        'n_features': input_dim,
        'trained_at': datetime.now().isoformat(),
    }, save_path)
    log.info(f"Model saved to {save_path}")

    scaler_path = checkpoint_dir / "scaler.joblib"
    engineer.save_scalers(str(scaler_path))
    log.info(f"Scaler saved to {scaler_path}")

    return model, engineer, features_df.columns.tolist(), history


def make_prediction(model, engineer, feature_columns, data_path, device):
    import torch
    import numpy as np
    import pandas as pd
    from config import config

    log.info("Generating prediction from latest data...")

    df = pd.read_parquet(data_path)
    from data.pipeline import FeatureEngineer
    feat_engineer = FeatureEngineer()
    features_df = feat_engineer.compute_technical_features(df)
    features_df = features_df.fillna(0)

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
        output = model(x)

    current_price = float(df.iloc[-1]['close'])

    probs = torch.softmax(output.classification, dim=-1).cpu().numpy()[0]
    action_idx = int(np.argmax(probs))
    action_map = {0: "SHORT", 1: "HOLD", 2: "LONG"}
    action = action_map[action_idx]
    confidence = float(probs[action_idx])

    direction_probs = {
        "SHORT": float(probs[0]),
        "HOLD": float(probs[1]),
        "LONG": float(probs[2]),
    }

    quantiles = {}
    if output.quantiles is not None:
        q = output.quantiles.cpu().numpy()[0]
        quantile_keys = ["q10", "q25", "q50", "q75", "q90"]
        for i, key in enumerate(quantile_keys):
            if i < len(q):
                quantiles[key] = float(q[i])

    vol_state = "neutral"
    vol_state_probs = {"contraction": 0.33, "neutral": 0.34, "expansion": 0.33}
    if output.vol_state is not None:
        vs_probs = torch.softmax(output.vol_state, dim=-1).cpu().numpy()[0]
        vol_states = ["contraction", "neutral", "expansion"]
        vol_state = vol_states[int(np.argmax(vs_probs))]
        vol_state_probs = {s: float(vs_probs[i]) for i, s in enumerate(vol_states)}

    mu_val = 0.0
    if output.mu is not None:
        mu_val = float(output.mu.cpu().numpy()[0])

    sigma_val = 0.01
    if output.sigma is not None:
        sigma_val = float(abs(output.sigma.cpu().numpy()[0]))
        if sigma_val < 0.0001:
            sigma_val = 0.01

    edge = abs(mu_val) / sigma_val if sigma_val > 0 else 0
    edge = min(edge, 1.0)

    atr = float(df.iloc[-20:]['high'].max() - df.iloc[-20:]['low'].min()) / 20
    if action == "LONG":
        sl_price = current_price - 2 * atr
        tp_price = current_price + 3 * atr
    elif action == "SHORT":
        sl_price = current_price + 2 * atr
        tp_price = current_price - 3 * atr
    else:
        sl_price = current_price - 1.5 * atr
        tp_price = current_price + 1.5 * atr

    sl_pct = abs(current_price - sl_price) / current_price
    tp_pct = abs(tp_price - current_price) / current_price
    rr = tp_pct / sl_pct if sl_pct > 0 else 1.0

    position_size = min(max(confidence * 20, 5), 25)

    prediction = {
        "action": action,
        "confidence": round(confidence, 4),
        "direction_probs": {k: round(v, 4) for k, v in direction_probs.items()},
        "quantiles": {k: round(v, 6) for k, v in quantiles.items()},
        "vol_state": vol_state,
        "vol_state_probs": {k: round(v, 4) for k, v in vol_state_probs.items()},
        "expected_return": round(mu_val, 6),
        "uncertainty": round(sigma_val, 6),
        "edge": round(edge, 4),
        "entry_price": round(current_price, 2),
        "stop_loss_price": round(sl_price, 2),
        "take_profit_price": round(tp_price, 2),
        "stop_loss_pct": round(sl_pct, 4),
        "take_profit_pct": round(tp_pct, 4),
        "risk_reward_ratio": round(rr, 2),
        "position_size_pct": round(position_size, 1),
        "current_price": round(current_price, 2),
        "model_name": "enhanced_mlp_quickstart",
        "is_multihead": True,
        "urgency": "medium" if confidence > 0.6 else "low",
        "suggested_order_type": "limit",
        "reasons": [],
    }

    reasons = []
    if action == "LONG" and probs[2] > 0.5:
        reasons.append("Strong bullish classification")
    elif action == "SHORT" and probs[0] > 0.5:
        reasons.append("Strong bearish classification")
    if mu_val > 0.001:
        reasons.append("Positive expected return")
    elif mu_val < -0.001:
        reasons.append("Negative expected return")
    if edge > 0.3:
        reasons.append(f"Favorable edge ({edge:.2f})")
    if vol_state == "expansion":
        reasons.append("Volatility expansion detected")
    prediction["reasons"] = reasons if reasons else ["Model prediction"]

    return prediction


def push_prediction(replit_url: str, prediction: dict):
    import requests

    url = f"{replit_url.rstrip('/')}/api/gpu/push-prediction"
    log.info(f"Pushing prediction to dashboard...")
    log.info(f"  Action: {prediction['action']} | Confidence: {prediction['confidence']:.1%}")
    log.info(f"  Price: ${prediction['current_price']:,.2f}")
    log.info(f"  Entry: ${prediction['entry_price']:,.2f} | SL: ${prediction['stop_loss_price']:,.2f} | TP: ${prediction['take_profit_price']:,.2f}")
    log.info(f"  Vol State: {prediction['vol_state']}")

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
        description="BTC Futures GPU Trainer - Quick Start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_start.py --url https://your-app.replit.app
  python quick_start.py --url https://your-app.replit.app --epochs 100
  python quick_start.py --url https://your-app.replit.app --predict-only
        """
    )
    parser.add_argument("--url", required=True, help="Your Replit dashboard URL (e.g. https://your-app.replit.app)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--predict-only", action="store_true", help="Skip training, just make a prediction from existing model")
    parser.add_argument("--no-push", action="store_true", help="Train but don't push prediction to dashboard")

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  BTC FUTURES GPU TRAINER - QUICK START")
    print("=" * 60)
    print()

    device = check_gpu()
    data_dir = Path("data_cache")

    if not args.predict_only:
        data_path = download_data(args.url, data_dir)

        model, engineer, feature_columns, history = train_model(
            data_path, device, args.epochs, args.batch_size, args.lr
        )

        print()
        log.info("=" * 60)
        log.info("  TRAINING COMPLETE")
        log.info("=" * 60)
    else:
        import torch
        checkpoint_path = Path("checkpoints/best_enhanced_mlp.pt")
        if not checkpoint_path.exists():
            log.error("No trained model found! Run without --predict-only first.")
            sys.exit(1)

        data_path = data_dir / "BTCUSDT_15m.parquet"
        if not data_path.exists():
            data_path = download_data(args.url, data_dir)

        log.info("Loading saved model...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        from models.simple_mlp import EnhancedMultiHeadMLP, EnhancedMultiHeadMLP_Config
        cfg = checkpoint.get('model_config', {})
        mlp_config = EnhancedMultiHeadMLP_Config(
            input_dim=cfg.get('input_dim', 41),
            hidden_dims=cfg.get('hidden_dims', [512, 256, 128, 64]),
            num_classes=3,
            dropout=0.3,
            use_layer_norm=True,
            use_residual=True,
            enable_quantile_head=True,
            enable_vol_state_head=True,
            enable_mu_head=True,
            enable_sigma_head=True,
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

        feature_columns = checkpoint.get('feature_columns', [])

    if not args.no_push:
        prediction = make_prediction(model, engineer, feature_columns, data_path, device)

        print()
        log.info("-" * 40)
        log.info(f"PREDICTION: {prediction['action']} ({prediction['confidence']:.1%})")
        log.info(f"  Price: ${prediction['current_price']:,.2f}")
        log.info(f"  SL: ${prediction['stop_loss_price']:,.2f} | TP: ${prediction['take_profit_price']:,.2f}")
        log.info(f"  R:R = {prediction['risk_reward_ratio']:.1f}")
        log.info(f"  Vol State: {prediction['vol_state']}")
        log.info(f"  Edge: {prediction['edge']:.2f}")
        log.info("-" * 40)

        push_prediction(args.url, prediction)
    else:
        log.info("Skipping prediction push (--no-push)")

    print()
    log.info("Done! Check your dashboard to see the prediction.")
    print()


if __name__ == "__main__":
    main()
