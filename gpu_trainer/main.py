#!/usr/bin/env python3
"""
BTC Futures Trading - GPU Neural Network Trainer

This is the main entry point for training deep learning models
on your local GPU for cryptocurrency trading signals.

Usage:
    python main.py train --model transformer --epochs 100
    python main.py serve --port 8000
    python main.py backtest --start 2024-01-01 --end 2024-12-31
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_gpu():
    """Check GPU availability and print info."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU Available: {gpu_name} ({gpu_memory:.1f} GB)")
            return True
        else:
            logger.warning("No GPU available. Training will use CPU (slower).")
            return False
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install -r requirements.txt")
        return False

async def fetch_data(args):
    """Fetch historical data from Binance (or via Replit proxy if configured)."""
    from data.pipeline import BinanceDataFetcher
    from config import config
    
    replit_url = getattr(args, 'replit_proxy', None) or config.replit_proxy_url
    
    if replit_url:
        logger.info(f"Using Replit proxy at: {replit_url}")
    else:
        logger.info("No Replit proxy configured. Trying direct Binance access...")
        logger.info("Tip: Set REPLIT_PROXY_URL or use --replit-proxy <url>")
    
    fetcher = BinanceDataFetcher(
        config.data.symbols, 
        config.data.timeframes,
        replit_proxy_url=replit_url
    )
    
    try:
        # Use bulk download from Replit if proxy is configured (much faster)
        if replit_url:
            logger.info("Attempting bulk download from Replit (faster)...")
            data = fetcher.fetch_bulk_from_replit()
            
            # Check if we got any data
            has_data = any(
                any(len(df) > 0 for df in tfs.values())
                for tfs in data.values()
            ) if data else False
            
            if not has_data:
                logger.warning("Bulk download empty, falling back to individual fetches...")
                data = await fetcher.fetch_all_historical(args.candles)
        else:
            data = await fetcher.fetch_all_historical(args.candles)
        
        total_candles = 0
        for symbol, timeframes in data.items():
            for tf, df in timeframes.items():
                if len(df) > 0:
                    path = config.data_dir / f"{symbol}_{tf}.parquet"
                    df.to_parquet(path)
                    total_candles += len(df)
                    logger.info(f"Saved {len(df)} candles for {symbol} {tf}")
                else:
                    logger.warning(f"No data received for {symbol} {tf}")
        
        if total_candles > 0:
            logger.info(f"Data fetch complete! Total: {total_candles} candles")
        else:
            logger.error("No data was fetched. Check your connection or Replit proxy URL.")
        
    finally:
        await fetcher.close()

def train(args):
    """Train a neural network model.
    
    IMPORTANT: This function implements proper train/val separation to prevent data leakage:
    1. Chronological split FIRST (before any scaling)
    2. Fit scalers ONLY on training data
    3. Purge gap at train/val boundary to prevent lookahead from label computation
    """
    import torch
    import numpy as np
    import pandas as pd
    from config import config
    from data.pipeline import FeatureEngineer, TradingDataset, create_labels
    from torch.utils.data import DataLoader
    from training.trainer import Trainer
    
    logger.info(f"Starting training for model: {args.model}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    
    check_gpu()
    
    logger.info("Loading training data...")
    data_path = config.data_dir / "BTCUSDT_15m.parquet"
    
    if data_path.exists():
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} candles from {data_path}")
    else:
        logger.error("No cached data found. Run 'python main.py fetch' first to download data.")
        logger.error("Training on synthetic data produces meaningless models - aborting.")
        return
    
    # === STEP 1: Compute features (before split, features don't leak future) ===
    engineer = FeatureEngineer()
    features_df = engineer.compute_technical_features(df)
    features_df = features_df.fillna(0)
    
    # === STEP 2: Create labels with lookahead (horizon candles into future) ===
    horizon = getattr(args, 'horizon', 5)
    labels = create_labels(df, horizon=horizon, threshold=0.001)
    labels = (labels + 1).astype(int)  # Convert -1/0/1 to 0/1/2
    
    # === STEP 3: CHRONOLOGICAL SPLIT FIRST (before scaling!) ===
    # This prevents scaler from learning distribution info from validation/test data
    sequence_length = config.data.sequence_length
    valid_start = sequence_length  # Skip warmup period for indicators
    
    features_np = features_df.values[valid_start:].astype(np.float32)
    labels_np = labels[valid_start:].astype(np.int64)
    
    # === STEP 4: PURGE GAP and EXPLICIT SPLIT SIZING ===
    # Labels near train end look `horizon` candles ahead, which may be in val
    # Purge gap must be at least horizon + sequence_length to prevent lookahead
    purge_gap = horizon + sequence_length
    
    n_total = len(features_np)
    
    # EXPLICIT SIZING (not implicit remainder)
    # Validation must have at least horizon + sequence_length samples to be meaningful
    min_val_samples = horizon + sequence_length
    min_train_samples = sequence_length * 3  # At least 3x sequence for meaningful training
    
    # Reserve explicit validation window: ~10% of total but at least min_val_samples
    val_samples = max(int(n_total * 0.1), min_val_samples)
    
    # Train gets the rest after subtracting purge gap and validation
    train_samples = n_total - purge_gap - val_samples
    
    # Validate we have enough data
    if train_samples < min_train_samples:
        logger.error(f"Insufficient training data: {train_samples} < {min_train_samples}")
        logger.error(f"  Total: {n_total}, purge_gap: {purge_gap}, val_samples: {val_samples}")
        logger.error(f"  Need at least {min_train_samples + purge_gap + min_val_samples} total samples")
        return
    
    # Compute actual indices
    train_end = train_samples
    val_start = train_end + purge_gap  # Val starts AFTER purge gap
    val_end = val_start + val_samples
    
    # Final bounds check - fail rather than clamp to preserve validation integrity
    if val_end > n_total:
        logger.error(f"Val window exceeds data bounds: val_end={val_end} > n_total={n_total}")
        logger.error(f"  Reduce val_samples or provide more data")
        return
    
    # Log explicit split sizes
    logger.info(f"Data splits (total={n_total}):")
    logger.info(f"  Train: [0, {train_end}) = {train_samples} samples")
    logger.info(f"  Purge: [{train_end}, {val_start}) = {purge_gap} samples (discarded)")
    logger.info(f"  Val:   [{val_start}, {val_end}) = {val_samples} samples")
    tail_discarded = n_total - val_end
    if tail_discarded > 0:
        logger.info(f"  Tail:  [{val_end}, {n_total}) = {tail_discarded} samples (unused)")
    
    # Assert correct layout
    assert train_end + purge_gap == val_start, "Purge gap must be exactly between train and val"
    assert val_end <= n_total, "Val must not exceed data"
    assert val_samples >= min_val_samples, f"Val samples {val_samples} < minimum {min_val_samples}"
    
    # CRITICAL: Verify labels' lookahead never crosses into validation
    # Labels at index i look ahead `horizon` candles to compute target
    # Train labels at train_end-1 look at index train_end-1+horizon
    # This must be strictly less than val_start
    max_label_lookahead = train_end - 1 + horizon
    if max_label_lookahead >= val_start:
        logger.error(f"LEAKAGE DETECTED: Train labels look into validation!")
        logger.error(f"  Train ends at {train_end-1}, label lookahead={horizon}")
        logger.error(f"  Max lookahead index: {max_label_lookahead} >= val_start {val_start}")
        return
    
    logger.info(f"Leakage check PASSED: max_label_lookahead={max_label_lookahead} < val_start={val_start}")
    
    # Split the raw (unscaled) features
    train_features_raw = features_np[:train_end]
    train_labels = labels_np[:train_end]
    val_features_raw = features_np[val_start:val_end]
    val_labels = labels_np[val_start:val_end]
    
    # === STEP 5: FIT SCALER ON TRAINING DATA ONLY ===
    # This is critical - scaler must not see validation/test distribution
    train_features_df = pd.DataFrame(train_features_raw, columns=features_df.columns)
    engineer.fit_scalers(train_features_df)
    logger.info("Scaler fitted on TRAINING data only (no leakage)")
    
    # Transform both train and val with the train-fitted scaler
    train_features_scaled = engineer.transform(train_features_df).values.astype(np.float32)
    val_features_df = pd.DataFrame(val_features_raw, columns=features_df.columns)
    val_features_scaled = engineer.transform(val_features_df).values.astype(np.float32)
    
    # === STEP 6: Create datasets ===
    train_dataset = TradingDataset(train_features_scaled, train_labels, sequence_length)
    val_dataset = TradingDataset(val_features_scaled, val_labels, sequence_length)
    
    # Note: shuffle=True is OK for training since we've already done chronological split
    # and purged the boundary. Shuffling within train set is fine.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    input_dim = features_np.shape[1]
    logger.info(f"Input dimension: {input_dim}, Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    if args.model == "transformer":
        from models.transformer import TransformerPriceModel
        model = TransformerPriceModel(
            input_dim=input_dim,
            d_model=config.model.transformer_dim,
            nhead=config.model.transformer_heads,
            num_layers=config.model.transformer_layers
        )
    elif args.model == "tft":
        from models.transformer import TemporalFusionTransformer
        model = TemporalFusionTransformer(
            input_dim=input_dim,
            d_model=config.model.transformer_dim,
            nhead=config.model.transformer_heads
        )
    elif args.model == "lstm":
        from models.lstm import BidirectionalLSTM
        model = BidirectionalLSTM(
            input_dim=input_dim,
            hidden_dim=config.model.lstm_hidden,
            num_layers=config.model.lstm_layers
        )
    elif args.model == "cnn":
        from models.cnn import ResNetPrice
        model = ResNetPrice(
            input_dim=input_dim,
            channels=config.model.cnn_channels
        )
    elif args.model == "vae":
        from models.vae import MarketVAE
        model = MarketVAE(
            input_dim=input_dim,
            sequence_length=config.data.sequence_length,
            latent_dim=config.model.vae_latent_dim
        )
    elif args.model == "gnn":
        from models.gnn import CrossAssetGNN
        model = CrossAssetGNN(
            input_dim=input_dim,
            num_assets=len(config.data.symbols)
        )
    else:
        logger.error(f"Unknown model type: {args.model}")
        return
        
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    config.training.epochs = args.epochs
    config.training.learning_rate = args.lr
    
    # === STEP 7: Compute class weights for imbalanced dataset ===
    # With threshold=0.001 and costs=0.0009, HOLD class often dominates
    # Class weights help the model learn from minority classes (LONG/SHORT)
    unique_labels, label_counts = np.unique(train_labels, return_counts=True)
    total_samples = len(train_labels)
    
    # Compute inverse frequency weights (higher weight for rare classes)
    # Formula: weight[i] = total_samples / (num_classes * count[i])
    num_classes = 3  # SHORT, HOLD, LONG
    class_weights_list = []
    for class_idx in range(num_classes):
        if class_idx in unique_labels:
            idx = np.where(unique_labels == class_idx)[0][0]
            weight = total_samples / (num_classes * label_counts[idx])
        else:
            weight = 1.0  # Default weight if class not present
        class_weights_list.append(weight)
    
    class_weights = torch.FloatTensor(class_weights_list)
    logger.info(f"Class distribution: SHORT={label_counts[0] if 0 in unique_labels else 0}, "
                f"HOLD={label_counts[1] if 1 in unique_labels else 0}, "
                f"LONG={label_counts[2] if 2 in unique_labels else 0}")
    logger.info(f"Class weights: {class_weights.numpy()}")
    
    trainer = Trainer(model, train_loader, val_loader, config, device=config.device, 
                      class_weights=class_weights)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    history = trainer.train(epochs=args.epochs)
    
    save_path = config.model_dir / f"{args.model}_trained.pt"
    model.save(str(save_path))
    logger.info(f"Model saved to {save_path}")
    
    engineer.save_scalers(str(config.model_dir / f"{args.model}_scalers.joblib"))
    logger.info("Training complete!")

def train_rl(args):
    """Train reinforcement learning agent."""
    import torch
    from config import config
    from models.rl_agent import PPOAgent, TradingEnvironment
    import numpy as np
    
    logger.info("Training RL Agent with PPO...")
    
    check_gpu()
    
    dummy_data = np.random.randn(10000, 5)
    env = TradingEnvironment(
        data=dummy_data,
        initial_balance=config.rl.initial_capital,
        transaction_cost=config.rl.transaction_cost
    )
    
    agent = PPOAgent(
        state_dim=env._get_state().shape[0],
        action_dim=3,
        hidden_dim=256,
        gamma=config.rl.gamma,
        gae_lambda=config.rl.gae_lambda,
        clip_epsilon=config.rl.clip_epsilon,
        device=config.device
    )
    
    logger.info(f"RL Agent initialized. Training for {args.episodes} episodes...")
    
    for episode in range(args.episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            from models.rl_agent import Experience
            exp = Experience(state, action, reward, next_state, done, log_prob, value)
            agent.store_experience(exp)
            
            state = next_state
            total_reward += reward
            
        if len(agent.buffer) >= 256:
            metrics = agent.update()
            
        if (episode + 1) % 10 == 0:
            logger.info(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Trades = {info['num_trades']}, Sharpe = {info['sharpe']:.2f}")
            
    save_path = config.model_dir / "ppo_agent.pt"
    agent.save(str(save_path))
    logger.info(f"RL Agent saved to {save_path}")

def serve(args):
    """Start the FastAPI prediction server."""
    from api.server import start_server
    
    logger.info(f"Starting prediction server on port {args.port}...")
    check_gpu()
    
    start_server(host="0.0.0.0", port=args.port)

def backtest(args):
    """Run backtest on historical data using walk-forward evaluation.
    
    This implements proper hedge fund-style backtesting:
    - Purged time splits (gap between train/test)
    - Walk-forward: train on window A, test on B, roll forward
    - After-cost PnL with realistic fills
    - Per-regime performance reporting
    """
    import torch
    import numpy as np
    import pandas as pd
    from datetime import datetime as dt
    from config import config
    from data.pipeline import FeatureEngineer, create_labels
    from training.walk_forward import WalkForwardEvaluator, WalkForwardSplitter
    
    logger.info(f"Running walk-forward backtest from {args.start} to {args.end}")
    
    check_gpu()
    
    # Load data
    data_path = config.data_dir / "BTCUSDT_15m.parquet"
    if not data_path.exists():
        logger.error("No data found. Run 'python main.py fetch' first.")
        return
    
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} candles")
    
    # Filter by date range if timestamps are available
    if 'timestamp' in df.columns:
        start_ts = pd.Timestamp(args.start).timestamp() * 1000
        end_ts = pd.Timestamp(args.end).timestamp() * 1000
        df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]
        logger.info(f"Filtered to {len(df)} candles in date range")
    
    if len(df) < 1000:
        logger.error(f"Not enough data for backtest: {len(df)} candles")
        return
    
    # Compute features
    engineer = FeatureEngineer()
    features_df = engineer.compute_technical_features(df)
    features_df = features_df.fillna(0)
    features_np = features_df.values.astype(np.float32)
    
    # Load model
    model_name = args.model or "transformer"
    model_path = config.model_dir / f"{model_name}_trained.pt"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error(f"Run 'python main.py train --model {model_name}' first")
        return
    
    # Load model based on type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = features_np.shape[1]
    
    if model_name == "transformer":
        from models.transformer import TransformerPriceModel
        model = TransformerPriceModel(input_dim=input_dim)
    elif model_name == "tft":
        from models.transformer import TemporalFusionTransformer
        model = TemporalFusionTransformer(input_dim=input_dim)
    elif model_name == "lstm":
        from models.lstm import BidirectionalLSTM
        model = BidirectionalLSTM(input_dim=input_dim)
    elif model_name == "cnn":
        from models.cnn import ResNetPrice
        model = ResNetPrice(input_dim=input_dim)
    else:
        logger.error(f"Unknown model type: {model_name}")
        return
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model: {model_name}")
    
    # Configure walk-forward evaluation
    n_folds = args.folds
    purge_gap = args.purge
    n_samples = len(features_np)
    
    # Convert days to 15m samples: 1 day = 24 hours * 4 samples/hour = 96 samples
    samples_per_day = 96
    train_days = getattr(args, 'train_days', 30)
    test_days = getattr(args, 'test_days', 7)
    
    train_periods = train_days * samples_per_day
    test_periods = test_days * samples_per_day
    embargo_periods = 48  # 12 hours at 15m
    
    logger.info(f"Window config: train={train_days}d ({train_periods} samples), test={test_days}d ({test_periods} samples)")
    
    # Validate we have enough data for at least one fold
    total_fold_size = train_periods + purge_gap + test_periods + embargo_periods
    if n_samples < total_fold_size:
        logger.error(f"Not enough data for walk-forward: need {total_fold_size}, have {n_samples}")
        logger.error(f"  Required: train={train_periods} (~30 days), purge={purge_gap}, test={test_periods} (~7 days), embargo={embargo_periods}")
        logger.error(f"  Try a longer date range or fetch more data first")
        return
    
    # Calculate step size between folds
    step_size = (n_samples - total_fold_size) // max(n_folds - 1, 1)
    if step_size <= 0:
        logger.warning(f"Data only supports 1 fold (step_size={step_size}), reducing n_folds to 1")
        n_folds = 1
    
    splitter = WalkForwardSplitter(
        n_splits=n_folds,
        train_periods=train_periods,
        test_periods=test_periods,
        purge_periods=purge_gap,
        embargo_periods=embargo_periods
    )
    
    evaluator = WalkForwardEvaluator(
        splitter=splitter,
        holding_periods=48  # 12 hours at 15m intervals
    )
    
    logger.info(f"Walk-forward config: {n_folds} folds, train={train_periods} (~30d), test={test_periods} (~7d), purge={purge_gap}")
    
    # Generate and validate ALL splits before running any evaluation
    splits = list(splitter.split(n_samples))
    
    if len(splits) == 0:
        logger.error("No valid walk-forward splits could be generated")
        return
    
    # Validate ALL folds have valid boundaries before execution - FAIL FAST on any invalid fold
    valid_splits = []
    invalid_folds = []
    
    for i, (train_idx, test_idx) in enumerate(splits):
        errors = []
        
        # Check non-empty
        if len(train_idx) == 0 or len(test_idx) == 0:
            errors.append(f"empty indices (train={len(train_idx)}, test={len(test_idx)})")
        
        # Check non-overlapping (with purge gap)
        if len(train_idx) > 0 and len(test_idx) > 0 and test_idx[0] <= train_idx[-1]:
            errors.append(f"overlapping train/test")
        
        # Check purge gap is maintained
        if len(train_idx) > 0 and len(test_idx) > 0:
            actual_gap = test_idx[0] - train_idx[-1] - 1
            if actual_gap < purge_gap:
                errors.append(f"purge gap {actual_gap} < required {purge_gap}")
        
        # Check test end doesn't exceed data
        if len(test_idx) > 0 and test_idx[-1] >= n_samples:
            errors.append(f"test exceeds data bounds")
        
        # Check expected train/test lengths (institutional requirement)
        if len(train_idx) != train_periods:
            errors.append(f"train length {len(train_idx)} != expected {train_periods}")
        if len(test_idx) != test_periods:
            errors.append(f"test length {len(test_idx)} != expected {test_periods}")
        
        if errors:
            invalid_folds.append((i, errors))
        else:
            valid_splits.append((i, train_idx, test_idx))
    
    # FAIL FAST: If any fold is invalid, abort entirely
    if invalid_folds:
        logger.error(f"{len(invalid_folds)}/{len(splits)} folds failed validation:")
        for fold_id, errors in invalid_folds:
            logger.error(f"  Fold {fold_id}: {', '.join(errors)}")
        logger.error("Aborting backtest - reduce --folds or provide more data")
        return
    
    logger.info(f"All {len(valid_splits)} folds validated successfully")
    results = []
    
    for orig_fold_id, train_idx, test_idx in valid_splits:
        logger.info(f"Fold {orig_fold_id + 1}: train[{train_idx[0]}:{train_idx[-1]}] test[{test_idx[0]}:{test_idx[-1]}]")
        
        result = evaluator.evaluate_fold(
            model=model,
            candles=df.reset_index(drop=True),
            features=features_np,
            train_idx=train_idx,
            test_idx=test_idx,
            fold_id=orig_fold_id,
            device=device
        )
        
        results.append(result)
        
        logger.info(f"  Trades: {result.n_trades}, Win Rate: {result.win_rate:.1%}, "
                   f"Sharpe: {result.sharpe_ratio:.2f}, Max DD: {result.max_drawdown:.1%}")
    
    # Aggregate results
    total_trades = sum(r.n_trades for r in results)
    avg_win_rate = np.mean([r.win_rate for r in results if r.n_trades > 0])
    avg_sharpe = np.mean([r.sharpe_ratio for r in results if r.n_trades > 0])
    avg_expectancy = np.mean([r.expectancy for r in results if r.n_trades > 0])
    max_drawdown = max(r.max_drawdown for r in results) if results else 0
    
    logger.info("\n" + "="*60)
    logger.info("WALK-FORWARD BACKTEST RESULTS")
    logger.info("="*60)
    logger.info(f"Total Trades:    {total_trades}")
    logger.info(f"Avg Win Rate:    {avg_win_rate:.1%}")
    logger.info(f"Avg Sharpe:      {avg_sharpe:.2f}")
    logger.info(f"Avg Expectancy:  {avg_expectancy:.4f}")
    logger.info(f"Max Drawdown:    {max_drawdown:.1%}")
    logger.info("="*60)
    
    # Decision: is model worth deploying?
    if avg_sharpe > 0.5 and avg_expectancy > 0:
        logger.info("✓ Model shows positive edge after costs - consider deploying")
    elif avg_sharpe > 0:
        logger.info("⚠ Model shows marginal edge - needs improvement")
    else:
        logger.info("✗ Model does NOT beat costs - do not deploy")

def main():
    parser = argparse.ArgumentParser(
        description="BTC Futures Trading - GPU Neural Network Trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    fetch_parser = subparsers.add_parser("fetch", help="Fetch historical data")
    fetch_parser.add_argument("--candles", type=int, default=175000, help="Number of candles to fetch (default: 5 years of 15m data)")
    fetch_parser.add_argument("--replit-proxy", type=str, dest="replit_proxy",
                              help="Replit proxy URL for Binance data (e.g., https://your-app.replit.app)")
    
    train_parser = subparsers.add_parser("train", help="Train a neural network model")
    train_parser.add_argument("--model", type=str, required=True,
                             choices=["transformer", "tft", "lstm", "cnn", "vae", "gnn"],
                             help="Model type to train")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--horizon", type=int, default=5, help="Label lookahead horizon (candles)")
    train_parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    rl_parser = subparsers.add_parser("train-rl", help="Train reinforcement learning agent")
    rl_parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    
    serve_parser = subparsers.add_parser("serve", help="Start prediction API server")
    serve_parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    backtest_parser = subparsers.add_parser("backtest", help="Run walk-forward backtest")
    backtest_parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--model", type=str, default="transformer", help="Model to use")
    backtest_parser.add_argument("--folds", type=int, default=5, help="Number of walk-forward folds")
    backtest_parser.add_argument("--purge", type=int, default=100, help="Purge gap (samples) between train/test")
    backtest_parser.add_argument("--train-days", type=int, default=30, dest="train_days", help="Training window in days (default: 30)")
    backtest_parser.add_argument("--test-days", type=int, default=7, dest="test_days", help="Test window in days (default: 7)")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
        
    if args.command == "fetch":
        asyncio.run(fetch_data(args))
    elif args.command == "train":
        train(args)
    elif args.command == "train-rl":
        train_rl(args)
    elif args.command == "serve":
        serve(args)
    elif args.command == "backtest":
        backtest(args)

if __name__ == "__main__":
    main()
