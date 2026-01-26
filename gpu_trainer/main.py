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
    """Train a neural network model."""
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
        logger.warning("No cached data found. Generating synthetic data for testing...")
        n_samples = 10000
        df = pd.DataFrame({
            'open': np.cumsum(np.random.randn(n_samples) * 0.001) + 50000,
            'high': np.cumsum(np.random.randn(n_samples) * 0.001) + 50100,
            'low': np.cumsum(np.random.randn(n_samples) * 0.001) + 49900,
            'close': np.cumsum(np.random.randn(n_samples) * 0.001) + 50000,
            'volume': np.abs(np.random.randn(n_samples) * 1000000) + 500000,
        })
    
    engineer = FeatureEngineer()
    features_df = engineer.compute_technical_features(df)
    features_df = features_df.fillna(0)
    
    engineer.fit_scalers(features_df)
    scaled_features = engineer.transform(features_df)
    
    labels = create_labels(df, horizon=5, threshold=0.001)
    labels = (labels + 1).astype(int)
    
    features_np = scaled_features.values.astype(np.float32)
    labels_np = labels.astype(np.int64)
    
    valid_start = config.data.sequence_length
    features_np = features_np[valid_start:]
    labels_np = labels_np[valid_start:]
    
    n_train = int(len(features_np) * 0.8)
    n_val = int(len(features_np) * 0.1)
    
    train_dataset = TradingDataset(features_np[:n_train], labels_np[:n_train], config.data.sequence_length)
    val_dataset = TradingDataset(features_np[n_train:n_train+n_val], labels_np[n_train:n_train+n_val], config.data.sequence_length)
    
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
    
    trainer = Trainer(model, train_loader, val_loader, config, device=config.device)
    
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
    """Run backtest on historical data."""
    logger.info(f"Running backtest from {args.start} to {args.end}")
    
    logger.info("Backtest functionality would run here with loaded models...")

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
    train_parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    rl_parser = subparsers.add_parser("train-rl", help="Train reinforcement learning agent")
    rl_parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    
    serve_parser = subparsers.add_parser("serve", help="Start prediction API server")
    serve_parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--model", type=str, help="Model to use")
    
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
