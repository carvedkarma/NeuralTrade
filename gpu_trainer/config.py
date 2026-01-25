import os
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

@dataclass
class DataConfig:
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "4h"])
    lookback_candles: int = 50000
    sequence_length: int = 100
    prediction_horizon: int = 5
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

@dataclass
class ModelConfig:
    transformer_dim: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 6
    transformer_dropout: float = 0.1
    
    lstm_hidden: int = 256
    lstm_layers: int = 3
    lstm_dropout: float = 0.2
    
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    
    vae_latent_dim: int = 64
    
    gnn_hidden: int = 128
    gnn_layers: int = 3
    
    sentiment_model: str = "distilbert-base-uncased"
    
@dataclass 
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 10
    gradient_clip: float = 1.0
    warmup_steps: int = 1000
    
    use_curriculum: bool = True
    use_contrastive: bool = True
    use_online_learning: bool = True
    
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
@dataclass
class RLConfig:
    algorithm: str = "PPO"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_steps: int = 2048
    n_epochs: int = 10
    
    initial_capital: float = 10000.0
    max_position_size: float = 1.0
    transaction_cost: float = 0.001
    
@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    num_workers: int = 4
    
    base_dir: Path = Path(__file__).parent
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data_cache")
    model_dir: Path = field(default_factory=lambda: Path(__file__).parent / "saved_models")
    
    db_url: str = os.getenv("DATABASE_URL", "postgresql://localhost:5432/btc_signals")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    api_port: int = 8000
    
    # Replit proxy URL for fetching Binance data (bypasses Australian geoblocking)
    # Set this to your Replit app URL, e.g., "https://your-app.replit.app"
    replit_proxy_url: str = os.getenv("REPLIT_PROXY_URL", "")
    
    def __post_init__(self):
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        Path(self.training.checkpoint_dir).mkdir(exist_ok=True)
        Path(self.training.log_dir).mkdir(exist_ok=True)
        
        torch.manual_seed(self.seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.seed)

config = Config()
