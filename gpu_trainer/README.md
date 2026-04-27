# BTC Futures Trading - GPU Neural Network Trainer

A comprehensive deep learning system for cryptocurrency trading signals, designed to run on your local GPU.

> Important: this repository can train research models and generate signals, but no model can guarantee profit or be "better than every platform" in live markets. Use paper trading, exchange-side stops, small sizing, and independent monitoring before risking capital.

## New: 10-symbol self-training CLI

`crypto_ai_system.py` is a self-contained Python CLI for training a GPU-capable multi-symbol crypto model across 10 default symbols:

`BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, LINKUSDT, LTCUSDT, AVAXUSDT`

It includes the practical controls that matter for financial ML:

- Binance spot/futures candle download with local parquet cache
- rolling technical features only from current/past candles
- future-return labels that include configurable fees and slippage
- chronological train/validation split with a purge gap to reduce lookahead leakage
- robust scaler fitted on training rows only
- Transformer encoder with symbol embeddings and GPU mixed precision
- classification plus return-distribution heads
- validation metrics with costs, max drawdown, trade rate, profit factor, and Sharpe
- signal JSON export for all configured symbols
- `auto` mode that downloads fresh data, retrains, exports signals, sleeps, and repeats

### Install on your GPU machine

```bash
cd gpu_trainer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the CUDA build of PyTorch that matches your driver if the default wheel is CPU-only. Verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Download data

```bash
python crypto_ai_system.py download --market futures --bars 80000
```

### Train

```bash
python crypto_ai_system.py train \
  --device cuda \
  --epochs 40 \
  --batch-size 256 \
  --sequence-length 96 \
  --horizon 16 \
  --fee-bps 6 \
  --slippage-bps 4 \
  --purge-bars 160
```

Outputs:

- `gpu_trainer/checkpoints/crypto_ai_best.pt`
- `gpu_trainer/checkpoints/crypto_ai_report.json`

### Generate signals

```bash
python crypto_ai_system.py signal \
  --device cuda \
  --checkpoint gpu_trainer/checkpoints/crypto_ai_best.pt \
  --min-confidence 0.50 \
  --output gpu_trainer/signals/crypto_ai_signals.json
```

### Self-training loop

```bash
python crypto_ai_system.py auto \
  --device cuda \
  --market futures \
  --bars 80000 \
  --epochs 30 \
  --retrain-hours 12
```

This is intended for research/paper trading. For live execution, keep a separate risk engine with position limits, kill switches, max daily drawdown, exchange-side stops, and alerting.

## Features

### Neural Network Models
- **Transformer** - Attention-based sequence learning (100M+ parameters)
- **Temporal Fusion Transformer** - State-of-the-art time series model
- **Bidirectional LSTM** - Recurrent network for sequential patterns
- **ResNet CNN** - Convolutional network for candlestick patterns
- **WaveNet** - Dilated convolutions for long-range dependencies
- **VAE** - Variational autoencoder for market state compression
- **Graph Neural Network** - Cross-asset relationship modeling

### Advanced Training
- **Contrastive Learning** - Distinguish winning vs losing setups
- **Curriculum Learning** - Easy patterns first, hard patterns later
- **Online Learning** - Update weights with each new candle
- **Mixed Precision Training** - 2x faster training with FP16

### Reinforcement Learning
- **PPO Agent** - Learns optimal entry/exit timing
- **Sharpe-based Rewards** - Optimizes risk-adjusted returns
- **Position Sizing** - Learns dynamic position management

### Sentiment Analysis
- **DistilBERT Fine-tuning** - Crypto news understanding
- **Multi-modal Fusion** - Combines price + sentiment signals
- **News Aggregation** - Attention over multiple headlines

## Requirements

### Hardware
- NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better recommended)
- 16GB+ RAM
- 50GB+ disk space for data and models

### Software
- Python 3.10+
- CUDA 11.8+ and cuDNN 8+
- NVIDIA Driver 525+

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo>
cd gpu_trainer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify GPU

```bash
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 3. Fetch Historical Data

```bash
python main.py fetch --candles 50000
```

### 4. Train Models

```bash
# Train Transformer model
python main.py train --model transformer --epochs 100

# Train LSTM model
python main.py train --model lstm --epochs 100

# Train all models
for model in transformer tft lstm cnn vae gnn; do
    python main.py train --model $model --epochs 50
done
```

### 5. Train RL Agent

```bash
python main.py train-rl --episodes 1000
```

### 6. Start Prediction Server

```bash
python main.py serve --port 8000
```

## Docker Setup (Recommended)

### Start All Services

```bash
# Start PostgreSQL, Redis, Trainer, and TensorBoard
docker-compose up -d

# View logs
docker-compose logs -f trainer
```

### Access Services
- **Prediction API**: http://localhost:8000
- **TensorBoard**: http://localhost:6006
- **API Docs**: http://localhost:8000/docs

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Make Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[...]], "sequence_length": 100}'
```

### Training Status
```bash
curl http://localhost:8000/training/status
```

### Start Training
```bash
curl -X POST http://localhost:8000/training/start \
  -H "Content-Type: application/json" \
  -d '{"model_type": "transformer", "epochs": 100}'
```

## Model Architecture

### Transformer Price Model
```
Input (100 candles × 64 features)
    ↓
Linear Projection (64 → 256)
    ↓
Positional Encoding
    ↓
6× Attention Blocks (8 heads)
    ↓
Global Average Pooling
    ↓
MLP Classifier
    ↓
Output (3 classes: LONG, SHORT, HOLD)
```

### Ensemble System
```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Transformer │  │    LSTM     │  │     CNN     │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                ┌───────▼───────┐
                │  Meta-Learner │
                │  (Learns      │
                │   weights)    │
                └───────┬───────┘
                        │
                ┌───────▼───────┐
                │   Ensemble    │
                │   Output      │
                └───────────────┘
```

## Configuration

Edit `config.py` to customize:

```python
@dataclass
class ModelConfig:
    transformer_dim: int = 256      # Increase for more capacity
    transformer_heads: int = 8      # Number of attention heads
    transformer_layers: int = 6     # Depth of transformer
    lstm_hidden: int = 256          # LSTM hidden size
    lstm_layers: int = 3            # LSTM depth
```

## Performance Tuning

### For RTX 4070 (12GB VRAM)
- Batch size: 64-128
- Sequence length: 100-200
- Transformer layers: 6-8
- Hidden dimension: 256-512

### For RTX 4090 (24GB VRAM)
- Batch size: 128-256
- Sequence length: 200-500
- Transformer layers: 12+
- Hidden dimension: 512-1024

## Monitoring

### TensorBoard
```bash
tensorboard --logdir logs
```

### GPU Usage
```bash
watch -n 1 nvidia-smi
```

## Integration with Replit Dashboard

The prediction server can be connected to your Replit dashboard:

1. Start the local server: `python main.py serve --port 8000`
2. Use ngrok or similar to expose: `ngrok http 8000`
3. Update Replit app to call your ngrok URL for predictions

## Directory Structure

```
gpu_trainer/
├── main.py              # CLI entry point
├── config.py            # Configuration
├── requirements.txt     # Dependencies
├── docker-compose.yml   # Docker setup
├── Dockerfile           # Container build
├── models/              # Neural network architectures
│   ├── base.py          # Base model class
│   ├── transformer.py   # Transformer models
│   ├── lstm.py          # LSTM models
│   ├── cnn.py           # CNN models
│   ├── vae.py           # Variational autoencoders
│   ├── gnn.py           # Graph neural networks
│   ├── rl_agent.py      # Reinforcement learning
│   ├── sentiment.py     # Sentiment models
│   └── ensemble.py      # Ensemble systems
├── data/                # Data pipeline
│   └── pipeline.py      # Data fetching & processing
├── training/            # Training logic
│   └── trainer.py       # Training loops
├── api/                 # FastAPI server
│   └── server.py        # Prediction API
├── saved_models/        # Trained models
├── checkpoints/         # Training checkpoints
├── logs/                # TensorBoard logs
└── data_cache/          # Cached historical data
```

## License

MIT License - Use at your own risk for trading.
