# BTC Futures Trading - GPU Neural Network Trainer

A comprehensive deep learning system for cryptocurrency trading signals, designed to run on your local GPU.

## Important reality check

No trading model can honestly be guaranteed to be "better than any available platform".
Markets are non-stationary, execution quality matters, and most apparent edge disappears
under realistic costs, slippage, and out-of-sample testing. This repository now includes
a local-first research CLI aimed at making the process more robust:

- fixed 10-symbol liquid crypto universe by default,
- GPU-oriented multi-symbol V5 training,
- walk-forward validation before promotion,
- local self-training loop with guarded promotion,
- CLI workflows that do not require a hosted dashboard.

Use it as a research and experimentation framework, not as a promise of profit.

## Local GPU research CLI

The recommended path for local multi-symbol crypto research is the new `crypto-research`
CLI (or `python -m gpu_trainer.research_cli`).

### Default 10-symbol universe

```text
BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT,
ADAUSDT, DOGEUSDT, LINKUSDT, AVAXUSDT, LTCUSDT
```

### Install

From the repo root:

```bash
pip install -e .
pip install -r gpu_trainer/requirements.txt
```

If you prefer not to install the package, you can still run:

```bash
python -m gpu_trainer.research_cli --help
```

### Prepare local data

Binance-only:

```bash
crypto-research --dashboard-url "" prepare-data
```

Custom symbols:

```bash
crypto-research --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,LINKUSDT,AVAXUSDT,LTCUSDT prepare-data
```

### Train on your GPU

```bash
crypto-research train
```

### Run walk-forward evaluation and export a best policy

```bash
crypto-research evaluate
```

There is also an explicit alias if you want the command name to match the task:

```bash
crypto-research walk-forward
```

This writes:

- `checkpoints/v5_walkforward_report.json`
- `checkpoints/v5_run_metrics.json`
- `checkpoints/best_policy.json`

### Recommended walk-forward workflow for existing local data

If you already have 5 years of parquet data across your symbols, use this order:

1. Validate local files and environment:

```bash
crypto-research doctor
```

2. Run walk-forward only:

```bash
crypto-research --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,LINKUSDT,AVAXUSDT,LTCUSDT walk-forward
```

For a first baseline-seeding run, prefer:

```bash
crypto-research --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,LINKUSDT,AVAXUSDT,LTCUSDT run --skip-data --set-baseline-on-first-success
```

This path now skips the standalone pre-train stage and performs walk-forward first,
which avoids noisy single-train sweep logs when your main goal is walk-forward validation.

3. Print the saved fold-by-fold results:

```bash
crypto-research inspect-results
```

The `doctor` command checks:

- whether PyTorch is installed,
- whether CUDA is visible,
- whether each parquet file exists,
- whether required columns are present,
- whether timestamps are sorted and unique,
- whether row counts meet the minimum threshold,
- whether the date range is long enough for walk-forward training.

### Full research pipeline

```bash
crypto-research run --dashboard-url "" --refresh --set-baseline-on-first-success
```

This will:

1. refresh/download local data,
2. train the multi-symbol model,
3. run walk-forward evaluation,
4. export the best policy,
5. promote checkpoints only if configured gates pass.

### Self-training loop

```bash
crypto-research self-train --cycles 3 --sleep-seconds 600
```

This loop refreshes cached parquet data and fine-tunes symbol models locally using the
existing V5 learning stack. Promotion remains gated.

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
