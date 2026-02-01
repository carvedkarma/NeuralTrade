# BTC Futures Signal Dashboard

## Overview
This project is an institutional-grade, AI-driven dashboard designed to generate sophisticated BTCUSDT futures trading signals. It integrates machine learning, real-time market data, and sentiment analysis to provide AI-powered trade plans. The system features a continuous learning loop with ongoing data refreshing and model retraining, aiming to deliver a robust, selective trading system that leverages advanced AI and comprehensive market insights for high-confidence trading opportunities. The project's vision is to deliver a cutting-edge platform for futures trading, capitalizing on market potential through AI-driven precision and continuous adaptation.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend
- **Framework**: React with TypeScript (Vite).
- **Routing**: Wouter.
- **State Management**: TanStack React Query for real-time data.
- **UI Components**: shadcn/ui (Radix UI, Tailwind CSS).
- **Charts**: Recharts.
- **Animations**: Framer Motion.
- **Navigation**: Tabbed interface including Overview, Signal, Neural Network (displaying quantile-based predictions and predicted price bands), and Paper Trading.

### Backend
- **Runtime**: Node.js with Express.js (TypeScript, ESM).
- **API Pattern**: RESTful.
- **AI Integration**: OpenAI via Replit AI Integrations.
- **Market Data**: Binance Vision API, with CoinGecko and CryptoCompare fallbacks, augmented by a Replit-hosted data proxy.
- **GPU Trainer Communication**: Bi-directional communication with a local GPU trainer via dedicated API endpoints.

### Data Layer
- **ORM**: Drizzle ORM for PostgreSQL.
- **Schema**: Zod for type-safe validation.
- **Data Separation**: Live sentiment data is separated from historical price/volume data.
- **Key Data Models**: Covers Candles, Signals, Futures Data, Technical Indicators, AI Analysis, Trading, and Paper Trading entities.
- **Persistence**: All learning states, including pattern clusters and social media statistics, are persisted in a database.

### Machine Learning and Signal Generation

#### Regression-Based Signal System
- **Output**: Comprehensive signals including action, confidence, expected_move, uncertainty, edge, cost_estimate, suggested_order_type, urgency, position_size_pct, stop_loss_pct, take_profit_pct, regime, and expert_weights.
- **Regression Targets**: Focuses on μ (4h forward return), σ (uncertainty), P(move>cost), quantiles (p10/p50/p90), MFE/MAE.
- **Market Microstructure**: Incorporates funding rates, open interest, liquidations, order book depth, and taker buy/sell volume.
- **Evaluation**: Uses purged time-series splits with walk-forward evaluation, after-cost PnL, Sharpe ratio, and maximum drawdown.

#### Regime Detection & Mixture-of-Experts (MoE)
- **Regime Types**: Identifies TRENDING, MEAN_REVERTING, HIGH_VOLATILITY, LOW_VOLATILITY, TRANSITION, UNKNOWN.
- **MoE Architecture**: Employs a gating network and four specialized expert models that adapt to the detected market regime.
- **Self-Supervised Pretraining**: Utilizes techniques like Masked Time-Series, Next-Step Distribution prediction, Contrastive Learning, and Deep clustering for regime discovery.

#### GPU Neural Network Training
- **Architectures**: Supports 12+ deep learning architectures (Transformers, LSTMs, CNNs, VAEs, GNNs, Ensembles) with local GPU training.
- **GPU Trainer API**: FastAPI server for predictions (including regression and multihead) and model management.
- **Advanced Exit Logic**: Implements dynamic take-profit targets, MFE tracking, and failure stop detection.
- **Unified Learning Controller**: Synchronizes Strategy Learner, Pattern Memory, and GPU Trainer for consistent data processing.
- **Regime-Balanced Training**: Utilizes a 4-regime classification (BULL, BEAR, HIGH_VOL, LOW_VOL_CHOP) with balanced sampling and per-regime validation.

#### Multi-Head Model Architecture
- **Output Heads**: Five distinct head types: Classification (Direction probabilities), Regression (Expected return μ and uncertainty σ), Quantile (price projections), VolState (3-class volatility state), Acceleration (Scalar momentum change prediction).
- **Loss Function**: Combines CrossEntropyLoss, HuberLoss, GaussianNLLLoss, and Pinball loss.

#### Flow Forecast System
- **Purpose**: Replaces triangle probability cones with regime-conditioned quantile path projections.
- **Vol_State Classification**: Predicts forward volatility regime based on forward_vol/current_vol ratio.
- **Acceleration Prediction**: Predicts momentum change.
- **Quantile Path Generation**: Generates three paths (q10, q50, q90) using alpha shaping based on volatility state for a 16-bar horizon.
- **Volatility Gate**: Triggers "NO_FORECAST" mode when volatility is contracting or expected spread is insufficient.

#### Advanced Labeling and Prediction
- **Cost-Aware Labeling**: Signals generated only when net edge (accounting for trading costs) exceeds a minimum threshold and confidence is sufficient.
- **Gaussian NLL with Log-Sigma**: Predicts `log_sigma` for calibrated uncertainty estimates.
- **Constrained Candle Parameterization**: Predicts `delta_close`, `log_range`, and `skew` to guarantee valid candle predictions.
- **Quantile-Based SL/TP Derivation**: Stop-loss and take-profit levels are derived directly from predicted quantiles.

#### Model Management and Monitoring
- **Walk-Forward Weight Saving**: Saves real trading metrics for ensemble model weighting.
- **Feature Version Locking**: Ensures feature consistency between training and inference.
- **Prediction Drift Monitoring**: Utilizes PSI, KL Divergence, and ECE to detect and report feature distribution shifts and prediction calibration changes.

#### Professional Ensemble Predictor
- **Ensemble Voting**: Combines predictions from Transformer, TFT, LSTM, and CNN models with confidence-based voting.
- **Regime & Risk Gating**: Incorporates VAE for market regime detection and GNN for risk regime detection to adjust thresholds and position sizing.
- **Metric Weighting**: Models are weighted based on actual trading metrics.

### Build System
- **Client Build**: Vite bundles React app to `dist/public`.
- **Server Build**: esbuild bundles server to `dist/index.cjs`.

### Data Architecture
- **Dashboard**: Uses 35K+ database candles + 1 live Binance candle.
- **GPU Trainer**: Uses 140K+ parquet candles for training (separate data flow).
- **Feature Pipeline Alignment**: Explicit STF/MTF mode routing with mode auto-detection from saved feature config.
- **FeatureEngineer Version Tracking**: Prevents silent signal degradation from train/inference feature computation mismatch.
- **Centralized Timeframe Configuration**: Uses `gpu_trainer/config/timeframe_config.py` as a single source of truth for timeframes and horizons, defaulting to BTCUSDT 15m.

### Runtime Diagnostic System
- **GPU Trainer /health Endpoint**: Provides comprehensive capability information, including available features and disconnect reasons.
- **Server Health Polling**: Logs all health fields with `[GPU HEALTH]` prefix and explicitly logs disconnect reasons.
- **UI Console Logging**: Logs `[FLOW FORECAST UI]` with forecastMode, volState, acceleration, path lengths, and whether NO_FORECAST or QUANTILE_PATHS rendering occurs.

### Training, Monitoring, and Policy Architecture

The system separates model training from policy selection to ensure consistent live trading behavior.

#### 1. Neural Network Training (MultiheadTrainer.train)
- **Purpose**: Learns model weights via gradient descent.
- **No policy logic**: Training does NOT save or freeze any execution policy.
- **Outputs**: Model checkpoints saved to disk.

#### 2. Monitoring Sweep (during training)
- **Purpose**: Informational only - tracks training quality.
- **Location**: `_compute_trading_metrics()` in multihead_trainer.py.
- **Runs**: Every 5 epochs (configurable via MONITORING_EPOCH_INTERVAL).
- **Logs**: Prefixed with "MONITORING SWEEP" - clearly marked as NOT for live trading.
- **Does NOT**: Save policies, alter training, or affect live trading.

#### 3. Post-Training Policy Selection (PolicySelector)
- **Purpose**: The ONLY source of truth for live execution policy.
- **Location**: `gpu_trainer/training/policy_selector.py`.
- **Flow**:
  1. Load best model checkpoint.
  2. Run sequential out-of-sample (OOS) evaluation (>=5 time folds).
     - NOTE: Model is NOT retrained per fold - tests fixed model across time periods.
  3. Sweep confidence thresholds with Pareto selection.
  4. Filter by MIN_TRADES=30 eligibility.
  5. Select best by risk-adjusted score (expectancy - 0.5*max_drawdown).
  6. Save frozen policy to `execution_policy.json`.
  7. Print "FROZEN POLICY" summary.

#### 4. Live Trading
- **Uses**: Frozen policy from `execution_policy.json`.
- **Gate order**: spread → confidence → direction → cooldown → trade.
- **No adaptation**: Policy is static until next retrain cycle.
- **Retraining trigger**: Walk-forward instability detection.

#### Execution Policy Fields (execution_policy.json)
```json
{
  "min_confidence": 0.15,
  "spread_multiplier": 3.0,
  "cooldown": 8,
  "fixed_cost": 0.0009,
  "tp_quantile": "q75",
  "sl_quantile": "q10",
  "min_trades": 30,
  "expectancy": 0.0023,
  "risk_adjusted_score": 0.0015,
  "hit_rate": 0.542,
  "max_drawdown": 0.0016,
  "sharpe": 1.23,
  "num_trades": 142,
  "created_at": "2026-02-01T12:00:00",
  "checkpoint_path": "models/best_multihead.pt",
  "walk_forward_folds": 5
}
```

## External Dependencies

### Database
- PostgreSQL (via `DATABASE_URL`).
- Drizzle Kit for schema migrations.
- connect-pg-simple for session storage.

### UI Framework
- Radix UI primitives.
- Lucide React for icons.
- class-variance-authority for component variants.

### Data & Validation
- Zod for runtime schema validation.
- drizzle-zod for database schema to Zod type generation.
- date-fns for date formatting.

### Development Tools
- Replit-specific plugins for dev banner and error overlay.
- TypeScript.