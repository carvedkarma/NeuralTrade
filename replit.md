# BTC Futures Signal Dashboard

## Overview
This project is an institutional-grade, AI-driven dashboard designed to generate sophisticated BTCUSDT futures trading signals. It integrates machine learning, real-time market data, and sentiment analysis to provide AI-powered trade plans. The system features a continuous learning loop with ongoing data refreshing and model retraining, aiming to deliver a robust, selective trading system that leverages advanced AI and comprehensive market insights for high-confidence trading opportunities. The project's vision is to deliver a cutting-edge platform for futures trading, capitalizing on market potential through AI-driven precision and continuous adaptation.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend
- **Framework**: React with TypeScript (Vite).
- **UI Components**: shadcn/ui (Radix UI, Tailwind CSS).
- **Charts**: Recharts.
- **Navigation**: Tabbed interface including Overview, Signal, Neural Network, and Paper Trading.

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
- **Persistence**: All learning states, including pattern clusters and social media statistics, are persisted in a database.

### Machine Learning and Signal Generation
- **Regression-Based Signal System**: Generates comprehensive signals including action, confidence, expected_move, uncertainty, and position sizing. Targets μ (4h forward return), σ (uncertainty), P(move>cost), quantiles, MFE/MAE.
- **Regime Detection & Mixture-of-Experts (MoE)**: Identifies market regimes (TRENDING, MEAN_REVERTING, HIGH_VOLATILITY, LOW_VOLATILITY, TRANSITION, UNKNOWN) and employs a gating network with specialized expert models.
- **GPU Neural Network Training**: Supports 12+ deep learning architectures (Transformers, LSTMs, CNNs, VAEs, GNNs, Ensembles) with local GPU training. Implements dynamic take-profit targets and failure stop detection.
- **Multi-Head Model Architecture**: Uses five distinct output heads: Classification, Regression, Quantile, VolState, and Acceleration, with combined loss functions.
- **Flow Forecast System**: Replaces triangle probability cones with regime-conditioned quantile path projections (q10, q50, q90) for a 16-bar horizon.
- **Advanced Labeling and Prediction**: Addresses HOLD-heavy label distribution with a 3-stage HOLD fix, cost-aware labeling, Gaussian NLL with Log-Sigma for uncertainty, and constrained candle parameterization.
- **Model Management and Monitoring**: Includes walk-forward weight saving, feature version locking, prediction drift monitoring (PSI, KL Divergence, ECE), and label metadata tracking.
- **Feature Schema Enforcement**: Ensures strict consistency of features between training and inference using `.features.json` files and `FeatureValidator.enforce_schema`.
- **Professional Ensemble Predictor**: Combines predictions from multiple models (Transformer, TFT, LSTM, CNN) with confidence-based voting, regime, and risk gating.
- **Training, Monitoring, and Policy Architecture**: Separates model training (weights learning) from policy selection (live execution rules). Post-training, a `PolicySelector` performs sequential out-of-sample evaluation to select and save a frozen `execution_policy.json` for live trading.

### Build System
- **Client Build**: Vite bundles React app to `dist/public`.
- **Server Build**: esbuild bundles server to `dist/index.cjs`.

### Data Architecture
- **Dashboard**: Uses 35K+ database candles + 1 live Binance candle.
- **GPU Trainer**: Uses 140K+ parquet candles for training (separate data flow).
- **Feature Pipeline Alignment**: Explicit STF/MTF mode routing with auto-detection.
- **Centralized Timeframe Configuration**: Uses `gpu_trainer/config/timeframe_config.py` as a single source of truth.

### Runtime Diagnostic System
- **GPU Trainer /health Endpoint**: Provides comprehensive capability information and disconnect reasons.
- **UI Console Logging**: Logs `[FLOW FORECAST UI]` with forecastMode, volState, acceleration, and path lengths.

### GPU Training API
- **`/training/start`**: Triggers MultiHeadTrainer training.
- **`/training/status`**: Provides real-time training progress, epoch_history, ETA, per-head losses, and health warnings.
- **`/api/retrain/daily`**: Daily retraining endpoint.
- **Training Configuration**: Uses inverse frequency class weights, 80/20 train/val split, 100-step sequences.

## External Dependencies

### Database
- PostgreSQL (via `DATABASE_URL`).
- Drizzle Kit for schema migrations.

### UI Framework
- Radix UI primitives.
- Lucide React for icons.
- class-variance-authority for component variants.

### Data & Validation
- Zod for runtime schema validation.
- drizzle-zod for database schema to Zod type generation.
- date-fns for date formatting.