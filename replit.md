# BTC Futures Signal Dashboard

## Overview
This project is an institutional-grade, AI-driven dashboard designed to generate sophisticated BTCUSDT futures trading signals. It integrates machine learning, real-time market data, and sentiment analysis to provide AI-powered trade plans. The system features a continuous learning loop with ongoing data refreshing and model retraining, aiming to deliver a robust, selective trading system that leverages advanced AI and comprehensive market insights for high-confidence trading opportunities.

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
- **Navigation**: Tabbed interface including Overview, Signal, Neural Network (displaying quantile-based predictions and predicted price bands), Paper Trading, and various learning/analysis sections.

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
- **MoE Architecture**: Employs a gating network and four specialized expert models (trend, mean-reversion, volatility, chaos) that adapt to the detected market regime.
- **Self-Supervised Pretraining**: Utilizes techniques like Masked Time-Series, Next-Step Distribution prediction, Contrastive Learning, and Deep clustering for regime discovery, pre-trained on extensive datasets.

#### GPU Neural Network Training
- **Architectures**: Supports 12+ deep learning architectures (Transformers, LSTMs, CNNs, VAEs, GNNs, Ensembles) with local GPU training.
- **GPU Trainer API**: FastAPI server for predictions (including regression and multihead) and model management.
- **Advanced Exit Logic**: Implements dynamic take-profit targets, MFE tracking, and failure stop detection.
- **Unified Learning Controller**: Synchronizes Strategy Learner, Pattern Memory, and GPU Trainer for consistent data processing.
- **Regime-Balanced Training**: Utilizes a 4-regime classification (BULL, BEAR, HIGH_VOL, LOW_VOL_CHOP) with balanced sampling and per-regime validation.

#### Multi-Head Model Architecture
- **Output Heads**: Three distinct heads for Classification (direction probabilities), Regression (expected return μ and uncertainty σ), and Quantile (q10, q25, q50, q75, q90).
- **Loss Function**: Combines CrossEntropyLoss, HuberLoss, GaussianNLLLoss, and Pinball loss.

#### Advanced Labeling and Prediction
- **Cost-Aware Labeling**: Signals generated only when net edge (accounting for trading costs) exceeds a minimum threshold and confidence is sufficient.
- **Gaussian NLL with Log-Sigma**: Predicts `log_sigma` to ensure properly calibrated uncertainty estimates.
- **Constrained Candle Parameterization**: Predicts `delta_close`, `log_range`, and `skew` to guarantee valid candle predictions where high >= low.
- **Quantile-Based SL/TP Derivation**: Stop-loss and take-profit levels are derived directly from predicted quantiles for consistency.

#### Model Management and Monitoring
- **Walk-Forward Weight Saving**: Saves real trading metrics for ensemble model weighting.
- **Feature Version Locking**: Ensures feature consistency between training and inference, returning HOLD on mismatch.
- **Prediction Drift Monitoring**: Utilizes PSI, KL Divergence, and ECE to detect and report feature distribution shifts and prediction calibration changes.

#### Professional Ensemble Predictor
- **Ensemble Voting**: Combines predictions from Transformer, TFT, LSTM, and CNN models with confidence-based voting.
- **Regime & Risk Gating**: Incorporates VAE for market regime detection and GNN for risk regime detection to adjust thresholds and position sizing.
- **Metric Weighting**: Models are weighted based on actual trading metrics (expectancy, precision, profit factor, F1, Sharpe).

### Build System
- **Client Build**: Vite bundles React app to `dist/public`.
- **Server Build**: esbuild bundles server to `dist/index.cjs`.

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

## Enabled Institutional Features (Phase Summary)

The following institutional-grade features are **enabled by default**:

### Phase 1a: Cost-Aware Labeling
- Trading costs computed from volatility + 4h hold time (fees, slippage, funding)
- `net_edge = |μ| - cost` for proper edge calculation
- Signals only generated when net edge exceeds minimum threshold

### Phase 1b: Log-Sigma NLL
- `RegressionHead` defaults to `use_log_sigma=True`
- All multi-head models (Transformer, LSTM, CNN, GNN, VAE, TFT) use log_sigma
- `MultiHeadLossConfig.use_log_sigma=True` by default
- Prevents σ from being "gamed" and couples uncertainty to prediction error

### Phase 1c: Quantile-Derived SL/TP
- API derives SL/TP from quantiles at inference (not separate learned heads)
- LONG: SL from q10/q25 (downside), TP from q75/q90 (upside)
- SHORT: SL from q75/q90 (upside risk), TP from q10/q25 (downside)
- Ensures internal consistency with quantile distribution

### Phase 2: Constrained Candle Parameterization
- `ConstrainedCandleHead` class available (experimental, not wired into models by default)
- Predicts (Δclose, log_range, skew) and reconstructs valid candles
- **Guarantees** high >= low for all predictions

### Phase 3: Mandatory Ensemble Weights
- `model_weights.json` required for production ensemble predictions
- **LOUD WARNINGS** logged if missing (uses placeholder defaults)
- `using_default_weights` property tracks if real weights are loaded

### Phase 4b: Drift Monitoring
- PSI, KL Divergence, ECE tracking available via `DriftMonitor`
- History saved to `checkpoints/drift_history/drift_history.json`

### Phase 5: Configurable Label Generation (NEW)
- **Removed hardcoded thresholds** that caused 99% HOLD signals
- `generate_multihead_targets()` now accepts:
  - `min_net_edge`: Minimum net edge after costs (default 0.0 for debugging)
  - `min_confidence`: Minimum mu/sigma ratio (default 0.3 for debugging)
  - `use_volatility_cost`: Use volatility-based vs fixed cost
  - `fixed_cost`: Fixed round-trip cost (default 0.09% = 0.0009)
- **Label Density Debug Report**: Logs BEFORE training:
  - Total samples, mean |mu|, mean sigma, mean cost
  - Gate pass rates (edge gate %, confidence gate %, BOTH gates %)
  - Class distribution (SHORT %, HOLD %, LONG %)
- **CLI arguments**: `--cost`, `--min-net-edge`, `--min-confidence`, `--volatility-cost`
- Target class distribution: SHORT 7-15%, HOLD 70-85%, LONG 7-15%

### Data Architecture
- **Dashboard**: Uses 35K+ database candles + 1 live Binance candle (display only)
- **GPU Trainer**: Uses 140K+ parquet candles for training (separate data flow)
- **Training is NOT affected by live candle fetch** - completely separate data paths

### Feature Pipeline Alignment (Critical Fix - Jan 2026)
- **Root Cause Fixed**: Model trained on `compute_technical_features` (41 features) but inference used MTF fusion (66 features with different names)
- **Training Feature Names**: `return_50`, `bb_upper`, `ema_5`, `rsi_14`, `atr_14`, etc.
- **MTF Fusion Names (incompatible)**: `ret_1_15m`, `rolling_vol_20_4h`, `ema_slope_26_1h`, etc.
- **Solution**: `/predict/ensemble/candles` endpoint now ALWAYS uses `compute_technical_features` from `data/pipeline.py`
- **Impact**: Resolves 100% missing features issue and silent HOLD fallback during predictions

### GUI Configuration
- Fixed to BTCUSDT 15m timeframe only (no multi-timeframe/multi-asset options)