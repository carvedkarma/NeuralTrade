# BTC Futures Signal Dashboard

## Overview

This project is an institutional-grade, AI-driven dashboard for generating BTCUSDT futures trading signals. Its primary goal is to deliver sophisticated, AI-powered trade plans by integrating machine learning, real-time market data, and sentiment analysis. The system features a continuous learning loop, adapting to market changes through ongoing data refreshing and model retraining. Key ambitions include providing a robust, selective trading system that leverages advanced AI techniques and comprehensive market insights to generate high-confidence trading opportunities.

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
- **Navigation**: Tabbed interface covering Overview, Signal, Neural Network, Paper Trading, GPU Training, Strategy Learner, Learning, AI Analysis, Indicators, and Performance. The Neural Network tab displays quantile-based predictions (Entry/SL/TP) and predicted price bands.

### Backend
- **Runtime**: Node.js with Express.js.
- **Language**: TypeScript with ESM.
- **API Pattern**: RESTful.
- **AI Integration**: OpenAI via Replit AI Integrations.
- **Market Data**: Binance Vision API, with CoinGecko and CryptoCompare fallbacks, augmented by a Replit-hosted data proxy.
- **GPU Trainer Communication**: Bi-directional communication with a local GPU trainer via dedicated API endpoints.

### Data Layer
- **ORM**: Drizzle ORM for PostgreSQL.
- **Schema**: Zod for type-safe validation.
- **Data Separation**: Live sentiment data is separated from historical price/volume data to prevent leakage.
- **Key Data Models**: Candle, Signal, FuturesData, TechnicalIndicator, MultiTimeframeScore, WhaleActivity, PerformanceStats, AIAnalysis, Trade, PaperPortfolio, PaperPosition, PaperTrade, PaperEquityCurve.
- **Persistence**: All learning states, including pattern clusters and social media stats, are persisted in a database.

### Machine Learning and Signal Generation

#### Regression-Based Signal System (Institutional Upgrade)
- **Edge Calculation**: `edge = (μ - cost) / σ`.
- **Signal Output Format**: Comprehensive, including action, confidence, expected_move, uncertainty, edge, cost_estimate, suggested_order_type, urgency, position_size_pct, stop_loss_pct, take_profit_pct, regime, expert_weights.
- **Regression Targets**: μ (4h forward return), σ (uncertainty), P(move>cost), quantiles (p10/p50/p90), MFE/MAE.
- **Market Microstructure Data**: Funding rates, open interest, liquidations, order book depth, taker buy/sell volume from Binance Futures API.
- **Walk-Forward Evaluation**: Purged time-series splits with gap between train/test, after-cost PnL metrics, Sharpe ratio, maximum drawdown.

#### Regime Detection & Mixture-of-Experts
- **Regime Types**: TRENDING, MEAN_REVERTING, HIGH_VOLATILITY, LOW_VOLATILITY, TRANSITION, UNKNOWN.
- **Expert Models**: 4 specialists (trend, mean-reversion, volatility, chaos) plus a gating network.
- **MoE Architecture**: GatingNetwork outputs soft weights, RegimeAwareExpert adapts to detected regime.

#### Self-Supervised Pretraining
- **Techniques**: Masked Time-Series, Next-Step Distribution prediction, Contrastive Learning, Deep clustering for regime discovery.
- **Pretraining Flow**: Self-supervised on billions of timesteps then fine-tuned on trading objectives.

#### GPU Neural Network Training
- **Deep Learning Architectures**: Supports 12+ architectures including Transformers, LSTMs, CNNs, VAEs, GNNs, and Ensembles, trainable on local GPU with real-time status.
- **GPU Trainer API**: FastAPI server providing `/predict` (including regression and multihead) and model management endpoints.
- **Advanced Exit Logic**: Dynamic take-profit targets, MFE tracking, and failure stop detection.
- **Unified Learning Controller**: Synchronizes Strategy Learner, Pattern Memory, and GPU Trainer for consistent historical data processing.
- **Dual Decision Display**: Separate outputs for Combined Learning (Strategy Learner + Pattern Memory) and GPU Neural Network decisions with confidence levels.
- **Edge Tracker**: File-based persistence for monitoring actual signal performance (avg net return, hit rate, expectancy, Sharpe ratio, monthly stability scores).

#### Multi-Head Model Architecture (Institutional Upgrade)
- **Three Output Heads**: Classification (Direction probabilities), Regression (Expected return μ and uncertainty σ), and Quantile (q10, q25, q50, q75, q90).
- **Combined Loss Function**: Integrates CrossEntropyLoss, HuberLoss, GaussianNLLLoss, and Pinball loss.

#### Feature Version Locking (Safety Critical)
- **Mandatory for Live Trading**: Every trained model saves its feature configuration (`FeatureConfig`).
- **FeatureValidator**: Validates and aligns incoming features at inference.
- **Safe Prediction**: Returns HOLD with 0 confidence if feature mismatch.

#### Training-Inference Alignment
- **Key Fixes**: Removed cross-asset features, uses forward-fill for missing features (aborts if >15% missing), explicitly drops OHLCV columns before feature extraction, and locks `sequence_length` to 100.
- **Retraining Required**: Models must be retrained after these fixes.

#### Quantile-Based Predictions & SL/TP Derivation
- **Learned Quantiles**: Multi-head models output true learned quantiles.
- **Entry/SL/TP Derivation**: Mathematically derived from current price and learned quantiles.
- **Probabilistic Fan Chart**: Visualizes return path quantiles.

#### Professional Ensemble Predictor
- **Model Voting**: Transformer, TFT, LSTM, CNN models vote on direction with confidence margin.
- **Regime Gating**: VAE detects market regime (TRENDING, RANGING, CHOPPY, HIGH_VOLATILITY) to adjust thresholds.
- **Risk Filtering**: GNN detects risk regime (RISK_ON, RISK_OFF, CORRELATION_SHOCK) to adjust position sizing.
- **Metric Weighting**: Models weighted by trading metrics (expectancy, precision, profit factor, F1, Sharpe).
- **Position Sizing Adjustment**: Regime-aware position sizing.

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

## API Unit Standards

### Return Values (Critical for UI Display)
- **Standard**: All return values (mu, sigma, quantiles, expectedMove, uncertainty) are returned as **decimal returns** from the API (e.g., 0.01 = 1%, -0.0099 = -0.99%).
- **UI Conversion**: The UI multiplies by 100 only at display time using `formatDecimalAsPercent(val) => (val * 100).toFixed(2)%`.
- **Price Derivation**: `low_price = current_price * (1 + q10)`, `high_price = current_price * (1 + q90)`. No rounding before calculation.
- **API Response Fields**:
  - `units: "decimal_return"` - Indicates values are decimals
  - `derived_low_price` / `derived_high_price` - Backend-computed prices for frontend verification
- **DEBUG Panel**: Neural Network Prediction card has a bug icon toggle that shows raw API values vs displayed values for unit verification.

### Chart-Prediction Anchoring (Critical for Visual Alignment)
- **Problem**: GPU trainer uses live ticker price for predictions, but chart candles may be stale from database.
- **Solution**: Chart always anchors predictions to last candle's close price (not live ticker).
- **Re-anchoring Logic**: Predicted candle prices are converted back to returns relative to ticker price, then re-applied to last candle close: `anchoredPrice = lastCandleClose * (1 + (predictedPrice / tickerPrice - 1))`.
- **STALE Warning**: Amber warning banner appears when `|lastCandleClose - tickerPrice| / tickerPrice > 0.2%` with "Refresh Data" button.
- **Quantile Components**: QuantileFanChart and DerivedTradeLevels receive lastCandleClose as currentPrice with quantiles unchanged (since quantiles are returns, not absolute prices).