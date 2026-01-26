# BTC Futures Signal Dashboard

## Overview

An institutional-grade AI-driven BTCUSDT futures trading signal dashboard. The project's main purpose is to provide sophisticated, AI-driven trading signals for BTCUSDT futures, integrating machine learning, real-time market data, and sentiment analysis to generate comprehensive trade plans.

Key capabilities include:
- **ML Ensemble Predictor**: Combines rule-based, pattern-based, and OpenAI models with weighted voting for action-based predictions (P(LONG), P(SHORT), P(HOLD)).
- **Pattern Memory System**: Stores and searches historical trade setups using cosine similarity.
- **Comprehensive Feature Engine**: Computes 81 features total (57 core + 24 embedding), including OHLCV data, momentum, volatility, regime, and Kalman filters.
- **Cross-Asset Learning**: Integrates ETH, SOL, BNB data with 10 cross-asset features: correlations (ethBtcCorrelation, solBtcCorrelation, bnbBtcCorrelation), relative strength (ethRelativeStrength, solRelativeStrength, bnbRelativeStrength), momentum divergence (ethMomentumDivergence, solMomentumDivergence, bnbMomentumDivergence), and cryptoSectorMomentum.
- **Shot Plan Generation**: Provides detailed trade plans including entry/exit zones, risk-reward ratios, and estimated hold times.
- **Gatekeeper Logic**: Ensures trades are only taken with high confidence, positive edge, and sufficient supporting reasons, promoting selective trading.
- **Sentiment Integration**: Incorporates Fear & Greed Index, social sentiment, and news sentiment with caching.
- **Real-time Market Data**: Utilizes Binance Vision API as primary, with CoinGecko and CryptoCompare as fallbacks, and includes a data proxy for geoblocked regions.
- **Multi-timeframe Confluence**: Scores signals across 5m, 15m, 1h, and 4h timeframes.
- **Automated Paper Trading**: Features an ATR-based risk management system and performance analytics for simulated trading.
- **GPU Neural Network Training**: 6+ deep learning architectures (Transformer, TFT, LSTM, CNN, VAE, GNN) trainable on local GPU with real-time status push.
- **Continuous Learning Loop**: The system continuously refreshes data and trains models to adapt to market changes.
- **Advanced Exit Logic**: Implements dynamic take-profit targets, MFE tracking, and failure stop detection for optimized trade exits.
- **Persistence System**: All learning states, including pattern clusters and social media stats, are persisted in a database to survive restarts.
- **Multi-Asset Data Management**: GUI-based download/clear for 1-15 years of historical data (BTC, ETH, SOL, BNB). Auto-detects stored data, all 3 learning systems share the same dataset.
- **Unified Learning Controller**: Synchronizes Strategy Learner, Pattern Memory, and GPU Trainer to process the same historical data range.
- **Manual Training Controls**: Start Learning buttons for Strategy Learner and Deep Learning systems - training only begins when explicitly triggered after data is downloaded.
- **Multi-Timeframe Neural Network Data**: Separate data pipeline for GPU neural networks (1m, 5m, 1h, 4h timeframes) distinct from the 15m data used by Strategy Learner and Pattern Memory.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: React with TypeScript, using Vite.
- **Routing**: Wouter.
- **State Management**: TanStack React Query for real-time data updates.
- **UI Components**: shadcn/ui built on Radix UI, styled with Tailwind CSS for theming.
- **Charts**: Recharts for data visualization.
- **Animations**: Framer Motion.
- **Navigation**: Tabbed interface including Overview, Signal, Paper Trading, GPU Training, Strategy Learner, Learning, AI Analysis, Indicators, Performance views.
- **GPU Training Tab**: Dedicated tab displaying real-time GPU trainer status, loss curves, model comparison, and cross-asset analysis (correlations, relative strength, normalized price charts).

### Backend Architecture
- **Runtime**: Node.js with Express.js.
- **Language**: TypeScript with ESM modules.
- **API Pattern**: RESTful endpoints.
- **AI Integration**: OpenAI via Replit AI Integrations.
- **Market Data**: Binance Vision API (primary), CoinGecko, CryptoCompare (fallbacks), with a Replit-hosted data proxy.
- **Development**: Vite dev server with HMR.
- **Production**: Static file serving.
- **GPU Trainer Communication**: Bi-directional communication with a local GPU trainer via dedicated API endpoints (`/api/gpu/push-status`, `/api/gpu/pushed-status`).

### Data Layer
- **ORM**: Drizzle ORM for PostgreSQL.
- **Schema**: Zod for type-safe validation.
- **Data Separation**: Critical design principle separating live sentiment data (for current signals) from historical price/volume data (for backtesting) to prevent data leakage.
- **Key Data Models**: Candle, Signal, FuturesData, TechnicalIndicator, MultiTimeframeScore, WhaleActivity, PerformanceStats, AIAnalysis, Trade, PaperPortfolio, PaperPosition, PaperTrade, PaperEquityCurve.

### Build System
- **Client Build**: Vite bundles React app to `dist/public`.
- **Server Build**: esbuild bundles server to `dist/index.cjs`.

## External Dependencies

### Database
- PostgreSQL (configured via `DATABASE_URL`).
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
- TypeScript with strict mode.

## Recent Changes (January 2026)

### Unified Regime Detection System
- **Shared ATR-Percentile Classifier**: `classifyRegime(candles, idx?)` in `feature-engine.ts` provides consistent regime detection across all components.
- **6 Market Regimes**: trend_up, trend_down, shock (75th+ ATR percentile), quiet (<25th percentile), ranging (25-50th), chop (default).
- **Shared Risk Parameters**: `getRegimeRiskParams(regime)` provides consistent stop/TP multipliers and R:R ratios.
- **Full Consistency**: signal-engine, paper-engine, and strategy-learner all use the shared classifier and risk params.

### Research-Backed Improvements
- **Triple Barrier Labeling**: TP/SL/Time barrier simulation with intrabar timing heuristic (20-50% accuracy improvement over fixed-horizon).
- **Pattern Quality**: Similarity threshold increased 0.6→0.75, MIN_SAMPLES 50→100.
- **Meta-Labeling Filter (WIRED)**: Secondary confidence filter integrated into paper trading gate logic. Requires 55%+ meta-label confidence for execution (research: improves precision 37%→56%).
- **Half-Kelly Position Sizing (WIRED)**: Dynamic position sizing based on historical edge now active in paper trading. Captures ~75% optimal growth with ~50% less drawdown. Caps at 20% max position, 1% minimum.
- **Feature Sanitization**: All 81 features validated with `sanitizeFeatureVector()` to prevent NaN/Infinity propagation into ML models. Safe defaults for all feature values.
- **Trade Audit Enhancement**: Added `sizingMethod` tracking to audit logs for risk governance (shows Half-Kelly vs fixed sizing).
- **Max 2 Vetoes Rule**: Gating logic allows up to 2 vetoes before blocking trades. Improves trade flow while maintaining selectivity.

### Precision Audit Improvements (January 2026)
- **Adaptive Normalization (DAIN-style)**: `AdaptiveNormalizer` class uses rolling z-score normalization (100-sample window, ±3 clipping) for 30+ features. Uses only past data to prevent leakage.
- **Feature-Vector Embargo**: `applyFeatureVectorEmbargo()` filters patterns with >95% similarity to train/test boundary, preventing data leakage from feature-similar patterns.
- **Sharpe Ratio Sanity Check**: Warning system when Sharpe > 3.0 (overfitting) or > 2.5 (unusually high). Requires ≥20 samples for reliable detection.
- **Monte Carlo Simulation**: Runs 1000 simulations with shuffled trade returns to calculate 5th/50th/95th percentile final equity and max drawdown. Determines statistical significance and confidence level.
- **Validation Logging**: Verifies win rate calculations match actual returns using same threshold criteria to detect computation bugs.
- **Train/Test Split**: 80/20 walk-forward validation ensures win rate stats are computed on out-of-sample data only.
- **Model Display Fix**: Strategy Learner appears as 4th model entry with weight=0 (backtest stats only), separated from live predictions to prevent double-counting.

### GPU Export API (January 2026)
- **Multi-Timeframe Data Export**: `/api/gpu-export/multi-tf` exports aligned candles across 1m/5m/15m/1h/4h with as-of joins.
- **Feature Specs Endpoint**: `/api/gpu-export/feature-specs` returns 28 feature definitions (returns, volatility, EMA ratios, RSI, MACD, ATR, candle shape, volume) for Python trainer parity.
- **Trainer Config Endpoint**: `/api/gpu-export/trainer-config` returns RTX 4070 optimized settings (Transformer 6 layers, d_model=256, Huber+directional loss).
- **Walk-Forward Folds**: `/api/gpu-export/walk-forward-folds` implements rolling 12mo train/2mo val/2mo test windows.
- **Prediction Ingestion**: `/api/gpu-export/predictions` receives GPU predictions, stores in ml-predictor cache with 5min TTL.
- **Ensemble Integration**: GPU predictions integrated with weights: rule=0.25, pattern=0.25, ai=0.20, gpu=0.30 (when GPU available).
- **Quantile Uncertainty**: Uses (q90-q10)/|q50| spread for confidence, with NaN guards and divide-by-zero protection.
- **Python Pipeline**: `gpu_trainer/data/pipeline.py` contains `DashboardAPIFetcher` class with async/sync methods.
- **Feature Parity**: `compute_features_from_spec()` in Python matches TypeScript FEATURE_SPECS exactly, including `volatility_regime` with quantile_bucket.

### Key Files for ML/Learning
- `server/feature-engine.ts`: Feature computation + shared regime classifier
- `server/signal-engine.ts`: Signal generation and shot plans
- `server/strategy-learner.ts`: Reinforcement learning on historical data
- `server/pattern-memory.ts`: Pattern storage and similarity matching
- `server/paper/engine.ts`: Paper trading execution with regime-adaptive stops/TPs
- `server/gpu-data-export.ts`: GPU export API endpoints and data preparation
- `server/ml-predictor.ts`: ML ensemble predictor with GPU integration
- `gpu_trainer/data/pipeline.py`: Python data fetcher for GPU trainer
- `gpu_trainer/config.py`: GPU trainer configuration