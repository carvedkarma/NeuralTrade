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

### Key Files for ML/Learning
- `server/feature-engine.ts`: Feature computation + shared regime classifier
- `server/signal-engine.ts`: Signal generation and shot plans
- `server/strategy-learner.ts`: Reinforcement learning on historical data
- `server/pattern-memory.ts`: Pattern storage and similarity matching
- `server/paper/engine.ts`: Paper trading execution with regime-adaptive stops/TPs