# BTC Futures Signal Dashboard

## Overview

This project is an institutional-grade, AI-driven dashboard for generating BTCUSDT futures trading signals. Its primary goal is to deliver sophisticated, AI-powered trade plans by integrating machine learning, real-time market data, and sentiment analysis. The system features a continuous learning loop, adapting to market changes through ongoing data refreshing and model retraining. Key ambitions include providing a robust, selective trading system that leverages advanced AI techniques and comprehensive market insights to generate high-confidence trading opportunities.

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
- **Navigation**: Tabbed interface covering Overview, Signal, Paper Trading, GPU Training, Strategy Learner, Learning, AI Analysis, Indicators, and Performance views. A dedicated GPU Training tab displays real-time status, loss curves, model comparisons, and cross-asset analysis.

### Backend Architecture
- **Runtime**: Node.js with Express.js.
- **Language**: TypeScript with ESM modules.
- **API Pattern**: RESTful endpoints.
- **AI Integration**: OpenAI via Replit AI Integrations.
- **Market Data**: Primary reliance on Binance Vision API, with CoinGecko and CryptoCompare as fallbacks, augmented by a Replit-hosted data proxy.
- **Development**: Vite dev server with HMR.
- **Production**: Static file serving.
- **GPU Trainer Communication**: Bi-directional communication with a local GPU trainer via dedicated API endpoints.

### Data Layer
- **ORM**: Drizzle ORM for PostgreSQL.
- **Schema**: Zod for type-safe validation.
- **Data Separation**: Critical design principle ensuring separation of live sentiment data from historical price/volume data to prevent data leakage.
- **Key Data Models**: Includes Candle, Signal, FuturesData, TechnicalIndicator, MultiTimeframeScore, WhaleActivity, PerformanceStats, AIAnalysis, Trade, PaperPortfolio, PaperPosition, PaperTrade, PaperEquityCurve.
- **Persistence System**: All learning states, including pattern clusters and social media stats, are persisted in a database to ensure continuity across restarts.

### Machine Learning and Signal Generation
- **ML Ensemble Predictor**: Combines rule-based, pattern-based, and OpenAI models with weighted voting for action-based predictions (P(LONG), P(SHORT), P(HOLD)).
- **Pattern Memory System**: Stores and retrieves historical trade setups using cosine similarity.
- **Comprehensive Feature Engine**: Computes 81 features (57 core + 24 embedding) including OHLCV, momentum, volatility, regime, Kalman filters, and 10 cross-asset features (ETH, SOL, BNB correlations, relative strength, momentum divergence, crypto sector momentum).
- **Shot Plan Generation**: Provides detailed trade plans with entry/exit zones, risk-reward ratios, and estimated hold times.
- **Gatekeeper Logic**: Ensures trades are executed only with high confidence, positive edge, and sufficient supporting reasons, promoting selective trading. This includes horizon-specific thresholds, confidence ratio gates, multi-horizon decision logic, and 'NO-TRADE' conditions based on market state or horizon disagreement.
- **Sentiment Integration**: Incorporates Fear & Greed Index, social sentiment, and news sentiment with caching.
- **Multi-timeframe Confluence**: Scores signals across 5m, 15m, 1h, and 4h timeframes.
- **Automated Paper Trading**: Features an ATR-based risk management system, Half-Kelly position sizing, and performance analytics.
- **GPU Neural Network Training**: Supports 12 deep learning architectures trainable on local GPU with real-time status push and multi-timeframe data:
  - Transformers: TransformerPriceModel, TemporalFusionTransformer (TFT)
  - LSTMs: BidirectionalLSTM, StackedLSTM, ConvLSTM
  - CNNs: ResNetPrice, InceptionNet, WaveNet
  - VAEs: MarketVAE, ConditionalVAE
  - GNNs: CrossAssetGNN, TemporalGNN
  - Ensembles: MetaLearner, AttentionEnsemble, MasterEnsemble
- **GPU Trainer API**: FastAPI server at port 8000 with:
  - `/predict` and `/predict/candles` endpoints for model inference
  - `/models/load` and `/models/status` for model management
  - Automatic model loading at startup from checkpoints
  - Architecture-specific factory method handling each model's unique constructor signature
  - Label mapping: 0=SHORT, 1=HOLD/NEUTRAL, 2=LONG (matches training labels)
  - Instantiation error tracking with detailed status reporting
- **Advanced Exit Logic**: Implements dynamic take-profit targets, MFE tracking, and failure stop detection.
- **Unified Learning Controller**: Synchronizes Strategy Learner, Pattern Memory, and GPU Trainer to process consistent historical data ranges, with enhanced training status, ETA calculation, and staged decision logic.
- **Multi-timeframe Data Download**: Parallel download of 1m, 5m, 15m, 1h, 4h data across 4 assets (BTC, ETH, SOL, BNB) with 8 concurrent streams, ordered by timeframe for optimal parallelism.
- **Dual Decision Display**: Separate outputs for Combined Learning (Strategy Learner + Pattern Memory) and GPU Neural Network decisions with confidence levels.
- **Signal Threshold Tuning**: Classification probability-based threshold (scoreThreshold=0.15, minConfidence=0.45, minMargin=0.10) that controls signal frequency to target 2-3 trades/day using directional score (pLong - pShort).
- **Edge Tracker**: File-based persistence system (edge_tracker_state.json) that monitors actual signal performance including avg net return, hit rate, expectancy, Sharpe ratio, and monthly stability scores. API endpoints at /api/edge-metrics and /api/edge-metrics/clear.

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