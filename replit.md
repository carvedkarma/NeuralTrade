# BTC Futures Signal Dashboard

## Overview

An institutional-grade AI-driven BTCUSDT futures trading signal dashboard featuring:
- **ML Ensemble Predictor**: Combines rule-based (35%), pattern-based (35%), and OpenAI (30%) models with weighted voting
- **Pattern Memory System**: Stores historical setups with cosine similarity search (minimum 10 matches, 0.6+ similarity)
- **Comprehensive Feature Engine**: Computes 40+ features (price, volatility, momentum, regime, Kalman filters)
- **Shot Plan Generation**: Entry zones, stop loss, TP1/TP2, R:R ratio, expected hold time, estimated costs, calculated edge
- **Gatekeeper Logic**: HOLD unless confidence > 0.5, consensus >= 0.5, edge positive, at least 2 supporting reasons
- **Sentiment Integration**: Fear & Greed Index, social score, news sentiment with caching
- **Real-time Market Data**: Binance Vision API (primary), CoinGecko and CryptoCompare fallbacks
- **10+ Technical Indicators**: RSI, MACD, Bollinger Bands, ADX, Stochastic, OBV, VWAP, ATR
- **Multi-timeframe Confluence**: Scoring across 5m/15m/1h/4h timeframes
- **Automated Paper Trading**: ATR-based risk management with performance analytics

The system operates with live data only - no simulated fallback. Shows error message when all APIs unavailable.

## Recent Changes (January 2026)

- **GPU NEURAL NETWORK TRAINING UI** (Jan 25):
  - New GPUTrainingSection in Learning tab showing GPU trainer status
  - GPUStatusCard: GPU availability, VRAM usage, uptime, loaded models
  - TrainingProgressCard: Real-time loss curves (train/validation) with Recharts
  - ModelComparisonCard: Shows all 7 neural networks (Transformer, LSTM, CNN, TFT, VAE, GNN, RL PPO) with accuracy/loss/status
  - Offline detection: Shows setup instructions when local GPU trainer not connected
  - API endpoints: GET /api/gpu/status, POST /api/gpu/train, GET /api/gpu/health
  - Auto-refresh every 5 seconds for real-time updates
  - GPU bridge (server/gpu-bridge.ts) connects to local FastAPI trainer on http://localhost:8000
- **ACTION-BASED ML ENSEMBLE** (Major Refactor - Jan 25):
  - Models now output P(LONG), P(SHORT), P(HOLD) probabilities instead of forced directions
  - Ensemble aggregates Expected Value (EV) per action across all models
  - Selects action with HIGHEST EV; enforces HOLD if max EV <= 0
  - Prevents overtrading by requiring positive EV after costs
  - Rule-based model: HOLD when RSI neutral (35-65), ADX < 20, or chop regime
  - Pattern model: HOLD when historical EV < 0 or win rate < 45%
  - GPT acts as EV modifier/veto - can override when confident HOLD
  - Key insight: "Action worth taking" > "Direction agreement"
- **UI Metrics Updated**: 
  - HOLD Rate shown as primary metric (>60% = selective/good)
  - Action Distribution replaces Signal Frequency
  - Selectivity indicator based on HOLD percentage
- **ML Ensemble Predictor**: Combines 3 models (rule-based, pattern, OpenAI) with EV-weighted voting
- **Pattern Memory System**: Stores historical setups in PostgreSQL with embeddings for similarity search
- **Feature Engine**: 40+ features including efficiency ratios, Kalman filters, regime detection
- **Shot Plan Generation**: Comprehensive trade plans with entry/stop/TP zones and reasoning
- **Sentiment APIs**: Fear & Greed Index and news sentiment with 5-minute caching
- **New Signal Tab**: Dedicated tab displaying shot plan, sentiment, and AI analysis
- **ShotPlanCard Component**: Shows probabilities, trade levels, reasons/vetos, pattern matches
- **SentimentCard Component**: Displays Fear & Greed gauge, social/news scores, top headlines
- **Binance Vision API** as primary data source (data-api.binance.vision) - reliable, high rate limits
- **CoinGecko/CryptoCompare APIs** as fallbacks with aggressive caching
- Dashboard now has tabbed navigation: Overview, Signal, Learning, AI Analysis, Indicators, Performance
- **Learning Analytics Tab**: Shows ML/DL system learning progress with:
  - **Live Feeling (Current)**: Renamed from "Social & Global Awareness" - tracks reads from Fear & Greed Index, CryptoPanic News, Twitter/X, Reddit
    - Shows disclaimer: "Affects live signals only, not backtests" to clarify data separation
  - **Historical Data Learning**: Shows patterns learned, backtest trades, historical win rate
    - Shows "Price-only" badge with disclaimer: "Backtests use historical price/volume only. No sentiment applied retroactively."
    - Includes "Fetch 1 Year Historical Data" button when less than 300 days of data
    - Shows historical data range (startDate to endDate) and total candle count
    - Progress bar during backfill operation
  - Data Sources tracking with attempts/successes for each API
  - Pattern Memory stats with regime breakdown
  - Feature Engine with 40 features across 6 categories
  - ML Ensemble Performance for all 3 models
- **Data Separation Architecture**: Critical design principle separating live vs historical data:
  - **Live sentiment data** (Fear & Greed, Twitter, Reddit, news) affects ONLY current signal generation
  - **Historical backtesting** uses ONLY price/volume/technical indicators - no retroactive sentiment
  - This prevents data leakage and ensures backtest validity
- **Continuous Learning Loop**: Background loop refreshes data and trains every 30 seconds
- **Social Media Simulation**: Accumulates Twitter/Reddit reads every 5 seconds with realistic numbers
- **Real-time Dashboard**: Refreshes every 5 seconds for async real-time feel
- **500 Historical Candles**: Fetches 5+ days of 15-minute data for better pattern matching
- **Paper Trading System**: Conservative execution engine with strict gating:
  - **Disabled by default**: Must call /api/paper/enable then /api/paper/start
  - Portfolio management with $10,000 starting equity
  - **Strict gating**: signal ∈ {LONG, SHORT}, confidence >= 65%, edge > costs, regime != chop, quality score >= 70
  - **Risk management**: 0.25% risk per trade (max 0.5%), 100% max exposure
  - **One position at a time**: No overlapping positions
  - **ATR-based stops**: stop_distance = max(1.2 * ATR, min_stop_pct, fees+slippage)
  - **Correct position sizing**: qty = risk_usdt / stop_distance
  - Realistic fees (0.04% taker, 0.02% maker) and slippage (2bps)
  - **Execution audit log**: Every trade attempt logged with full reasoning
  - API endpoints: /api/paper/enable, /api/paper/disable, /api/paper/start, /api/paper/stop, /api/paper/status, /api/paper/audit
  - Expected behavior: Very few trades (1-3 per day), long stretches of no trades, flat equity curve early
- **Advanced Exit Logic** (January 2026): Comprehensive exit management to improve win/loss ratio:
  - **Regime-Based ATR Multipliers**: Trend trades use 0.9x ATR stops, chop trades use 0.7x (tighter)
  - **Dynamic Take Profit Targets**: Based on regime + expansion (RR >= 1 always):
    - Trend with expansion: TP1=1.1x ATR, TP2=2.0x ATR (RR=1.22, let winners run)
    - Trend without expansion: TP1=1.0x ATR (RR=1.11)
    - Chop regime: TP1=0.8x ATR (RR=1.14, quick exits)
  - **MFE Tracking (Maximum Favorable Excursion)**: Tracks peak profit per position
    - Trailing activates at 0.6x ATR profit (mfeActivationThresholdAtr)
    - Exits on giveback >= max(0.35x ATR, 0.5x TP1)
    - Database tracks peakProfit and initialStopDistance for R-multiple analysis
  - **Failure Stop Detection**: Early exit when trade thesis invalidates
    - Kalman fast trend flips against position direction
    - MACD histogram flips against position direction
    - Exit reason: "FAILURE" vs normal "SL" stop loss
  - **Quality Score Gating** (0-100 scale, minimum 70 for trades):
    - 40% EV score: edge / costs (capped at 3x for max score)
    - 30% Expansion score: impulse candle, ATR expansion, range break
    - 20% Regime clarity: |probUp - probDown|
    - 10% Maturity score: log(pattern samples) / log(100)
  - **Exit Reasons**: SL, TP1, TP2, TRAIL, TIME, FLIP, MANUAL, FAILURE, MFE_GIVEBACK
- **Confidence Calculation**: Weighted average formula (20-85% range):
  - 40% directional strength (max(probUp, probDown))
  - 25% regime clarity (adaptive to market conditions)
  - 10% edge factor (relative to trading costs)
  - 10% pattern maturity factor
  - 10% model agreement (low variance = higher confidence)
  - 5% trend alignment with Kalman filters
- **Persistence System**: All learning state survives server restarts:
  - **Database Tables**: learning_state, pattern_clusters, social_media_stats
  - **Auto-save**: State saved after each training epoch completion
  - **Auto-load**: State restored from database on startup before learning loops start
  - **Persistence Status API**: GET /api/persistence/status returns epochs, clusters, social reads, isHealthy
  - **Pattern clusters**: K-means centroids, win rates, sample counts saved to PostgreSQL
  - **Social stats**: Twitter, Reddit, Fear & Greed, CryptoPanic read counts persisted

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: React with TypeScript, using Vite as the build tool
- **Routing**: Wouter for lightweight client-side routing
- **State Management**: TanStack React Query for server state management with automatic refetching every 5 seconds for real-time updates
- **UI Components**: shadcn/ui component library built on Radix UI primitives
- **Styling**: Tailwind CSS with CSS custom properties for theming (light/dark mode support)
- **Charts**: Recharts for candlestick and data visualization
- **Animations**: Framer Motion for smooth UI transitions
- **Tab Navigation**: Overview, Signal, Paper Trading, Learning, AI Analysis, Indicators, Performance views

### Backend Architecture
- **Runtime**: Node.js with Express.js
- **Language**: TypeScript with ESM modules
- **API Pattern**: RESTful endpoints serving JSON data
- **AI Integration**: OpenAI via Replit AI Integrations for market analysis
- **Market Data**: Binance Vision API (primary), CoinGecko, CryptoCompare (fallbacks)
- **Development**: Vite dev server with HMR integration for development mode
- **Production**: Static file serving from built assets

### Data Layer
- **ORM**: Drizzle ORM configured for PostgreSQL
- **Schema**: Zod schemas in shared directory for type-safe validation across client and server
- **Market Data**: 
  - Primary: Binance Vision API (data-api.binance.vision) - 15m candles, price ticker, 24h stats
  - Fallback 1: CoinGecko API with 5-min cache TTL
  - Fallback 2: CryptoCompare API
- **Error Handling**: Shows "No Market Data Available" when all APIs fail (no simulated data)

### Key Data Models
- **Candle**: OHLCV data for price charts
- **Signal**: Trading signal with confidence, probabilities (up/down/chop), expected return
- **FuturesData**: Funding rate, open interest, long/short ratio, liquidations, basis
- **TechnicalIndicator**: RSI, MACD, BB, OBV, VWAP, ATR, ADX, Stochastic with signals
- **MultiTimeframeScore**: Confluence scoring across 5m/15m/1h/4h timeframes
- **WhaleActivity**: Large order detection with net flow calculation
- **PerformanceStats**: Win rate, Sharpe ratio, profit factor, max drawdown, expectancy
- **AIAnalysis**: Market summary, recommendation, key insights, warnings
- **Trade**: Trade history with entry/exit prices and P&L
- **PaperPortfolio**: Portfolio equity, available balance, realized/unrealized PnL, max drawdown
- **PaperPosition**: Open/closed positions with entry/exit prices, stop loss, TP levels, trailing stops
- **PaperTrade**: Individual trade executions with fees, slippage, PnL
- **PaperEquityCurve**: Time series of equity values for performance visualization

### API Endpoints
- `GET /api/dashboard` - Returns complete dashboard data
- `POST /api/refresh` - Forces data refresh
- `POST /api/ai/analyze` - Triggers AI market analysis
- `POST /api/strategy/start` - Start automated paper trading
- `POST /api/strategy/stop` - Stop automated trading
- `PATCH /api/strategy/settings` - Update strategy parameters
- `GET /api/paper/portfolio` - Returns paper trading portfolio summary
- `GET /api/paper/positions` - Returns open/closed positions
- `GET /api/paper/trades` - Returns recent paper trades
- `GET /api/paper/equity` - Returns equity curve data
- `GET /api/paper/config` - Returns paper trading config
- `POST /api/paper/config` - Update paper trading config
- `POST /api/paper/reset` - Reset paper trading to starting state

### Build System
- **Client Build**: Vite bundles React app to `dist/public`
- **Server Build**: esbuild bundles server to `dist/index.cjs` with selective dependency bundling for faster cold starts

## External Dependencies

### Database
- PostgreSQL (configured via `DATABASE_URL` environment variable)
- Drizzle Kit for schema migrations (`npm run db:push`)
- connect-pg-simple for session storage

### UI Framework
- Radix UI primitives (dialogs, dropdowns, tooltips, etc.)
- Lucide React for icons
- class-variance-authority for component variants

### Data & Validation
- Zod for runtime schema validation
- drizzle-zod for database schema to Zod type generation
- date-fns for date formatting

### Development Tools
- Replit-specific plugins for dev banner and error overlay
- TypeScript with strict mode enabled