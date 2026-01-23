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

- **ML Ensemble Predictor**: Combines 3 models (rule-based, pattern, OpenAI) with weighted voting
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
  - Social & Global Awareness: Tracks reads from Fear & Greed Index, CryptoPanic News, Twitter/X, Reddit
  - Historical Data Learning: Shows patterns learned, backtest trades, historical win rate
  - Data Sources tracking with attempts/successes for each API
  - Pattern Memory stats with regime breakdown
  - Feature Engine with 40 features across 6 categories
  - ML Ensemble Performance for all 3 models

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: React with TypeScript, using Vite as the build tool
- **Routing**: Wouter for lightweight client-side routing
- **State Management**: TanStack React Query for server state management with automatic refetching every 15 seconds
- **UI Components**: shadcn/ui component library built on Radix UI primitives
- **Styling**: Tailwind CSS with CSS custom properties for theming (light/dark mode support)
- **Charts**: Recharts for candlestick and data visualization
- **Animations**: Framer Motion for smooth UI transitions
- **Tab Navigation**: Overview, Signal, AI Analysis, Indicators, Performance views

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

### API Endpoints
- `GET /api/dashboard` - Returns complete dashboard data
- `POST /api/refresh` - Forces data refresh
- `POST /api/ai/analyze` - Triggers AI market analysis
- `POST /api/strategy/start` - Start automated paper trading
- `POST /api/strategy/stop` - Stop automated trading
- `PATCH /api/strategy/settings` - Update strategy parameters

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