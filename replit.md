# BTC Futures Signal Dashboard

## Overview

A powerful AI-driven BTCUSDT futures trading signal dashboard featuring:
- Kalman filter trend detection (70/250 periods)
- Real-time market data from Binance Vision API (primary), CoinGecko and CryptoCompare fallbacks
- OpenAI-powered market analysis and signal generation
- 10+ technical indicators (RSI, MACD, Bollinger Bands, ADX, Stochastic, OBV, VWAP, ATR)
- Multi-timeframe confluence scoring (5m/15m/1h/4h)
- Whale activity detection for large order flow ($50K+ orders)
- Automated paper trading with ATR-based risk management
- Comprehensive performance analytics

The system operates with live data only - no simulated fallback. Shows error message when all APIs unavailable.

## Recent Changes (January 2026)

- **Binance Vision API** as primary data source (data-api.binance.vision) - reliable, high rate limits
- **CoinGecko API** as first fallback with aggressive caching
- **CryptoCompare API** as second fallback
- Removed simulated data - shows error when all APIs fail
- Added OpenAI integration for AI market analysis (uses Replit AI Integrations)
- Implemented 8 technical indicators with bull/bear signals
- Added multi-timeframe analysis scoring system
- Created whale activity detection for large orders
- Added performance statistics tracking
- Dashboard now has tabbed navigation: Overview, AI Analysis, Indicators, Performance
- Added Live data source indicator badge (Binance/CoinGecko/CryptoCompare)

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
- **Tab Navigation**: Overview, AI Analysis, Indicators, Performance views

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