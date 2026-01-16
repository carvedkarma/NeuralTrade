# BTC Futures Signal Dashboard

## Overview

A real-time BTCUSDT futures trading signal dashboard that displays trading signals, regime detection, feature analysis, and risk management tools for 15-minute timeframe trading. The application predicts market direction (LONG/SHORT/HOLD) with probability estimates for trend-up, trend-down, and chop regimes, incorporating futures-specific data like funding rates, open interest, and liquidations.

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

### Backend Architecture
- **Runtime**: Node.js with Express.js
- **Language**: TypeScript with ESM modules
- **API Pattern**: RESTful endpoints serving JSON data
- **Development**: Vite dev server with HMR integration for development mode
- **Production**: Static file serving from built assets

### Data Layer
- **ORM**: Drizzle ORM configured for PostgreSQL
- **Schema**: Zod schemas in shared directory for type-safe validation across client and server
- **Current State**: Uses in-memory mock data generation in storage.ts (generates random candles, signals, trades, and futures metrics)

### Key Data Models
- **Candle**: OHLCV data for price charts
- **Signal**: Trading signal with confidence, probabilities (up/down/chop), expected return
- **FuturesData**: Funding rate, open interest, long/short ratio, liquidations, basis
- **Feature**: Signal driver with importance scoring
- **Trade**: Trade history with entry/exit prices and P&L

### API Endpoints
- `GET /api/dashboard` - Returns complete dashboard data including candles, signals, futures metrics, features, trades, and performance stats
- `POST /api/refresh` - Forces data refresh and returns updated dashboard data

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