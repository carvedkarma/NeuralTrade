# Neural Terminal — AI Trading Dashboard

## Overview
An institutional-grade, GPU-accelerated AI trading system for multi-asset crypto futures. The v5 neural network runs on a local GPU trainer; this Replit web app serves as the trading terminal — displaying signals, managing paper/live trading, and tracking performance via WebSocket + ingest API.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend (5-Page Trading Terminal)
Built with React + TypeScript + Vite, using shadcn/ui (Radix UI, Tailwind CSS), Recharts for charts, and a dark navy theme with neon accents.

**Pages:**
- `/` — Command Center: Metrics bar, market grid (6 symbols with sparklines), live signal feed, active positions (with quick close), mini equity curve
- `/live` — Live Trading: Symbol selector tabs, price chart, signal detail panel, positions table (close/partial-close/edit SL-TP actions, SL/TP progress bars), new manual trade panel, signal history
- `/paper` — Paper Trading: Enable/disable toggles, portfolio metrics, equity curve, open/closed positions (with close/partial-close/edit SL-TP actions), configuration
- `/analytics` — Analytics: Performance summary (8 cards), equity curve, per-symbol breakdown, trade distribution, directional analysis
- `/settings` — Settings: GPU connection, account config, model info, risk parameters, data freshness, danger zone

**Key Files:**
- `client/src/App.tsx` — Router with 5 routes wrapped in AppLayout
- `client/src/components/layout/app-layout.tsx` — Collapsible sidebar with GPU status, theme toggle
- `client/src/hooks/use-trading-ws.ts` — WebSocket hook with auto-reconnect, event subscription
- `client/src/index.css` — Dark theme CSS vars, glow effects, glassmorphism, animations

**Design System:**
- Background: deep navy (HSL 225 40% 6%), cards slightly lighter
- Primary: neon green (#22c55e / emerald-400) for profits/longs
- Red-400 for losses/shorts, cyan-500 for info, amber-400 for warnings
- Custom classes: glass-card, glow-green/red/cyan, number-mono, pulse-dot, gradient-border, shimmer, scanline, animate-signal-arrive

### Backend (Node.js + Express + TypeScript)
- **API Routes** (`server/routes.ts`):
  - `GET /api/v5/signals` — v5 model signals from DB
  - `GET /api/v5/performance` — Aggregated stats (total R, win rate, profit factor, per-symbol)
  - `GET /api/v5/equity-curve` — Equity curve from trade records
  - `GET /api/v5/trades` — Filtered trade history
  - `GET /api/system/status` — GPU health, sync status, paper trading state
  - `GET /api/market/prices` — Current prices for 6 symbols with 24h change
  - `GET /api/market/candles` — Candle data for charts
  - `GET/POST /api/paper/*` — Paper trading engine (positions, portfolio, config, enable/disable, start/stop)
  - `POST /api/paper/positions/:id/close` — Manual close position at market
  - `POST /api/paper/positions/:id/partial-close` — Partial close (percent)
  - `PATCH /api/paper/positions/:id/sl` — Update stop loss
  - `PATCH /api/paper/positions/:id/tp` — Update take profit
  - `POST /api/paper/manual-open` — Open manual position
  - `GET /api/paper/risk-alerts` — Real-time risk alerts (SL proximity, drawdown, exposure, duration)
  - `POST /api/ingest/*` — Signal ingest from GPU trainer

- **Key Server Files:**
  - `server/ingest.ts` — Receives GPU trainer events (CYCLE_UPDATE, TRADE_OPEN/CLOSE)
  - `server/gpu-bridge.ts` — GPU trainer connection bridge
  - `server/live-candle-sync.ts` — Real-time 15m candle sync from Binance
  - `server/paper/` — Paper trading engine (routes, storage, engine, config)
  - `server/ws.ts` — WebSocket server for real-time event streaming

### Database (PostgreSQL via Drizzle ORM)
Key tables: `v5_signals`, `live_trade_records`, `live_cycle_logs`, `paper_positions`, `paper_portfolio`, `paper_trades`, `candles`, `settings`

Schema in `shared/schema.ts` with Drizzle + Zod validation.

### v5 Neural Network (runs on local GPU)
- Architecture: V5Forecaster (multi-head: ret_dist, mfe, mae, action[HOLD/LONG/SHORT]) with 85 features, 15m timeframe
- Also supports legacy EnhancedMultiHeadMLP (auto-detected from checkpoint `model_type` field)
- Feature version: v5.0.1_forecaster (FeatureEngineer: STF44 + ENH24 + HTF12 + Regime5 = 85 features)
- Triple-Lane Aggression Engine: CORE/FLOW/SCALP routed by HTF score
- 6 symbols: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, AVAXUSDT
- Per-symbol edge learning (v5.4.0): per-symbol scalers, thresholds, kill switches
- Walk-forward validation: 21/25 folds profitable, +520R cumulative
- Checkpoint loading: searches best_enter_prauc.pt → best_v5_expectancy.pt → best_enter_loss.pt → best_v5_loss.pt
- Scaler: embedded in checkpoint (scaler_center/scaler_scale) or loaded from per_symbol_scalers.joblib/scaler.joblib

## External Dependencies

### Database
- PostgreSQL (Drizzle ORM)

### UI Framework
- Radix UI, Lucide React, class-variance-authority, Recharts

### Data & Validation
- Zod, drizzle-zod, date-fns

### AI / Machine Learning
- OpenAI (for AI analysis features)

### Market Data
- Binance Vision API
