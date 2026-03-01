# Neural Terminal — AI Trading Dashboard

## Overview
An institutional-grade, GPU-accelerated AI trading system for multi-asset crypto futures. The v5 neural network runs on a local GPU trainer; this Replit web app serves as the trading terminal — displaying signals, managing paper/live trading, and tracking performance via WebSocket + ingest API.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend (5-Page Trading Terminal)
Built with React + TypeScript + Vite, using shadcn/ui (Radix UI, Tailwind CSS), Recharts for charts, and a dark navy theme with neon accents.

**Pages:**
- `/` — Command Center: Live status banner ("V5 Neural Engine LIVE"), metrics bar (6 KPIs), market grid (6 symbols with sparklines), real-time signal feed with V5 model outputs (action probs, ret_mu, MFE/MAE, lane routing), heartbeat sparkline, active positions (with quick close), mini equity curve
- `/live` — Live Trading: Symbol selector tabs, price chart, market scanner indicator with per-symbol scan times, signal detail panel with V5 cycle log data (p_enter, direction, decision, lane, action prob bar, ret_mu/MFE/MAE, reasons), recent cycles mini-history, positions table (close/partial-close/edit SL-TP actions, SL/TP progress bars), new manual trade panel, signal history
- `/paper` — Paper Trading: Enable/disable toggles, portfolio metrics, equity curve, open/closed positions (with close/partial-close/edit SL-TP actions), configuration
- `/analytics` — Analytics: 10 stat cards (Total R, Trades, Win Rate, Profit Factor, Expectancy, Sharpe Ratio, Sortino Ratio, Max Drawdown, Max Consec Wins, Avg Hold Time), equity curve, rolling 7d/30d performance, hourly heatmap, per-symbol equity curves, win/loss streaks chart, trade duration histogram, R-multiple distribution, per-symbol breakdown table with edge status, monthly/weekly P&L table
- `/settings` — Settings: GPU connection (push-based detection with "Last activity: X ago"), account config, model info, risk parameters, data freshness, danger zone

**Key Files:**
- `client/src/App.tsx` — Router with 5 routes wrapped in AppLayout
- `client/src/components/layout/app-layout.tsx` — Collapsible sidebar with GPU status, system live indicator (pulse dot + "LIVE" text, last scan time, cycle count today), theme toggle
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
  - `GET /api/v5/performance` — Aggregated stats (total R, win rate, profit factor, per-symbol, Sharpe ratio, Sortino ratio, expectancy, max consecutive wins/losses, avg hold duration, hourly heatmap, per-symbol equity curves, streaks, trade duration distribution, monthly/weekly P&L, rolling 7d/30d metrics)
  - `GET /api/v5/equity-curve` — Equity curve from trade records
  - `GET /api/v5/trades` — Filtered trade history (supports `symbol`, `source=paper|all|live`, `limit`, `offset`; paper source queries `paper_trade_history` table)
  - `GET /api/system/status` — GPU health (push-based detection via lastActivity), sync status, paper trading state, cyclesToday, lastCycleTs
  - `GET /api/market/prices` — Current prices for 6 symbols with 24h change
  - `GET /api/market/candles` — Candle data for charts
  - `GET /api/live/cycle-logs` — Cycle log history (supports `symbol`, `limit` params)
  - `POST /api/live/cycle-log` — Receive cycle log from GPU trainer, broadcasts via WebSocket. **Auto-trade**: when decision=ENTER and paper trading is enabled, automatically opens a paper position via `manualOpenPosition` with source="v5_signal", SL/TP calculated from threshold_used, 2:1 R:R ratio. Safety checks: max 1 position per symbol, max 6 total open positions.
  - `GET/POST /api/paper/*` — Paper trading engine (positions, portfolio, config, enable/disable, start/stop). Positions endpoint returns enriched data: `currentPrice`, `pnlR`, `pnlUsdt`, `takeProfit` (mapped from `tp1`), `entryTime` (mapped from `entryTs`)
  - `POST /api/paper/positions/:id/close` — Manual close position at market
  - `POST /api/paper/positions/:id/partial-close` — Partial close (percent)
  - `PATCH /api/paper/positions/:id/sl` — Update stop loss
  - `PATCH /api/paper/positions/:id/tp` — Update take profit
  - `POST /api/paper/manual-open` — Open manual position
  - `GET /api/paper/risk-alerts` — Real-time risk alerts (SL proximity, drawdown, exposure, duration)
  - `GET /api/paper/trade-history` — Paper trade history (supports `symbol`, `limit`, `offset`)
  - `POST /api/ingest/*` — Signal ingest from GPU trainer
  - `POST /api/gpu/push-prediction` — Multi-head prediction push from GPU trainer (records activity for connection detection)

- **Key Server Files:**
  - `server/ingest.ts` — Receives GPU trainer events (CYCLE_UPDATE, TRADE_OPEN/CLOSE), saves V5 model outputs (retMu, mfePred, maePred, pHold, pLong, pShort)
  - `server/gpu-bridge.ts` — GPU trainer connection bridge with push-based detection (lastIngestActivity timestamp, recordActivity(), 5-minute window for availability)
  - `server/live-candle-sync.ts` — Real-time 15m candle sync from Binance
  - `server/paper/` — Paper trading engine (routes, storage, engine, config)
  - `server/paper/storage.ts` — Paper storage with getPositionsBySymbol(), recordTradeClose(), getTradeHistory()
  - `server/paper/engine.ts` — Paper engine with position monitor (30s interval, checks SL/TP1/TP2/time-stop for all open positions across all symbols, broadcasts TRADE_CLOSE via WebSocket)
  - `server/ws.ts` — WebSocket server for real-time event streaming (broadcasts CYCLE_UPDATE on push)

### Database (PostgreSQL via Drizzle ORM)
Key tables: `v5_signals`, `live_trade_records`, `live_cycle_logs` (with V5 fields: ret_mu, mfe_pred, mae_pred, p_hold, p_long, p_short), `paper_positions` (with `source` field: "v5_signal"|"manual"|"auto"), `paper_portfolio`, `paper_trades`, `paper_trade_history` (complete trade records with R metrics per asset), `candles`, `settings`

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
