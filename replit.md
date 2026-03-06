# Neural Terminal — AI Trading Dashboard

## Overview
An institutional-grade, GPU-accelerated AI trading system for multi-asset crypto futures. The v5 neural network runs on a local GPU trainer; this Replit web app serves as the trading terminal — displaying signals, managing paper/live trading, and tracking performance via WebSocket + ingest API.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend (6-Page Trading Terminal)
Built with React + TypeScript + Vite, using shadcn/ui (Radix UI, Tailwind CSS), Recharts for charts, and a dark navy theme with neon accents.

**Pages:**
- `/` — Command Center: Live status banner ("V5 Neural Engine LIVE"), metrics bar (6 KPIs), market grid (6 symbols with sparklines), real-time signal feed with V5 model outputs (action probs, ret_mu, MFE/MAE, V5 composite score), heartbeat sparkline, active positions (with quick close), mini equity curve
- `/live` — Live Trading: Symbol selector tabs, price chart, market scanner indicator with per-symbol scan times, signal detail panel with V5 cycle log data (p_enter, direction, decision, V5 score/threshold, action prob bar, ret_mu/MFE/MAE, reasons), recent cycles mini-history, positions table (close/partial-close/edit SL-TP actions, SL/TP progress bars), new manual trade panel, signal history
- `/paper` — Paper Trading: Enable/disable toggles, portfolio metrics, equity curve, open/closed positions (with single-click close/partial-close/edit SL-TP actions, live 1s price updates), **position health gauges** (0-100 score with color-coded risk levels), **neural status badges** (shows latest Neural PM action), **MFE tracker** (peak profit vs giveback), **breakeven indicator**, **adaptive trail visualization**, ping latency badge, configuration
- `/analytics` — Analytics: Live/Paper source toggle (default: Paper), 10 stat cards (Total R, Trades, Win Rate, Profit Factor, Expectancy, Sharpe Ratio, Sortino Ratio, Max Drawdown, Max Consec Wins, Avg Hold Time), equity curve, rolling 7d/30d performance, hourly heatmap, per-symbol equity curves, win/loss streaks chart, trade duration histogram, R-multiple distribution, per-symbol breakdown table with edge status, monthly/weekly P&L table. Paper mode adds: Total P&L (USDT), Total Risk (USDT), ROI on Risk, Leverage Tier Performance table. **Advanced Analytics**: MFE vs Result scatter plot, Wasted Edge card, Capture Ratio chart, Optimal Exit Simulator (interactive slider), Neural Manager Performance table, Edge Decay Analysis chart, AI-powered trade analysis (OpenAI)
- `/training` — Training Monitor: Live GPU training progress page. Status banner (pulse when training active, ETA, progress bar), overview cards (folds/epochs/elapsed/ETA/total R/avg expectancy), model knowledge gauge, live loss curves (train+val with fold boundaries, togglable L_ret/L_mfe/L_mae/L_action components), action accuracy trend chart, expectancy & threshold evolution chart, walk-forward fold results table (per-fold R/WR/PF/Sharpe/MaxDD), per-fold R bar chart, per-symbol edge heatmap (symbols×folds), collapsible config panel, session history list. Real-time via WebSocket (TRAINING_EPOCH, TRAINING_FOLD_END, TRAINING_SESSION_START/END events) + 10s polling
- `/settings` — Settings: GPU connection (push-based detection with "Last activity: X ago"), account config, model info, risk parameters, data freshness, danger zone

**Key Files:**
- `client/src/App.tsx` — Router with 6 routes wrapped in AppLayout
- `client/src/components/layout/app-layout.tsx` — Collapsible sidebar with GPU status, system live indicator (pulse dot + "LIVE" text, last scan time, cycle count today), theme toggle
- `client/src/hooks/use-trading-ws.ts` — WebSocket hook with auto-reconnect, event subscription
- `client/src/hooks/use-ping.ts` — Ping/latency monitor hook (5s interval, round-trip to /api/ping)
- `client/src/components/ping-badge.tsx` — Color-coded ping latency badge (green/amber/red)
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
  - `POST /api/live/cycle-log` — Receive cycle log from GPU trainer, broadcasts via WebSocket (includes v5Score, v5Threshold, v5Side). Runs Neural Position Manager for symbols with open positions. Does NOT auto-open trades (auto-trade moved to `/api/live/trade`).
  - `POST /api/live/trade` — Receive executed trade record from GPU trainer (only called after portfolio filtering + execution module confirmation). **Auto-trade**: when paper trading is enabled, automatically opens a paper position via `manualOpenPosition` with source="v5_signal", using GPU trainer's SL/TP (required, no fallback). Safety checks: max 1 position per symbol, max 6 total open positions. Also triggers live Bybit auto-trade when live trading is enabled. **Signal-strength leverage**: v5Score is passed to `computeSignalLeverage()` which maps score to leverage tiers (≥0.02→1x, ≥0.05→2x, ≥0.10→3x, ≥0.15→5x, ≥0.40→10x, ≥0.50→30x). Position qty is multiplied by leverage; risk per trade stays constant.
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
  - `GET /api/bybit/status` — Bybit connection status, live trading enabled flag, config, execution service status
  - `GET /api/bybit/positions` — Current open positions (from execution service cache if connected, otherwise direct Bybit API)
  - `GET /api/bybit/balance` — USDT wallet balance (from execution service cache if connected, otherwise direct)
  - `POST /api/bybit/close/:symbol` — Close position on Bybit via proxy (market order, reduceOnly)
  - `POST /api/bybit/amend/:symbol` — Amend SL/TP on open Bybit position via proxy (setTradingStop)
  - `POST /api/bybit/toggle` — Enable/disable live trading (requires testConnection success)
  - `PATCH /api/bybit/config` — Update live trading config (riskPerTradePct, maxDailyLossUsdt)
  - `POST /api/execution/push-state` — Receive Bybit positions/balance push from GPU trainer execution service
  - `GET /api/execution/state` — Current execution service state and connection status

- **Key Server Files:**
  - `server/ingest.ts` — Receives GPU trainer events (CYCLE_UPDATE, TRADE_OPEN/CLOSE), saves V5 model outputs (retMu, mfePred, maePred, pHold, pLong, pShort)
  - `server/gpu-bridge.ts` — GPU trainer connection bridge with push-based detection (lastIngestActivity timestamp, recordActivity(), 5-minute window for availability). **GPU URL auto-registration**: GPU trainer sends its ngrok URL via `gpu_callback_url` field in push payloads or `POST /api/gpu/register`. Bybit client uses registered URL via `getEffectiveGpuUrl()`.
  - `server/live-candle-sync.ts` — Real-time 15m candle sync from Binance
  - `server/paper/` — Paper trading engine (routes, storage, engine, config)
  - `server/paper/storage.ts` — Paper storage with getPositionsBySymbol(), recordTradeClose(), getTradeHistory()
  - `server/paper/engine.ts` — Paper engine with position monitor (30s interval, checks SL/TP1/TP2 for all open positions across all symbols, broadcasts TRADE_CLOSE via WebSocket). `processCandle()` skips v5_signal-sourced positions to avoid cross-symbol price contamination from the legacy BTCUSDT-only `executePaperTrade()` loop in storage.ts. **Neural Position Manager**: `neuralPositionManager()` re-evaluates open positions every V5 cycle — direction flip exit, MFE protection exit, confidence decay tightening, breakeven automation, adaptive trailing (giveback % scales with V5 score). Neural PM propagates SL/TP changes and closes to Bybit when live trading is enabled. **Position Health**: `computePositionHealth()` returns 0-100 health score from weighted factors (P&L, SL/TP distance, model confidence, time, MFE trend).
  - `server/execution-bridge.ts` — Execution Service bridge: caches positions/balance pushed from GPU trainer, tracks connection status (30s timeout), broadcasts EXECUTION_STATE via WebSocket
  - `server/bybit/client.ts` — Bybit V5 REST API client with HMAC-SHA256 auth. Methods: getWalletBalance, getTicker, getPositions, createOrder, amendOrder, cancelOrder, setLeverage, setTradingStop, getKlines, getOrderHistory, getClosedPnl. Base URL: api.bybit.com (mainnet). Supports direct mode and proxy mode (routes through GPU trainer `/bybit-proxy` endpoint when registered).
  - `server/bybit/live-engine.ts` — Live trading execution engine. openLivePosition (market order with SL/TP, signal-strength leverage), closeLivePosition (reduceOnly), amendLiveSLTP (setTradingStop). Config persisted to DB (settings table). Safety: max 6 positions, 1 per symbol, daily loss limit, requires testConnection before enable. formatQty/formatPrice with per-symbol decimal precision.
  - `server/ws.ts` — WebSocket server for real-time event streaming (broadcasts CYCLE_UPDATE on push, LIVE_TRADE_OPEN/CLOSE events)

### Database (PostgreSQL via Drizzle ORM)
Key tables: `v5_signals`, `live_trade_records`, `live_cycle_logs` (with V5 fields: ret_mu, mfe_pred, mae_pred, p_hold, p_long, p_short, v5_score, v5_threshold, v5_side), `paper_positions` (with `source` field: "v5_signal"|"manual"|"auto"), `paper_portfolio`, `paper_trades`, `paper_trade_history` (complete trade records with R metrics per asset), `candles`, `settings`, `training_sessions` (walk-forward session tracking with config, aggregate metrics, ETA), `training_epochs` (per-epoch loss curves, accuracy, sweep metrics), `training_folds` (per-fold results with per-symbol breakdown), `neural_adjustments` (Neural PM actions: breakeven, trail, flip exit, MFE protection, confidence decay — per position with V5 model state at time of adjustment)

Schema in `shared/schema.ts` with Drizzle + Zod validation.

### v5 Neural Network (runs on local GPU)
- Architecture: V5Forecaster (multi-head: ret_dist, mfe, mae, action[HOLD/LONG/SHORT]) with 85 features, 15m timeframe
- Also supports legacy EnhancedMultiHeadMLP (auto-detected from checkpoint `model_type` field)
- Feature version: v5.0.1_forecaster (FeatureEngineer: STF44 + ENH24 + HTF12 + Regime5 = 85 features)
- V5 Composite Scoring Engine (replaced Triple-Lane system): score = p_side × |mu|/risk - λ × (1-p_side) × |mu|/risk, threshold=0.02, min_mu_r=0.03, lambda=0.5
- Side determined by model (edge_long vs edge_short), not HTF alignment
- 6 symbols: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, AVAXUSDT
- Per-symbol edge learning (v5.4.0): per-symbol scalers, thresholds, kill switches
- Walk-forward validation: 21/25 folds profitable, +520R cumulative
- Checkpoint loading: searches best_enter_prauc.pt → best_v5_expectancy.pt → best_enter_loss.pt → best_v5_loss.pt
- Scaler: embedded in checkpoint (scaler_center/scaler_scale) or loaded from per_symbol_scalers.joblib/scaler.joblib
- **Live feature pipeline** (`gpu_trainer/live_runner.py`): `_compute_features_for_symbol()` fetches real funding rate (from Binance FAPI `/fapi/v1/fundingRate`) and open interest (`/futures/data/openInterestHist`) during live inference, matching the training pipeline. Funding cached 30min (TTL), OI cached 15min. Falls back to zeros on fetch failure. One-time `[Feature Check]` diagnostic log per symbol per session.

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
