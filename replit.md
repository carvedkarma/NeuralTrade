# Neural Terminal — AI Trading Dashboard

## Overview
Neural Terminal is an institutional-grade, GPU-accelerated AI trading system for multi-asset crypto futures. It functions as a trading terminal, providing real-time signals, managing paper and live trading operations, and tracking performance. The system leverages a v5 neural network running on a local GPU trainer, with this web application serving as the interface. The project aims to deliver a comprehensive solution for AI-driven crypto futures trading, offering advanced analytics and automated trading capabilities.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend (9-Page Trading Terminal)
The frontend is built with React, TypeScript, and Vite, utilizing `shadcn/ui` (Radix UI, Tailwind CSS) for UI components and Recharts for data visualization. The design adheres to a dark navy theme with distinct neon accents.

**Core Pages:**
-   **Command Center:** Live system status, KPIs, market overview, real-time signal feed, active positions, mini equity curve.
-   **Live Trading:** Symbol selector, price charts, market scanner, detailed signal analysis, order flow panel (OB imbalance, aggressor ratio, CVD, composite score), position management, manual trade entry, signal history.
-   **Paper Trading:** Portfolio metrics, equity curve, open/closed positions, health gauges, neural status, MFE tracker, breakeven indicator, adaptive trail visualization, configuration. **Leverage Monitor row** (Avg Leverage Closed, Peak Leverage, Open Avg Leverage, Effective Exposure) powered by `/api/paper/leverage-stats`.
-   **Analytics:** Performance metrics, equity curves, rolling performance, hourly heatmaps, per-symbol breakdowns, advanced analytics.
-   **Training Monitor:** Visualizes live GPU training progress, status, model knowledge, loss curves, action accuracy, walk-forward validation.
-   **Bitget Trading:** Live Bitget exchange connection, positions table, balance overview, trading toggle, recent V5 signals, risk config display.
-   **Neural Monitor:** Per-symbol neural intelligence dashboard. Shows V5 model outputs (p_long/p_hold/p_short gauges, expected return, MFE/MAE predicted risk-reward, HTF trend alignment, V5 score vs threshold, decision status) for all 20 symbols in real-time. Live WebSocket updates. Market bias summary bar. Route: `/neural`.
-   **Trade History:** Complete trade history with filtering and export.
-   **Settings:** Manages GPU connection, Bybit connection, Bitget credentials & config, account configuration, model information, risk parameters, data freshness.

### Backend (Node.js + Express + TypeScript)
The backend provides API routes and services to support the frontend and interact with external systems.

**Key Features:**
-   **API Routes:** Manages data retrieval for signals, performance, equity curves, trade history, system status, market data, and cycle logs. Handles POST requests for cycle logs and executed trades, enabling auto-trading.
-   **GPU Trainer Bridge:** Facilitates communication with the local GPU trainer, including push-based activity detection and auto-registration.
-   **Live Candle Sync:** Synchronizes real-time 15-minute candle data.
-   **Paper Trading Engine:** Simulates trades, monitors positions, manages SL/TP, and applies neural position management strategies (direction flip exit, MFE protection, confidence decay tightening, breakeven automation, adaptive trailing).
-   **Execution Service Bridge:** Caches execution state and monitors connection status from the GPU trainer's execution service.
-   **Bitget Client & Live Engine:** Interfaces with the Bitget V2 REST API for live trading operations (HMAC-SHA256 auth, credentials stored in DB settings table). Auto-trade signals route through Bitget when enabled (takes priority over Bybit).
-   **Bybit Client & Live Engine:** Interfaces with the Bybit V5 REST API for live trading operations.
-   **Market Regime / Chop Protection:** `server/market-regime.ts` computes ADX (14-period), Chop Index, and Bollinger Band Width from DB candles. Three tiers: HARD_CHOP (ADX<15, signal blocked), SOFT_CHOP (ADX 15-25, threshold raised to 0.62 + leverage cut to 0.4x), TRENDING (ADX>25, no changes). Applied as Gate 3 in auto-trade ingestion. `GET /api/market/regime` returns all 20 symbols' regime state.
-   **Order Flow Pipeline (Gate 4):** `server/order-flow.ts` fetches real-time orderbook, recent trades, and ticker data from Bybit V5 public API. Computes OB imbalance, buy/sell aggressor ratio, CVD (cumulative volume delta) with slope, liquidation proximity, and a weighted composite score (40% OB + 35% aggressor + 25% CVD normalized by total traded volume). Applied as Gate 4 in both auto-trade ingestion (routes.ts) and paper trading engine (paper/engine.ts) after chop protection. LONG blocked when OB imbalance<0.35 AND aggressor<0.40, or CVD falling AND composite<-0.3. SHORT blocked when OB imbalance>0.65 AND aggressor>0.60, or CVD rising AND composite>0.3. Falls back gracefully (allows trade) when Bybit public API is unreachable. `GET /api/market/order-flow` and `GET /api/orderflow/:symbol` return per-symbol snapshots (60s TTL cache). 6 new columns on `v5_signals` table (`ob_imbalance`, `aggressor_ratio`, `cvd_at_signal`, `liq_proximity`, `of_gate_passed`, `of_gate_reason`) are enriched at auto-trade time by matching the most recent signal within a 5-minute window.
-   **WebSocket Server:** Enables real-time event streaming for continuous updates.

### Symbol Configuration
All 20 trading symbols are defined in `shared/symbols.ts`, including `TRADING_SYMBOLS`, `QTY_PRECISION`, and `PRICE_PRECISION`.

**20 Symbols:** BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, AVAXUSDT, ADAUSDT, DOGEUSDT, LINKUSDT, LTCUSDT, NEARUSDT, PEPEUSDT, SUIUSDT, AAVEUSDT, ARBUSDT, DOTUSDT, MATICUSDT, FILUSDT, APTUSDT, OPUSDT.

### Database (PostgreSQL via Drizzle ORM)
PostgreSQL with Drizzle ORM is used for data persistence.

**Key Tables:** `v5_signals`, `live_trade_records`, `live_cycle_logs`, `paper_positions`, `paper_portfolio`, `paper_trades`, `paper_trade_history`, `candles`, `settings`, `training_sessions`, `training_epochs`, `training_folds`, `neural_adjustments`.

### v5 Neural Network
A v5 neural network, `V5Forecaster`, runs on a local GPU.
-   **Architecture:** Multi-head output (return distribution, MFE, MAE, action probabilities) based on 95 features from a 15-minute timeframe (85 base + 3 funding rate + 3 OI + 4 L/S ratio).
-   **Symbols:** Supports trading for all 20 defined symbols.
-   **Composite Scoring:** Employs a V5 Composite Scoring Engine.
-   **Per-symbol Edge Learning:** Incorporates symbol-specific scalers, thresholds, and kill switches.
-   **Live Feature Pipeline:** Integrates real-time funding rates, open interest, and long/short ratio data during live inference.
-   **Improvements:** Side balance fix for biased training labels, sigma discount for penalizing uncertain predictions, minimum conviction gate, larger MFE/MAE heads, asymmetric MAE loss, per-symbol cooldowns, per-symbol edge topN, EMA200 auto-skip with multi-regime, correlation max-block cap, prediction quality diagnostics, v5.5 soft gates (regime soft sizing, graduated symbol kill, edge topn decay, gate impact diagnostics), dead-fold diagnostics (cause logging when 0 trades), threshold EMA decay on dead folds (`--v5-wf-threshold-decay`, default 0.5), min_trades soft floor (trades kept as LOW_CONF instead of wiped, threshold EMA skips low-confidence folds), cooldown CLI threading (`--cooldown` now properly passed through all layers), sizing stacking fix (position sizers use max-of-modifiers instead of multiplicative stacking to avoid pinning all trades to size_floor), per-symbol threshold cap (no-edge symbols get HIGH_BAR = global_threshold×3 instead of inf, allowing recovery when market conditions change), side-aware scoring (`--v5-side-aware-scoring`: shorts require mu_R<0, longs require mu_R>0 — both heads must agree), per-side short confidence gate (`--v5-min-p-short`: separate minimum p_short threshold for short trades), per-side quality diagnostics (forward test reports head agreement %, avg mu_R, avg p_side per direction, and short head-disagree trade expectancy), v5.6 soft gate floor (`--v5-soft-gate-floor`: clamps soft gate multiplier minimum to size_floor so position sizers can amplify above floor instead of all trades pinned at 0.5x), dynamic weekly cap (`--v5-weekly-cap-dynamic`: scales weekly loss cap based on rolling 4-week performance — widens when profitable, tightens when losing), rolling quality gate (`--v5-quality-gate`: tracks rolling action accuracy and win rate over last N trades, reduces sizing to 0.25x when accuracy drops below 30% and WR below 35%, to 0.1x when accuracy below 20%), direction balance cap (`--v5-direction-balance-cap`: reduces sizing on the dominant direction when one side exceeds 75% of recent candidates, 0.5x at 75% imbalance, 0.25x at 85%), v5.7 recency-weighted training (`--v5-recency-weight`, `--v5-recency-half-life`: exponential time-decay weighting so recent samples get higher loss weight, configurable half-life in days), walk-forward warm-start (`--v5-wf-warm-start`: conditional warm-start — carries previous fold model weights as initialization for next fold only when the previous fold was profitable; resets to random init after negative or dead folds to prevent loss cascading), fine-tuning phase (`--v5-finetune-months`, `--v5-finetune-epochs`, `--v5-finetune-lr-mult`: after main training, fine-tunes on last N months of training window with reduced LR), and long/short ratio features (4 features: ls_ratio, ls_deviation, ls_extreme, crowd_sentiment — fetched from Binance globalLongShortAccountRatio API, wired into both training and live inference pipelines).

### v6 Neural Network (V6Forecaster)
The next-generation `V6Forecaster` offers advanced capabilities while maintaining the same output interface as V5.
-   **Architecture:** Causal Conv1D, Positional Encoding, Transformer Blocks, Mixture-of-Experts trunk (4 experts), and 6 output heads.
-   **New capabilities:** Temporal context (16 bars of history), MoE routing with collapse recovery mechanisms, feature masking, auxiliary self-supervised loss for next-bar feature prediction, and a confidence calibration head for signal gating.
-   **Training:** Supports V6-specific arguments for architecture and loss components, including MoE balance, aux next-bar MSE, and confidence calibration BCE. Balanced sampling uses inverse-frequency loss weighting.
-   **Live Inference:** Automatically loads V6 models, computes sequential features, and uses confidence output to gate signals. Supports flexible scaler loading.

## External Dependencies

### Database
-   PostgreSQL (via Drizzle ORM)

### UI Framework
-   Radix UI, Lucide React, class-variance-authority, Recharts

### Data & Validation
-   Zod, drizzle-zod, date-fns

### AI / Machine Learning
-   OpenAI

### Market Data
-   Binance Vision API
-   Bybit V5 REST API