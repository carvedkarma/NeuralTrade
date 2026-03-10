# Neural Terminal — AI Trading Dashboard

## Overview
Neural Terminal is an institutional-grade, GPU-accelerated AI trading system for multi-asset crypto futures. It functions as a trading terminal, providing real-time signals, managing paper and live trading operations, and tracking performance. The system leverages a v5 neural network running on a local GPU trainer, with this web application serving as the interface. The project aims to deliver a comprehensive solution for AI-driven crypto futures trading, offering advanced analytics and automated trading capabilities.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend (6-Page Trading Terminal)
The frontend is built with React, TypeScript, and Vite, utilizing `shadcn/ui` (Radix UI, Tailwind CSS) for UI components and Recharts for data visualization. The design adheres to a dark navy theme with distinct neon accents.

**Core Pages:**
-   **Command Center:** Live system status, KPIs, market overview, real-time signal feed, active positions, mini equity curve.
-   **Live Trading:** Symbol selector, price charts, market scanner, detailed signal analysis, position management, manual trade entry, signal history.
-   **Paper Trading:** Portfolio metrics, equity curve, open/closed positions, health gauges, neural status, MFE tracker, breakeven indicator, adaptive trail visualization, configuration.
-   **Analytics:** Performance metrics, equity curves, rolling performance, hourly heatmaps, per-symbol breakdowns, advanced analytics.
-   **Training Monitor:** Visualizes live GPU training progress, status, model knowledge, loss curves, action accuracy, walk-forward validation.
-   **Settings:** Manages GPU connection, account configuration, model information, risk parameters, data freshness.

### Backend (Node.js + Express + TypeScript)
The backend provides API routes and services to support the frontend and interact with external systems.

**Key Features:**
-   **API Routes:** Manages data retrieval for signals, performance, equity curves, trade history, system status, market data, and cycle logs. Handles POST requests for cycle logs and executed trades, enabling auto-trading.
-   **GPU Trainer Bridge:** Facilitates communication with the local GPU trainer, including push-based activity detection and auto-registration.
-   **Live Candle Sync:** Synchronizes real-time 15-minute candle data.
-   **Paper Trading Engine:** Simulates trades, monitors positions, manages SL/TP, and applies neural position management strategies (direction flip exit, MFE protection, confidence decay tightening, breakeven automation, adaptive trailing).
-   **Execution Service Bridge:** Caches execution state and monitors connection status from the GPU trainer's execution service.
-   **Bybit Client & Live Engine:** Interfaces with the Bybit V5 REST API for live trading operations.
-   **WebSocket Server:** Enables real-time event streaming for continuous updates.

### Symbol Configuration
All 20 trading symbols are defined in `shared/symbols.ts`, including `TRADING_SYMBOLS`, `QTY_PRECISION`, and `PRICE_PRECISION`.

**20 Symbols:** BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, AVAXUSDT, ADAUSDT, DOGEUSDT, LINKUSDT, LTCUSDT, NEARUSDT, PEPEUSDT, SUIUSDT, AAVEUSDT, ARBUSDT, DOTUSDT, MATICUSDT, FILUSDT, APTUSDT, OPUSDT.

### Database (PostgreSQL via Drizzle ORM)
PostgreSQL with Drizzle ORM is used for data persistence.

**Key Tables:** `v5_signals`, `live_trade_records`, `live_cycle_logs`, `paper_positions`, `paper_portfolio`, `paper_trades`, `paper_trade_history`, `candles`, `settings`, `training_sessions`, `training_epochs`, `training_folds`, `neural_adjustments`.

### v5 Neural Network
A v5 neural network, `V5Forecaster`, runs on a local GPU.
-   **Architecture:** Multi-head output (return distribution, MFE, MAE, action probabilities) based on 85 features from a 15-minute timeframe.
-   **Symbols:** Supports trading for all 20 defined symbols.
-   **Composite Scoring:** Employs a V5 Composite Scoring Engine.
-   **Per-symbol Edge Learning:** Incorporates symbol-specific scalers, thresholds, and kill switches.
-   **Live Feature Pipeline:** Integrates real-time funding rates and open interest data during live inference.
-   **Improvements:** Side balance fix for biased training labels, sigma discount for penalizing uncertain predictions, minimum conviction gate, larger MFE/MAE heads, asymmetric MAE loss, per-symbol cooldowns, per-symbol edge topN, EMA200 auto-skip with multi-regime, correlation max-block cap, prediction quality diagnostics, v5.5 soft gates (regime soft sizing, graduated symbol kill, edge topn decay, gate impact diagnostics), dead-fold diagnostics (cause logging when 0 trades), threshold EMA decay on dead folds (`--v5-wf-threshold-decay`, default 0.5), min_trades soft floor (trades kept as LOW_CONF instead of wiped, threshold EMA skips low-confidence folds), cooldown CLI threading (`--cooldown` now properly passed through all layers), sizing stacking fix (position sizers use max-of-modifiers instead of multiplicative stacking to avoid pinning all trades to size_floor), per-symbol threshold cap (no-edge symbols get HIGH_BAR = global_threshold×3 instead of inf, allowing recovery when market conditions change), side-aware scoring (`--v5-side-aware-scoring`: shorts require mu_R<0, longs require mu_R>0 — both heads must agree), per-side short confidence gate (`--v5-min-p-short`: separate minimum p_short threshold for short trades), per-side quality diagnostics (forward test reports head agreement %, avg mu_R, avg p_side per direction, and short head-disagree trade expectancy), v5.6 soft gate floor (`--v5-soft-gate-floor`: clamps soft gate multiplier minimum to size_floor so position sizers can amplify above floor instead of all trades pinned at 0.5x), dynamic weekly cap (`--v5-weekly-cap-dynamic`: scales weekly loss cap based on rolling 4-week performance — widens when profitable, tightens when losing), rolling quality gate (`--v5-quality-gate`: tracks rolling action accuracy and win rate over last N trades, reduces sizing to 0.25x when accuracy drops below 30% and WR below 35%, to 0.1x when accuracy below 20%), and direction balance cap (`--v5-direction-balance-cap`: reduces sizing on the dominant direction when one side exceeds 75% of recent candidates, 0.5x at 75% imbalance, 0.25x at 85%).

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