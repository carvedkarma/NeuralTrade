# Neural Terminal — AI Trading Dashboard

## Overview
Neural Terminal is an institutional-grade, GPU-accelerated AI trading system designed for multi-asset crypto futures. Its primary purpose is to serve as a trading terminal, providing real-time signals, managing paper and live trading operations, and tracking performance. The system utilizes a v5 neural network that runs on a local GPU trainer, with this web application acting as the interface. The project aims to deliver a comprehensive and robust solution for AI-driven crypto futures trading, offering advanced analytics and automated trading capabilities.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend (6-Page Trading Terminal)
The frontend is built with React, TypeScript, and Vite, utilizing `shadcn/ui` (Radix UI, Tailwind CSS) for UI components and Recharts for data visualization. The design adheres to a dark navy theme with distinct neon accents.

**Core Pages:**
-   `/` — Command Center: Displays live system status, key performance indicators, market overview, real-time signal feed with V5 model outputs, active positions, and a mini equity curve.
-   `/live` — Live Trading: Features a symbol selector, price charts, market scanner, detailed signal analysis, position management tools (close, partial-close, SL/TP edits), manual trade entry, and signal history.
-   `/paper` — Paper Trading: Offers portfolio metrics, an equity curve, open/closed positions with advanced management, position health gauges, neural status badges, MFE tracker, breakeven indicator, adaptive trail visualization, and configuration options.
-   `/analytics` — Analytics: Provides comprehensive performance metrics, equity curves, rolling performance data, hourly heatmaps, per-symbol breakdowns, and advanced analytics such as MFE vs. Result scatter plots, capture ratio, and optimal exit simulation.
-   `/training` — Training Monitor: Visualizes live GPU training progress with status banners, overview cards, model knowledge gauges, live loss curves, action accuracy trends, and walk-forward validation results.
-   `/settings` — Settings: Manages GPU connection status, account configuration, model information, risk parameters, and data freshness settings.

**Design System:**
The system employs a deep navy background with primary neon green for positive indicators, red for losses, cyan for information, and amber for warnings. Custom CSS classes enhance visual elements with glassmorphism, glow effects, and animations.

### Backend (Node.js + Express + TypeScript)
The backend provides a comprehensive set of API routes and services to support the frontend and interact with external systems.

**Key Features:**
-   **API Routes:** Manages data retrieval for v5 signals, performance statistics, equity curves, trade history, system status, market data (prices, candles), and cycle logs. It also handles POST requests for receiving cycle logs and executed trade records from the GPU trainer, enabling auto-trading (both paper and live) with signal-strength leverage. Specific endpoints are dedicated to paper trading operations (managing positions, risk alerts, manual trades) and integrations with external services like Bybit.
-   **GPU Trainer Bridge:** Facilitates communication with the local GPU trainer, including push-based detection of activity and auto-registration of the GPU trainer's callback URL.
-   **Live Candle Sync:** Synchronizes real-time 15-minute candle data from external sources.
-   **Paper Trading Engine:** A robust module for simulating trades, monitoring positions, managing SL/TP, and applying neural position management strategies (direction flip exit, MFE protection, confidence decay tightening, breakeven automation, adaptive trailing). It also computes position health scores based on various factors.
-   **Execution Service Bridge:** Caches execution state (positions, balance) pushed from the GPU trainer's execution service and monitors its connection status.
-   **Bybit Client & Live Engine:** Provides an interface for interacting with the Bybit V5 REST API for live trading operations, including opening/closing positions, amending SL/TP, and managing account configurations. It supports both direct and proxy modes for API calls.
-   **WebSocket Server:** Enables real-time event streaming for continuous updates on cycles, trades, and execution states.

### Database (PostgreSQL via Drizzle ORM)
The system leverages PostgreSQL with Drizzle ORM for data persistence.

**Key Tables:**
-   `v5_signals`: Stores processed signals from the v5 model.
-   `live_trade_records`, `live_cycle_logs`: Records of live trades and detailed cycle log data, including V5 model outputs (ret_mu, mfe_pred, mae_pred, p_hold, p_long, p_short, v5_score, v5_threshold, v5_side).
-   `paper_positions`, `paper_portfolio`, `paper_trades`, `paper_trade_history`: Comprehensive data for paper trading, including position details, portfolio metrics, and historical trade records with R metrics.
-   `candles`: Stores market candle data.
-   `settings`: General application settings.
-   `training_sessions`, `training_epochs`, `training_folds`: Tracks GPU training progress, session configurations, epoch-level metrics, and fold-specific results.
-   `neural_adjustments`: Records actions taken by the Neural Position Manager for specific positions.

### v5 Neural Network
The core trading intelligence is provided by a v5 neural network, `V5Forecaster`, running on a local GPU.
-   **Architecture:** Multi-head output (return distribution, MFE, MAE, action probabilities) based on 85 features from a 15-minute timeframe.
-   **Symbols:** Supports trading for BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, AVAXUSDT.
-   **Composite Scoring:** Employs a V5 Composite Scoring Engine for signal evaluation.
-   **Per-symbol Edge Learning:** Incorporates symbol-specific scalers, thresholds, and kill switches for refined trading.
-   **Live Feature Pipeline:** Integrates real-time funding rates and open interest data during live inference to ensure feature consistency with training.

## External Dependencies

### Database
-   PostgreSQL (managed via Drizzle ORM)

### UI Framework
-   Radix UI, Lucide React, class-variance-authority, Recharts

### Data & Validation
-   Zod, drizzle-zod, date-fns

### AI / Machine Learning
-   OpenAI (for advanced AI analysis features)

### Market Data
-   Binance Vision API (for real-time market data)
-   Bybit V5 REST API (for live trading execution)