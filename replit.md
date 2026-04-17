# Neural Terminal — AI Trading Dashboard

## Overview
Neural Terminal is an institutional-grade, GPU-accelerated AI trading system for multi-asset crypto futures. It functions as a trading terminal, providing real-time signals, managing paper and live trading operations, and tracking performance. The system aims to deliver a comprehensive solution for AI-driven crypto futures trading, offering advanced analytics and automated trading capabilities.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend (9-Page Trading Terminal)
The frontend is built with React, TypeScript, and Vite, utilizing `shadcn/ui` (Radix UI, Tailwind CSS) for UI components and Recharts for data visualization. The design adheres to a dark navy theme with distinct neon accents.

**Core Pages:**
-   **Command Center:** Live system status, KPIs, market overview, real-time signal feed, active positions, mini equity curve.
-   **Live Trading:** Symbol selector, price charts, market scanner, detailed signal analysis, order flow panel, position management, manual trade entry, signal history.
-   **Paper Trading:** Portfolio metrics, equity curve, open/closed positions, health gauges, neural status, MFE tracker, breakeven indicator, adaptive trail visualization, configuration, and leverage monitoring.
-   **Analytics:** Performance metrics, equity curves, rolling performance, hourly heatmaps, per-symbol breakdowns.
-   **Training Monitor:** Visualizes live GPU training progress, status, model knowledge, loss curves, action accuracy, walk-forward validation.
-   **Bitget Trading:** Live Bitget exchange connection, positions, balance, trading toggle, V5 signals, risk config.
-   **Neural Monitor:** Per-symbol neural intelligence dashboard displaying V5 model outputs in real-time via WebSockets.
-   **World Intel:** Macro Oracle dashboard providing global signals from various sources (RSS feeds, macro indicators, Reddit sentiment), a GPT-4o generated Global Macro Climate Score and narrative, 24h prediction, sentiment scores, and crypto impact explanations. Includes a Risk Calendar.
-   **Signal Dashboard:** Manual leveraged trading page displaying high-confidence V5 SHORT signals in real-time with entry/SL/TP, score, predicted MFE/MAE/R:R, and an integrated Leverage Calculator.
-   **Trade History:** Complete trade history with filtering and export.
-   **Settings:** Manages connections, credentials, account configuration, model information, risk parameters, and data freshness.

### Backend (Node.js + Express + TypeScript)
The backend provides API routes and services to support the frontend and interact with external systems.

**Key Features:**
-   **API Routes:** Manages data retrieval for signals, performance, equity curves, trade history, system status, market data, and cycle logs. Handles POST requests for cycle logs and executed trades.
-   **GPU Trainer Bridge:** Facilitates communication with the local GPU trainer.
-   **Live Candle Sync:** Synchronizes real-time 15-minute candle data.
-   **Paper Trading Engine:** Simulates trades, monitors positions, manages SL/TP, and applies neural position management strategies.
-   **Execution Service Bridge:** Caches execution state and monitors connection status.
-   **Bitget & Bybit Client & Live Engine:** Interfaces with respective exchange APIs for live trading operations.
-   **Market Regime / Chop Protection:** Computes ADX, Chop Index, and Bollinger Band Width to apply trade gating based on market conditions (HARD_CHOP, SOFT_CHOP, TRENDING).
-   **Order Flow Pipeline:** Fetches real-time order book, trades, and ticker data to compute OB imbalance, aggressor ratio, CVD, liquidation proximity, and a composite score for trade gating.
-   **WebSocket Server:** Enables real-time event streaming for continuous updates.

### Symbol Configuration
All 20 trading symbols (e.g., BTCUSDT, ETHUSDT) are defined in `shared/symbols.ts` with their respective precision settings.

### Database (PostgreSQL via Drizzle ORM)
PostgreSQL with Drizzle ORM is used for data persistence across various trading and training tables.

### v5 Neural Network
A v5 neural network, `V5Forecaster`, runs on a local GPU.
-   **Architecture:** Multi-head output (return distribution, MFE, MAE, action probabilities) based on 95 features from a 15-minute timeframe.
-   **Capabilities:** Supports trading for 20 symbols, uses a Composite Scoring Engine, incorporates per-symbol edge learning, and integrates real-time funding rates, open interest, and long/short ratio data.
-   **Enhancements:** Includes various improvements for training stability, prediction accuracy, risk management, and sizing adjustments, such as side balance fixes, sigma discount, conviction gates, adaptive loss functions, dynamic caps, and quality gates.
-   **Specialist Fixes:** Addressed structural bugs in the LONG/SHORT specialist training pipeline by improving gradient flow, adjusting entropy regularization, recalibrating KL targets, and implementing intelligent oversampling.

### v11 Green-Field Brain (research, off-dashboard)
A separate `gpu_trainer_v11/` package contains a complete green-field redesign after V5/V6/V10 Phase 1 all proved no edge: dollar bars, fractionally-differentiated returns, two specialist primary rules (LONG momentum-after-vol-contraction; SHORT mean-reversion-after-vol-expansion), horizon-conditional triple barriers, sample-uniqueness weighting, transfer-entropy causal feature selection, a 4-layer causal Transformer meta-classifier (d=128, 4 heads, seq=128) pretrained with masked-feature reconstruction and finetuned in a bagged ensemble (N=5), Mondrian conformal calibration per ATR-pct regime bucket, honest 6-fold walk-forward with adversarial-validation drift detection, per-symbol diversification probe, and a verdict writer. Anti-tuning policy is locked in `gpu_trainer_v11/README.md`: no hyperparameter changes after pre-flight; ship only on PF≥1.3 with ≥500 trades/fold. Independent of the production dashboard, paper engine, and exchange clients.

### v6 Neural Network (V6Forecaster)
The next-generation `V6Forecaster` offers advanced capabilities while maintaining output compatibility with V5.
-   **Architecture:** Causal Conv1D, Positional Encoding, Transformer Blocks, Mixture-of-Experts trunk (4 experts), and 6 output heads.
-   **New Features:** Incorporates temporal context, MoE routing with collapse recovery, feature masking, auxiliary self-supervised loss for next-bar feature prediction, and a confidence calibration head for signal gating.
-   **Training & Inference:** Supports V6-specific arguments for architecture and loss components, balanced sampling, and automatic model loading with sequential feature computation and confidence-based signal gating.

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