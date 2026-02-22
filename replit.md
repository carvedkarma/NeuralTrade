# BTC Futures Signal Dashboard

## Overview
This project is an institutional-grade, AI-driven dashboard for generating sophisticated BTCUSDT futures trading signals. It uses machine learning, real-time market data, and sentiment analysis to provide AI-powered trade plans, aiming for continuous learning and adaptation to maximize market potential through AI-driven precision.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### UI/UX Decisions
The frontend uses React, TypeScript, and Vite, with `shadcn/ui` (Radix UI, Tailwind CSS) for components and Recharts for data visualization. It includes a tabbed interface with a Pro Dashboard and an R/$ toggle for financial metrics.

### Technical Implementations
The backend is built with Node.js and Express.js (TypeScript, ESM) providing a RESTful API. It integrates AI via OpenAI and sources market data from Binance Vision API with fallbacks. A WebSocket server enables real-time event streaming. The system features a Triple-Lane Aggression Engine (CORE/FLOW/SCALP) routed by an HTF score, managing per-symbol daily R budgets and lane-specific thresholds. A live learning system supports scheduled retraining and safe model promotion, while a money management system converts R-based metrics to USD.

The core AI model predicts trend-following trade quality using 77 features on a 15-minute timeframe with a 24-bar horizon. It employs an EnhancedMultiHeadMLP architecture with HTF-gated Triple Barrier labeling, optimized for PR-AUC. Inference involves a probability threshold and HTF alignment for LONG/SHORT/HOLD signals, with dynamic position sizing based on ATR, account risk, and confidence. A policy auto-tuner optimizes trade frequency, and a cost model accounts for trading fees. Safety kill-switches include per-symbol daily drawdown caps and a Per-Asset Trade Quota Controller.

The Triple-Lane Aggression Engine routes trades based on HTF score, each with specific entry requirements, sizing multipliers, horizons, and daily R budgets. A Smart Trade Manager dynamically evaluates open positions using priority-based exit rules. The training pipeline supports multi-asset training with time-based splits. The EnhancedMultiHeadMLP includes a `value_head` for E[net R] regression and optional symbol embedding. Bias initialization and post-training temperature scaling calibrate logits, with lane gating adding E[net R] minimums. Promotion gates enforce performance and calibration. Training stability is improved with a three-stage loss schedule. Multi-asset data ingestion includes preflight checks and a HTF Warmup & Candle History mechanism.

Key enhancements include:
- **Directional Separation & Quality Scoring**: Improved labeling and refined TP quality scores.
- **Distributional Trade Forecasting**: Shifted from binary classification to distributional outputs for expected returns and quantile forecasts.
- **Advanced Risk Management**: Incorporated candidate filtering, multi-preset barriers, Kelly-like position sizing, adaptive sizing, regime scaling, daily loss caps, trailing equity stops, and conviction-based sizing.
- **Robust Training Pipeline**: Addressed critical bugs related to time-based data splits, scaling, quality gates, loss scheduling, and side-conditional outcomes.
- **Market Regime Classification**: Implemented an ADX-based regime gate and a multi-regime classifier to adapt to market conditions.
- **Capital Protection**: Added weekly loss cap kill-switches and a drawdown-adaptive throttle.
- **Multi-Asset Support**: Extended to 7 symbols with symbol-balanced sampling and per-symbol reporting.
- **Performance Optimization**: Introduced metric-based checkpoint promotion, hard threshold floors, and temperature scaling for model calibration.
- **Ultra-Conviction Tier**: Allowed for higher risk in rare, high-conviction setups under strict gating conditions.
- **v5.1.0 Edge-First Strategy**: Shifted from volume-maximization (TPD target) to quality-maximization (edge-per-trade). Includes edge-first pre-filtering (`--v5-edge-first`, `--v5-edge-min`, `--v5-edge-pct-floor`, `--v5-edge-topn-per-day`), regime-conditional side filtering (`--v5-regime-side-map`), and size floor clamping (`--v5-size-floor`). Recommended: `--v5-edge-first --v5-edge-min 0.03 --v5-edge-pct-floor 70 --v5-edge-topn-per-day 3 --v5-regime-side-map "trending_up=LONG,trending_down=SHORT,choppy=NONE" --v5-size-floor 0.5`.

### System Design Choices
Data management uses Drizzle ORM for PostgreSQL and Zod for type-safe validation. The system persists learning states and separates live sentiment from historical data. The client is bundled by Vite, and the server by esbuild. Centralized timeframe configuration ensures consistency. A runtime diagnostic system provides health endpoints and UI console logging. A GPU training API supports training, status checks, and daily retraining.

## External Dependencies

### Database
- PostgreSQL

### UI Framework
- Radix UI
- Lucide React
- class-variance-authority

### Data & Validation
- Zod
- drizzle-zod
- date-fns

### AI / Machine Learning
- OpenAI

### Market Data
- Binance Vision API
- CoinGecko
- CryptoCompare