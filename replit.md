# BTC Futures Signal Dashboard

## Overview
This project is an institutional-grade, AI-driven dashboard for generating sophisticated BTCUSDT futures trading signals. It leverages machine learning, real-time market data, and sentiment analysis to provide AI-powered trade plans, aiming for continuous learning and adaptation to maximize market potential through AI-driven precision.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### UI/UX Decisions
The frontend is built with React, TypeScript, and Vite, utilizing `shadcn/ui` (Radix UI, Tailwind CSS) for components and Recharts for data visualization. Navigation is managed via a tabbed interface, including a Pro Dashboard with real-time analytics across six tabs. An R/$ toggle allows switching between R-units and USD for financial metrics.

### Technical Implementations
The backend uses Node.js and Express.js (TypeScript, ESM) with a RESTful API. AI integration is through OpenAI. Market data is sourced from Binance Vision API, with fallbacks to CoinGecko and CryptoCompare, supported by a Replit-hosted data proxy. A WebSocket server handles real-time event streaming for the Pro Dashboard. The system incorporates a Triple-Lane Aggression Engine (CORE/FLOW/SCALP) routed by HTF score, per-symbol daily R budgets, and lane-specific thresholds. A live learning system supports scheduled retraining and safe model promotion, including per-symbol model management and stringent promotion gates. A money management system converts R-based metrics to USD values.

### Feature Specifications
The core AI model predicts the quality of trend-following trades based on 63 features on a 15-minute timeframe with a 24-bar horizon. Training uses an EnhancedMultiHeadMLP architecture with HTF-gated Triple Barrier labeling, optimizing for PR-AUC. Inference applies a probability threshold and HTF alignment for LONG/SHORT/HOLD signals, with position sizing based on ATR, dynamic account risk, and confidence. A policy auto-tuner optimizes trade frequency, and a sophisticated cost model accounts for fees, spread, and slippage. Safety kill-switches, including per-symbol daily drawdown caps, are implemented. A Per-Asset Trade Quota Controller dynamically adjusts FLOW thresholds for under-trading symbols.

The Triple-Lane Aggression Engine routes trades (CORE > FLOW > SCALP > HOLD) based on HTF score, each with specific entry requirements, sizing multipliers, horizons, and daily R budgets. SCALP trades have advanced volatility/momentum gates and specific TP/SL. Execution modes (signal_only, paper, live) are controlled via CLI flags.

A Smart Trade Manager dynamically evaluates open positions with priority-based exit rules: TIME_EXIT, ADVERSE_FLIP, STALL_TAKEPROFIT, TRAIL_SL, BREAKEVEN.

The v5.0 training pipeline supports multi-asset training with per-symbol time-based splits. The EnhancedMultiHeadMLP includes a value_head for E[net R] regression and optional symbol_embedding. Bias initialization and post-training temperature scaling calibrate logits, with lane gating adding E[net R] minimums. Promotion gates enforce performance and calibration. Training stability improvements include a three-stage loss schedule to prevent classifier collapse. P_enter percentiles are audited for validity during evaluation. A `--verify-enter-metrics` mode validates model performance after training.

Multi-asset data ingestion supports per-symbol downloads, with a preflight check for data sufficiency and options to download missing data or allow partial data. A HTF Warmup & Candle History mechanism ensures sufficient historical data for indicator computation.

The v4.6 "Directional Separation" introduces bidirectional triple-barrier labeling (LONG/SHORT outcomes, direction, confidence, HTF alignment). The model includes `dir_head` and `htf_head` with configurable composite loss weights. The v4.7.1 "TP Quality Score Balancing" refines labeling with a continuous TP quality score, auto-balancing `q_min_tp` to hit a target `enter_rate`.

The v4.9.0 "Distributional Trade Forecaster" replaces binary classification with distributional outputs: E[R], quantile forecasts (q10/q50/q90), p(R>0), and optional regime classification. Trade selection uses a score-based ranking (p_win * E_R - lambda * max(0, -q10)). Validation uses a percentile-based sweep. The v4.9.1 "Enhanced Distributional Trade Forecaster" adds candidate filtering, multi-preset barriers, a money-score formula, risk controls (daily loss limit, max concurrent trades, per-symbol exposure cap), Kelly-like position sizing, and multi-horizon support.

The v5.0 "Forecaster" separates market forecasting from the decision layer. The v5.0.1 update fixes three critical issues: (1) ALL continuous targets (ret_R, mfe_R, mae_R) are now in R-units (price_change/ATR) for unit consistency — no more log-return/R-unit mixing in scores; (2) Adaptive deadzone targets ~30% HOLD rate via `--v5-hold-target` with class-balanced CE loss (inverse frequency weighting) to prevent HOLD collapse; (3) Candidate auto-relax stepwise lowers ATR threshold, disables chop filter, then disables trigger gating to reach `--cand-min-rate` (default 0.25). Score formula: edge = p_dir * mu_R/(mae_R + eps) - lambda*max(0, -mu_R). Candidate warmup disables candidate mask for first `--v5-cand-warmup` epochs (default 3). It uses a 3-class `action_head` (HOLD, LONG, SHORT) and optional `barrier_head` for preset selection. The V5Forecaster model uses a ResidualBlock trunk with specialized heads. Target generation ensures no data leakage by computing targets only from future bars.

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