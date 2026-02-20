# BTC Futures Signal Dashboard

## Overview
This project is an institutional-grade, AI-driven dashboard designed to generate sophisticated BTCUSDT futures trading signals. It leverages machine learning, real-time market data, and sentiment analysis to provide AI-powered trade plans, aiming for continuous learning and adaptation to maximize market potential through AI-driven precision.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### UI/UX Decisions
The frontend is built with React, TypeScript, and Vite, utilizing `shadcn/ui` (Radix UI, Tailwind CSS) for components and Recharts for data visualization. It features a tabbed interface, including a Pro Dashboard with real-time analytics and an R/$ toggle for financial metrics.

### Technical Implementations
The backend uses Node.js and Express.js (TypeScript, ESM) with a RESTful API. AI integration is through OpenAI, and market data is sourced from Binance Vision API with fallbacks. A WebSocket server provides real-time event streaming. The system incorporates a Triple-Lane Aggression Engine (CORE/FLOW/SCALP) routed by HTF score, per-symbol daily R budgets, and lane-specific thresholds. A live learning system supports scheduled retraining and safe model promotion. A money management system converts R-based metrics to USD values.

The core AI model predicts trend-following trade quality using 63 features on a 15-minute timeframe with a 24-bar horizon. It employs an EnhancedMultiHeadMLP architecture with HTF-gated Triple Barrier labeling, optimizing for PR-AUC. Inference applies a probability threshold and HTF alignment for LONG/SHORT/HOLD signals, with dynamic position sizing based on ATR, account risk, and confidence. A policy auto-tuner optimizes trade frequency, and a cost model accounts for fees, spread, and slippage. Safety kill-switches, including per-symbol daily drawdown caps and a Per-Asset Trade Quota Controller, are implemented.

The Triple-Lane Aggression Engine routes trades (CORE > FLOW > SCALP > HOLD) based on HTF score, each with specific entry requirements, sizing multipliers, horizons, and daily R budgets. A Smart Trade Manager dynamically evaluates open positions with priority-based exit rules.

The training pipeline supports multi-asset training with per-symbol time-based splits. The EnhancedMultiHeadMLP includes a value_head for E[net R] regression and optional symbol_embedding. Bias initialization and post-training temperature scaling calibrate logits, with lane gating adding E[net R] minimums. Promotion gates enforce performance and calibration. Training stability is improved with a three-stage loss schedule. Multi-asset data ingestion includes preflight checks for data sufficiency and a HTF Warmup & Candle History mechanism.

The model evolved through several versions:
- **v4.6 "Directional Separation"**: Introduced bidirectional triple-barrier labeling and included `dir_head` and `htf_head`.
- **v4.7.1 "TP Quality Score Balancing"**: Refined labeling with a continuous TP quality score.
- **v4.9.0 "Distributional Trade Forecaster"**: Replaced binary classification with distributional outputs (E[R], quantile forecasts, p(R>0)).
- **v4.9.1 "Enhanced Distributional Trade Forecaster"**: Added candidate filtering, multi-preset barriers, a money-score formula, risk controls, Kelly-like position sizing, and multi-horizon support.
- **v5.0 "Forecaster"**: Separated market forecasting from the decision layer.
- **v5.0.1 Update**: Standardized continuous targets to R-units, introduced adaptive deadzone targets for HOLD rate, and implemented candidate auto-relax.
- **v5.0.2 "Pipeline Fix"**: Addressed 10 critical bugs including time-based data splits, robust scaling, data-adaptive quality gates, and improved loss scheduling.
- **v5.0.3 "Time-Based Split & Forward Test"**: Added proper out-of-sample validation through optional time-based train/test splits, forward testing with frozen decision layers, and walk-forward analysis.
- **v5.0.4 "Side-Conditional Outcomes"**: Eliminated oracle best-side bias by using side-conditional outcome selection and fixed Sharpe annualization.
- **v5.0.5 "Trade Frequency Control & Side Diagnostics"**: Enhanced trade frequency management with CLI flags and added comprehensive side diagnostics and an optional EMA200 regime gate.
- **v5.0.6 "Zero-SHORT Fix & Side-Specific Targets"**: Addressed model bias by fixing the zero-SHORT bug, introducing side-specific MFE/MAE targets, and adding an R/R ratio scoring bonus.
- **v5.0.7 "Capital Protection Layers"**: Implemented a regime gate in training sweep, a weekly loss cap kill-switch, and a warmup skip guard to enhance capital protection.
- **v5.0.8 "Cross-Asset Correlation Tracking & Smart Blocking"**: Added correlation-aware trade gating for multi-asset runs using a RollingDailyCorr tracker and CorrBlocker.
- **v5.0.8+ "Adaptive Sizing & Risk Management"**: Added AdaptivePositionSizer (Kelly-criterion based, fractional Kelly with confidence boost), RegimeScaler (ATR ratio + EMA200 alignment + rolling Sharpe → risk multiplier), DailyLossTracker (daily loss cap + per-symbol R budget), and TrailingEquityStop (drawdown-based pause with recovery). CLI flags: `--v5-adaptive-sizing`, `--v5-regime-scaling`, `--v5-daily-loss-cap`, `--v5-trailing-equity-stop`, `--v5-per-symbol-daily-r`, `--v5-min-threshold`, `--v5-max-trades-per-day`. All features opt-in, integrated into forward test loop with sizing diagnostics in fold reports.
- **v5.0.8+ RegimeScaler Hardening**: Fixed 4 root causes of regime instability: (1) replaced zero-padded ATR convolution with proper NaN-gated rolling mean, (2) normalized equity signal from [-1,1] to [-0.5,0.5] matching ATR/EMA scales, (3) added low-confidence dampening when <2 signals active (configurable `low_confidence_dampen`, default 0.5), (4) raised equity minimum trades threshold from 5 to 15 (configurable `min_equity_trades`). Added `--v5-min-threshold` (score floor) and `--v5-max-trades-per-day` (daily trade cap) CLI flags.
- **v5.0.8+ Adaptive Percentile Threshold**: Added `--v5-min-threshold-pct` for adaptive score filtering. Computes the Nth percentile of valid test scores as a threshold floor, adapting to each fold's score distribution. Solves the problem where fixed `--v5-min-threshold` kills all trades when model scores are uniformly low. Effective threshold = max(calibrated, fixed_floor, percentile_floor). Enhanced logging shows all three floors and which one was applied.
- **v5.0.8+ Trailing Stop Exit System**: Added 3-phase trailing stop-loss via `--v5-trailing-sl`. Phase 1: fixed SL. Phase 2: after `--v5-trail-activation` ATR multiples of favorable move, SL moves to breakeven then trails at `--v5-trail-distance` ATR behind best price. Phase 3 (optional): `--v5-allow-runner` keeps trades open past TP with tighter trail (0.5x distance). New outcomes: TRAIL_WIN (positive trailing exit) and TRAIL_BE (breakeven trailing exit). Implemented in `_simulate_trade_trailing()` and `generate_v5_sweep_outcomes_trailing()` in data/common.py.
- **v5.0.8+ Conviction-Based Position Sizing**: Added `--v5-conviction-sizing` for score-tiered position sizing. Uses percentile-based tiers computed from rolling score history: top `--v5-conviction-top-pct`% → `--v5-conviction-top-mult`x, top `--v5-conviction-high-pct`% → `--v5-conviction-high-mult`x, middle → 1.0x, bottom 50% → 0.5x. Directional confidence boost (`--v5-conviction-conf-thresh`, `--v5-conviction-conf-boost`) multiplies size when p_dir is high. Stacks multiplicatively with Kelly and regime sizing (final_mult = kelly × regime × conviction). Implemented as ConvictionSizer class in v5_position_sizer.py.

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