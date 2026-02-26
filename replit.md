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

The core AI model predicts trend-following trade quality using 85 features on a 15-minute timeframe with a volatility-adaptive horizon (8-48 bars). It employs an EnhancedMultiHeadMLP architecture with HTF-gated Triple Barrier labeling, optimized for PR-AUC. Inference involves a probability threshold and HTF alignment for LONG/SHORT/HOLD signals, with dynamic position sizing based on ATR, account risk, and confidence. A policy auto-tuner optimizes trade frequency, and a cost model accounts for trading fees. Safety kill-switches include per-symbol daily drawdown caps and a Per-Asset Trade Quota Controller.

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
- **v5.2.0 Precision Audit Fixes**: Six bugs fixed to improve forward test accuracy: (1) Quality gate percentiles now use training-set reference arrays instead of test-set (fixes lookahead bias via `_build_train_ref_arrays`); (2) Oracle best-side fallback removed — `r_long/r_short/out_long/out_short` now required (raises `ValueError` if missing); (3) Head disagreement gate blocks trades when 2+ model heads conflict (`--v5-head-disagree-gate`); (4) Statistical edge metrics added — Sortino ratio, t-statistic, p-value, bootstrap 95% CI on expectancy, tighter sanity checks (Sharpe>5, PF>3, WR>75%); (5) Train/test purge gap — `horizon` bars removed from end of training set to prevent label leakage across the boundary; (6) Slippage deduction in score computation (`--slippage-base-bps`, default 0). Tests: 18 new tests in `test_v5_precision_audit.py`.
- **v5.3.0 Signal Quality Upgrade**: Structural improvements to the signal pipeline addressing feature quality, regime segmentation, label quality, and horizon suitability. (1) Enhanced feature engineering — removed 6 redundant features (RSI_7, ATR_7, vol_weighted_mom_5, roc_accel_5, vol_regime_ratio, efficiency_ratio), added 7 information-dense features (ATR_ratio_7_28, bb_squeeze, vol_regime_roc, trend_efficiency, momentum_acceleration, cvd_zscore, volume_price_divergence) and 2 cross-timeframe features (rsi_divergence_15m_1h, macd_hist_slope_1h); (2) Regime-adaptive labels — ADX-based deadzone (20th pctl trending, 50th pctl choppy, 30th normal), MAE penalty (0.5x for near-SL trades), clean entry bonus (1.3x for early favorable moves), side-confidence sample weighting; (3) 4D regime vector as model input — 5 continuous features (regime_trend, regime_volatility, regime_momentum, regime_session_sin, regime_session_cos) with regime-conditional sample weighting (1.3x trending, 0.7x choppy); (4) Volatility-adaptive horizon — per-bar `effective_horizon = base * (median_ATR_50 / current_ATR_14)` clamped [8,48], vol-adjusted SL (1.15x in high-vol); (5) Feature importance report — permutation importance + Spearman correlation matrix via `--v5-feature-report` flag. Total: 85 features (STF:44, ENH:24, HTF:12, REGIME:5). Tests: 29 new tests in `test_v5_features.py`, 224 total passing.
- **v5.3.1 Directional Balance Fixes**: Fixed massive LONG bias (228L/4S in bear market). (1) Directional penalty no longer uses `mu_R` sign — replaced with conviction penalty scaled by edge magnitude (`score_lambda * (1 - p_side) * mu_over_risk`) so penalty is proportional to signal strength and never dominates when mu_R is small (e.g. after debiasing). Score positive when `p_side > λ/(1+λ)`. Independent of mu_R sign; (2) Side-balance KL regularization added to training loss — penalizes model when batch-level LONG/SHORT prediction ratio deviates from label ratio (weight 0.1, on LONG/SHORT only, targets detached); (3) Per-symbol mu_R EMA debiasing in forward test — removes persistent positive/negative drift from `mu_R` predictions per-symbol using global chronological EMA (alpha=0.01), `--v5-mu-debias` (default True); (4) Forward test directional balance diagnostics — stage-by-stage side distribution tracking (pre/post-ema200/post-regime/final) with >80% dominance warnings, per-gate block counts, score component breakdown (edge/penalty/mu_R post-debias/p_side stats/threshold % above); (5) EMA200 gate no longer superseded by multi-regime (both work simultaneously). Tests: 17 tests in `test_v5_directional.py`, 229 total passing.

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