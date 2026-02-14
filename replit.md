# BTC Futures Signal Dashboard

## Overview
This project is an institutional-grade, AI-driven dashboard for generating sophisticated BTCUSDT futures trading signals. It uses machine learning, real-time market data, and sentiment analysis to provide AI-powered trade plans, aiming for continuous learning and adaptation. The system delivers a robust and selective trading solution by leveraging advanced AI and comprehensive market insights to identify high-confidence trading opportunities. The goal is to create a cutting-edge platform for futures trading, maximizing market potential through AI-driven precision and continuous adaptation.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### UI/UX Decisions
The frontend is built with React, TypeScript, and Vite, featuring a modern UI. It uses `shadcn/ui` (Radix UI, Tailwind CSS) for components and Recharts for data visualization, with navigation managed via a tabbed interface. A Pro Dashboard provides real-time analytics across six dedicated tabs. An R/$ toggle allows switching between R-units and USD for financial metrics.

### Technical Implementations
The backend is developed with Node.js and Express.js (TypeScript, ESM), following a RESTful API pattern. AI integration is handled via OpenAI. Market data is sourced from Binance Vision API, with fallbacks to CoinGecko and CryptoCompare, supplemented by a Replit-hosted data proxy. Bi-directional communication with a local GPU trainer occurs via dedicated API endpoints. A WebSocket server handles real-time event streaming for the Pro Dashboard. The system incorporates a Triple-Lane Aggression Engine (CORE/FLOW/SCALP) routed by HTF score, per-symbol daily R budgets, and lane-specific thresholds. A live learning system supports scheduled retraining and safe model promotion, including per-symbol model management and stringent promotion gates. A money management system converts R-based metrics to USD values based on configurable account equity and risk settings.

### Feature Specifications
The core AI model predicts the quality of entering a trend-following trade (binary 0/1) based on 63 features on a 15-minute timeframe with a 24-bar horizon. Training utilizes an EnhancedMultiHeadMLP architecture with HTF-gated Triple Barrier labeling and optimizes for PR-AUC. Inference applies a probability threshold and HTF alignment gates to generate LONG/SHORT/HOLD signals, with position sizing based on ATR, dynamic account risk, and confidence. A policy auto-tuner optimizes trade frequency by evaluating configurations across multiple regimes. A sophisticated cost model accounts for fees, spread, and slippage in R-unit calculations. Safety kill-switches, including per-symbol daily drawdown caps and performance-based disabling, are implemented. A Per-Asset Trade Quota Controller dynamically relaxes FLOW thresholds when a symbol is under-trading.

The Triple-Lane Aggression Engine routes trades as CORE > FLOW > SCALP > HOLD based on HTF score. Each lane has specific entry requirements, sizing multipliers, horizons, and daily R budgets. SCALP trades have advanced volatility and momentum gates, a short horizon, and specific TP/SL. Execution modes are strictly controlled (signal_only, paper, live) via CLI flags, impacting trade recording and exchange interaction.

A Smart Trade Manager system dynamically evaluates open positions each cycle with priority-based exit rules: TIME_EXIT, ADVERSE_FLIP, STALL_TAKEPROFIT, TRAIL_SL, BREAKEVEN. It tracks MFE/MAE and manages SL updates.

The v5.0 training pipeline supports multi-asset training with per-symbol time-based splits and symbol_id tracking. The EnhancedMultiHeadMLP includes a value_head for E[net R] regression and optional symbol_embedding. Bias initialization and post-training temperature scaling calibrate logits. Lane gating adds E[net R] minimum gates per lane. Promotion gates enforce performance metrics and calibration sanity.

Training stability improvements include fixes for logit polarity and PR-AUC stabilization. The v4.5.3 stabilization patch reduces focal loss gamma from 1.5 to 1.0, OHEM hardest negative fraction from 25% to 8%, max learning rate from 1e-4 to 6e-5, and tightens enter logit clamp from [-10,10] to [-5,5]. Earlier patches added soft labels, edge regression head, and logit separation regularizer. A PR-AUC anti-collapse tuning reduces "stacked aggressiveness" after FULL stage switch: focal alpha 0.35->0.45, OHEM neg_pct 0.20->0.08, edge_head weight 0.30->0.15, soft_label temp 1.5->1.2. A [PR_TUNE] summary log prints all tuning parameters on startup, and a collapse guard logs [ALERT] if PR-AUC drops >0.06 immediately after WARMUP->FULL transition.

A two-stage loss schedule prevents ENTER classifier early collapse (Pred%=0). During warmup (default 10 epochs), training uses plain BCEWithLogitsLoss with capped pos_weight (default 2.0), no focal loss, and no OHEM. After warmup, training transitions to the full loss configuration (focal loss, OHEM if enabled, uncapped pos_weight). CLI flags: --loss-warmup-epochs, --warmup-pos-weight. Each epoch logs [LOSS_STAGE] with stage=WARMUP or stage=FULL and active settings.

p_enter percentiles are computed from the exact same sigmoid(enter_logits) tensor used for PR-AUC computation, with no calibration or policy mapping applied. Each eval epoch logs [PENTER_AUDIT] with p_min/p_max/p_mean/logits_min/logits_max and [PENTER_PCTL] with p50/p75/p90/p95/p99. NaN/Inf in logits or p_enter raises RuntimeError immediately.

A --verify-enter-metrics mode runs 3 validation passes after training, computes Pred%/PR-AUC/sep/p75/p99 per pass, asserts: (A) Pred% in [2%,60%], (B) sep increases from epoch 1 to epoch 10, (C) p99 > p75. Writes verify_enter_metrics.md report with pass results table and assertion outcomes.

Multi-asset data ingestion: download_data() supports per-symbol downloads (data_cache/{SYMBOL}_15m.parquet). A preflight_data_check() verifies all requested symbols have parquet files with >= 20k bars before training, with [DATA_CHECK] log lines per symbol. CLI flags: --download-missing-data auto-fetches missing parquets from the dashboard API; --allow-partial-data trains on available symbols only instead of aborting.

A HTF Warmup & Candle History mechanism ensures sufficient historical data for indicator computation, with a strict WARMUP gate checking minimum bar counts before trading. An optional direct HTF fetch provides more stable indicators.

The v4.6 "Directional Separation" feature introduces bidirectional triple-barrier labeling (no HTF gating). For every bar, both LONG and SHORT outcomes are computed, producing y_quality (best-direction trade quality), y_dir (binary direction: 1=LONG, 0=SHORT), y_dir_conf (sigmoid-mapped confidence), and y_htf_score (4-class HTF alignment from past-only features). The model gains dir_head (binary BCE) and htf_head (4-class CrossEntropy) with configurable composite loss weights (--w-quality, --w-dir, --w-htf). Evaluation tracks direction AUC/accuracy and HTF macro-F1. A --verify-v46-separation mode checks label distributions, target ranges, and model output shapes. Backward compatibility with v4.5 checkpoints is maintained via strict=False loading and default-disabled head flags.

The v4.7 "Label Geometry Fix" addresses overly permissive labeling (~90% positive rate) by implementing strict ENTER=1 criteria. ENTER=1 requires: (A) TP hit BEFORE SL within horizon, OR (B) expiry with R >= r_min_expiry_strict (default 1.0), AND in both cases best_R >= r_min_enter (default 0.8). An auto-balance positive rate targeter (--auto-balance-enter-labels) searches r_min_enter in [0.3, 1.5] to achieve ~18% positive rate (configurable via --target-enter-rate, --target-enter-rate-min/max). pos_weight guardrails cap between --pos-weight-min (0.5) and --pos-weight-max (6.0). OHEM default changed to 0.15. v4.7 is the default labeling mode (--use-v47-labels); --no-v47-labels falls back to v4.6. A --verify-v47-labels mode validates label distributions and generates verify_v47_labels.md. All v4.7 config is saved to checkpoints for reproducibility. Changes are training-only — no live trading or dashboard modifications.

### System Design Choices
Data management uses Drizzle ORM for PostgreSQL and Zod for type-safe validation. The system persists all learning states and separates live sentiment data from historical price/volume. The client is bundled by Vite and the server by esbuild. Centralized timeframe configuration ensures consistency. A runtime diagnostic system provides health endpoints and UI console logging. The GPU training API supports training, status checks, and daily retraining.

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