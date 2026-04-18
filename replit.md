# Neural Terminal — AI Trading Dashboard

## Overview
Neural Terminal is an institutional-grade, GPU-accelerated AI trading system designed for multi-asset crypto futures. It acts as a comprehensive trading terminal, providing real-time signals, managing both paper and live trading operations, and tracking performance. The system integrates a v5 neural network, with the web application serving as the primary interface for advanced analytics and automated trading capabilities. The project's ambition is to deliver a cutting-edge solution for AI-driven crypto futures trading.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend (9-Page Trading Terminal)
The frontend is built with React, TypeScript, and Vite, leveraging `shadcn/ui` (Radix UI, Tailwind CSS) for UI components and Recharts for data visualization. It features a dark navy theme with neon accents. Key pages include Command Center, Live Trading, Paper Trading, Analytics, Training Monitor, Bitget Trading, Neural Monitor, Trade History, and Settings.

### Backend (Node.js + Express + TypeScript)
The backend provides API routes and services to support the frontend and interact with external systems. It manages data retrieval, handles cycle logs and executed trades, and integrates with a local GPU trainer. Key features include a Paper Trading Engine, Bitget and Bybit client integrations for live trading, Market Regime/Chop Protection, and an Order Flow Pipeline for signal validation. A WebSocket server enables real-time event streaming.

### Symbol Configuration
All 20 trading symbols, including precision settings, are defined in `shared/symbols.ts`.

### V5 Neural Network (`V5Forecaster`)
The core v5 neural network runs on a local GPU, featuring multi-head output for various predictions based on 95 features. It includes a V5 Composite Scoring Engine, per-symbol edge learning, and a live feature pipeline for real-time data integration. The network has undergone continuous improvements for biased training labels, confidence gating, and dynamic risk management.

### V6 Neural Network (`V6Forecaster`)
The next-generation `V6Forecaster` utilizes a Causal Conv1D, Positional Encoding, Transformer Blocks, Mixture-of-Experts, and 6 output heads. It incorporates temporal context, MoE routing, feature masking, auxiliary self-supervised loss, and a confidence calibration head for signal gating. It supports flexible scaler loading and V6-specific training arguments.

### V7 Augmented Dataset Ingest (`gpu_trainer/data_ingest/`)
Hydrates the V7 augmented dataset from `data.binance.vision` (live `fapi.binance.com` is geo-blocked from Replit; the public CDN is reachable). Modules: `binance_api.py` (streaming ZIP iterators), `klines_backfill.py`, `flow_features.py` (1m kline aggregator → `flow_features_15m`), `funding_backfill.py`, `oi_backfill.py`, `cli.py` (`python -m gpu_trainer.data_ingest.cli {funding|klines|oi|flow}`), `run_all.sh` (run inside a Workflow so the subprocess survives shell teardown).

### V7 Truth-Discovery Audit (`gpu_trainer/eval/v7_signal_audit_augmented.py`)
Walk-forward signal-learnability audit on the augmented dataset. 7 target variants (ret_15m, ret_60m, ret_240m, sign_60m, ret_60m_volnorm, ret_60m_quintile, mfe_minus_mae_4) × 2 simple models (Ridge, HGBR) × 5 walk-forward folds (24m train / 6m test). **16-bar embargo** between train end and test start (>= longest forward target horizon, prevents label-overlap leakage at fold boundaries). Ridge uses median-impute + clip + standardize; HGBR is NaN-native (so symbols without OI history audit cleanly). Verdict logic is fail-closed on the cost gate: a positive top-decile mean net per-trade return after 8 bps round-trip cost is the minimum evidence that a discovered IC is *tradeable*. Outputs `.local/reports/v7_truth_discovery_augmented.{md,json}`.

### V7 Audit Result (2026-04-18) — VERDICT: PIVOT
Ran on 7 full-history symbols (BTC/ETH/BNB/SOL/ADA/AVAX/XRP), each ~180k 15-min bars from 2021-02 → 2026-04, 30 features (price/vol + flow_features_15m + funding + OI where available).
- **Signal IS present and is not a leakage artefact.** Adding the 16-bar embargo only moved best cross-fold mean IC 0.0712 → 0.0695 (a 2% drop, not the collapse leakage would produce). `sign_60m` (Ridge): cross-symbol mean IC `+0.055`, sign positive on every fold for all 7 symbols. `ret_60m_quintile` (Ridge): mean IC `+0.052`, all-folds-positive on 7/7.
- **Cost gate fails uniformly.** Zero (symbol × target × model) cells produce a positive top-decile mean net per-trade return after 8 bps round-trip. Best is `-1.3 bps`. The naive top-decile-take-and-hold-60m policy loses money everywhere.
- **PIVOT, per locked rules.** Best IC ≥ 0.05 but `best_td_net ≤ 0` ⇒ PIVOT. This audit, with this cost assumption and this naïve trading proxy, did *not* refute the null that the discovered signal is non-tradeable. The V7 5-layer organism build (Task #104) is **NOT authorised** on this evidence alone.
- **What would unlock GO before building the organism.** Three follow-ups, runnable on the same dataset: (1) cost-sensitivity sweep at 2/4/6 bps round-trip, (2) holding-horizon optimiser per (symbol × target) instead of fixed 60 min, (3) ex-ante threshold tuner using prior-fold predicted distribution. Any of these flipping the cost gate on ≥3 symbols upgrades to REDESIGN.
- Full report: `.local/reports/v7_truth_discovery_augmented.md`. Raw fold-level data: `.local/reports/v7_truth_discovery_augmented.json`.

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