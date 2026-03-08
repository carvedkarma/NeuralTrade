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

### Symbol Configuration
All 20 trading symbols are defined in a single source of truth: `shared/symbols.ts`. This file exports:
- `TRADING_SYMBOLS`: Array of all 20 symbol strings
- `QTY_PRECISION`: Bybit quantity precision per symbol (decimal places for order qty)
- `PRICE_PRECISION`: Bybit price precision per symbol (decimal places for order price)

All server files, client pages, and GPU trainer scripts import from this shared config.

**20 Symbols:** BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, AVAXUSDT, ADAUSDT, DOGEUSDT, LINKUSDT, LTCUSDT, NEARUSDT, PEPEUSDT, SUIUSDT, AAVEUSDT, ARBUSDT, DOTUSDT, MATICUSDT, FILUSDT, APTUSDT, OPUSDT

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
-   **Symbols:** Supports trading for all 20 symbols defined in `shared/symbols.ts`.
-   **Composite Scoring:** Employs a V5 Composite Scoring Engine for signal evaluation.
-   **Per-symbol Edge Learning:** Incorporates symbol-specific scalers, thresholds, and kill switches for refined trading.
-   **Live Feature Pipeline:** Integrates real-time funding rates and open interest data during live inference to ensure feature consistency with training.
-   **Side Balance fix** (`v5_train.py`): KL divergence loss uses balanced 50/50 LONG/SHORT target instead of biased training label distribution. Training data (2021-2026 bull market) had more LONG labels, causing the model to systematically favor LONG. Fix requires retraining.
-   **Live feature pipeline** (`gpu_trainer/live_runner.py`): `_compute_features_for_symbol()` fetches real funding rate (from Binance FAPI `/fapi/v1/fundingRate`) and open interest (`/futures/data/openInterestHist`) during live inference, matching the training pipeline. Funding cached 30min (TTL), OI cached 15min. Falls back to zeros on fetch failure. One-time `[Feature Check]` diagnostic log per symbol per session.
-   **V5 Sharpness Improvements (v5.3):**
    - **Sigma discount** (`--v5-sigma-discount`, default ON): Multiplies scores by `1/(1+sigma)`, penalizing uncertain predictions. `--v5-no-sigma-discount` to disable.
    - **Minimum conviction gate** (`--v5-min-p-side 0.45`, default 0.45): Kills trades where `p_side < min_p_side`, filtering low-conviction signals.
    - **Larger MFE/MAE heads**: Expanded from `[32, 1]` to `[64, 32, 1]` for better risk prediction across 20 symbols. Requires retraining.
    - **Asymmetric MAE loss** (`--v5-mae-asym-weight 2.0`, default 2.0): Penalizes MAE underestimation 2x more than overestimation, producing conservative risk estimates. `1.0` = symmetric (old behavior).

### v6 Neural Network (V6Forecaster)
Next-generation model upgrade in `gpu_trainer/models/v6_forecaster.py`. Same output dict interface as V5 — all existing scoring, forward test, and live runner infrastructure works unchanged.
-   **Architecture:** Causal Conv1D (3 layers, 128ch) → Positional Encoding → 2x Transformer Blocks (4-head self-attention, causal mask) → Mixture-of-Experts trunk (4 experts, top-2 sparse routing) → 6 output heads. ~1.5M params vs V5's ~300K.
-   **New capabilities:**
    - Temporal context: sees 16 bars (4 hours) of history via sliding window, not just 1 bar.
    - MoE routing: 4 specialized expert MLPs (trending, reverting, volatile, breakout), top-2 gating per sample. Uses noisy top-k gating (learnable Gaussian noise on gate logits during training, Shazeer et al.), expert dropout (p=0.1, randomly masks one expert per batch), and Switch Transformer load balance loss (f×P cross-term, w=0.05). Auto-recovery detects expert collapse (<5% hard routing for 3 epochs) and temporarily boosts balance weight 20x. If collapse persists for 5 epochs, dead expert gate weights are reinitialized by cloning from the strongest alive expert + noise (one-time per training run). Gate biases initialized with random uniform ±0.1 and gate weights with Xavier (gain=1.0) to prevent identical expert starts. `get_expert_usage()` reports hard routing fractions (not soft probs).
    - MoE collapse recovery (v2): `reinit_dead_experts()` deep-copies full MLP state dict (not just gate weights) from strongest alive expert to dead experts with 0.02 noise perturbation. Gate noise increased to 0.05 (was 0.01), bias offset to +0.2 (was +0.1). Multi-attempt reinit: up to 3 reinits per run with 10-epoch cooldown (was one-shot). Entropy bonus on gate probability distribution activated when collapse detected (penalizes low entropy = encourages uniform routing). Entropy bonus auto-disabled on recovery.
    - Feature masking: randomly zeros 15% of features during training (like BERT). Forces robustness.
    - Auxiliary self-supervised loss: next-bar feature prediction forces trunk to learn market structure.
    - Confidence calibration head: sigmoid output (0-1) predicting its own accuracy. Live runner gates signals with `confidence >= 0.4`.
-   **Training:** `python quick_start.py --train-v5 --v6` activates V6. V6-specific args: `--v6-seq-len`, `--v6-conv-channels`, `--v6-n-conv-layers`, `--v6-attn-heads`, `--v6-attn-layers`, `--v6-n-experts`, `--v6-expert-top-k`, `--v6-feature-mask-ratio`, `--v6-aux-weight`, `--v6-confidence-weight`, `--v6-moe-balance-weight`.
-   **Dataset:** `V6SequenceDataset` builds sliding windows per symbol (no cross-symbol boundaries), with zero-padding for early bars.
-   **Loss:** `compute_v6_loss` = all V5 loss components + MoE balance (w=0.05, Switch Transformer f×P cross-term) + aux next-bar MSE (w=0.1) + confidence calibration BCE (w=0.15).
-   **Balanced sampling:** `--balanced-sampling-mode weighted` uses inverse-frequency loss weighting instead of truncation.
-   **Live inference:** `live_runner.py` detects `model_type='v6_forecaster'` in checkpoint, instantiates V6Forecaster, computes seq_len bars of scaled features per symbol, and uses confidence output to gate signals. Scaler loading supports both V5 and V6 model types, with priority: per-symbol scalers from checkpoint → global scaler from checkpoint → per-symbol scalers from disk (joblib) → column scalers from disk.
-   **Per-symbol scaling:** `_compute_features_for_symbol` checks `engineer._per_symbol_scalers[symbol]` first, then `_v5_global_scaler`, then falls back to `transform_and_clip`. Per-symbol scalers loaded from checkpoint or `per_symbol_scalers.joblib` are stored as `{symbol: RobustScaler}` on the engineer.
-   **Checkpoint model_type:** `'v6_forecaster'`. Config saves all V6 hyperparameters for reproducible loading. Per-symbol scaler keys standardized to `center_`/`scale_` (matching scikit-learn attributes). Loading code handles both old `center`/`scale` and new `center_`/`scale_` keys for backwards compatibility.
-   **Files:** `gpu_trainer/models/v6_forecaster.py`, `gpu_trainer/train/v5_train.py` (V6SequenceDataset, compute_v6_loss, train_v5_model with model_version='v6'), `gpu_trainer/quick_start.py` (--v6 CLI args), `gpu_trainer/live_runner.py` (V6 model loading + seq inference + confidence gating).

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
