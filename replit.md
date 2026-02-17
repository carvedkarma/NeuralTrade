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

The v5.0.2 "Pipeline Fix" addresses 10 critical bugs in the training pipeline: (1) Per-symbol 80/20 time-based data split then concat (prevents future leakage); (2) RobustScaler fitted on train, applied to val (replaces fillna(0)); (3) V5-consistent sweep outcomes replacing v4.7 generate_v47_quality_targets; (4) Data-adaptive quality gates using percentile thresholds (p75 sigma/mae, p50 |mu_R|, p60 p_trade); (5) TPD controller clamp widened to [p5,p99] allowing negative thresholds; (6) Three-stage loss schedule (epochs 0-5 action-heavy, 6-15 balanced, 16+ full) preventing NLL dominance; (7) Clamped Gaussian NLL (sigma in [0.01,5.0]) for stable training; (8) Feature warmup mask invalidating first 50 bars per symbol; (9) Model output distribution diagnostics logged every sweep epoch; (10) Scaler parameters saved in checkpoints for inference reproducibility.

The v5.0.3 "Time-Based Split & Forward Test" adds proper out-of-sample validation: (1) Optional time-based train/test split via `--v5-train-end-date` and `--v5-test-start-date` (YYYY-MM-DD), replacing the default 80/20 percentage split per symbol when dates provided; (2) Forward test with frozen decision layer (`--v5-forward-test`) using the best checkpoint's fixed threshold, quality gates, and cooldown — no TPD adaptation, no calibration tuning, no percentile sweep — providing unbiased performance measurement on held-out data; (3) Walk-forward analysis (`--v5-walk-forward`) with rolling N-month train / M-month test windows (`--v5-wf-train-months`, `--v5-wf-test-months`), training and forward-testing each fold independently, with per-fold and aggregated metrics (win rate, expectancy, PF, Sharpe, max drawdown, equity curve). Reports saved as JSON. The forward test validates test_valid mask, sanitizes NaN scores, and guards against empty splits.

The v5.0.4 "Side-Conditional Outcomes" eliminates the oracle best-side bias that inflated backtest metrics. Key fixes: (1) `generate_v5_sweep_outcomes` now returns `r_long`, `r_short`, `out_long`, `out_short` arrays independently — evaluation selects outcomes based on the model's predicted side, not `max(long_r, short_r)`. Legacy `realized_r`/`outcome` keys preserved but marked DEPRECATED. (2) Both `_run_v5_sweep` and `run_v5_forward_test` use `np.where(sides==1, r_long, r_short)` for side-conditional selection. (3) Sharpe annualization fixed: uses `sqrt(trades_per_year)` instead of `sqrt(252*96)` which treated per-trade returns as per-bar returns. (4) Walk-forward entry fee bug fixed (was using `size=1.0` instead of `position_size`). (5) Runtime sanity warning added: if Sharpe>20, PF>5, WR>85%, logs a leakage/oracle warning. (6) Tests in `gpu_trainer/tests/test_side_conditional.py` validate all fixes. Forward test now logs whether it uses side-conditional or deprecated oracle outcomes.

The v5.0.5 "Trade Frequency Control & Side Diagnostics" enhances trade frequency management and adds diagnostic tools: (1) Intuitive CLI flags `--v5-target-trades-per-day` and `--v5-target-trades-per-day-band` override internal `--v5-target-tpd`/`--v5-tpd-tol` for easier TPD targeting (e.g., 3-4 trades/day). (2) Comprehensive side diagnostics log three-stage direction breakdown: all-bar distribution, eligible after quality/threshold, and taken after cooldown/gate — with one-sided warnings when LONG=0 or SHORT=0 and trades>10, identifying model bias vs code bugs. (3) Optional `--v5-ema200-regime-gate` blocks LONG trades when close < EMA200 and SHORT trades when close > EMA200, using a leakage-free recursive EMA computation. (4) Enhanced score side analysis logging (p_long_mean, p_short_mean, edge_long_mean, edge_short_mean) in forward test. (5) 8 new tests covering CLI parsing, side counts, one-sided detection, EMA computation, and gate blocking logic.

The v5.0.6 "Zero-SHORT Fix & Side-Specific Targets" addresses critical model bias and oracle contamination in risk estimates. Key fixes: (1) **Zero-SHORT bug fix**: The scoring formula `edge_short = p_short * (-mu_R) / risk` made edge_short always negative when mu_R > 0 (common in crypto bull markets), producing 1041 LONG / 0 SHORT observed. New `--v5-score-side-mode action_head` (default) uses `abs_mu = |mu_R|` for magnitude, letting `p_long` vs `p_short` from the trained action head drive side selection. Legacy `mu_sign` mode preserved via `--v5-score-side-mode mu_sign`. Penalty formula updated: applies when chosen side conflicts with mu_R sign (LONG with mu_R < 0, SHORT with mu_R > 0). (2) **Side-specific MFE/MAE targets**: Target generator now computes `mfe_R_long`, `mae_R_long`, `mfe_R_short`, `mae_R_short` independently — long_mfe = (max_high - entry)/ATR, short_mfe = (entry - min_low)/ATR. Eliminates oracle bias where previous code selected best-side MFE/MAE using future data (lines 85-90 selected whichever side looked better). Training uses `np.where(act_arr == 2, mfe_short, mfe_long)` for side-conditional target selection based on action_label. Action labeling now checks side-appropriate MFE for the mfe_min gate. (3) **R/R ratio scoring bonus**: `--v5-rr-weight` (default 0.0) adds `rr_weight * (mfe_pred / (mae_pred + eps)) * abs_mu / risk` to score, rewarding setups with high MFE/low MAE profiles. (4) **12 new tests** (26 total, all passing) covering: action_head SHORT generation, mu_sign backward compatibility, mixed side selection, penalty conflict detection, R/R boost verification, side-specific target generation, Long MFE == Short MAE relationship, CLI flag parsing.

The v5.0.7 "Capital Protection Layers" adds three hard capital protection mechanisms to address production issues (Week 1-2 negative performance, fragile expectancy swinging -0.47 to +0.38 weekly). Key additions: (1) **Regime gate in training sweep**: `_run_v5_sweep` now accepts `close_prices` and `ema200_regime_gate` params. When enabled, computes EMA200 inside the sweep and hard-blocks LONG trades when close < EMA200, SHORT trades when close > EMA200 — matching forward test behavior. This eliminates the train/eval mismatch where models got credit for trades that would be blocked in production. (2) **Weekly loss cap kill-switch** (`--v5-weekly-loss-cap`): Both `_run_v5_sweep` and `run_v5_forward_test` track cumulative R within each calendar week. When weekly R drops below the cap (e.g., -5.0R), all remaining trades that week are skipped. Resets at each new week boundary. Prevents catastrophic drawdown in regime-shift weeks. (3) **Warmup skip guard** (`--v5-warmup-skip-bars`): Forward test skips all trades in the first N bars of the test window (e.g., 96 = 1 day at 15min timeframe). Addresses cold-start losses observed consistently in Week 1 across all forward test windows. All three features are wired through CLI flags, `V5ForwardTestConfig`, and the `train_v5_model` function signature. **7 new tests** (33 total, 28 passing + 5 skipped/torch) covering: regime gate blocking in sweep, weekly cap triggering, warmup config, CLI flag parsing, EMA direction correctness, weekly reset between weeks.

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