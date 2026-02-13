# BTC Futures Signal Dashboard

## Overview
This project is an institutional-grade, AI-driven dashboard designed to generate sophisticated BTCUSDT futures trading signals. It leverages machine learning, real-time market data, and sentiment analysis to provide AI-powered trade plans. The system aims for continuous learning and adaptation, delivering a robust and selective trading solution by utilizing advanced AI and comprehensive market insights to identify high-confidence trading opportunities. The ultimate vision is to create a cutting-edge platform for futures trading, maximizing market potential through AI-driven precision and continuous adaptation.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### UI/UX Decisions
The frontend is built using React and TypeScript with Vite, featuring a modern UI. It utilizes shadcn/ui (Radix UI, Tailwind CSS) for components and Recharts for data visualization, with navigation managed via a tabbed interface. The Pro Dashboard (`/pro`) provides a premium, real-time analytics experience across six dedicated tabs. An R/$ toggle allows switching between R-units and USD values for financial metrics.

### Technical Implementations
The backend is developed with Node.js and Express.js (TypeScript, ESM), following a RESTful API pattern. AI integration is handled via OpenAI. Market data is sourced primarily from Binance Vision API, with fallbacks to CoinGecko and CryptoCompare, supplemented by a Replit-hosted data proxy. Bi-directional communication with a local GPU trainer occurs via dedicated API endpoints. A WebSocket server handles real-time event streaming for the Pro Dashboard. The system implements a Triple-Lane Aggression Engine (v4.3.0) replacing the dual-policy engine, with CORE/FLOW/SCALP lanes routed by HTF score (0-3), per-symbol daily R budgets, and lane-specific thresholds. A live learning system supports scheduled retraining and safe model promotion, including per-symbol model management and stringent promotion gates. A money management system converts R-based metrics to USD values based on configurable account equity and risk settings.

### Feature Specifications
The core AI model (v3.3.0) predicts the quality of entering a trend-following trade (binary 0/1) based on 63 features (47 Short-TimeFrame, 10 HTF, 3 Funding, 3 Open Interest) on a 15-minute timeframe with a 24-bar horizon. Training utilizes an EnhancedMultiHeadMLP architecture with HTF-gated Triple Barrier labeling (v3.1.0) and optimizes for PR-AUC. Inference applies a probability threshold and HTF alignment gates to generate LONG/SHORT/HOLD signals, with position sizing based on ATR, dynamic account risk, and confidence. A policy auto-tuner (v3.4.x) optimizes trade frequency by evaluating configurations across multiple regimes and selecting the best policy based on net profitability, positive expectancy, and target trades per day. A sophisticated cost model (v3.3.2) accounts for fees, spread, and slippage in R-unit calculations. Safety kill-switches, including per-symbol daily drawdown caps and performance-based disabling of the FLOW policy, are implemented. A Per-Asset Trade Quota Controller (v4.2.0) dynamically relaxes FLOW thresholds when a symbol is under-trading relative to its daily target (~2-3 trades/day), with stepped percentile relaxation (p95→p93→p91→p89), proportional risk reduction (size mults 0.60→0.50→0.40→0.30), absolute threshold floor (0.25), and hard daily trade max per symbol. Quota state is stored on cycle logs (quota_step, flow_pct_used, trades_today_total/target/max, quota_flow_risk_mult) and displayed in the Quota Status dashboard card and Cycle Monitor columns.

### Triple-Lane Aggression Engine (v4.3.0 → v4.5.0 "Aggression with Separation")
The system replaces the dual-policy engine with a Triple-Lane router: CORE > FLOW > SCALP > HOLD. HTF score (0-3) replaces the binary trend_aligned gate, computed as +1 per matching h1_trend, h4_trend, slope_ok. Lane routing: CORE requires htf_score>=3 + range_ok + p_enter>=p99 (size_mult=1.0, horizon=24); FLOW requires htf_score>=2 + (range_ok OR momentum_ok) + p_enter>=p95-stepped (size_mult via quota 0.60-0.30, horizon=24); SCALP requires htf_score>=1 + vol_expansion_ok + momentum_ok + p_enter>=p90 (size_mult=0.25, horizon=4 bars, TP=1.20R, SL=0.80R, half cooldown). Per-symbol daily R budgets: CORE=1.2R, FLOW=0.6R, SCALP=0.2R (total 2.0R, CLI-configurable via --budget-core/flow/scalp), reset at UTC midnight. SCALP time-stop exits after 4 bars. Position close callback propagates exit_reason to dashboard trades. All lane fields stored on cycle logs; trade records store lane, htf_score, lane_threshold_used, lane_size_mult, exit_reason, lane_horizon.

**v4.5 SCALP Separation Gates**: Advanced volatility expansion gate requires ATR14/ATR50>=1.20 AND (true_range_z>=1.0 OR bb_width_z>=1.0). Momentum gate requires (ema20_slope>=0.0005 OR macd_hist>=0.0001) AND volume_ratio>=1.2. range_ok=False halves SCALP size_mult (0.25→0.125). Structured logging: [SCALP_GATES], [BUDGET], [EXIT_RESOLVE] with gate metrics, budget states, intrabar exit details. SCALP gate fields (scalpAtrRatio, scalpTrZ, scalpBbZ, scalpEma20Slope, scalpMacdHist, scalpVolRatio, scalpVolExpansionOk, scalpMomentumOk) stored on cycle logs and displayed in ScalpGatesPanel. `--verify-separation` CLI flag runs 200-cycle dry-run with SeparationVerifier asserting gate enforcement, router priority, budget bounds, and exit resolve logs, outputting verify_report_v4.5.md.

### Execution Mode Gating (v4.5.0)
Strict, mutually exclusive execution modes: signal_only (default), paper, live. Controlled via `--execution-mode`, `--paper`, or `--live` CLI flags. In signal_only mode: no Position created, no trade records POSTed, no TradeManager runs — only cycle logs with decision=SIGNAL_ONLY and full trade geometry (entry, SL, TP, lane, p_enter, htf_score). In paper mode: positions simulated, trades recorded, TradeManager active. In live mode: exchange orders placed, trades recorded, TradeManager active. Startup log shows `[MODE] execution_mode=<mode>`. Per-trade guard logs: `[NO_EXEC]` (signal_only), `[PAPER_OPEN]` (paper), `[LIVE_OPEN]` (live). Default `--paper` is now `False` to prevent accidental trade recording.

### Smart Trade Manager (v4.4.0)
Dynamic exit intelligence system (gpu_trainer/trade_manager.py) that evaluates open positions each cycle. Five exit rules evaluated in priority order: TIME_EXIT > ADVERSE_FLIP > STALL_TAKEPROFIT > TRAIL_SL > BREAKEVEN > HOLD. Breakeven moves SL to entry at 0.35R MFE; trailing stop engages at 0.60R MFE with 0.40R distance; stall profit-taking closes after 2 bars of no MFE progress above 0.70R; adverse flip closes at -0.60R with HTF score drop >=2 or p_enter decay <80%. TradeManager tracks per-position MFE/MAE (max favorable/adverse R excursion), breakeven_moved flag, and stall counters. Actions returned: HOLD, MOVE_SL, TRAIL_SL, CLOSE_FULL. LiveRunner._run_trade_manager() processes actions each cycle — SL updates propagated to both Position object and dashboard via PATCH /api/live/trade/:id. Position.check_exit uses intrabar high/low for accurate TP/SL resolution (worst-case SL on same-bar conflict). Trade records store max_favorable_r, max_adverse_r, time_exit, breakeven_moved, bars_held, exit_reason. Trade Journal table shows Lane, Exit Reason, MFE, MAE columns. Trade Detail Popup displays lane info, HTF score, thresholds, MFE/MAE, horizon, and exit management badges.

### v5.0 Multi-Asset Training & Calibration
The v5.0 training pipeline (gpu_trainer/quick_start.py) supports multi-asset training (--symbols BTCUSDT,ETHUSDT,SOLUSDT) with per-symbol 70/15/15 time-based splits and symbol_id tracking. The EnhancedMultiHeadMLP (gpu_trainer/models/multihead.py) adds a value_head for E[net R] regression (HuberLoss, --value-loss-weight 0.5, --value-clip 3.0) and optional symbol_embedding for n_symbols > 1. Bias initialization (log(pos_rate/(1-pos_rate))) on the enter_head prevents p_enter collapse from sigmoid saturation. Post-training temperature scaling calibrates logits via LBFGS on validation data (saved as temp_scale_v5.0.json). Inference (gpu_trainer/live_runner.py) loads temperature and value head config, applies calibrated p_enter = sigmoid(logit/T), and returns e_net_pred from the value head. Lane gating adds E[net R] minimum gates per lane (--min-enet-core 0.00, --min-enet-flow -0.05, --min-enet-scalp -0.02). Promotion gates (gpu_trainer/learning.py) enforce PF_net >= 1.05, E[net] >= 0.0, profitable_regimes >= 2, maxDD_r <= 6.0R, and p95 calibration sanity (0.40-0.98) with CLI-configurable thresholds. Dashboard extensions add eNetPred, enterLogit, temperatureUsed to cycle log schema, Cycle Monitor UI columns (E[net], Lane, Hold Reason), and trade detail popup. Smoke test flags (--smoke-calib, --smoke-infer) enable quick validation of the full pipeline.

### v4.5.1 PR-AUC Stable Patch
Stabilization patch for the PR-AUC Upgrade Pack, fixing Pred% collapse to 0% and PR-AUC stalling at ~0.26. Changes: (A) Soft labels ON by default (temp=1.5 down from 2.0) with --no-soft-labels toggle; (B) OHEM neg_pct reduced from 0.35 to 0.25 to reduce over-aggressive hard-negative mining; (C) Horizon standardized to 16 bars across triple_barrier, model, and regression targets with [HORIZON_CHECK] verification log; (D) Label purity tightened: r_min_expiry raised from 0.5 to 1.0 to remove weak expiry wins; (E) OI pipeline sanity assert ([OI_CHECK] PASS/FAIL) before training validates 30-day coverage >= 70% and nonzero >= 2000; (F) --verify-pr-auc-upgrade extended with 13 checks including soft_labels, ohem_neg_pct, r_min_expiry, horizon alignment; (G) Version bumped to v4.5.1_pr_auc_stable.

### v4.5.0 PR-AUC Upgrade Pack
Training enhancements to improve binary enter quality prediction. Focal BCE loss (gamma=1.5, alpha=0.60) replaces standard BCE, focusing gradients on hard-to-classify samples. Online Hard Example Mining (OHEM) keeps all positives and top 25% hardest negatives per batch. An auxiliary edge regression head (edge = mfe_r - mae_r, weight=0.3) provides trade-quality gradient signal. Soft quality labels (default ON, --no-soft-labels to disable) use sigmoid(edge/temp) with temp=1.5. ECE (Expected Calibration Error) computed with 15 bins before/after temperature scaling and saved in temp_scale_v5.0.json. Promotion gate min PR-AUC raised to 0.42 (from 0.35). CLI flags: --use-focal-loss/--no-focal-loss, --focal-gamma, --focal-alpha, --use-ohem/--no-ohem, --ohem-neg-pct, --use-edge-head/--no-edge-head, --edge-loss-weight, --use-soft-labels/--no-soft-labels, --soft-label-temp, --promote-min-pr-auc, --verify-pr-auc-upgrade. Training config saved in checkpoints for reproducibility.

### HTF Warmup & Candle History (v3.5.1)
The live runner fetches 800 x 15m candles by default (configurable via `--limit-15m`) to guarantee ~50 H4 bars and ~200 H1 bars for HTF indicator computation (SMA20, RSI14, ATR). A strict WARMUP gate checks minimum bar counts (h1 >= 100, h4 >= 50) before allowing HTF indicators and trading for each symbol; symbols in WARMUP emit `decision=WARMUP` cycle logs and skip inference entirely. An optional `--direct-htf` flag fetches 1H (limit=300) and 4H (limit=200) candles directly from the exchange API instead of resampling from 15m data, providing more stable HTF indicators. HTF NaN logging is condensed to a single summary line per cycle to reduce log spam.

### System Design Choices
Data management uses Drizzle ORM for PostgreSQL and Zod for type-safe validation. The system persists all learning states and separates live sentiment data from historical price/volume. The client is bundled by Vite and the server by esbuild. Centralized timeframe configuration ensures consistency. A runtime diagnostic system provides health endpoints and UI console logging. The GPU training API supports training, status checks, and daily retraining, incorporating gradient clipping and learning rate adjustments. A data diagnostics system audits features and labels.

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