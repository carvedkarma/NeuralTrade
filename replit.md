# BTC Futures Signal Dashboard

## Overview
This project is an institutional-grade, AI-driven dashboard for generating sophisticated BTCUSDT futures trading signals. It integrates machine learning, real-time market data, and sentiment analysis to provide AI-powered trade plans. The system focuses on continuous learning and adaptation, aiming to deliver a robust and selective trading solution by leveraging advanced AI and comprehensive market insights for high-confidence trading opportunities. The project's vision is to establish a cutting-edge platform for futures trading, capitalizing on market potential through AI-driven precision and continuous adaptation.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### UI/UX Decisions
The frontend is built with React and TypeScript using Vite, featuring a modern UI with shadcn/ui (Radix UI, Tailwind CSS) for components and Recharts for data visualization. Navigation is handled through a tabbed interface.

### Technical Implementations
The backend uses Node.js with Express.js (TypeScript, ESM) and follows a RESTful API pattern. AI integration is managed via OpenAI. Market data is sourced primarily from Binance Vision API, with fallbacks to CoinGecko and CryptoCompare, augmented by a Replit-hosted data proxy. Bi-directional communication with a local GPU trainer is established via dedicated API endpoints.

### Feature Specifications
The system utilizes an AI model (v3.3.0) that predicts the quality of entering a trend-following trade (binary 0/1) rather than direction, which is derived from Higher Time Frame (HTF) trend alignment. GPU-accelerated training uses the EnhancedMultiHeadMLP architecture with specific residual blocks and a binary classifier head. The model employs 63 features (47 Short-TimeFrame, 10 HTF, 3 Funding, 3 Open Interest) on a 15-minute timeframe with a 24-bar horizon. Inference applies a probability threshold and HTF alignment gates for generating LONG/SHORT/HOLD signals. The model incorporates walk-forward weight saving, feature version locking, and prediction drift monitoring.

The training label strategy (v3.1.0) uses HTF-gated Triple Barrier labeling for binary entry quality, considering HTF trend alignment, slope strength, and range position. Training uses BCEWithLogitsLoss with `pos_weight` for class imbalance, with PR-AUC as the primary metric.

Key input features (v3.3.0) include 47 STF features (e.g., returns, SMAs, RSIs, MACD, Bollinger Bands, ATR, volume metrics, ADX, Stochastic, OBV, and enhanced features like RSI divergence and VWAP deviation), 10 HTF features (derived from 1H and 4H bars for slope, trend, RSI, ATR ratio, and range position), 3 Funding features (rate, delta, z-score), and 3 Open Interest features (normalized OI, 1-hour delta, z-score). Feature versioning ensures data consistency and model integrity.

Training involves an optimized learning rate schedule (warmup and cosine annealing), early stopping, and dual checkpoint saving based on validation loss and a combined trading score. The current model (v3.1.0) focuses on binary ENTER quality classification, with all auxiliary heads disabled during training to maximize classification accuracy. Evaluation metrics include Precision, Recall, F1, and PR-AUC. Inference applies an enter threshold and HTF alignment gates. Position sizing is ATR-based with dynamic account risk, scaled by confidence and edge, and signals below thresholds are downgraded.

The system integrates multi-head predictions from the GPU trainer into the dashboard, with API endpoints for pushing, retrieving the latest, and viewing history of predictions. The `MultiheadSignalCard` component displays unified predictions, including direction probabilities, quantile spread, volatility state, expected return, uncertainty, edge, and derived trade levels.

### GPU Trainer Commands
Training: `python quick_start.py --url URL --epochs 300`
Prediction: `python quick_start.py --url URL --predict-only`
Regime Eval: `python quick_start.py --url URL --regime-eval --policy threshold:0.85 --cooldown 6 --tp-mult 3.0 --sl-mult 1.5`
Regime Eval (custom costs): `python quick_start.py --url URL --regime-eval --fees-entry-bps 2 --fees-exit-bps 2 --slip-k 0.05`
Geometry Sweep (full): `python quick_start.py --url URL --regime-eval --geometry-sweep --thresholds 0.80,0.85 --topn-list 8,10,12,15,18 --paired-tp-sl 3.0:1.25,3.0:1.5,3.5:1.5 --cooldowns 4,6,8 --target-tpd 2.5 --target-tpd-tol 1.0`
Geometry Sweep (threshold-only): `python quick_start.py --url URL --regime-eval --geometry-sweep --thresholds 0.80,0.85 --cooldowns 4,6,8`
Geometry Sweep (debug costs): `python quick_start.py --url URL --regime-eval --geometry-sweep --debug-costs`
Live (auto-load best policy): `python quick_start.py --url URL --live --paper --symbols BTCUSDT,ETHUSDT,SOLUSDT --interval 15m --enable-learning`
Live (explicit policy): `python quick_start.py --url URL --live --paper --symbols BTCUSDT,ETHUSDT,SOLUSDT --interval 15m --enter-threshold 0.85 --tp-mult 3.0 --sl-mult 1.5 --cooldown 6`
Live (dry run): `python quick_start.py --url URL --live --dry-run --dry-run-candles 200`
Live (no exec module): `python quick_start.py --url URL --live --paper --no-exec`
Live (per-symbol models): `python quick_start.py --url URL --live --paper --per-symbol-models --enable-learning`
Live (learning, no auto-promote): `python quick_start.py --url URL --live --paper --enable-learning --no-auto-promote`

Key files: `gpu_trainer/quick_start.py`, `gpu_trainer/training/triple_barrier.py` (shared barrier simulator + cost model), `gpu_trainer/data/regression_targets.py` (labeling), `gpu_trainer/data/pipeline.py` (features/data), `gpu_trainer/live_runner.py` (multi-asset live loop), `gpu_trainer/execution.py` (lower-TF entry), `gpu_trainer/portfolio.py` (position tracking/risk caps), `gpu_trainer/learning.py` (scheduled retrain + safe promotion).

### Policy Auto-Tuner (v3.4.x)
The geometry sweep supports both threshold and percentile (topN) policies for optimizing trade frequency while maintaining net edge. The sweep evaluates configs across multiple regimes and selects the BEST config using NET-first priority:
1. PF_net >= 1.05
2. E_net > 0
3. Trades/day within target band (default 2.5 +/- 1.0)
4. Profitable regimes >= 2/3
Fallback: highest PF_net among configs with TPD >= 1.5

The winning policy is saved to `checkpoints/best_policy.json` with full metadata (policy_type, threshold, TP/SL/cooldown, cost config, metrics, timestamp). The live runner automatically loads this policy if `--enter-threshold` or `--policy` is not explicitly set on the CLI.

CLI flags for sweep: `--topn-list`, `--paired-tp-sl`, `--thresholds`, `--cooldowns`, `--target-tpd`, `--target-tpd-tol`

Enhanced HOLD debug output: when live inference outputs non-ENTER decisions, the CLI prints p_enter percentile ranks (p50/p75/p90/p95/p99) from recent history, plus threshold and HTF gate details.

### Live Learning System (v3.5.0)
The live system supports scheduled retraining with safe model promotion. Key components:
- `learning.py`: LearningManager with LearningConfig for scheduled daily retrain + walk-forward evaluation
- Per-symbol model management: deployed models stored under `checkpoints/deployed/{symbol}/`
- Safe promotion gates: PR-AUC threshold, PF_net minimum, profitable regime count, TPD range
- Dashboard integration: cycle logs, trade records, and learning stats pushed to `/api/live/*` endpoints
- Candle data caching with append/dedupe (up to 2000 bars per symbol)
- Exchange time sync via Binance serverTime API
- Cooldown tracking per symbol after trade entry
- Retry logic (3 attempts) for all dashboard HTTP pushes

### Live System Dashboard
The "Live System" tab shows:
- Overview cards: open positions, closed trades count, win rate, total net R
- Model Learning Stats: per-symbol cards with PR-AUC, PF_net, E[Net R], trades/day, profitable regimes, trend indicator, promotion status
- Live Trade History: table with symbol, side, entry/exit prices, SL/TP, p_enter, outcome, net R
- Inference Cycle Log: table with timestamp, symbol, price, p_enter, HTF alignment, direction, decision, reasons

DB tables: `live_trade_records`, `model_learning_stats`, `live_cycle_logs`
API endpoints: POST/GET `/api/live/trade`, PATCH `/api/live/trade/:id`, POST/GET `/api/live/learning-stats`, GET `/api/live/learning-stats/latest`, POST/GET `/api/live/cycle-logs`, GET `/api/live/summary`

### Cost Model (v3.3.2)
Default execution: MARKET orders (taker) for entry and exit. Cost components computed in R-units via `compute_trade_cost_r()` in `training/triple_barrier.py`:
- Fees: `(entry_bps + exit_bps) / 10000` (default 5+5 bps taker)
- Spread: `spread_bps / 10000` (default 1 bps, half each side)
- Slippage: `2 * slip_k * ATR/price` (default slip_k=0.10, entry+exit)
- Converted to R: `cost_R = total_cost_pct / (sl_mult * ATR/price)`
- Net R = Gross R - Cost R

Confidence-based sizing: linear scale from 1.0x at threshold to `size_cap` (default 2.0x) at p_enter=1.0. Sized R = Net R * size_mult.

### System Design Choices
Data is managed with Drizzle ORM for PostgreSQL and Zod for type-safe validation. Live sentiment data is separated from historical price/volume data, and all learning states are persisted. The client is bundled by Vite, and the server by esbuild. A centralized timeframe configuration ensures consistency. A runtime diagnostic system provides health endpoints and UI console logging. The GPU training API supports starting training, checking status, and daily retraining, incorporating gradient clipping and learning rate adjustments for stability. A data diagnostics system audits features and labels.

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