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

### Triple-Lane Aggression Engine (v4.3.0)
The system replaces the dual-policy engine with a Triple-Lane router: CORE > FLOW > SCALP > HOLD. HTF score (0-3) replaces the binary trend_aligned gate, computed as +1 per matching h1_trend, h4_trend, slope_ok. Lane routing: CORE requires htf_score>=3 + range_ok + p_enter>=p99 (size_mult=1.0, horizon=24); FLOW requires htf_score>=2 + (range_ok OR momentum_ok) + p_enter>=p95-stepped (size_mult via quota 0.60-0.30, horizon=24); SCALP requires htf_score>=1 + volatility_ok + momentum_ok + p_enter>=p90 (size_mult=0.25, horizon=4 bars, TP=1.20R, SL=0.80R, half cooldown). Per-symbol daily R budgets: CORE=1.2R, FLOW=0.6R, SCALP=0.2R (total 2.0R), reset at UTC midnight. SCALP time-stop exits after 4 bars. Position close callback propagates exit_reason to dashboard trades. All lane fields (lane_selected, htf_score, core_thr, flow_thr, scalp_thr, lane_size_mult, lane_budget_remaining_r, hold_reason) stored on cycle logs; trade records store lane, htf_score, lane_threshold_used, lane_size_mult, exit_reason, lane_horizon.

### Execution Mode Gating (v4.5.0)
Strict, mutually exclusive execution modes: signal_only (default), paper, live. Controlled via `--execution-mode`, `--paper`, or `--live` CLI flags. In signal_only mode: no Position created, no trade records POSTed, no TradeManager runs — only cycle logs with decision=SIGNAL_ONLY and full trade geometry (entry, SL, TP, lane, p_enter, htf_score). In paper mode: positions simulated, trades recorded, TradeManager active. In live mode: exchange orders placed, trades recorded, TradeManager active. Startup log shows `[MODE] execution_mode=<mode>`. Per-trade guard logs: `[NO_EXEC]` (signal_only), `[PAPER_OPEN]` (paper), `[LIVE_OPEN]` (live). Default `--paper` is now `False` to prevent accidental trade recording.

### Smart Trade Manager (v4.4.0)
Dynamic exit intelligence system (gpu_trainer/trade_manager.py) that evaluates open positions each cycle. Five exit rules evaluated in priority order: TIME_EXIT > ADVERSE_FLIP > STALL_TAKEPROFIT > TRAIL_SL > BREAKEVEN > HOLD. Breakeven moves SL to entry at 0.35R MFE; trailing stop engages at 0.60R MFE with 0.40R distance; stall profit-taking closes after 2 bars of no MFE progress above 0.70R; adverse flip closes at -0.60R with HTF score drop >=2 or p_enter decay <80%. TradeManager tracks per-position MFE/MAE (max favorable/adverse R excursion), breakeven_moved flag, and stall counters. Actions returned: HOLD, MOVE_SL, TRAIL_SL, CLOSE_FULL. LiveRunner._run_trade_manager() processes actions each cycle — SL updates propagated to both Position object and dashboard via PATCH /api/live/trade/:id. Position.check_exit uses intrabar high/low for accurate TP/SL resolution (worst-case SL on same-bar conflict). Trade records store max_favorable_r, max_adverse_r, time_exit, breakeven_moved, bars_held, exit_reason. Trade Journal table shows Lane, Exit Reason, MFE, MAE columns. Trade Detail Popup displays lane info, HTF score, thresholds, MFE/MAE, horizon, and exit management badges.

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