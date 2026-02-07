# BTC Futures Signal Dashboard

## Overview
This project is an institutional-grade, AI-driven dashboard for generating sophisticated BTCUSDT futures trading signals. It integrates machine learning, real-time market data, and sentiment analysis to provide AI-powered trade plans. The system focuses on continuous learning and adaptation, aiming to deliver a robust and selective trading solution by leveraging advanced AI and comprehensive market insights for high-confidence trading opportunities. The project's vision is to establish a cutting-edge platform for futures trading, capitalizing on market potential through AI-driven precision and continuous adaptation.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### UI/UX Decisions
The frontend is built with React and TypeScript using Vite, featuring a modern UI with shadcn/ui (Radix UI, Tailwind CSS) for components and Recharts for data visualization. Navigation is handled through a tabbed interface, including sections for Overview, Signal, Neural Network, and Paper Trading.

### Technical Implementations
The backend uses Node.js with Express.js (TypeScript, ESM) and follows a RESTful API pattern. AI integration is managed via OpenAI. Market data is sourced primarily from Binance Vision API, with fallbacks to CoinGecko and CryptoCompare, augmented by a Replit-hosted data proxy. Bi-directional communication with a local GPU trainer is established via dedicated API endpoints.

### Feature Specifications
The system incorporates a regression-based signal system generating comprehensive signals including action, confidence, expected_move, uncertainty, and position sizing. It features regime detection with a Mixture-of-Experts (MoE) model, identifying market states like TRENDING or MEAN_REVERTING. GPU-accelerated training uses the stable EnhancedMultiHeadMLP architecture with [512, 256, 128, 64] residual blocks and progressive head enablement. A multi-head model architecture uses five distinct output heads (Classification, Quantile, VolState, Mu, Sigma) with combined loss functions. A flow forecast system provides regime-conditioned quantile path projections. Advanced labeling uses Triple Barrier Method (ATR-scaled TP/SL/time-expiry barriers) for clean, outcome-based training labels, with Focal Loss (gamma=2.0) and configurable class weight caps for class imbalance. Model management includes walk-forward weight saving, feature version locking, prediction drift monitoring, and label metadata tracking. A professional ensemble predictor combines multiple model predictions with confidence-based voting. A robust training, monitoring, and policy architecture separates model training from live execution policy selection.

### Training Label Strategy
The Triple Barrier Method (Stage 4) is the recommended labeling approach:
- For each bar, three barriers are placed: TP (2.0x ATR above), SL (1.5x ATR below), time expiry (24 bars)
- Whichever barrier gets hit first determines the label: TP hit = LONG, SL hit = SHORT, time expiry = HOLD
- ATR-scaled barriers automatically adapt to current volatility regime
- Horizon: 24 bars (6 hours on 15m timeframe) for meaningful directional separation
- Produces cleaner labels than simple return thresholds because labels reflect actual trade outcomes

### Input Features (57 total, v3.0.0)

**STF features (47, Single-TimeFrame 15m):**
Base features: returns, log_returns, SMA/EMA/std/return at 5/10/20/50/100 periods, RSI-14/7, MACD/signal/hist, Bollinger Bands (upper/middle/lower/width/position), ATR-14/7, volume SMA/ratio, ADX-14, Stochastic K/D, OBV/OBV-SMA.
Enhanced features: RSI divergence (price vs RSI slope mismatch), volume-weighted momentum (5/10 bar), VWAP deviation, close-to-high ratio (intra-bar position), volume delta (buy/sell pressure proxy).

**HTF features (10, Higher-TimeFrame context):**
Resampled from 15m candles into 1H and 4H bars. Each 15m row receives features from the most recently COMPLETED HTF bar (shifted by 1 to prevent leakage). Merged via `merge_asof(direction="backward")`.
For each HTF (1H and 4H):
- `{h1,h4}_sma20_slope` - SMA(20) slope normalized by ATR: `(sma20 - sma20.shift(3)) / (atr14 + 1e-9)`
- `{h1,h4}_trend_sign` - Sign of SMA slope (-1/0/+1), indicates HTF trend direction
- `{h1,h4}_rsi14` - RSI(14) on HTF bar, overbought/oversold context
- `{h1,h4}_atr_ratio` - Ratio of 15m ATR to HTF ATR: `atr_15m / (atr_htf + 1e-9)`, measures relative volatility
- `{h1,h4}_range_pos` - Price position within HTF range: `(close - htf_low) / (htf_high - htf_low)`, clipped [0,1]

Feature versioning: `VERSION = "3.0.0-stf47-htf10"`. Saved in checkpoint metadata. Inference verifies version match and **hard-fails** on mismatch (sys.exit or RuntimeError). Column order is locked at training time and enforced via `reindex()` at inference. Missing or extra columns also trigger hard failure.

### GPU Training CLI Reference
Current stable model: `enhanced_mlp` (EnhancedMultiHeadMLP)
- Gradient norms: 0.35-0.63 (well under 10 threshold)
- Prediction distribution with focal loss: SHORT ~13%, HOLD ~47%, LONG ~40%
- Training data: 5 years of BTCUSDT 15m candles (~175k target, backfilled from Binance)
- Quick start: `python quick_start.py --url https://APP.replit.app` (300 epochs default)

Quick start training flags:
- `--epochs 300` - Training epochs (default: 300)
- `--batch-size 64` - Batch size (default: 64)
- `--lr 0.0001` - Learning rate (default: 0.0001)
- `--warmup-epochs 5` - LR warmup epochs (default: 5)
- `--min-lr` - Minimum LR for cosine annealing (default: lr * 0.05)
- `--predict-only` - Skip training, just predict from saved model
- `--no-push` - Train but don't push prediction to dashboard
- `--min-confidence 0.40` - Minimum confidence to push trade signal (default: 0.40)
- `--min-edge 0.10` - Minimum edge to push trade signal (default: 0.10)
- `--checkpoint-interval 25` - Pause every N epochs to show results and wait for user to continue or stop (default: 25, use 0 to disable)
- `--stf-only` - A/B test mode: train with STF (47) features only, no HTF context. Use for baseline comparison

Training improvements (v3):
- LR schedule: 5-epoch linear warmup -> cosine annealing to eta_min (lr * 0.05)
- Early stopping patience: 50 epochs (was 30), min_epochs: 40
- Dual checkpoint saving: `best_loss.pt` (lowest val loss) + `best_trading.pt` (best trading score)
- Trading score: expectancy + 0.1*sharpe + 0.02*log(profit_factor), requires >= 150 trades
- Epoch-level scheduler stepping (was per-batch), current LR logged each epoch

Training improvements (v4 - classification focus):
- CRITICAL FIX: Disabled all auxiliary heads (mu, sigma, quantile, vol_state) during training
  - Previously these heads had combined lambda=1.0, meaning classification only got 50% of gradient signal
  - Now classification gets 100% of learning capacity, dramatically improving directional accuracy
  - Auxiliary heads still exist in the model for inference (prediction output uses all 5 heads)
- Focal Loss enabled by default (gamma=2.0) with class weight caps
- Per-class accuracy (SHORT/HOLD/LONG) logged every epoch for training visibility
- CLI flags: --focal-loss/--no-focal-loss, --focal-gamma, --class-weight-cap
- Monitoring sweep now uses softmax probability as confidence (not mu/sigma which are untrained)
  - Thresholds: 35%-70% (above 33% random baseline for 3-class)
  - Removed spread gate and move gate that depended on untrained auxiliary heads
  - Trading sweep output is compact tabular format with Trades/Expect/WinRate/Sharpe/PF columns
- Per-epoch log format: `Epoch N | Loss T:X V:X | Acc:X S:X H:X L:X | Pred S:X H:X L:X | LR:X`
  - Shows both per-class accuracy (recall) AND prediction distribution every epoch
  - Diagnostic logging only appears when relevant (gradient norms > 5, active auxiliary heads)
- Checkpoint display includes prediction distribution and full trading metrics (PF, avg win/loss)

Position sizing: ATR-based with 2% account risk per trade, scaled by confidence/edge, hard capped at 0.5-5.0% of account. Trade signals below confidence/edge thresholds are automatically downgraded to HOLD.

Progressive head enablement flags (add incrementally via main.py train, NOT quick_start.py):
1. `--enable-quantile` - Quantile head (PinballLoss, λ=0.3, output clamped ±0.1)
2. `--enable-vol-state` - Volatility state classification (CrossEntropy, λ=0.2, 3-class)
3. `--enable-mu` - Expected return regression (HuberLoss, λ=0.3, clamped ±0.1)
4. `--enable-sigma` - Uncertainty estimation (GaussianNLLLoss, λ=0.2, most unstable - enable last)

Quick start loss tuning flags:
- `--focal-loss` / `--no-focal-loss` - Enable/disable Focal Loss (default: enabled)
- `--focal-gamma 2.0` - Focal focusing parameter (default 2.0)
- `--class-weight-cap 10.0` - Max class weight multiplier (default 10.0)

Disabled/removed features:
- LSTM/Transformer architectures (caused gradient explosions)
- Label smoothing (hurt imbalanced classification)
- Prior bias initialization (caused model collapse)
- Cross-asset features (can't compute at inference with BTC-only data)
- Mixed precision FP16 (caused NaN with class weights)

### Multi-Head Dashboard Integration
The `multihead_predictions` table stores full 5-head prediction outputs from the GPU trainer. API endpoints:
- `POST /api/gpu/push-prediction` - Receives predictions from local GPU trainer with validation (action must be LONG/SHORT/HOLD, confidence 0-1)
- `GET /api/gpu/multihead/latest` - Returns latest prediction normalized for the dashboard (5-minute staleness threshold)
- `GET /api/gpu/multihead/history` - Returns prediction history normalized to frontend shape
- `GET /api/gpu/multihead/current` - Real-time prediction via GPU trainer with auto-save to DB

The `MultiheadSignalCard` component (`client/src/components/multihead-signal-card.tsx`) displays all 5 heads unified: direction probabilities, quantile spread visualization (q10-q90), vol state regime badge (Contraction/Neutral/Expansion), mu (expected return), sigma (uncertainty level), edge, and derived trade levels (entry/SL/TP). It is integrated into both Signal tab (right column) and Neural Network tab (with prediction history).

### System Design Choices
Data is managed with Drizzle ORM for PostgreSQL, using Zod for type-safe validation. Live sentiment data is separated from historical price/volume data, and all learning states are persisted. The client is bundled by Vite, and the server by esbuild. A centralized timeframe configuration ensures consistency. A runtime diagnostic system provides health endpoints for the GPU trainer and detailed UI console logging. The GPU training API includes endpoints for starting training, checking status, and daily retraining. Gradient clipping and learning rate adjustments are implemented for training stability. A data diagnostics system audits features and labels, while a stable `SimpleMLP` model serves as a baseline for complex model development. The system supports progressive re-enablement of multi-head models to identify sources of instability.

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