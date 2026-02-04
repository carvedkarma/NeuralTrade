# BTC Futures Signal Dashboard

## Overview
This project is an institutional-grade, AI-driven dashboard designed to generate sophisticated BTCUSDT futures trading signals. It integrates machine learning, real-time market data, and sentiment analysis to provide AI-powered trade plans. The system features a continuous learning loop with ongoing data refreshing and model retraining, aiming to deliver a robust, selective trading system that leverages advanced AI and comprehensive market insights for high-confidence trading opportunities. The project's vision is to deliver a cutting-edge platform for futures trading, capitalizing on market potential through AI-driven precision and continuous adaptation.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend
- **Framework**: React with TypeScript (Vite).
- **UI Components**: shadcn/ui (Radix UI, Tailwind CSS).
- **Charts**: Recharts.
- **Navigation**: Tabbed interface including Overview, Signal, Neural Network, and Paper Trading.

### Backend
- **Runtime**: Node.js with Express.js (TypeScript, ESM).
- **API Pattern**: RESTful.
- **AI Integration**: OpenAI via Replit AI Integrations.
- **Market Data**: Binance Vision API, with CoinGecko and CryptoCompare fallbacks, augmented by a Replit-hosted data proxy.
- **GPU Trainer Communication**: Bi-directional communication with a local GPU trainer via dedicated API endpoints.

### Data Layer
- **ORM**: Drizzle ORM for PostgreSQL.
- **Schema**: Zod for type-safe validation.
- **Data Separation**: Live sentiment data is separated from historical price/volume data.
- **Persistence**: All learning states, including pattern clusters and social media statistics, are persisted in a database.

### Machine Learning and Signal Generation
- **Regression-Based Signal System**: Generates comprehensive signals including action, confidence, expected_move, uncertainty, and position sizing. Targets μ (4h forward return), σ (uncertainty), P(move>cost), quantiles, MFE/MAE.
- **Regime Detection & Mixture-of-Experts (MoE)**: Identifies market regimes (TRENDING, MEAN_REVERTING, HIGH_VOLATILITY, LOW_VOLATILITY, TRANSITION, UNKNOWN) and employs a gating network with specialized expert models.
- **GPU Neural Network Training**: Supports 12+ deep learning architectures (Transformers, LSTMs, CNNs, VAEs, GNNs, Ensembles) with local GPU training. Implements dynamic take-profit targets and failure stop detection.
- **Multi-Head Model Architecture**: Uses five distinct output heads: Classification, Regression, Quantile, VolState, and Acceleration, with combined loss functions.
- **Flow Forecast System**: Replaces triangle probability cones with regime-conditioned quantile path projections (q10, q50, q90) for a 16-bar horizon.
- **Advanced Labeling and Prediction**: Addresses HOLD-heavy label distribution with a 3-stage HOLD fix, cost-aware labeling, Gaussian NLL with Log-Sigma for uncertainty, and constrained candle parameterization.
- **Model Management and Monitoring**: Includes walk-forward weight saving, feature version locking, prediction drift monitoring (PSI, KL Divergence, ECE), and label metadata tracking.
- **Feature Schema Enforcement**: Ensures strict consistency of features between training and inference using `.features.json` files and `FeatureValidator.enforce_schema`.
- **Professional Ensemble Predictor**: Combines predictions from multiple models (Transformer, TFT, LSTM, CNN) with confidence-based voting, regime, and risk gating.
- **Training, Monitoring, and Policy Architecture**: Separates model training (weights learning) from policy selection (live execution rules). Post-training, a `PolicySelector` performs sequential out-of-sample evaluation to select and save a frozen `execution_policy.json` for live trading.

### Build System
- **Client Build**: Vite bundles React app to `dist/public`.
- **Server Build**: esbuild bundles server to `dist/index.cjs`.

### Data Architecture
- **Dashboard**: Uses 35K+ database candles + 1 live Binance candle.
- **GPU Trainer**: Uses 140K+ parquet candles for training (separate data flow).
- **Feature Pipeline Alignment**: Explicit STF/MTF mode routing with auto-detection.
- **Centralized Timeframe Configuration**: Uses `gpu_trainer/config/timeframe_config.py` as a single source of truth.

### Runtime Diagnostic System
- **GPU Trainer /health Endpoint**: Provides comprehensive capability information and disconnect reasons.
- **UI Console Logging**: Logs `[FLOW FORECAST UI]` with forecastMode, volState, acceleration, and path lengths.

### GPU Training API
- **`/training/start`**: Triggers MultiHeadTrainer training.
- **`/training/status`**: Provides real-time training progress, epoch_history, ETA, per-head losses, and health warnings.
- **`/api/retrain/daily`**: Daily retraining endpoint.
- **Training Configuration**: Uses inverse frequency class weights, 80/20 train/val split, 100-step sequences.

### Critical Bug Fixes (February 2026)
Six critical bugs were identified and fixed that were causing mode collapse and uniform predictions:

1. **Bug #1-4: Class Weight Propagation** - OHEMLoss now has `set_alpha()` pass-through method to forward alpha weights to underlying FocalLoss. Previously, class weights were never applied when OHEM+Focal was enabled.

2. **Bug #5: Confidence Penalty Direction** - Fixed ConfidencePenaltyLoss to return POSITIVE penalty for overconfidence. Previously returned negative entropy which REDUCED loss for uniform predictions, actively encouraging mode collapse.

3. **Bug #6: Feature Scaling** - Added RobustScaler to training path. Previously raw features (RSI 0-100, MACD arbitrary, returns -0.1 to 0.1) caused gradient instability.

**Validation Logging**: Training now logs verification that FocalLoss alpha weights were successfully applied through OHEM wrapper.

### Training Stability Fixes (February 2026)
Following persistent mode collapse (100% HOLD predictions) despite bug fixes, aggressive stability measures were implemented:

**Scheduler Changes:**
- OneCycleLR max_lr reduced from 3x to 1.2x (was causing gradient explosion)
- Base LR clamped to max 1e-4 for stability
- Added 10% warmup period with cosine annealing

**Loss Configuration - Stability Mode:**
- All aggressive classification tricks DISABLED: use_focal_loss=False, use_ohem=False, use_confidence_penalty=False
- Loss weights reduced: lambda_class=1.0 (was 3.0), other heads=0.1-0.2 (was 0.3-0.5)
- Prior bias initialization DISABLED (was causing early collapse)

**Stability Guardrails:**
- HOLD Collapse Guardrail: Aborts training if HOLD predictions > 95% for 3 consecutive epochs
- Auto LR Reduction: If gradient norm > 20 for 3 consecutive epochs, scheduler is reinitialized with 50% reduced max_lr

**Re-enablement Strategy:** Once stable training is achieved (no mode collapse, gradient norms < 20), re-enable features ONE AT A TIME in this order: (1) Focal Loss, (2) OHEM, (3) Confidence Penalty.

### Final Stabilization - Survival Mode (February 2026)
After continued gradient explosions at epoch 14+ despite all previous fixes, the system was reduced to bare-minimum training:

**Disabled Auxiliary Heads:**
- lambda_quantile=0.0, lambda_trading=0.0, lambda_candle=0.0, lambda_vol_state=0.0, lambda_acceleration=0.0
- Only 3 heads active: Classification (1.0), Regression μ (0.3), Regression σ (0.2)
- Total enabled heads: 3/8

**Additional Stability Measures:**
- Gradient clip reduced from 1.0 to 0.7
- Stability proof logging shows enabled/disabled heads at training start

### Aggressive Stability Fixes (February 2026)
After continued gradient explosions, the following aggressive measures were implemented:

**1. Sigma Head Clamping:**
- RegressionHead.forward() now clamps log_sigma to [-8, 2] before returning
- exp(-8) ≈ 0.00034 (min σ), exp(2) ≈ 7.4 (max σ)
- Prevents extreme uncertainty values from destabilizing gradients

**2. Scheduler Replacement:**
- OneCycleLR completely REMOVED (was causing cyclic gradient spikes)
- Replaced with SequentialLR: Linear warmup (10% steps) + CosineAnnealingLR decay
- No cyclic LR behavior - monotonic decay after warmup

**3. Lower Learning Rate:**
- Base LR clamped to 5e-5 (was 1e-4)
- Final LR = base_lr / 100 = 5e-7

**4. Enhanced Diagnostics:**
- Per-loss means logged each epoch (class, mu, sigma, quantile, etc.)
- On gradient explosion (>20): logs per-layer norms (trunk/classifier/regression)
- Top-5 parameters by gradient norm printed on explosion

**5. Guardrail 3/3 Reset (Optimizer Momentum Reset):**
- When gradient norm exceeds 20 for 3 consecutive epochs:
  - AdamW optimizer RECREATED (clears exp_avg/exp_avg_sq momentum states)
  - Scheduler reinitialized with 50% reduced LR
  - This prevents accumulated momentum from causing continued explosions

**Success Criteria:** After epoch 10, avg_pre_grad_norm < 10 and no guardrail resets

**Live Prediction Display:**
- Neural Network tab shows real-time SHORT/HOLD/LONG prediction distribution
- Color-coded bars with actual counts and percentages
- Mode collapse warning when HOLD > 95%
- Gradient norm display (green < 10, amber > 10)

**TrainingHealthMonitor Enhancements:**
- Now tracks actual prediction counts in `class_counts_history` (not just percentages)
- Provides real argmax counts from model predictions per epoch

## External Dependencies

### Database
- PostgreSQL (via `DATABASE_URL`).
- Drizzle Kit for schema migrations.

### UI Framework
- Radix UI primitives.
- Lucide React for icons.
- class-variance-authority for component variants.

### Data & Validation
- Zod for runtime schema validation.
- drizzle-zod for database schema to Zod type generation.
- date-fns for date formatting.