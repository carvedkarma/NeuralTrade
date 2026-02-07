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
The system incorporates a regression-based signal system generating comprehensive signals including action, confidence, expected_move, uncertainty, and position sizing. It features regime detection with a Mixture-of-Experts (MoE) model, identifying market states like TRENDING or MEAN_REVERTING. GPU-accelerated training uses the stable EnhancedMultiHeadMLP architecture with [512, 256, 128, 64] residual blocks and progressive head enablement. A multi-head model architecture uses five distinct output heads (Classification, Quantile, VolState, Mu, Sigma) with combined loss functions. A flow forecast system provides regime-conditioned quantile path projections. Advanced labeling addresses class imbalance with Focal Loss (gamma=2.0) and configurable class weight caps. Model management includes walk-forward weight saving, feature version locking, prediction drift monitoring, and label metadata tracking. A professional ensemble predictor combines multiple model predictions with confidence-based voting. A robust training, monitoring, and policy architecture separates model training from live execution policy selection.

### GPU Training CLI Reference
Current stable model: `enhanced_mlp` (EnhancedMultiHeadMLP)
- Gradient norms: 0.35-0.63 (well under 10 threshold)
- Prediction distribution with focal loss: SHORT ~13%, HOLD ~47%, LONG ~40%
- Training data: 5 years of BTCUSDT 15m candles (~175k target, backfilled from Binance)
- Quick start: `python quick_start.py --url https://APP.replit.app` (200 epochs default)

Quick start training flags:
- `--epochs 200` - Training epochs (default: 200)
- `--batch-size 64` - Batch size (default: 64)
- `--lr 0.0001` - Learning rate (default: 0.0001)
- `--predict-only` - Skip training, just predict from saved model
- `--no-push` - Train but don't push prediction to dashboard
- `--min-confidence 0.40` - Minimum confidence to push trade signal (default: 0.40)
- `--min-edge 0.10` - Minimum edge to push trade signal (default: 0.10)

Position sizing: ATR-based with 2% account risk per trade, scaled by confidence/edge, hard capped at 0.5-5.0% of account. Trade signals below confidence/edge thresholds are automatically downgraded to HOLD.

Progressive head enablement flags (add incrementally):
1. `--enable-quantile` - Quantile head (PinballLoss, λ=0.3, output clamped ±0.1)
2. `--enable-vol-state` - Volatility state classification (CrossEntropy, λ=0.2, 3-class)
3. `--enable-mu` - Expected return regression (HuberLoss, λ=0.3, clamped ±0.1)
4. `--enable-sigma` - Uncertainty estimation (GaussianNLLLoss, λ=0.2, most unstable - enable last)

Loss tuning flags:
- `--focal-loss` - Enable Focal Loss (down-weights easy HOLD predictions)
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