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
The system incorporates a regression-based signal system generating comprehensive signals including action, confidence, expected_move, uncertainty, and position sizing. It features regime detection with a Mixture-of-Experts (MoE) model, identifying market states like TRENDING or MEAN_REVERTING. GPU-accelerated training supports multiple deep learning architectures (Transformers, LSTMs, CNNs) with dynamic take-profit targets and failure stop detection. A multi-head model architecture uses five distinct output heads (Classification, Regression, Quantile, VolState, Acceleration) with combined loss functions. A flow forecast system provides regime-conditioned quantile path projections. Advanced labeling addresses class imbalance with a 3-stage HOLD fix and cost-aware labeling. Model management includes walk-forward weight saving, feature version locking, prediction drift monitoring, and label metadata tracking. A professional ensemble predictor combines multiple model predictions with confidence-based voting. A robust training, monitoring, and policy architecture separates model training from live execution policy selection.

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