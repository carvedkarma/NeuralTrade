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