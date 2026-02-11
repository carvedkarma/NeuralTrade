# BTC Futures Signal Dashboard

## Overview
This project is an institutional-grade, AI-driven dashboard designed to generate sophisticated BTCUSDT futures trading signals. It leverages machine learning, real-time market data, and sentiment analysis to provide AI-powered trade plans. The system aims for continuous learning and adaptation, delivering a robust and selective trading solution by utilizing advanced AI and comprehensive market insights to identify high-confidence trading opportunities. The ultimate vision is to create a cutting-edge platform for futures trading, maximizing market potential through AI-driven precision and continuous adaptation.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### UI/UX Decisions
The frontend is built using React and TypeScript with Vite, featuring a modern UI. It utilizes shadcn/ui (Radix UI, Tailwind CSS) for components and Recharts for data visualization, with navigation managed via a tabbed interface. The Pro Dashboard (`/pro`) provides a premium, real-time analytics experience across six dedicated tabs. An R/$ toggle allows switching between R-units and USD values for financial metrics.

### Technical Implementations
The backend is developed with Node.js and Express.js (TypeScript, ESM), following a RESTful API pattern. AI integration is handled via OpenAI. Market data is sourced primarily from Binance Vision API, with fallbacks to CoinGecko and CryptoCompare, supplemented by a Replit-hosted data proxy. Bi-directional communication with a local GPU trainer occurs via dedicated API endpoints. A WebSocket server handles real-time event streaming for the Pro Dashboard. The system supports a dual-policy engine (CORE and FLOW) for trading, enabling increased trade frequency with distinct risk profiles and dynamic, percentile-based thresholds. A live learning system supports scheduled retraining and safe model promotion, including per-symbol model management and stringent promotion gates. A money management system converts R-based metrics to USD values based on configurable account equity and risk settings.

### Feature Specifications
The core AI model (v3.3.0) predicts the quality of entering a trend-following trade (binary 0/1) based on 63 features (47 Short-TimeFrame, 10 HTF, 3 Funding, 3 Open Interest) on a 15-minute timeframe with a 24-bar horizon. Training utilizes an EnhancedMultiHeadMLP architecture with HTF-gated Triple Barrier labeling (v3.1.0) and optimizes for PR-AUC. Inference applies a probability threshold and HTF alignment gates to generate LONG/SHORT/HOLD signals, with position sizing based on ATR, dynamic account risk, and confidence. A policy auto-tuner (v3.4.x) optimizes trade frequency by evaluating configurations across multiple regimes and selecting the best policy based on net profitability, positive expectancy, and target trades per day. A sophisticated cost model (v3.3.2) accounts for fees, spread, and slippage in R-unit calculations. Safety kill-switches, including per-symbol daily drawdown caps and performance-based disabling of the FLOW policy, are implemented.

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