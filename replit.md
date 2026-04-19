# Neural Terminal — AI Trading Dashboard

## Overview
Neural Terminal is an institutional-grade, GPU-accelerated AI trading system designed for multi-asset crypto futures. It acts as a comprehensive trading terminal, providing real-time signals, managing both paper and live trading operations, and tracking performance. The system integrates advanced neural networks (v5, v6, and v7 iterations) with a web application for advanced analytics and automated trading capabilities. The project's ambition is to deliver a cutting-edge solution for AI-driven crypto futures trading, focusing on real-time data ingestion, sophisticated signal generation, and robust trade execution.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend (9-Page Trading Terminal)
The frontend is built with React, TypeScript, and Vite, utilizing `shadcn/ui` (Radix UI, Tailwind CSS) for UI components and Recharts for data visualization. It features a dark navy theme with neon accents. Key pages include Command Center, Live Trading, Paper Trading, Analytics, and various monitoring and history views.

### Backend (Node.js + Express + TypeScript)
The backend provides API routes and services to support the frontend and interact with external systems. It manages data retrieval, handles cycle logs and executed trades, and integrates with a local GPU trainer. Key features include a Paper Trading Engine, Bitget and Bybit client integrations for live trading, Market Regime/Chop Protection, and an Order Flow Pipeline for signal validation. A WebSocket server enables real-time event streaming.

### Symbol Configuration
All 20 trading symbols, including precision settings, are defined in `shared/symbols.ts`.

### Neural Network Models (V5, V6, V7)
The system incorporates multiple iterations of neural networks:
-   **V5 Forecaster:** Runs on a local GPU, features multi-head output for predictions based on 95 features, includes a Composite Scoring Engine, per-symbol edge learning, and a live feature pipeline.
-   **V6 Forecaster:** A next-generation model utilizing Causal Conv1D, Positional Encoding, Transformer Blocks, Mixture-of-Experts, and 6 output heads. It incorporates temporal context, MoE routing, feature masking, auxiliary self-supervised loss, and a confidence calibration head.
-   **V7 Data Ingestion:** Hydrates augmented datasets from `data.binance.vision`, processing klines, funding, and open interest data.
-   **V7 Truth-Discovery Audit & Payoff Geometry:** Performs walk-forward signal-learnability audits and payoff geometry sweeps to validate tradeable signals, assess cost sensitivity, and optimize exit strategies. Initial findings have led to refined strategies focusing on selectivity, absence of hard stops, and careful consideration of trade execution costs.
-   **V7 Path A Paper Engine** (`server/paper/v7_path_a.ts`): Live paper-trading implementation of the GO-verdict V7 Path A strategy. Tradeable book = ADA/XRP/AVAX (equal notional), probationary bucket = SOL, ETH disabled. Selectivity = top 0.5% of |returnH2| via rolling 30-day quantile per symbol (50-sample warm-up). Fixed 120-min market exit, no stop/TP/trail. Per-symbol kill switch at cumulative net < -1500 bps with manual-resume only. Hooks `updateGPUPrediction()`; uses 15m candle close as entry/exit price. State persisted to `.local/v7_path_a_state.json`. Uses existing `paperPositions` table with `source` field segregating tradeable (`v7_path_a`) and probationary (`v7_path_a_prob`). Default DISABLED on boot — operator enables via `POST /api/v7/enable` after Bybit cost verification clears. Routes: `/api/v7/{state,performance,enable,disable,notional,resume/:symbol}`. Dashboard panel rendered at the top of the Paper Trading page (`client/src/components/V7PathAPanel.tsx`) with per-symbol buffer/threshold/cum-bps tiles, kill-switch resume controls, and live-vs-back-test divergence band (±5 bps around back-test refs: tradeable +24.01 bps net@6, probationary +21.37 bps net@6).

## External Dependencies

### Database
-   PostgreSQL (via Drizzle ORM)

### UI Framework
-   Radix UI, Lucide React, class-variance-authority, Recharts

### Data & Validation
-   Zod, drizzle-zod, date-fns

### AI / Machine Learning
-   OpenAI

### Market Data
-   Binance Vision API
-   Bybit V5 REST API