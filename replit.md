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