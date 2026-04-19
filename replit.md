# Neural Terminal — AI Trading Dashboard

## Overview
Neural Terminal is an institutional-grade, GPU-accelerated AI trading system designed for multi-asset crypto futures. It acts as a comprehensive trading terminal, providing real-time signals, managing both paper and live trading operations, and tracking performance. The system integrates a v5 neural network, with the web application serving as the primary interface for advanced analytics and automated trading capabilities. The project's ambition is to deliver a cutting-edge solution for AI-driven crypto futures trading.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend (9-Page Trading Terminal)
The frontend is built with React, TypeScript, and Vite, leveraging `shadcn/ui` (Radix UI, Tailwind CSS) for UI components and Recharts for data visualization. It features a dark navy theme with neon accents. Key pages include Command Center, Live Trading, Paper Trading, Analytics, Training Monitor, Bitget Trading, Neural Monitor, Trade History, and Settings.

### Backend (Node.js + Express + TypeScript)
The backend provides API routes and services to support the frontend and interact with external systems. It manages data retrieval, handles cycle logs and executed trades, and integrates with a local GPU trainer. Key features include a Paper Trading Engine, Bitget and Bybit client integrations for live trading, Market Regime/Chop Protection, and an Order Flow Pipeline for signal validation. A WebSocket server enables real-time event streaming.

### Symbol Configuration
All 20 trading symbols, including precision settings, are defined in `shared/symbols.ts`.

### V5 Neural Network (`V5Forecaster`)
The core v5 neural network runs on a local GPU, featuring multi-head output for various predictions based on 95 features. It includes a V5 Composite Scoring Engine, per-symbol edge learning, and a live feature pipeline for real-time data integration. The network has undergone continuous improvements for biased training labels, confidence gating, and dynamic risk management.

### V6 Neural Network (`V6Forecaster`)
The next-generation `V6Forecaster` utilizes a Causal Conv1D, Positional Encoding, Transformer Blocks, Mixture-of-Experts, and 6 output heads. It incorporates temporal context, MoE routing, feature masking, auxiliary self-supervised loss, and a confidence calibration head for signal gating. It supports flexible scaler loading and V6-specific training arguments.

### V7 Augmented Dataset Ingest (`gpu_trainer/data_ingest/`)
Hydrates the V7 augmented dataset from `data.binance.vision` (live `fapi.binance.com` is geo-blocked from Replit; the public CDN is reachable). Modules: `binance_api.py` (streaming ZIP iterators), `klines_backfill.py`, `flow_features.py` (1m kline aggregator → `flow_features_15m`), `funding_backfill.py`, `oi_backfill.py`, `cli.py` (`python -m gpu_trainer.data_ingest.cli {funding|klines|oi|flow}`), `run_all.sh` (run inside a Workflow so the subprocess survives shell teardown).

### V7 Truth-Discovery Audit (`gpu_trainer/eval/v7_signal_audit_augmented.py`)
Walk-forward signal-learnability audit on the augmented dataset. 7 target variants (ret_15m, ret_60m, ret_240m, sign_60m, ret_60m_volnorm, ret_60m_quintile, mfe_minus_mae_4) × 2 simple models (Ridge, HGBR) × 5 walk-forward folds (24m train / 6m test). **16-bar embargo** between train end and test start (>= longest forward target horizon, prevents label-overlap leakage at fold boundaries). Ridge uses median-impute + clip + standardize; HGBR is NaN-native (so symbols without OI history audit cleanly). Verdict logic is fail-closed on the cost gate: a positive top-decile mean net per-trade return after 8 bps round-trip cost is the minimum evidence that a discovered IC is *tradeable*. Outputs `.local/reports/v7_truth_discovery_augmented.{md,json}`.

### V7 Path A — Payoff-Geometry Sweep (2026-04-18) — VERDICT: PARTIAL PASS at maker fees only
Tested 33 exit geometries across 5 families on the same E2 universe (whitelist ADA/AVAX/ETH/SOL/XRP, top 2% |pred|, HGBR sign_60m, 5-fold WF, 16-bar embargo, 8,560 trades). Module: `gpu_trainer/eval/v7_payoff_geometry.py`. Cache: `.local/cache/v7_payoff_geometry/*.parquet` (per-trade entry + 16 forward bars OHLC + vol_16). Report: `.local/reports/v7_payoff_geometry.md`.
- **Strict criterion FAILS**: no variant has both Expectancy R > 0 AND mean net bps > 0 at 8 bps. Best by Exp R = stop=3×vol, hold=60m at Exp R = -0.043, net = -1.66 bps.
- **Dollar-P&L criterion PASSES at all cost levels for one geometry**: `4.time_exit | no stop, hold=120m` — book mean net per trade: **+6.36 bps @ 4bps, +4.36 @ 6bps, +2.36 @ 8bps**. Per-symbol @ 8 bps: ADA +7.11, AVAX +1.21, ETH -3.46, SOL +0.31, XRP +6.58 — **3/5 positive, 2/5 negative (ETH and SOL)**.
- **Key lever found**: REMOVING the hard stop is the largest single improvement. The 1.5×vol stop was chopping winners more than it saved on losers (4.time_exit hold=60m beats 1.stop_sweep stop=1.5×vol hold=60m by ~+2 bps net @ 8bps).
- **Worst family**: trailing stops (-10 to -21 bps net @ 8bps across all variants). Trailing in 15-min crypto noise locks in chop, not trend.
- **Symmetric TP at 1.5×vol = -3.7 bps net@8** vs the same 1.5×vol stop alone = -1.47 bps. The TP is hurting, not helping (caps the upside path that pays for the losers).
- **Asymmetric TP** (e.g. stop=1.25×vol, TP=3.0×vol, hold=240m) gets to net = -0.62 @ 8bps but never positive — the wide TP fires too rarely to dominate the cost line.
- **Path-quality early-exit at bar 1** does NOT work as intended: early-exiting trades that already moved against you at bar 1's close just locks in the small loss without giving the held branch enough upside concentration to compensate. All 6 path-gate variants negative.
- **Why R disagrees with bps**: R = return / planned_risk. Time-exit (no stop) trades use a reference 1.5×vol risk denominator, but actual losses can far exceed 1.5×vol → big-loss trades blow up the R denominator → Exp R looks worse than the dollar P&L. **For a $15k-notional, $1k/day target, dollar bps is the right metric, not R.**
- **Path B (cost ladder for `time_exit hold=120m`):**
  - 2 bps: +8.36 bps net/trade — clearly tradeable
  - 4 bps: +6.36 bps net/trade — clearly tradeable (achievable on Bybit/Bitget with maker rebates + tier-1 fees on sub-$50k notional)
  - 6 bps: +4.36 bps — tradeable (mixed maker/taker)
  - 8 bps: +2.36 bps — marginal at retail taker fees
  - 10 bps: +0.36 bps — break-even
- **Production-readiness verdict**: V7 Path A unlocks a CONDITIONAL GO for paper-trading the `time_exit hold=120m no-stop` geometry on the 5-symbol whitelist, contingent on (1) achievable round-trip cost ≤ 6 bps, (2) ETH and SOL kept on a per-symbol kill switch (negative cumulative net at any point → halt), (3) max drawdown bound at -27,488 bps × position-size scales must fit risk budget.

### V7.1 "Brilliant" — Postmortem-driven Redesign (2026-04-18)
Added `gpu_trainer/eval/v7_brilliant.py`: keeps what worked in V7.0 (tight threshold + cell filter + confidence sizing), drops what hurt (1.5×vol stops + time stop), adds three new levers (5-symbol whitelist excluding BTC/BNB; per-symbol kill switch on negative prior-fold net; low-vol guard).
- **E1 Brilliant baseline** (top 2% + filter + whitelist + fixed 60m hold + sizing): 6,511 trades, mean gross +5.92 bps, mean net @8 bps **−2.08 bps** (vs D's −2.64), Sharpe-like −1.07 (vs D's −1.98). Pass 2/5 @8bps, 2/5 @6bps, 3/5 @4bps.
- **E2 Brilliant + per-symbol kill switch**: 4,099 trades, mean gross +6.25 bps, mean net @8 bps **−1.75 bps** (best), Sharpe-like −0.74, max DD −21,439 bps. **Passes the 6 bps gate book-wide for the first time** (+0.25 bps). Pass 1/5 @8bps (SOL), 2/5 @6bps, 3/5 @4bps. The kill switch correctly axed ADA + XRP early.
- **E3 Brilliant + low-vol guard** (vol_q ≤ 2): **catastrophic failure** — gross collapses to +0.29 bps, net @8 bps −7.71 bps. **The drift edge actually lives in higher-vol regimes**, not lower; this disproves the "low vol = clean signal" intuition.
- **Verdict per locked gates (best = E2)**: still REDESIGN — 1/5 @8bps below the GO threshold. But E2 is the right config to forward to paper-trading: closer to 8 bps gate, much better Sharpe, lower DD, and SOL alone runs +5.71 bps NET @8bps (gross +13.7 bps).
- **Key learnings rewritten:** (1) edge lives in HIGH-vol regimes (E3 kills it); (2) per-symbol kill switch is a real lever (E1→E2 = +0.33 bps and ½ DD); (3) brilliant fixed-hold beats stops on 4/5 symbols but XRP got worse (−1.52 bps in D → −9.76 bps in E1) — XRP-specific path dependence; (4) the entire book-wide profitability is one symbol (SOL) carrying the rest.
- Full report: `.local/reports/v7_brilliant.md`. JSON: `.local/reports/v7_brilliant.json`.

### V7 Asymmetric Payoff Fix (2026-04-18) — VERDICT: REDESIGN
Tested 4 configs on the same 7-symbol / 59,890-trade base (HGBR `sign_60m`, 5-fold walk-forward, 16-bar embargo). _Note: results below are post-bug-fix; the first run had a `gross_fixed = fwd[sel_idx]` indexing bug (should be `fwd[global_idx]`) that misaligned A/B baselines. C/D were unaffected because the vectorized exit simulator already used `global_idx`. Verdict unchanged after fix._
- **A. Baseline (top 10%, fixed 60m, equal size, no filter):** mean net @8 bps = **−6.51 bps**; cum −390k bps; 0/7 pass 8 bps, 0/7 at 6, 1/7 at 4, 2/7 at 2.
- **B. + Cell filter (no other changes):** 40,191 trades; mean net @8 bps = **−5.89 bps** (+0.62 over A); cum DD −241k. 0/7 at 8, 1/7 at 6, 1/7 at 4, 3/7 at 2.
- **C. + Adaptive 1.5×vol stops + 30-min time-stop + P1–P5 confidence sizing (top 10%):** 37,856 trades; gross WR drops 51% → 41% (stops chop winners almost as often as losers), mean net @8 bps = **−6.09 bps** — roughly net-neutral vs B. The naive trailing-vol stop is NOT the lever.
- **D. C + tight threshold top ~2% (P5 only):** 8,306 trades; mean gross **+5.36 bps** (3.6× B's gross), mean net @8 bps = **−2.64 bps**. **5/7 symbols pass 4 bps, 3/7 pass 6 bps, 2/7 (ADA + SOL) pass 8 bps.** Cum DD shrinks from −390k → −22.6k (94% reduction).
- **Verdict per locked gates:** GO needs ≥3/7 at 8 bps (only 2/7 — fail). CONDITIONAL GO needs ≥4/7 at 6 bps (only 3/7 — fail). REDESIGN needs ≥4/7 at 4 bps (5/7 — **PASS**). The full V7 5-layer organism (Task #104) is **NOT** authorised; a narrow-scope redesign on the cheapest-to-trade slice is.
- **What actually fixed the asymmetry:** tightening the threshold (10% → 2%) drove almost the entire improvement. Stops + sizing alone (B → C) added nothing material. The edge is concentrated in the highest-confidence tail (top 2% of |pred|), and the cheap symbols to trade are SOL, ADA, XRP, ETH, AVAX. BNB and BTC remain unprofitable at 8 bps even at the tight threshold.
- **Caveats called out by code review:** stop-fill assumes execution exactly at stop level (no adverse intrabar slippage); ablation between "stops alone" vs "sizing alone" not separately tested (C bundles both); 5+ years includes 2021 high-fee/high-vol regime that may inflate D's edge.
- Full report: `.local/reports/v7_payoff_fix.md`. Cached per-symbol trades: `.local/cache/v7_payoff_fix/*.parquet`.

### V7 Audit Result (2026-04-18) — VERDICT: PIVOT
Ran on 7 full-history symbols (BTC/ETH/BNB/SOL/ADA/AVAX/XRP), each ~180k 15-min bars from 2021-02 → 2026-04, 30 features (price/vol + flow_features_15m + funding + OI where available).
- **Signal IS present and is not a leakage artefact.** Adding the 16-bar embargo only moved best cross-fold mean IC 0.0712 → 0.0695 (a 2% drop, not the collapse leakage would produce). `sign_60m` (Ridge): cross-symbol mean IC `+0.055`, sign positive on every fold for all 7 symbols. `ret_60m_quintile` (Ridge): mean IC `+0.052`, all-folds-positive on 7/7.
- **Cost gate fails uniformly.** Zero (symbol × target × model) cells produce a positive top-decile mean net per-trade return after 8 bps round-trip. Best is `-1.3 bps`. The naive top-decile-take-and-hold-60m policy loses money everywhere.
- **PIVOT, per locked rules.** Best IC ≥ 0.05 but `best_td_net ≤ 0` ⇒ PIVOT. This audit, with this cost assumption and this naïve trading proxy, did *not* refute the null that the discovered signal is non-tradeable. The V7 5-layer organism build (Task #104) is **NOT authorised** on this evidence alone.
- **What would unlock GO before building the organism.** Three follow-ups, runnable on the same dataset: (1) cost-sensitivity sweep at 2/4/6 bps round-trip, (2) holding-horizon optimiser per (symbol × target) instead of fixed 60 min, (3) ex-ante threshold tuner using prior-fold predicted distribution. Any of these flipping the cost gate on ≥3 symbols upgrades to REDESIGN.
- Full report: `.local/reports/v7_truth_discovery_augmented.md`. Raw fold-level data: `.local/reports/v7_truth_discovery_augmented.json`.

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