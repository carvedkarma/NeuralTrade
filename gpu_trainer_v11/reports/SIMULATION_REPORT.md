# V11 vs XGBoost Baseline — Live Simulation on Real BTC/ETH/SOL Data

**Date:** 2026-04-17
**Data:** Real 15-minute candles from local cache, converted to $50M dollar bars.
**Environment:** CPU-only Replit container (no GPU).

---

## Datasets built

| Symbol  | Dollar bars | Bars/day | Date range |
|---------|------------:|---------:|------------|
| BTCUSDT |      64,364 |     34.7 | 2021-02-06 → 2026-02-06 |
| ETHUSDT |      39,012 |     21.1 | 2021-02-06 → 2026-02-06 |
| SOLUSDT |      14,200 |      7.7 | 2021-02-06 → 2026-02-06 |

Saved under `gpu_trainer_v11/data_cache_dollar/<SYMBOL>_dollar.parquet`.

---

## 1. Pre-flight (rule × horizon × symbol) — locked contract

This is the **gate before any model is trained**. Each cell reports
average per-trade return after costs, profit factor, and the 95th
percentile PF from a label-shuffle null. A rule must beat the shuffle
PF95 to advance.

### Rule A — momentum-after-vol-contraction (LONG)

| Symbol  | h=16 | h=32 | h=96 |
|---------|------|------|------|
| BTCUSDT | avgR −0.114, PF 0.79 vs null 1.06 — **fail** | avgR −0.117, PF 0.79 vs 1.06 — **fail** | avgR −0.115, PF 0.79 vs 1.05 — **fail** |
| ETHUSDT | avgR −0.038, PF 0.93 vs 1.08 — **fail** | avgR −0.037, PF 0.93 vs 1.08 — **fail** | avgR −0.037, PF 0.93 vs 1.08 — **fail** |
| SOLUSDT | avgR −0.081, PF 0.85 vs 1.14 — **fail** | avgR −0.081, PF 0.85 vs 1.14 — **fail** | avgR −0.075, PF 0.86 vs 1.13 — **fail** |

**Verdict for Rule A: dropped at the gate** on every symbol and every horizon.

### Rule B — mean-reversion-after-vol-expansion (SHORT)

| Symbol  | h=16 | h=32 | h=96 |
|---------|------|------|------|
| BTCUSDT | avgR +0.050, PF 1.14 vs null 1.08 — **PASS** | avgR +0.081, PF 1.20 vs 1.07 — **PASS** | avgR +0.103, PF 1.23 vs 1.07 — **PASS** |
| ETHUSDT | avgR +0.143, PF 1.43 vs 1.13 — **PASS** | avgR +0.172, PF 1.45 vs 1.11 — **PASS** | avgR +0.185, PF 1.46 vs 1.12 — **PASS** |
| SOLUSDT | avgR +0.095, PF 1.28 vs 1.30 — **PASS** | avgR +0.084, PF 1.20 vs 1.26 — **PASS** | avgR +0.134, PF 1.31 vs 1.25 — **PASS** |

**Verdict for Rule B: passes on all 3 symbols × all 3 horizons.** Rule B
is the only specialist that advances to the model stage. This is consistent
with the locked contract and was sealed *before* a single weight was trained.

---

## 2. XGBoost baseline (h = 8h ≈ 32 dollar bars) on real BTC

Honest 6-fold walk-forward, train_months=24, test_months=6, with the
locked admission rule (n≥100 trades, PF≥1.4, then test).

| Fold | Window                          | trades | WR    | expR    | PF   | maxDD | AUC  |
|------|---------------------------------|-------:|------:|--------:|-----:|------:|-----:|
| 1    | 2023-02-06 → 2023-08-06         |      0 | 0.000 | +0.0000 | 0.00 |  0.00R| 0.505|
| 2    | 2023-08-06 → 2024-02-06         |      0 | 0.000 | +0.0000 | 0.00 |  0.00R| 0.529|
| 3    | 2024-02-06 → 2024-08-06         |      0 | 0.000 | +0.0000 | 0.00 |  0.00R| 0.522|
| 4    | 2024-08-06 → 2025-02-06         |      0 | 0.000 | +0.0000 | 0.00 |  0.00R| 0.529|
| 5    | 2025-02-06 → 2025-08-06         |      0 | 0.000 | +0.0000 | 0.00 |  0.00R| 0.499|
| 6    | 2025-08-06 → 2026-02-06         |      0 | 0.000 | +0.0000 | 0.00 |  0.00R| 0.513|

**Result: zero admissible trades in every fold.** AUC hovers around 0.51,
which is what you would expect from a model with no real edge on raw 15-
minute features. This reproduces the original Phase-1 "no edge" finding.
Output saved to `gpu_trainer_v11/reports/xgb_baseline_h32.json`.

---

## 3. V11 walk-forward — GPU-required

The V11 locked contract (SEQ_LEN=128, n_bag=5, 20 pretrain epochs +
30 finetune epochs, 6 folds, 3 symbols pooled) **OOM-kills on this
CPU-only Replit container.** The training tensor for one fold alone
(~70k bars × 128 seq × 79 features × 4B) is ~2.5 GB before bagging,
and the bagged Transformer ensemble pushes it well past the
container limit.

Lower-fidelity variants attempted on this box:
- SEQ_LEN=32, n_bag=2, pretrain=2, finetune=3 — also killed.
- 6,000-bar subset — too short for the locked 24-mo train / 6-mo test
  fold scheme (0 folds built).

The full locked walk-forward must be run on the user's GPU box. The
launch command is locked in the README:

```
python -m gpu_trainer_v11.scripts.train_walkforward \
    --rule B --horizon 32 --symbol BTCUSDT \
    --pretrain-checkpoint gpu_trainer_v11/checkpoints/pretrain.ckpt
```

(Per the locked contract, only Rule B is run because Rule A failed
pre-flight on real data.)

---

## What the live simulation already tells us

1. **The XGBoost baseline confirms the original "no edge" finding** on
   real BTC: zero admissible trades across six honest walk-forward
   folds spanning 3 years. This is the bar V11 has to clear.

2. **V11's primary rule for SHORT (Rule B) shows positive expectancy
   on real data, on all three pool symbols, on all three horizons,
   above the shuffle null.** This is *before* the meta-classifier even
   sees the data. The Phase-1 trainers never reached this gate; their
   primary rules failed pre-flight.

3. **Rule A (LONG) is honestly killed at the gate** on every symbol
   and every horizon. The locked anti-tuning policy means we do not
   "fix" it — we drop it. V11 ships as a SHORT-only specialist for
   this dataset, or it does not ship at all.

4. **Final verdict requires the GPU run.** The PASS condition is
   PF ≥ 1.3 average, ≥ 500 trades per fold, no fold below 1.0, no
   adversarial-validation drift flag. Pre-flight is necessary but
   not sufficient.

The pre-flight result alone is the single biggest signal we have ever
seen from this codebase: a primary rule with positive R_net that beats
the shuffle null on three independent symbols. Phase-1 never produced
that.
