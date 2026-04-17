# V11 Pre-Flight Report

Pass criterion per (symbol × rule × horizon) cell: **avg R_net > 0**.

Shuffle PF p95 is reported as a sanity metric only (not gated on).

Pass criterion per rule: at least one passing cell.

| Symbol | Rule | H | n_signals | base_rate | avg R_net | PF | shuffle PF p95 | cell |
|---|---|---:|---:|---:|---:|---:|---:|---|
| BTCUSDT | A | 16 | 2766 | 0.452 | -0.1142 | 0.79 | 1.06 | fail |
| BTCUSDT | A | 32 | 2766 | 0.479 | -0.1166 | 0.79 | 1.06 | fail |
| BTCUSDT | A | 96 | 2765 | 0.485 | -0.1148 | 0.79 | 1.05 | fail |
| BTCUSDT | B | 16 | 1924 | 0.402 | +0.0504 | 1.14 | 1.08 | PASS |
| BTCUSDT | B | 32 | 1924 | 0.497 | +0.0811 | 1.20 | 1.07 | PASS |
| BTCUSDT | B | 96 | 1924 | 0.563 | +0.1029 | 1.23 | 1.07 | PASS |
| ETHUSDT | A | 16 | 1731 | 0.486 | -0.0375 | 0.93 | 1.08 | fail |
| ETHUSDT | A | 32 | 1730 | 0.501 | -0.0369 | 0.93 | 1.08 | fail |
| ETHUSDT | A | 96 | 1730 | 0.503 | -0.0372 | 0.93 | 1.08 | fail |
| ETHUSDT | B | 16 | 853 | 0.451 | +0.1433 | 1.43 | 1.13 | PASS |
| ETHUSDT | B | 32 | 853 | 0.553 | +0.1723 | 1.45 | 1.11 | PASS |
| ETHUSDT | B | 96 | 853 | 0.603 | +0.1848 | 1.46 | 1.12 | PASS |
| SOLUSDT | A | 16 | 642 | 0.449 | -0.0806 | 0.85 | 1.14 | fail |
| SOLUSDT | A | 32 | 642 | 0.469 | -0.0808 | 0.85 | 1.14 | fail |
| SOLUSDT | A | 96 | 642 | 0.474 | -0.0751 | 0.86 | 1.13 | fail |
| SOLUSDT | B | 16 | 203 | 0.379 | +0.0953 | 1.28 | 1.30 | PASS |
| SOLUSDT | B | 32 | 203 | 0.488 | +0.0842 | 1.20 | 1.26 | PASS |
| SOLUSDT | B | 96 | 203 | 0.571 | +0.1344 | 1.31 | 1.25 | PASS |

## Decisions per (rule, horizon)

- **Rule A**: passing horizons []
  - h=16: FAIL (0 passing symbols)
  - h=32: FAIL (0 passing symbols)
  - h=96: FAIL (0 passing symbols)
- **Rule B**: passing horizons [16, 32, 96]
  - h=16: PASS (3 passing symbols)
  - h=32: PASS (3 passing symbols)
  - h=96: PASS (3 passing symbols)
