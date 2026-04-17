# V11 — Green-Field Brain

Causal Transformer meta-classifier on dollar bars, two specialists,
honest contract. Built to either pass a pre-committed bar or die clean.

> **This directory is the entire brain.** It does not depend on
> anything inside `../gpu_trainer/` except for the four proven Phase-1
> modules already copied here (`labels/triple_barrier.py`,
> `labels/meta_label.py`, `baselines_xgboost_meta.py`,
> `eval/honest_walkforward_xgb.py`). The old directory stays frozen
> during build and serves the dashboard untouched.

---

## Locked contract (do NOT change after seeing eval results)

### Bar formation
| Setting | Value |
|---|---|
| Bar type | Dollar bars |
| Bar size (BTC) | Locked by diagnostic sweep targeting ~96 bars/day, written below before training |
| Source | 15-minute OHLCV from `gpu_trainer/data_cache/{SYMBOL}_15m.parquet` |
| Symbol pool (training) | BTCUSDT + ETHUSDT + SOLUSDT |
| Symbol pool (diversification) | All other 17 symbols, post-2024 windows |
| Construction rule | Walk forward, accumulate `close × volume` until threshold; emit OHLC + close timestamp |

### Feature layer
| Group | Count | Notes |
|---|---|---|
| Fractionally differentiated returns | 4 | d ∈ {0.3, 0.5, 0.7, 0.9}; weights truncated at 1e-4 |
| Base technical | 62 | Returns at multiple lags, EMAs/EMA-crosses, RSI family, Bollinger, MACD, ATR-pct, candle anatomy, volume features, realized vol/skew/kurt, drawdown ratios, return z-scores, return autocorr |
| Microstructure | 6 | OHLCV-only proxies: signed-volume z, bar velocity, realized vol cone (3), bar duration |
| Cross-asset | 4 | BTC-as-numeraire alt return, BTC dominance proxy, BTC-alt corr, BTC-alt beta |
| Regime | 3 | Vol-of-vol, ATR-percentile bucket, ADX regime tag |
| **Total** | **79** | Dimensions before feature selection (top-K = 64 selected per fold) |

Strict causality: every feature uses only bars at or before the entry bar.
Reproducibility test enforced (same input → bit-for-bit same output).

### Labels
| Setting | Value |
|---|---|
| Triple barrier | López de Prado AFML ch. 3, first-touch |
| Barrier width | `barrier_mult × ATR(14)` where barrier_mult is **horizon-conditional**: 1.5× in normal vol regime, 2.0× in high-vol, 1.2× in low-vol |
| Horizons | 16, 32, 96 dollar bars (≈ 4h / 8h / 1d on the locked bar size) |
| Slippage | 6 bps round-trip (matches `shared_v5_trade_config.slippage_base_bps`) |
| Sample weight | Inverse co-occurrence (López de Prado AFML ch. 4, eq. 4.2) |

### Primary rules (defines the two specialists)
| Specialist | Rule | Side |
|---|---|---|
| A — momentum-after-vol-contraction | `prior_4h_return > 0` AND `bb_squeeze_pct < 0.30` AND `adx_14 > median(adx_14, 200)` | LONG only |
| B — mean-reversion-after-vol-expansion | `prior_4h_return < 0` AND `atr_expansion > 1.30` AND `bb_width_pct > 0.70` | SHORT only |

Each specialist sees only its own rule's eligible bars. They never share
a model checkpoint and never see each other's labels.

### Causal feature selection
| Setting | Value |
|---|---|
| Method | Discretized lagged mutual information (transfer-entropy proxy), 8-bin equal-frequency, lag = 1 dollar bar |
| Computed on | Train portion of fold only — never on test |
| Top-K kept | 64 features per fold |
| Used as | **Audit log only** — the trunk consumes all 79 bundle features so a single pooled pretrained checkpoint can be reused across folds. Per-fold transfer-entropy ranking is still computed and the top-64 list is written into every fold report for verdict review. |
| Logged | Yes — full ranking written to fold report for verdict audit |

### Model architecture (frozen)
| Setting | Value |
|---|---|
| Encoder | Causal Transformer, 4 layers, model dim 128, 4 heads, FFN 256 |
| Sequence length | 128 dollar bars |
| Positional encoding | Sinusoidal |
| Causal mask | Strict (lower-triangular) |
| Pretraining head | Linear `128 → n_features`, applied at masked positions only |
| Classification head | Linear `128 → 1`, applied at last position only |
| Auxiliary loss | None during finetuning |

### Training (frozen)
| Stage | Setting | Value |
|---|---|---|
| Pretrain | Mask ratio | 0.15 (per-bar feature mask) |
| Pretrain | Epochs | 20 |
| Pretrain | Batch | 128 |
| Pretrain | LR | 3e-4 with cosine decay |
| Pretrain | Optimizer | AdamW (β=0.9/0.95, wd=0.01) |
| Pretrain | Symbols | **Pooled BTC + ETH + SOL, fixed window 2021-02-01 → 2023-02-01.** Window is locked so it lies inside or before every walk-forward TRAIN window — never inside any walk-forward TEST window. Produced by `scripts/pretrain.py` once and reused by every fold. If the checkpoint is missing the harness falls back to per-fold train-slice pretraining (smoke-test only; production runs MUST use the pooled checkpoint). |
| Finetune | Bagging | N=5 independent runs per specialist, bootstrap-resampled meta-labels |
| Finetune | Epochs | 30 |
| Finetune | Batch | 64 |
| Finetune | LR | 1e-4 with cosine decay, head LR 5e-4 |
| Finetune | Loss | Binary cross-entropy with sample-uniqueness weights |
| Finetune | Class balance | Per-batch weighted sampler |
| Finetune | Early stop | Validation logloss, patience 5 |

### Calibration & gating (frozen)
| Setting | Value |
|---|---|
| Method | Mondrian conformal predictor (split conformal, per regime bucket) |
| Regime buckets | 3 — vol-low / vol-mid / vol-high (ATR percentile within fold's train) |
| Calibration set | Last 20% of train fold (held out from finetuning AND from the early-stop validation slice — three-way split: train 60% / val 20% / cal 20%, with `horizon_bars` purge at every boundary) |
| Threshold rule | Smallest cal-prob T s.t. `cal_trades ≥ 100 AND cal_PF ≥ 1.4`, evaluated **per regime bucket** |

### Walk-forward (frozen)
| Setting | Value |
|---|---|
| Train window | 24 months |
| Test window | 6 months |
| Folds | 6 |
| Purge | `horizon_bars` rows dropped from train at the train/test boundary |
| Embargo | None (purge handles it for first-touch labels) |
| Bagged ensemble policy | Average of 5 calibrated probabilities at inference |

### Stop criterion (PASS gate, identical to V10 Phase 1)
- Avg PF (across 6 test folds) ≥ **1.3**
- Per-fold trade count ≥ **500**
- No fold PF < **1.0**
- AND beats both the V5 forward report and the XGBoost Phase 1 baseline on the identical fold scheme

If any of the above fails for **both** specialists → V11 **fails clean**.
The verdict is written, no retries, no tuning.

### Anti-tuning policy (load-bearing)
- Hyperparameters above are LOCKED before any training run.
- Bug fixes are allowed; numerical tweaks to make a fold pass are not.
- Each specialist's final 6-fold walk-forward executes **exactly once**.
- A failed fold goes into the verdict; it does not trigger retraining.

---

## Honest expectations

| Stage | Probability of clearing |
|---|---|
| Pre-flight finds positive R_net for at least one (rule, horizon) | ~40% |
| Model trains cleanly given pre-flight passed | ~85% |
| Walk-forward clears the PF ≥ 1.3 gate given model trained | ~45% |
| Generalizes to ≥ 8 of 17 non-pool symbols | ~50% |
| **Joint: V11 ships** | **~7–8%** |

| Outcome | Specialists active | Symbols passing | Trades/day across fleet |
|---|---|---|---|
| BTC-only edge | 1 | 1 | 3–5 |
| Major-coin edge | 2 | BTC + ETH + SOL | 18–30 |
| Broad edge (best case) | 2 | 8 of 20 | 50–80 |
| No edge | 0 | 0 | 0 — system stays silent, V11 dies clean |

---

## Run instructions (on the GPU box)

```bash
# 0. Sync repo, install deps (torch, xgboost, sklearn, pyarrow, statsmodels, scipy)
cd gpu_trainer_v11
pip install -r requirements.txt        # if not already installed

# 1. Build dollar bars for all 20 symbols (writes to gpu_trainer_v11/data_cache_dollar/)
python -m scripts.build_bars --diagnostic   # sweep, picks size, then locks
python -m scripts.build_bars --build         # emits per-symbol dollar-bar parquets

# 2. Pre-flight signal report (per rule × horizon × symbol)
python -m scripts.preflight                  # writes reports/preflight.md + .json
                                             # KILL SWITCH: stops here if no rule passes

# 3. Pooled masked-feature pretraining (BTC+ETH+SOL, 2021-02 → 2023-02, locked)
python -m scripts.pretrain                   # writes reports/pretrained_trunk.pt + meta

# 4. (Optional) XGBoost Phase-1 baseline on dollar bars; needed for verdict beat-check
python -m gpu_trainer_v11.eval.honest_walkforward_xgb \
    --symbol BTCUSDT --horizon 32           # produces reports/xgb_baseline.json

# 5. Train + walk-forward each specialist that survived pre-flight
python -m scripts.train_walkforward --rule A   # momentum specialist
python -m scripts.train_walkforward --rule B   # mean-reversion specialist

# 6. Per-symbol diversification probe (loads last-fold artifacts persisted in step 5)
python -m scripts.diversify --rule all

# 7. Write verdict (appends to .local/tasks/v5-static-postmortem-verdict.md)
python -m scripts.write_verdict
```

## Smoke test (works on CPU, no GPU needed)

```bash
cd gpu_trainer_v11
python -m pytest tests/ -v
```

Smoke tests use tiny synthetic data so the pipeline is verified end-to-end
before committing GPU time.

---

## Directory layout

```
gpu_trainer_v11/
├── README.md                       # this file — the locked contract
├── bars/                           # dollar-bar construction
│   ├── dollar_bars.py
│   └── diagnostics.py
├── features/                       # extended feature layer
│   ├── frac_diff.py                # Hosking fractional differentiation
│   ├── base_technical.py           # 88 base technical features (port)
│   ├── microstructure.py           # OHLCV-only signed-volume proxy etc.
│   ├── cross_asset.py              # BTC-as-numeraire, dominance, beta
│   ├── regime.py                   # vol-of-vol, ATR percentile, ADX regime
│   └── compose.py                  # stitches all groups together
├── labels/                         # primary rules + meta-labels
│   ├── primary_rules.py            # the two specialists' entry rules
│   ├── triple_barrier.py           # PROVEN, copied from Phase 1
│   ├── meta_label.py               # PROVEN, copied from Phase 1
│   ├── horizon_conditional.py      # vol-regime-scaled barrier widths
│   └── sample_weights.py           # inverse co-occurrence weights (AFML 4.2)
├── selection/                      # causal feature selection
│   └── transfer_entropy.py         # discretized lagged MI per fold
├── models/                         # the brain itself
│   ├── causal_transformer.py       # 4-layer causal Transformer
│   ├── pretrain.py                 # masked-feature self-supervised
│   ├── finetune.py                 # bagged supervised finetune
│   └── conformal.py                # Mondrian conformal calibration
├── eval/                           # honest evaluation
│   ├── walkforward.py              # 6-fold harness, single-shot
│   ├── adversarial_drift.py        # train-vs-test classifier per fold
│   ├── diversification.py          # cross-symbol generalization probe
│   └── honest_walkforward_xgb.py   # PROVEN, copied from Phase 1 (baseline)
├── baselines_xgboost_meta.py       # PROVEN, copied from Phase 1
├── reports/                        # generated per run
├── scripts/                        # CLI entry points
│   ├── build_bars.py
│   ├── preflight.py
│   ├── pretrain.py                 # pooled BTC+ETH+SOL pretrain
│   ├── train_walkforward.py
│   ├── diversify.py                # loads last-fold artifacts
│   └── write_verdict.py
└── tests/                          # CPU smoke tests
    └── test_end_to_end.py
```

## Locked dollar-bar size

> Filled in after `scripts/build_bars.py --diagnostic` runs.
> Until that runs, the size is `TBD`.
