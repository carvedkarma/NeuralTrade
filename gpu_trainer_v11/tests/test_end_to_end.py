"""
End-to-end CPU smoke tests on tiny synthetic data.

These tests verify that the entire V11 pipeline wires together
correctly. They do NOT verify edge — they verify that the code runs.

Run:  cd .. && python -m pytest gpu_trainer_v11/tests -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gpu_trainer_v11.bars.dollar_bars import DollarBarConfig, build_dollar_bars
from gpu_trainer_v11.eval.adversarial_drift import adversarial_auc
from gpu_trainer_v11.features.compose import compute_features
from gpu_trainer_v11.features.frac_diff import find_min_d, frac_diff_ffd
from gpu_trainer_v11.labels.horizon_conditional import per_bar_barrier_mult
from gpu_trainer_v11.labels.meta_label_v11 import compute_meta_labels_v11
from gpu_trainer_v11.labels.primary_rules import primary_rule
from gpu_trainer_v11.labels.sample_weights import sample_uniqueness_weights
from gpu_trainer_v11.models.causal_transformer import CausalTransformer, V11ModelConfig
from gpu_trainer_v11.models.conformal import fit_mondrian
from gpu_trainer_v11.models.datasets import build_sequences
from gpu_trainer_v11.models.finetune import finetune_bagged, predict_proba_bagged
from gpu_trainer_v11.models.pretrain import pretrain
from gpu_trainer_v11.selection.transfer_entropy import select_top_k


def _synthetic_15m(n: int = 4000, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.005, n)
    rets[1000:1100] += 0.003
    rets[2000:2200] -= 0.004
    close = 30000 * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0, 0.003, n))
    low = close * (1 - rng.uniform(0, 0.003, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = rng.uniform(50, 200, n) + (np.abs(rets) * 1e5)
    ts = np.arange(n) * (15 * 60 * 1000) + 1_700_000_000_000
    return pd.DataFrame({
        "timestamp": ts.astype(np.int64),
        "open": open_, "high": high, "low": low, "close": close, "volume": vol,
    })


def test_dollar_bars_build_deterministic():
    df = _synthetic_15m()
    cfg = DollarBarConfig(threshold_dollars=2_000_000)
    a = build_dollar_bars(df, cfg)
    b = build_dollar_bars(df, cfg)
    assert len(a) > 50
    pd.testing.assert_frame_equal(a, b)
    # Each bar's dollar_volume reaches threshold or capped by max_bars
    assert ((a["dollar_volume"] >= cfg.threshold_dollars) | (a["n_15m_bars"] >= cfg.max_bars_per_dollar_bar)).all()


def test_frac_diff_stationarity():
    df = _synthetic_15m(n=2000)
    log_close = np.log(df["close"])
    out = frac_diff_ffd(log_close, 0.5)
    assert out.notna().sum() > 1500
    d = find_min_d(df["close"])
    assert 0.0 <= d <= 1.0


def test_features_reproducible():
    df = _synthetic_15m()
    bars = build_dollar_bars(df, DollarBarConfig(threshold_dollars=2_000_000))
    f1 = compute_features(bars, btc_bars=None, symbol="BTCUSDT").features
    f2 = compute_features(bars, btc_bars=None, symbol="BTCUSDT").features
    pd.testing.assert_frame_equal(f1, f2)
    assert f1.shape[1] >= 70  # locked feature count is 79
    assert np.isfinite(f1.to_numpy()).all()


def test_primary_rules_emit_signals():
    df = _synthetic_15m()
    bars = build_dollar_bars(df, DollarBarConfig(threshold_dollars=2_000_000))
    bundle = compute_features(bars, None, "BTCUSDT")
    sigA = primary_rule(bars, bundle.side_data, "A")
    sigB = primary_rule(bars, bundle.side_data, "B")
    # Signals should be in correct sign space
    assert set(sigA.unique()).issubset({0, 1})
    assert set(sigB.unique()).issubset({0, -1})


def test_meta_label_horizon_conditional():
    df = _synthetic_15m()
    bars = build_dollar_bars(df, DollarBarConfig(threshold_dollars=2_000_000))
    bundle = compute_features(bars, None, "BTCUSDT")
    sigA = primary_rule(bars, bundle.side_data, "A")
    bm = per_bar_barrier_mult(bundle.features["atr_pct_bucket"])
    meta = compute_meta_labels_v11(bars, sigA, horizon_bars=16, barrier_mult_per_bar=bm)
    assert "meta_label" in meta.columns
    assert meta["barrier_mult"].notna().all()
    elig = meta[meta["eligible"]]
    if len(elig) > 0:
        assert elig["R_net"].notna().all()
        # By construction R_gross should be in [-1.05, +1.05] after slippage
        assert elig["R_gross"].between(-1.5, 1.5).all()


def test_sample_weights_average_to_unity():
    n_bars = 100
    entries = np.array([0, 5, 10, 50, 60])
    exits = np.array([20, 25, 40, 70, 80])
    w = sample_uniqueness_weights(entries, exits, n_bars)
    assert len(w) == 5
    assert abs(w.mean() - 1.0) < 1e-6


def test_transfer_entropy_selection_returns_kept_features():
    df = _synthetic_15m()
    bars = build_dollar_bars(df, DollarBarConfig(threshold_dollars=2_000_000))
    bundle = compute_features(bars, None, "BTCUSDT")
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, size=len(bundle.features))
    res = select_top_k(bundle.features, y, k=10)
    assert res.keep_mask.sum() == 10


def test_causal_transformer_forward_shapes():
    cfg = V11ModelConfig(n_features=12, seq_len=16)
    model = CausalTransformer(cfg)
    import torch
    x = torch.randn(4, 16, 12)
    logits = model.forward_classify(x)
    assert logits.shape == (4,)
    recon = model.forward_pretrain(x)
    assert recon.shape == (4, 16, 12)


def test_pretrain_then_finetune_runs():
    cfg = V11ModelConfig(n_features=8, seq_len=16, d_model=32, n_heads=4, n_layers=2, d_ff=64)
    rng = np.random.default_rng(17)
    X = rng.standard_normal((128, 16, 8)).astype(np.float32)
    trunk = pretrain(cfg, X, epochs=2, batch=32)
    Xtr = rng.standard_normal((96, 16, 8)).astype(np.float32)
    ytr = rng.integers(0, 2, size=96).astype(np.int8)
    wtr = np.ones(96, dtype=np.float32)
    Xv = rng.standard_normal((32, 16, 8)).astype(np.float32)
    yv = rng.integers(0, 2, size=32).astype(np.int8)
    wv = np.ones(32, dtype=np.float32)
    models = finetune_bagged(trunk, Xtr, ytr, wtr, Xv, yv, wv,
                             n_bag=2, epochs=2, batch=16)
    assert len(models) == 2
    p = predict_proba_bagged(models, Xv)
    assert p.shape == (32,)
    assert np.all((p >= 0) & (p <= 1))


def test_mondrian_conformal():
    rng = np.random.default_rng(17)
    p = rng.uniform(0, 1, 600)
    y = (p > 0.5).astype(np.int8)
    R = np.where(y == 1, rng.uniform(0.2, 0.9, 600), rng.uniform(-0.9, -0.1, 600))
    regime = rng.integers(0, 3, 600).astype(np.int8)
    cal = fit_mondrian(p, y, R, regime, min_trades=20, min_pf=1.2)
    admit = cal.admit(p, regime)
    assert admit.dtype == bool
    assert admit.shape == p.shape


def test_adversarial_drift_runs():
    rng = np.random.default_rng(17)
    Xa = rng.standard_normal((400, 12)).astype(np.float32)
    Xb = rng.standard_normal((400, 12)).astype(np.float32) + 0.5
    auc = adversarial_auc(Xa, Xb)
    assert 0.0 <= auc <= 1.0


def test_walkforward_smoke_runs():
    """Tiny synthetic dataset: harness must complete without exceptions
    even if not enough data to produce real folds."""
    from gpu_trainer_v11.eval.walkforward import run_walk_forward
    df = _synthetic_15m(n=3000)
    bars = build_dollar_bars(df, DollarBarConfig(threshold_dollars=1_000_000))
    if len(bars) < 600:
        pytest.skip("synthetic dataset too small")
    rep = run_walk_forward(
        bars, btc_bars=None, symbol="BTCUSDT", rule="A",
        horizon_bars=8, n_bag=1, pretrain_epochs=1, finetune_epochs=1,
        top_k_features=24,
    )
    # No assertion on PASS — synthetic data has no edge by design.
    assert rep.rule == "A"
