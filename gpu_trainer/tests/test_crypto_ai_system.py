import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crypto_ai_system import (
    FeatureConfig,
    add_targets,
    compute_metrics,
    engineer_features,
    parse_symbols,
    prepare_datasets,
)


def _sample_klines(rows: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    timestamp = np.arange(rows, dtype=np.int64) * 15 * 60 * 1000
    drift = np.linspace(100.0, 125.0, rows)
    cycle = np.sin(np.linspace(0, 18, rows)) * 2.0
    noise = rng.normal(0, 0.2, rows).cumsum()
    close = drift + cycle + noise
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0.05, 0.5, rows)
    low = np.minimum(open_, close) - rng.uniform(0.05, 0.5, rows)
    volume = rng.uniform(100, 250, rows)
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "trades": rng.integers(50, 200, rows),
            "taker_buy_base": volume * rng.uniform(0.35, 0.65, rows),
            "taker_buy_quote": volume * close * rng.uniform(0.35, 0.65, rows),
        }
    )


def test_parse_symbols_requires_unique_values():
    assert parse_symbols("btcusdt, ETHUSDT ") == ["BTCUSDT", "ETHUSDT"]
    try:
        parse_symbols("BTCUSDT,btcusdt")
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate symbols should fail")


def test_targets_use_round_trip_cost_threshold():
    df = _sample_klines(8)
    df["close"] = [100.0, 100.1, 100.2, 100.3, 100.45, 100.2, 99.0, 98.0]
    targeted = add_targets(df, horizon=2, threshold=0.001, round_trip_cost=0.002)
    assert targeted.loc[0, "label"] == 1
    assert targeted.loc[2, "label"] == 1
    assert targeted.loc[4, "label"] == 0
    assert "future_return" in targeted.columns


def test_prepare_datasets_fits_scaler_on_train_rows(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    symbols = ["BTCUSDT", "ETHUSDT"]
    for i, sym in enumerate(symbols):
        frame = _sample_klines()
        frame["close"] = frame["close"] + i * 20.0
        frame["open"] = frame["open"] + i * 20.0
        frame["high"] = frame["high"] + i * 20.0
        frame["low"] = frame["low"] + i * 20.0
        frame.to_parquet(data_dir / f"{sym}_15m.parquet", index=False)

    cfg = FeatureConfig(sequence_length=16, horizon=4, purge_bars=12, val_fraction=0.2)
    prepared = prepare_datasets(data_dir, symbols, cfg)

    assert len(prepared.train) > 0
    assert len(prepared.val) > 0
    assert prepared.symbols == symbols
    assert prepared.train.sequences.shape[-1] == len(prepared.feature_cols)
    assert prepared.scaler.center_.shape[0] == len(prepared.feature_cols)


def test_compute_metrics_charges_only_traded_rows():
    labels = np.array([2, 1, 0, 2])
    preds = np.array([2, 1, 0, 2])
    returns = np.array([0.02, 0.50, 0.01, -0.01])
    positions = np.array([1.0, 0.0, -1.0, 1.0])

    metrics = compute_metrics(labels, preds, returns, positions, round_trip_cost=0.001)

    assert metrics.accuracy == 1.0
    assert metrics.trades == 3
    assert metrics.trade_rate == 0.75
    assert metrics.total_return < 0.02


def test_feature_engineering_is_finite_after_warmup():
    features = engineer_features(_sample_klines(160))
    feature_cols = [c for c in features.columns if c not in {"timestamp", "open", "high", "low", "close", "volume"}]
    assert np.isfinite(features[feature_cols].to_numpy()).all()
