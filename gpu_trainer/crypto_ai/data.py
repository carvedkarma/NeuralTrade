from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from .config import SystemConfig

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
INTERVAL_TO_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


@dataclass(slots=True)
class DataBundle:
    symbols: list[str]
    timestamps: pd.DatetimeIndex
    feature_names: list[str]
    features: np.ndarray
    class_labels: np.ndarray
    horizon_returns: np.ndarray
    one_step_returns: np.ndarray
    close_prices: np.ndarray


def _symbol_file(data_dir: Path, symbol: str, interval: str) -> Path:
    return data_dir / f"{symbol}_{interval}.parquet"


def _request_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    retries: int = 4,
) -> list[list]:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            response = requests.get(BINANCE_KLINES_URL, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise RuntimeError(f"Unexpected Binance payload for {symbol}: {payload}")
            return payload
        except Exception as exc:  # pragma: no cover - network behavior
            if attempt >= retries:
                raise RuntimeError(
                    f"Failed to request Binance klines for {symbol} after {retries + 1} attempts"
                ) from exc
            time.sleep(delay)
            delay *= 2.0
    return []


def download_symbol_klines(
    symbol: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"Unsupported interval {interval}. Supported: {sorted(INTERVAL_TO_MS)}")
    interval_ms = INTERVAL_TO_MS[interval]
    cursor = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    frames: list[pd.DataFrame] = []
    while cursor < end_ms:
        klines = _request_klines(symbol, interval, cursor, end_ms)
        if not klines:
            break
        chunk = pd.DataFrame(
            klines,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "n_trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )
        chunk = chunk[
            ["open_time", "open", "high", "low", "close", "volume", "quote_volume", "n_trades"]
        ].copy()
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            chunk[col] = chunk[col].astype(float)
        chunk["n_trades"] = chunk["n_trades"].astype(int)
        chunk["timestamp"] = pd.to_datetime(chunk["open_time"], unit="ms", utc=True)
        chunk = chunk.drop(columns=["open_time"])
        frames.append(chunk)

        last_open_ms = int(klines[-1][0])
        next_cursor = last_open_ms + interval_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        # Respect API limits.
        time.sleep(0.05)

        if len(klines) < 1000:
            break

    if not frames:
        raise RuntimeError(f"No data downloaded for {symbol} {interval}")
    data = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)
    return data


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_gain = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    close = x["close"]
    high = x["high"]
    low = x["low"]
    open_ = x["open"]
    volume = x["volume"]
    quote_volume = x["quote_volume"]

    log_close = np.log(close.clip(lower=1e-12))
    ret_1 = log_close.diff(1)
    ret_3 = log_close.diff(3)
    ret_6 = log_close.diff(6)
    ret_24 = log_close.diff(24)

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    ema_signal = close.ewm(span=50, adjust=False).mean()

    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = true_range.rolling(window=14, min_periods=14).mean()

    vol_24 = ret_1.rolling(window=24, min_periods=24).std()
    vol_72 = ret_1.rolling(window=72, min_periods=72).std()
    volume_z = (volume - volume.rolling(48, min_periods=48).mean()) / (
        volume.rolling(48, min_periods=48).std() + 1e-8
    )

    features = pd.DataFrame(
        {
            "ret_1": ret_1,
            "ret_3": ret_3,
            "ret_6": ret_6,
            "ret_24": ret_24,
            "ema_12_spread": (close / ema_fast) - 1.0,
            "ema_26_spread": (close / ema_slow) - 1.0,
            "ema_50_spread": (close / ema_signal) - 1.0,
            "ema_fast_slow_ratio": (ema_fast / (ema_slow + 1e-8)) - 1.0,
            "hl_spread": (high - low) / (close + 1e-8),
            "oc_spread": (close - open_) / (open_ + 1e-8),
            "atr14_norm": atr14 / (close + 1e-8),
            "rsi_14": _rsi(close, period=14) / 100.0,
            "vol_24": vol_24,
            "vol_ratio_24_72": vol_24 / (vol_72 + 1e-8),
            "volume_z_48": volume_z,
            "quote_volume_z_48": (
                (quote_volume - quote_volume.rolling(48, min_periods=48).mean())
                / (quote_volume.rolling(48, min_periods=48).std() + 1e-8)
            ),
            "trade_intensity_proxy": x["n_trades"] / (volume + 1e-8),
        },
        index=x["timestamp"],
    )
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    return features


def _load_or_fetch_symbol(config: SystemConfig, symbol: str, refresh: bool) -> pd.DataFrame:
    path = _symbol_file(config.data_dir, symbol, config.interval)
    if path.exists() and not refresh:
        return pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=config.history_days)
    df = download_symbol_klines(symbol, config.interval, start_dt=start_dt, end_dt=end_dt)
    df.to_parquet(path, index=False)
    return df


def _intersection_index(frames: Iterable[pd.DataFrame]) -> pd.DatetimeIndex:
    idx: pd.DatetimeIndex | None = None
    for frame in frames:
        current = pd.DatetimeIndex(frame.index)
        idx = current if idx is None else idx.intersection(current)
    if idx is None or len(idx) == 0:
        raise RuntimeError("No common timestamps found across symbols.")
    return idx.sort_values()


def build_data_bundle(config: SystemConfig, horizon: int, refresh: bool = False) -> DataBundle:
    if len(config.symbols) != 10:
        raise ValueError(
            f"System expects exactly 10 symbols, got {len(config.symbols)}: {config.symbols}"
        )
    config.ensure_directories()
    raw_by_symbol: dict[str, pd.DataFrame] = {}
    feat_by_symbol: dict[str, pd.DataFrame] = {}
    close_by_symbol: dict[str, pd.Series] = {}

    for symbol in config.symbols:
        raw = _load_or_fetch_symbol(config, symbol, refresh=refresh)
        if raw.empty:
            raise RuntimeError(f"No raw data loaded for {symbol}")
        raw_by_symbol[symbol] = raw

        engineered = compute_symbol_features(raw)
        if engineered.empty:
            raise RuntimeError(f"No engineered features for {symbol}")
        feat_by_symbol[symbol] = engineered
        close_series = raw.set_index("timestamp")["close"].astype(float)
        close_by_symbol[symbol] = close_series

    common_idx = _intersection_index(feat_by_symbol.values())
    feature_names = list(next(iter(feat_by_symbol.values())).columns)
    symbols = list(config.symbols)

    feat_stack = []
    close_stack = []
    for symbol in symbols:
        feat_stack.append(feat_by_symbol[symbol].loc[common_idx, feature_names].to_numpy(dtype=np.float32))
        close_stack.append(close_by_symbol[symbol].reindex(common_idx).to_numpy(dtype=np.float32))

    features = np.stack(feat_stack, axis=1)  # [T, S, F]
    close_prices = np.stack(close_stack, axis=1)  # [T, S]

    horizon_returns = (np.roll(close_prices, -horizon, axis=0) / (close_prices + 1e-8)) - 1.0
    one_step_returns = (np.roll(close_prices, -1, axis=0) / (close_prices + 1e-8)) - 1.0

    class_labels = np.full_like(horizon_returns, fill_value=1, dtype=np.int64)
    class_labels[horizon_returns > config.label_threshold] = 2
    class_labels[horizon_returns < -config.label_threshold] = 0

    valid_until = len(common_idx) - max(horizon, 1)
    if valid_until <= config.sequence_length + 32:
        raise RuntimeError(
            f"Insufficient history after alignment: {valid_until} rows, "
            f"need at least {config.sequence_length + 32}."
        )

    features = features[:valid_until]
    close_prices = close_prices[:valid_until]
    horizon_returns = horizon_returns[:valid_until]
    one_step_returns = one_step_returns[:valid_until]
    class_labels = class_labels[:valid_until]
    timestamps = common_idx[:valid_until]

    finite_mask = np.isfinite(features).all(axis=(1, 2))
    finite_mask &= np.isfinite(horizon_returns).all(axis=1)
    finite_mask &= np.isfinite(one_step_returns).all(axis=1)
    features = features[finite_mask]
    close_prices = close_prices[finite_mask]
    horizon_returns = horizon_returns[finite_mask]
    one_step_returns = one_step_returns[finite_mask]
    class_labels = class_labels[finite_mask]
    timestamps = timestamps[finite_mask]

    return DataBundle(
        symbols=symbols,
        timestamps=timestamps,
        feature_names=feature_names,
        features=features.astype(np.float32),
        class_labels=class_labels.astype(np.int64),
        horizon_returns=horizon_returns.astype(np.float32),
        one_step_returns=one_step_returns.astype(np.float32),
        close_prices=close_prices.astype(np.float32),
    )
