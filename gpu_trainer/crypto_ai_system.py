#!/usr/bin/env python3
"""
Multi-symbol crypto AI trading research system.

This module is intentionally self-contained so it can be copied to a local GPU
machine and operated from the CLI.  It trains on chronological historical data,
fits scalers on training rows only, uses a purge gap before validation, includes
fees/slippage in validation metrics, and can keep retraining itself on a timer.

It is a research and signal-generation tool, not a promise of profitability.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


log = logging.getLogger("CryptoAISystem")
BASE_DIR = Path(__file__).resolve().parent

DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "AVAXUSDT",
]

BINANCE_SPOT_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_FUTURES_KLINES = "https://fapi.binance.com/fapi/v1/klines"


@dataclass
class FeatureConfig:
    interval: str = "15m"
    sequence_length: int = 96
    horizon: int = 16
    label_threshold: float = 0.0015
    fee_bps: float = 6.0
    slippage_bps: float = 4.0
    purge_bars: int = 160
    val_fraction: float = 0.18

    @property
    def round_trip_cost(self) -> float:
        return 2.0 * (self.fee_bps + self.slippage_bps) / 10_000.0


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    d_model: int = 192
    n_heads: int = 6
    layers: int = 4
    dropout: float = 0.15
    patience: int = 6
    min_confidence: float = 0.48
    seed: int = 42


@dataclass
class BacktestMetrics:
    samples: int
    accuracy: float
    macro_f1: float
    trades: int
    trade_rate: float
    total_return: float
    sharpe: float
    max_drawdown: float
    profit_factor: float


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_symbols(symbols: str | Sequence[str]) -> List[str]:
    if isinstance(symbols, str):
        values = [s.strip().upper() for s in symbols.split(",")]
    else:
        values = [str(s).strip().upper() for s in symbols]
    values = [s for s in values if s]
    if len(values) != len(set(values)):
        raise ValueError("Symbols must be unique")
    return values


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log.info("GPU: %s (%.1f GB)", name, mem_gb)
    else:
        log.warning("Using CPU. Pass --device cuda on a CUDA machine for GPU training.")
    return resolved


def _interval_to_millis(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    multipliers = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    if unit not in multipliers:
        raise ValueError(f"Unsupported interval: {interval}")
    return value * multipliers[unit]


class BinanceDownloader:
    """Public Binance kline downloader with local parquet caching."""

    def __init__(self, data_dir: Path, market: str = "futures", timeout: int = 30):
        self.data_dir = data_dir
        self.market = market
        self.timeout = timeout
        self.endpoint = BINANCE_FUTURES_KLINES if market == "futures" else BINANCE_SPOT_KLINES
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def parquet_path(self, symbol: str, interval: str) -> Path:
        return self.data_dir / f"{symbol}_{interval}.parquet"

    def download_symbol(
        self,
        symbol: str,
        interval: str,
        limit_bars: int,
        force: bool = False,
    ) -> Path:
        path = self.parquet_path(symbol, interval)
        if path.exists() and not force:
            existing = pd.read_parquet(path)
            if len(existing) >= limit_bars:
                log.info("%s: cached %d bars at %s", symbol, len(existing), path)
                return path

        rows: List[list] = []
        remaining = int(limit_bars)
        end_time = None
        interval_ms = _interval_to_millis(interval)

        log.info("%s: downloading %d %s candles from Binance %s", symbol, limit_bars, interval, self.market)
        while remaining > 0:
            batch_limit = min(1000, remaining)
            params = {"symbol": symbol, "interval": interval, "limit": batch_limit}
            if end_time is not None:
                params["endTime"] = end_time
            response = requests.get(self.endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break
            rows = batch + rows
            earliest_open = int(batch[0][0])
            end_time = earliest_open - interval_ms
            remaining -= len(batch)
            if len(batch) < batch_limit:
                break
            time.sleep(0.05)

        if not rows:
            raise RuntimeError(f"No klines downloaded for {symbol}")

        df = self._klines_to_frame(rows).drop_duplicates("timestamp").sort_values("timestamp")
        if len(df) > limit_bars:
            df = df.tail(limit_bars)
        df.to_parquet(path, index=False)
        log.info("%s: saved %d bars to %s", symbol, len(df), path)
        return path

    @staticmethod
    def _klines_to_frame(rows: Sequence[Sequence[object]]) -> pd.DataFrame:
        columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ]
        df = pd.DataFrame(rows, columns=columns)
        numeric = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
        ]
        for col in numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[numeric].dropna()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling features using only current and historical candles."""
    out = df.copy().sort_values("timestamp").reset_index(drop=True)
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    open_ = out["open"].astype(float)
    volume = out["volume"].astype(float)

    log_close = np.log(close.replace(0.0, np.nan))
    out["ret_1"] = log_close.diff()
    out["ret_4"] = log_close.diff(4)
    out["ret_16"] = log_close.diff(16)
    out["ret_96"] = log_close.diff(96)
    out["range_pct"] = (high - low) / close
    out["body_pct"] = (close - open_) / open_.replace(0.0, np.nan)
    out["upper_wick_pct"] = (high - np.maximum(open_, close)) / close
    out["lower_wick_pct"] = (np.minimum(open_, close) - low) / close

    for window in (8, 16, 48, 96):
        ret = out["ret_1"]
        out[f"vol_{window}"] = ret.rolling(window, min_periods=window // 2).std()
        out[f"volume_z_{window}"] = (
            (volume - volume.rolling(window, min_periods=window // 2).mean())
            / volume.rolling(window, min_periods=window // 2).std().replace(0.0, np.nan)
        )
        ema = close.ewm(span=window, adjust=False).mean()
        out[f"ema_dist_{window}"] = close / ema.replace(0.0, np.nan) - 1.0

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["macd_pct"] = macd / close.replace(0.0, np.nan)
    out["macd_signal_pct"] = macd.ewm(span=9, adjust=False).mean() / close.replace(0.0, np.nan)
    out["rsi_14"] = (_rsi(close, 14) - 50.0) / 50.0

    tr = _true_range(out)
    atr = tr.ewm(span=14, adjust=False).mean()
    out["atr_pct"] = atr / close.replace(0.0, np.nan)

    mid = close.rolling(20, min_periods=10).mean()
    std = close.rolling(20, min_periods=10).std()
    out["bb_z_20"] = (close - mid) / std.replace(0.0, np.nan)

    taker_ratio = out["taker_buy_base"] / volume.replace(0.0, np.nan)
    out["taker_buy_ratio"] = taker_ratio.clip(0.0, 1.0)
    out["quote_volume_log"] = np.log1p(out["quote_volume"])
    out["trades_log"] = np.log1p(out["trades"])

    dt = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
    minutes = dt.dt.hour * 60 + dt.dt.minute
    out["tod_sin"] = np.sin(2.0 * np.pi * minutes / 1440.0)
    out["tod_cos"] = np.cos(2.0 * np.pi * minutes / 1440.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * dt.dt.dayofweek / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * dt.dt.dayofweek / 7.0)

    feature_cols = get_feature_columns(out)
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan)
    out[feature_cols] = out[feature_cols].ffill().fillna(0.0).clip(-20.0, 20.0)
    return out


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "future_return",
        "label",
    }
    return [c for c in df.columns if c not in excluded]


def add_targets(df: pd.DataFrame, horizon: int, threshold: float, round_trip_cost: float) -> pd.DataFrame:
    out = df.copy()
    future_close = out["close"].shift(-horizon)
    out["future_return"] = (future_close / out["close"]) - 1.0
    net_threshold = threshold + round_trip_cost
    out["label"] = 1
    out.loc[out["future_return"] > net_threshold, "label"] = 2
    out.loc[out["future_return"] < -net_threshold, "label"] = 0
    return out.dropna(subset=["future_return"]).reset_index(drop=True)


class RobustStandardizer:
    """Median/IQR standardizer fitted on training rows only."""

    def __init__(self) -> None:
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "RobustStandardizer":
        self.center_ = np.nanmedian(x, axis=0)
        q75 = np.nanpercentile(x, 75, axis=0)
        q25 = np.nanpercentile(x, 25, axis=0)
        scale = q75 - q25
        scale[scale < 1e-8] = 1.0
        self.scale_ = scale
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted")
        z = (x - self.center_) / self.scale_
        return np.clip(z, -8.0, 8.0).astype(np.float32)

    def to_dict(self) -> Dict[str, list]:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted")
        return {"center": self.center_.tolist(), "scale": self.scale_.tolist()}

    @classmethod
    def from_dict(cls, payload: Dict[str, list]) -> "RobustStandardizer":
        scaler = cls()
        scaler.center_ = np.asarray(payload["center"], dtype=np.float32)
        scaler.scale_ = np.asarray(payload["scale"], dtype=np.float32)
        return scaler


@dataclass
class PreparedData:
    train: "MultiAssetWindowDataset"
    val: "MultiAssetWindowDataset"
    feature_cols: List[str]
    scaler: RobustStandardizer
    symbols: List[str]


class MultiAssetWindowDataset(Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        returns: np.ndarray,
        symbol_ids: np.ndarray,
    ):
        self.sequences = torch.from_numpy(sequences.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.returns = torch.from_numpy(returns.astype(np.float32))
        self.symbol_ids = torch.from_numpy(symbol_ids.astype(np.int64))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.symbol_ids[idx], self.labels[idx], self.returns[idx]


def load_symbol_frame(data_dir: Path, symbol: str, interval: str, cfg: FeatureConfig) -> pd.DataFrame:
    path = data_dir / f"{symbol}_{interval}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the download command first")
    raw = pd.read_parquet(path)
    featured = engineer_features(raw)
    targeted = add_targets(featured, cfg.horizon, cfg.label_threshold, cfg.round_trip_cost)
    if len(targeted) < cfg.sequence_length + cfg.horizon + cfg.purge_bars + 100:
        raise ValueError(f"{symbol}: not enough bars after feature/target creation ({len(targeted)})")
    return targeted


def _make_windows(
    matrix: np.ndarray,
    labels: np.ndarray,
    returns: np.ndarray,
    symbol_id: int,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    rs: List[float] = []
    sids: List[int] = []
    for end in range(sequence_length - 1, len(matrix)):
        start = end - sequence_length + 1
        xs.append(matrix[start : end + 1])
        ys.append(int(labels[end]))
        rs.append(float(returns[end]))
        sids.append(symbol_id)
    if not xs:
        return (
            np.empty((0, sequence_length, matrix.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.stack(xs), np.asarray(ys), np.asarray(rs), np.asarray(sids)


def prepare_datasets(data_dir: Path, symbols: List[str], cfg: FeatureConfig) -> PreparedData:
    frames = {sym: load_symbol_frame(data_dir, sym, cfg.interval, cfg) for sym in symbols}
    feature_cols = get_feature_columns(next(iter(frames.values())))

    train_feature_blocks = []
    splits: Dict[str, Tuple[int, int]] = {}
    for sym, frame in frames.items():
        val_start = int(len(frame) * (1.0 - cfg.val_fraction))
        train_end = max(cfg.sequence_length, val_start - cfg.purge_bars)
        if val_start <= train_end:
            raise ValueError(f"{sym}: validation split too small; lower --purge-bars or --val-fraction")
        splits[sym] = (train_end, val_start)
        train_feature_blocks.append(frame.iloc[:train_end][feature_cols].to_numpy(dtype=np.float32))
        log.info("%s split: train=%d purge=%d val=%d", sym, train_end, val_start - train_end, len(frame) - val_start)

    scaler = RobustStandardizer().fit(np.vstack(train_feature_blocks))

    train_parts = []
    val_parts = []
    for symbol_id, sym in enumerate(symbols):
        frame = frames[sym]
        train_end, val_start = splits[sym]

        train_matrix = scaler.transform(frame.iloc[:train_end][feature_cols].to_numpy(dtype=np.float32))
        val_context_start = max(0, val_start - cfg.sequence_length + 1)
        val_matrix = scaler.transform(frame.iloc[val_context_start:][feature_cols].to_numpy(dtype=np.float32))

        train_parts.append(
            _make_windows(
                train_matrix,
                frame.iloc[:train_end]["label"].to_numpy(),
                frame.iloc[:train_end]["future_return"].to_numpy(),
                symbol_id,
                cfg.sequence_length,
            )
        )
        raw_val = _make_windows(
            val_matrix,
            frame.iloc[val_context_start:]["label"].to_numpy(),
            frame.iloc[val_context_start:]["future_return"].to_numpy(),
            symbol_id,
            cfg.sequence_length,
        )
        keep_from = val_start - val_context_start
        val_parts.append(tuple(part[keep_from - cfg.sequence_length + 1 :] for part in raw_val))

    def concat(parts: Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> MultiAssetWindowDataset:
        seq, label, ret, sid = zip(*parts)
        return MultiAssetWindowDataset(
            np.concatenate(seq, axis=0),
            np.concatenate(label, axis=0),
            np.concatenate(ret, axis=0),
            np.concatenate(sid, axis=0),
        )

    train_ds = concat(train_parts)
    val_ds = concat(val_parts)
    log.info("Prepared datasets: train=%d validation=%d features=%d", len(train_ds), len(val_ds), len(feature_cols))
    return PreparedData(train=train_ds, val=val_ds, feature_cols=feature_cols, scaler=scaler, symbols=symbols)


class CryptoTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_symbols: int,
        d_model: int,
        n_heads: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.symbol_embedding = nn.Embedding(n_symbols, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, 512, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        self.return_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.log_sigma_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, symbol_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        seq_len = x.shape[1]
        if seq_len > self.position_embedding.shape[1]:
            raise ValueError(f"sequence length {seq_len} exceeds max 512")
        h = self.input_projection(x)
        h = h + self.position_embedding[:, :seq_len, :]
        h = h + self.symbol_embedding(symbol_ids).unsqueeze(1)
        h = self.encoder(h)
        pooled = self.norm(h[:, -1, :])
        logits = self.action_head(pooled)
        mu = 0.05 * torch.tanh(self.return_head(pooled).squeeze(-1))
        sigma = F.softplus(self.log_sigma_head(pooled).squeeze(-1)) + 1e-4
        return {"logits": logits, "mu": mu, "sigma": sigma}


def class_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=3).float()
    weights = counts.sum() / (3.0 * counts.clamp_min(1.0))
    return (weights / weights.mean()).float()


def train_epoch(
    model: CryptoTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weights: torch.Tensor,
    amp: bool,
) -> float:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    total = 0.0
    seen = 0
    for x, symbol_ids, labels, returns in loader:
        x = x.to(device, non_blocking=True)
        symbol_ids = symbol_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        returns = returns.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(x, symbol_ids)
            ce = F.cross_entropy(out["logits"], labels, weight=weights)
            huber = F.smooth_l1_loss(out["mu"], returns)
            nll = 0.5 * (((returns - out["mu"]) / out["sigma"]) ** 2 + 2.0 * torch.log(out["sigma"])).mean()
            loss = ce + 4.0 * huber + 0.25 * nll
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        batch = labels.shape[0]
        total += float(loss.detach().cpu()) * batch
        seen += batch
    return total / max(1, seen)


@torch.no_grad()
def evaluate(
    model: CryptoTransformer,
    loader: DataLoader,
    device: torch.device,
    min_confidence: float,
    round_trip_cost: float,
) -> Tuple[float, BacktestMetrics]:
    model.eval()
    losses = []
    all_labels = []
    all_preds = []
    all_returns = []
    all_positions = []
    for x, symbol_ids, labels, returns in loader:
        x = x.to(device, non_blocking=True)
        symbol_ids = symbol_ids.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        returns_device = returns.to(device, non_blocking=True)
        out = model(x, symbol_ids)
        loss = F.cross_entropy(out["logits"], labels_device) + 4.0 * F.smooth_l1_loss(out["mu"], returns_device)
        probs = torch.softmax(out["logits"], dim=-1)
        confidence, pred = probs.max(dim=-1)
        position = torch.zeros_like(out["mu"])
        position[(pred == 2) & (confidence >= min_confidence)] = 1.0
        position[(pred == 0) & (confidence >= min_confidence)] = -1.0

        losses.append(float(loss.cpu()) * labels.shape[0])
        all_labels.append(labels.numpy())
        all_preds.append(pred.cpu().numpy())
        all_returns.append(returns.numpy())
        all_positions.append(position.cpu().numpy())

    labels_np = np.concatenate(all_labels)
    preds_np = np.concatenate(all_preds)
    returns_np = np.concatenate(all_returns)
    positions_np = np.concatenate(all_positions)
    val_loss = sum(losses) / max(1, len(labels_np))
    metrics = compute_metrics(labels_np, preds_np, returns_np, positions_np, round_trip_cost)
    return val_loss, metrics


def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    returns: np.ndarray,
    positions: np.ndarray,
    round_trip_cost: float,
) -> BacktestMetrics:
    accuracy = float((labels == preds).mean()) if len(labels) else 0.0
    f1s = []
    for cls in (0, 1, 2):
        tp = np.sum((preds == cls) & (labels == cls))
        fp = np.sum((preds == cls) & (labels != cls))
        fn = np.sum((preds != cls) & (labels == cls))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1s.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
    macro_f1 = float(np.mean(f1s))

    trades = np.abs(positions) > 0
    gross = positions * returns
    net = np.where(trades, gross - round_trip_cost, 0.0)
    equity = np.cumprod(1.0 + net)
    if len(equity):
        peak = np.maximum.accumulate(equity)
        max_drawdown = float(((equity - peak) / peak).min())
        total_return = float(equity[-1] - 1.0)
    else:
        max_drawdown = 0.0
        total_return = 0.0
    if net.std() > 1e-12:
        sharpe = float(net.mean() / net.std() * math.sqrt(365 * 24 * 4))
    else:
        sharpe = 0.0
    wins = net[net > 0].sum()
    losses = abs(net[net < 0].sum())
    profit_factor = float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0)
    return BacktestMetrics(
        samples=int(len(labels)),
        accuracy=accuracy,
        macro_f1=macro_f1,
        trades=int(trades.sum()),
        trade_rate=float(trades.mean()) if len(trades) else 0.0,
        total_return=total_return,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        profit_factor=profit_factor,
    )


def train_model(args: argparse.Namespace) -> Path:
    symbols = parse_symbols(args.symbols)
    set_seed(args.seed)
    device = resolve_device(args.device)
    feature_cfg = FeatureConfig(
        interval=args.interval,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
        label_threshold=args.label_threshold,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        purge_bars=args.purge_bars,
        val_fraction=args.val_fraction,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        n_heads=args.n_heads,
        layers=args.layers,
        dropout=args.dropout,
        patience=args.patience,
        min_confidence=args.min_confidence,
        seed=args.seed,
    )
    prepared = prepare_datasets(Path(args.data_dir), symbols, feature_cfg)
    train_loader = DataLoader(
        prepared.train,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        prepared.val,
        batch_size=train_cfg.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = CryptoTransformer(
        n_features=len(prepared.feature_cols),
        n_symbols=len(symbols),
        d_model=train_cfg.d_model,
        n_heads=train_cfg.n_heads,
        layers=train_cfg.layers,
        dropout=train_cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, train_cfg.epochs))
    weights = class_weights(prepared.train.labels).to(device)
    amp = bool(args.amp and device.type == "cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "crypto_ai_best.pt"
    best_loss = float("inf")
    best_metrics: Optional[BacktestMetrics] = None
    no_improve = 0

    log.info("Model parameters: %,d", sum(p.numel() for p in model.parameters() if p.requires_grad))
    for epoch in range(1, train_cfg.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, weights, amp)
        val_loss, metrics = evaluate(
            model,
            val_loader,
            device,
            min_confidence=train_cfg.min_confidence,
            round_trip_cost=feature_cfg.round_trip_cost,
        )
        scheduler.step()
        log.info(
            "epoch=%03d train=%.5f val=%.5f acc=%.3f f1=%.3f trades=%d sharpe=%.2f ret=%.2f%% dd=%.2f%%",
            epoch,
            train_loss,
            val_loss,
            metrics.accuracy,
            metrics.macro_f1,
            metrics.trades,
            metrics.sharpe,
            100.0 * metrics.total_return,
            100.0 * metrics.max_drawdown,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_metrics = metrics
            no_improve = 0
            save_checkpoint(best_path, model, prepared, feature_cfg, train_cfg, metrics, epoch)
            log.info("saved best checkpoint: %s", best_path)
        else:
            no_improve += 1
            if no_improve >= train_cfg.patience:
                log.info("early stopping after %d epochs without validation improvement", no_improve)
                break

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(best_path),
        "symbols": symbols,
        "feature_config": asdict(feature_cfg),
        "train_config": asdict(train_cfg),
        "best_val_loss": best_loss,
        "best_metrics": asdict(best_metrics) if best_metrics else None,
        "risk_note": "Validation includes configured fees/slippage. This is research software, not a profit guarantee.",
    }
    report_path = output_dir / "crypto_ai_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("wrote report: %s", report_path)
    return best_path


def save_checkpoint(
    path: Path,
    model: CryptoTransformer,
    prepared: PreparedData,
    feature_cfg: FeatureConfig,
    train_cfg: TrainConfig,
    metrics: BacktestMetrics,
    epoch: int,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "feature_cols": prepared.feature_cols,
        "symbols": prepared.symbols,
        "scaler": prepared.scaler.to_dict(),
        "feature_config": asdict(feature_cfg),
        "train_config": asdict(train_cfg),
        "metrics": asdict(metrics),
        "epoch": epoch,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> Tuple[CryptoTransformer, dict, RobustStandardizer]:
    checkpoint = torch.load(path, map_location=device)
    train_cfg = checkpoint["train_config"]
    model = CryptoTransformer(
        n_features=len(checkpoint["feature_cols"]),
        n_symbols=len(checkpoint["symbols"]),
        d_model=train_cfg["d_model"],
        n_heads=train_cfg["n_heads"],
        layers=train_cfg["layers"],
        dropout=train_cfg["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, RobustStandardizer.from_dict(checkpoint["scaler"])


@torch.no_grad()
def generate_signals(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    model, checkpoint, scaler = load_checkpoint(Path(args.checkpoint), device)
    feature_cfg = FeatureConfig(**checkpoint["feature_config"])
    symbols = checkpoint["symbols"]
    data_dir = Path(args.data_dir)
    signals = []
    for symbol_id, symbol in enumerate(symbols):
        frame = load_symbol_frame(data_dir, symbol, feature_cfg.interval, feature_cfg)
        feature_cols = checkpoint["feature_cols"]
        matrix = scaler.transform(frame[feature_cols].to_numpy(dtype=np.float32))
        if len(matrix) < feature_cfg.sequence_length:
            raise ValueError(f"{symbol}: not enough rows for sequence_length={feature_cfg.sequence_length}")
        sequence = torch.from_numpy(matrix[-feature_cfg.sequence_length :][None, :, :]).to(device)
        sid = torch.tensor([symbol_id], dtype=torch.long, device=device)
        out = model(sequence, sid)
        probs = torch.softmax(out["logits"], dim=-1)[0].cpu().numpy()
        pred = int(np.argmax(probs))
        confidence = float(np.max(probs))
        action = {0: "SHORT", 1: "HOLD", 2: "LONG"}[pred]
        if confidence < float(args.min_confidence):
            action = "HOLD"
        signals.append(
            {
                "symbol": symbol,
                "timestamp": int(frame["timestamp"].iloc[-1]),
                "datetime": datetime.fromtimestamp(frame["timestamp"].iloc[-1] / 1000, tz=timezone.utc).isoformat(),
                "action": action,
                "confidence": confidence,
                "prob_short": float(probs[0]),
                "prob_hold": float(probs[1]),
                "prob_long": float(probs[2]),
                "expected_return": float(out["mu"].cpu().item()),
                "uncertainty": float(out["sigma"].cpu().item()),
                "last_close": float(frame["close"].iloc[-1]),
            }
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "min_confidence": float(args.min_confidence),
        "signals": signals,
        "risk_note": "Signals are model outputs only. Use paper trading and exchange-side risk controls before live capital.",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("wrote signals: %s", output_path)
    for item in signals:
        log.info(
            "%s %-5s conf=%.3f pS=%.3f pH=%.3f pL=%.3f mu=%.4f close=%.4f",
            item["symbol"],
            item["action"],
            item["confidence"],
            item["prob_short"],
            item["prob_hold"],
            item["prob_long"],
            item["expected_return"],
            item["last_close"],
        )
    return output_path


def download_data(args: argparse.Namespace) -> None:
    symbols = parse_symbols(args.symbols)
    downloader = BinanceDownloader(Path(args.data_dir), market=args.market)
    for symbol in symbols:
        downloader.download_symbol(symbol, args.interval, args.bars, force=args.force)


def auto_loop(args: argparse.Namespace) -> None:
    cycle = 0
    while True:
        cycle += 1
        log.info("auto cycle %d starting", cycle)
        download_data(args)
        checkpoint = train_model(args)
        args.checkpoint = str(checkpoint)
        generate_signals(args)
        log.info("auto cycle %d complete; sleeping %.1f hours", cycle, args.retrain_hours)
        time.sleep(float(args.retrain_hours) * 3600.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GPU-capable multi-symbol crypto AI research CLI. "
            "Uses walk-forward-style chronological validation with purge gaps and trading costs."
        )
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    common_symbols = {
        "--symbols": dict(type=str, default=",".join(DEFAULT_SYMBOLS), help="Comma-separated symbols; default is 10 USDT pairs"),
        "--interval": dict(type=str, default="15m", help="Binance kline interval"),
        "--data-dir": dict(type=str, default=str(BASE_DIR / "data_cache"), help="Directory for parquet data"),
    }

    download = sub.add_parser("download", help="Download historical Binance candles")
    for flag, kwargs in common_symbols.items():
        download.add_argument(flag, **kwargs)
    download.add_argument("--market", choices=["spot", "futures"], default="futures")
    download.add_argument("--bars", type=int, default=80_000, help="Bars per symbol")
    download.add_argument("--force", action="store_true", help="Re-download even when cached data exists")
    download.set_defaults(func=download_data)

    train = sub.add_parser("train", help="Train the multi-symbol Transformer on GPU/CPU")
    add_train_args(train, common_symbols)
    train.set_defaults(func=train_model)

    signal = sub.add_parser("signal", help="Generate latest signals from a trained checkpoint")
    signal.add_argument("--checkpoint", type=str, default=str(BASE_DIR / "checkpoints" / "crypto_ai_best.pt"))
    signal.add_argument("--data-dir", type=str, default=str(BASE_DIR / "data_cache"))
    signal.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    signal.add_argument("--min-confidence", type=float, default=0.50)
    signal.add_argument("--output", type=str, default=str(BASE_DIR / "signals" / "crypto_ai_signals.json"))
    signal.set_defaults(func=generate_signals)

    auto = sub.add_parser("auto", help="Download, train, signal, sleep, and repeat")
    add_train_args(auto, common_symbols)
    auto.add_argument("--market", choices=["spot", "futures"], default="futures")
    auto.add_argument("--bars", type=int, default=80_000)
    auto.add_argument("--force", action="store_true")
    auto.add_argument("--checkpoint", type=str, default=str(BASE_DIR / "checkpoints" / "crypto_ai_best.pt"))
    auto.add_argument("--output", type=str, default=str(BASE_DIR / "signals" / "crypto_ai_signals.json"))
    auto.add_argument("--retrain-hours", type=float, default=12.0)
    auto.set_defaults(func=auto_loop)

    return parser


def add_train_args(parser: argparse.ArgumentParser, common_symbols: Dict[str, dict]) -> None:
    for flag, kwargs in common_symbols.items():
        parser.add_argument(flag, **kwargs)
    parser.add_argument("--output-dir", type=str, default=str(BASE_DIR / "checkpoints"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--label-threshold", type=float, default=0.0015)
    parser.add_argument("--fee-bps", type=float, default=6.0)
    parser.add_argument("--slippage-bps", type=float, default=4.0)
    parser.add_argument("--purge-bars", type=int, default=160)
    parser.add_argument("--val-fraction", type=float, default=0.18)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-confidence", type=float, default=0.48)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True, help="Use CUDA mixed precision")
    parser.add_argument("--no-amp", dest="amp", action="store_false")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
