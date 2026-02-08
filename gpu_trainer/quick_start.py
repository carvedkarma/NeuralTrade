#!/usr/bin/env python3
"""
BTC Futures GPU Trainer - Quick Start (v3.3.0 ENTER QUALITY + Funding + OI)
=============================================================
One-script setup: Downloads data from your Replit dashboard,
trains the ENTER QUALITY model on your GPU, and pushes predictions back.

The model predicts WHETHER to enter a trend-following trade (binary ENTER=0/1),
not WHICH direction. Direction comes from HTF (1H/4H) trend alignment.

Usage:
    python quick_start.py --url https://YOUR-APP.replit.app

That's it. Everything else is automatic.
"""

import argparse
import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("QuickStart")

FEATURE_VERSION = "v3.3.0_enter_quality_stf47_htf10_funding3_oi3"

FUNDING_FEATURE_NAMES = ["funding_rate", "funding_rate_delta_8h", "funding_rate_zscore_30d"]
FUNDING_FEATURE_COUNT = len(FUNDING_FEATURE_NAMES)

OI_FEATURE_NAMES = ["open_interest", "oi_delta_1h", "oi_zscore_30d"]
OI_FEATURE_COUNT = len(OI_FEATURE_NAMES)


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            log.info(f"GPU: {name} ({mem:.1f} GB)")
            return "cuda"
        else:
            log.warning("No GPU found - training will be slow on CPU")
            return "cpu"
    except ImportError:
        log.error("PyTorch not installed! Run: pip install -r requirements.txt")
        sys.exit(1)


def download_data(replit_url: str, data_dir: Path, force_fresh: bool = False):
    import requests

    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "BTCUSDT_15m.csv"
    parquet_path = data_dir / "BTCUSDT_15m.parquet"

    if parquet_path.exists() and not force_fresh:
        import pandas as pd
        existing = pd.read_parquet(parquet_path)
        log.info(f"Found existing data: {len(existing)} candles")
        resp = input("Re-download fresh data? (y/N): ").strip().lower()
        if resp != 'y':
            return parquet_path

    url = f"{replit_url.rstrip('/')}/api/data/export-csv?symbol=BTCUSDT&timeframe=15m"
    log.info(f"Downloading BTC 15m data from dashboard...")
    log.info(f"  URL: {url}")

    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        log.error(f"Cannot connect to {replit_url}")
        log.error("Make sure your Replit dashboard is running!")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        log.error(f"Server returned error: {e}")
        sys.exit(1)

    with open(csv_path, 'wb') as f:
        total = 0
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            total += len(chunk)
    log.info(f"  Downloaded {total / 1024:.0f} KB")

    import pandas as pd
    df = pd.read_csv(csv_path)
    log.info(f"  Loaded {len(df)} candles")

    if len(df) < 1000:
        log.error(f"Only {len(df)} candles - need at least 1,000 for training")
        log.error("Go to your dashboard's Neural Network tab and download more historical data first")
        sys.exit(1)

    date_min = datetime.fromtimestamp(df['timestamp'].min() / 1000).strftime('%Y-%m-%d')
    date_max = datetime.fromtimestamp(df['timestamp'].max() / 1000).strftime('%Y-%m-%d')
    log.info(f"  Date range: {date_min} to {date_max}")

    df.to_parquet(parquet_path, index=False)
    log.info(f"  Saved to {parquet_path}")

    csv_path.unlink(missing_ok=True)
    return parquet_path


def fetch_funding_rates(candle_df, data_dir: Path):
    """Fetch historical funding rates from Binance Futures API, paginating to cover full candle range."""
    import requests
    import pandas as pd

    cache_path = data_dir / "funding_rates.parquet"

    candle_start_ms = int(candle_df['timestamp'].min())
    candle_end_ms = int(candle_df['timestamp'].max())

    if cache_path.exists():
        existing = pd.read_parquet(cache_path)
        if len(existing) > 0:
            cached_start = existing['timestamp'].min()
            cached_end = existing['timestamp'].max()
            if cached_start <= candle_start_ms and cached_end >= candle_end_ms - 8 * 3600 * 1000:
                log.info(f"Using cached funding rates: {len(existing)} records")
                return existing

    log.info("Fetching historical funding rates from Binance Futures...")
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    all_records = []
    current_start = candle_start_ms
    page = 0

    while current_start < candle_end_ms:
        params = {
            "symbol": "BTCUSDT",
            "startTime": current_start,
            "endTime": candle_end_ms,
            "limit": 1000,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                import time as _time
                _time.sleep(2)
                continue
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning(f"Funding rate fetch error (page {page}): {e}")
            break

        if not data:
            break

        for item in data:
            all_records.append({
                "timestamp": int(item["fundingTime"]),
                "funding_rate": float(item["fundingRate"]),
            })

        last_ts = int(data[-1]["fundingTime"])
        if last_ts <= current_start:
            break
        current_start = last_ts + 1
        page += 1

        if page % 5 == 0:
            log.info(f"  Fetched {len(all_records)} funding records so far...")
        import time as _time
        _time.sleep(0.1)

    if not all_records:
        log.warning("No funding data fetched - funding features will be zero")
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    funding_df = pd.DataFrame(all_records)
    funding_df = funding_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    funding_start = funding_df['timestamp'].min()
    funding_end = funding_df['timestamp'].max()
    expected_records = (candle_end_ms - candle_start_ms) / (8 * 3600 * 1000)
    coverage_pct = len(funding_df) / max(expected_records, 1) * 100
    log.info(f"Funding coverage: {coverage_pct:.0f}% ({len(funding_df)} records for ~{expected_records:.0f} expected 8h intervals)")
    if coverage_pct < 50:
        log.warning(f"Low funding coverage ({coverage_pct:.0f}%) - some candles will have zero funding features")

    funding_df.to_parquet(cache_path, index=False)
    log.info(f"Fetched {len(funding_df)} funding rate records (cached to {cache_path})")

    return funding_df


def compute_funding_features(candle_df, funding_df):
    """Compute funding features aligned to 15m candle timestamps via merge_asof backward.

    Returns DataFrame with 3 columns: funding_rate, funding_rate_delta_8h, funding_rate_zscore_30d
    All values are z-scored/normalized and clipped to ±5.
    """
    import pandas as pd
    import numpy as np

    n = len(candle_df)

    if funding_df.empty:
        log.warning("Empty funding data - returning zero features")
        return pd.DataFrame(
            np.zeros((n, FUNDING_FEATURE_COUNT)),
            columns=FUNDING_FEATURE_NAMES,
            index=candle_df.index,
        )

    candle_ts = candle_df[['timestamp']].copy()
    candle_ts = candle_ts.reset_index(drop=True)
    candle_ts['_candle_idx'] = candle_ts.index

    funding_sorted = funding_df[['timestamp', 'funding_rate']].copy()
    funding_sorted = funding_sorted.sort_values('timestamp').reset_index(drop=True)

    funding_sorted['funding_rate_prev'] = funding_sorted['funding_rate'].shift(1)
    funding_sorted['funding_rate_delta_8h'] = funding_sorted['funding_rate'] - funding_sorted['funding_rate_prev']

    rolling_window = 90
    rolling_mean = funding_sorted['funding_rate'].rolling(rolling_window, min_periods=1).mean()
    rolling_std = funding_sorted['funding_rate'].rolling(rolling_window, min_periods=1).std().clip(lower=1e-8)
    funding_sorted['funding_rate_zscore_30d'] = (funding_sorted['funding_rate'] - rolling_mean) / rolling_std

    funding_sorted = funding_sorted.fillna(0)

    merged = pd.merge_asof(
        candle_ts.sort_values('timestamp'),
        funding_sorted[['timestamp', 'funding_rate', 'funding_rate_delta_8h', 'funding_rate_zscore_30d']],
        on='timestamp',
        direction='backward',
    )

    merged = merged.sort_values('_candle_idx').reset_index(drop=True)

    result = pd.DataFrame(index=candle_df.index)
    result['funding_rate'] = merged['funding_rate'].values * 100
    result['funding_rate_delta_8h'] = merged['funding_rate_delta_8h'].values * 100
    result['funding_rate_zscore_30d'] = merged['funding_rate_zscore_30d'].values

    result = result.fillna(0)
    result = result.clip(lower=-5, upper=5)

    n_nonzero = (result.abs() > 1e-8).any(axis=1).sum()
    log.info(f"Funding features: {n_nonzero}/{n} rows with non-zero funding data")

    import random
    sample_indices = sorted(random.sample(range(min(100, n), n), min(10, max(1, n - 100))))
    log.info("FUNDING ALIGNMENT CHECK (10 random rows):")
    log.info(f"{'Row':>8} | {'Candle TS':>15} | {'FR':>10} | {'Delta8h':>10} | {'Z30d':>10}")
    log.info("-" * 65)
    for idx in sample_indices:
        ts = candle_df.iloc[idx].get('timestamp', 0)
        fr = result.iloc[idx]['funding_rate']
        delta = result.iloc[idx]['funding_rate_delta_8h']
        zscore = result.iloc[idx]['funding_rate_zscore_30d']
        log.info(f"{idx:>8} | {int(ts):>15} | {fr:>+10.4f} | {delta:>+10.4f} | {zscore:>+10.4f}")
    log.info("-" * 65)

    return result


def fetch_open_interest_hist(candle_df, data_dir: Path, period: str = "5m"):
    """Fetch historical Open Interest from Binance Futures API, paginating to cover full candle range."""
    import requests
    import pandas as pd

    cache_path = data_dir / "open_interest_hist.parquet"

    candle_start_ms = int(candle_df['timestamp'].min())
    candle_end_ms = int(candle_df['timestamp'].max())

    if cache_path.exists():
        existing = pd.read_parquet(cache_path)
        if len(existing) > 0:
            cached_start = existing['oi_time_ms'].min()
            cached_end = existing['oi_time_ms'].max()
            cached_period = existing.iloc[0].get('period', 'unknown') if 'period' in existing.columns else 'unknown'
            period_ms = {"5m": 5*60*1000, "15m": 15*60*1000, "1h": 3600*1000}.get(cached_period, 15*60*1000)
            if cached_start <= candle_start_ms and cached_end >= candle_end_ms - period_ms:
                if cached_period != period:
                    log.info(f"Cached OI uses period={cached_period} (requested {period}) - using cached data as-is")
                log.info(f"Using cached OI data: {len(existing)} records (period={cached_period})")
                return existing

    log.info(f"Fetching historical Open Interest from Binance Futures (period={period})...")
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    all_records = []
    current_start = candle_start_ms
    page = 0

    while current_start < candle_end_ms:
        params = {
            "symbol": "BTCUSDT",
            "period": period,
            "startTime": current_start,
            "endTime": candle_end_ms,
            "limit": 500,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                import time as _time
                log.warning("OI rate limited - sleeping 3s")
                _time.sleep(3)
                continue
            if resp.status_code == 403 or resp.status_code == 451:
                if period == "5m":
                    log.warning(f"OI period={period} not available (HTTP {resp.status_code}), falling back to 15m")
                    return fetch_open_interest_hist(candle_df, data_dir, period="15m")
                elif period == "15m":
                    log.warning(f"OI period={period} not available (HTTP {resp.status_code}), falling back to 1h")
                    return fetch_open_interest_hist(candle_df, data_dir, period="1h")
                else:
                    log.error(f"OI fetch failed for all periods (HTTP {resp.status_code})")
                    return pd.DataFrame(columns=["oi_time_ms", "sumOpenInterest", "symbol", "period"])
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError as e:
            if period == "5m":
                log.warning(f"OI period={period} error: {e}, falling back to 15m")
                return fetch_open_interest_hist(candle_df, data_dir, period="15m")
            elif period == "15m":
                log.warning(f"OI period={period} error: {e}, falling back to 1h")
                return fetch_open_interest_hist(candle_df, data_dir, period="1h")
            log.warning(f"OI fetch error (page {page}): {e}")
            break
        except Exception as e:
            log.warning(f"OI fetch error (page {page}): {e}")
            break

        if not data:
            break

        for item in data:
            all_records.append({
                "oi_time_ms": int(item["timestamp"]),
                "sumOpenInterest": float(item["sumOpenInterest"]),
                "symbol": item.get("symbol", "BTCUSDT"),
                "period": period,
            })

        last_ts = int(data[-1]["timestamp"])
        if last_ts <= current_start:
            break
        current_start = last_ts + 1
        page += 1

        if page % 10 == 0:
            log.info(f"  Fetched {len(all_records)} OI records so far (page {page})...")
        import time as _time
        _time.sleep(0.2)

    if not all_records:
        log.warning("No OI data fetched - OI features will be zero")
        return pd.DataFrame(columns=["oi_time_ms", "sumOpenInterest", "symbol", "period"])

    oi_df = pd.DataFrame(all_records)
    oi_df = oi_df.drop_duplicates(subset=["oi_time_ms"]).sort_values("oi_time_ms").reset_index(drop=True)

    period_minutes = {"5m": 5, "15m": 15, "1h": 60}.get(period, 15)
    total_minutes = (candle_end_ms - candle_start_ms) / (60 * 1000)
    expected_records = total_minutes / period_minutes
    coverage_pct = len(oi_df) / max(expected_records, 1) * 100
    log.info(f"OI coverage: {coverage_pct:.0f}% ({len(oi_df)} records for ~{expected_records:.0f} expected {period} intervals)")
    if coverage_pct < 95:
        log.warning(f"Low OI coverage ({coverage_pct:.0f}%) - some candles may have zero OI features")

    oi_df.to_parquet(cache_path, index=False)
    log.info(f"Fetched {len(oi_df)} OI records (period={period}, cached to {cache_path})")

    return oi_df


def compute_oi_features(candle_df, oi_df):
    """Compute OI features aligned to 15m candle timestamps via merge_asof backward.

    Returns DataFrame with 3 columns: open_interest, oi_delta_1h, oi_zscore_30d
    All features computed on OI event series BEFORE alignment (leak-free).
    """
    import pandas as pd
    import numpy as np

    n = len(candle_df)

    if oi_df.empty or len(oi_df) < 2:
        log.warning("Empty/insufficient OI data - returning zero features")
        return pd.DataFrame(
            np.zeros((n, OI_FEATURE_COUNT)),
            columns=OI_FEATURE_NAMES,
            index=candle_df.index,
        )

    oi_sorted = oi_df[['oi_time_ms', 'sumOpenInterest']].copy()
    oi_sorted = oi_sorted.sort_values('oi_time_ms').reset_index(drop=True)

    period = oi_df['period'].iloc[0] if 'period' in oi_df.columns else '15m'
    if period == '5m':
        delta_lookback = 12
        zscore_window = 8640
    elif period == '15m':
        delta_lookback = 4
        zscore_window = 2880
    else:
        delta_lookback = 1
        zscore_window = 720

    oi_sorted['oi_delta_1h'] = oi_sorted['sumOpenInterest'] - oi_sorted['sumOpenInterest'].shift(delta_lookback)

    rolling_mean = oi_sorted['oi_delta_1h'].rolling(zscore_window, min_periods=max(delta_lookback + 1, 10)).mean()
    rolling_std = oi_sorted['oi_delta_1h'].rolling(zscore_window, min_periods=max(delta_lookback + 1, 10)).std().clip(lower=1e-8)
    oi_sorted['oi_zscore_30d'] = (oi_sorted['oi_delta_1h'] - rolling_mean) / rolling_std

    oi_sorted = oi_sorted.fillna(0)

    oi_median = oi_sorted['sumOpenInterest'].median()
    if oi_median > 0:
        oi_sorted['open_interest_scaled'] = oi_sorted['sumOpenInterest'] / oi_median
    else:
        oi_sorted['open_interest_scaled'] = oi_sorted['sumOpenInterest']

    delta_std = oi_sorted['oi_delta_1h'].std()
    if delta_std > 0:
        oi_sorted['oi_delta_1h_scaled'] = oi_sorted['oi_delta_1h'] / delta_std
    else:
        oi_sorted['oi_delta_1h_scaled'] = oi_sorted['oi_delta_1h']

    candle_ts = candle_df[['timestamp']].copy().reset_index(drop=True)
    candle_ts['_candle_idx'] = candle_ts.index

    oi_for_merge = oi_sorted[['oi_time_ms', 'open_interest_scaled', 'oi_delta_1h_scaled', 'oi_zscore_30d']].copy()
    oi_for_merge = oi_for_merge.rename(columns={'oi_time_ms': 'timestamp'})

    merged = pd.merge_asof(
        candle_ts.sort_values('timestamp'),
        oi_for_merge.sort_values('timestamp'),
        on='timestamp',
        direction='backward',
    )

    merged = merged.sort_values('_candle_idx').reset_index(drop=True)

    result = pd.DataFrame(index=candle_df.index)
    result['open_interest'] = merged['open_interest_scaled'].values
    result['oi_delta_1h'] = merged['oi_delta_1h_scaled'].values
    result['oi_zscore_30d'] = merged['oi_zscore_30d'].values

    result = result.fillna(0)
    result = result.clip(lower=-5, upper=5)

    n_nonzero = (result.abs() > 1e-8).any(axis=1).sum()
    log.info(f"OI features: {n_nonzero}/{n} rows with non-zero OI data")

    import random
    if n > 10:
        start_idx = min(100, n - 1)
        sample_pool = list(range(start_idx, n))
        sample_size = min(10, len(sample_pool))
        sample_indices = sorted(random.sample(sample_pool, sample_size)) if sample_size > 0 else []
    else:
        sample_indices = list(range(n))
    log.info("OI ALIGNMENT CHECK (10 random rows):")
    log.info(f"{'Row':>8} | {'Candle TS':>15} | {'OI':>10} | {'Delta1h':>10} | {'Z30d':>10}")
    log.info("-" * 65)
    for idx in sample_indices:
        ts = candle_df.iloc[idx].get('timestamp', 0)
        oi_val = result.iloc[idx]['open_interest']
        delta = result.iloc[idx]['oi_delta_1h']
        zscore = result.iloc[idx]['oi_zscore_30d']
        log.info(f"{idx:>8} | {int(ts):>15} | {oi_val:>+10.4f} | {delta:>+10.4f} | {zscore:>+10.4f}")
    log.info("-" * 65)

    return result


def train_enter_model(data_path: Path, device: str, epochs: int, batch_size: int, lr: float,
                      checkpoint_interval: int = 25, warmup_epochs: int = 5, min_lr: float = None,
                      tp_mult: float = 2.0, sl_mult: float = 1.5, horizon: int = 24, slope_eps: float = 0.05,
                      r_min_expiry: float = 0.5, target_tpd: float = 5.5, target_tpd_tol: float = 1.5):
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    from config import config

    log.info("=" * 60)
    log.info("  ENTER QUALITY MODEL - TRAINING")
    log.info("=" * 60)
    log.info(f"Version: {FEATURE_VERSION}")

    df = pd.read_parquet(data_path)
    log.info(f"Loaded {len(df)} candles")

    from data.pipeline import FeatureEngineer
    engineer = FeatureEngineer()
    features_df = engineer.compute_all_features(df)
    features_df = features_df.fillna(0)
    log.info(f"Computed {len(features_df.columns)} base features ({engineer.STF_FEATURE_COUNT} STF + {engineer.HTF_FEATURE_COUNT} HTF)")

    data_dir = Path("data_cache")
    funding_df = fetch_funding_rates(df, data_dir)
    funding_features = compute_funding_features(df, funding_df)
    features_df = pd.concat([features_df, funding_features], axis=1)
    features_df = features_df.fillna(0)

    oi_df = fetch_open_interest_hist(df, data_dir)
    oi_features = compute_oi_features(df, oi_df)
    features_df = pd.concat([features_df, oi_features], axis=1)
    features_df = features_df.fillna(0)

    total_features = engineer.STF_FEATURE_COUNT + engineer.HTF_FEATURE_COUNT + FUNDING_FEATURE_COUNT + OI_FEATURE_COUNT
    actual_cols = len(features_df.columns)
    log.info(f"Total features: {actual_cols} ({engineer.STF_FEATURE_COUNT} STF + {engineer.HTF_FEATURE_COUNT} HTF + {FUNDING_FEATURE_COUNT} funding + {OI_FEATURE_COUNT} OI)")
    if actual_cols != total_features:
        log.error(f"FATAL: Feature count mismatch! Expected {total_features}, got {actual_cols}")
        log.error(f"Columns: {sorted(features_df.columns.tolist())}")
        sys.exit(1)

    htf_cols = [c for c in features_df.columns if c.startswith('h1_') or c.startswith('h4_')]
    htf_features_df = features_df[htf_cols].copy()
    log.info(f"HTF features for labeling: {htf_cols}")

    from data.regression_targets import generate_enter_quality_targets
    label_df = generate_enter_quality_targets(
        df, htf_features_df,
        horizon_periods=horizon,
        tp_atr_mult=tp_mult, sl_atr_mult=sl_mult,
        slope_eps=slope_eps,
        r_min_expiry=r_min_expiry,
    )

    enter_labels = label_df['enter_label'].values.astype(np.float32)
    side_hints = label_df['side_hint'].values.astype(np.int64)
    precomputed_outcomes = label_df['outcome'].values
    precomputed_r = label_df['realized_r'].values.astype(np.float64)

    sequence_length = config.data.sequence_length
    valid_start = sequence_length
    features_np = features_df.values[valid_start:].astype(np.float32)
    enter_np = enter_labels[valid_start:].astype(np.float32)
    side_np = side_hints[valid_start:].astype(np.int64)
    outcomes_np = precomputed_outcomes[valid_start:]
    r_np = precomputed_r[valid_start:]

    n_total = len(features_np)
    purge_gap = horizon + sequence_length
    val_samples = max(int(n_total * 0.1), purge_gap)
    train_samples = n_total - purge_gap - val_samples

    if train_samples < sequence_length * 3:
        log.error(f"Not enough data for training: {train_samples} samples")
        sys.exit(1)

    train_end = train_samples
    val_start_idx = train_end + purge_gap
    val_end = val_start_idx + val_samples

    log.info(f"Data split: train={train_samples}, purge={purge_gap}, val={val_samples}")

    train_features_raw = features_np[:train_end]
    train_enter = enter_np[:train_end]
    train_side = side_np[:train_end]

    val_features_raw = features_np[val_start_idx:val_end]
    val_enter = enter_np[val_start_idx:val_end]
    val_side = side_np[val_start_idx:val_end]
    val_outcomes = outcomes_np[val_start_idx:val_end]
    val_r = r_np[val_start_idx:val_end]
    val_bars = val_samples

    train_features_df_scaled = pd.DataFrame(train_features_raw, columns=features_df.columns)
    engineer.fit_scalers(train_features_df_scaled)
    clip_range = 5.0
    train_scaled = engineer.transform_and_clip(train_features_df_scaled, clip_range=clip_range).values.astype(np.float32)
    val_features_df_scaled = pd.DataFrame(val_features_raw, columns=features_df.columns)
    val_scaled = engineer.transform_and_clip(val_features_df_scaled, clip_range=clip_range).values.astype(np.float32)

    def clean_enter(features, enter, side, outcomes, r_vals, name):
        features = np.where(np.isinf(features), np.nan, features)
        mask = np.isnan(features).any(axis=1)
        valid = ~mask
        dropped = mask.sum()
        if dropped > 0:
            log.info(f"  {name}: dropped {dropped} NaN rows")
        return features[valid], enter[valid], side[valid], outcomes[valid], r_vals[valid]

    train_outcomes_dummy = np.full(len(train_enter), "NO_CANDIDATE", dtype=object)
    train_r_dummy = np.zeros(len(train_enter), dtype=np.float64)
    train_scaled, train_enter, train_side, _, _ = clean_enter(train_scaled, train_enter, train_side, train_outcomes_dummy, train_r_dummy, "Train")
    val_scaled, val_enter, val_side, val_outcomes, val_r = clean_enter(val_scaled, val_enter, val_side, val_outcomes, val_r, "Val")

    pos_count = train_enter.sum()
    neg_count = len(train_enter) - pos_count
    pos_weight = neg_count / max(pos_count, 1)
    pos_weight = min(pos_weight, 10.0)
    log.info(f"ENTER label distribution: ENTER=1: {int(pos_count)} ({100*pos_count/len(train_enter):.1f}%), ENTER=0: {int(neg_count)} ({100*neg_count/len(train_enter):.1f}%)")
    log.info(f"BCE pos_weight: {pos_weight:.2f}")

    class EnterDataset(Dataset):
        def __init__(self, features, enter_labels, side_hints, seq_len):
            self.features = features.astype(np.float32)
            self.enter_labels = enter_labels.astype(np.float32)
            self.side_hints = side_hints.astype(np.int64)
            self.seq_len = seq_len
            self.valid_indices = list(range(seq_len, len(features)))

        def __len__(self):
            return len(self.valid_indices)

        def __getitem__(self, idx):
            actual_idx = self.valid_indices[idx]
            start = actual_idx - self.seq_len
            seq = self.features[start:actual_idx]
            return (
                torch.from_numpy(seq),
                torch.tensor(self.enter_labels[actual_idx], dtype=torch.float32),
                torch.tensor(self.side_hints[actual_idx], dtype=torch.long),
            )

    train_dataset = EnterDataset(train_scaled, train_enter, train_side, sequence_length)
    val_dataset = EnterDataset(val_scaled, val_enter, val_side, sequence_length)

    log.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    input_dim = features_np.shape[1]

    from models.simple_mlp import EnhancedMultiHeadMLP, EnhancedMultiHeadMLP_Config
    mlp_config = EnhancedMultiHeadMLP_Config(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128, 64],
        num_classes=3,
        dropout=0.3,
        use_layer_norm=True,
        use_residual=True,
        enable_quantile_head=False,
        enable_vol_state_head=False,
        enable_mu_head=False,
        enable_sigma_head=False,
        enable_enter_head=True,
    )
    model = EnhancedMultiHeadMLP(mlp_config)
    model.name = "EnterQualityMLP"
    model.to(device)
    log.info(f"Model: EnterQualityMLP ({model.parameters_count():,} parameters)")
    log.info(f"Architecture: [512, 256, 128, 64] with residual connections")
    log.info(f"Active head: enter_head (binary) | All other heads DISABLED")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    effective_min_lr = min_lr if min_lr is not None else lr * 0.05
    warmup_sched = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=effective_min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    for pg in optimizer.param_groups:
        pg['lr'] = lr * 1e-3

    log.info(f"Training for {epochs} epochs (lr={lr}, batch={batch_size})")
    log.info(f"LR schedule: {warmup_epochs}-epoch warmup -> cosine annealing to {effective_min_lr:.2e}")
    log.info(f"Early stopping: patience=50, min_epochs=40")
    log.info(f"Barriers: TP={tp_mult}x ATR, SL={sl_mult}x ATR, horizon={horizon} bars")
    log.info("-" * 60)

    best_val_loss = float('inf')
    best_val_prauc = 0.0
    patience = 0
    max_patience = 50
    min_epochs = 40
    history = {'train_loss': [], 'val_loss': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_prauc': []}

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            features_batch, enter_batch, side_batch = batch
            features_batch = features_batch.to(device)
            enter_batch = enter_batch.to(device)

            optimizer.zero_grad()
            output = model.forward_multihead(features_batch)
            enter_logits = output.enter_logits.squeeze(-1)
            loss = criterion(enter_logits, enter_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(n_batches, 1)

        model.eval()
        val_loss_total = 0
        val_n = 0
        all_probs = []
        all_targets = []
        all_sides = []

        with torch.no_grad():
            for batch in val_loader:
                features_batch, enter_batch, side_batch = batch
                features_batch = features_batch.to(device)
                enter_batch = enter_batch.to(device)

                output = model.forward_multihead(features_batch)
                enter_logits = output.enter_logits.squeeze(-1)
                v_loss = criterion(enter_logits, enter_batch)
                val_loss_total += v_loss.item()
                val_n += 1

                probs = torch.sigmoid(enter_logits).cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(enter_batch.cpu().numpy())
                all_sides.extend(side_batch.numpy())

        avg_val_loss = val_loss_total / max(val_n, 1)

        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        all_sides = np.array(all_sides)

        threshold = 0.5
        preds = (all_probs >= threshold).astype(int)
        tp = ((preds == 1) & (all_targets == 1)).sum()
        fp = ((preds == 1) & (all_targets == 0)).sum()
        fn = ((preds == 0) & (all_targets == 1)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        pos_rate = all_targets.mean()

        try:
            from sklearn.metrics import average_precision_score
            prauc = average_precision_score(all_targets, all_probs) if all_targets.sum() > 0 else 0.0
        except ImportError:
            prauc = 0.0

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)
        history['val_prauc'].append(prauc)

        current_lr = optimizer.param_groups[0]['lr']

        log.info(
            f"Epoch {epoch+1}/{epochs} | Loss T:{avg_train_loss:.4f} V:{avg_val_loss:.4f} | "
            f"P:{precision:.1%} R:{recall:.1%} F1:{f1:.1%} | PR-AUC:{prauc:.3f} | "
            f"Pos:{pos_rate:.1%} | Pred1:{preds.mean():.1%} | LR:{current_lr:.2e}"
        )

        if prauc > best_val_prauc:
            best_val_prauc = prauc
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_dim': input_dim,
                    'hidden_dims': [512, 256, 128, 64],
                    'num_classes': 3,
                    'dropout': 0.3,
                    'use_layer_norm': True,
                    'use_residual': True,
                    'enable_enter_head': True,
                    'enable_quantile_head': False,
                    'enable_vol_state_head': False,
                    'enable_mu_head': False,
                    'enable_sigma_head': False,
                },
                'feature_columns': list(features_df.columns),
                'n_features': input_dim,
                'feature_version': FEATURE_VERSION,
                'model_type': 'enter_quality',
                'barrier_config': {'tp_mult': tp_mult, 'sl_mult': sl_mult, 'horizon': horizon, 'slope_eps': slope_eps, 'r_min_expiry': r_min_expiry},
                'best_prauc': best_val_prauc,
                'trained_at': datetime.now().isoformat(),
            }, checkpoint_dir / "best_enter_prauc.pt")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_dim': input_dim,
                    'hidden_dims': [512, 256, 128, 64],
                    'num_classes': 3,
                    'dropout': 0.3,
                    'use_layer_norm': True,
                    'use_residual': True,
                    'enable_enter_head': True,
                    'enable_quantile_head': False,
                    'enable_vol_state_head': False,
                    'enable_mu_head': False,
                    'enable_sigma_head': False,
                },
                'feature_columns': list(features_df.columns),
                'n_features': input_dim,
                'feature_version': FEATURE_VERSION,
                'model_type': 'enter_quality',
                'barrier_config': {'tp_mult': tp_mult, 'sl_mult': sl_mult, 'horizon': horizon, 'slope_eps': slope_eps, 'r_min_expiry': r_min_expiry},
                'best_val_loss': best_val_loss,
                'trained_at': datetime.now().isoformat(),
            }, checkpoint_dir / "best_enter_loss.pt")
        else:
            patience += 1

        if epoch + 1 >= min_epochs and patience >= max_patience:
            log.info(f"Early stopping at epoch {epoch+1} (patience={max_patience})")
            break

        MONITORING_INTERVAL = 5
        if (epoch + 1) % MONITORING_INTERVAL == 0:
            sweep_outcomes = val_outcomes[sequence_length:]
            sweep_r = val_r[sequence_length:]
            n_sweep = min(len(all_probs), len(sweep_outcomes))
            if len(all_probs) != len(sweep_outcomes):
                log.warning(f"Sweep alignment: probs={len(all_probs)} vs outcomes={len(sweep_outcomes)}, using min={n_sweep}")
            _run_enter_trading_sweep(
                all_probs[:n_sweep], all_targets[:n_sweep], all_sides[:n_sweep],
                sweep_outcomes[:n_sweep], sweep_r[:n_sweep],
                val_bars, epoch + 1, tp_mult, sl_mult,
                target_tpd=target_tpd, target_tpd_tol=target_tpd_tol,
            )

        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0 and (epoch + 1) < epochs:
            log.info("=" * 60)
            log.info(f"  CHECKPOINT @ Epoch {epoch+1}/{epochs}")
            log.info("=" * 60)
            log.info(f"  Val Loss: {avg_val_loss:.4f} | Best: {best_val_loss:.4f}")
            log.info(f"  PR-AUC: {prauc:.3f} | Best: {best_val_prauc:.3f}")
            log.info(f"  Patience: {patience}/{max_patience}")
            log.info(f"  P:{precision:.1%} R:{recall:.1%} F1:{f1:.1%}")
            try:
                resp = input("Continue training? (Y/n): ").strip().lower()
                if resp == 'n':
                    log.info("User stopped training at checkpoint")
                    break
            except EOFError:
                pass

    scaler_path = checkpoint_dir / "scaler.joblib"
    engineer.save_scalers(str(scaler_path))
    log.info(f"Scaler saved to {scaler_path}")

    best_ckpt = checkpoint_dir / "best_enter_prauc.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        log.info(f"Loaded best PR-AUC checkpoint (PR-AUC={best_val_prauc:.3f})")

    return model, engineer, list(features_df.columns), history


def _select_trades_with_cooldown(probs, sides, precomputed_outcomes, precomputed_r, threshold, cooldown):
    """Select trades using threshold + cooldown, return precomputed outcomes for selected trades.
    
    Uses PRECOMPUTED outcomes from labeling triple-barrier (single source of truth).
    Only selects indices where side_hint != 0 (candidates) and p_enter >= threshold.
    """
    import numpy as np
    
    candidate_mask = (probs >= threshold) & (sides != 0)
    
    selected_indices = []
    last_trade = -cooldown - 1
    for i in range(len(candidate_mask)):
        if candidate_mask[i] and (i - last_trade) > cooldown:
            selected_indices.append(i)
            last_trade = i
    
    if not selected_indices:
        return np.array([]), np.array([]), np.array([], dtype=int)
    
    sel = np.array(selected_indices)
    sel_outcomes = precomputed_outcomes[sel]
    sel_r = precomputed_r[sel]
    
    valid_mask = ~np.isnan(sel_r.astype(float))
    sel_outcomes = sel_outcomes[valid_mask]
    sel_r = sel_r[valid_mask]
    sel = sel[valid_mask]
    
    return sel_outcomes, sel_r, sel




def _compute_sweep_metrics(outcomes, r_values, val_bars):
    """Compute metrics from precomputed triple-barrier outcomes.
    
    Args:
        outcomes: Array of "TP", "SL", "EXP_WIN", "EXP_LOSS" strings
        r_values: Array of realized R-multiples
        val_bars: Total number of validation bars (for trades_per_day)
    """
    import numpy as np
    
    n_trades = len(outcomes)
    if n_trades == 0:
        return {
            'trades': 0, 'expect': 0, 'winrate': 0, 'sharpe': 0, 'pf': 0,
            'avg_win_r': 0, 'avg_loss_r': 0, 'median_r': 0,
            'pct_tp': 0, 'pct_sl': 0, 'pct_exp': 0, 'trades_per_day': 0,
        }
    
    r_values = np.array(r_values, dtype=np.float64)
    
    expect = float(np.mean(r_values))
    wins = (r_values > 0).sum()
    winrate = wins / n_trades
    
    pos_r = r_values[r_values > 0]
    neg_r = r_values[r_values < 0]
    gross_profit = float(pos_r.sum()) if len(pos_r) > 0 else 0.0
    gross_loss = float(abs(neg_r.sum())) if len(neg_r) > 0 else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0
    
    avg_win_r = float(np.mean(pos_r)) if len(pos_r) > 0 else 0.0
    avg_loss_r = float(np.mean(neg_r)) if len(neg_r) > 0 else 0.0
    median_r = float(np.median(r_values))
    
    val_days = val_bars / 96.0
    trades_per_day = n_trades / val_days if val_days > 0 else 0.0
    trades_per_year = trades_per_day * 365.0
    
    std_r = float(np.std(r_values))
    if std_r > 1e-8 and n_trades > 1:
        sharpe = float(np.mean(r_values) / std_r * np.sqrt(max(trades_per_year, 1)))
    else:
        sharpe = 0.0
    
    outcomes_arr = np.array(outcomes)
    pct_tp = float((outcomes_arr == "TP").sum() / n_trades)
    pct_sl = float((outcomes_arr == "SL").sum() / n_trades)
    n_exp = ((outcomes_arr == "EXP_WIN") | (outcomes_arr == "EXP_LOSS")).sum()
    pct_exp = float(n_exp / n_trades)
    
    return {
        'trades': n_trades, 'expect': expect, 'winrate': winrate, 'sharpe': sharpe, 'pf': pf,
        'avg_win_r': avg_win_r, 'avg_loss_r': avg_loss_r, 'median_r': median_r,
        'pct_tp': pct_tp, 'pct_sl': pct_sl, 'pct_exp': pct_exp,
        'trades_per_day': trades_per_day,
    }


def _run_enter_trading_sweep(probs, targets, sides, precomputed_outcomes, precomputed_r,
                              val_bars, epoch, tp_mult, sl_mult,
                              target_tpd=5.5, target_tpd_tol=1.5, min_trades=50):
    """ENTER trading sweep using PRECOMPUTED triple-barrier outcomes (parity with labeling).
    
    Args:
        probs: Model p_enter probabilities for val set
        targets: True enter labels for val set
        sides: side_hint values for val set
        precomputed_outcomes: Outcome strings from labeling ("TP","SL","EXP_WIN","EXP_LOSS","NO_CANDIDATE")
        precomputed_r: Realized R-multiples from labeling (NaN for non-candidates)
        val_bars: Number of validation bars (for trades_per_day)
        epoch: Current epoch number
        tp_mult/sl_mult: Barrier config (for display only)
        target_tpd: Target trades per day
        target_tpd_tol: Tolerance band around target
        min_trades: Minimum trades for a valid sweep row
    """
    import numpy as np
    THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    PERCENTILES = [90, 85, 80, 75, 70]
    COOLDOWN = 4

    safe_outcomes = np.where(
        np.isin(precomputed_outcomes, ["TP", "SL", "EXP_WIN", "EXP_LOSS"]),
        precomputed_outcomes,
        "NO_CANDIDATE"
    )
    safe_r = np.where(np.isnan(precomputed_r.astype(float)), 0.0, precomputed_r.astype(float))

    sweep_results = []

    for thresh in THRESHOLDS:
        sel_outcomes, sel_r, sel_idx = _select_trades_with_cooldown(
            probs, sides, safe_outcomes, safe_r, thresh, COOLDOWN
        )
        m = _compute_sweep_metrics(sel_outcomes, sel_r, val_bars)
        m['label'] = f"{thresh*100:.0f}%"
        m['thresh'] = thresh
        sweep_results.append(m)

    active_probs = probs[sides != 0]
    if len(active_probs) > 0:
        for pct in PERCENTILES:
            pct_thresh = float(np.percentile(active_probs, pct))
            sel_outcomes, sel_r, sel_idx = _select_trades_with_cooldown(
                probs, sides, safe_outcomes, safe_r, pct_thresh, COOLDOWN
            )
            m = _compute_sweep_metrics(sel_outcomes, sel_r, val_bars)
            m['label'] = f"top{100-pct}%"
            m['thresh'] = pct_thresh
            sweep_results.append(m)

    tpd_lo = target_tpd - target_tpd_tol
    tpd_hi = target_tpd + target_tpd_tol
    best_freq_score = float('-inf')
    best_freq_label = ""
    best_any_score = float('-inf')
    best_any_label = ""

    for m in sweep_results:
        if m['trades'] < min_trades:
            continue
        if tpd_lo <= m['trades_per_day'] <= tpd_hi:
            if m['expect'] > best_freq_score:
                best_freq_score = m['expect']
                best_freq_label = m['label']
        if m['expect'] > best_any_score:
            best_any_score = m['expect']
            best_any_label = m['label']

    if best_freq_label:
        best_label = best_freq_label
        best_score = best_freq_score
    elif best_any_label:
        best_label = best_any_label
        best_score = best_any_score
    else:
        best_label = ""
        best_score = float('-inf')

    probs_arr = np.array(probs)
    p50 = float(np.percentile(probs_arr, 50))
    p75 = float(np.percentile(probs_arr, 75))
    p90 = float(np.percentile(probs_arr, 90))
    p95 = float(np.percentile(probs_arr, 95))
    p99 = float(np.percentile(probs_arr, 99))
    log.info("-" * 115)
    log.info("p_enter percentiles (val): p50=%.3f p75=%.3f p90=%.3f p95=%.3f p99=%.3f", p50, p75, p90, p95, p99)
    val_days = val_bars / 96.0
    log.info("ENTER TRADING SWEEP (epoch %d) | cooldown=%d | TP=%.1fx SL=%.1fx ATR | val_days=%.1f | target=%.1f±%.1f tpd",
             epoch, COOLDOWN, tp_mult, sl_mult, val_days, target_tpd, target_tpd_tol)
    log.info("%-8s %5s %8s %6s %6s %5s | %6s %6s %6s | %4s %4s %4s | %5s",
             "Select", "Trds", "Expect", "WR", "Shrpe", "PF",
             "WinR", "LosR", "MedR", "%TP", "%SL", "%EX", "T/Day")
    log.info("-" * 115)
    for m in sweep_results:
        in_freq = tpd_lo <= m['trades_per_day'] <= tpd_hi
        marker = ""
        if m['label'] == best_label and m['trades'] >= min_trades and best_score > float('-inf'):
            marker = " << BEST" + (" (freq)" if best_freq_label else " (any)")
        log.info("%-8s %5d %+.4f %5.1f%% %+6.2f %5.2f | %+5.2f %+5.2f %+5.2f | %3.0f%% %3.0f%% %3.0f%% | %5.1f%s",
                 m['label'], m['trades'], m['expect'], m['winrate'] * 100, m['sharpe'], m['pf'],
                 m['avg_win_r'], m['avg_loss_r'], m['median_r'],
                 m['pct_tp'] * 100, m['pct_sl'] * 100, m['pct_exp'] * 100,
                 m['trades_per_day'],
                 marker)
    log.info("-" * 115)


def make_enter_prediction(model, engineer, feature_columns, data_path, device):
    import torch
    import numpy as np
    import pandas as pd

    log.info("Generating ENTER QUALITY prediction from latest data...")

    df = pd.read_parquet(data_path)
    from data.pipeline import FeatureEngineer
    feat_engineer = FeatureEngineer()

    expected_count = FeatureEngineer.TOTAL_FEATURE_COUNT + FUNDING_FEATURE_COUNT + OI_FEATURE_COUNT
    if len(feature_columns) != expected_count:
        raise RuntimeError(
            f"FATAL: feature_columns has {len(feature_columns)} cols, expected {expected_count} (57 base + {FUNDING_FEATURE_COUNT} funding + {OI_FEATURE_COUNT} OI). "
            f"Checkpoint mismatch - retrain the model."
        )

    features_df = feat_engineer.compute_all_features(df)
    features_df = features_df.fillna(0)

    data_dir = Path("data_cache")
    funding_df = fetch_funding_rates(df, data_dir)
    funding_features = compute_funding_features(df, funding_df)
    features_df = pd.concat([features_df, funding_features], axis=1)
    features_df = features_df.fillna(0)

    oi_df = fetch_open_interest_hist(df, data_dir)
    oi_features = compute_oi_features(df, oi_df)
    features_df = pd.concat([features_df, oi_features], axis=1)
    features_df = features_df.fillna(0)

    missing = set(feature_columns) - set(features_df.columns)
    extra = set(features_df.columns) - set(feature_columns)
    if missing or extra:
        log.error(f"FATAL: Feature column mismatch!")
        if missing:
            log.error(f"  Missing: {sorted(missing)}")
        if extra:
            log.error(f"  Extra: {sorted(extra)}")
        raise RuntimeError(f"Feature column mismatch: {len(missing)} missing, {len(extra)} extra. Retrain.")

    features_df = features_df.reindex(columns=feature_columns, fill_value=0)

    last_features = features_df.iloc[-1:].copy()
    last_scaled = engineer.transform_and_clip(
        pd.DataFrame(last_features.values, columns=feature_columns),
        clip_range=5.0
    ).values.astype(np.float32)
    last_scaled = np.where(np.isinf(last_scaled), 0, last_scaled)
    last_scaled = np.where(np.isnan(last_scaled), 0, last_scaled)

    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(last_scaled).to(device)
        output = model.forward_multihead(x)

    p_enter = float(torch.sigmoid(output.enter_logits).cpu().item())

    last_row = features_df.iloc[-1]
    h1_trend = last_row.get('h1_trend_sign', 0)
    h4_trend = last_row.get('h4_trend_sign', 0)
    h1_slope = last_row.get('h1_sma20_slope', 0)
    h1_range_pos = last_row.get('h1_range_pos', 0.5)

    trend_aligned = (h1_trend == h4_trend) and (h1_trend != 0)
    slope_ok = abs(h1_slope) > 0.05
    range_ok = True
    if h1_trend > 0 and h1_range_pos < 0.2:
        range_ok = False
    if h1_trend < 0 and h1_range_pos > 0.8:
        range_ok = False

    if h1_trend > 0:
        side = "LONG"
    elif h1_trend < 0:
        side = "SHORT"
    else:
        side = "NEUTRAL"

    current_price = float(df.iloc[-1]['close'])

    atr_window = min(20, len(df) - 1)
    if atr_window < 2:
        atr = current_price * 0.005
    else:
        highs = df.iloc[-atr_window:]['high'].values
        lows = df.iloc[-atr_window:]['low'].values
        true_ranges = []
        for i in range(1, len(highs)):
            prev_close = float(df.iloc[-atr_window + i - 1]['close'])
            tr = max(float(highs[i]) - float(lows[i]),
                     abs(float(highs[i]) - prev_close),
                     abs(float(lows[i]) - prev_close))
            true_ranges.append(tr)
        atr = float(np.mean(true_ranges))

    log.info("=" * 70)
    log.info("INFERENCE DIAGNOSTICS")
    log.info("=" * 70)
    log.info(f"Current price: {current_price:.2f} | ATR(14): {atr:.2f} ({100*atr/current_price:.2f}% of price)")
    log.info(f"p_enter: {p_enter:.4f}")
    log.info(f"HTF gates:")
    log.info(f"  h1_trend_sign={h1_trend:+.0f}  h4_trend_sign={h4_trend:+.0f}  aligned={'YES' if trend_aligned else 'NO'}")
    log.info(f"  h1_sma20_slope={h1_slope:.4f}  |slope|>0.05={'YES' if slope_ok else 'NO'}")
    log.info(f"  h1_range_pos={h1_range_pos:.3f}  range_ok={'YES' if range_ok else 'NO'}")
    log.info(f"  HTF direction: {side}")
    
    feature_diagnostics = {}
    for col in ['rsi_14', 'rsi_7', 'macd', 'adx_14', 'bb_position', 'volume_ratio',
                'funding_rate', 'funding_rate_zscore_30d', 'open_interest', 'oi_delta_1h']:
        if col in features_df.columns:
            val = float(last_row.get(col, 0))
            feature_diagnostics[col] = val
    
    log.info(f"Key features: {' | '.join(f'{k}={v:.4f}' for k, v in feature_diagnostics.items())}")
    
    nan_count = int(np.isnan(last_scaled).sum()) + int(np.isinf(last_scaled).sum())
    zero_count = int((last_scaled == 0).sum())
    log.info(f"Scaled feature coverage: {len(feature_columns)} features | NaN/Inf={nan_count} | zeros={zero_count}")
    log.info("=" * 70)

    enter_threshold = 0.55
    should_trade = p_enter >= enter_threshold and trend_aligned and slope_ok and range_ok

    if should_trade:
        action = side
    else:
        action = "HOLD"

    if action == "LONG":
        sl_price = current_price - 1.5 * atr
        tp_price = current_price + 2.0 * atr
    elif action == "SHORT":
        sl_price = current_price + 1.5 * atr
        tp_price = current_price - 2.0 * atr
    else:
        sl_price = current_price - 1.0 * atr
        tp_price = current_price + 1.0 * atr

    sl_pct = abs(current_price - sl_price) / current_price
    tp_pct = abs(tp_price - current_price) / current_price
    rr = tp_pct / sl_pct if sl_pct > 0 else 1.0

    ACCOUNT_RISK_PER_TRADE = 0.02
    if sl_pct > 0:
        position_size = ACCOUNT_RISK_PER_TRADE / sl_pct * 100
    else:
        position_size = 1.0
    if p_enter > 0.75:
        position_size *= 1.25
    elif p_enter < 0.55:
        position_size *= 0.5
    position_size = min(max(position_size, 0.5), 5.0)

    confidence = p_enter
    edge = p_enter - 0.5

    prediction = {
        "action": action,
        "confidence": round(confidence, 4),
        "direction_probs": {"SHORT": round(1.0 if side == "SHORT" else 0.0, 4),
                            "HOLD": round(1.0 if action == "HOLD" else 0.0, 4),
                            "LONG": round(1.0 if side == "LONG" else 0.0, 4)},
        "quantiles": {},
        "vol_state": "neutral",
        "vol_state_probs": {"contraction": 0.33, "neutral": 0.34, "expansion": 0.33},
        "expected_return": round(edge, 6),
        "uncertainty": round(1.0 - p_enter, 6),
        "edge": round(edge, 4),
        "entry_price": round(current_price, 2),
        "stop_loss_price": round(sl_price, 2),
        "take_profit_price": round(tp_price, 2),
        "stop_loss_pct": round(sl_pct, 4),
        "take_profit_pct": round(tp_pct, 4),
        "risk_reward_ratio": round(rr, 2),
        "position_size_pct": round(position_size, 1),
        "current_price": round(current_price, 2),
        "model_name": "enter_quality_v3.3_funding_oi",
        "is_multihead": True,
        "urgency": "high" if p_enter > 0.7 and should_trade else ("medium" if should_trade else "low"),
        "suggested_order_type": "limit",
        "reasons": [],
    }

    reasons = []
    if should_trade:
        reasons.append(f"ENTER signal: p_enter={p_enter:.1%}")
        reasons.append(f"HTF trend: {side} (1H={h1_trend:+.0f}, 4H={h4_trend:+.0f})")
        if abs(h1_slope) > 0.1:
            reasons.append(f"Strong trend slope ({h1_slope:.2f})")
    else:
        if not trend_aligned:
            reasons.append("HTF trends not aligned")
        if not slope_ok:
            reasons.append(f"Weak slope ({h1_slope:.2f})")
        if not range_ok:
            reasons.append(f"Range position against trend ({h1_range_pos:.2f})")
        if p_enter < enter_threshold:
            reasons.append(f"p_enter {p_enter:.1%} < {enter_threshold:.0%} threshold")

    prediction["reasons"] = reasons if reasons else ["No signal"]

    return prediction


def push_prediction(replit_url: str, prediction: dict):
    import requests

    url = f"{replit_url.rstrip('/')}/api/gpu/push-prediction"
    log.info(f"Pushing prediction to dashboard...")
    log.info(f"  Action: {prediction['action']} | Confidence: {prediction['confidence']:.1%}")
    log.info(f"  Price: ${prediction['current_price']:,.2f}")
    log.info(f"  Entry: ${prediction['entry_price']:,.2f} | SL: ${prediction['stop_loss_price']:,.2f} | TP: ${prediction['take_profit_price']:,.2f}")

    try:
        resp = requests.post(url, json=prediction, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        log.info(f"  Pushed successfully! (id={result.get('id', '?')})")
        return True
    except Exception as e:
        log.error(f"  Failed to push: {e}")
        return False


def run_regime_eval(data_path: Path, device: str, regimes_str: str, policy_str: str,
                    cooldown: int, tp_mult: float, sl_mult: float, horizon: int,
                    slope_eps: float, r_min_expiry: float):
    """Evaluate one fixed trading policy across multiple date regimes.
    
    No re-training, no per-slice optimization. Same checkpoint, same policy for all regimes.
    Uses training/triple_barrier.py for trade scoring (parity with labeling).
    """
    import torch
    import numpy as np
    import pandas as pd
    from config import config
    from training.triple_barrier import compute_atr_14, triple_barrier_outcome_for_index

    log.info("=" * 80)
    log.info("  REGIME ROBUSTNESS EVALUATION")
    log.info("=" * 80)

    checkpoint_path = Path("checkpoints/best_enter_prauc.pt")
    if not checkpoint_path.exists():
        checkpoint_path = Path("checkpoints/best_enter_loss.pt")
    if not checkpoint_path.exists():
        log.error("No trained ENTER model found! Train first.")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_version = checkpoint.get('feature_version', 'unknown')
    if saved_version != FEATURE_VERSION:
        log.error(f"FATAL: Feature version mismatch! Model: '{saved_version}', current: '{FEATURE_VERSION}'")
        sys.exit(1)
    log.info(f"Checkpoint: {checkpoint_path.name} | Feature version: {saved_version}")

    feature_columns = checkpoint.get('feature_columns', [])
    if not feature_columns:
        log.error("FATAL: No feature_columns in checkpoint - retrain.")
        sys.exit(1)

    from models.simple_mlp import EnhancedMultiHeadMLP, EnhancedMultiHeadMLP_Config
    cfg = checkpoint.get('model_config', {})
    mlp_config = EnhancedMultiHeadMLP_Config(
        input_dim=cfg.get('input_dim', 63),
        hidden_dims=cfg.get('hidden_dims', [512, 256, 128, 64]),
        num_classes=3, dropout=0.3, use_layer_norm=True, use_residual=True,
        enable_enter_head=True, enable_quantile_head=False,
        enable_vol_state_head=False, enable_mu_head=False, enable_sigma_head=False,
    )
    model = EnhancedMultiHeadMLP(mlp_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    log.info(f"Model loaded: {model.parameters_count():,} parameters")

    from data.pipeline import FeatureEngineer
    engineer = FeatureEngineer()
    scaler_path = Path("checkpoints/scaler.joblib")
    if scaler_path.exists():
        engineer.load_scalers(str(scaler_path))
    else:
        log.error("No saved scaler found! Predictions will be unreliable.")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    log.info(f"Loaded {len(df)} candles")

    if 'timestamp' not in df.columns:
        log.error("Data must contain 'timestamp' column")
        sys.exit(1)

    features_df = engineer.compute_all_features(df)
    features_df = features_df.fillna(0)

    data_dir = Path("data_cache")
    funding_df = fetch_funding_rates(df, data_dir)
    funding_features = compute_funding_features(df, funding_df)
    features_df = pd.concat([features_df, funding_features], axis=1)
    features_df = features_df.fillna(0)

    oi_df = fetch_open_interest_hist(df, data_dir)
    oi_features = compute_oi_features(df, oi_df)
    features_df = pd.concat([features_df, oi_features], axis=1)
    features_df = features_df.fillna(0)

    missing = set(feature_columns) - set(features_df.columns)
    extra = set(features_df.columns) - set(feature_columns)
    if missing or extra:
        log.error(f"Feature column mismatch!")
        if missing:
            log.error(f"  Missing: {sorted(missing)}")
        if extra:
            log.error(f"  Extra: {sorted(extra)}")
        sys.exit(1)
    features_df = features_df.reindex(columns=feature_columns, fill_value=0)
    log.info(f"Features: {len(feature_columns)} columns")

    scaled_df = engineer.transform_and_clip(
        features_df, clip_range=5.0
    )
    scaled_np = scaled_df.values.astype(np.float32)
    scaled_np = np.where(np.isinf(scaled_np), 0, scaled_np)
    scaled_np = np.where(np.isnan(scaled_np), 0, scaled_np)

    log.info(f"Running single-row inference over {len(df)} bars (matching make_enter_prediction)...")

    all_p_enter = np.full(len(df), np.nan, dtype=np.float64)
    BATCH_SIZE = 1024

    valid_indices = list(range(len(scaled_np)))
    with torch.no_grad():
        for batch_start in range(0, len(valid_indices), BATCH_SIZE):
            batch_idx = valid_indices[batch_start:batch_start + BATCH_SIZE]
            batch_rows = scaled_np[batch_idx]
            batch_tensor = torch.FloatTensor(batch_rows).to(device)
            output = model.forward_multihead(batch_tensor)
            p_batch = torch.sigmoid(output.enter_logits).cpu().numpy().flatten()
            for k, idx in enumerate(batch_idx):
                all_p_enter[idx] = p_batch[k]

    valid_predictions = np.sum(~np.isnan(all_p_enter))
    log.info(f"Inference complete: {valid_predictions}/{len(df)} bars have predictions")

    h1_trend = features_df['h1_trend_sign'].values if 'h1_trend_sign' in features_df.columns else np.zeros(len(df))
    h4_trend = features_df['h4_trend_sign'].values if 'h4_trend_sign' in features_df.columns else np.zeros(len(df))
    h1_slope = features_df['h1_sma20_slope'].values if 'h1_sma20_slope' in features_df.columns else np.zeros(len(df))
    h1_range_pos = features_df['h1_range_pos'].values if 'h1_range_pos' in features_df.columns else np.full(len(df), 0.5)

    aligned = (h1_trend == h4_trend) & (h1_trend != 0)
    slope_ok_arr = np.abs(h1_slope) > slope_eps
    range_ok_arr = np.ones(len(df), dtype=bool)
    range_ok_arr[(h1_trend > 0) & (h1_range_pos < 0.2)] = False
    range_ok_arr[(h1_trend < 0) & (h1_range_pos > 0.8)] = False

    htf_pass = aligned & slope_ok_arr & range_ok_arr
    side_arr = np.where(h1_trend > 0, 1, np.where(h1_trend < 0, -1, 0)).astype(int)

    candidate_mask = htf_pass & (~np.isnan(all_p_enter))
    n_candidates = candidate_mask.sum()
    log.info(f"HTF-gated candidates: {n_candidates}/{valid_predictions} ({100*n_candidates/max(valid_predictions,1):.1f}%)")

    policy_type, policy_value = policy_str.split(":")
    policy_type = policy_type.lower().strip()
    policy_value_clean = policy_value.lower().replace("top", "").strip()
    policy_value_num = float(policy_value_clean)
    log.info(f"Policy: {policy_type}={policy_value} | Cooldown: {cooldown} | TP={tp_mult}x SL={sl_mult}x | Horizon={horizon} | r_min_expiry={r_min_expiry}")

    atr_full = compute_atr_14(df)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)

    timestamps_ms = df['timestamp'].values.astype(np.int64)

    regimes = []
    for regime_str in regimes_str.split(","):
        parts = regime_str.strip().split(":")
        start_str, end_str = parts[0], parts[1]
        start_dt = datetime.strptime(start_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_str, "%Y-%m-%d")
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000) + 86400 * 1000 - 1
        regimes.append((f"{start_str} to {end_str}", start_ms, end_ms))

    log.info(f"Regimes: {len(regimes)}")
    for name, s, e in regimes:
        mask = (timestamps_ms >= s) & (timestamps_ms <= e)
        log.info(f"  {name}: {mask.sum()} bars")

    all_regime_results = []
    all_selected_r = []

    for regime_name, start_ms, end_ms in regimes:
        regime_mask = (timestamps_ms >= start_ms) & (timestamps_ms <= end_ms)
        regime_indices = np.where(regime_mask)[0]

        if len(regime_indices) == 0:
            log.warning(f"  {regime_name}: No bars in range")
            all_regime_results.append({
                'name': regime_name, 'bars': 0, 'trades': 0, 'trades_per_day': 0,
                'expect': 0, 'winrate': 0, 'sharpe': 0, 'pf': 0,
                'avg_win_r': 0, 'avg_loss_r': 0, 'median_r': 0,
                'pct_tp': 0, 'pct_sl': 0, 'pct_exp': 0, 'pct_exp_win': 0, 'pct_exp_loss': 0,
            })
            continue

        regime_candidates = candidate_mask[regime_indices]
        regime_p_enter = all_p_enter[regime_indices]
        regime_sides = side_arr[regime_indices]

        if policy_type == "threshold":
            threshold = policy_value_num
        elif policy_type == "percentile":
            active_p = regime_p_enter[regime_candidates]
            if len(active_p) == 0:
                threshold = 1.0
            else:
                pct = 100.0 - policy_value_num
                threshold = float(np.percentile(active_p, max(pct, 0)))
        else:
            log.error(f"Unknown policy type: {policy_type}")
            sys.exit(1)

        trade_mask = regime_candidates & (regime_p_enter >= threshold)

        selected_local = []
        last_trade = -cooldown - 1
        for i in range(len(trade_mask)):
            if trade_mask[i] and (i - last_trade) > cooldown:
                selected_local.append(i)
                last_trade = i

        if not selected_local:
            n_bars_regime = len(regime_indices)
            all_regime_results.append({
                'name': regime_name, 'bars': n_bars_regime, 'trades': 0, 'trades_per_day': 0,
                'expect': 0, 'winrate': 0, 'sharpe': 0, 'pf': 0,
                'avg_win_r': 0, 'avg_loss_r': 0, 'median_r': 0,
                'pct_tp': 0, 'pct_sl': 0, 'pct_exp': 0, 'pct_exp_win': 0, 'pct_exp_loss': 0,
            })
            continue

        global_indices = regime_indices[np.array(selected_local)]

        outcomes = []
        r_values = []
        for gi in global_indices:
            side = int(side_arr[gi])
            atr_i = float(atr_full[gi])
            outcome, r = triple_barrier_outcome_for_index(
                highs, lows, closes, gi, side, atr_i,
                tp_mult, sl_mult, horizon, r_min_expiry,
            )
            outcomes.append(outcome)
            r_values.append(r)

        outcomes = np.array(outcomes)
        r_values = np.array(r_values, dtype=np.float64)

        valid_r = ~np.isnan(r_values)
        outcomes = outcomes[valid_r]
        r_values = r_values[valid_r]

        n_trades = len(r_values)
        n_bars_regime = len(regime_indices)
        regime_days = n_bars_regime / 96.0

        if n_trades == 0:
            all_regime_results.append({
                'name': regime_name, 'bars': n_bars_regime, 'trades': 0, 'trades_per_day': 0,
                'expect': 0, 'winrate': 0, 'sharpe': 0, 'pf': 0,
                'avg_win_r': 0, 'avg_loss_r': 0, 'median_r': 0,
                'pct_tp': 0, 'pct_sl': 0, 'pct_exp': 0, 'pct_exp_win': 0, 'pct_exp_loss': 0,
            })
            continue

        expect = float(np.mean(r_values))
        wins = (r_values > 0).sum()
        winrate = wins / n_trades
        trades_per_day = n_trades / regime_days if regime_days > 0 else 0

        pos_r = r_values[r_values > 0]
        neg_r = r_values[r_values < 0]
        gross_profit = float(pos_r.sum()) if len(pos_r) > 0 else 0.0
        gross_loss = float(abs(neg_r.sum())) if len(neg_r) > 0 else 0.0
        pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

        avg_win_r = float(np.mean(pos_r)) if len(pos_r) > 0 else 0.0
        avg_loss_r = float(np.mean(neg_r)) if len(neg_r) > 0 else 0.0
        median_r = float(np.median(r_values))

        trades_per_year = trades_per_day * 365.0
        std_r = float(np.std(r_values))
        if std_r > 1e-8 and n_trades > 1:
            sharpe = float(np.mean(r_values) / std_r * np.sqrt(max(trades_per_year, 1)))
        else:
            sharpe = 0.0

        pct_tp = float((outcomes == "TP").sum() / n_trades)
        pct_sl = float((outcomes == "SL").sum() / n_trades)
        pct_exp_win = float((outcomes == "EXP_WIN").sum() / n_trades)
        pct_exp_loss = float((outcomes == "EXP_LOSS").sum() / n_trades)
        pct_exp = pct_exp_win + pct_exp_loss

        all_selected_r.extend(r_values.tolist())

        all_regime_results.append({
            'name': regime_name, 'bars': n_bars_regime, 'trades': n_trades,
            'trades_per_day': trades_per_day, 'expect': expect, 'winrate': winrate,
            'sharpe': sharpe, 'pf': pf, 'avg_win_r': avg_win_r, 'avg_loss_r': avg_loss_r,
            'median_r': median_r, 'pct_tp': pct_tp, 'pct_sl': pct_sl, 'pct_exp': pct_exp,
            'pct_exp_win': pct_exp_win, 'pct_exp_loss': pct_exp_loss,
        })

    total_bars = sum(r['bars'] for r in all_regime_results)
    total_trades = sum(r['trades'] for r in all_regime_results)
    total_days = total_bars / 96.0
    overall_r = np.array(all_selected_r, dtype=np.float64)

    if len(overall_r) > 0:
        overall_expect = float(np.mean(overall_r))
        overall_wins = (overall_r > 0).sum()
        overall_winrate = overall_wins / len(overall_r)
        overall_tpd = total_trades / total_days if total_days > 0 else 0

        pos_r_all = overall_r[overall_r > 0]
        neg_r_all = overall_r[overall_r < 0]
        gp = float(pos_r_all.sum()) if len(pos_r_all) > 0 else 0
        gl = float(abs(neg_r_all.sum())) if len(neg_r_all) > 0 else 0
        overall_pf = gp / gl if gl > 0 else 0

        overall_avg_win = float(np.mean(pos_r_all)) if len(pos_r_all) > 0 else 0
        overall_avg_loss = float(np.mean(neg_r_all)) if len(neg_r_all) > 0 else 0
        overall_median = float(np.median(overall_r))

        tpy = overall_tpd * 365.0
        std_all = float(np.std(overall_r))
        overall_sharpe = float(np.mean(overall_r) / std_all * np.sqrt(max(tpy, 1))) if std_all > 1e-8 and len(overall_r) > 1 else 0
    else:
        overall_expect = overall_winrate = overall_tpd = overall_pf = 0
        overall_avg_win = overall_avg_loss = overall_median = overall_sharpe = 0

    log.info("")
    log.info("=" * 140)
    log.info("REGIME ROBUSTNESS RESULTS")
    log.info(f"Policy: {policy_str} | Cooldown: {cooldown} | TP={tp_mult}x SL={sl_mult}x | Horizon={horizon}")
    log.info("=" * 140)
    log.info("%-26s %6s %5s %5s %8s %6s %+7s %5s | %+5s %+5s %+5s | %4s %4s %4s %4s %4s",
             "Regime", "Bars", "Trds", "T/Day", "Expect", "WR", "Sharpe", "PF",
             "WinR", "LosR", "MedR", "%TP", "%SL", "%EX", "%EW", "%EL")
    log.info("-" * 140)

    for r in all_regime_results:
        log.info("%-26s %6d %5d %5.1f %+8.4f %5.1f%% %+7.2f %5.2f | %+5.2f %+5.2f %+5.2f | %3.0f%% %3.0f%% %3.0f%% %3.0f%% %3.0f%%",
                 r['name'], r['bars'], r['trades'], r['trades_per_day'],
                 r['expect'], r['winrate'] * 100, r['sharpe'], r['pf'],
                 r['avg_win_r'], r['avg_loss_r'], r['median_r'],
                 r['pct_tp'] * 100, r['pct_sl'] * 100, r['pct_exp'] * 100,
                 r['pct_exp_win'] * 100, r['pct_exp_loss'] * 100)

    log.info("-" * 140)
    log.info("%-26s %6d %5d %5.1f %+8.4f %5.1f%% %+7.2f %5.2f | %+5.2f %+5.2f %+5.2f |",
             "OVERALL", total_bars, total_trades, overall_tpd,
             overall_expect, overall_winrate * 100, overall_sharpe, overall_pf,
             overall_avg_win, overall_avg_loss, overall_median)
    log.info("=" * 140)

    if len(all_regime_results) > 1:
        expects = [r['expect'] for r in all_regime_results if r['trades'] > 0]
        if len(expects) > 1:
            expect_std = float(np.std(expects))
            expect_mean = float(np.mean(expects))
            log.info(f"Cross-regime consistency: mean(expect)={expect_mean:+.4f} std={expect_std:.4f} CV={expect_std/abs(expect_mean) if abs(expect_mean)>1e-8 else float('inf'):.2f}")
            positive_regimes = sum(1 for e in expects if e > 0)
            log.info(f"Profitable regimes: {positive_regimes}/{len(expects)}")

    log.info("")
    return all_regime_results


def main():
    parser = argparse.ArgumentParser(
        description="BTC Futures GPU Trainer - ENTER QUALITY Model (v3.3.0 + Funding + OI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_start.py --url https://your-app.replit.app
  python quick_start.py --url https://your-app.replit.app --epochs 300
  python quick_start.py --url https://your-app.replit.app --predict-only
  python quick_start.py --url https://your-app.replit.app --regime-eval --policy threshold:0.70 --cooldown 4
  python quick_start.py --url https://your-app.replit.app --regime-eval --policy percentile:top20
        """
    )
    parser.add_argument("--url", required=True, help="Your Replit dashboard URL")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs (default: 300)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="LR warmup epochs (default: 5)")
    parser.add_argument("--min-lr", type=float, default=None, help="Min LR for cosine annealing")
    parser.add_argument("--predict-only", action="store_true", help="Skip training, predict from saved model")
    parser.add_argument("--no-push", action="store_true", help="Train but don't push prediction")
    parser.add_argument("--checkpoint-interval", type=int, default=25, help="Pause every N epochs (0=no pausing)")
    parser.add_argument("--tp-mult", type=float, default=2.0, help="TP ATR multiplier (default: 2.0)")
    parser.add_argument("--sl-mult", type=float, default=1.5, help="SL ATR multiplier (default: 1.5)")
    parser.add_argument("--horizon", type=int, default=24, help="Horizon bars (default: 24)")
    parser.add_argument("--slope-eps", type=float, default=0.05, help="Min slope for trend gate (default: 0.05)")
    parser.add_argument("--r-min-expiry", type=float, default=0.5, help="Min R-multiple at expiry for ENTER=1 (default: 0.5)")
    parser.add_argument("--target-tpd", type=float, default=5.5, help="Target trades per day for BEST selection (default: 5.5)")
    parser.add_argument("--target-tpd-tol", type=float, default=1.5, help="Tolerance band for trades/day (default: 1.5)")
    parser.add_argument("--regime-eval", action="store_true", help="Run regime robustness evaluation (no training)")
    parser.add_argument("--regimes", type=str,
                        default="2019-01-01:2020-12-31,2021-01-01:2021-12-31,2022-01-01:2022-12-31,2023-01-01:2024-12-31",
                        help="Comma-separated date ranges as START:END (YYYY-MM-DD)")
    parser.add_argument("--policy", type=str, default="threshold:0.70",
                        help="Trade selection policy: 'threshold:0.70' or 'percentile:top20'")
    parser.add_argument("--cooldown", type=int, default=4, help="Cooldown bars after each trade (default: 4)")

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  BTC FUTURES - ENTER QUALITY MODEL v3.3.0 + FUNDING + OI")
    print("=" * 60)
    print()

    device = check_gpu()
    data_dir = Path("data_cache")

    if args.regime_eval:
        data_path = download_data(args.url, data_dir)
        run_regime_eval(
            data_path, device, args.regimes, args.policy,
            args.cooldown, args.tp_mult, args.sl_mult, args.horizon,
            args.slope_eps, args.r_min_expiry,
        )
        return

    if not args.predict_only:
        data_path = download_data(args.url, data_dir)

        model, engineer, feature_columns, history = train_enter_model(
            data_path, device, args.epochs, args.batch_size, args.lr,
            checkpoint_interval=args.checkpoint_interval,
            warmup_epochs=args.warmup_epochs, min_lr=args.min_lr,
            tp_mult=args.tp_mult, sl_mult=args.sl_mult,
            horizon=args.horizon, slope_eps=args.slope_eps,
            r_min_expiry=args.r_min_expiry,
            target_tpd=args.target_tpd, target_tpd_tol=args.target_tpd_tol,
        )

        print()
        log.info("=" * 60)
        log.info("  TRAINING COMPLETE")
        log.info("=" * 60)

        if history['val_loss']:
            best_loss = min(history['val_loss'])
            best_prauc = max(history['val_prauc']) if history['val_prauc'] else 0
            log.info(f"  Best val loss: {best_loss:.4f}")
            log.info(f"  Best PR-AUC: {best_prauc:.3f}")
            log.info(f"  Epochs trained: {len(history['val_loss'])}")
    else:
        import torch
        checkpoint_path = Path("checkpoints/best_enter_prauc.pt")
        if not checkpoint_path.exists():
            checkpoint_path = Path("checkpoints/best_enter_loss.pt")
        if not checkpoint_path.exists():
            log.error("No trained ENTER model found! Run without --predict-only first.")
            sys.exit(1)

        log.info("Downloading fresh data for prediction...")
        data_path = download_data(args.url, data_dir, force_fresh=True)

        log.info("Loading saved model...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        saved_version = checkpoint.get('feature_version', 'unknown')
        if saved_version != FEATURE_VERSION:
            log.error(f"FATAL: Feature version mismatch! Model: '{saved_version}', current: '{FEATURE_VERSION}'")
            sys.exit(1)
        log.info(f"Feature version: {saved_version} (matches)")

        feature_columns = checkpoint.get('feature_columns', [])
        if not feature_columns:
            log.error("FATAL: No feature_columns in checkpoint - retrain.")
            sys.exit(1)

        from models.simple_mlp import EnhancedMultiHeadMLP, EnhancedMultiHeadMLP_Config
        cfg = checkpoint.get('model_config', {})
        mlp_config = EnhancedMultiHeadMLP_Config(
            input_dim=cfg.get('input_dim', 63),
            hidden_dims=cfg.get('hidden_dims', [512, 256, 128, 64]),
            num_classes=3,
            dropout=0.3,
            use_layer_norm=True,
            use_residual=True,
            enable_enter_head=True,
            enable_quantile_head=False,
            enable_vol_state_head=False,
            enable_mu_head=False,
            enable_sigma_head=False,
        )
        model = EnhancedMultiHeadMLP(mlp_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        from data.pipeline import FeatureEngineer
        engineer = FeatureEngineer()
        scaler_path = Path("checkpoints/scaler.joblib")
        if scaler_path.exists():
            engineer.load_scalers(str(scaler_path))
        else:
            log.warning("No saved scaler found - prediction quality may be reduced")

    if not args.no_push:
        prediction = make_enter_prediction(model, engineer, feature_columns, data_path, device)

        print()
        log.info("=" * 60)
        log.info(f"  SIGNAL: {prediction['action']} | p_enter: {prediction['confidence']:.1%}")
        log.info("=" * 60)
        log.info(f"  Price: ${prediction['current_price']:,.2f}")
        log.info(f"  Entry: ${prediction['entry_price']:,.2f}")
        log.info(f"  SL:    ${prediction['stop_loss_price']:,.2f} ({prediction['stop_loss_pct']:.2%})")
        log.info(f"  TP:    ${prediction['take_profit_price']:,.2f} ({prediction['take_profit_pct']:.2%})")
        log.info(f"  R:R = {prediction['risk_reward_ratio']:.1f} | Position: {prediction['position_size_pct']:.1f}%")
        if prediction.get('reasons'):
            log.info(f"  Reasons: {', '.join(prediction['reasons'])}")
        log.info("=" * 60)

        is_hold = prediction['action'] == "HOLD"
        if is_hold:
            log.info("Signal: HOLD - pushing to dashboard (no trade)")
        else:
            log.info(f"Signal: {prediction['action']} PASSED - pushing to dashboard")
        push_prediction(args.url, prediction)
    else:
        log.info("Skipping prediction push (--no-push)")

    print()
    log.info("Done! Check your dashboard to see the prediction.")
    print()


if __name__ == "__main__":
    main()
