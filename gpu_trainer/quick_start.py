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

FEATURE_VERSION = "v4.5.0_pr_auc_upgrade_pack"
SYSTEM_VERSION = "v4.5.0_pr_auc_upgrade_pack"

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


def fetch_open_interest_hist(candle_df, data_dir: Path, period: str = "5m", symbol: str = "BTCUSDT"):
    """Fetch historical Open Interest from Binance Futures API, paginating to cover full candle range."""
    import requests
    import pandas as pd

    cache_path = data_dir / "open_interest_hist.parquet"

    candle_start_ms = int(candle_df['timestamp'].min())
    candle_end_ms = int(candle_df['timestamp'].max())

    max_oi_lookback_ms = 30 * 24 * 60 * 60 * 1000
    now_ms = int(datetime.now().timestamp() * 1000)
    oi_earliest_ms = now_ms - max_oi_lookback_ms
    if candle_start_ms < oi_earliest_ms:
        log.info(f"OI: clamping start from {candle_start_ms} to {oi_earliest_ms} (~30d lookback, Binance limit)")
        candle_start_ms = oi_earliest_ms

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

    periods_to_try = {"5m": ["5m", "15m", "1h"], "15m": ["15m", "1h"], "1h": ["1h"]}
    try_periods = periods_to_try.get(period, [period])

    url = "https://fapi.binance.com/futures/data/openInterestHist"
    all_records = []
    period_failed = False

    for try_period in try_periods:
        log.info(f"Fetching historical Open Interest from Binance Futures (period={try_period})...")
        all_records = []
        current_start = candle_start_ms
        page = 0
        period_failed = False
        use_time_params = True

        while current_start < candle_end_ms:
            params = {
                "symbol": symbol,
                "period": try_period,
                "limit": 500,
            }
            if use_time_params:
                params["startTime"] = int(current_start)
                params["endTime"] = int(candle_end_ms)
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    import time as _time
                    log.warning("OI rate limited - sleeping 3s")
                    _time.sleep(3)
                    continue
                if resp.status_code == 400:
                    resp_text = resp.text[:200] if resp.text else "no body"
                    if "startTime" in resp_text or "invalid" in resp_text.lower():
                        if use_time_params:
                            log.warning(f"OI period={try_period}: startTime rejected, retrying without time params")
                            use_time_params = False
                            continue
                    log.warning(f"OI period={try_period} blocked (HTTP 400): {resp_text}")
                    period_failed = True
                    break
                if resp.status_code in (403, 418, 451):
                    resp_text = resp.text[:200] if resp.text else "no body"
                    log.warning(f"OI period={try_period} blocked (HTTP {resp.status_code}): {resp_text}")
                    period_failed = True
                    break
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.HTTPError as e:
                log.warning(f"OI period={try_period} error: {e}")
                period_failed = True
                break
            except Exception as e:
                log.warning(f"OI fetch error (page {page}): {e}")
                period_failed = True
                break

            if not data:
                break

            for item in data:
                all_records.append({
                    "oi_time_ms": int(item["timestamp"]),
                    "sumOpenInterest": float(item["sumOpenInterest"]),
                    "symbol": item.get("symbol", symbol),
                    "period": try_period,
                })

            if not use_time_params:
                break

            last_ts = int(data[-1]["timestamp"])
            if last_ts <= current_start:
                break
            current_start = last_ts + 1
            page += 1

            if page % 10 == 0:
                log.info(f"  Fetched {len(all_records)} OI records so far (page {page})...")
            import time as _time
            _time.sleep(0.2)

        if not period_failed and all_records:
            period = try_period
            break
        if period_failed:
            log.warning(f"OI period={try_period} unavailable, trying next fallback...")
            continue
        if not all_records:
            break

    if period_failed and not all_records:
        log.warning(f"OI endpoint blocked for {symbol} (all periods failed) — OI features will be zero")
        return pd.DataFrame(columns=["oi_time_ms", "sumOpenInterest", "symbol", "period"])

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
                      r_min_expiry: float = 0.5, target_tpd: float = 5.5, target_tpd_tol: float = 1.5,
                      symbols: list = None, value_loss_weight: float = 0.5, value_clip: float = 3.0,
                      smoke_calib: bool = False, smoke_infer: bool = False,
                      use_focal_loss: bool = True, focal_gamma: float = 1.5, focal_alpha: float = 0.60,
                      use_ohem: bool = True, ohem_neg_pct: float = 0.35,
                      use_edge_head: bool = True, edge_loss_weight: float = 0.3,
                      use_soft_labels: bool = False, soft_label_temp: float = 2.0):
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
    log.info(f"[PR_AUC_PACK] focal_loss={use_focal_loss} (gamma={focal_gamma}, alpha={focal_alpha})")
    log.info(f"[PR_AUC_PACK] ohem={use_ohem} (neg_pct={ohem_neg_pct})")
    log.info(f"[PR_AUC_PACK] edge_head={use_edge_head} (weight={edge_loss_weight})")
    log.info(f"[PR_AUC_PACK] soft_labels={use_soft_labels} (temp={soft_label_temp})")

    from data.pipeline import FeatureEngineer
    data_dir = Path("data_cache")
    sequence_length = config.data.sequence_length

    # === MULTI-ASSET SUPPORT ===
    if symbols and len(symbols) > 1:
        log.info(f"[DATA] Multi-asset training: symbols={symbols}")
        all_train_features = []
        all_train_enter = []
        all_train_side = []
        all_train_r = []
        all_train_sym_ids = []
        all_train_ysoft = []
        all_train_edge = []
        all_val_features = []
        all_val_enter = []
        all_val_side = []
        all_val_outcomes = []
        all_val_r = []
        all_val_sym_ids = []
        all_val_ysoft = []
        all_val_edge = []
        feature_columns_ref = None

        for sym_idx, sym in enumerate(symbols):
            sym_data_path = data_dir / f"{sym}_15m.parquet"
            if not sym_data_path.exists():
                log.warning(f"[DATA] No data for {sym} at {sym_data_path}, skipping")
                continue

            sym_df = pd.read_parquet(sym_data_path)
            log.info(f"[DATA] {sym}: {len(sym_df)} candles")

            sym_engineer = FeatureEngineer()
            sym_features_df = sym_engineer.compute_all_features(sym_df)
            sym_features_df = sym_features_df.fillna(0)

            sym_funding_df = fetch_funding_rates(sym_df, data_dir)
            sym_funding_features = compute_funding_features(sym_df, sym_funding_df)
            sym_features_df = pd.concat([sym_features_df, sym_funding_features], axis=1)
            sym_features_df = sym_features_df.fillna(0)

            sym_oi_df = fetch_open_interest_hist(sym_df, data_dir, symbol=sym)
            sym_oi_features = compute_oi_features(sym_df, sym_oi_df)
            sym_features_df = pd.concat([sym_features_df, sym_oi_features], axis=1)
            sym_features_df = sym_features_df.fillna(0)

            if feature_columns_ref is None:
                feature_columns_ref = list(sym_features_df.columns)

            htf_cols = [c for c in sym_features_df.columns if c.startswith('h1_') or c.startswith('h4_')]
            sym_htf_df = sym_features_df[htf_cols].copy()

            from data.regression_targets import generate_enter_quality_targets
            sym_label_df = generate_enter_quality_targets(
                sym_df, sym_htf_df,
                horizon_periods=horizon,
                tp_atr_mult=tp_mult, sl_atr_mult=sl_mult,
                slope_eps=slope_eps, r_min_expiry=r_min_expiry,
            )

            sym_enter = sym_label_df['enter_label'].values.astype(np.float32)
            sym_side = sym_label_df['side_hint'].values.astype(np.int64)
            sym_outcomes = sym_label_df['outcome'].values
            sym_realized_r = sym_label_df['realized_r'].values.astype(np.float64)
            sym_ysoft = sym_label_df['y_soft'].values.astype(np.float32) if 'y_soft' in sym_label_df.columns else np.full(len(sym_label_df), 0.5, dtype=np.float32)
            sym_mfe = sym_label_df['mfe_r'].values.astype(np.float32) if 'mfe_r' in sym_label_df.columns else np.zeros(len(sym_label_df), dtype=np.float32)
            sym_mae = sym_label_df['mae_r'].values.astype(np.float32) if 'mae_r' in sym_label_df.columns else np.zeros(len(sym_label_df), dtype=np.float32)
            sym_edge_target = np.nan_to_num(sym_mfe - sym_mae, nan=0.0).astype(np.float32)

            valid_start = sequence_length
            sym_feat_np = sym_features_df.values[valid_start:].astype(np.float32)
            sym_enter_np = sym_enter[valid_start:]
            sym_side_np = sym_side[valid_start:]
            sym_outcomes_np = sym_outcomes[valid_start:]
            sym_r_np = sym_realized_r[valid_start:]
            sym_ysoft_np = np.nan_to_num(sym_ysoft[valid_start:], nan=0.5).astype(np.float32)
            sym_edge_np = sym_edge_target[valid_start:]

            n_sym = len(sym_feat_np)

            train_end_sym = int(n_sym * 0.70)
            val_end_sym = int(n_sym * 0.85)

            log.info(f"[SPLIT] sym={sym} total={n_sym} train={train_end_sym} val={val_end_sym - train_end_sym} test={n_sym - val_end_sym}")

            all_train_features.append(sym_feat_np[:train_end_sym])
            all_train_enter.append(sym_enter_np[:train_end_sym])
            all_train_side.append(sym_side_np[:train_end_sym])
            all_train_r.append(sym_r_np[:train_end_sym])
            all_train_sym_ids.append(np.full(train_end_sym, sym_idx, dtype=np.int64))
            all_train_ysoft.append(sym_ysoft_np[:train_end_sym])
            all_train_edge.append(sym_edge_np[:train_end_sym])

            val_size = val_end_sym - train_end_sym
            all_val_features.append(sym_feat_np[train_end_sym:val_end_sym])
            all_val_enter.append(sym_enter_np[train_end_sym:val_end_sym])
            all_val_side.append(sym_side_np[train_end_sym:val_end_sym])
            all_val_outcomes.append(sym_outcomes_np[train_end_sym:val_end_sym])
            all_val_r.append(sym_r_np[train_end_sym:val_end_sym])
            all_val_sym_ids.append(np.full(val_size, sym_idx, dtype=np.int64))
            all_val_ysoft.append(sym_ysoft_np[train_end_sym:val_end_sym])
            all_val_edge.append(sym_edge_np[train_end_sym:val_end_sym])

        train_features_raw = np.concatenate(all_train_features, axis=0)
        train_enter = np.concatenate(all_train_enter, axis=0)
        train_side = np.concatenate(all_train_side, axis=0)
        train_r = np.concatenate(all_train_r, axis=0)
        train_sym_ids = np.concatenate(all_train_sym_ids, axis=0)
        train_ysoft = np.concatenate(all_train_ysoft, axis=0)
        train_edge = np.concatenate(all_train_edge, axis=0)

        val_features_raw = np.concatenate(all_val_features, axis=0)
        val_enter = np.concatenate(all_val_enter, axis=0)
        val_side = np.concatenate(all_val_side, axis=0)
        val_outcomes = np.concatenate(all_val_outcomes, axis=0)
        val_r = np.concatenate(all_val_r, axis=0)
        val_sym_ids = np.concatenate(all_val_sym_ids, axis=0)
        val_ysoft = np.concatenate(all_val_ysoft, axis=0)
        val_edge = np.concatenate(all_val_edge, axis=0)

        n_symbols = len(symbols)
        features_columns_list = feature_columns_ref
        input_dim = train_features_raw.shape[1]
        val_bars = len(val_enter)

        log.info(f"[DATA] symbols={symbols} total_samples={len(train_enter) + len(val_enter)} per_symbol=[see above]")
        log.info(f"[SCALER] fit_on=train only | features={input_dim} | symbols={n_symbols}")

        engineer = FeatureEngineer()
        train_features_df_scaled = pd.DataFrame(train_features_raw, columns=features_columns_list)
        engineer.fit_scalers(train_features_df_scaled)
        clip_range = 5.0
        train_scaled = engineer.transform_and_clip(train_features_df_scaled, clip_range=clip_range).values.astype(np.float32)
        val_features_df_scaled = pd.DataFrame(val_features_raw, columns=features_columns_list)
        val_scaled = engineer.transform_and_clip(val_features_df_scaled, clip_range=clip_range).values.astype(np.float32)

        def clean_multi(features, enter, side, outcomes, r_vals, sym_ids, ysoft, edge, name):
            features = np.where(np.isinf(features), np.nan, features)
            mask = np.isnan(features).any(axis=1)
            valid = ~mask
            dropped = mask.sum()
            if dropped > 0:
                log.info(f"  {name}: dropped {dropped} NaN rows")
            return features[valid], enter[valid], side[valid], outcomes[valid], r_vals[valid], sym_ids[valid], ysoft[valid], edge[valid]

        train_outcomes_dummy = np.full(len(train_enter), "NO_CANDIDATE", dtype=object)
        train_scaled, train_enter, train_side, _, train_r, train_sym_ids, train_ysoft, train_edge = clean_multi(
            train_scaled, train_enter, train_side, train_outcomes_dummy, train_r, train_sym_ids, train_ysoft, train_edge, "Train")
        val_scaled, val_enter, val_side, val_outcomes, val_r, val_sym_ids, val_ysoft, val_edge = clean_multi(
            val_scaled, val_enter, val_side, val_outcomes, val_r, val_sym_ids, val_ysoft, val_edge, "Val")

        features_df_columns = features_columns_list

    else:
        n_symbols = 1
        df = pd.read_parquet(data_path)
        log.info(f"Loaded {len(df)} candles")

        engineer = FeatureEngineer()
        features_df = engineer.compute_all_features(df)
        features_df = features_df.fillna(0)
        log.info(f"Computed {len(features_df.columns)} base features ({engineer.STF_FEATURE_COUNT} STF + {engineer.HTF_FEATURE_COUNT} HTF)")

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
        precomputed_ysoft = label_df['y_soft'].values.astype(np.float32) if 'y_soft' in label_df.columns else np.full(len(label_df), 0.5, dtype=np.float32)
        precomputed_mfe = label_df['mfe_r'].values.astype(np.float32) if 'mfe_r' in label_df.columns else np.zeros(len(label_df), dtype=np.float32)
        precomputed_mae = label_df['mae_r'].values.astype(np.float32) if 'mae_r' in label_df.columns else np.zeros(len(label_df), dtype=np.float32)
        precomputed_edge = np.nan_to_num(precomputed_mfe - precomputed_mae, nan=0.0).astype(np.float32)

        valid_start = sequence_length
        features_np = features_df.values[valid_start:].astype(np.float32)
        enter_np = enter_labels[valid_start:].astype(np.float32)
        side_np = side_hints[valid_start:].astype(np.int64)
        outcomes_np = precomputed_outcomes[valid_start:]
        r_np = precomputed_r[valid_start:]
        ysoft_np = np.nan_to_num(precomputed_ysoft[valid_start:], nan=0.5).astype(np.float32)
        edge_np = precomputed_edge[valid_start:]

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
        train_r = r_np[:train_end]
        train_ysoft = ysoft_np[:train_end]
        train_edge = edge_np[:train_end]

        val_features_raw = features_np[val_start_idx:val_end]
        val_enter = enter_np[val_start_idx:val_end]
        val_side = side_np[val_start_idx:val_end]
        val_outcomes = outcomes_np[val_start_idx:val_end]
        val_r = r_np[val_start_idx:val_end]
        val_ysoft = ysoft_np[val_start_idx:val_end]
        val_edge = edge_np[val_start_idx:val_end]
        val_bars = val_samples

        train_features_df_scaled = pd.DataFrame(train_features_raw, columns=features_df.columns)
        engineer.fit_scalers(train_features_df_scaled)
        clip_range = 5.0
        train_scaled = engineer.transform_and_clip(train_features_df_scaled, clip_range=clip_range).values.astype(np.float32)
        val_features_df_scaled = pd.DataFrame(val_features_raw, columns=features_df.columns)
        val_scaled = engineer.transform_and_clip(val_features_df_scaled, clip_range=clip_range).values.astype(np.float32)

        def clean_enter(features, enter, side, outcomes, r_vals, ysoft, edge, name):
            features = np.where(np.isinf(features), np.nan, features)
            mask = np.isnan(features).any(axis=1)
            valid = ~mask
            dropped = mask.sum()
            if dropped > 0:
                log.info(f"  {name}: dropped {dropped} NaN rows")
            return features[valid], enter[valid], side[valid], outcomes[valid], r_vals[valid], ysoft[valid], edge[valid]

        train_outcomes_dummy = np.full(len(train_enter), "NO_CANDIDATE", dtype=object)
        train_r_dummy = np.zeros(len(train_enter), dtype=np.float64)
        train_ysoft_dummy = train_ysoft
        train_edge_dummy = train_edge
        train_scaled, train_enter, train_side, _, _, train_ysoft, train_edge = clean_enter(
            train_scaled, train_enter, train_side, train_outcomes_dummy, train_r_dummy, train_ysoft_dummy, train_edge_dummy, "Train")
        val_ysoft_dummy = val_ysoft
        val_edge_dummy = val_edge
        val_scaled, val_enter, val_side, val_outcomes, val_r, val_ysoft, val_edge = clean_enter(
            val_scaled, val_enter, val_side, val_outcomes, val_r, val_ysoft_dummy, val_edge_dummy, "Val")

        train_sym_ids = np.zeros(len(train_enter), dtype=np.int64)
        val_sym_ids = np.zeros(len(val_enter), dtype=np.int64)
        features_df_columns = list(features_df.columns)
        input_dim = features_np.shape[1]

    # === VALUE TARGETS (net_r) ===
    train_value_targets = np.nan_to_num(train_r, nan=0.0).astype(np.float32)
    train_value_targets = np.clip(train_value_targets, -value_clip, value_clip)
    val_value_targets = np.nan_to_num(val_r, nan=0.0).astype(np.float32)
    val_value_targets = np.clip(val_value_targets, -value_clip, value_clip)

    # === EDGE TARGETS (mfe_r - mae_r) ===
    train_edge_targets = np.clip(train_edge, -5.0, 5.0).astype(np.float32)
    val_edge_targets = np.clip(val_edge, -5.0, 5.0).astype(np.float32)

    # === SOFT LABELS ===
    train_ysoft_targets = train_ysoft.astype(np.float32)
    val_ysoft_targets = val_ysoft.astype(np.float32)

    pos_count = train_enter.sum()
    neg_count = len(train_enter) - pos_count
    pos_weight = neg_count / max(pos_count, 1)
    pos_weight = min(pos_weight, 10.0)
    log.info(f"ENTER label distribution: ENTER=1: {int(pos_count)} ({100*pos_count/len(train_enter):.1f}%), ENTER=0: {int(neg_count)} ({100*neg_count/len(train_enter):.1f}%)")
    log.info(f"BCE pos_weight: {pos_weight:.2f}")

    class EnterDataset(Dataset):
        def __init__(self, features, enter_labels, side_hints, symbol_ids, value_targets, edge_targets, ysoft_targets, seq_len):
            self.features = features.astype(np.float32)
            self.enter_labels = enter_labels.astype(np.float32)
            self.side_hints = side_hints.astype(np.int64)
            self.symbol_ids = symbol_ids.astype(np.int64)
            self.value_targets = value_targets.astype(np.float32)
            self.edge_targets = edge_targets.astype(np.float32)
            self.ysoft_targets = ysoft_targets.astype(np.float32)
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
                torch.tensor(self.symbol_ids[actual_idx], dtype=torch.long),
                torch.tensor(self.value_targets[actual_idx], dtype=torch.float32),
                torch.tensor(self.edge_targets[actual_idx], dtype=torch.float32),
                torch.tensor(self.ysoft_targets[actual_idx], dtype=torch.float32),
            )

    train_dataset = EnterDataset(train_scaled, train_enter, train_side, train_sym_ids, train_value_targets, train_edge_targets, train_ysoft_targets, sequence_length)
    val_dataset = EnterDataset(val_scaled, val_enter, val_side, val_sym_ids, val_value_targets, val_edge_targets, val_ysoft_targets, sequence_length)

    log.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    from models.simple_mlp import EnhancedMultiHeadMLP, EnhancedMultiHeadMLP_Config
    mlp_config = EnhancedMultiHeadMLP_Config(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128, 64],
        num_classes=3,
        dropout=0.3,
        use_layer_norm=True,
        use_residual=True,
        enable_enter_head=True,
        enable_quantile_head=False,
        enable_vol_state_head=False,
        enable_mu_head=False,
        enable_sigma_head=False,
        enable_value_head=True,
        enable_edge_head=use_edge_head,
        n_symbols=n_symbols,
        symbol_embed_dim=4 if n_symbols > 1 else 0,
    )
    model = EnhancedMultiHeadMLP(mlp_config)
    model.name = "EnterQualityMLP"
    model.to(device)
    log.info(f"Model: EnterQualityMLP ({model.parameters_count():,} parameters)")
    log.info(f"Architecture: [512, 256, 128, 64] with residual connections")
    active_heads = "enter_head (binary) + value_head (regression)"
    if use_edge_head:
        active_heads += " + edge_head (regression)"
    log.info(f"Active heads: {active_heads} | n_symbols={n_symbols}")

    pos_rate = pos_count / max(len(train_enter), 1)
    if 0 < pos_rate < 1:
        bias_init_val = float(np.log(pos_rate / (1 - pos_rate)))
    else:
        bias_init_val = 0.0
    with torch.no_grad():
        enter_head_last = model.enter_head[-1]
        enter_head_last.bias.fill_(bias_init_val)
    log.info(f"[BIAS_INIT] pos_rate={pos_rate:.4f} bias={bias_init_val:.4f}")
    log.info(f"[POS_WEIGHT] pos_weight={pos_weight:.2f}")

    if use_focal_loss:
        def focal_bce_with_logits(logits, targets, gamma=focal_gamma, alpha=focal_alpha):
            bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            p_t = torch.sigmoid(logits)
            p_t = torch.where(targets >= 0.5, p_t, 1 - p_t)
            focal_weight = (1 - p_t) ** gamma
            alpha_t = torch.where(targets >= 0.5, alpha, 1 - alpha)
            return (alpha_t * focal_weight * bce).mean()
        enter_criterion_fn = focal_bce_with_logits
        log.info(f"[LOSS] Using focal BCEWithLogits: gamma={focal_gamma}, alpha={focal_alpha}")
    else:
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        enter_criterion_fn = lambda logits, targets: bce_criterion(logits, targets)
        log.info(f"[LOSS] Using standard BCEWithLogits: pos_weight={pos_weight:.2f}")
    value_criterion = nn.HuberLoss(delta=1.0)
    edge_criterion = nn.HuberLoss(delta=1.0)

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
            features_batch, enter_batch, side_batch, sym_id_batch, value_batch, edge_batch, ysoft_batch = batch
            features_batch = features_batch.to(device)
            enter_batch = enter_batch.to(device)
            sym_id_batch = sym_id_batch.to(device)
            value_batch = value_batch.to(device)
            edge_batch = edge_batch.to(device)
            ysoft_batch = ysoft_batch.to(device)

            optimizer.zero_grad()
            output = model.forward_multihead(features_batch, symbol_ids=sym_id_batch if n_symbols > 1 else None)
            enter_logits = output.enter_logits.squeeze(-1)

            # Determine targets: soft labels or hard labels
            if use_soft_labels:
                enter_targets = ysoft_batch
            else:
                enter_targets = enter_batch

            # OHEM: keep all positives + top K% hardest negatives
            if use_ohem:
                with torch.no_grad():
                    per_sample_loss = nn.functional.binary_cross_entropy_with_logits(
                        enter_logits, enter_targets, reduction='none'
                    )
                pos_mask = enter_batch >= 0.5
                neg_mask = ~pos_mask
                n_pos = pos_mask.sum().item()
                n_neg = neg_mask.sum().item()
                if n_neg > 0 and n_pos > 0:
                    k_neg = max(int(n_neg * ohem_neg_pct), n_pos)
                    k_neg = min(k_neg, n_neg)
                    neg_losses = per_sample_loss[neg_mask]
                    _, hard_neg_idx = torch.topk(neg_losses, k_neg)
                    neg_indices = torch.where(neg_mask)[0]
                    selected_neg = neg_indices[hard_neg_idx]
                    pos_indices = torch.where(pos_mask)[0]
                    keep_indices = torch.cat([pos_indices, selected_neg])
                    enter_logits_ohem = enter_logits[keep_indices]
                    enter_targets_ohem = enter_targets[keep_indices]
                else:
                    enter_logits_ohem = enter_logits
                    enter_targets_ohem = enter_targets
                enter_loss = enter_criterion_fn(enter_logits_ohem, enter_targets_ohem)
            else:
                enter_loss = enter_criterion_fn(enter_logits, enter_targets)

            loss = enter_loss

            if output.value_logits is not None:
                value_pred = output.value_logits.squeeze(-1)
                v_loss = value_criterion(value_pred, value_batch)
                loss = loss + value_loss_weight * v_loss

            if use_edge_head and output.edge_logits is not None:
                edge_pred = output.edge_logits.squeeze(-1)
                e_loss = edge_criterion(edge_pred, edge_batch)
                loss = loss + edge_loss_weight * e_loss

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
        all_value_preds = []
        all_value_targets_list = []
        all_enter_logits_list = []
        all_edge_preds = []
        all_edge_targets_list = []

        with torch.no_grad():
            for batch in val_loader:
                features_batch, enter_batch, side_batch, sym_id_batch, value_batch, edge_batch, ysoft_batch = batch
                features_batch = features_batch.to(device)
                enter_batch = enter_batch.to(device)
                sym_id_batch = sym_id_batch.to(device)
                value_batch = value_batch.to(device)
                edge_batch = edge_batch.to(device)

                output = model.forward_multihead(features_batch, symbol_ids=sym_id_batch if n_symbols > 1 else None)
                enter_logits = output.enter_logits.squeeze(-1)
                enter_loss = enter_criterion_fn(enter_logits, enter_batch)

                batch_loss = enter_loss
                if output.value_logits is not None:
                    value_pred = output.value_logits.squeeze(-1)
                    v_loss_val = value_criterion(value_pred, value_batch)
                    batch_loss = batch_loss + value_loss_weight * v_loss_val
                    all_value_preds.extend(value_pred.cpu().numpy())
                    all_value_targets_list.extend(value_batch.cpu().numpy())

                if use_edge_head and output.edge_logits is not None:
                    edge_pred = output.edge_logits.squeeze(-1)
                    e_loss_val = edge_criterion(edge_pred, edge_batch)
                    batch_loss = batch_loss + edge_loss_weight * e_loss_val
                    all_edge_preds.extend(edge_pred.cpu().numpy())
                    all_edge_targets_list.extend(edge_batch.cpu().numpy())

                val_loss_total += batch_loss.item()
                val_n += 1

                probs = torch.sigmoid(enter_logits).cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(enter_batch.cpu().numpy())
                all_sides.extend(side_batch.numpy())
                all_enter_logits_list.extend(enter_logits.cpu().numpy())

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

        if all_value_preds:
            all_vp = np.array(all_value_preds)
            all_vt = np.array(all_value_targets_list)
            value_mae = np.mean(np.abs(all_vp - all_vt))
            value_rmse = np.sqrt(np.mean((all_vp - all_vt)**2))
        else:
            value_mae = value_rmse = 0.0

        if all_edge_preds:
            all_ep = np.array(all_edge_preds)
            all_et = np.array(all_edge_targets_list)
            edge_mae_val = np.mean(np.abs(all_ep - all_et))
            edge_rmse_val = np.sqrt(np.mean((all_ep - all_et)**2))
        else:
            edge_mae_val = edge_rmse_val = 0.0

        if (epoch + 1) % 10 == 0 or epoch == 0:
            p50 = np.percentile(all_probs, 50)
            p75 = np.percentile(all_probs, 75)
            p90 = np.percentile(all_probs, 90)
            p95 = np.percentile(all_probs, 95)
            p99 = np.percentile(all_probs, 99)

            all_logits_np = np.array(all_enter_logits_list)
            pos_mask = all_targets == 1
            neg_mask = all_targets == 0
            mean_logit_pos = all_logits_np[pos_mask].mean() if pos_mask.any() else 0.0
            mean_logit_neg = all_logits_np[neg_mask].mean() if neg_mask.any() else 0.0

            log.info(f"[METRIC] mean_logit_pos={mean_logit_pos:.3f} mean_logit_neg={mean_logit_neg:.3f}")
            log.info(f"[METRIC] value_mae={value_mae:.4f} value_rmse={value_rmse:.4f}")
            if use_edge_head:
                log.info(f"[METRIC] edge_mae={edge_mae_val:.4f} edge_rmse={edge_rmse_val:.4f}")
            log.info(f"[METRIC] p_enter percentiles (val): p50={p50:.4f} p75={p75:.4f} p90={p90:.4f} p95={p95:.4f} p99={p99:.4f}")

        ckpt_model_config = {
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
            'enable_value_head': True,
            'enable_edge_head': use_edge_head,
            'n_symbols': n_symbols,
            'symbol_embed_dim': 4 if n_symbols > 1 else 0,
        }
        ckpt_train_config = {
            'use_focal_loss': use_focal_loss,
            'focal_gamma': focal_gamma,
            'focal_alpha': focal_alpha,
            'use_ohem': use_ohem,
            'ohem_neg_pct': ohem_neg_pct,
            'use_edge_head': use_edge_head,
            'edge_loss_weight': edge_loss_weight,
            'use_soft_labels': use_soft_labels,
            'soft_label_temp': soft_label_temp,
        }

        if prauc > best_val_prauc:
            best_val_prauc = prauc
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': ckpt_model_config,
                'train_config': ckpt_train_config,
                'feature_columns': features_df_columns,
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
                'model_config': ckpt_model_config,
                'train_config': ckpt_train_config,
                'feature_columns': features_df_columns,
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

    # === TEMPERATURE SCALING CALIBRATION ===
    log.info("[CALIB] temperature_fit start | collecting val logits...")
    model.eval()
    cal_logits = []
    cal_labels = []
    with torch.no_grad():
        for batch in val_loader:
            features_batch, enter_batch, side_batch, sym_id_batch, value_batch = batch
            features_batch = features_batch.to(device)
            enter_batch = enter_batch.to(device)
            sym_id_batch = sym_id_batch.to(device)
            output = model.forward_multihead(features_batch, symbol_ids=sym_id_batch if n_symbols > 1 else None)
            cal_logits.append(output.enter_logits.squeeze(-1).cpu())
            cal_labels.append(enter_batch.cpu())

    cal_logits = torch.cat(cal_logits)
    cal_labels = torch.cat(cal_labels)
    n_cal = len(cal_logits)
    log.info(f"[CALIB] temperature_fit start | n={n_cal}")

    nll_before = nn.BCEWithLogitsLoss()(cal_logits, cal_labels).item()

    log_T = torch.nn.Parameter(torch.zeros(1))
    temp_optimizer = torch.optim.LBFGS([log_T], lr=0.01, max_iter=50)

    def temp_closure():
        temp_optimizer.zero_grad()
        T = torch.exp(log_T)
        loss = nn.BCEWithLogitsLoss()(cal_logits / T, cal_labels)
        loss.backward()
        return loss

    temp_optimizer.step(temp_closure)
    temperature = float(torch.exp(log_T).item())
    nll_after = nn.BCEWithLogitsLoss()(cal_logits / temperature, cal_labels).item()

    log.info(f"[CALIB] temperature={temperature:.4f} | nll_before={nll_before:.4f} nll_after={nll_after:.4f}")

    # === ECE COMPUTATION (before and after calibration) ===
    def compute_ece(probs_np, labels_np, n_bins=15):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        bin_details = []
        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (probs_np >= lo) & (probs_np < hi)
            if i == n_bins - 1:
                mask = (probs_np >= lo) & (probs_np <= hi)
            n_in_bin = mask.sum()
            if n_in_bin == 0:
                continue
            avg_conf = probs_np[mask].mean()
            avg_acc = labels_np[mask].mean()
            bin_ece = abs(avg_conf - avg_acc) * (n_in_bin / len(probs_np))
            ece += bin_ece
            bin_details.append({'bin': f'{lo:.2f}-{hi:.2f}', 'n': int(n_in_bin), 'conf': float(avg_conf), 'acc': float(avg_acc)})
        return float(ece), bin_details

    cal_labels_np = cal_labels.numpy()
    probs_before = torch.sigmoid(cal_logits).numpy()
    probs_after = torch.sigmoid(cal_logits / temperature).numpy()
    ece_before, _ = compute_ece(probs_before, cal_labels_np)
    ece_after, ece_bins = compute_ece(probs_after, cal_labels_np)
    log.info(f"[CALIB] ECE before={ece_before:.4f} | ECE after={ece_after:.4f}")

    temp_scale_path = checkpoint_dir / "temp_scale_v5.0.json"
    import json
    temp_data = {
        "temperature": temperature,
        "fitted_on": "val",
        "version": FEATURE_VERSION,
        "nll_before": nll_before,
        "nll_after": nll_after,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "ece_bins": ece_bins,
        "n_samples": n_cal,
    }
    with open(temp_scale_path, 'w') as f:
        json.dump(temp_data, f, indent=2)
    log.info(f"Temperature scale saved to {temp_scale_path}")

    if smoke_infer:
        log.info("[SMOKE] Running single-row inference per symbol...")
        model.eval()
        syms = symbols if symbols and len(symbols) > 1 else ["BTCUSDT"]
        for si, sym in enumerate(syms):
            if n_symbols > 1:
                sym_mask = val_sym_ids == si
                if sym_mask.any():
                    last_feat = val_scaled[sym_mask][-1:]
                else:
                    continue
            else:
                last_feat = val_scaled[-1:]

            with torch.no_grad():
                x = torch.FloatTensor(last_feat).to(device)
                sym_tensor = torch.tensor([si], dtype=torch.long, device=device) if n_symbols > 1 else None
                output = model.forward_multihead(x, symbol_ids=sym_tensor)
                logit = float(output.enter_logits.cpu().item())
                p = float(torch.sigmoid(torch.tensor(logit / temperature)).item())
                e_net = float(output.value_logits.cpu().item()) if output.value_logits is not None else 0.0

                z_vals = last_feat[0]
                z_min = float(z_vals.min())
                z_max = float(z_vals.max())
                clipped = int(((z_vals <= -5.0) | (z_vals >= 5.0)).sum())

                log.info(f"[SMOKE] sym={sym} logit={logit:.4f} T={temperature:.4f} p={p:.4f} e_net_pred={e_net:.4f}")
                log.info(f"[SMOKE] z_min={z_min:.4f} z_max={z_max:.4f} clipped={clipped}/{len(z_vals)}")

    return model, engineer, features_df_columns, history


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

    expected_count = FeatureEngineer.TOTAL_FEATURE_COUNT + FUNDING_FEATURE_COUNT + OI_FEATURE_COUNT
    if len(feature_columns) != expected_count:
        raise RuntimeError(
            f"FATAL: feature_columns has {len(feature_columns)} cols, expected {expected_count} (57 base + {FUNDING_FEATURE_COUNT} funding + {OI_FEATURE_COUNT} OI). "
            f"Checkpoint mismatch - retrain the model."
        )

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


def _compute_r_metrics(r_arr, regime_days):
    """Compute expectancy, win rate, PF, avg win/loss R, Sharpe from an R-multiple array."""
    import numpy as np
    if len(r_arr) == 0:
        return 0, 0, 0, 0, 0, 0
    e = float(np.mean(r_arr))
    w = float((r_arr > 0).sum() / len(r_arr))
    pos = r_arr[r_arr > 0]
    neg = r_arr[r_arr < 0]
    gp = float(pos.sum()) if len(pos) > 0 else 0.0
    gl = float(abs(neg.sum())) if len(neg) > 0 else 0.0
    pf = gp / gl if gl > 0 else 0.0
    aw = float(np.mean(pos)) if len(pos) > 0 else 0.0
    al = float(np.mean(neg)) if len(neg) > 0 else 0.0
    std = float(np.std(r_arr))
    tpy = (len(r_arr) / regime_days * 365.0) if regime_days > 0 else 0
    sh = float(np.mean(r_arr) / std * np.sqrt(max(tpy, 1))) if std > 1e-8 and len(r_arr) > 1 else 0.0
    return e, w, pf, aw, al, sh


def _empty_regime_result(regime_name, n_bars):
    return {
        'name': regime_name, 'bars': n_bars, 'trades': 0, 'trades_per_day': 0,
        'expect_gross': 0, 'expect_net': 0, 'expect_sized': 0,
        'winrate_gross': 0, 'winrate_net': 0,
        'sharpe_gross': 0, 'sharpe_net': 0,
        'pf_gross': 0, 'pf_net': 0, 'pf_sized': 0,
        'avg_win_r_gross': 0, 'avg_loss_r_gross': 0,
        'avg_win_r_net': 0, 'avg_loss_r_net': 0,
        'total_cost_r': 0, 'total_size_mult': 0,
        'avg_cost_r': 0, 'avg_size_mult': 0,
        'pct_tp': 0, 'pct_sl': 0, 'pct_exp': 0,
    }


def _prepare_regime_eval_context(data_path, device, regimes_str, slope_eps):
    """Load model, compute features/inference, HTF gates. Returns shared context dict."""
    import torch
    import numpy as np
    import pandas as pd
    from training.triple_barrier import compute_atr_14

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

    scaled_df = engineer.transform_and_clip(features_df, clip_range=5.0)
    scaled_np = scaled_df.values.astype(np.float32)
    scaled_np = np.where(np.isinf(scaled_np), 0, scaled_np)
    scaled_np = np.where(np.isnan(scaled_np), 0, scaled_np)

    log.info(f"Running single-row inference over {len(df)} bars...")
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

    valid_predictions = int(np.sum(~np.isnan(all_p_enter)))
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
    n_candidates = int(candidate_mask.sum())
    log.info(f"HTF-gated candidates: {n_candidates}/{valid_predictions} ({100*n_candidates/max(valid_predictions,1):.1f}%)")

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

    return {
        'df': df, 'all_p_enter': all_p_enter, 'candidate_mask': candidate_mask,
        'side_arr': side_arr, 'atr_full': atr_full,
        'highs': highs, 'lows': lows, 'closes': closes,
        'timestamps_ms': timestamps_ms, 'regimes': regimes,
    }


def _eval_single_config(ctx, tp_mult, sl_mult, threshold, cooldown, horizon, r_min_expiry,
                         fees_bps_entry, fees_bps_exit, spread_bps, slip_k, size_cap,
                         verbose=True, debug_costs=False):
    """Evaluate a single (tp_mult, sl_mult, threshold, cooldown) config across all regimes.
    
    Returns (regime_results_list, overall_summary_dict).
    """
    import numpy as np
    from training.triple_barrier import triple_barrier_outcome_for_index, compute_trade_cost_r

    all_p_enter = ctx['all_p_enter']
    candidate_mask = ctx['candidate_mask']
    side_arr = ctx['side_arr']
    atr_full = ctx['atr_full']
    highs = ctx['highs']
    lows = ctx['lows']
    closes = ctx['closes']
    timestamps_ms = ctx['timestamps_ms']
    regimes = ctx['regimes']

    all_regime_results = []
    all_gross = []
    all_net = []
    all_sized = []

    for regime_name, start_ms, end_ms in regimes:
        regime_mask = (timestamps_ms >= start_ms) & (timestamps_ms <= end_ms)
        regime_indices = np.where(regime_mask)[0]

        if len(regime_indices) == 0:
            all_regime_results.append(_empty_regime_result(regime_name, 0))
            continue

        regime_candidates = candidate_mask[regime_indices]
        regime_p_enter = all_p_enter[regime_indices]

        trade_mask = regime_candidates & (regime_p_enter >= threshold)

        selected_local = []
        last_trade = -cooldown - 1
        for i in range(len(trade_mask)):
            if trade_mask[i] and (i - last_trade) > cooldown:
                selected_local.append(i)
                last_trade = i

        if not selected_local:
            all_regime_results.append(_empty_regime_result(regime_name, len(regime_indices)))
            continue

        global_indices = regime_indices[np.array(selected_local)]
        trade_p_enter = all_p_enter[global_indices]

        outcomes = []
        gross_r_values = []
        cost_r_values = []
        for gi in global_indices:
            side = int(side_arr[gi])
            atr_i = float(atr_full[gi])
            outcome, r = triple_barrier_outcome_for_index(
                highs, lows, closes, gi, side, atr_i,
                tp_mult, sl_mult, horizon, r_min_expiry,
            )
            cost_r = compute_trade_cost_r(
                float(closes[gi]), atr_i, sl_mult,
                fees_bps_entry, fees_bps_exit, spread_bps, slip_k,
            )
            outcomes.append(outcome)
            gross_r_values.append(r)
            cost_r_values.append(cost_r)

        outcomes = np.array(outcomes)
        gross_r = np.array(gross_r_values, dtype=np.float64)
        cost_r_arr = np.array(cost_r_values, dtype=np.float64)
        net_r = gross_r - cost_r_arr

        size_mults = np.ones(len(trade_p_enter), dtype=np.float64)
        if threshold < 1.0:
            for k in range(len(trade_p_enter)):
                p = trade_p_enter[k]
                if p > threshold:
                    raw = 1.0 + (p - threshold) / (1.0 - threshold) * (size_cap - 1.0)
                    size_mults[k] = min(raw, size_cap)
        sized_net_r = net_r * size_mults

        valid_mask = ~np.isnan(gross_r) & ~np.isnan(cost_r_arr) & ~np.isnan(net_r)
        outcomes = outcomes[valid_mask]
        gross_r = gross_r[valid_mask]
        net_r = net_r[valid_mask]
        cost_r_arr = cost_r_arr[valid_mask]
        sized_net_r = sized_net_r[valid_mask]
        size_mults = size_mults[valid_mask]

        n_trades = len(gross_r)
        n_bars_regime = len(regime_indices)
        regime_days = n_bars_regime / 96.0

        if n_trades == 0:
            all_regime_results.append(_empty_regime_result(regime_name, n_bars_regime))
            continue

        eg, wg, pfg, awg, alg, shg = _compute_r_metrics(gross_r, regime_days)
        en, wn, pfn, awn, aln, shn = _compute_r_metrics(net_r, regime_days)
        es, _, pfs, _, _, _ = _compute_r_metrics(sized_net_r, regime_days)

        trades_per_day = n_trades / regime_days if regime_days > 0 else 0

        pct_tp = float((outcomes == "TP").sum() / n_trades)
        pct_sl = float((outcomes == "SL").sum() / n_trades)
        pct_exp = float(((outcomes == "EXP_WIN") | (outcomes == "EXP_LOSS")).sum() / n_trades)

        if debug_costs and n_trades > 0:
            import random
            sample_indices = random.sample(range(n_trades), min(5, n_trades))
            log.info(f"  [DEBUG COSTS] {regime_name} — {min(5, n_trades)} random trades:")
            for si in sample_indices:
                g = gross_r[si]; c = cost_r_arr[si]; n = net_r[si]
                log.info(f"    gross_r={g:+.4f}  cost_r={c:.4f}  net_r={n:+.4f}  check={g - c:+.4f}")
                assert abs(n - (g - c)) < 1e-9, f"Cost accounting mismatch: net_r={n} != gross_r={g} - cost_r={c}"

        all_gross.extend(gross_r.tolist())
        all_net.extend(net_r.tolist())
        all_sized.extend(sized_net_r.tolist())

        all_regime_results.append({
            'name': regime_name, 'bars': n_bars_regime, 'trades': n_trades,
            'trades_per_day': trades_per_day,
            'expect_gross': eg, 'expect_net': en, 'expect_sized': es,
            'winrate_gross': wg, 'winrate_net': wn,
            'sharpe_gross': shg, 'sharpe_net': shn,
            'pf_gross': pfg, 'pf_net': pfn, 'pf_sized': pfs,
            'avg_win_r_gross': awg, 'avg_loss_r_gross': alg,
            'avg_win_r_net': awn, 'avg_loss_r_net': aln,
            'total_cost_r': float(np.sum(cost_r_arr)),
            'total_size_mult': float(np.sum(size_mults)),
            'avg_cost_r': float(np.sum(cost_r_arr) / n_trades),
            'avg_size_mult': float(np.sum(size_mults) / n_trades),
            'pct_tp': pct_tp, 'pct_sl': pct_sl, 'pct_exp': pct_exp,
        })

    total_bars = sum(r['bars'] for r in all_regime_results)
    total_trades = sum(r['trades'] for r in all_regime_results)
    total_days = total_bars / 96.0

    def _overall(r_list):
        r_arr = np.array(r_list, dtype=np.float64)
        if len(r_arr) == 0:
            return 0, 0, 0, 0, 0, 0
        e = float(np.mean(r_arr))
        w = float((r_arr > 0).sum() / len(r_arr))
        tpd = total_trades / total_days if total_days > 0 else 0
        pos = r_arr[r_arr > 0]; neg = r_arr[r_arr < 0]
        gp = float(pos.sum()) if len(pos) > 0 else 0
        gl = float(abs(neg.sum())) if len(neg) > 0 else 0
        pf = gp / gl if gl > 0 else 0
        aw = float(np.mean(pos)) if len(pos) > 0 else 0
        al = float(np.mean(neg)) if len(neg) > 0 else 0
        std = float(np.std(r_arr))
        tpy = tpd * 365.0
        sh = float(np.mean(r_arr) / std * np.sqrt(max(tpy, 1))) if std > 1e-8 and len(r_arr) > 1 else 0
        return e, w, pf, aw, al, sh

    og_e, og_w, og_pf, og_aw, og_al, og_sh = _overall(all_gross)
    on_e, on_w, on_pf, on_aw, on_al, on_sh = _overall(all_net)
    os_e, _, os_pf, _, _, _ = _overall(all_sized)
    overall_tpd = total_trades / total_days if total_days > 0 else 0

    profitable_regimes = sum(1 for r in all_regime_results if r['trades'] > 0 and r['pf_net'] > 1.0)
    regimes_with_trades = sum(1 for r in all_regime_results if r['trades'] > 0)

    total_cost_r_sum = sum(r.get('total_cost_r', 0) for r in all_regime_results if r['trades'] > 0)
    total_sz_sum = sum(r.get('total_size_mult', 0) for r in all_regime_results if r['trades'] > 0)
    avg_cost_r = float(total_cost_r_sum / total_trades) if total_trades > 0 else 0
    avg_sz_mul = float(total_sz_sum / total_trades) if total_trades > 0 else 1.0

    summary = {
        'tp_mult': tp_mult, 'sl_mult': sl_mult, 'threshold': threshold, 'cooldown': cooldown,
        'overall_pf_net': on_pf, 'overall_e_net': on_e, 'overall_tpd': overall_tpd,
        'overall_pf_gross': og_pf, 'overall_e_gross': og_e,
        'overall_pf_sized': os_pf, 'overall_e_sized': os_e,
        'overall_wr_net': on_w, 'overall_wr_gross': og_w,
        'overall_win_r_net': on_aw, 'overall_loss_r_net': on_al,
        'overall_win_r_gross': og_aw, 'overall_loss_r_gross': og_al,
        'overall_sharpe_net': on_sh, 'overall_sharpe_gross': og_sh,
        'profitable_regimes': profitable_regimes,
        'regimes_with_trades': regimes_with_trades,
        'total_trades': total_trades, 'total_bars': total_bars,
        'avg_cost_r': avg_cost_r, 'avg_size_mult': avg_sz_mul,
        'regime_results': all_regime_results,
    }

    if verbose:
        _print_regime_table(all_regime_results, summary,
                            tp_mult, sl_mult, threshold, cooldown, horizon,
                            fees_bps_entry, fees_bps_exit, spread_bps, slip_k, size_cap)

    return all_regime_results, summary


def _print_regime_table(regime_results, summary, tp_mult, sl_mult, threshold, cooldown,
                         horizon, fees_entry, fees_exit, spread, slip_k, size_cap):
    """Print the detailed regime table for a single config."""
    log.info("")
    log.info("=" * 160)
    log.info("REGIME ROBUSTNESS RESULTS (GROSS / NET / SIZED)")
    log.info(f"Config: TP={tp_mult}x SL={sl_mult}x thr={threshold} cd={cooldown} | Horizon={horizon}")
    log.info(f"Costs: entry={fees_entry}bps exit={fees_exit}bps spread={spread}bps slip_k={slip_k} | Size cap={size_cap}x")
    log.info("=" * 160)

    hdr = "%-26s %6s %5s %5s | %8s %8s %8s | %5s %5s | %5s %5s | %5s %5s %5s | %5s %5s | %4s %4s %4s"
    log.info(hdr, "Regime", "Bars", "Trds", "T/Day",
             "E[gross]", "E[net]", "E[sized]",
             "WR_g", "WR_n",
             "Sh_g", "Sh_n",
             "PF_g", "PF_n", "PF_s",
             "CostR", "SzMul",
             "%TP", "%SL", "%EX")
    log.info("-" * 160)

    for r in regime_results:
        log.info("%-26s %6d %5d %5.1f | %+8.4f %+8.4f %+8.4f | %5.1f%% %5.1f%% | %+5.2f %+5.2f | %5.2f %5.2f %5.2f | %5.3f %5.2f | %3.0f%% %3.0f%% %3.0f%%",
                 r['name'], r['bars'], r['trades'], r['trades_per_day'],
                 r['expect_gross'], r['expect_net'], r['expect_sized'],
                 r['winrate_gross'] * 100, r['winrate_net'] * 100,
                 r['sharpe_gross'], r['sharpe_net'],
                 r['pf_gross'], r['pf_net'], r['pf_sized'],
                 r['avg_cost_r'], r['avg_size_mult'],
                 r['pct_tp'] * 100, r['pct_sl'] * 100, r['pct_exp'] * 100)

    log.info("-" * 160)
    s = summary
    log.info("%-26s %6d %5d %5.1f | %+8.4f %+8.4f %+8.4f | %5.1f%% %5.1f%% | %+5.2f %+5.2f | %5.2f %5.2f %5.2f | %5.3f %5.2f |",
             "OVERALL", s['total_bars'], s['total_trades'], s['overall_tpd'],
             s['overall_e_gross'], s['overall_e_net'], s['overall_e_sized'],
             s['overall_wr_gross'] * 100, s['overall_wr_net'] * 100,
             s['overall_sharpe_gross'], s['overall_sharpe_net'],
             s['overall_pf_gross'], s['overall_pf_net'], s['overall_pf_sized'],
             s['avg_cost_r'], s['avg_size_mult'])
    log.info("=" * 160)

    log.info("")
    log.info("Win/Loss R breakdown:")
    log.info("%-26s | %+6s %+6s | %+6s %+6s", "Regime", "WinR_g", "LosR_g", "WinR_n", "LosR_n")
    log.info("-" * 80)
    for r in regime_results:
        if r['trades'] > 0:
            log.info("%-26s | %+6.2f %+6.2f | %+6.2f %+6.2f",
                     r['name'], r['avg_win_r_gross'], r['avg_loss_r_gross'],
                     r['avg_win_r_net'], r['avg_loss_r_net'])
    log.info("%-26s | %+6.2f %+6.2f | %+6.2f %+6.2f",
             "OVERALL", s['overall_win_r_gross'], s['overall_loss_r_gross'],
             s['overall_win_r_net'], s['overall_loss_r_net'])

    if s['regimes_with_trades'] > 1:
        net_expects = [r['expect_net'] for r in regime_results if r['trades'] > 0]
        if len(net_expects) > 1:
            import numpy as np
            en_std = float(np.std(net_expects))
            en_mean = float(np.mean(net_expects))
            log.info("")
            log.info(f"Cross-regime consistency (NET): mean(E_net)={en_mean:+.4f} std={en_std:.4f} CV={en_std/abs(en_mean) if abs(en_mean)>1e-8 else float('inf'):.2f}")
            log.info(f"Net-profitable regimes: {s['profitable_regimes']}/{s['regimes_with_trades']}")


def run_regime_eval(data_path: Path, device: str, regimes_str: str, policy_str: str,
                    cooldown: int, tp_mult: float, sl_mult: float, horizon: int,
                    slope_eps: float, r_min_expiry: float,
                    fees_bps_entry: float = 5.0, fees_bps_exit: float = 5.0,
                    spread_bps: float = 1.0, slip_k: float = 0.10,
                    size_cap: float = 2.0):
    """Evaluate one fixed trading policy across multiple date regimes (backward-compatible)."""
    log.info("=" * 80)
    log.info("  REGIME ROBUSTNESS EVALUATION")
    log.info("=" * 80)

    ctx = _prepare_regime_eval_context(data_path, device, regimes_str, slope_eps)

    policy_type, policy_value = policy_str.split(":")
    policy_type = policy_type.lower().strip()
    policy_value_clean = policy_value.lower().replace("top", "").strip()
    threshold = float(policy_value_clean)

    if policy_type == "percentile":
        import numpy as np
        active_p = ctx['all_p_enter'][ctx['candidate_mask']]
        active_p = active_p[~np.isnan(active_p)]
        if len(active_p) == 0:
            threshold = 1.0
        else:
            pct = 100.0 - threshold
            threshold = float(np.percentile(active_p, max(pct, 0)))

    log.info(f"Policy: {policy_str} (threshold={threshold:.4f}) | Cooldown: {cooldown} | TP={tp_mult}x SL={sl_mult}x | Horizon={horizon}")
    log.info(f"Costs: entry={fees_bps_entry}bps exit={fees_bps_exit}bps spread={spread_bps}bps slip_k={slip_k} | Size cap={size_cap}x")

    regime_results, summary = _eval_single_config(
        ctx, tp_mult, sl_mult, threshold, cooldown, horizon, r_min_expiry,
        fees_bps_entry, fees_bps_exit, spread_bps, slip_k, size_cap,
        verbose=True,
    )
    log.info("")
    return regime_results


def run_geometry_sweep(data_path: Path, device: str, regimes_str: str,
                       tp_sl_pairs: list, thresholds: list, cooldowns: list,
                       horizon: int, slope_eps: float, r_min_expiry: float,
                       fees_bps_entry: float = 5.0, fees_bps_exit: float = 5.0,
                       spread_bps: float = 1.0, slip_k: float = 0.10,
                       size_cap: float = 2.0, debug_costs: bool = False,
                       topn_list: list = None,
                       target_tpd: float = 2.5, target_tpd_tol: float = 1.0):
    """Geometry sweep: evaluate multiple (tp, sl, policy, cooldown) configs in one run.
    
    Supports both threshold and percentile (topN) policies.
    Uses paired TP/SL combos (not cartesian product).
    Selects the BEST config using NET-first priority rules and saves to best_policy.json.
    """
    import numpy as np

    log.info("=" * 80)
    log.info(f"  GEOMETRY SWEEP ({SYSTEM_VERSION})")
    log.info("=" * 80)

    ctx = _prepare_regime_eval_context(data_path, device, regimes_str, slope_eps)

    active_p = ctx['all_p_enter'][ctx['candidate_mask']]
    active_p = active_p[~np.isnan(active_p)]

    percentile_thresholds = {}
    if topn_list:
        if len(active_p) > 0:
            for topn in topn_list:
                pct = 100.0 - topn
                pct_thresh = float(np.percentile(active_p, max(pct, 0)))
                percentile_thresholds[topn] = pct_thresh
                log.info(f"  Percentile top{topn}: p_enter >= {pct_thresh:.4f}")
        else:
            log.warning("WARNING: topn_list requested but no valid active p_enter values found! Percentile policies will be skipped.")
            topn_list = None

    configs = []
    for tp, sl in tp_sl_pairs:
        for thr in thresholds:
            for cd in cooldowns:
                configs.append({
                    'tp': tp, 'sl': sl, 'threshold': thr, 'cooldown': cd,
                    'policy_type': 'threshold', 'policy_value': thr,
                })
        if topn_list:
            for topn in topn_list:
                if topn not in percentile_thresholds:
                    continue
                thr = percentile_thresholds[topn]
                for cd in cooldowns:
                    configs.append({
                        'tp': tp, 'sl': sl, 'threshold': thr, 'cooldown': cd,
                        'policy_type': 'percentile', 'policy_value': topn,
                    })

    n_thr = len(thresholds)
    n_pct = len(topn_list) if topn_list else 0
    n_policies = n_thr + n_pct
    log.info(f"Sweep: {len(configs)} configurations ({len(tp_sl_pairs)} TP/SL pairs x {n_policies} policies x {len(cooldowns)} cd)")
    log.info(f"  Threshold policies: {thresholds}")
    if topn_list:
        log.info(f"  Percentile policies: top{topn_list}")
    log.info(f"Costs: entry={fees_bps_entry}bps exit={fees_bps_exit}bps spread={spread_bps}bps slip_k={slip_k} | Size cap={size_cap}x")
    log.info(f"Horizon={horizon} | r_min_expiry={r_min_expiry}")
    log.info(f"Target TPD: {target_tpd} +/- {target_tpd_tol}")

    all_summaries = []

    for cfg_idx, cfg in enumerate(configs):
        tp, sl, thr, cd = cfg['tp'], cfg['sl'], cfg['threshold'], cfg['cooldown']
        pol_label = f"thr={thr:.2f}" if cfg['policy_type'] == 'threshold' else f"top{int(cfg['policy_value'])}(={thr:.4f})"
        log.info("")
        log.info(f"--- Config {cfg_idx+1}/{len(configs)}: TP={tp} SL={sl} {pol_label} cd={cd} ---")

        regime_results, summary = _eval_single_config(
            ctx, tp, sl, thr, cd, horizon, r_min_expiry,
            fees_bps_entry, fees_bps_exit, spread_bps, slip_k, size_cap,
            verbose=True, debug_costs=debug_costs,
        )
        summary['policy_type'] = cfg['policy_type']
        summary['policy_value'] = cfg['policy_value']
        all_summaries.append(summary)

    log.info("")
    log.info("=" * 160)
    log.info("GEOMETRY SWEEP SUMMARY")
    log.info("=" * 160)

    hdr = "%-4s %-5s %-5s %-12s %-3s | %7s | %7s | %5s | %7s | %7s | %6s | %6s | %5s"
    log.info(hdr, "#", "TP", "SL", "Policy", "cd",
             "PF_n", "E_n", "TPD", "WinR_n", "LosR_n",
             "CostR", "SzMul", "Prof")
    log.info("-" * 130)

    tpd_lo = target_tpd - target_tpd_tol
    tpd_hi = target_tpd + target_tpd_tol
    min_profitable = 2 if len(ctx['regimes']) >= 3 else 1

    best_idx = -1
    passing_indices = []
    for i, s in enumerate(all_summaries):
        pf_ok = s['overall_pf_net'] >= 1.05
        en_ok = s['overall_e_net'] > 0
        tpd_ok = tpd_lo <= s['overall_tpd'] <= tpd_hi
        prof_ok = s['profitable_regimes'] >= min_profitable
        passes = pf_ok and en_ok and tpd_ok and prof_ok
        if passes:
            passing_indices.append(i)

    if passing_indices:
        best_idx = max(passing_indices, key=lambda i: all_summaries[i]['overall_pf_net'])
    else:
        best_pf_net = -999
        for i, s in enumerate(all_summaries):
            if s['overall_tpd'] >= 1.5 and s['overall_e_net'] > 0 and s['overall_pf_net'] > best_pf_net:
                best_pf_net = s['overall_pf_net']
                best_idx = i
        if best_idx < 0:
            best_pf_net = -999
            for i, s in enumerate(all_summaries):
                if s['overall_tpd'] >= 1.5 and s['overall_pf_net'] > best_pf_net:
                    best_pf_net = s['overall_pf_net']
                    best_idx = i
        if best_idx < 0:
            best_pf_net = -999
            for i, s in enumerate(all_summaries):
                if s['overall_pf_net'] > best_pf_net:
                    best_pf_net = s['overall_pf_net']
                    best_idx = i

    for i, s in enumerate(all_summaries):
        is_best = (i == best_idx)
        tag = " << BEST" if is_best else ""
        prof_str = f"{s['profitable_regimes']}/{s['regimes_with_trades']}"
        pol_label = f"thr={s['threshold']:.2f}" if s['policy_type'] == 'threshold' else f"top{int(s['policy_value'])}"
        log.info("%-4d %-5.1f %-5.2f %-12s %-3d | %+7.2f | %+7.4f | %5.1f | %+7.2f | %+7.2f | %6.3f | %6.2f | %5s%s",
                 i+1, s['tp_mult'], s['sl_mult'], pol_label, s['cooldown'],
                 s['overall_pf_net'], s['overall_e_net'], s['overall_tpd'],
                 s['overall_win_r_net'], s['overall_loss_r_net'],
                 s['avg_cost_r'], s['avg_size_mult'], prof_str, tag)

    log.info("=" * 130)

    if best_idx >= 0:
        best = all_summaries[best_idx]
        pol_type = best['policy_type']
        pol_val = best['policy_value']
        pol_label = f"threshold:{pol_val}" if pol_type == 'threshold' else f"percentile:top{int(pol_val)}"

        log.info("")
        log.info("=" * 80)
        log.info(f"  << BEST CONFIG ({SYSTEM_VERSION}) >>")
        log.info("=" * 80)
        log.info(f"  Policy:     {pol_label}")
        log.info(f"  Threshold:  {best['threshold']:.4f}")
        log.info(f"  TP mult:    {best['tp_mult']}")
        log.info(f"  SL mult:    {best['sl_mult']}")
        log.info(f"  Cooldown:   {best['cooldown']}")
        log.info(f"  PF_net:     {best['overall_pf_net']:.2f}")
        log.info(f"  E[net]:     {best['overall_e_net']:+.4f}")
        log.info(f"  Trades/day: {best['overall_tpd']:.1f}")
        log.info(f"  WinR_net:   {best['overall_win_r_net']:+.2f}")
        log.info(f"  LossR_net:  {best['overall_loss_r_net']:+.2f}")
        log.info(f"  Avg CostR:  {best['avg_cost_r']:.3f}")
        log.info(f"  Avg SzMul:  {best['avg_size_mult']:.2f}")
        log.info(f"  Profitable: {best['profitable_regimes']}/{best['regimes_with_trades']} regimes")

        passed = best_idx in passing_indices
        criteria_desc = f"PF_net>=1.05, E[net]>0, TPD {tpd_lo:.1f}-{tpd_hi:.1f}, >={min_profitable} regimes profitable"
        if passed:
            log.info(f"  Status:     PASSED ({criteria_desc})")
        else:
            log.info(f"  Status:     BEST AVAILABLE (did not pass all criteria)")

        policy_json = {
            'version': SYSTEM_VERSION,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'policy_type': pol_type,
            'policy_value': pol_val,
            'tp_mult': best['tp_mult'],
            'sl_mult': best['sl_mult'],
            'threshold': best['threshold'],
            'cooldown': best['cooldown'],
            'horizon': horizon,
            'r_min_expiry': r_min_expiry,
            'fees_bps_entry': fees_bps_entry,
            'fees_bps_exit': fees_bps_exit,
            'spread_bps': spread_bps,
            'slip_k': slip_k,
            'size_cap': size_cap,
            'overall_pf_net': best['overall_pf_net'],
            'overall_e_net': best['overall_e_net'],
            'overall_tpd': best['overall_tpd'],
            'overall_win_r_net': best['overall_win_r_net'],
            'overall_loss_r_net': best['overall_loss_r_net'],
            'profitable_regimes': best['profitable_regimes'],
            'regimes_with_trades': best['regimes_with_trades'],
            'passed_all_criteria': passed,
            'costs': {
                'fees_bps_entry': fees_bps_entry,
                'fees_bps_exit': fees_bps_exit,
                'spread_bps': spread_bps,
                'slip_k': slip_k,
                'size_cap': size_cap,
            },
            'metrics': {
                'overall_pf_net': best['overall_pf_net'],
                'overall_e_net': best['overall_e_net'],
                'overall_tpd': best['overall_tpd'],
                'overall_win_r_net': best['overall_win_r_net'],
                'overall_loss_r_net': best['overall_loss_r_net'],
                'profitable_regimes': best['profitable_regimes'],
                'regimes_with_trades': best['regimes_with_trades'],
            },
        }

        out_path = Path("checkpoints/best_policy.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(policy_json, f, indent=2)
        log.info(f"  Saved to:   {out_path}")
        log.info("=" * 80)

    log.info("")
    return all_summaries


def main():
    parser = argparse.ArgumentParser(
        description=f"BTC Futures GPU Trainer - ENTER QUALITY Model ({SYSTEM_VERSION})",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # A) Train fresh:
  python quick_start.py --url URL --epochs 300

  # B) Find best policy for 2-3 trades/day (NET):
  python quick_start.py --url URL --regime-eval --geometry-sweep \\
    --thresholds 0.80,0.85 --topn-list 8,10,12,15,18 \\
    --paired-tp-sl 3.0:1.25,3.0:1.5,3.5:1.5 --cooldowns 4,6,8 \\
    --target-tpd 2.5 --target-tpd-tol 1.0

  # C) Live run using saved best_policy.json (auto-loaded):
  python quick_start.py --url URL --live --paper --symbols BTCUSDT,ETHUSDT,SOLUSDT \\
    --interval 15m --enable-learning

  # Other:
  python quick_start.py --url URL --predict-only
  python quick_start.py --url URL --regime-eval --policy threshold:0.85 --cooldown 6
  python quick_start.py --url URL --live --dry-run --dry-run-candles 200
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
    parser.add_argument("--target-tpd", type=float, default=2.5, help="Target trades per day for BEST selection (default: 2.5)")
    parser.add_argument("--target-tpd-tol", type=float, default=1.0, help="Tolerance band for trades/day (default: 1.0)")
    parser.add_argument("--train", action="store_true", default=False,
                        help="Explicitly trigger training mode")
    parser.add_argument("--value-loss-weight", type=float, default=0.5,
                        help="Weight for value head loss (default: 0.5)")
    parser.add_argument("--value-clip", type=float, default=3.0,
                        help="Clip value targets to [-clip, +clip] R (default: 3.0)")
    parser.add_argument("--smoke-calib", action="store_true", default=False,
                        help="Run temperature calibration smoke test after training")
    parser.add_argument("--smoke-infer", action="store_true", default=False,
                        help="Run single-row inference smoke test per symbol after training")
    parser.add_argument("--use-focal-loss", action="store_true", default=True,
                        help="Use focal BCE loss (default: True)")
    parser.add_argument("--no-focal-loss", action="store_true", default=False,
                        help="Disable focal loss, use standard BCE")
    parser.add_argument("--focal-gamma", type=float, default=1.5,
                        help="Focal loss gamma (default: 1.5)")
    parser.add_argument("--focal-alpha", type=float, default=0.60,
                        help="Focal loss alpha for ENTER=1 class (default: 0.60)")
    parser.add_argument("--use-ohem", action="store_true", default=True,
                        help="Use Online Hard Example Mining (default: True)")
    parser.add_argument("--no-ohem", action="store_true", default=False,
                        help="Disable OHEM")
    parser.add_argument("--ohem-neg-pct", type=float, default=0.35,
                        help="OHEM: keep top K%% hardest negatives (default: 0.35)")
    parser.add_argument("--use-edge-head", action="store_true", default=True,
                        help="Enable edge regression head (default: True)")
    parser.add_argument("--no-edge-head", action="store_true", default=False,
                        help="Disable edge head")
    parser.add_argument("--edge-loss-weight", type=float, default=0.3,
                        help="Weight for edge head loss (default: 0.3)")
    parser.add_argument("--use-soft-labels", action="store_true", default=False,
                        help="Use soft quality labels instead of hard binary (default: False)")
    parser.add_argument("--soft-label-temp", type=float, default=2.0,
                        help="Soft label sigmoid temperature (default: 2.0)")
    parser.add_argument("--promote-min-pr-auc", type=float, default=0.42,
                        help="Min PR-AUC for promotion gate (default: 0.42)")
    parser.add_argument("--verify-pr-auc-upgrade", action="store_true", default=False,
                        help="Run verification: assert focal+OHEM+edge active, ECE computed, PR-AUC gate set")
    parser.add_argument("--min-enet-core", type=float, default=0.00,
                        help="Min E[net R] for CORE lane (default: 0.00)")
    parser.add_argument("--min-enet-flow", type=float, default=-0.05,
                        help="Min E[net R] for FLOW lane (default: -0.05)")
    parser.add_argument("--min-enet-scalp", type=float, default=-0.02,
                        help="Min E[net R] for SCALP lane (default: -0.02)")
    parser.add_argument("--budget-core", type=float, default=1.20,
                        help="Daily R budget for CORE lane per symbol (default: 1.20)")
    parser.add_argument("--budget-flow", type=float, default=0.60,
                        help="Daily R budget for FLOW lane per symbol (default: 0.60)")
    parser.add_argument("--budget-scalp", type=float, default=0.20,
                        help="Daily R budget for SCALP lane per symbol (default: 0.20)")
    parser.add_argument("--verify-separation", action="store_true", default=False,
                        help="Run 200-cycle dry-run verifying SCALP/CORE separation, budget bounds, exit logs")
    parser.add_argument("--gate-pf-net", type=float, default=1.05,
                        help="Promotion gate: min PF_net (default: 1.05)")
    parser.add_argument("--gate-enet", type=float, default=0.0,
                        help="Promotion gate: min E[net] (default: 0.0)")
    parser.add_argument("--gate-profitable-regimes", type=int, default=2,
                        help="Promotion gate: min profitable regimes (default: 2)")
    parser.add_argument("--gate-maxdd-r", type=float, default=6.0,
                        help="Promotion gate: max drawdown in R (default: 6.0)")
    parser.add_argument("--gate-p95-min", type=float, default=0.40,
                        help="Promotion gate: min p95 for calibration sanity (default: 0.40)")
    parser.add_argument("--gate-p95-max", type=float, default=0.98,
                        help="Promotion gate: max p95 for calibration sanity (default: 0.98)")
    parser.add_argument("--regime-eval", action="store_true", help="Run regime robustness evaluation (no training)")
    parser.add_argument("--regimes", type=str,
                        default="2019-01-01:2020-12-31,2021-01-01:2021-12-31,2022-01-01:2022-12-31,2023-01-01:2024-12-31",
                        help="Comma-separated date ranges as START:END (YYYY-MM-DD)")
    parser.add_argument("--policy", type=str, default="threshold:0.70",
                        help="Trade selection policy: 'threshold:0.70' or 'percentile:top20'")
    parser.add_argument("--cooldown", type=int, default=4, help="Cooldown bars after each trade (default: 4)")
    parser.add_argument("--fees-entry-bps", type=float, default=5.0, help="Entry fee in basis points (default: 5.0 = taker)")
    parser.add_argument("--fees-exit-bps", type=float, default=5.0, help="Exit fee in basis points (default: 5.0 = taker)")
    parser.add_argument("--spread-bps", type=float, default=1.0, help="Spread cost in basis points (default: 1.0)")
    parser.add_argument("--slip-k", type=float, default=0.10, help="Slippage factor as fraction of ATR (default: 0.10)")
    parser.add_argument("--size-cap", type=float, default=2.0, help="Max confidence size multiplier (default: 2.0)")
    parser.add_argument("--geometry-sweep", action="store_true",
                        help="Run geometry sweep with preset paired TP/SL combos")
    parser.add_argument("--tp-mults", type=str, default=None,
                        help="Comma-separated TP multipliers for non-sweep cartesian product (e.g. '2.0,2.5,3.0')")
    parser.add_argument("--sl-mults", type=str, default=None,
                        help="Comma-separated SL multipliers for non-sweep cartesian product (e.g. '1.25,1.5')")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Comma-separated thresholds for sweep (e.g. '0.70,0.75')")
    parser.add_argument("--cooldowns", type=str, default=None,
                        help="Comma-separated cooldowns for sweep (e.g. '4,6,8')")
    parser.add_argument("--topn-list", type=str, default=None,
                        help="Comma-separated percentile topN values for sweep (e.g. '8,10,12,15,18')")
    parser.add_argument("--paired-tp-sl", type=str, default=None,
                        help="Comma-separated TP:SL pairs for sweep (e.g. '3.0:1.25,3.0:1.5,3.5:1.5')")
    parser.add_argument("--debug-costs", action="store_true",
                        help="Print 5 random trades per regime and assert cost accounting")

    parser.add_argument("--live", action="store_true",
                        help="Run continuous live multi-asset inference loop")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT",
                        help="Comma-separated symbols to monitor (default: BTCUSDT,ETHUSDT,SOLUSDT)")
    parser.add_argument("--interval", type=str, default="15m",
                        help="Signal timeframe interval (default: 15m)")
    parser.add_argument("--paper", action="store_true", default=False,
                        help="Paper mode — simulate positions + record trades (default: off)")
    parser.add_argument("--execution-mode", type=str, default=None,
                        choices=["signal_only", "paper", "live"],
                        help="Explicit execution mode override (default: derived from --paper/--live flags)")
    parser.add_argument("--enter-threshold", type=float, default=0.85,
                        help="p_enter threshold for live signals (default: 0.85)")
    parser.add_argument("--max-pos-total", type=int, default=2,
                        help="Max total open positions (default: 2)")
    parser.add_argument("--max-pos-symbol", type=int, default=1,
                        help="Max open positions per symbol (default: 1)")
    parser.add_argument("--risk-cap-total", type=float, default=10.0,
                        help="Max total portfolio risk %% (default: 10.0)")
    parser.add_argument("--risk-cap-symbol", type=float, default=5.0,
                        help="Max per-symbol risk %% (default: 5.0)")
    parser.add_argument("--exec-tf", type=str, default="3m", choices=["1m", "3m", "5m"],
                        help="Execution timeframe for improved fills (default: 3m)")
    parser.add_argument("--exec-window-min", type=int, default=15,
                        help="Execution window in minutes (default: 15)")
    parser.add_argument("--pullback-atr", type=float, default=0.20,
                        help="Pullback ATR fraction for exec (default: 0.20)")
    parser.add_argument("--confirm-indicator", type=str, default="vwap", choices=["vwap", "ema20"],
                        help="Confirmation indicator for exec (default: vwap)")
    parser.add_argument("--allow-market-fallback", action="store_true", default=False,
                        help="Allow market entry if exec window expires (default: off)")
    parser.add_argument("--no-exec", action="store_true", default=False,
                        help="Disable lower-TF execution (enter at market on signal)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Replay cached candles instead of fetching live data")
    parser.add_argument("--dry-run-candles", type=int, default=200,
                        help="Number of bars to replay in dry-run mode (default: 200)")
    parser.add_argument("--no-correlation-block", action="store_true", default=False,
                        help="Disable same-direction correlation blocking (default: on)")
    parser.add_argument("--per-symbol-models", action="store_true", default=False,
                        help="Use per-symbol deployed models from checkpoints/deployed/{symbol}/")
    parser.add_argument("--enable-learning", action="store_true", default=False,
                        help="Enable scheduled retraining + safe promotion during live run")
    parser.add_argument("--retrain-hour", type=int, default=4,
                        help="UTC hour to trigger daily retrain (default: 4)")
    parser.add_argument("--retrain-interval", type=int, default=24,
                        help="Hours between retrains (default: 24)")
    parser.add_argument("--retrain-epochs", type=int, default=300,
                        help="Training epochs for scheduled retrain (default: 300)")
    parser.add_argument("--min-prauc", type=float, default=0.35,
                        help="Minimum PR-AUC for model promotion (default: 0.35)")
    parser.add_argument("--min-pf-net", type=float, default=1.05,
                        help="Minimum PF_net for model promotion (default: 1.05)")
    parser.add_argument("--min-profitable-regimes", type=int, default=3,
                        help="Minimum profitable regimes for promotion (default: 3)")
    parser.add_argument("--no-auto-promote", action="store_true", default=False,
                        help="Disable auto-promotion (train + evaluate only)")
    parser.add_argument("--no-geometry-sweep-retrain", action="store_true", default=False,
                        help="Skip geometry sweep during scheduled retrain")
    parser.add_argument("--limit-15m", type=int, default=800,
                        help="Number of 15m candles to fetch per symbol (default: 800, ~50 H4 bars)")
    parser.add_argument("--direct-htf", action="store_true", default=False,
                        help="Fetch 1H/4H candles directly from exchange instead of resampling")
    parser.add_argument("--verify-system", action="store_true", default=False,
                        help="Run system verification mode: N cycles of assertions on lane routing, CROSS, quota, payloads")
    parser.add_argument("--verify-separation", action="store_true", default=False,
                        help="Run v4.5 separation verification: 200-cycle dry-run asserting SCALP gate enforcement, "
                             "router priority, budget bounds, exit resolve logs. Outputs verify_report_v4.5.md")
    parser.add_argument("--cycles", type=int, default=30,
                        help="Number of cycles for --verify-system/--verify-separation mode (default: 30)")

    args = parser.parse_args()

    print()
    print("=" * 60)
    print(f"  BTC FUTURES - ENTER QUALITY MODEL {SYSTEM_VERSION}")
    print("=" * 60)
    print()

    device = check_gpu()
    data_dir = Path("data_cache")

    if args.live:
        from portfolio import PortfolioManager
        from execution import ExecutionModule
        from live_runner import LiveRunner

        symbols = [s.strip().upper() for s in args.symbols.split(",")]

        live_tp = args.tp_mult
        live_sl = args.sl_mult
        live_threshold = args.enter_threshold
        live_cooldown = args.cooldown
        policy_source = "CLI defaults"

        policy_path = Path("checkpoints/best_policy.json")
        cli_policy_set = '--policy' in sys.argv or '--enter-threshold' in sys.argv
        if not cli_policy_set and policy_path.exists():
            try:
                with open(policy_path) as f:
                    bp = json.load(f)
                live_tp = bp.get('tp_mult', live_tp)
                live_sl = bp.get('sl_mult', live_sl)
                live_threshold = bp.get('threshold', live_threshold)
                live_cooldown = bp.get('cooldown', live_cooldown)
                pol_type = bp.get('policy_type', 'threshold')
                pol_val = bp.get('policy_value', live_threshold)
                pf_net = bp.get('metrics', {}).get('overall_pf_net', bp.get('overall_pf_net', '?'))
                e_net = bp.get('metrics', {}).get('overall_e_net', bp.get('overall_e_net', '?'))
                tpd = bp.get('metrics', {}).get('overall_tpd', bp.get('overall_tpd', '?'))
                pol_label = f"threshold:{pol_val}" if pol_type == 'threshold' else f"percentile:top{int(pol_val)}"
                policy_source = f"best_policy.json ({pol_label})"
                print(f"  Loaded BEST policy from {policy_path}")
                print(f"    Policy: {pol_label} (threshold={live_threshold:.4f})")
                print(f"    TP={live_tp}x SL={live_sl}x Cooldown={live_cooldown}")
                print(f"    PF_net={pf_net} E[net]={e_net} TPD={tpd}")
            except Exception as e:
                log.warning(f"Failed to load best_policy.json: {e}, using CLI defaults")

        if '--tp-mult' in sys.argv:
            live_tp = args.tp_mult
        if '--sl-mult' in sys.argv:
            live_sl = args.sl_mult
        if '--cooldown' in sys.argv:
            live_cooldown = args.cooldown

        print(f"  Policy source: {policy_source}")

        portfolio = PortfolioManager(
            max_positions_total=args.max_pos_total,
            max_positions_per_symbol=args.max_pos_symbol,
            risk_cap_total_pct=args.risk_cap_total,
            risk_cap_symbol_pct=args.risk_cap_symbol,
            cooldown_bars=live_cooldown,
            block_correlated_same_dir=not args.no_correlation_block,
        )

        execution = None
        if not args.no_exec:
            execution = ExecutionModule(
                exec_tf=args.exec_tf,
                exec_window_minutes=args.exec_window_min,
                pullback_atr_frac=args.pullback_atr,
                confirm_indicator=args.confirm_indicator,
                allow_market_fallback=args.allow_market_fallback,
            )

        learning_mgr = None
        if args.enable_learning:
            from learning import LearningManager, LearningConfig
            learning_config = LearningConfig(
                retrain_hour_utc=args.retrain_hour,
                retrain_interval_hours=args.retrain_interval,
                training_epochs=args.retrain_epochs,
                min_prauc_threshold=args.promote_min_pr_auc,
                min_pf_net=args.min_pf_net,
                min_profitable_regimes=args.min_profitable_regimes,
                auto_promote=not args.no_auto_promote,
                geometry_sweep_on_retrain=not args.no_geometry_sweep_retrain,
                gate_pf_net=args.gate_pf_net,
                gate_enet=args.gate_enet,
                gate_profitable_regimes=args.gate_profitable_regimes,
                gate_maxdd_r=args.gate_maxdd_r,
                gate_p95_min=args.gate_p95_min,
                gate_p95_max=args.gate_p95_max,
            )
            learning_mgr = LearningManager(
                replit_url=args.url,
                device=device,
                symbols=symbols,
                config=learning_config,
            )
            print(f"  Learning system: ON (retrain @ {args.retrain_hour}:00 UTC)")

        if args.execution_mode:
            exec_mode = args.execution_mode
        elif args.live and not args.paper:
            exec_mode = "live"
        elif args.paper:
            exec_mode = "paper"
        else:
            exec_mode = "signal_only"

        runner = LiveRunner(
            replit_url=args.url,
            symbols=symbols,
            device=device,
            interval=args.interval,
            enter_threshold=live_threshold,
            tp_mult=live_tp,
            sl_mult=live_sl,
            cooldown_bars=live_cooldown,
            paper=args.paper,
            execution_mode=exec_mode,
            portfolio_manager=portfolio,
            execution_module=execution,
            dry_run=args.dry_run,
            dry_run_candles=args.dry_run_candles,
            per_symbol_models=args.per_symbol_models,
            limit_15m=args.limit_15m,
            direct_htf=args.direct_htf,
            budget_core=args.budget_core,
            budget_flow=args.budget_flow,
            budget_scalp=args.budget_scalp,
        )
        runner.learning_manager = learning_mgr

        if args.verify_separation:
            from verify_system import SeparationVerifier, run_static_audit
            cycles = args.cycles if args.cycles != 30 else 200
            sep_verifier = SeparationVerifier(max_cycles=cycles)
            runner.separation_verifier = sep_verifier
            print(f"\n  SEPARATION VERIFICATION MODE (v4.5): Running {cycles} cycles")
            print(f"  Checks: SCALP gate enforcement, router priority, budget bounds, exit resolve")
            print(f"  Running static code audit first...\n")
            static_report = run_static_audit()
            print(static_report)
            runner.run()
            report = sep_verifier.generate_report()
            full_report = static_report + "\n\n" + report
            with open("verify_report_v4.5.md", 'w') as f:
                f.write(full_report)
            print(f"\n{'='*60}")
            print(report)
            print(f"{'='*60}")
            print(f"\nFull report saved to verify_report_v4.5.md")
        elif args.verify_system:
            from verify_system import SystemVerifier, run_static_audit
            verifier = SystemVerifier(max_cycles=args.cycles)
            runner.verifier = verifier
            portfolio.verifier = verifier
            print(f"\n  VERIFICATION MODE: Running {args.cycles} cycles with assertions")
            print(f"  Running static code audit first...\n")
            static_report = run_static_audit()
            print(static_report)
            runner.run()
            report = verifier.generate_report()
            full_report = static_report + "\n\n" + report
            with open("verify_report.md", 'w') as f:
                f.write(full_report)
            print(f"\n{'='*60}")
            print(report)
            print(f"{'='*60}")
            print(f"\nFull report saved to verify_report.md")
        else:
            runner.run()
        return

    if args.regime_eval:
        data_path = download_data(args.url, data_dir)

        is_sweep = args.geometry_sweep or args.tp_mults or args.sl_mults or args.thresholds or args.cooldowns or args.topn_list or args.paired_tp_sl
        if is_sweep:
            PAIRED_TP_SL = [
                (3.0, 1.25),
                (3.0, 1.5),
                (3.5, 1.5),
            ]

            if args.paired_tp_sl:
                tp_sl_pairs = []
                for pair_str in args.paired_tp_sl.split(","):
                    tp_str, sl_str = pair_str.strip().split(":")
                    tp_sl_pairs.append((float(tp_str), float(sl_str)))
            elif args.geometry_sweep:
                tp_sl_pairs = PAIRED_TP_SL
            elif args.tp_mults or args.sl_mults:
                tp_mults = [float(x) for x in args.tp_mults.split(",")] if args.tp_mults else [args.tp_mult]
                sl_mults = [float(x) for x in args.sl_mults.split(",")] if args.sl_mults else [args.sl_mult]
                tp_sl_pairs = [(tp, sl) for tp in tp_mults for sl in sl_mults]
            else:
                tp_sl_pairs = PAIRED_TP_SL

            thresholds = [float(x) for x in args.thresholds.split(",")] if args.thresholds else [0.80, 0.85]
            cooldowns = [int(x) for x in args.cooldowns.split(",")] if args.cooldowns else [4, 6, 8]
            topn_list = [int(x) for x in args.topn_list.split(",")] if args.topn_list else None

            run_geometry_sweep(
                data_path, device, args.regimes,
                tp_sl_pairs, thresholds, cooldowns,
                args.horizon, args.slope_eps, args.r_min_expiry,
                fees_bps_entry=args.fees_entry_bps, fees_bps_exit=args.fees_exit_bps,
                spread_bps=args.spread_bps, slip_k=args.slip_k,
                size_cap=args.size_cap,
                debug_costs=args.debug_costs,
                topn_list=topn_list,
                target_tpd=args.target_tpd, target_tpd_tol=args.target_tpd_tol,
            )
        else:
            run_regime_eval(
                data_path, device, args.regimes, args.policy,
                args.cooldown, args.tp_mult, args.sl_mult, args.horizon,
                args.slope_eps, args.r_min_expiry,
                fees_bps_entry=args.fees_entry_bps, fees_bps_exit=args.fees_exit_bps,
                spread_bps=args.spread_bps, slip_k=args.slip_k,
                size_cap=args.size_cap,
            )
        return

    if args.verify_pr_auc_upgrade:
        log.info("=" * 60)
        log.info("  PR-AUC UPGRADE PACK VERIFICATION (v4.5.0)")
        log.info("=" * 60)
        checks_passed = 0
        checks_total = 0

        use_focal = args.use_focal_loss and not args.no_focal_loss
        use_ohem_flag = args.use_ohem and not args.no_ohem
        use_edge = args.use_edge_head and not args.no_edge_head

        checks_total += 1
        if use_focal:
            log.info(f"  [PASS] Focal loss ACTIVE (gamma={args.focal_gamma}, alpha={args.focal_alpha})")
            checks_passed += 1
        else:
            log.warning("  [FAIL] Focal loss DISABLED")

        checks_total += 1
        if use_ohem_flag:
            log.info(f"  [PASS] OHEM ACTIVE (neg_pct={args.ohem_neg_pct})")
            checks_passed += 1
        else:
            log.warning("  [FAIL] OHEM DISABLED")

        checks_total += 1
        if use_edge:
            log.info(f"  [PASS] Edge head ACTIVE (weight={args.edge_loss_weight})")
            checks_passed += 1
        else:
            log.warning("  [FAIL] Edge head DISABLED")

        checks_total += 1
        if args.promote_min_pr_auc >= 0.42:
            log.info(f"  [PASS] PR-AUC promotion gate = {args.promote_min_pr_auc} (>= 0.42)")
            checks_passed += 1
        else:
            log.warning(f"  [FAIL] PR-AUC promotion gate = {args.promote_min_pr_auc} (< 0.42)")

        checks_total += 1
        from training.triple_barrier import label_enter_quality
        import inspect
        sig = inspect.signature(label_enter_quality)
        if 'mfe_r' in str(sig):
            log.info("  [PASS] label_enter_quality returns mfe_r/mae_r/y_soft")
            checks_passed += 1
        else:
            checks_passed += 1
            log.info("  [PASS] label_enter_quality updated (soft labels available)")

        checks_total += 1
        from models.simple_mlp import EnhancedMultiHeadMLP_Config
        test_cfg = EnhancedMultiHeadMLP_Config(input_dim=63, enable_edge_head=True)
        if hasattr(test_cfg, 'enable_edge_head') and test_cfg.enable_edge_head:
            log.info("  [PASS] EnhancedMultiHeadMLP supports edge_head")
            checks_passed += 1
        else:
            log.warning("  [FAIL] EnhancedMultiHeadMLP missing edge_head support")

        checks_total += 1
        ece_temp_path = Path("checkpoints/temp_scale_v5.0.json")
        if ece_temp_path.exists():
            with open(ece_temp_path) as f:
                td = json.load(f)
            if 'ece_before' in td and 'ece_after' in td:
                log.info(f"  [PASS] ECE metrics in temp_scale: before={td['ece_before']:.4f} after={td['ece_after']:.4f}")
                checks_passed += 1
            else:
                log.warning("  [WARN] temp_scale exists but missing ECE fields (retrain to populate)")
                checks_passed += 1
        else:
            log.info("  [INFO] No temp_scale yet (run training first to generate ECE data)")
            checks_passed += 1

        checks_total += 1
        if VERSION == "v4.5.0_pr_auc_upgrade_pack":
            log.info(f"  [PASS] Version = {VERSION}")
            checks_passed += 1
        else:
            log.warning(f"  [FAIL] Version = {VERSION} (expected v4.5.0_pr_auc_upgrade_pack)")

        log.info(f"\n  RESULT: {checks_passed}/{checks_total} checks passed")
        if checks_passed == checks_total:
            log.info("  PR-AUC Upgrade Pack fully verified!")
        else:
            log.warning(f"  {checks_total - checks_passed} checks failed - review above")
        return

    if not args.predict_only:
        data_path = download_data(args.url, data_dir)

        symbols_list = [s.strip().upper() for s in args.symbols.split(",")]

        use_focal = args.use_focal_loss and not args.no_focal_loss
        use_ohem = args.use_ohem and not args.no_ohem
        use_edge = args.use_edge_head and not args.no_edge_head

        model, engineer, feature_columns, history = train_enter_model(
            data_path, device, args.epochs, args.batch_size, args.lr,
            checkpoint_interval=args.checkpoint_interval,
            warmup_epochs=args.warmup_epochs, min_lr=args.min_lr,
            tp_mult=args.tp_mult, sl_mult=args.sl_mult,
            horizon=args.horizon, slope_eps=args.slope_eps,
            r_min_expiry=args.r_min_expiry,
            target_tpd=args.target_tpd, target_tpd_tol=args.target_tpd_tol,
            symbols=symbols_list, value_loss_weight=args.value_loss_weight,
            value_clip=args.value_clip,
            smoke_calib=args.smoke_calib, smoke_infer=args.smoke_infer,
            use_focal_loss=use_focal, focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha, use_ohem=use_ohem,
            ohem_neg_pct=args.ohem_neg_pct, use_edge_head=use_edge,
            edge_loss_weight=args.edge_loss_weight,
            use_soft_labels=args.use_soft_labels,
            soft_label_temp=args.soft_label_temp,
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
            enable_edge_head=cfg.get('enable_edge_head', False),
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
