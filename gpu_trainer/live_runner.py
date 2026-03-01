"""Multi-asset live inference loop (v4.5.2 — Triple-Lane Aggression Engine).

Monitors multiple symbols in parallel on 15m intervals, runs the ENTER QUALITY
model inference, applies HTF gates, computes HTF score, routes trades through
CORE / FLOW / SCALP lanes, manages per-symbol daily R budgets, and handles
SCALP time-stop exits.

v4.3.0 additions:
  - HTF score (0–3) replacing binary aligned gate
  - Triple-lane policy router (CORE / FLOW / SCALP)
  - Per-symbol daily R budget with lane-specific caps
  - SCALP lane with time-stop exit after 4 bars
  - Enhanced cycle log + trade record payloads

Usage:
    python quick_start.py --live --paper --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,AVAXUSDT,XRPUSDT,ADAUSDT
"""

import sys
import time
import json
import logging
import traceback
import requests
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("LiveRunner")


SYSTEM_VERSION = "v4.5.2_directional_sep_fix"

REQUIRED_CANDLES = 800
MAX_CACHE_BARS = 2000
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2.0

MIN_H1_BARS = 100
MIN_H4_BARS = 50

SCALP_HORIZON = 4
SCALP_TP_R = 1.20
SCALP_SL_R = 0.80
SCALP_SIZE_MULT = 0.25

DAILY_BUDGET_R_TOTAL = 2.0
LANE_BUDGET = {
    "CORE": 1.20,
    "FLOW": 0.60,
    "SCALP": 0.20,
}

FLOW_QUOTA_STEPS = {
    0: {"percentile": 95, "size_mult": 0.60},
    1: {"percentile": 93, "size_mult": 0.50},
    2: {"percentile": 91, "size_mult": 0.40},
    3: {"percentile": 89, "size_mult": 0.30},
}

COST_BPS = 8.0


def _retry_request(method: str, url: str, **kwargs) -> Optional[requests.Response]:
    timeout = kwargs.pop('timeout', 15)
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
            if resp.status_code < 500:
                return resp
        except (requests.ConnectionError, requests.Timeout) as e:
            log.warning(f"Request {method} {url} attempt {attempt+1}/{RETRY_ATTEMPTS} failed: {e}")
        if attempt < RETRY_ATTEMPTS - 1:
            time.sleep(RETRY_DELAY * (attempt + 1))
    log.error(f"All {RETRY_ATTEMPTS} attempts failed for {method} {url}")
    return None


def _get_exchange_time_offset() -> float:
    try:
        resp = requests.get("https://api.binance.com/api/v3/time", timeout=5)
        if resp.status_code == 200:
            server_ms = resp.json()["serverTime"]
            local_ms = time.time() * 1000
            offset_ms = server_ms - local_ms
            if abs(offset_ms) > 1000:
                log.warning(f"Clock offset vs Binance: {offset_ms:.0f}ms")
            return offset_ms / 1000.0
        return 0.0
    except Exception:
        return 0.0


def _load_model(device: str, symbol: Optional[str] = None):
    """Load the trained model, scaler, feature columns, and temperature.

    Supports both legacy EnhancedMultiHeadMLP and V5Forecaster models.
    If symbol is provided, first checks checkpoints/deployed/{symbol}/ for a
    per-symbol model. Falls back to the global checkpoints/ directory.

    Returns: (model, engineer, feature_columns, temperature, symbol_map)
    """
    import torch
    from data.pipeline import FeatureEngineer

    search_dirs = []
    if symbol:
        search_dirs.append(Path(f"checkpoints/deployed/{symbol}"))
    search_dirs.append(Path("checkpoints"))

    checkpoint_names = [
        "best_enter_prauc.pt",
        "best_v5_expectancy.pt",
        "best_enter_loss.pt",
        "best_v5_loss.pt",
    ]

    checkpoint_path = None
    for d in search_dirs:
        for name in checkpoint_names:
            p = d / name
            if p.exists():
                checkpoint_path = p
                break
        if checkpoint_path:
            break

    if not checkpoint_path:
        log.error(f"No trained model found{' for '+symbol if symbol else ''}! Run training first.")
        sys.exit(1)

    log.info(f"Loading model from {checkpoint_path}{' ('+symbol+')' if symbol else ''}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    saved_version = checkpoint.get('feature_version', 'unknown')
    from quick_start import FEATURE_VERSION
    ACCEPTED_VERSIONS = {FEATURE_VERSION, "v5.0.1_forecaster"}
    if saved_version not in ACCEPTED_VERSIONS:
        log.error(f"Feature version mismatch! Model: '{saved_version}', accepted: {ACCEPTED_VERSIONS}")
        sys.exit(1)

    feature_columns = checkpoint.get('feature_columns', [])
    if not feature_columns:
        log.error("No feature_columns in checkpoint — retrain.")
        sys.exit(1)

    cfg = checkpoint.get('model_config', {})
    model_type = checkpoint.get('model_type', 'legacy')

    if model_type == 'v5_forecaster':
        from models.v5_forecaster import V5Forecaster, V5ForecasterConfig
        v5_config = V5ForecasterConfig(
            input_dim=cfg.get('input_dim', 85),
            hidden_dims=cfg.get('hidden_dims', [512, 256, 128, 64]),
            dropout=cfg.get('dropout', 0.3),
            use_layer_norm=True,
            use_residual=True,
            n_barrier_presets=cfg.get('n_barrier_presets', 0),
            enable_regime_head=cfg.get('enable_regime_head', False),
            n_symbols=cfg.get('n_symbols', 1),
            symbol_embed_dim=cfg.get('symbol_embed_dim', 8),
        )
        model = V5Forecaster(v5_config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model._is_v5 = True
        log.info(f"Loaded V5Forecaster ({model.parameters_count():,} params)")
    else:
        from models.simple_mlp import EnhancedMultiHeadMLP, EnhancedMultiHeadMLP_Config
        n_symbols = cfg.get('n_symbols', 1)
        symbol_embed_dim = cfg.get('symbol_embed_dim', 0)
        enable_value_head = cfg.get('enable_value_head', False)
        enable_edge_head = cfg.get('enable_edge_head', False)
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
            enable_value_head=enable_value_head,
            enable_edge_head=enable_edge_head,
            n_symbols=n_symbols,
            symbol_embed_dim=symbol_embed_dim,
        )
        model = EnhancedMultiHeadMLP(mlp_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model._is_v5 = False
        log.info(f"Loaded EnhancedMultiHeadMLP")

    model.to(device)
    model.eval()

    engineer = FeatureEngineer()
    scaler_loaded = False

    if model_type == 'v5_forecaster' and 'scaler_center' in checkpoint and 'scaler_scale' in checkpoint:
        from sklearn.preprocessing import RobustScaler
        v5_scaler = RobustScaler()
        v5_scaler.center_ = np.array(checkpoint['scaler_center'])
        v5_scaler.scale_ = np.array(checkpoint['scaler_scale'])
        engineer._v5_global_scaler = v5_scaler
        scaler_loaded = True
        log.info("Scaler loaded from checkpoint (embedded)")

    if not scaler_loaded:
        scaler_path = None
        for d in search_dirs:
            for sname in ["per_symbol_scalers.joblib", "scaler.joblib"]:
                sp = d / sname
                if sp.exists():
                    scaler_path = sp
                    break
            if scaler_path:
                break
        if scaler_path:
            engineer.load_scalers(str(scaler_path))
            log.info(f"Scaler loaded from {scaler_path}")
        else:
            log.warning("No saved scaler — prediction quality may be reduced")

    temperature = 1.0
    for d in search_dirs:
        tp = d / "temp_scale_v5.0.json"
        if tp.exists():
            try:
                with open(tp) as f:
                    temp_data = json.load(f)
                temperature = float(temp_data.get("temperature", 1.0))
                log.info(f"[INFER] using_temperature={temperature:.4f} (from {tp})")
            except Exception as e:
                log.warning(f"Failed to load temperature: {e}")
            break

    symbol_map = checkpoint.get('symbol_map', None)

    n_symbols = cfg.get('n_symbols', 1)
    log.info(f"Model loaded: {len(feature_columns)} features, version {saved_version}, "
             f"model_type={model_type}, n_symbols={n_symbols}, temperature={temperature:.4f}")
    return model, engineer, feature_columns, temperature, symbol_map


def _fetch_candles_for_symbol(fetcher, symbol: str, timeframe: str = "15m",
                               limit: int = REQUIRED_CANDLES) -> Optional[pd.DataFrame]:
    """Fetch recent candles for a symbol via the BinanceDataFetcher."""
    try:
        log.info(f"Fetching {symbol} {timeframe} (limit={limit}...)")
        raw = fetcher.fetch_klines_sync(symbol, timeframe, limit=limit)
        if not raw or len(raw) < 100:
            log.warning(f"Insufficient candles for {symbol}: got {len(raw) if raw else 0}")
            return None

        df = pd.DataFrame(raw)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype(int)

        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    except Exception as e:
        log.error(f"Failed to fetch candles for {symbol}: {e}")
        return None


def _check_htf_warmup(df: pd.DataFrame, symbol: str) -> Optional[str]:
    """Check if there are enough candles to form valid HTF bars.

    Returns None if OK, or a WARMUP reason string if insufficient bars.
    Logs once per cycle when in warmup state.
    """
    if 'timestamp' not in df.columns:
        return f"{symbol} WARMUP: no timestamp column"

    ts = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

    ohlcv = pd.DataFrame({
        'open': df['open'].values,
        'high': df['high'].values,
        'low': df['low'].values,
        'close': df['close'].values,
        'volume': df['volume'].values,
    }, index=ts)

    h1_bars = ohlcv.resample('1h', label='left', closed='left').agg({
        'open': 'first'
    }).dropna()
    h4_bars = ohlcv.resample('4h', label='left', closed='left').agg({
        'open': 'first'
    }).dropna()

    n_h1 = len(h1_bars)
    n_h4 = len(h4_bars)

    if n_h1 < MIN_H1_BARS or n_h4 < MIN_H4_BARS:
        msg = (f"{symbol} WARMUP: h1_bars={n_h1} h4_bars={n_h4} "
               f"(min {MIN_H1_BARS}/{MIN_H4_BARS}) -> skip gates/trading")
        return msg

    return None


def _fetch_htf_candles_direct(fetcher, symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
    """Fetch 1H and 4H candles directly from the exchange instead of resampling.

    Returns dict with '1h' and '4h' DataFrames, or None on failure.
    """
    result = {}
    for tf, limit in [("1h", 300), ("4h", 200)]:
        try:
            log.info(f"Fetching {symbol} {tf} (limit={limit}, direct HTF...)")
            raw = fetcher.fetch_klines_sync(symbol, tf, limit=limit)
            if not raw or len(raw) < 10:
                log.warning(f"Insufficient direct {tf} candles for {symbol}: got {len(raw) if raw else 0}")
                return None
            df = pd.DataFrame(raw)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            if 'timestamp' in df.columns:
                df['timestamp'] = df['timestamp'].astype(int)
            df = df.sort_values('timestamp').reset_index(drop=True)
            result[tf] = df
            log.info(f"  {symbol} {tf}: got {len(df)} bars")
        except Exception as e:
            log.error(f"Failed to fetch direct {tf} candles for {symbol}: {e}")
            return None
    return result


def _compute_features_for_symbol(df: pd.DataFrame, engineer, feature_columns: list,
                                  symbol: str) -> Optional[np.ndarray]:
    """Compute features for the latest bar of a symbol's candle data.

    Uses the same pipeline as make_enter_prediction: FeatureEngineer.compute_all_features,
    then funding + OI features, then scale + clip.
    """
    from quick_start import (
        FUNDING_FEATURE_COUNT, OI_FEATURE_COUNT,
        FUNDING_FEATURE_NAMES, OI_FEATURE_NAMES,
    )

    try:
        feat_engineer_local = engineer.__class__()
        features_df = feat_engineer_local.compute_all_features(df)
        features_df = features_df.fillna(0)

        n = len(df)
        funding_features = pd.DataFrame(
            np.zeros((n, FUNDING_FEATURE_COUNT)),
            columns=FUNDING_FEATURE_NAMES,
            index=df.index,
        )
        features_df = pd.concat([features_df, funding_features], axis=1)

        oi_features = pd.DataFrame(
            np.zeros((n, OI_FEATURE_COUNT)),
            columns=OI_FEATURE_NAMES,
            index=df.index,
        )
        features_df = pd.concat([features_df, oi_features], axis=1)
        features_df = features_df.fillna(0)

        features_df = features_df.reindex(columns=feature_columns, fill_value=0)

        last_features = features_df.iloc[-1:].copy()
        if hasattr(engineer, '_v5_global_scaler'):
            raw = last_features.values.astype(np.float32)
            last_scaled = engineer._v5_global_scaler.transform(raw).astype(np.float32)
            last_scaled = np.clip(last_scaled, -5.0, 5.0)
        else:
            last_scaled = engineer.transform_and_clip(
                pd.DataFrame(last_features.values, columns=feature_columns),
                clip_range=5.0
            ).values.astype(np.float32)
        last_scaled = np.where(np.isinf(last_scaled), 0, last_scaled)
        last_scaled = np.where(np.isnan(last_scaled), 0, last_scaled)

        return last_scaled, features_df

    except Exception as e:
        log.error(f"Feature computation failed for {symbol}: {e}")
        traceback.print_exc()
        return None, None


def _run_inference(model, scaled_features: np.ndarray, device: str,
                   temperature: float = 1.0, symbol_id: Optional[int] = None) -> dict:
    """Run single-row model inference with temperature calibration.
    
    Supports both legacy EnhancedMultiHeadMLP and V5Forecaster models.
    
    Returns dict with:
      p_enter: calibrated probability of entering a trade
      e_net_pred: predicted E[net R] (ret_mu for V5, value_logits for legacy)
      enter_logit: raw logit before calibration
      v5_action_probs: [hold, long, short] probabilities (V5 only)
      v5_ret_mu: predicted return in R-units (V5 only)
      v5_mfe: predicted max favorable excursion (V5 only)
      v5_mae: predicted max adverse excursion (V5 only)
    """
    import torch
    is_v5 = getattr(model, '_is_v5', False)

    with torch.no_grad():
        x = torch.FloatTensor(scaled_features).to(device)

        if is_v5:
            sym_ids = None
            if symbol_id is not None and model.symbol_embedding is not None:
                sym_ids = torch.tensor([symbol_id], dtype=torch.long, device=device)
            output = model(x, symbol_ids=sym_ids)

            action_logits = output['action_logits']
            calibrated_logits = action_logits / max(temperature, 0.01)
            action_probs = torch.softmax(calibrated_logits, dim=-1).cpu().numpy().flatten()

            p_hold = float(action_probs[0])
            p_enter = 1.0 - p_hold

            ret_mu = float(output['ret_mu'].cpu().item())
            mfe = float(output['mfe'].cpu().item())
            mae = float(output['mae'].cpu().item())

            enter_logit = float(action_logits[0, 1].cpu().item() - action_logits[0, 0].cpu().item())

            return {
                'p_enter': p_enter,
                'e_net_pred': ret_mu,
                'enter_logit': enter_logit,
                'temperature_used': temperature,
                'v5_action_probs': action_probs.tolist(),
                'v5_ret_mu': ret_mu,
                'v5_mfe': mfe,
                'v5_mae': mae,
            }
        else:
            sym_ids = None
            if symbol_id is not None and model.symbol_embedding is not None:
                sym_ids = torch.tensor([symbol_id], dtype=torch.long, device=device)
            output = model.forward_multihead(x, symbol_ids=sym_ids)

            enter_logit = float(output.enter_logits.cpu().item())
            calibrated_logit = enter_logit / max(temperature, 0.01)
            p_enter = float(torch.sigmoid(torch.tensor(calibrated_logit)).item())

            e_net_pred = 0.0
            if output.value_logits is not None:
                e_net_pred = float(output.value_logits.cpu().item())

            return {
                'p_enter': p_enter,
                'e_net_pred': e_net_pred,
                'enter_logit': enter_logit,
                'temperature_used': temperature,
            }


def _apply_htf_gates(features_df: pd.DataFrame) -> dict:
    """Apply HTF gates on the last row — identical to make_enter_prediction."""
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

    return {
        'trend_aligned': trend_aligned,
        'slope_ok': slope_ok,
        'range_ok': range_ok,
        'side': side,
        'h1_trend': h1_trend,
        'h4_trend': h4_trend,
        'h1_slope': h1_slope,
        'h1_range_pos': h1_range_pos,
    }


def _compute_htf_score(htf: dict, direction: str) -> int:
    """Compute HTF score (0–3) for a given trade direction.

    +1 if h1_trend matches direction
    +1 if h4_trend matches direction
    +1 if slope_ok == True
    """
    score = 0
    dir_sign = 1 if direction == "LONG" else (-1 if direction == "SHORT" else 0)
    if dir_sign == 0:
        return 0

    h1 = htf.get('h1_trend', 0)
    h4 = htf.get('h4_trend', 0)
    slope_ok = htf.get('slope_ok', False)
    range_ok = htf.get('range_ok', False)

    if int(h1) == dir_sign:
        score += 1
    if int(h4) == dir_sign:
        score += 1
    if slope_ok:
        score += 1

    log.info(f"HTF_SCORE: score={score} h1={h1} h4={h4} slope_ok={slope_ok} range_ok={range_ok} dir={direction}")
    return score


SCALP_ATR_RATIO_MIN = 1.20
SCALP_TR_Z_MIN = 1.0
SCALP_BB_Z_MIN = 1.0
SCALP_SLOPE_MIN = 0.0005
SCALP_MACD_MIN = 0.0001
SCALP_VOL_RATIO_MIN = 1.2


def _compute_scalp_gates(df_candles: pd.DataFrame, features_df: pd.DataFrame,
                         direction: str, atr: float) -> dict:
    """Compute all SCALP gate metrics and pass/fail flags.

    Returns dict with:
      vol_expansion_ok, momentum_ok, atr_ratio, true_range_z, bb_width_z,
      ema20_slope, macd_hist_val, volume_ratio, range_ok_adjusted_mult
    """
    n = len(df_candles)
    last_row = features_df.iloc[-1] if len(features_df) > 0 else {}

    atr14 = atr
    atr50 = _compute_atr(df_candles, window=50) if n > 51 else atr
    atr_ratio = atr14 / atr50 if atr50 > 0 else 0.0

    highs = df_candles['high'].values.astype(float)
    lows = df_candles['low'].values.astype(float)
    closes = df_candles['close'].values.astype(float)
    trs = []
    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    if len(trs) >= 20:
        tr_arr = np.array(trs)
        tr_mean = tr_arr[-20:].mean()
        tr_std = tr_arr[-20:].std()
        true_range_z = (tr_arr[-1] - tr_mean) / tr_std if tr_std > 0 else 0.0
    else:
        true_range_z = 0.0

    bb_width = float(last_row.get('bb_width', 0))
    if n > 20 and 'bb_width' in features_df.columns:
        bw_vals = features_df['bb_width'].iloc[-20:].values.astype(float)
        bw_mean = np.nanmean(bw_vals)
        bw_std = np.nanstd(bw_vals)
        bb_width_z = (bb_width - bw_mean) / bw_std if bw_std > 0 else 0.0
    else:
        bb_width_z = 0.0

    vol_expansion_ok = (atr_ratio >= SCALP_ATR_RATIO_MIN and
                        (true_range_z >= SCALP_TR_Z_MIN or bb_width_z >= SCALP_BB_Z_MIN))

    if n >= 20:
        ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean()
        ema20_slope = (ema20.iloc[-1] - ema20.iloc[-2]) / ema20.iloc[-2] if ema20.iloc[-2] != 0 else 0.0
    else:
        ema20_slope = 0.0

    macd_hist_val = float(last_row.get('macd_hist', 0))

    slope_pass = abs(ema20_slope) >= SCALP_SLOPE_MIN
    macd_pass = abs(macd_hist_val) >= SCALP_MACD_MIN
    signal_component = slope_pass or macd_pass

    recent_vol = df_candles['volume'].iloc[-4:].mean() if n >= 4 else 0
    avg_vol = df_candles['volume'].iloc[-20:].mean() if n >= 20 else (df_candles['volume'].mean() if n > 0 else 1)
    volume_ratio = float(recent_vol / avg_vol) if avg_vol > 0 else 0.0

    momentum_ok = signal_component and volume_ratio >= SCALP_VOL_RATIO_MIN

    return {
        'vol_expansion_ok': vol_expansion_ok,
        'momentum_ok': momentum_ok,
        'atr_ratio': round(atr_ratio, 4),
        'true_range_z': round(true_range_z, 4),
        'bb_width_z': round(bb_width_z, 4),
        'ema20_slope': round(ema20_slope, 6),
        'macd_hist_val': round(macd_hist_val, 6),
        'volume_ratio': round(volume_ratio, 4),
    }


def _compute_momentum_ok(features_df: pd.DataFrame, direction: str) -> bool:
    """Check momentum conditions for FLOW: ADX >= 18 OR MACD matches direction."""
    last_row = features_df.iloc[-1]
    adx = last_row.get('adx_14', 0)
    if adx >= 18:
        return True
    macd_val = last_row.get('macd', 0)
    if direction == "LONG" and macd_val > 0:
        return True
    if direction == "SHORT" and macd_val < 0:
        return True
    return False


def _compute_atr(df: pd.DataFrame, window: int = 14) -> float:
    """Compute ATR from candle data."""
    n = min(window + 1, len(df))
    if n < 3:
        return float(df.iloc[-1]['close']) * 0.005

    highs = df.iloc[-n:]['high'].values.astype(float)
    lows = df.iloc[-n:]['low'].values.astype(float)
    closes = df.iloc[-n:]['close'].values.astype(float)

    true_ranges = []
    for i in range(1, len(highs)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i] - closes[i - 1]))
        true_ranges.append(tr)
    return float(np.mean(true_ranges))


def _build_prediction_payload(
    symbol: str, side: str, p_enter: float, current_price: float, atr: float,
    entry_price: float, tp_mult: float, sl_mult: float, htf: dict,
    exec_result=None
) -> dict:
    """Build a prediction dict compatible with the dashboard push endpoint."""
    if side == "LONG":
        sl_price = entry_price - sl_mult * atr
        tp_price = entry_price + tp_mult * atr
    else:
        sl_price = entry_price + sl_mult * atr
        tp_price = entry_price - tp_mult * atr

    sl_pct = abs(entry_price - sl_price) / entry_price
    tp_pct = abs(tp_price - entry_price) / entry_price
    rr = tp_pct / sl_pct if sl_pct > 0 else 1.0

    ACCOUNT_RISK = 0.02
    position_size = ACCOUNT_RISK / sl_pct * 100 if sl_pct > 0 else 1.0
    if p_enter > 0.75:
        position_size *= 1.25
    elif p_enter < 0.55:
        position_size *= 0.5
    position_size = min(max(position_size, 0.5), 5.0)

    risk_pct = min(position_size * sl_pct * 100, 5.0)

    reasons = []
    reasons.append(f"ENTER signal: p_enter={p_enter:.1%}")
    reasons.append(f"HTF trend: {side} (1H={htf['h1_trend']:+.0f}, 4H={htf['h4_trend']:+.0f})")
    if exec_result and exec_result.executed and exec_result.method == "pullback_confirm":
        reasons.append(f"Improved entry via {exec_result.method} ({exec_result.cost_improvement_bps:+.1f} bps)")

    return {
        "symbol": symbol,
        "action": side,
        "confidence": round(p_enter, 4),
        "direction_probs": {
            "SHORT": round(1.0 if side == "SHORT" else 0.0, 4),
            "HOLD": 0.0,
            "LONG": round(1.0 if side == "LONG" else 0.0, 4),
        },
        "quantiles": {},
        "vol_state": "neutral",
        "vol_state_probs": {"contraction": 0.33, "neutral": 0.34, "expansion": 0.33},
        "expected_return": round(p_enter - 0.5, 6),
        "uncertainty": round(1.0 - p_enter, 6),
        "edge": round(p_enter - 0.5, 4),
        "entry_price": round(entry_price, 2),
        "stop_loss_price": round(sl_price, 2),
        "take_profit_price": round(tp_price, 2),
        "stop_loss_pct": round(sl_pct, 4),
        "take_profit_pct": round(tp_pct, 4),
        "risk_reward_ratio": round(rr, 2),
        "position_size_pct": round(position_size, 1),
        "current_price": round(current_price, 2),
        "model_name": "enter_quality_v3.3_live_multi",
        "is_multihead": True,
        "urgency": "high" if p_enter > 0.7 else "medium",
        "suggested_order_type": "limit",
        "reasons": reasons,
        "risk_pct": round(risk_pct, 2),
        "p_enter": round(p_enter, 4),
        "atr": round(atr, 2),
    }


class LiveRunner:
    def __init__(
        self,
        replit_url: str,
        symbols: List[str],
        device: str,
        interval: str = "15m",
        enter_threshold: float = 0.85,
        tp_mult: float = 3.5,
        sl_mult: float = 1.5,
        cooldown_bars: int = 8,
        paper: bool = False,
        execution_mode: str = "signal_only",
        record_trades: bool = False,
        portfolio_manager=None,
        execution_module=None,
        dry_run: bool = False,
        dry_run_candles: int = 200,
        per_symbol_models: bool = False,
        limit_15m: int = REQUIRED_CANDLES,
        direct_htf: bool = False,
        budget_core: float = None,
        budget_flow: float = None,
        budget_scalp: float = None,
    ):
        self.replit_url = replit_url
        self.symbols = symbols
        self.device = device
        self.interval = interval
        self.enter_threshold = enter_threshold
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.cooldown_bars = cooldown_bars
        self.paper = paper
        self.execution_mode = execution_mode
        self.record_trades = record_trades
        self.portfolio = portfolio_manager
        self.execution = execution_module
        self.dry_run = dry_run
        self.dry_run_candles = dry_run_candles
        self.per_symbol_models = per_symbol_models
        self.limit_15m = limit_15m
        self.direct_htf = direct_htf

        self.lane_budgets = {
            "CORE": budget_core if budget_core is not None else LANE_BUDGET["CORE"],
            "FLOW": budget_flow if budget_flow is not None else LANE_BUDGET["FLOW"],
            "SCALP": budget_scalp if budget_scalp is not None else LANE_BUDGET["SCALP"],
        }
        log.info(f"[INIT] LiveRunner {SYSTEM_VERSION} execution_mode={execution_mode} "
                 f"record_trades={record_trades} symbols={symbols}")
        log.info(f"[CONFIG] Lane budgets: CORE={self.lane_budgets['CORE']:.2f}R "
                 f"FLOW={self.lane_budgets['FLOW']:.2f}R SCALP={self.lane_budgets['SCALP']:.2f}R")

        self.model = None
        self.engineer = None
        self.feature_columns = None
        self.symbol_models: Dict[str, Tuple] = {}
        self.fetcher = None
        self.cycle_count = 0
        self.candle_cache: Dict[str, pd.DataFrame] = {}
        self.exchange_time_offset = 0.0
        self.cooldown_tracker: Dict[str, int] = {}
        self.p_enter_history: Dict[str, List[float]] = {}
        self.warmup_logged: Dict[str, bool] = {}

        self.daily_budget: Dict[str, Dict[str, float]] = {}
        self.daily_budget_date: Dict[str, str] = {}
        self.budget_block_logged: Dict[str, Dict[str, bool]] = {}

        from trade_manager import TradeManager
        self.trade_manager = TradeManager()

    def _init_fetcher(self):
        from data.pipeline import BinanceDataFetcher
        self.fetcher = BinanceDataFetcher(
            symbols=self.symbols,
            timeframes=[self.interval],
            replit_proxy_url=self.replit_url.rstrip('/'),
            use_sync=True,
        )
        if self.execution and self.execution.fetcher is None:
            self.execution.fetcher = self.fetcher

    def _on_position_close(self, pos, exit_price: float, outcome: str, gross_r: float):
        """Callback when portfolio closes a position — push trade update to dashboard."""
        if hasattr(self, 'separation_verifier') and self.separation_verifier:
            self.separation_verifier.record_exit_resolve()
        from portfolio import Position as _Pos

        initial_sl = pos.original_sl if pos.original_sl is not None else pos.sl_price
        original_risk_abs = abs(pos.entry_price - initial_sl)

        if pos.is_long:
            gross_r_calc = (exit_price - pos.entry_price) / original_risk_abs if original_risk_abs > 0 else 0
        else:
            gross_r_calc = (pos.entry_price - exit_price) / original_risk_abs if original_risk_abs > 0 else 0

        if abs(gross_r_calc - gross_r) > 0.01:
            log.warning(f"[R_CHECK] MISMATCH sym={pos.symbol} side={pos.side} "
                        f"entry={pos.entry_price} exit={exit_price} initial_sl={initial_sl} "
                        f"orig_risk_abs={original_risk_abs:.4f} "
                        f"gross_r_calc={gross_r_calc:.6f} gross_r_stored={gross_r:.6f}")
            gross_r = gross_r_calc

        cost_bps = COST_BPS
        cost_r = (cost_bps / 10000) * 2 / (original_risk_abs / pos.entry_price) if original_risk_abs > 0 else 0.0
        net_r = gross_r - cost_r
        sized_r = net_r * pos.size_mult

        log.info(f"[R_CHECK] sym={pos.symbol} side={pos.side} entry={pos.entry_price:.2f} "
                 f"exit={exit_price:.2f} initial_sl={initial_sl:.2f} orig_risk_abs={original_risk_abs:.4f} "
                 f"gross_r_calc={gross_r_calc:.6f} gross_r_stored={gross_r:.6f} "
                 f"cost_r={cost_r:.6f} net_r={net_r:.6f}")

        check = abs(net_r - (gross_r - cost_r))
        if check > 1e-6:
            log.error(f"[NET_CHECK] INVARIANT VIOLATED: net_r={net_r:.6f} != gross_r={gross_r:.6f} - cost_r={cost_r:.6f} (diff={check:.8f})")

        if hasattr(self, 'verifier') and self.verifier:
            vf = self.verifier
            fs = vf.verify_net_r(vf.stats.total_cycles, gross_r, cost_r, net_r)
            vf.add_failures(fs)

        risk_usd = 0.0
        try:
            import requests as _req
            money_resp = _req.get(f"{self.replit_url.rstrip('/')}/api/money-config", timeout=5)
            if money_resp.status_code == 200:
                mc = money_resp.json()
                equity = mc.get('account_equity_usd', 1500)
                risk_pct = mc.get('risk_per_trade_pct', 1.0)
                risk_usd = equity * (risk_pct / 100) * pos.size_mult
        except Exception:
            pass
        gross_usd = round(gross_r * risk_usd, 2)
        cost_usd = round(cost_r * risk_usd, 2)
        net_usd = round(net_r * risk_usd, 2)

        exit_reason = outcome
        if outcome == "TIME_EXIT":
            exit_reason = f"SCALP_TIME_STOP ({pos.horizon} bars)"

        tm_state = self.trade_manager.get_state(pos.symbol)
        mfe = tm_state['max_favorable_r'] if tm_state else None
        mae = tm_state['max_adverse_r'] if tm_state else None
        be_moved = tm_state['breakeven_moved'] if tm_state else pos.breakeven_moved
        bars_held = self.cycle_count - pos.bar_index if pos.bar_index > 0 else None
        if bars_held is None and pos.entry_time > 0:
            bars_held = max(1, int((time.time() - pos.entry_time) / (15 * 60)))
        if bars_held is None:
            bars_held = 0
        is_time_exit = outcome == "TIME_EXIT"

        self.trade_manager.clear_position(pos.symbol)

        try:
            self._update_trade_record(
                trade_id=pos.dashboard_trade_id,
                exit_price=exit_price,
                outcome=outcome,
                gross_r=gross_r,
                net_r=net_r,
                sized_r=sized_r,
                cost_r=cost_r,
                exit_reason=exit_reason,
                max_favorable_r=mfe,
                max_adverse_r=mae,
                time_exit=is_time_exit,
                breakeven_moved=be_moved,
                bars_held=bars_held,
                initial_sl=initial_sl,
                risk_usd=risk_usd,
                gross_usd=gross_usd,
                cost_usd=cost_usd,
                net_usd=net_usd,
            )
        except Exception as e:
            log.warning(f"Failed to update trade record {pos.dashboard_trade_id}: {e}")

    def _push_prediction(self, prediction: dict):
        from quick_start import push_prediction
        push_prediction(self.replit_url, prediction)

    def _push_cycle_log(self, symbol: str, price: float, p_enter: float,
                        htf: dict, direction: str, decision: str, reasons: list,
                        lane_info: Optional[dict] = None,
                        e_net_pred: float = None, enter_logit: float = None,
                        temperature_used: float = None):
        url = f"{self.replit_url.rstrip('/')}/api/live/cycle-log"
        li = lane_info or {}
        payload = {
            "symbol": symbol,
            "cycle_ts": int(time.time() * 1000),
            "price": float(price),
            "p_enter": float(p_enter),
            "htf_h1_trend": str(htf.get('h1_trend', '')),
            "htf_h4_trend": str(htf.get('h4_trend', '')),
            "slope_ok": bool(htf.get('slope_ok', False)),
            "range_ok": bool(htf.get('range_ok', False)),
            "direction": str(direction),
            "threshold_used": float(li.get('threshold_used', self.enter_threshold)),
            "decision": str(decision),
            "reasons": [str(r) for r in reasons] if reasons else [],
            "lane_selected": li.get('lane_selected'),
            "htf_score": li.get('htf_score'),
            "core_thr": li.get('core_thr'),
            "flow_thr": li.get('flow_thr'),
            "scalp_thr": li.get('scalp_thr'),
            "lane_size_mult": li.get('lane_size_mult'),
            "lane_budget_remaining_r": li.get('lane_budget_remaining_r'),
            "hold_reason": li.get('hold_reason'),
            "quota_step": li.get('quota_step'),
            "e_net_pred": round(float(e_net_pred), 4) if e_net_pred is not None else li.get('e_net_pred'),
            "enter_logit": round(float(enter_logit), 4) if enter_logit is not None else None,
            "temperature_used": round(float(temperature_used), 4) if temperature_used is not None else None,
        }
        sg = li.get('scalp_gates')
        if sg:
            payload["scalp_atr_ratio"] = sg.get('atr_ratio')
            payload["scalp_tr_z"] = sg.get('true_range_z')
            payload["scalp_bb_z"] = sg.get('bb_width_z')
            payload["scalp_ema20_slope"] = sg.get('ema20_slope')
            payload["scalp_macd_hist"] = sg.get('macd_hist_val')
            payload["scalp_vol_ratio"] = sg.get('volume_ratio')
            payload["scalp_vol_expansion_ok"] = sg.get('vol_expansion_ok')
            payload["scalp_momentum_ok"] = sg.get('momentum_ok')
        payload_keys = [k for k, v in payload.items() if v is not None]
        log.debug(f"[CYCLE_PAYLOAD] sym={symbol} fields_present={payload_keys}")
        _retry_request("POST", url, json=payload)

    def _push_trade_record(self, symbol: str, side: str, entry_price: float,
                           sl_price: float, tp_price: float, p_enter: float,
                           size_pct: float, lane_info: Optional[dict] = None) -> Optional[int]:
        url = f"{self.replit_url.rstrip('/')}/api/live/trade"
        li = lane_info or {}
        payload = {
            "symbol": symbol,
            "side": side,
            "entry_time": int(time.time() * 1000),
            "entry_price": entry_price,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "initial_sl": sl_price,
            "p_enter": p_enter,
            "size_pct": size_pct,
            "status": "open",
            "lane": li.get('lane'),
            "htf_score": li.get('htf_score'),
            "lane_threshold_used": li.get('threshold_used'),
            "lane_size_mult": li.get('lane_size_mult'),
            "lane_horizon": li.get('lane_horizon'),
        }
        resp = _retry_request("POST", url, json=payload)
        if resp and resp.status_code == 200:
            data = resp.json()
            return data.get("id")
        return None

    def _update_trade_record(self, trade_id: int, exit_price: float,
                             outcome: str, gross_r: float, net_r: float, sized_r: float,
                             cost_r: float = 0.0,
                             exit_reason: Optional[str] = None,
                             max_favorable_r: Optional[float] = None,
                             max_adverse_r: Optional[float] = None,
                             time_exit: Optional[bool] = None,
                             breakeven_moved: Optional[bool] = None,
                             bars_held: Optional[int] = None,
                             initial_sl: Optional[float] = None,
                             risk_usd: float = 0.0,
                             gross_usd: float = 0.0,
                             cost_usd: float = 0.0,
                             net_usd: float = 0.0):
        url = f"{self.replit_url.rstrip('/')}/api/live/trade/{trade_id}"
        payload = {
            "exit_time": int(time.time() * 1000),
            "exit_price": exit_price,
            "outcome": outcome,
            "gross_r": round(gross_r, 6),
            "net_r": round(net_r, 6),
            "sized_r": round(sized_r, 6),
            "cost_r": round(cost_r, 6),
            "status": "closed",
            "exit_reason": exit_reason or outcome,
            "bars_held": bars_held if bars_held is not None else 0,
            "risk_usd_used": round(risk_usd, 2),
            "pnl_usd_gross": round(gross_usd, 2),
            "pnl_usd_cost": round(cost_usd, 2),
            "pnl_usd": round(net_usd, 2),
        }
        if initial_sl is not None:
            payload["initial_sl"] = round(initial_sl, 6)
        if max_favorable_r is not None:
            payload["max_favorable_r"] = round(max_favorable_r, 4)
        if max_adverse_r is not None:
            payload["max_adverse_r"] = round(max_adverse_r, 4)
        if time_exit is not None:
            payload["time_exit"] = time_exit
        if breakeven_moved is not None:
            payload["breakeven_moved"] = breakeven_moved
        _retry_request("PATCH", url, json=payload)

    def _update_trade_sl(self, trade_id: int, new_sl: float):
        """Update stop loss on an open trade record in the dashboard."""
        url = f"{self.replit_url.rstrip('/')}/api/live/trade/{trade_id}"
        payload = {"stop_loss": new_sl}
        _retry_request("PATCH", url, json=payload)

    def _get_model_for_symbol(self, symbol: str):
        if self.per_symbol_models:
            if symbol not in self.symbol_models:
                try:
                    m, e, fc, temp, sm = _load_model(self.device, symbol=symbol)
                    self.symbol_models[symbol] = (m, e, fc, temp, sm)
                except SystemExit:
                    log.warning(f"No per-symbol model for {symbol}, using global model")
                    self.symbol_models[symbol] = (self.model, self.engineer, self.feature_columns, self.temperature, self.symbol_map)
            return self.symbol_models[symbol]
        return self.model, self.engineer, self.feature_columns, self.temperature, self.symbol_map

    def _update_candle_cache(self, symbol: str, new_df: pd.DataFrame) -> pd.DataFrame:
        if symbol not in self.candle_cache:
            self.candle_cache[symbol] = new_df.copy()
        else:
            existing = self.candle_cache[symbol]
            combined = pd.concat([existing, new_df], ignore_index=True)
            if 'timestamp' in combined.columns:
                combined = combined.drop_duplicates(subset='timestamp', keep='last')
                combined = combined.sort_values('timestamp').reset_index(drop=True)
            if len(combined) > MAX_CACHE_BARS:
                combined = combined.iloc[-MAX_CACHE_BARS:].reset_index(drop=True)
            self.candle_cache[symbol] = combined
        return self.candle_cache[symbol]

    def _interval_seconds(self) -> int:
        if self.interval == "15m":
            return 15 * 60
        elif self.interval == "5m":
            return 5 * 60
        elif self.interval == "1m":
            return 60
        return 15 * 60

    def run(self):
        """Main loop — runs continuously until interrupted."""
        mode_label = {"signal_only": "SIGNAL_ONLY", "paper": "PAPER", "live": "LIVE"}.get(self.execution_mode, "UNKNOWN")
        log.info("=" * 80)
        log.info(f"  LIVE RUNNER v4.5.2 ({mode_label})")
        log.info(f"  [MODE] execution_mode={self.execution_mode} record_trades={self.record_trades} "
                 f"paper={self.paper} live={self.execution_mode == 'live'}")
        log.info(f"  Symbols: {', '.join(self.symbols)}")
        log.info(f"  Interval: {self.interval} | Threshold: {self.enter_threshold}")
        log.info(f"  TP={self.tp_mult}x SL={self.sl_mult}x | Cooldown: {self.cooldown_bars} bars")
        log.info(f"  Per-symbol models: {self.per_symbol_models}")
        log.info(f"  Triple-Lane Engine: CORE(p99/1.0x) FLOW(p95-stepped) SCALP(p90/0.25x/4bar)")
        total_budget = sum(self.lane_budgets.values())
        log.info(f"  Daily R Budget: {total_budget:.2f}R total | CORE={self.lane_budgets['CORE']:.2f}R FLOW={self.lane_budgets['FLOW']:.2f}R SCALP={self.lane_budgets['SCALP']:.2f}R")

        self.portfolio.on_close_callback = self._on_position_close
        log.info(f"  15m fetch limit: {self.limit_15m} | Direct HTF fetch: {self.direct_htf}")
        log.info(f"  HTF warmup gates: min h1={MIN_H1_BARS} h4={MIN_H4_BARS} bars")
        if self.dry_run:
            log.info(f"  DRY RUN MODE — replaying cached candles")
        log.info("=" * 80)

        self.exchange_time_offset = _get_exchange_time_offset()
        log.info(f"Exchange time offset: {self.exchange_time_offset*1000:.0f}ms")

        self.model, self.engineer, self.feature_columns, self.temperature, self.symbol_map = _load_model(self.device)
        self._init_fetcher()

        for sym in self.symbols:
            self.cooldown_tracker[sym] = 0

        if self.dry_run:
            self._run_dry()
            return

        self._should_stop = False
        try:
            while not self._should_stop:
                self._run_cycle()

                if self._should_stop:
                    log.info("Verification cycle limit reached — exiting run loop.")
                    break

                if hasattr(self, 'learning_manager') and self.learning_manager:
                    try:
                        self.learning_manager.check_and_retrain_all()
                    except Exception as e:
                        log.error(f"Learning check failed: {e}")

                interval_s = self._interval_seconds()
                now = time.time() + self.exchange_time_offset
                next_bar = (int(now) // interval_s + 1) * interval_s
                wait = max(next_bar - now + 5, 10)
                log.info(f"Next cycle in {wait:.0f}s...")
                time.sleep(wait)
        except KeyboardInterrupt:
            log.info("Live runner stopped by user.")
            self._print_summary()

    def _run_dry(self):
        """Dry-run mode: replay cached data instead of fetching live."""
        log.info("Loading cached data for dry run...")

        data_dir = Path("data_cache")

        for symbol in self.symbols:
            parquet_path = data_dir / f"{symbol}_15m.parquet"
            if not parquet_path.exists():
                parquet_path = data_dir / "BTCUSDT_15m.parquet"
                if not parquet_path.exists():
                    log.error(f"No cached data for dry run. Run --predict-only first to download data.")
                    return

            df_full = pd.read_parquet(parquet_path)
            n_bars = min(self.dry_run_candles, len(df_full) - REQUIRED_CANDLES)
            if n_bars <= 0:
                log.error(f"Insufficient cached data for {symbol}")
                continue

            log.info(f"Dry run: {symbol} — replaying {n_bars} bars from cache")

            for i in range(n_bars):
                end_idx = REQUIRED_CANDLES + i
                df_slice = df_full.iloc[end_idx - REQUIRED_CANDLES:end_idx].copy().reset_index(drop=True)

                self.portfolio.set_bar(self.cycle_count)
                prices = {symbol: float(df_slice.iloc[-1]['close'])}
                self.portfolio.check_exits(prices)

                self._process_symbol(symbol, df_candles=df_slice)
                self.cycle_count += 1

        self._print_summary()

    def _run_trade_manager(self, prices: Dict[str, float],
                           highs: Dict[str, float], lows: Dict[str, float]):
        """Run Smart Trade Manager over all open positions.

        For each open position, compute current HTF score and p_enter,
        then ask TradeManager for an action. Handle MOVE_SL, TRAIL_SL, CLOSE_FULL.
        """
        open_positions = dict(self.portfolio.open_positions)
        if not open_positions:
            return

        for symbol, pos in open_positions.items():
            price = prices.get(symbol)
            if price is None:
                continue

            candle_high = highs.get(symbol)
            candle_low = lows.get(symbol)

            current_htf_score = getattr(pos, 'htf_score', None)
            current_p_enter = None

            action = self.trade_manager.update_position(
                symbol=symbol,
                pos=pos,
                current_price=price,
                current_bar=self.cycle_count,
                current_htf_score=current_htf_score,
                current_p_enter=current_p_enter,
                candle_high=candle_high,
                candle_low=candle_low,
            )

            if action.action == "HOLD":
                continue

            if action.action in ("MOVE_SL", "TRAIL_SL") and action.new_sl is not None:
                pos.sl_price = action.new_sl
                if action.reason == "BREAKEVEN":
                    pos.breakeven_moved = True
                if pos.dashboard_trade_id:
                    try:
                        self._update_trade_sl(pos.dashboard_trade_id, action.new_sl)
                    except Exception as e:
                        log.warning(f"Failed to update SL on dashboard for {symbol}: {e}")

            elif action.action == "CLOSE_FULL":
                if symbol not in self.portfolio.open_positions:
                    continue
                close_price = action.close_price or price
                self.portfolio.close_position(symbol, close_price, action.reason)

    def _run_cycle(self):
        """One 15m cycle: fetch, predict, rank, execute for all symbols."""
        self.cycle_count += 1
        self.portfolio.set_bar(self.cycle_count)

        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        log.info("")
        log.info(f"{'='*60}")
        log.info(f"CYCLE {self.cycle_count} — {timestamp}")
        log.info(f"{'='*60}")

        prices = {}
        highs = {}
        lows = {}
        candle_dfs = {}
        for symbol in self.symbols:
            df = _fetch_candles_for_symbol(self.fetcher, symbol, self.interval, limit=self.limit_15m)
            if df is not None and len(df) > 0:
                prices[symbol] = float(df.iloc[-1]['close'])
                if 'high' in df.columns:
                    highs[symbol] = float(df.iloc[-1]['high'])
                if 'low' in df.columns:
                    lows[symbol] = float(df.iloc[-1]['low'])
                candle_dfs[symbol] = df
        if self.execution_mode in ("paper", "live") and self.record_trades:
            self.portfolio.check_exits(prices, highs=highs, lows=lows)
            self._run_trade_manager(prices, highs, lows)

        candidates = []
        for symbol in self.symbols:
            result = self._process_symbol(symbol)
            if result:
                candidates.append(result)

        if not candidates:
            log.info("No candidates this cycle.")
            return

        accepted = self.portfolio.filter_and_rank(candidates)
        if not accepted:
            log.info("All candidates rejected by portfolio rules.")
            return

        for c in accepted:
            self._execute_candidate(c)

    def _reset_daily_budget_if_needed(self, symbol: str):
        """Reset daily R budget for symbol at UTC day boundary."""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if self.daily_budget_date.get(symbol) != today:
            self.daily_budget[symbol] = {
                "CORE": self.lane_budgets["CORE"],
                "FLOW": self.lane_budgets["FLOW"],
                "SCALP": self.lane_budgets["SCALP"],
            }
            self.daily_budget_date[symbol] = today
            self.budget_block_logged[symbol] = {"CORE": False, "FLOW": False, "SCALP": False}

    def _check_lane_budget(self, symbol: str, lane: str, size_mult: float) -> bool:
        """Check if lane has enough R budget remaining. Reserve = 1.0R * size_mult."""
        self._reset_daily_budget_if_needed(symbol)
        reserve_r = 1.0 * size_mult
        remaining = self.daily_budget.get(symbol, {}).get(lane, 0.0)
        if remaining < reserve_r:
            if not self.budget_block_logged.get(symbol, {}).get(lane, False):
                log.warning(f"BUDGET_BLOCK: {symbol} lane={lane} remaining={remaining:.2f} < need={reserve_r:.2f} -> disabled today")
                self.budget_block_logged.setdefault(symbol, {})[lane] = True
            return False
        return True

    def _spend_lane_budget(self, symbol: str, lane: str, size_mult: float):
        """Deduct R budget after a trade is opened."""
        reserve_r = 1.0 * size_mult
        self.daily_budget.setdefault(symbol, {})
        self.daily_budget[symbol][lane] = self.daily_budget[symbol].get(lane, 0.0) - reserve_r

    def _get_lane_budget_remaining(self, symbol: str, lane: str) -> float:
        """Get remaining R budget for a lane."""
        self._reset_daily_budget_if_needed(symbol)
        return self.daily_budget.get(symbol, {}).get(lane, 0.0)

    def _get_percentile(self, symbol: str, pct: int) -> Optional[float]:
        """Get p_enter percentile from history for a symbol."""
        hist = self.p_enter_history.get(symbol, [])
        if len(hist) < 20:
            return None
        return float(np.percentile(hist, pct))

    def _select_lane(self, symbol: str, p_enter: float, htf: dict,
                     htf_score: int, features_df: pd.DataFrame,
                     df_candles: pd.DataFrame, atr: float,
                     e_net_pred: float = 0.0) -> dict:
        """Triple-lane router: CORE > FLOW > SCALP > HOLD.

        Returns dict with lane selection info including:
        lane_selected, threshold_used, lane_size_mult, lane_horizon,
        core_thr, flow_thr, scalp_thr, hold_reason, quota_step, e_net_pred
        """
        range_ok = htf.get('range_ok', False)
        side = htf.get('side', 'NEUTRAL')
        if side == 'NEUTRAL':
            return {
                'lane_selected': 'HOLD', 'threshold_used': 0.0,
                'lane_size_mult': 0.0, 'lane_horizon': 24,
                'core_thr': None, 'flow_thr': None, 'scalp_thr': None,
                'hold_reason': 'NEUTRAL_SIDE', 'htf_score': htf_score,
                'lane_budget_remaining_r': 0.0, 'quota_step': None, 'lane': None,
            }

        p99 = self._get_percentile(symbol, 99)
        p95 = self._get_percentile(symbol, 95)
        p90 = self._get_percentile(symbol, 90)

        core_thr = max(self.enter_threshold, p99) if p99 is not None else self.enter_threshold
        scalp_thr = p90 if p90 is not None else 0.65

        flow_momentum_ok = _compute_momentum_ok(features_df, side)
        scalp_gates = _compute_scalp_gates(df_candles, features_df, side, atr)
        log.info(f"[SCALP_GATES] sym={symbol} vol_ok={scalp_gates['vol_expansion_ok']} "
                 f"mom_ok={scalp_gates['momentum_ok']} atr_ratio={scalp_gates['atr_ratio']:.4f} "
                 f"bb_z={scalp_gates['bb_width_z']:.4f} tr_z={scalp_gates['true_range_z']:.4f} "
                 f"vol_ratio={scalp_gates['volume_ratio']:.4f}")

        quota_step = 0
        sym_hist = self.p_enter_history.get(symbol, [])
        trades_today = self.daily_budget_date.get(symbol, '')
        if len(sym_hist) >= 20:
            recent_hist = np.array(sym_hist)
            target_trades_per_day = 3
            actual_today = self.lane_budgets["FLOW"] - self._get_lane_budget_remaining(symbol, "FLOW")
            if actual_today > 0:
                ratio = actual_today / self.lane_budgets["FLOW"]
                if ratio < 0.25:
                    quota_step = 3
                elif ratio < 0.5:
                    quota_step = 2
                elif ratio < 0.75:
                    quota_step = 1
                else:
                    quota_step = 0

        flow_config = FLOW_QUOTA_STEPS.get(quota_step, FLOW_QUOTA_STEPS[0])
        flow_pct = flow_config["percentile"]
        flow_size_mult = flow_config["size_mult"]
        flow_thr_val = self._get_percentile(symbol, flow_pct)
        flow_thr = flow_thr_val if flow_thr_val is not None else 0.75

        flow_budget_used = self.lane_budgets["FLOW"] - self._get_lane_budget_remaining(symbol, "FLOW")
        log.info(f"[QUOTA] sym={symbol} trades_today={flow_budget_used:.2f}R_used "
                 f"target_tpd=3 step={quota_step} pressure={quota_step} "
                 f"flow_pct={flow_pct} flow_size_mult={flow_size_mult:.2f}")

        min_enet_core = getattr(self, 'min_enet_core', 0.00)
        min_enet_flow = getattr(self, 'min_enet_flow', -0.05)
        min_enet_scalp = getattr(self, 'min_enet_scalp', -0.02)

        result_base = {
            'htf_score': htf_score,
            'core_thr': round(core_thr, 4),
            'flow_thr': round(flow_thr, 4),
            'scalp_thr': round(scalp_thr, 4),
            'quota_step': quota_step,
            'e_net_pred': round(e_net_pred, 4),
        }

        if htf_score >= 3 and range_ok:
            if p_enter >= core_thr:
                if e_net_pred < min_enet_core:
                    log.info(f"[ENET_GATE] CORE blocked: e_net={e_net_pred:.4f} < min={min_enet_core:.4f}")
                elif self._check_lane_budget(symbol, "CORE", 1.0):
                    log.info(f"[BUDGET] sym={symbol} core_rem={self._get_lane_budget_remaining(symbol, 'CORE'):.2f} "
                             f"flow_rem={self._get_lane_budget_remaining(symbol, 'FLOW'):.2f} "
                             f"scalp_rem={self._get_lane_budget_remaining(symbol, 'SCALP'):.2f}")
                    return {**result_base,
                        'lane_selected': 'CORE', 'lane': 'CORE',
                        'threshold_used': core_thr, 'lane_size_mult': 1.0,
                        'lane_horizon': 24,
                        'lane_budget_remaining_r': self._get_lane_budget_remaining(symbol, "CORE"),
                        'hold_reason': None,
                        'scalp_gates': scalp_gates,
                    }

        if htf_score >= 2 and (range_ok or flow_momentum_ok):
            if p_enter >= flow_thr:
                if e_net_pred < min_enet_flow:
                    log.info(f"[ENET_GATE] FLOW blocked: e_net={e_net_pred:.4f} < min={min_enet_flow:.4f}")
                elif self._check_lane_budget(symbol, "FLOW", flow_size_mult):
                    log.info(f"[BUDGET] sym={symbol} core_rem={self._get_lane_budget_remaining(symbol, 'CORE'):.2f} "
                             f"flow_rem={self._get_lane_budget_remaining(symbol, 'FLOW'):.2f} "
                             f"scalp_rem={self._get_lane_budget_remaining(symbol, 'SCALP'):.2f}")
                    return {**result_base,
                        'lane_selected': 'FLOW', 'lane': 'FLOW',
                        'threshold_used': flow_thr, 'lane_size_mult': flow_size_mult,
                        'lane_horizon': 24,
                        'lane_budget_remaining_r': self._get_lane_budget_remaining(symbol, "FLOW"),
                        'hold_reason': None,
                        'scalp_gates': scalp_gates,
                    }

        if htf_score >= 1 and scalp_gates['vol_expansion_ok'] and scalp_gates['momentum_ok']:
            if p_enter >= scalp_thr:
                if e_net_pred < min_enet_scalp:
                    log.info(f"[ENET_GATE] SCALP blocked: e_net={e_net_pred:.4f} < min={min_enet_scalp:.4f}")
                else:
                    scalp_size = SCALP_SIZE_MULT
                    if not range_ok:
                        scalp_size *= 0.5
                    if self._check_lane_budget(symbol, "SCALP", scalp_size):
                        log.info(f"[BUDGET] sym={symbol} core_rem={self._get_lane_budget_remaining(symbol, 'CORE'):.2f} "
                                 f"flow_rem={self._get_lane_budget_remaining(symbol, 'FLOW'):.2f} "
                                 f"scalp_rem={self._get_lane_budget_remaining(symbol, 'SCALP'):.2f}")
                        return {**result_base,
                            'lane_selected': 'SCALP', 'lane': 'SCALP',
                            'threshold_used': scalp_thr, 'lane_size_mult': scalp_size,
                            'lane_horizon': SCALP_HORIZON,
                            'lane_budget_remaining_r': self._get_lane_budget_remaining(symbol, "SCALP"),
                            'hold_reason': None,
                            'scalp_gates': scalp_gates,
                        }

        hold_reasons = []
        if htf_score < 1:
            hold_reasons.append(f"htf_score={htf_score}<1")
        elif htf_score < 2:
            if p_enter < scalp_thr:
                hold_reasons.append(f"p_enter={p_enter:.4f}<scalp_thr={scalp_thr:.4f}")
            if not scalp_gates['vol_expansion_ok']:
                hold_reasons.append("vol_expansion_fail")
            if not scalp_gates['momentum_ok']:
                hold_reasons.append("momentum_fail")
        elif htf_score < 3:
            if p_enter < flow_thr:
                hold_reasons.append(f"p_enter={p_enter:.4f}<flow_thr={flow_thr:.4f}")
        else:
            if p_enter < core_thr:
                hold_reasons.append(f"p_enter={p_enter:.4f}<core_thr={core_thr:.4f}")
        if e_net_pred < min_enet_scalp:
            hold_reasons.append(f"e_net={e_net_pred:.4f}<min_enet_scalp={min_enet_scalp:.4f}")

        core_rem = self._get_lane_budget_remaining(symbol, "CORE")
        flow_rem = self._get_lane_budget_remaining(symbol, "FLOW")
        scalp_rem = self._get_lane_budget_remaining(symbol, "SCALP")
        log.info(f"[BUDGET] sym={symbol} core_rem={core_rem:.2f} flow_rem={flow_rem:.2f} scalp_rem={scalp_rem:.2f}")
        total_remaining = core_rem + flow_rem + scalp_rem
        return {**result_base,
            'lane_selected': 'HOLD', 'lane': None,
            'threshold_used': 0.0, 'lane_size_mult': 0.0,
            'lane_horizon': 24,
            'lane_budget_remaining_r': total_remaining,
            'hold_reason': '; '.join(hold_reasons) if hold_reasons else 'NO_LANE_MATCH',
            'scalp_gates': scalp_gates,
        }

    def _compute_htf_from_direct(self, htf_direct: Dict[str, pd.DataFrame],
                                  df_candles: pd.DataFrame) -> dict:
        """Compute HTF gate values from directly-fetched 1H/4H candle data.

        Mirrors _apply_htf_gates but uses real HTF bars instead of resampled features.
        Uses the last COMPLETED bar (second-to-last) for each timeframe to avoid lookahead.
        """
        result = {
            'trend_aligned': False, 'slope_ok': False, 'range_ok': False,
            'side': 'NEUTRAL', 'h1_trend': 0, 'h4_trend': 0,
            'h1_slope': 0.0, 'h1_range_pos': 0.5,
        }

        for tf_key, tf_label in [('1h', 'h1'), ('4h', 'h4')]:
            df_tf = htf_direct.get(tf_key)
            if df_tf is None or len(df_tf) < 22:
                continue

            closes = df_tf['close'].values.astype(float)
            highs = df_tf['high'].values.astype(float)
            lows = df_tf['low'].values.astype(float)

            sma20 = pd.Series(closes).rolling(20, min_periods=1).mean().values
            atr_vals = []
            for i in range(1, len(closes)):
                tr = max(highs[i] - lows[i],
                         abs(highs[i] - closes[i-1]),
                         abs(lows[i] - closes[i-1]))
                atr_vals.append(tr)
            atr_series = pd.Series([atr_vals[0]] + atr_vals).rolling(14, min_periods=1).mean().values

            idx = -2
            slope = (sma20[idx] - sma20[max(idx-3, 0)]) / (atr_series[idx] + 1e-9)
            trend_sign = 1 if slope > 0 else (-1 if slope < 0 else 0)

            result[f'{tf_label}_trend'] = trend_sign

            if tf_label == 'h1':
                result['h1_slope'] = float(slope)
                htf_high = highs[idx]
                htf_low = lows[idx]
                current_close = float(df_candles.iloc[-1]['close'])
                result['h1_range_pos'] = float(np.clip(
                    (current_close - htf_low) / (htf_high - htf_low + 1e-9), 0.0, 1.0
                ))

        h1_trend = result['h1_trend']
        h4_trend = result['h4_trend']
        result['trend_aligned'] = (h1_trend == h4_trend) and (h1_trend != 0)
        result['slope_ok'] = abs(result['h1_slope']) > 0.05

        h1_range_pos = result['h1_range_pos']
        result['range_ok'] = True
        if h1_trend > 0 and h1_range_pos < 0.2:
            result['range_ok'] = False
        if h1_trend < 0 and h1_range_pos > 0.8:
            result['range_ok'] = False

        if h1_trend > 0:
            result['side'] = 'LONG'
        elif h1_trend < 0:
            result['side'] = 'SHORT'
        else:
            result['side'] = 'NEUTRAL'

        log.info(f"  HTF gates (direct): aligned={result['trend_aligned']} "
                 f"h1={h1_trend:+d} h4={h4_trend:+d} slope={result['h1_slope']:.3f} "
                 f"range_pos={h1_range_pos:.2f}")

        return result

    def _process_symbol(self, symbol: str, df_candles: Optional[pd.DataFrame] = None) -> Optional[dict]:
        """Process one symbol: fetch data, compute features, run inference, apply gates."""
        if df_candles is None:
            df_candles = _fetch_candles_for_symbol(self.fetcher, symbol, self.interval, limit=self.limit_15m)
        if df_candles is None:
            return None

        df_candles = self._update_candle_cache(symbol, df_candles)

        htf_direct = None
        use_direct_htf = False

        if self.direct_htf:
            htf_direct = _fetch_htf_candles_direct(self.fetcher, symbol)
            if htf_direct:
                n_h1 = len(htf_direct.get('1h', []))
                n_h4 = len(htf_direct.get('4h', []))
                log.info(f"  {symbol} direct HTF: h1={n_h1} bars, h4={n_h4} bars")
                if n_h1 < MIN_H1_BARS or n_h4 < MIN_H4_BARS:
                    warmup_msg = (f"{symbol} WARMUP (direct HTF): h1_bars={n_h1} h4_bars={n_h4} "
                                  f"(min {MIN_H1_BARS}/{MIN_H4_BARS}) -> skip gates/trading")
                    if not self.warmup_logged.get(symbol):
                        log.warning(warmup_msg)
                        self.warmup_logged[symbol] = True
                    try:
                        current_price = float(df_candles.iloc[-1]['close'])
                        self._push_cycle_log(
                            symbol=symbol, price=current_price, p_enter=0.0,
                            htf={'h1_trend': 0, 'h4_trend': 0, 'slope_ok': False, 'range_ok': False},
                            direction="NEUTRAL", decision="WARMUP",
                            reasons=[warmup_msg],
                        )
                    except Exception:
                        pass
                    return None
                else:
                    self.warmup_logged[symbol] = False
                    use_direct_htf = True
            else:
                log.warning(f"{symbol}: direct HTF fetch failed, falling back to resampled warmup check")

        if not use_direct_htf:
            warmup_reason = _check_htf_warmup(df_candles, symbol)
            if warmup_reason:
                if not self.warmup_logged.get(symbol):
                    log.warning(warmup_reason)
                    self.warmup_logged[symbol] = True
                try:
                    current_price = float(df_candles.iloc[-1]['close'])
                    self._push_cycle_log(
                        symbol=symbol, price=current_price, p_enter=0.0,
                        htf={'h1_trend': 0, 'h4_trend': 0, 'slope_ok': False, 'range_ok': False},
                        direction="NEUTRAL", decision="WARMUP",
                        reasons=[warmup_reason],
                    )
                except Exception:
                    pass
                return None
            else:
                self.warmup_logged[symbol] = False

        model, engineer, feature_columns, temperature, symbol_map = self._get_model_for_symbol(symbol)

        scaled, features_df = _compute_features_for_symbol(
            df_candles, engineer, feature_columns, symbol
        )
        if scaled is None:
            return None

        sym_id = None
        if symbol_map and symbol in symbol_map:
            sym_id = symbol_map[symbol]

        infer_result = _run_inference(model, scaled, self.device,
                                      temperature=temperature, symbol_id=sym_id)
        p_enter = infer_result['p_enter']
        e_net_pred = infer_result['e_net_pred']
        enter_logit = infer_result['enter_logit']
        temperature_used = infer_result['temperature_used']

        if use_direct_htf and htf_direct:
            htf = self._compute_htf_from_direct(htf_direct, df_candles)
        else:
            htf = _apply_htf_gates(features_df)

        current_price = float(df_candles.iloc[-1]['close'])
        atr = _compute_atr(df_candles)

        if symbol not in self.p_enter_history:
            self.p_enter_history[symbol] = []
        self.p_enter_history[symbol].append(p_enter)
        if len(self.p_enter_history[symbol]) > 2000:
            self.p_enter_history[symbol] = self.p_enter_history[symbol][-2000:]

        side = htf.get('side', 'NEUTRAL')
        if side == 'NEUTRAL':
            dir_for_score = 'LONG' if htf.get('h1_trend', 0) >= 0 else 'SHORT'
        else:
            dir_for_score = side
        htf_score = _compute_htf_score(htf, dir_for_score)

        log.info(f"  {symbol}: price={current_price:.2f} p_enter={p_enter:.4f} e_net={e_net_pred:.4f} "
                 f"logit={enter_logit:.3f} T={temperature_used:.3f} "
                 f"side={side} htf_score={htf_score} "
                 f"slope_ok={htf['slope_ok']} range_ok={htf['range_ok']}")

        reasons = []
        decision = "HOLD"

        if self.cooldown_tracker.get(symbol, 0) > 0:
            bars_left = self.cooldown_tracker[symbol]
            self.cooldown_tracker[symbol] -= 1
            reasons.append(f"Cooldown active ({bars_left} bars left)")
            decision = "COOLDOWN"
            lane_info = {
                'lane_selected': 'HOLD', 'htf_score': htf_score,
                'hold_reason': f'COOLDOWN ({bars_left} bars)',
                'lane_size_mult': 0.0, 'threshold_used': 0.0,
                'core_thr': None, 'flow_thr': None, 'scalp_thr': None,
            }
            try:
                self._push_cycle_log(symbol=symbol, price=current_price, p_enter=p_enter,
                    htf=htf, direction=side, decision=decision, reasons=reasons,
                    lane_info=lane_info,
                    e_net_pred=e_net_pred, enter_logit=enter_logit,
                    temperature_used=temperature_used)
            except Exception as e:
                log.warning(f"Failed to push cycle log for {symbol}: {e}")
            return None

        self._reset_daily_budget_if_needed(symbol)
        lane_result = self._select_lane(
            symbol=symbol, p_enter=p_enter, htf=htf,
            htf_score=htf_score, features_df=features_df,
            df_candles=df_candles, atr=atr,
            e_net_pred=e_net_pred,
        )

        lane_selected = lane_result['lane_selected']

        if hasattr(self, 'verifier') and self.verifier:
            vf = self.verifier
            vf.stats.total_cycles += 1
            cycle_n = vf.stats.total_cycles
            fs = vf.verify_lane_routing(cycle_n, lane_result, p_enter, htf_score, htf)
            vf.add_failures(fs)
            payload_for_check = {
                'lane_selected': lane_result.get('lane_selected'),
                'htf_score': lane_result.get('htf_score'),
                'hold_reason': lane_result.get('hold_reason'),
                'quota_step': lane_result.get('quota_step'),
                'core_thr': lane_result.get('core_thr'),
                'flow_thr': lane_result.get('flow_thr'),
                'scalp_thr': lane_result.get('scalp_thr'),
                'lane_size_mult': lane_result.get('lane_size_mult'),
                'lane_budget_remaining_r': lane_result.get('lane_budget_remaining_r'),
            }
            fs2 = vf.verify_payload(cycle_n, payload_for_check)
            vf.add_failures(fs2)
            qs = lane_result.get('quota_step')
            if qs is not None:
                vf.record_quota_step(qs)
            if cycle_n >= vf.max_cycles:
                log.info(f"[VERIFY] Reached {vf.max_cycles} cycles -- stopping for report generation")
                self._should_stop = True

        if hasattr(self, 'separation_verifier') and self.separation_verifier:
            sv = self.separation_verifier
            scalp_gates = lane_result.get('scalp_gates', {})
            budgets = {
                'core_remaining': self._get_lane_budget_remaining(symbol, "CORE"),
                'flow_remaining': self._get_lane_budget_remaining(symbol, "FLOW"),
                'scalp_remaining': self._get_lane_budget_remaining(symbol, "SCALP"),
            }
            sv.verify_cycle(
                cycle=sv.stats.total_cycles,
                lane_result=lane_result,
                p_enter=p_enter,
                htf_score=htf_score,
                htf=htf,
                scalp_gates=scalp_gates,
                budgets=budgets,
            )
            if sv.stats.total_cycles >= sv.max_cycles:
                log.info(f"[VERIFY_SEPARATION] Reached {sv.max_cycles} cycles -- stopping for report generation")
                self._should_stop = True

        if lane_selected == 'HOLD':
            decision = "HOLD"
            hold_reason = lane_result.get('hold_reason', 'NO_LANE_MATCH')
            reasons.append(hold_reason)
            log.info(f"[LANE_DECISION] sym={symbol} lane=HOLD p={p_enter:.4f} "
                     f"htf_score={htf_score} core_thr={lane_result.get('core_thr','?')} "
                     f"flow_thr={lane_result.get('flow_thr','?')} scalp_thr={lane_result.get('scalp_thr','?')} "
                     f"reason={hold_reason}")

            sym_hist = self.p_enter_history.get(symbol, [])
            if len(sym_hist) >= 20:
                hist = np.array(sym_hist)
                p90 = float(np.percentile(hist, 90))
                p95 = float(np.percentile(hist, 95))
                p99 = float(np.percentile(hist, 99))
                log.info(f"    {symbol} p_enter percentiles (last {len(hist)}): "
                         f"p90={p90:.3f} p95={p95:.3f} p99={p99:.3f}")
                log.info(f"    thresholds: core={lane_result.get('core_thr','?')} "
                         f"flow={lane_result.get('flow_thr','?')} "
                         f"scalp={lane_result.get('scalp_thr','?')}")

            try:
                self._push_cycle_log(symbol=symbol, price=current_price, p_enter=p_enter,
                    htf=htf, direction=side, decision=decision, reasons=reasons,
                    lane_info=lane_result,
                    e_net_pred=e_net_pred, enter_logit=enter_logit,
                    temperature_used=temperature_used)
            except Exception as e:
                log.warning(f"Failed to push cycle log for {symbol}: {e}")
            return None

        decision = f"ENTER {lane_selected}"
        reasons.append(f"p_enter={p_enter:.1%} side={side} lane={lane_selected}")
        log.info(f"[LANE_DECISION] sym={symbol} lane={lane_selected} p={p_enter:.4f} "
                 f"htf_score={htf_score} core_thr={lane_result.get('core_thr','?')} "
                 f"flow_thr={lane_result.get('flow_thr','?')} scalp_thr={lane_result.get('scalp_thr','?')} "
                 f"reason=ENTER size_mult={lane_result['lane_size_mult']:.2f} "
                 f"horizon={lane_result['lane_horizon']}")

        try:
            self._push_cycle_log(symbol=symbol, price=current_price, p_enter=p_enter,
                htf=htf, direction=side, decision=decision, reasons=reasons,
                lane_info=lane_result,
                e_net_pred=e_net_pred, enter_logit=enter_logit,
                temperature_used=temperature_used)
        except Exception as e:
            log.warning(f"Failed to push cycle log for {symbol}: {e}")

        sl_pct = self.sl_mult * atr / current_price
        risk_pct = min(2.0 * sl_pct * 100, 5.0) * lane_result['lane_size_mult']

        return {
            'symbol': symbol,
            'side': side,
            'p_enter': p_enter,
            'current_price': current_price,
            'atr': atr,
            'htf': htf,
            'risk_pct': risk_pct,
            'df_candles': df_candles,
            'expected_net_r': (p_enter - 0.5) * self.tp_mult / self.sl_mult,
            'lane_info': lane_result,
            'features_df': features_df,
        }

    def _execute_candidate(self, candidate: dict):
        """Execute a trade candidate — with lane-aware geometry and budget spending.

        Gated by execution_mode:
        - signal_only: log the signal, push cycle log, do NOT create Position or POST trade
        - paper: create Position, POST trade, simulate exits
        - live: create Position, POST trade, place exchange orders
        """
        from portfolio import Position

        symbol = candidate['symbol']
        side = candidate['side']
        current_price = candidate['current_price']
        atr = candidate['atr']
        p_enter = candidate['p_enter']
        htf = candidate['htf']
        lane_info = candidate.get('lane_info', {})
        lane = lane_info.get('lane', 'CORE')
        size_mult = lane_info.get('lane_size_mult', 1.0)
        horizon = lane_info.get('lane_horizon', 24)
        htf_score = lane_info.get('htf_score', 0)

        if self.execution_mode == "signal_only" or not self.record_trades:
            if lane == "SCALP":
                sig_sl_dist = SCALP_SL_R * atr
                sig_tp_dist = SCALP_TP_R * atr
            else:
                sig_sl_dist = self.sl_mult * atr
                sig_tp_dist = self.tp_mult * atr
            if side == "LONG":
                sig_sl = current_price - sig_sl_dist
                sig_tp = current_price + sig_tp_dist
            else:
                sig_sl = current_price + sig_sl_dist
                sig_tp = current_price - sig_tp_dist
            no_exec_reason = "SIGNAL_ONLY" if self.execution_mode == "signal_only" else "RECORD_TRADES_OFF"
            log.info(f"  [NO_EXEC] {no_exec_reason} would_open_trade symbol={symbol} side={side} lane={lane} "
                     f"p={p_enter:.4f} entry={current_price:.2f} sl={sig_sl:.2f} tp={sig_tp:.2f} "
                     f"htf_score={htf_score} size_mult={size_mult:.2f} horizon={horizon}")
            try:
                self._push_cycle_log(
                    symbol=symbol, price=current_price, p_enter=p_enter,
                    htf=htf, direction=side, decision=no_exec_reason,
                    reasons=[
                        f"would_enter=true",
                        f"side={side} entry={current_price:.2f} sl={sig_sl:.2f} tp={sig_tp:.2f}",
                        f"lane={lane} p={p_enter:.4f} htf_score={htf_score}",
                    ],
                    lane_info=lane_info,
                )
            except Exception as e:
                log.warning(f"Failed to push {no_exec_reason} cycle log for {symbol}: {e}")
            return

        entry_price = current_price
        exec_result = None

        if self.execution:
            exec_candles = None
            if self.dry_run:
                exec_candles = candidate['df_candles'].tail(15).reset_index(drop=True)

            exec_result = self.execution.attempt_entry(
                symbol=symbol, side=side,
                signal_price=current_price, atr=atr, p_enter=p_enter,
                dry_run_candles=exec_candles,
            )

            if not exec_result.executed:
                log.info(f"  {symbol}: execution module skipped trade — {exec_result.reason}")
                return

            entry_price = exec_result.entry_price

        if lane == "SCALP":
            sl_dist = SCALP_SL_R * atr
            tp_dist = SCALP_TP_R * atr
        else:
            sl_dist = self.sl_mult * atr
            tp_dist = self.tp_mult * atr

        if side == "LONG":
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        prediction = _build_prediction_payload(
            symbol=symbol, side=side, p_enter=p_enter,
            current_price=current_price, atr=atr,
            entry_price=entry_price, tp_mult=self.tp_mult, sl_mult=self.sl_mult,
            htf=htf, exec_result=exec_result,
        )

        sl_pct = abs(entry_price - sl_price) / entry_price
        risk_pct = min(2.0 * sl_pct * 100, 5.0) * size_mult
        size_pct = risk_pct

        pos = Position(
            symbol=symbol, side=side,
            entry_price=entry_price, entry_time=time.time(),
            atr=atr, tp_price=tp_price, sl_price=sl_price,
            p_enter=p_enter, size_mult=size_mult, risk_pct=risk_pct,
            bar_index=self.cycle_count,
            lane=lane, horizon=horizon, htf_score=htf_score,
            threshold_used=lane_info.get('threshold_used', self.enter_threshold),
        )
        self.portfolio.open_position(pos)

        self._spend_lane_budget(symbol, lane, size_mult)

        cooldown = self.cooldown_bars
        if lane == "SCALP":
            cooldown = max(2, self.cooldown_bars // 2)
        self.cooldown_tracker[symbol] = cooldown

        try:
            trade_id = self._push_trade_record(
                symbol=symbol, side=side, entry_price=entry_price,
                sl_price=sl_price, tp_price=tp_price,
                p_enter=p_enter, size_pct=size_pct, lane_info={
                    'lane': lane, 'htf_score': htf_score,
                    'threshold_used': lane_info.get('threshold_used'),
                    'lane_size_mult': size_mult,
                    'lane_horizon': horizon,
                },
            )
            if trade_id:
                pos.dashboard_trade_id = trade_id
        except Exception as e:
            log.warning(f"Failed to push trade record for {symbol}: {e}")

        if self.execution_mode == "paper":
            log.info(f"  [PAPER_OPEN] {lane} {symbol} {side} @ {entry_price:.2f} "
                     f"| size_mult={size_mult:.2f} horizon={horizon}")
        elif self.execution_mode == "live":
            log.info(f"  [LIVE_OPEN] placing_order {lane} {symbol} {side} @ {entry_price:.2f} "
                     f"| size_mult={size_mult:.2f} horizon={horizon}")
        self._push_prediction(prediction)

    def _print_summary(self):
        summary = self.portfolio.summary()
        log.info("")
        log.info("=" * 60)
        log.info("  SESSION SUMMARY")
        log.info("=" * 60)
        log.info(f"  Cycles: {self.cycle_count}")
        log.info(f"  Trades: {summary['total_trades']} (W:{summary['wins']} L:{summary['losses']})")
        log.info(f"  Win rate: {summary['win_rate']:.1%}")
        log.info(f"  Avg R: {summary['avg_r']:+.2f}")
        log.info(f"  Open: {summary['open_positions']} | Risk: {summary['total_risk_pct']:.1f}%")
        log.info("=" * 60)
