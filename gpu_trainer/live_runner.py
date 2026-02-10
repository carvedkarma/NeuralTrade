"""Multi-asset live inference loop (v3.5.0).

Monitors multiple symbols in parallel on 15m intervals, runs the ENTER QUALITY
model inference, applies HTF gates, ranks candidates, manages portfolio, and
optionally improves entries via lower-timeframe execution.

v3.5.0 additions:
  - Per-symbol data caching with append/dedupe
  - Dashboard cycle-log + trade-record push
  - Exchange time sync (Binance serverTime)
  - Robust retry logic for all HTTP calls
  - Per-symbol model management (deployed/{symbol}/)

Usage:
    python quick_start.py --live --paper --symbols BTCUSDT,ETHUSDT,SOLUSDT
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


REQUIRED_CANDLES = 300
MAX_CACHE_BARS = 2000
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2.0


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
    """Load the trained ENTER QUALITY model, scaler, and feature columns.

    If symbol is provided, first checks checkpoints/deployed/{symbol}/ for a
    per-symbol model. Falls back to the global checkpoints/ directory.
    """
    import torch
    from models.simple_mlp import EnhancedMultiHeadMLP, EnhancedMultiHeadMLP_Config
    from data.pipeline import FeatureEngineer

    from quick_start import FEATURE_VERSION

    search_dirs = []
    if symbol:
        search_dirs.append(Path(f"checkpoints/deployed/{symbol}"))
    search_dirs.append(Path("checkpoints"))

    checkpoint_path = None
    for d in search_dirs:
        for name in ["best_enter_prauc.pt", "best_enter_loss.pt"]:
            p = d / name
            if p.exists():
                checkpoint_path = p
                break
        if checkpoint_path:
            break

    if not checkpoint_path:
        log.error(f"No trained ENTER model found{' for '+symbol if symbol else ''}! Run training first.")
        sys.exit(1)

    log.info(f"Loading model from {checkpoint_path}{' ('+symbol+')' if symbol else ''}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    saved_version = checkpoint.get('feature_version', 'unknown')
    if saved_version != FEATURE_VERSION:
        log.error(f"Feature version mismatch! Model: '{saved_version}', current: '{FEATURE_VERSION}'")
        sys.exit(1)

    feature_columns = checkpoint.get('feature_columns', [])
    if not feature_columns:
        log.error("No feature_columns in checkpoint — retrain.")
        sys.exit(1)

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
    model.eval()

    scaler_path = None
    for d in search_dirs:
        sp = d / "scaler.joblib"
        if sp.exists():
            scaler_path = sp
            break

    engineer = FeatureEngineer()
    if scaler_path:
        engineer.load_scalers(str(scaler_path))
        log.info(f"Scaler loaded from {scaler_path}")
    else:
        log.warning("No saved scaler — prediction quality may be reduced")

    log.info(f"Model loaded: {len(feature_columns)} features, version {saved_version}")
    return model, engineer, feature_columns


def _fetch_candles_for_symbol(fetcher, symbol: str, timeframe: str = "15m",
                               limit: int = REQUIRED_CANDLES) -> Optional[pd.DataFrame]:
    """Fetch recent candles for a symbol via the BinanceDataFetcher."""
    try:
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


def _run_inference(model, scaled_features: np.ndarray, device: str) -> float:
    """Run single-row ENTER model inference, returning p_enter."""
    import torch
    with torch.no_grad():
        x = torch.FloatTensor(scaled_features).to(device)
        output = model.forward_multihead(x)
    p_enter = float(torch.sigmoid(output.enter_logits).cpu().item())
    return p_enter


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
        paper: bool = True,
        portfolio_manager=None,
        execution_module=None,
        dry_run: bool = False,
        dry_run_candles: int = 200,
        per_symbol_models: bool = False,
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
        self.portfolio = portfolio_manager
        self.execution = execution_module
        self.dry_run = dry_run
        self.dry_run_candles = dry_run_candles
        self.per_symbol_models = per_symbol_models

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

    def _push_prediction(self, prediction: dict):
        from quick_start import push_prediction
        push_prediction(self.replit_url, prediction)

    def _push_cycle_log(self, symbol: str, price: float, p_enter: float,
                        htf: dict, direction: str, decision: str, reasons: list):
        url = f"{self.replit_url.rstrip('/')}/api/live/cycle-log"
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
            "threshold_used": float(self.enter_threshold),
            "decision": str(decision),
            "reasons": [str(r) for r in reasons] if reasons else [],
        }
        _retry_request("POST", url, json=payload)

    def _push_trade_record(self, symbol: str, side: str, entry_price: float,
                           sl_price: float, tp_price: float, p_enter: float,
                           size_pct: float) -> Optional[int]:
        url = f"{self.replit_url.rstrip('/')}/api/live/trade"
        payload = {
            "symbol": symbol,
            "side": side,
            "entry_time": int(time.time() * 1000),
            "entry_price": entry_price,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "p_enter": p_enter,
            "size_pct": size_pct,
            "status": "open",
        }
        resp = _retry_request("POST", url, json=payload)
        if resp and resp.status_code == 200:
            data = resp.json()
            return data.get("id")
        return None

    def _update_trade_record(self, trade_id: int, exit_price: float,
                             outcome: str, gross_r: float, net_r: float, sized_r: float):
        url = f"{self.replit_url.rstrip('/')}/api/live/trade/{trade_id}"
        payload = {
            "exit_time": int(time.time() * 1000),
            "exit_price": exit_price,
            "outcome": outcome,
            "gross_r": gross_r,
            "net_r": net_r,
            "sized_r": sized_r,
            "status": "closed",
        }
        _retry_request("PATCH", url, json=payload)

    def _get_model_for_symbol(self, symbol: str):
        if self.per_symbol_models:
            if symbol not in self.symbol_models:
                try:
                    m, e, fc = _load_model(self.device, symbol=symbol)
                    self.symbol_models[symbol] = (m, e, fc)
                except SystemExit:
                    log.warning(f"No per-symbol model for {symbol}, using global model")
                    self.symbol_models[symbol] = (self.model, self.engineer, self.feature_columns)
            return self.symbol_models[symbol]
        return self.model, self.engineer, self.feature_columns

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
        log.info("=" * 80)
        log.info(f"  LIVE RUNNER v3.5.0 {'(PAPER)' if self.paper else '(LIVE)'}")
        log.info(f"  Symbols: {', '.join(self.symbols)}")
        log.info(f"  Interval: {self.interval} | Threshold: {self.enter_threshold}")
        log.info(f"  TP={self.tp_mult}x SL={self.sl_mult}x | Cooldown: {self.cooldown_bars} bars")
        log.info(f"  Per-symbol models: {self.per_symbol_models}")
        if self.dry_run:
            log.info(f"  DRY RUN MODE — replaying cached candles")
        log.info("=" * 80)

        self.exchange_time_offset = _get_exchange_time_offset()
        log.info(f"Exchange time offset: {self.exchange_time_offset*1000:.0f}ms")

        self.model, self.engineer, self.feature_columns = _load_model(self.device)
        self._init_fetcher()

        for sym in self.symbols:
            self.cooldown_tracker[sym] = 0

        if self.dry_run:
            self._run_dry()
            return

        try:
            while True:
                self._run_cycle()

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
        for symbol in self.symbols:
            df = _fetch_candles_for_symbol(self.fetcher, symbol, self.interval)
            if df is not None and len(df) > 0:
                prices[symbol] = float(df.iloc[-1]['close'])
        self.portfolio.check_exits(prices)

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

    def _process_symbol(self, symbol: str, df_candles: Optional[pd.DataFrame] = None) -> Optional[dict]:
        """Process one symbol: fetch data, compute features, run inference, apply gates."""
        if df_candles is None:
            df_candles = _fetch_candles_for_symbol(self.fetcher, symbol, self.interval)
        if df_candles is None:
            return None

        df_candles = self._update_candle_cache(symbol, df_candles)

        model, engineer, feature_columns = self._get_model_for_symbol(symbol)

        scaled, features_df = _compute_features_for_symbol(
            df_candles, engineer, feature_columns, symbol
        )
        if scaled is None:
            return None

        p_enter = _run_inference(model, scaled, self.device)
        htf = _apply_htf_gates(features_df)

        current_price = float(df_candles.iloc[-1]['close'])
        atr = _compute_atr(df_candles)

        if symbol not in self.p_enter_history:
            self.p_enter_history[symbol] = []
        self.p_enter_history[symbol].append(p_enter)
        if len(self.p_enter_history[symbol]) > 500:
            self.p_enter_history[symbol] = self.p_enter_history[symbol][-500:]

        log.info(f"  {symbol}: price={current_price:.2f} p_enter={p_enter:.4f} "
                 f"side={htf['side']} aligned={htf['trend_aligned']} "
                 f"slope_ok={htf['slope_ok']} range_ok={htf['range_ok']}")

        passes_gates = htf['trend_aligned'] and htf['slope_ok'] and htf['range_ok']
        passes_threshold = p_enter >= self.enter_threshold
        side = htf['side']

        reasons = []
        decision = "HOLD"

        if self.cooldown_tracker.get(symbol, 0) > 0:
            bars_left = self.cooldown_tracker[symbol]
            self.cooldown_tracker[symbol] -= 1
            reasons.append(f"Cooldown active ({bars_left} bars left)")
            decision = "COOLDOWN"
        elif not passes_gates:
            if not htf['trend_aligned']:
                reasons.append("HTF trend not aligned")
            if not htf['slope_ok']:
                reasons.append("Slope too flat")
            if not htf['range_ok']:
                reasons.append("Range position unfavorable")
            decision = "GATE_FAIL"
        elif not passes_threshold:
            reasons.append(f"p_enter {p_enter:.4f} < {self.enter_threshold}")
            decision = "BELOW_THRESHOLD"
        elif side == "NEUTRAL":
            reasons.append("Side is NEUTRAL")
            decision = "NEUTRAL"
        else:
            decision = "ENTER"
            reasons.append(f"p_enter={p_enter:.1%} side={side}")

        try:
            self._push_cycle_log(
                symbol=symbol, price=current_price, p_enter=p_enter,
                htf=htf, direction=side, decision=decision, reasons=reasons,
            )
        except Exception as e:
            log.warning(f"Failed to push cycle log for {symbol}: {e}")

        if decision != "ENTER":
            if decision != "COOLDOWN":
                log.info(f"  {symbol}: {decision} — {'; '.join(reasons)}")
                sym_hist = self.p_enter_history.get(symbol, [])
                if len(sym_hist) >= 10:
                    hist = np.array(sym_hist)
                    p50 = float(np.percentile(hist, 50))
                    p75 = float(np.percentile(hist, 75))
                    p90 = float(np.percentile(hist, 90))
                    p95 = float(np.percentile(hist, 95))
                    p99 = float(np.percentile(hist, 99))
                    log.info(f"    {symbol} p_enter percentiles (last {len(hist)}): "
                             f"p50={p50:.3f} p75={p75:.3f} p90={p90:.3f} p95={p95:.3f} p99={p99:.3f}")
                    log.info(f"    threshold={self.enter_threshold:.4f} | "
                             f"HTF: aligned={htf['trend_aligned']} slope_ok={htf['slope_ok']} range_ok={htf['range_ok']} | "
                             f"side={side}")
            return None

        sl_pct = self.sl_mult * atr / current_price
        risk_pct = min(2.0 * sl_pct * 100, 5.0)

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
        }

    def _execute_candidate(self, candidate: dict):
        """Execute a trade candidate — with optional lower-TF execution improvement."""
        from portfolio import Position

        symbol = candidate['symbol']
        side = candidate['side']
        current_price = candidate['current_price']
        atr = candidate['atr']
        p_enter = candidate['p_enter']
        htf = candidate['htf']

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

        prediction = _build_prediction_payload(
            symbol=symbol, side=side, p_enter=p_enter,
            current_price=current_price, atr=atr,
            entry_price=entry_price, tp_mult=self.tp_mult, sl_mult=self.sl_mult,
            htf=htf, exec_result=exec_result,
        )

        if side == "LONG":
            sl_price = entry_price - self.sl_mult * atr
            tp_price = entry_price + self.tp_mult * atr
        else:
            sl_price = entry_price + self.sl_mult * atr
            tp_price = entry_price - self.tp_mult * atr

        sl_pct = abs(entry_price - sl_price) / entry_price
        risk_pct = min(2.0 * sl_pct * 100, 5.0)
        size_pct = min(2.0 * sl_pct * 100, 5.0)

        pos = Position(
            symbol=symbol, side=side,
            entry_price=entry_price, entry_time=time.time(),
            atr=atr, tp_price=tp_price, sl_price=sl_price,
            p_enter=p_enter, size_mult=1.0, risk_pct=risk_pct,
            bar_index=self.cycle_count,
        )
        self.portfolio.open_position(pos)

        self.cooldown_tracker[symbol] = self.cooldown_bars

        try:
            trade_id = self._push_trade_record(
                symbol=symbol, side=side, entry_price=entry_price,
                sl_price=sl_price, tp_price=tp_price,
                p_enter=p_enter, size_pct=size_pct,
            )
            if trade_id:
                pos.dashboard_trade_id = trade_id
        except Exception as e:
            log.warning(f"Failed to push trade record for {symbol}: {e}")

        if self.paper:
            log.info(f"  [PAPER] {symbol} {side} @ {entry_price:.2f}")
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
