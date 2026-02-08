"""Multi-asset live inference loop.

Monitors multiple symbols in parallel on 15m intervals, runs the ENTER QUALITY
model inference, applies HTF gates, ranks candidates, manages portfolio, and
optionally improves entries via lower-timeframe execution.

Usage:
    python quick_start.py --live --paper --symbols BTCUSDT,ETHUSDT,SOLUSDT
"""

import sys
import time
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("LiveRunner")


REQUIRED_CANDLES = 300


def _load_model(device: str):
    """Load the trained ENTER QUALITY model, scaler, and feature columns."""
    import torch
    from models.simple_mlp import EnhancedMultiHeadMLP, EnhancedMultiHeadMLP_Config
    from data.pipeline import FeatureEngineer

    from quick_start import FEATURE_VERSION

    checkpoint_path = Path("checkpoints/best_enter_prauc.pt")
    if not checkpoint_path.exists():
        checkpoint_path = Path("checkpoints/best_enter_loss.pt")
    if not checkpoint_path.exists():
        log.error("No trained ENTER model found! Run training first.")
        sys.exit(1)

    log.info(f"Loading model from {checkpoint_path}...")
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

    engineer = FeatureEngineer()
    scaler_path = Path("checkpoints/scaler.joblib")
    if scaler_path.exists():
        engineer.load_scalers(str(scaler_path))
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

        self.model = None
        self.engineer = None
        self.feature_columns = None
        self.fetcher = None
        self.cycle_count = 0

    def _init_fetcher(self):
        from data.pipeline import BinanceDataFetcher
        self.fetcher = BinanceDataFetcher(
            symbols=self.symbols,
            timeframes=[self.interval],
            replit_proxy_url=f"{self.replit_url.rstrip('/')}/api/binance-proxy",
            use_sync=True,
        )
        if self.execution and self.execution.fetcher is None:
            self.execution.fetcher = self.fetcher

    def _push_prediction(self, prediction: dict):
        from quick_start import push_prediction
        push_prediction(self.replit_url, prediction)

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
        log.info(f"  LIVE RUNNER {'(PAPER)' if self.paper else '(LIVE)'}")
        log.info(f"  Symbols: {', '.join(self.symbols)}")
        log.info(f"  Interval: {self.interval} | Threshold: {self.enter_threshold}")
        log.info(f"  TP={self.tp_mult}x SL={self.sl_mult}x | Cooldown: {self.cooldown_bars} bars")
        if self.dry_run:
            log.info(f"  DRY RUN MODE — replaying cached candles")
        log.info("=" * 80)

        self.model, self.engineer, self.feature_columns = _load_model(self.device)
        self._init_fetcher()

        if self.dry_run:
            self._run_dry()
            return

        try:
            while True:
                self._run_cycle()
                interval_s = self._interval_seconds()
                now = time.time()
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

        scaled, features_df = _compute_features_for_symbol(
            df_candles, self.engineer, self.feature_columns, symbol
        )
        if scaled is None:
            return None

        p_enter = _run_inference(self.model, scaled, self.device)
        htf = _apply_htf_gates(features_df)

        current_price = float(df_candles.iloc[-1]['close'])
        atr = _compute_atr(df_candles)

        log.info(f"  {symbol}: price={current_price:.2f} p_enter={p_enter:.4f} "
                 f"side={htf['side']} aligned={htf['trend_aligned']} "
                 f"slope_ok={htf['slope_ok']} range_ok={htf['range_ok']}")

        passes_gates = htf['trend_aligned'] and htf['slope_ok'] and htf['range_ok']
        passes_threshold = p_enter >= self.enter_threshold
        side = htf['side']

        if not passes_gates:
            log.info(f"  {symbol}: HTF gates FAIL — skipping")
            return None
        if not passes_threshold:
            log.info(f"  {symbol}: p_enter {p_enter:.4f} < threshold {self.enter_threshold} — skipping")
            return None
        if side == "NEUTRAL":
            log.info(f"  {symbol}: side NEUTRAL — skipping")
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

        pos = Position(
            symbol=symbol, side=side,
            entry_price=entry_price, entry_time=time.time(),
            atr=atr, tp_price=tp_price, sl_price=sl_price,
            p_enter=p_enter, size_mult=1.0, risk_pct=risk_pct,
            bar_index=self.cycle_count,
        )
        self.portfolio.open_position(pos)

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
