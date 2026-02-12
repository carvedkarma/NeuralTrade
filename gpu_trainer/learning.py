"""Scheduled retraining + safe promotion system (v3.5.0).

Manages the continuous learning lifecycle:
  1. Scheduled daily retrain (trains candidate model on latest data)
  2. Walk-forward evaluation of candidate vs deployed model
  3. Safe promotion: only promote if candidate beats deployed on key metrics
  4. Push learning stats to dashboard after each cycle

Usage (called from quick_start.py via LiveRunner):
    runner = LiveRunner(..., learning_config=LearningConfig(...))

Key safety gates for promotion:
  - PF_net (profit factor after costs) must improve or stay above minimum
  - Profitable regime count must not decrease
  - Trades-per-day must stay within acceptable range
  - Model must pass minimum PR-AUC threshold
"""

import os
import sys
import time
import json
import shutil
import logging
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List

log = logging.getLogger("Learning")


@dataclass
class LearningConfig:
    retrain_hour_utc: int = 4
    retrain_interval_hours: int = 24
    min_new_bars: int = 96
    training_epochs: int = 300
    min_prauc_threshold: float = 0.35
    min_pf_net: float = 1.05
    min_profitable_regimes: int = 3
    min_tpd: float = 0.3
    max_tpd: float = 5.0
    pf_improvement_required: float = 0.0
    regime_loss_tolerance: int = 0
    auto_promote: bool = True
    geometry_sweep_on_retrain: bool = True
    sweep_thresholds: List[float] = field(default_factory=lambda: [0.60, 0.65, 0.70, 0.75, 0.80, 0.85])
    sweep_cooldowns: List[int] = field(default_factory=lambda: [2, 4, 6, 8])
    gate_pf_net: float = 1.05
    gate_enet: float = 0.0
    gate_profitable_regimes: int = 2
    gate_maxdd_r: float = 6.0
    gate_p95_min: float = 0.40
    gate_p95_max: float = 0.98


@dataclass
class RetrainResult:
    success: bool = False
    model_version: str = ""
    val_prauc: float = 0.0
    val_precision: float = 0.0
    val_recall: float = 0.0
    val_f1: float = 0.0
    training_samples: int = 0
    trained_until_ts: int = 0
    best_policy: Optional[Dict] = None
    pf_net: float = 0.0
    e_net: float = 0.0
    trades_per_day: float = 0.0
    profitable_regimes: int = 0
    total_regimes: int = 0
    error: str = ""
    max_drawdown_r: float = 0.0
    p95_val: float = 0.0
    temperature: float = 1.0


class LearningManager:
    """Manages scheduled retraining and safe model promotion."""

    def __init__(
        self,
        replit_url: str,
        device: str,
        symbols: List[str],
        config: LearningConfig = None,
    ):
        self.replit_url = replit_url
        self.device = device
        self.symbols = symbols
        self.config = config or LearningConfig()
        self.last_retrain_time: Dict[str, float] = {}
        self.deployed_stats: Dict[str, Dict] = {}

    def should_retrain(self, symbol: str) -> bool:
        now = datetime.now(timezone.utc)
        last = self.last_retrain_time.get(symbol, 0)
        if last == 0:
            if now.hour == self.config.retrain_hour_utc:
                return True
            return False

        hours_since = (time.time() - last) / 3600
        if hours_since < self.config.retrain_interval_hours:
            return False
        if now.hour == self.config.retrain_hour_utc:
            return True
        return False

    def retrain_symbol(self, symbol: str) -> RetrainResult:
        result = RetrainResult()
        log.info(f"[Learning] Starting retrain for {symbol}...")

        try:
            candidate_dir = Path(f"checkpoints/candidate/{symbol}")
            candidate_dir.mkdir(parents=True, exist_ok=True)

            deployed_dir = Path(f"checkpoints/deployed/{symbol}")
            deployed_dir.mkdir(parents=True, exist_ok=True)

            result = self._run_training(symbol, candidate_dir)
            if not result.success:
                log.error(f"[Learning] Training failed for {symbol}: {result.error}")
                return result

            if self.config.geometry_sweep_on_retrain:
                sweep_result = self._run_geometry_sweep(symbol, candidate_dir)
                if sweep_result:
                    result.best_policy = sweep_result.get('best_policy')
                    result.pf_net = sweep_result.get('pf_net', 0)
                    result.e_net = sweep_result.get('e_net', 0)
                    result.trades_per_day = sweep_result.get('trades_per_day', 0)
                    result.profitable_regimes = sweep_result.get('profitable_regimes', 0)
                    result.total_regimes = sweep_result.get('total_regimes', 0)

            should_promote, reason = self._evaluate_promotion(symbol, result)

            if should_promote and self.config.auto_promote:
                self._promote_model(symbol, candidate_dir, deployed_dir)
                log.info(f"[Learning] Model PROMOTED for {symbol}: {reason}")
            elif not should_promote:
                log.info(f"[Learning] Model NOT promoted for {symbol}: {reason}")

            self._push_learning_stats(symbol, result, should_promote, reason)

            self.last_retrain_time[symbol] = time.time()
            return result

        except Exception as e:
            log.error(f"[Learning] Retrain failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            result.error = str(e)
            return result

    def _run_training(self, symbol: str, output_dir: Path) -> RetrainResult:
        result = RetrainResult()

        try:
            from quick_start import FEATURE_VERSION
            from data.pipeline import FeatureEngineer, BinanceDataFetcher
            from training.train_loop import train_enter_model

            log.info(f"[Learning] Fetching data for {symbol}...")
            fetcher = BinanceDataFetcher(
                symbols=[symbol],
                timeframes=["15m"],
                replit_proxy_url=self.replit_url.rstrip('/'),
                use_sync=True,
            )

            raw = fetcher.fetch_klines_sync(symbol, "15m", limit=5000)
            if not raw or len(raw) < 500:
                result.error = f"Insufficient data: got {len(raw) if raw else 0} candles"
                return result

            import pandas as pd
            df = pd.DataFrame(raw)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            if 'timestamp' in df.columns:
                df['timestamp'] = df['timestamp'].astype(int)
            df = df.sort_values('timestamp').reset_index(drop=True)

            log.info(f"[Learning] Training with {len(df)} candles for {symbol}...")
            train_result = train_enter_model(
                df=df,
                epochs=self.config.training_epochs,
                device=self.device,
                checkpoint_dir=str(output_dir),
                symbol=symbol,
            )

            result.success = True
            result.model_version = FEATURE_VERSION
            result.val_prauc = train_result.get('val_prauc', 0)
            result.val_precision = train_result.get('val_precision', 0)
            result.val_recall = train_result.get('val_recall', 0)
            result.val_f1 = train_result.get('val_f1', 0)
            result.training_samples = train_result.get('training_samples', len(df))
            result.trained_until_ts = int(df.iloc[-1].get('timestamp', time.time() * 1000))

        except ImportError as e:
            result.error = f"Missing training module: {e}"
        except Exception as e:
            result.error = str(e)

        return result

    def _run_geometry_sweep(self, symbol: str, checkpoint_dir: Path) -> Optional[Dict]:
        try:
            from training.triple_barrier import run_geometry_sweep

            checkpoint_path = checkpoint_dir / "best_enter_prauc.pt"
            if not checkpoint_path.exists():
                checkpoint_path = checkpoint_dir / "best_enter_loss.pt"
            if not checkpoint_path.exists():
                log.warning(f"[Learning] No checkpoint for geometry sweep: {symbol}")
                return None

            results = run_geometry_sweep(
                checkpoint_path=str(checkpoint_path),
                thresholds=self.config.sweep_thresholds,
                cooldowns=self.config.sweep_cooldowns,
                device=self.device,
            )

            if not results:
                return None

            best = max(results, key=lambda r: r.get('pf_net', 0))
            profitable = sum(1 for r in results if r.get('pf_net', 0) > 1.0)

            return {
                'best_policy': {
                    'threshold': best.get('threshold', 0.7),
                    'cooldown': best.get('cooldown', 4),
                    'tp_mult': best.get('tp_mult', 3.5),
                    'sl_mult': best.get('sl_mult', 1.5),
                },
                'pf_net': best.get('pf_net', 0),
                'e_net': best.get('e_net', 0),
                'trades_per_day': best.get('trades_per_day', 0),
                'profitable_regimes': profitable,
                'total_regimes': len(results),
            }

        except ImportError:
            log.warning("[Learning] Geometry sweep module not available")
            return None
        except Exception as e:
            log.warning(f"[Learning] Geometry sweep failed for {symbol}: {e}")
            return None

    def _evaluate_promotion(self, symbol: str, result: RetrainResult) -> tuple:
        """Evaluate whether candidate model should be promoted.
        
        Three-stage gating (v5.0):
        1. Absolute thresholds: candidate must meet minimum quality bars
        2. v5.0 calibration gates: p95 sanity, E[net], max drawdown
        3. Relative comparison: if deployed stats exist, candidate must not regress
        
        When no deployed stats exist (first training), only absolute thresholds apply.
        """
        if result.val_prauc < self.config.min_prauc_threshold:
            return False, f"PR-AUC {result.val_prauc:.3f} < min {self.config.min_prauc_threshold}"

        if result.pf_net < self.config.gate_pf_net:
            return False, f"PF_net {result.pf_net:.2f} < gate {self.config.gate_pf_net}"

        if result.profitable_regimes < self.config.gate_profitable_regimes:
            return False, f"Profitable regimes {result.profitable_regimes} < gate {self.config.gate_profitable_regimes}"

        if result.trades_per_day < self.config.min_tpd:
            return False, f"TPD {result.trades_per_day:.1f} < min {self.config.min_tpd}"

        if result.trades_per_day > self.config.max_tpd:
            return False, f"TPD {result.trades_per_day:.1f} > max {self.config.max_tpd}"

        if result.e_net < self.config.gate_enet:
            return False, f"E[net] {result.e_net:.4f} < gate {self.config.gate_enet}"

        if result.max_drawdown_r > self.config.gate_maxdd_r:
            return False, f"MaxDD {result.max_drawdown_r:.2f}R > gate {self.config.gate_maxdd_r}R"

        if result.p95_val > 0:
            if result.p95_val < self.config.gate_p95_min:
                return False, f"p95={result.p95_val:.4f} < gate_p95_min={self.config.gate_p95_min} (collapsed predictions)"
            if result.p95_val > self.config.gate_p95_max:
                return False, f"p95={result.p95_val:.4f} > gate_p95_max={self.config.gate_p95_max} (overconfident predictions)"

        prev = self.deployed_stats.get(symbol)
        if prev and prev.get('pf_net', 0) > 0:
            prev_pf = prev.get('pf_net', 0)
            if result.pf_net < prev_pf - self.config.pf_improvement_required:
                return False, f"PF_net {result.pf_net:.2f} < deployed {prev_pf:.2f}"

            prev_regimes = prev.get('profitable_regimes', 0)
            if result.profitable_regimes < prev_regimes - self.config.regime_loss_tolerance:
                return False, f"Profitable regimes dropped: {result.profitable_regimes} < {prev_regimes}"
        elif not prev:
            log.info(f"[Learning] No deployed stats for {symbol} — first promotion uses absolute thresholds only")

        return True, (f"PF={result.pf_net:.2f} PR-AUC={result.val_prauc:.3f} "
                      f"E[net]={result.e_net:.4f} p95={result.p95_val:.4f} "
                      f"maxDD={result.max_drawdown_r:.2f}R "
                      f"regimes={result.profitable_regimes}/{result.total_regimes}")

    def _promote_model(self, symbol: str, candidate_dir: Path, deployed_dir: Path):
        for filename in ["best_enter_prauc.pt", "best_enter_loss.pt", "scaler.joblib", "temp_scale_v5.0.json"]:
            src = candidate_dir / filename
            if src.exists():
                dst = deployed_dir / filename
                shutil.copy2(str(src), str(dst))
                log.info(f"[Learning] Promoted {filename} for {symbol}")

        self.deployed_stats[symbol] = {
            'pf_net': 0,
            'e_net': 0,
            'profitable_regimes': 0,
        }

    def _push_learning_stats(self, symbol: str, result: RetrainResult,
                             promoted: bool, reason: str):
        url = f"{self.replit_url.rstrip('/')}/api/live/learning-stats"

        prev = self.deployed_stats.get(symbol)
        trend = "Flat"
        if prev:
            if result.pf_net > prev.get('pf_net', 0) * 1.05:
                trend = "Improving"
            elif result.pf_net < prev.get('pf_net', 0) * 0.95:
                trend = "Worse"

        payload = {
            "symbol": symbol,
            "model_version": result.model_version,
            "trained_until_ts": result.trained_until_ts,
            "training_samples": result.training_samples,
            "val_pr_auc": result.val_prauc,
            "val_precision": result.val_precision,
            "val_recall": result.val_recall,
            "val_f1": result.val_f1,
            "best_policy_threshold": result.best_policy.get('threshold') if result.best_policy else None,
            "best_policy_cooldown": result.best_policy.get('cooldown') if result.best_policy else None,
            "best_policy_tp_mult": result.best_policy.get('tp_mult') if result.best_policy else None,
            "best_policy_sl_mult": result.best_policy.get('sl_mult') if result.best_policy else None,
            "pf_net": result.pf_net,
            "e_net": result.e_net,
            "trades_per_day": result.trades_per_day,
            "profitable_regimes": result.profitable_regimes,
            "total_regimes": result.total_regimes,
            "promoted": promoted,
            "promotion_reason": reason,
            "trend_7d": trend,
            "prev_pf_net": prev.get('pf_net') if prev else None,
            "prev_e_net": prev.get('e_net') if prev else None,
            "prev_trades_per_day": prev.get('trades_per_day') if prev else None,
        }

        try:
            from live_runner import _retry_request
            _retry_request("POST", url, json=payload)
        except Exception as e:
            log.warning(f"[Learning] Failed to push stats for {symbol}: {e}")

        if promoted:
            self.deployed_stats[symbol] = {
                'pf_net': result.pf_net,
                'e_net': result.e_net,
                'profitable_regimes': result.profitable_regimes,
                'trades_per_day': result.trades_per_day,
            }

    def check_and_retrain_all(self):
        for symbol in self.symbols:
            if self.should_retrain(symbol):
                log.info(f"[Learning] Retrain triggered for {symbol}")
                self.retrain_symbol(symbol)
