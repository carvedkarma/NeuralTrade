"""v5.0.8+ Adaptive Position Sizing & Dynamic Risk Scaling.

Provides:
  - AdaptivePositionSizer: Kelly-inspired sizing from model outputs (mu_R, p_trade, mfe/mae)
  - RegimeScaler:          ATR-ratio + EMA-trend + rolling-equity regime detection → risk multiplier
  - DailyLossTracker:      Per-day and per-symbol daily R budget enforcement
  - TrailingEquityStop:    Pauses trading when equity drops too far from peak
  - SizingDiagnostics:     Collects stats for fold-level reporting
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class AdaptiveSizingConfig:
    enabled: bool = False
    kelly_fraction: float = 0.25
    max_size_mult: float = 2.5
    min_size_mult: float = 0.25


@dataclass
class RegimeScalingConfig:
    enabled: bool = False
    bull_mult: float = 1.5
    bear_mult: float = 0.5
    lookback_trades: int = 20
    atr_lookback: int = 96
    atr_bull_ratio: float = 0.8
    atr_bear_ratio: float = 1.5
    min_equity_trades: int = 15
    low_confidence_dampen: float = 0.5


@dataclass
class LossManagementConfig:
    daily_loss_cap: Optional[float] = None
    trailing_equity_stop: Optional[float] = None
    per_symbol_daily_r_budget: Optional[float] = None


class AdaptivePositionSizer:
    """Kelly-inspired position sizing from model predictions.

    Uses fractional Kelly criterion:
      f* = kelly_fraction * (p * b - q) / b
    where:
      p = estimated win probability (from action head)
      q = 1 - p
      b = estimated reward/risk ratio (mfe/mae)

    The result is clamped to [min_size_mult, max_size_mult].
    """

    def __init__(self, config: AdaptiveSizingConfig):
        self.config = config
        self.sizing_history: List[float] = []

    def compute_size_multiplier(self, score: float, p_win: float,
                                 mu_r: float, mfe: float, mae: float) -> float:
        if not self.config.enabled:
            return 1.0

        p = max(min(p_win, 0.99), 0.01)
        q = 1.0 - p

        safe_mae = max(mae, 0.01)
        b = max(mfe / safe_mae, 0.01)

        kelly_f = (p * b - q) / b

        if kelly_f <= 0:
            mult = self.config.min_size_mult
        else:
            mult = self.config.kelly_fraction * kelly_f

            confidence_boost = min(abs(score) / 2.0, 1.0)
            mult *= (1.0 + 0.5 * confidence_boost)

        mult = max(self.config.min_size_mult,
                   min(self.config.max_size_mult, mult))

        self.sizing_history.append(mult)
        return mult

    def get_diagnostics(self) -> dict:
        if not self.sizing_history:
            return {
                'adaptive_sizing_enabled': self.config.enabled,
                'total_sized_trades': 0,
            }
        arr = np.array(self.sizing_history)
        return {
            'adaptive_sizing_enabled': self.config.enabled,
            'total_sized_trades': len(arr),
            'avg_size_mult': float(np.mean(arr)),
            'median_size_mult': float(np.median(arr)),
            'min_size_mult': float(np.min(arr)),
            'max_size_mult': float(np.max(arr)),
            'std_size_mult': float(np.std(arr)),
            'pct_above_1x': float(np.mean(arr > 1.0) * 100),
            'pct_below_1x': float(np.mean(arr < 1.0) * 100),
            'size_p10': float(np.percentile(arr, 10)),
            'size_p90': float(np.percentile(arr, 90)),
        }


class RegimeScaler:
    """Dynamic risk scaling based on market regime.

    Three signals combined (all normalized to [-0.5, +0.5]):
      1. ATR ratio:          current ATR vs rolling average ATR (volatility regime)
      2. EMA200 alignment:   is price trending with or against EMA? (trend regime)
      3. Rolling equity:     recent trade performance (model regime)

    Regime score ∈ [-0.5, +0.5]:  -0.5 = hostile, 0 = neutral, +0.5 = favorable
    Mapped to [bear_mult, bull_mult] via linear interpolation.

    Safety features:
      - ATR signal gated until full lookback window is available (no zero-padding)
      - Equity signal requires min_equity_trades (default 15) before activating
      - When fewer than 2 signals are active, multiplier deviation from 1.0 is
        dampened by low_confidence_dampen (default 0.5) to prevent single-signal extremes
    """

    def __init__(self, config: RegimeScalingConfig):
        self.config = config
        self.recent_r: List[float] = []
        self.regime_history: List[dict] = []

    def compute_regime_multiplier(self, idx: int, side: int,
                                   close_prices: Optional[np.ndarray] = None,
                                   ema200: Optional[np.ndarray] = None,
                                   atr_values: Optional[np.ndarray] = None) -> float:
        if not self.config.enabled:
            return 1.0

        signals = []

        if atr_values is not None and idx >= self.config.atr_lookback:
            window = atr_values[idx - self.config.atr_lookback:idx]
            if len(window) == self.config.atr_lookback:
                rolling_atr = float(np.mean(window))
                current_atr = float(atr_values[idx])
                if rolling_atr > 0 and not np.isnan(current_atr) and not np.isnan(rolling_atr):
                    atr_ratio = current_atr / rolling_atr
                    if atr_ratio <= self.config.atr_bull_ratio:
                        signals.append(0.5)
                    elif atr_ratio >= self.config.atr_bear_ratio:
                        signals.append(-0.5)
                    else:
                        mid = (self.config.atr_bull_ratio + self.config.atr_bear_ratio) / 2
                        rng = (self.config.atr_bear_ratio - self.config.atr_bull_ratio) / 2
                        signals.append(-0.5 * (atr_ratio - mid) / max(rng, 0.01))

        if close_prices is not None and ema200 is not None and idx < len(close_prices):
            close_val = close_prices[idx]
            ema_val = ema200[idx]
            if ema_val > 0:
                trend_strength = (close_val - ema_val) / ema_val
                trend_strength = max(-0.1, min(0.1, trend_strength))
                if side == 1:
                    signals.append(trend_strength * 5)
                else:
                    signals.append(-trend_strength * 5)

        if len(self.recent_r) >= self.config.min_equity_trades:
            recent = self.recent_r[-self.config.lookback_trades:]
            recent_arr = np.array(recent)
            mean_r = np.mean(recent_arr)
            std_r = np.std(recent_arr) + 1e-6
            rolling_sharpe = mean_r / std_r
            equity_signal = max(-0.5, min(0.5, rolling_sharpe * 0.5))
            signals.append(equity_signal)

        if not signals:
            return 1.0

        n_signals = len(signals)
        regime_score = float(np.mean(signals))
        regime_score = max(-0.5, min(0.5, regime_score))

        norm_score = regime_score * 2.0

        if norm_score >= 0:
            mult = 1.0 + norm_score * (self.config.bull_mult - 1.0)
        else:
            mult = 1.0 + norm_score * (1.0 - self.config.bear_mult)

        if n_signals < 2:
            deviation = mult - 1.0
            mult = 1.0 + deviation * self.config.low_confidence_dampen

        self.regime_history.append({
            'idx': idx,
            'regime_score': float(regime_score),
            'multiplier': float(mult),
            'n_signals': n_signals,
        })

        return float(mult)

    def record_trade_result(self, r_value: float):
        if not np.isnan(r_value):
            self.recent_r.append(r_value)

    def get_diagnostics(self) -> dict:
        if not self.regime_history:
            return {
                'regime_scaling_enabled': self.config.enabled,
                'total_regime_trades': 0,
            }
        scores = [h['regime_score'] for h in self.regime_history]
        mults = [h['multiplier'] for h in self.regime_history]
        scores_arr = np.array(scores)
        mults_arr = np.array(mults)
        return {
            'regime_scaling_enabled': self.config.enabled,
            'total_regime_trades': len(self.regime_history),
            'avg_regime_score': float(np.mean(scores_arr)),
            'avg_regime_mult': float(np.mean(mults_arr)),
            'pct_bull': float(np.mean(scores_arr > 0.2) * 100),
            'pct_bear': float(np.mean(scores_arr < -0.2) * 100),
            'pct_neutral': float(np.mean(np.abs(scores_arr) <= 0.2) * 100),
            'regime_mult_p10': float(np.percentile(mults_arr, 10)),
            'regime_mult_p90': float(np.percentile(mults_arr, 90)),
        }


class DailyLossTracker:
    """Enforces daily loss cap and per-symbol daily R budgets.

    - daily_loss_cap: if cumulative R for the day drops below this, skip rest of day
    - per_symbol_daily_r_budget: if any symbol's daily R drops below this, skip that symbol rest of day
    """

    def __init__(self, config: LossManagementConfig):
        self.config = config
        self.current_date: Optional[str] = None
        self.daily_r: float = 0.0
        self.daily_killed: bool = False
        self.symbol_daily_r: Dict[str, float] = {}
        self.symbol_daily_killed: Dict[str, bool] = {}

        self.days_killed: int = 0
        self.trades_blocked_daily: int = 0
        self.trades_blocked_symbol: int = 0
        self.symbol_kills: Dict[str, int] = {}

    def new_bar(self, date_str: str):
        if date_str != self.current_date:
            self.current_date = date_str
            self.daily_r = 0.0
            self.daily_killed = False
            self.symbol_daily_r.clear()
            self.symbol_daily_killed.clear()

    def should_block(self, symbol: Optional[str] = None) -> bool:
        if self.config.daily_loss_cap is not None and self.daily_killed:
            self.trades_blocked_daily += 1
            return True

        if (self.config.per_symbol_daily_r_budget is not None
                and symbol is not None
                and self.symbol_daily_killed.get(symbol, False)):
            self.trades_blocked_symbol += 1
            return True

        return False

    def record_trade(self, r_value: float, symbol: Optional[str] = None):
        if np.isnan(r_value):
            return

        self.daily_r += r_value

        if self.config.daily_loss_cap is not None:
            if self.daily_r <= self.config.daily_loss_cap:
                if not self.daily_killed:
                    self.daily_killed = True
                    self.days_killed += 1
                    log.info("[V5_GATE] daily_loss_cap hit: date=%s cumR=%.2f cap=%.2f",
                             self.current_date, self.daily_r, self.config.daily_loss_cap)

        if symbol is not None:
            self.symbol_daily_r[symbol] = self.symbol_daily_r.get(symbol, 0.0) + r_value
            if self.config.per_symbol_daily_r_budget is not None:
                if self.symbol_daily_r[symbol] <= self.config.per_symbol_daily_r_budget:
                    if not self.symbol_daily_killed.get(symbol, False):
                        self.symbol_daily_killed[symbol] = True
                        self.symbol_kills[symbol] = self.symbol_kills.get(symbol, 0) + 1
                        log.info("[V5_GATE] per_symbol_cap hit: date=%s sym=%s cumR=%.2f cap=%.2f",
                                 self.current_date, symbol, self.symbol_daily_r[symbol],
                                 self.config.per_symbol_daily_r_budget)

    def get_diagnostics(self) -> dict:
        return {
            'daily_loss_cap': self.config.daily_loss_cap,
            'per_symbol_daily_r_budget': self.config.per_symbol_daily_r_budget,
            'days_killed': self.days_killed,
            'trades_blocked_daily_cap': self.trades_blocked_daily,
            'trades_blocked_symbol_cap': self.trades_blocked_symbol,
            'symbol_kill_counts': dict(self.symbol_kills),
        }


class TrailingEquityStop:
    """Pauses trading when equity drawdown exceeds threshold.

    Tracks cumulative R (equity curve). When equity drops more than
    `stop_distance` R from peak, blocks all trades until equity recovers
    to within `recovery_pct` of the stop distance from peak.
    """

    def __init__(self, stop_distance: float, recovery_pct: float = 0.5):
        self.stop_distance = stop_distance
        self.recovery_pct = recovery_pct
        self.cumulative_r: float = 0.0
        self.peak_r: float = 0.0
        self.stopped: bool = False

        self.stop_triggers: int = 0
        self.trades_blocked: int = 0
        self.max_drawdown_r: float = 0.0
        self.drawdown_history: List[float] = []

    def update(self, trade_r: float):
        if np.isnan(trade_r):
            return

        self.cumulative_r += trade_r

        if self.cumulative_r > self.peak_r:
            self.peak_r = self.cumulative_r

        drawdown = self.peak_r - self.cumulative_r
        self.drawdown_history.append(drawdown)
        self.max_drawdown_r = max(self.max_drawdown_r, drawdown)

        if not self.stopped and drawdown >= self.stop_distance:
            self.stopped = True
            self.stop_triggers += 1
            log.info("[V5_GATE] trailing_equity_stop TRIGGERED: peak=%.2f current=%.2f DD=%.2f stop=%.2f",
                     self.peak_r, self.cumulative_r, drawdown, self.stop_distance)

        recovery_threshold = self.stop_distance * self.recovery_pct
        if self.stopped and drawdown <= recovery_threshold:
            self.stopped = False
            log.info("[V5_GATE] trailing_equity_stop RECOVERED: peak=%.2f current=%.2f DD=%.2f",
                     self.peak_r, self.cumulative_r, drawdown)

    def should_block(self) -> bool:
        if self.stopped:
            self.trades_blocked += 1
            return True
        return False

    def get_diagnostics(self) -> dict:
        dd_arr = np.array(self.drawdown_history) if self.drawdown_history else np.array([0.0])
        return {
            'trailing_equity_stop': self.stop_distance,
            'stop_triggers': self.stop_triggers,
            'trades_blocked_equity_stop': self.trades_blocked,
            'max_drawdown_r': float(self.max_drawdown_r),
            'avg_drawdown_r': float(np.mean(dd_arr)),
            'final_equity_r': float(self.cumulative_r),
            'peak_equity_r': float(self.peak_r),
        }


@dataclass
class ConvictionSizingConfig:
    enabled: bool = False
    tier_top_pct: float = 5.0
    tier_top_mult: float = 2.5
    tier_high_pct: float = 20.0
    tier_high_mult: float = 1.5
    tier_mid_mult: float = 1.0
    tier_low_pct: float = 50.0
    tier_low_mult: float = 0.5
    confidence_boost_threshold: float = 0.65
    confidence_boost_mult: float = 1.3
    max_combined_mult: float = 3.5
    window_size: int = 500


class ConvictionSizer:
    """Score-tiered position sizing with directional confidence boost.

    Assigns size multipliers based on where the trade's score falls
    in the distribution of recent scores (percentile-based tiers):
      - Top tier_top_pct%:  tier_top_mult  (e.g., top 5% → 2.5x)
      - Top tier_high_pct%: tier_high_mult (e.g., top 20% → 1.5x)
      - Middle:             tier_mid_mult  (e.g., 1.0x)
      - Bottom tier_low_pct%: tier_low_mult (e.g., bottom 50% → 0.5x)

    Additionally, if directional confidence (p_long for LONG, p_short for SHORT)
    exceeds confidence_boost_threshold, the multiplier gets a confidence_boost_mult.

    Score percentiles are computed from a rolling window of the most recent
    window_size scores to adapt to changing model output distributions.
    """

    def __init__(self, config: ConvictionSizingConfig):
        self.config = config
        from collections import deque
        self.score_window: deque = deque(maxlen=config.window_size)
        self.sizing_history: List[dict] = []
        self._min_scores_for_tiers = 20

    def compute_size_multiplier(self, score: float, p_directional: float,
                                 side: int) -> float:
        if not self.config.enabled:
            return 1.0

        self.score_window.append(score)

        if len(self.score_window) < self._min_scores_for_tiers:
            tier_mult = self.config.tier_mid_mult
            tier_name = "warmup"
        else:
            scores_arr = np.array(self.score_window)
            pct = float(np.sum(scores_arr < score) / len(scores_arr) * 100)

            if pct >= (100 - self.config.tier_top_pct):
                tier_mult = self.config.tier_top_mult
                tier_name = "top"
            elif pct >= (100 - self.config.tier_high_pct):
                tier_mult = self.config.tier_high_mult
                tier_name = "high"
            elif pct < self.config.tier_low_pct:
                tier_mult = self.config.tier_low_mult
                tier_name = "low"
            else:
                tier_mult = self.config.tier_mid_mult
                tier_name = "mid"

        confidence_boost = 1.0
        if p_directional >= self.config.confidence_boost_threshold:
            confidence_boost = self.config.confidence_boost_mult

        combined = tier_mult * confidence_boost
        combined = min(combined, self.config.max_combined_mult)

        self.sizing_history.append({
            'score': score,
            'tier': tier_name,
            'tier_mult': tier_mult,
            'confidence_boost': confidence_boost,
            'combined_mult': combined,
            'p_directional': p_directional,
            'side': side,
        })

        return combined

    def get_diagnostics(self) -> dict:
        if not self.sizing_history:
            return {
                'conviction_sizing_enabled': self.config.enabled,
                'total_conviction_trades': 0,
            }
        mults = np.array([h['combined_mult'] for h in self.sizing_history])
        tiers = [h['tier'] for h in self.sizing_history]
        boosts = [h['confidence_boost'] for h in self.sizing_history]
        return {
            'conviction_sizing_enabled': self.config.enabled,
            'total_conviction_trades': len(self.sizing_history),
            'avg_conviction_mult': float(np.mean(mults)),
            'median_conviction_mult': float(np.median(mults)),
            'min_conviction_mult': float(np.min(mults)),
            'max_conviction_mult': float(np.max(mults)),
            'tier_distribution': {
                'top': tiers.count('top'),
                'high': tiers.count('high'),
                'mid': tiers.count('mid'),
                'low': tiers.count('low'),
                'warmup': tiers.count('warmup'),
            },
            'pct_confidence_boosted': float(np.mean([b > 1.0 for b in boosts]) * 100),
            'conviction_mult_p10': float(np.percentile(mults, 10)),
            'conviction_mult_p90': float(np.percentile(mults, 90)),
        }


@dataclass
class MultiRegimeConfig:
    enabled: bool = False
    adx_trending_threshold: float = 25.0
    adx_choppy_threshold: float = 20.0
    atr_high_vol_ratio: float = 1.3
    atr_low_vol_ratio: float = 0.7
    atr_rolling_window: int = 96
    ema_slope_window: int = 10
    ema_price_buffer: float = 0.005


class MultiRegimeClassifier:
    """Multi-regime market state classifier using ADX + ATR ratio + EMA200 slope.

    Classifies each bar into one of 5 regimes:
      - trending_up:   ADX >= adx_trending_threshold AND price > EMA200 AND EMA slope > 0
      - trending_down: ADX >= adx_trending_threshold AND price < EMA200 AND EMA slope < 0
      - choppy:        ADX < adx_choppy_threshold (no clear trend)
      - high_vol:      ATR ratio > atr_high_vol_ratio (danger zone, volatile)
      - low_vol:       ATR ratio < atr_low_vol_ratio AND not trending (compression)

    Priority order (for overlapping conditions):
      1. high_vol  (overrides everything — volatile markets are dangerous)
      2. choppy    (ADX says no trend — avoid directional trades)
      3. trending_up / trending_down (clear directional trend)
      4. low_vol   (quiet compression, potential breakout)

    Ultra-Conviction alignment:
      - LONG allowed:  trending_up, low_vol
      - SHORT allowed: trending_down, low_vol
      - BLOCKED:       choppy, high_vol
    """

    REGIMES = ("trending_up", "trending_down", "choppy", "high_vol", "low_vol")
    LONG_ALLOWED = {"trending_up", "low_vol"}
    SHORT_ALLOWED = {"trending_down", "low_vol"}
    BLOCKED = {"choppy", "high_vol"}

    def __init__(self, config: MultiRegimeConfig):
        self.config = config
        self.regime_counts: Dict[str, int] = {r: 0 for r in self.REGIMES}
        self.regime_counts["unknown"] = 0
        self.total_classified: int = 0

    def classify(self, adx_val: float, atr_current: float,
                 atr_rolling: float, close_price: float,
                 ema200_val: float, ema200_prev: float) -> str:
        self.total_classified += 1

        has_adx = not np.isnan(adx_val)
        has_atr = (not np.isnan(atr_current) and not np.isnan(atr_rolling)
                   and atr_rolling > 1e-10)
        has_ema = (not np.isnan(ema200_val) and not np.isnan(ema200_prev)
                   and ema200_val > 0)

        atr_ratio = (atr_current / atr_rolling) if has_atr else 1.0

        if has_atr and atr_ratio > self.config.atr_high_vol_ratio:
            self.regime_counts["high_vol"] += 1
            return "high_vol"

        if has_adx and adx_val < self.config.adx_choppy_threshold:
            if has_atr and atr_ratio < self.config.atr_low_vol_ratio:
                self.regime_counts["low_vol"] += 1
                return "low_vol"
            self.regime_counts["choppy"] += 1
            return "choppy"

        if has_adx and adx_val >= self.config.adx_trending_threshold and has_ema:
            ema_slope = (ema200_val - ema200_prev) / ema200_prev
            price_above = close_price > ema200_val * (1.0 + self.config.ema_price_buffer)
            price_below = close_price < ema200_val * (1.0 - self.config.ema_price_buffer)

            if price_above and ema_slope > 0:
                self.regime_counts["trending_up"] += 1
                return "trending_up"
            elif price_below and ema_slope < 0:
                self.regime_counts["trending_down"] += 1
                return "trending_down"

        if has_atr and atr_ratio < self.config.atr_low_vol_ratio:
            self.regime_counts["low_vol"] += 1
            return "low_vol"

        if has_adx and adx_val >= self.config.adx_choppy_threshold:
            if has_ema:
                if close_price > ema200_val:
                    self.regime_counts["trending_up"] += 1
                    return "trending_up"
                else:
                    self.regime_counts["trending_down"] += 1
                    return "trending_down"

        self.regime_counts["unknown"] += 1
        return "choppy"

    def is_long_allowed(self, regime: str) -> bool:
        return regime in self.LONG_ALLOWED

    def is_short_allowed(self, regime: str) -> bool:
        return regime in self.SHORT_ALLOWED

    def is_blocked(self, regime: str) -> bool:
        return regime in self.BLOCKED

    def get_diagnostics(self) -> dict:
        total = max(self.total_classified, 1)
        return {
            'multi_regime_enabled': self.config.enabled,
            'total_classified': self.total_classified,
            'regime_counts': dict(self.regime_counts),
            'regime_pct': {k: round(v / total * 100, 1)
                           for k, v in self.regime_counts.items()},
            'config': {
                'adx_trending': self.config.adx_trending_threshold,
                'adx_choppy': self.config.adx_choppy_threshold,
                'atr_high_vol': self.config.atr_high_vol_ratio,
                'atr_low_vol': self.config.atr_low_vol_ratio,
            }
        }


@dataclass
class UltraConvictionConfig:
    enabled: bool = False
    risk_cap: float = 0.05
    score_pct: float = 0.95
    adx_min: float = 25.0
    edge_min: float = 0.03
    dd_max: float = 0.10
    max_per_day: int = 1
    mult: float = 3.0
    score_window: int = 500


class UltraConvictionSizer:
    """Ultra-Conviction Risk Tier: allows rare, very-high-conviction trades
    to use up to risk_cap (default 5%) risk per trade.

    Applied AFTER conviction sizing but BEFORE final clamp.
    All 8 trigger conditions must be true for ultra to activate:
      1) Feature enabled
      2) Score >= rolling percentile(score_history, score_pct)
      3) ADX >= adx_min (strong trend)
      4) Edge for the trade side >= edge_min
      5) Regime aligned (bull→LONG, bear→SHORT; neutral disables)
      6) Equity drawdown <= dd_max
      7) Not blocked by any existing capital protection gate
      8) Ultra trades today < max_per_day
    """

    def __init__(self, config: UltraConvictionConfig):
        self.config = config
        from collections import deque
        self.score_window: deque = deque(maxlen=config.score_window)
        self.daily_ultra_counts: Dict[str, int] = {}
        self.peak_equity: float = 0.0
        self.current_equity: float = 0.0
        self.ultra_applied: int = 0
        self.ultra_skipped: int = 0
        self.skip_reasons: Dict[str, int] = {}
        self.ultra_history: List[dict] = []
        self._min_scores_for_pct = 20

    def update_equity(self, trade_r: float):
        if not np.isnan(trade_r):
            self.current_equity += trade_r
            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity

    def _get_drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity

    def _get_score_percentile(self, score: float) -> Optional[float]:
        if len(self.score_window) < self._min_scores_for_pct:
            return None
        scores_arr = np.array(self.score_window)
        pct_val = float(np.percentile(scores_arr, self.config.score_pct * 100))
        return pct_val

    def record_score(self, score: float):
        self.score_window.append(score)

    def _skip(self, reason: str, **log_kwargs) -> bool:
        self.ultra_skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1
        parts = [f"reason={reason}"]
        for k, v in log_kwargs.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        log.info("[V5_ULTRA] SKIP %s", " ".join(parts))
        return False

    def evaluate(self, symbol: str, side: int, score: float,
                 edge_l: float, edge_s: float, adx_val: float,
                 regime: str, date_str: str,
                 capital_blocked: bool = False) -> bool:
        if not self.config.enabled:
            return False

        pct_threshold = self._get_score_percentile(score)
        if pct_threshold is None:
            return self._skip("warmup", scores=len(self.score_window))
        if score < pct_threshold:
            return self._skip("score_low", score=score, threshold=pct_threshold)

        if not np.isnan(adx_val) and adx_val < self.config.adx_min:
            return self._skip("adx_low", adx=adx_val, min=self.config.adx_min)

        edge = edge_l if side == 1 else edge_s
        if edge < self.config.edge_min:
            return self._skip("edge_low", edge=edge, min=self.config.edge_min,
                              side="LONG" if side == 1 else "SHORT")

        _BLOCKED_REGIMES = {"choppy", "high_vol", "neutral", "chop", "unknown", ""}
        if regime in _BLOCKED_REGIMES:
            return self._skip("regime_blocked", regime=regime)
        _LONG_REGIMES = {"trending_up", "low_vol", "bull"}
        _SHORT_REGIMES = {"trending_down", "low_vol", "bear"}
        if side == 1 and regime not in _LONG_REGIMES:
            return self._skip("regime_mismatch", regime=regime,
                              side="LONG")
        if side == -1 and regime not in _SHORT_REGIMES:
            return self._skip("regime_mismatch", regime=regime,
                              side="SHORT")

        dd = self._get_drawdown()
        if dd > self.config.dd_max:
            return self._skip("drawdown_high", dd=dd, max=self.config.dd_max)

        if capital_blocked:
            return self._skip("capital_blocked")

        day_count = self.daily_ultra_counts.get(date_str, 0)
        if day_count >= self.config.max_per_day:
            return self._skip("max_per_day", count=day_count, max=self.config.max_per_day)

        self.daily_ultra_counts[date_str] = day_count + 1
        self.ultra_applied += 1
        side_str = "LONG" if side == 1 else "SHORT"
        log.info("[V5_ULTRA] APPLY symbol=%s side=%s score=%.2f p%.0f=%.2f adx=%.1f "
                 "edge_%s=%.3f regime=%s dd=%.2f mult=%.1f",
                 symbol, side_str, score, self.config.score_pct * 100,
                 pct_threshold, adx_val,
                 "L" if side == 1 else "S", edge,
                 regime, dd, self.config.mult)
        return True

    def apply_ultra_sizing(self, current_mult: float, stop_distance_pct: float) -> float:
        sized_mult = current_mult * self.config.mult
        if stop_distance_pct > 0:
            risk_at_size = sized_mult * stop_distance_pct
            if risk_at_size > self.config.risk_cap:
                sized_mult = self.config.risk_cap / stop_distance_pct
        self.ultra_history.append({
            'mult_before': current_mult,
            'mult_after': sized_mult,
            'risk_cap': self.config.risk_cap,
        })
        return sized_mult

    def get_diagnostics(self) -> dict:
        result = {
            'ultra_conviction_enabled': self.config.enabled,
            'ultra_applied': self.ultra_applied,
            'ultra_skipped': self.ultra_skipped,
            'skip_reasons': dict(self.skip_reasons),
            'risk_cap': self.config.risk_cap,
            'mult': self.config.mult,
            'score_pct': self.config.score_pct,
            'adx_min': self.config.adx_min,
            'edge_min': self.config.edge_min,
            'dd_max': self.config.dd_max,
            'max_per_day': self.config.max_per_day,
        }
        if self.ultra_history:
            mults_before = np.array([h['mult_before'] for h in self.ultra_history])
            mults_after = np.array([h['mult_after'] for h in self.ultra_history])
            result['avg_mult_before'] = float(np.mean(mults_before))
            result['avg_mult_after'] = float(np.mean(mults_after))
            result['max_mult_after'] = float(np.max(mults_after))
        return result


def build_sizing_diagnostics(sizer: Optional[AdaptivePositionSizer],
                              regime: Optional[RegimeScaler],
                              daily_tracker: Optional[DailyLossTracker],
                              equity_stop: Optional[TrailingEquityStop],
                              sized_r: Optional[np.ndarray] = None,
                              unsized_r: Optional[np.ndarray] = None,
                              conviction: Optional[ConvictionSizer] = None,
                              ultra: Optional['UltraConvictionSizer'] = None) -> dict:
    report = {}
    if sizer:
        report['adaptive_sizing'] = sizer.get_diagnostics()
    if regime:
        report['regime_scaling'] = regime.get_diagnostics()
    if daily_tracker:
        report['daily_loss_management'] = daily_tracker.get_diagnostics()
    if equity_stop:
        report['trailing_equity_stop'] = equity_stop.get_diagnostics()
    if conviction:
        report['conviction_sizing'] = conviction.get_diagnostics()
    if ultra:
        report['ultra_conviction'] = ultra.get_diagnostics()

    if sized_r is not None and unsized_r is not None and len(sized_r) > 0:
        report['sizing_comparison'] = {
            'unsized_total_r': float(np.sum(unsized_r)),
            'sized_total_r': float(np.sum(sized_r)),
            'sizing_impact_r': float(np.sum(sized_r) - np.sum(unsized_r)),
            'sizing_impact_pct': float((np.sum(sized_r) / max(abs(np.sum(unsized_r)), 0.01) - 1.0) * 100),
        }

    return report
