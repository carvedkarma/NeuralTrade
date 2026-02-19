"""Tests for v5.0.8+ Adaptive Position Sizing & Dynamic Risk Scaling."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from train.v5_position_sizer import (
    AdaptivePositionSizer, AdaptiveSizingConfig,
    RegimeScaler, RegimeScalingConfig,
    DailyLossTracker, LossManagementConfig,
    TrailingEquityStop,
    build_sizing_diagnostics,
)


class TestAdaptivePositionSizer:
    def test_disabled_returns_1(self):
        cfg = AdaptiveSizingConfig(enabled=False)
        sizer = AdaptivePositionSizer(cfg)
        assert sizer.compute_size_multiplier(1.0, 0.6, 0.5, 0.5, 0.3) == 1.0

    def test_high_pwin_high_rr_gives_above_min(self):
        cfg = AdaptiveSizingConfig(enabled=True, kelly_fraction=0.5, max_size_mult=3.0, min_size_mult=0.25)
        sizer = AdaptivePositionSizer(cfg)
        mult = sizer.compute_size_multiplier(score=2.0, p_win=0.7, mu_r=1.0, mfe=2.0, mae=0.5)
        assert mult > cfg.min_size_mult
        assert mult <= cfg.max_size_mult

    def test_low_pwin_gives_min_mult(self):
        cfg = AdaptiveSizingConfig(enabled=True, kelly_fraction=0.25, min_size_mult=0.25)
        sizer = AdaptivePositionSizer(cfg)
        mult = sizer.compute_size_multiplier(score=0.5, p_win=0.3, mu_r=-0.5, mfe=0.2, mae=1.0)
        assert mult == 0.25

    def test_higher_pwin_gives_higher_mult(self):
        cfg = AdaptiveSizingConfig(enabled=True, kelly_fraction=0.5, max_size_mult=3.0)
        sizer = AdaptivePositionSizer(cfg)
        mult_low = sizer.compute_size_multiplier(score=1.0, p_win=0.5, mu_r=0.5, mfe=1.0, mae=0.5)
        mult_high = sizer.compute_size_multiplier(score=1.0, p_win=0.8, mu_r=0.5, mfe=1.0, mae=0.5)
        assert mult_high > mult_low

    def test_diagnostics(self):
        cfg = AdaptiveSizingConfig(enabled=True, kelly_fraction=0.25)
        sizer = AdaptivePositionSizer(cfg)
        for i in range(20):
            sizer.compute_size_multiplier(score=1.0, p_win=0.5 + i*0.01, mu_r=0.5, mfe=1.0, mae=0.5)
        diag = sizer.get_diagnostics()
        assert diag['total_sized_trades'] == 20
        assert 'avg_size_mult' in diag
        assert 'pct_above_1x' in diag

    def test_safe_mae_zero(self):
        cfg = AdaptiveSizingConfig(enabled=True, kelly_fraction=0.25)
        sizer = AdaptivePositionSizer(cfg)
        mult = sizer.compute_size_multiplier(score=1.0, p_win=0.6, mu_r=0.5, mfe=1.0, mae=0.0)
        assert mult >= cfg.min_size_mult
        assert mult <= cfg.max_size_mult


class TestRegimeScaler:
    def test_disabled_returns_1(self):
        cfg = RegimeScalingConfig(enabled=False)
        scaler = RegimeScaler(cfg)
        assert scaler.compute_regime_multiplier(100, 1) == 1.0

    def test_ema_bullish_long(self):
        cfg = RegimeScalingConfig(enabled=True, bull_mult=1.5, bear_mult=0.5)
        scaler = RegimeScaler(cfg)
        close = np.full(300, 100.0)
        close[200:] = 110.0
        ema = np.full(300, 100.0)
        mult = scaler.compute_regime_multiplier(250, side=1, close_prices=close, ema200=ema)
        assert mult > 1.0

    def test_ema_bearish_long(self):
        cfg = RegimeScalingConfig(enabled=True, bull_mult=1.5, bear_mult=0.5)
        scaler = RegimeScaler(cfg)
        close = np.full(300, 90.0)
        ema = np.full(300, 100.0)
        mult = scaler.compute_regime_multiplier(250, side=1, close_prices=close, ema200=ema)
        assert mult < 1.0

    def test_rolling_equity_positive(self):
        cfg = RegimeScalingConfig(enabled=True, lookback_trades=10, min_equity_trades=15)
        scaler = RegimeScaler(cfg)
        for _ in range(20):
            scaler.record_trade_result(1.0)
        mult = scaler.compute_regime_multiplier(100, side=1)
        assert mult > 1.0

    def test_rolling_equity_negative(self):
        cfg = RegimeScalingConfig(enabled=True, lookback_trades=10, min_equity_trades=15)
        scaler = RegimeScaler(cfg)
        for _ in range(20):
            scaler.record_trade_result(-1.0)
        mult = scaler.compute_regime_multiplier(100, side=1)
        assert mult < 1.0

    def test_atr_low_volatility_bullish(self):
        cfg = RegimeScalingConfig(enabled=True, atr_lookback=50, atr_bull_ratio=0.8, atr_bear_ratio=1.5)
        scaler = RegimeScaler(cfg)
        atr = np.full(200, 1.0)
        atr[150:] = 0.5
        mult = scaler.compute_regime_multiplier(160, side=1, atr_values=atr)
        assert mult > 1.0

    def test_atr_high_volatility_bearish(self):
        cfg = RegimeScalingConfig(enabled=True, atr_lookback=50, atr_bull_ratio=0.8, atr_bear_ratio=1.5)
        scaler = RegimeScaler(cfg)
        atr = np.full(200, 1.0)
        atr[150:] = 2.0
        mult = scaler.compute_regime_multiplier(160, side=1, atr_values=atr)
        assert mult < 1.0

    def test_diagnostics(self):
        cfg = RegimeScalingConfig(enabled=True)
        scaler = RegimeScaler(cfg)
        close = np.full(300, 100.0)
        ema = np.full(300, 100.0)
        for i in range(10):
            scaler.compute_regime_multiplier(200 + i, side=1, close_prices=close, ema200=ema)
        diag = scaler.get_diagnostics()
        assert diag['total_regime_trades'] == 10
        assert 'avg_regime_score' in diag

    def test_equity_below_min_trades_ignored(self):
        cfg = RegimeScalingConfig(enabled=True, min_equity_trades=15)
        scaler = RegimeScaler(cfg)
        for _ in range(10):
            scaler.record_trade_result(-2.0)
        mult = scaler.compute_regime_multiplier(100, side=1)
        assert mult == 1.0

    def test_low_confidence_dampening_single_signal(self):
        cfg = RegimeScalingConfig(enabled=True, bull_mult=1.5, bear_mult=0.5,
                                  min_equity_trades=5, low_confidence_dampen=0.5)
        scaler = RegimeScaler(cfg)
        for _ in range(10):
            scaler.record_trade_result(1.0)
        mult_dampened = scaler.compute_regime_multiplier(100, side=1)
        assert mult_dampened > 1.0
        assert mult_dampened <= 1.25

    def test_no_dampening_with_multiple_signals(self):
        cfg = RegimeScalingConfig(enabled=True, bull_mult=1.5, bear_mult=0.5,
                                  atr_lookback=50, min_equity_trades=5)
        scaler = RegimeScaler(cfg)
        for _ in range(10):
            scaler.record_trade_result(1.0)
        atr = np.full(200, 1.0)
        atr[150:] = 0.5
        mult = scaler.compute_regime_multiplier(160, side=1, atr_values=atr)
        assert mult > 1.0
        history = scaler.regime_history[-1]
        assert history['n_signals'] >= 2

    def test_atr_nan_early_bars_no_signal(self):
        cfg = RegimeScalingConfig(enabled=True, atr_lookback=50)
        scaler = RegimeScaler(cfg)
        atr = np.full(200, np.nan)
        atr[100:] = 1.0
        mult = scaler.compute_regime_multiplier(30, side=1, atr_values=atr)
        assert mult == 1.0

    def test_equity_signal_capped_at_half(self):
        cfg = RegimeScalingConfig(enabled=True, bull_mult=1.5, bear_mult=0.5,
                                  min_equity_trades=5, low_confidence_dampen=1.0)
        scaler = RegimeScaler(cfg)
        for _ in range(20):
            scaler.record_trade_result(5.0)
        mult = scaler.compute_regime_multiplier(100, side=1)
        assert mult <= 1.5
        assert mult > 1.0


class TestDailyLossTracker:
    def test_daily_cap_blocks(self):
        cfg = LossManagementConfig(daily_loss_cap=-3.0)
        tracker = DailyLossTracker(cfg)
        tracker.new_bar("2025-01-01")
        assert not tracker.should_block()
        tracker.record_trade(-2.0)
        assert not tracker.should_block()
        tracker.record_trade(-1.5)
        assert tracker.should_block()

    def test_day_reset(self):
        cfg = LossManagementConfig(daily_loss_cap=-3.0)
        tracker = DailyLossTracker(cfg)
        tracker.new_bar("2025-01-01")
        tracker.record_trade(-4.0)
        assert tracker.should_block()
        tracker.new_bar("2025-01-02")
        assert not tracker.should_block()

    def test_per_symbol_cap(self):
        cfg = LossManagementConfig(per_symbol_daily_r_budget=-2.0)
        tracker = DailyLossTracker(cfg)
        tracker.new_bar("2025-01-01")
        tracker.record_trade(-1.5, symbol="BTCUSDT")
        assert not tracker.should_block(symbol="BTCUSDT")
        tracker.record_trade(-1.0, symbol="BTCUSDT")
        assert tracker.should_block(symbol="BTCUSDT")
        assert not tracker.should_block(symbol="ETHUSDT")

    def test_diagnostics(self):
        cfg = LossManagementConfig(daily_loss_cap=-3.0, per_symbol_daily_r_budget=-2.0)
        tracker = DailyLossTracker(cfg)
        tracker.new_bar("2025-01-01")
        tracker.record_trade(-4.0, symbol="BTCUSDT")
        tracker.should_block(symbol="BTCUSDT")
        diag = tracker.get_diagnostics()
        assert diag['days_killed'] == 1
        assert 'symbol_kill_counts' in diag

    def test_no_cap_no_block(self):
        cfg = LossManagementConfig()
        tracker = DailyLossTracker(cfg)
        tracker.new_bar("2025-01-01")
        tracker.record_trade(-10.0)
        assert not tracker.should_block()


class TestTrailingEquityStop:
    def test_triggers_on_drawdown(self):
        stop = TrailingEquityStop(stop_distance=5.0)
        stop.update(10.0)
        assert not stop.should_block()
        stop.update(-6.0)
        assert stop.should_block()

    def test_recovery(self):
        stop = TrailingEquityStop(stop_distance=5.0, recovery_pct=0.5)
        stop.update(10.0)
        stop.update(-6.0)
        assert stop.should_block()
        stop.update(4.0)
        assert not stop.should_block()

    def test_no_false_trigger(self):
        stop = TrailingEquityStop(stop_distance=10.0)
        for _ in range(20):
            stop.update(0.5)
        assert not stop.should_block()

    def test_diagnostics(self):
        stop = TrailingEquityStop(stop_distance=5.0)
        stop.update(10.0)
        stop.update(-7.0)
        stop.update(3.0)
        diag = stop.get_diagnostics()
        assert diag['stop_triggers'] == 1
        assert diag['max_drawdown_r'] >= 5.0
        assert diag['final_equity_r'] == 6.0

    def test_multiple_triggers(self):
        stop = TrailingEquityStop(stop_distance=3.0, recovery_pct=0.3)
        stop.update(5.0)
        stop.update(-4.0)
        assert stop.should_block()
        stop.update(3.5)
        assert not stop.should_block()
        stop.update(2.0)
        stop.update(-6.0)
        assert stop.should_block()
        diag = stop.get_diagnostics()
        assert diag['stop_triggers'] == 2


class TestBuildSizingDiagnostics:
    def test_empty(self):
        diag = build_sizing_diagnostics(None, None, None, None)
        assert diag == {}

    def test_full(self):
        sizer_cfg = AdaptiveSizingConfig(enabled=True)
        sizer = AdaptivePositionSizer(sizer_cfg)
        sizer.compute_size_multiplier(1.0, 0.6, 0.5, 1.0, 0.5)

        regime_cfg = RegimeScalingConfig(enabled=True)
        regime = RegimeScaler(regime_cfg)
        regime.record_trade_result(1.0)

        loss_cfg = LossManagementConfig(daily_loss_cap=-3.0)
        daily = DailyLossTracker(loss_cfg)

        equity = TrailingEquityStop(5.0)

        sized_r = np.array([1.0, -0.5, 2.0])
        unsized_r = np.array([0.8, -0.4, 1.5])

        diag = build_sizing_diagnostics(sizer, regime, daily, equity, sized_r, unsized_r)
        assert 'adaptive_sizing' in diag
        assert 'regime_scaling' in diag
        assert 'daily_loss_management' in diag
        assert 'trailing_equity_stop' in diag
        assert 'sizing_comparison' in diag
        assert abs(diag['sizing_comparison']['sizing_impact_r'] - (2.5 - 1.9)) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
