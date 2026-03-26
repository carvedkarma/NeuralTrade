"""Tests for v5.2.0 precision audit fixes (T001-T006).

Covers:
  T001: Lookahead bias fix in quality gate (ref_arrays)
  T002: Oracle fallback removal (ValueError on missing side-conditional)
  T003: Head disagreement gate
  T004: Statistical edge metrics (Sortino, t-stat, CI)
  T005: Train/test purge gap
  T006: Slippage deduction in score computation
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")

if HAS_TORCH:
    from train.v5_train import (
        v5_quality_mask,
        V5QualityGateConfig,
        V5ForwardTestConfig,
        compute_v5_scores,
        _compute_time_split,
        _compute_forward_metrics,
    )


def _make_arrays(n=500, seed=42):
    rng = np.random.RandomState(seed)
    return {
        'mu_R': rng.randn(n).astype(np.float32) * 0.1,
        'mae': np.abs(rng.randn(n).astype(np.float32) * 0.3) + 0.01,
        'mfe': np.abs(rng.randn(n).astype(np.float32) * 0.3) + 0.01,
        'sigma': np.abs(rng.randn(n).astype(np.float32) * 0.2) + 0.01,
        'p_trade': rng.uniform(0.2, 0.9, n).astype(np.float32),
        'p_long': rng.uniform(0.1, 0.8, n).astype(np.float32),
        'p_short': rng.uniform(0.1, 0.8, n).astype(np.float32),
    }


class TestT001LookaheadBiasQualityGate:

    def test_ref_arrays_used_for_percentiles(self):
        test_arrays = _make_arrays(n=500, seed=1)
        test_arrays['mu_R'][:] = 0.5
        # Heterogeneous mae: 450 bars at 0.01, 50 bars at 0.5.
        # Without ref: adaptive_mae ≈ percentile([0.01]*450+[0.5]*50, 90) ≈ 0.059
        #   → bars with mae=0.5 FAIL threshold → ~450 pass.
        # With ref (mae=5.0): adaptive_mae = min(1.0, 5.0 * <cfg factor>) = 1.0
        #   → all 500 bars pass.
        # 450 ≠ 500 confirms ref_arrays change the gate.
        test_arrays['mae'][:450] = 0.01
        test_arrays['mae'][450:] = 0.5
        test_arrays['sigma'][:] = 0.01
        test_arrays['p_trade'][:] = 0.99

        ref_arrays = _make_arrays(n=500, seed=2)
        ref_arrays['mu_R'][:] = 0.01
        ref_arrays['mae'][:] = 5.0
        ref_arrays['sigma'][:] = 5.0
        ref_arrays['p_trade'][:] = 0.1

        cfg = V5QualityGateConfig()

        mask_no_ref, _ = v5_quality_mask(test_arrays, cfg, epoch=999)
        mask_with_ref, _ = v5_quality_mask(test_arrays, cfg, epoch=999, ref_arrays=ref_arrays)

        pass_no_ref = np.sum(mask_no_ref)
        pass_with_ref = np.sum(mask_with_ref)
        assert pass_no_ref != pass_with_ref, \
            "ref_arrays should change which bars pass the quality gate"

    def test_warmup_bypass_ignores_ref(self):
        test_arrays = _make_arrays(n=100)
        ref_arrays = _make_arrays(n=100)
        ref_arrays['sigma'][:] = 0.001

        cfg = V5QualityGateConfig()
        mask, diag = v5_quality_mask(test_arrays, cfg, epoch=2, ref_arrays=ref_arrays)
        assert diag['warmup_bypass'] is True
        assert np.all(mask)

    def test_no_ref_arrays_backward_compatible(self):
        arrays = _make_arrays(n=500)
        cfg = V5QualityGateConfig()
        mask1, _ = v5_quality_mask(arrays, cfg, epoch=999)
        mask2, _ = v5_quality_mask(arrays, cfg, epoch=999, ref_arrays=None)
        np.testing.assert_array_equal(mask1, mask2)


class TestT002OracleFallbackRemoval:

    def test_missing_side_arrays_raises_error(self):
        from unittest.mock import MagicMock

        model = MagicMock()
        n = 50
        dummy_out = {
            'ret_mu': torch.randn(n, 1),
            'mae': torch.abs(torch.randn(n, 1)),
            'mfe': torch.abs(torch.randn(n, 1)),
            'action_logits': torch.randn(n, 3),
            'ret_log_sigma': torch.randn(n, 1),
        }
        model.return_value = dummy_out
        model.eval = MagicMock()
        model.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(2, 2))])

        config = V5ForwardTestConfig(score_threshold=0.1, horizon=16)

        with pytest.raises(ValueError, match="REQUIRED"):
            from train.v5_train import run_v5_forward_test
            run_v5_forward_test(
                model=model,
                device=torch.device('cpu'),
                test_features=np.random.randn(n, 10).astype(np.float32),
                test_outcomes=np.array(["TP"] * n),
                test_realized_r=np.random.randn(n).astype(np.float32),
                test_sym_ids=np.zeros(n, dtype=np.int64),
                test_cand_mask=np.ones(n, dtype=bool),
                test_valid=np.ones(n, dtype=np.float32),
                test_bars=n,
                config=config,
            )


class TestT003HeadDisagreementGate:

    def test_config_default_off(self):
        cfg = V5ForwardTestConfig()
        assert cfg.head_disagreement_gate is False

    def test_config_can_enable(self):
        cfg = V5ForwardTestConfig(head_disagreement_gate=True)
        assert cfg.head_disagreement_gate is True


class TestT004StatisticalMetrics:

    def test_sortino_in_report(self):
        rng = np.random.RandomState(42)
        n = 100
        t_r = rng.randn(n).astype(np.float32) * 0.1 + 0.02
        t_outcomes = np.array(["TP"] * 60 + ["SL"] * 40)
        t_sides = np.where(t_r > 0, 1, -1)
        config = V5ForwardTestConfig(score_threshold=0.1, horizon=16)

        report = _compute_forward_metrics(
            t_r, t_outcomes, t_sides, n, config,
            start_date="2024-01-01", end_date="2024-03-01"
        )

        assert 'sortino' in report
        assert isinstance(report['sortino'], float)
        assert np.isfinite(report['sortino'])

    def test_tstat_pvalue_in_report(self):
        rng = np.random.RandomState(42)
        n = 100
        t_r = rng.randn(n).astype(np.float32) * 0.1 + 0.05
        t_outcomes = np.array(["TP"] * n)
        t_sides = np.ones(n, dtype=int)
        config = V5ForwardTestConfig(score_threshold=0.1, horizon=16)

        report = _compute_forward_metrics(
            t_r, t_outcomes, t_sides, n, config,
            start_date="2024-01-01", end_date="2024-03-01"
        )

        assert 't_stat' in report
        assert 'p_value' in report
        assert report['t_stat'] > 0
        assert 0 < report['p_value'] < 1

    def test_bootstrap_ci_in_report(self):
        rng = np.random.RandomState(42)
        n = 100
        t_r = rng.randn(n).astype(np.float32) * 0.1 + 0.03
        t_outcomes = np.array(["TP"] * n)
        t_sides = np.ones(n, dtype=int)
        config = V5ForwardTestConfig(score_threshold=0.1, horizon=16)

        report = _compute_forward_metrics(
            t_r, t_outcomes, t_sides, n, config,
            start_date="2024-01-01", end_date="2024-03-01"
        )

        assert 'ci_95_lower' in report
        assert 'ci_95_upper' in report
        assert report['ci_95_lower'] < report['ci_95_upper']
        assert report['ci_95_lower'] < report['expectancy_r'] < report['ci_95_upper']

    def test_few_trades_no_crash(self):
        t_r = np.array([0.1], dtype=np.float32)
        t_outcomes = np.array(["TP"])
        t_sides = np.array([1])
        config = V5ForwardTestConfig(score_threshold=0.1, horizon=16)

        report = _compute_forward_metrics(
            t_r, t_outcomes, t_sides, 96, config,
            start_date="2024-01-01", end_date="2024-01-02"
        )
        assert report['total_trades'] == 1
        assert report['t_stat'] == 0.0
        assert report['ci_95_lower'] == 0.0


class TestT005PurgeGap:

    def test_purge_removes_end_of_train(self):
        import pandas as pd
        n = 1000
        timestamps = np.arange(n) * 900000
        sym_df = pd.DataFrame({'timestamp': timestamps})

        train_no_purge, test_no_purge = _compute_time_split(sym_df)
        train_purge, test_purge = _compute_time_split(sym_df, purge_bars=24)

        assert len(train_purge) == len(train_no_purge) - 24
        assert len(test_purge) == len(test_no_purge)

    def test_purge_with_date_split(self):
        import pandas as pd
        from datetime import datetime, timezone

        n = 2000
        base_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        timestamps = base_ts + np.arange(n) * 900000
        sym_df = pd.DataFrame({'timestamp': timestamps})

        train_no_purge, test_no_purge = _compute_time_split(
            sym_df, train_end_date="2024-01-15",
            test_start_date="2024-01-15"
        )
        train_purge, test_purge = _compute_time_split(
            sym_df, train_end_date="2024-01-15",
            test_start_date="2024-01-15",
            purge_bars=24
        )

        assert len(train_purge) < len(train_no_purge)
        assert len(train_purge) == len(train_no_purge) - 24
        assert len(test_purge) == len(test_no_purge)

    def test_zero_purge_unchanged(self):
        import pandas as pd
        n = 500
        timestamps = np.arange(n) * 900000
        sym_df = pd.DataFrame({'timestamp': timestamps})

        train_a, test_a = _compute_time_split(sym_df, purge_bars=0)
        train_b, test_b = _compute_time_split(sym_df)

        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)


class TestT006SlippageDeduction:

    def test_slippage_reduces_scores(self):
        arrays = _make_arrays(n=200, seed=42)
        arrays['mu_R'][:] = 0.1
        arrays['mae'][:] = 0.05
        arrays['mfe'][:] = 0.15
        arrays['p_long'][:] = 0.7
        arrays['p_short'][:] = 0.3

        scores_no_slip, _, _ = compute_v5_scores(
            None, _arrays=arrays, slippage_bps=0.0
        )
        scores_with_slip, _, _ = compute_v5_scores(
            None, _arrays=arrays, slippage_bps=5.0
        )

        finite_no = scores_no_slip[np.isfinite(scores_no_slip)]
        finite_with = scores_with_slip[np.isfinite(scores_with_slip)]
        assert np.mean(finite_with) < np.mean(finite_no), \
            "Slippage should reduce average scores"

    def test_zero_slippage_unchanged(self):
        arrays = _make_arrays(n=200, seed=42)

        scores_a, sides_a, _ = compute_v5_scores(None, _arrays=arrays, slippage_bps=0.0)
        scores_b, sides_b, _ = compute_v5_scores(None, _arrays=arrays)

        np.testing.assert_array_almost_equal(scores_a, scores_b)
        np.testing.assert_array_equal(sides_a, sides_b)

    def test_high_slippage_kills_small_edge(self):
        arrays = _make_arrays(n=100, seed=42)
        arrays['mu_R'][:] = 0.005
        arrays['mae'][:] = 0.05
        arrays['mfe'][:] = 0.01
        arrays['p_long'][:] = 0.6
        arrays['p_short'][:] = 0.4

        scores_no_slip, _, _ = compute_v5_scores(
            None, _arrays=arrays, slippage_bps=0.0, min_mu_r_score=0.0
        )
        scores_big_slip, _, _ = compute_v5_scores(
            None, _arrays=arrays, slippage_bps=10.0, min_mu_r_score=0.0
        )

        finite_no = scores_no_slip[np.isfinite(scores_no_slip)]
        finite_slip = scores_big_slip[np.isfinite(scores_big_slip)]
        if len(finite_slip) > 0 and len(finite_no) > 0:
            assert np.mean(finite_slip) < np.mean(finite_no)

    def test_config_has_slippage(self):
        cfg = V5ForwardTestConfig(slippage_base_bps=2.5)
        assert cfg.slippage_base_bps == 2.5

    def test_default_slippage_is_bitget_roundtrip(self):
        cfg = V5ForwardTestConfig()
        assert cfg.slippage_base_bps == 6.0, (
            "Default slippage must be 6 bps (Bitget 3 bps taker × 2 sides), "
            f"got {cfg.slippage_base_bps}"
        )


class TestProductionReadinessAuditFixes:
    """Regression tests for bugs found in the Mar-2026 production-readiness audit.

    T_BUG1: edge_pass now applied in final_mask (quality gate mu_R filter live)
    T_BUG2: score_threshold default is 0.02 (matches live shared config)
    T_BUG3: slippage_base_bps default is 6.0 bps (tested in T006 class above)
    T_BUG4: _compute_score_decile_table detects monotonicity
    """

    def test_bug1_edge_pass_filters_low_mu_r_bars(self):
        """Bug 1: quality_mask must now filter bars with |mu_R| < p25 threshold."""
        n = 500
        rng = np.random.RandomState(0)
        arrays = {
            'mu_R':    rng.uniform(-0.001, 0.001, n).astype(np.float32),  # tiny mu_R, all near-zero
            'mae':     np.full(n, 0.1, dtype=np.float32),
            'sigma':   np.full(n, 0.1, dtype=np.float32),
            'p_trade': np.full(n, 0.7, dtype=np.float32),
        }
        ref = {
            'mu_R':    np.full(n, 0.5, dtype=np.float32),  # ref has large mu_R → p25 will be large
            'mae':     np.full(n, 0.01, dtype=np.float32),
            'sigma':   np.full(n, 0.01, dtype=np.float32),
            'p_trade': np.full(n, 0.99, dtype=np.float32),
        }
        cfg = V5QualityGateConfig(mu_R_min=0.05)
        mask, diag = v5_quality_mask(arrays, cfg, epoch=999, ref_arrays=ref)
        n_passed = int(np.sum(mask))
        assert n_passed < n * 0.25, (
            f"edge_pass should filter most low-mu_R bars (only {n_passed}/{n} passed). "
            "If this test fails, edge_pass is still not applied in final_mask."
        )

    def test_bug2_score_threshold_default_matches_live(self):
        """Bug 2: V5ForwardTestConfig.score_threshold must match shared_v5_trade_config default."""
        from config.shared_v5_trade_config import V5TradeDefaults
        live_default = V5TradeDefaults().score_threshold
        backtest_default = V5ForwardTestConfig().score_threshold
        assert backtest_default == live_default, (
            f"score_threshold mismatch: backtest default={backtest_default}, "
            f"live default={live_default}. Backtest metrics will be optimistic."
        )

    def test_bug4_score_decile_table_monotonic(self):
        """Bug 4: _compute_score_decile_table returns monotonic=True on perfectly sorted data."""
        from train.v5_train import _compute_score_decile_table
        n = 200
        scores = np.linspace(0.0, 1.0, n)
        realized_r = scores * 2.0 - 0.5  # perfectly correlated with scores
        rows, monotonic = _compute_score_decile_table(scores, realized_r, n_deciles=5)
        assert len(rows) == 5
        assert monotonic is True, "Perfectly sorted scores should give monotonic=True"

    def test_bug4_score_decile_table_non_monotonic(self):
        """Bug 4: _compute_score_decile_table returns monotonic=False when top decile underperforms."""
        from train.v5_train import _compute_score_decile_table
        n = 200
        scores = np.linspace(0.0, 1.0, n)
        realized_r = np.full(n, 0.1)
        realized_r[int(n * 0.9):] = -0.5  # top decile is a loser
        rows, monotonic = _compute_score_decile_table(scores, realized_r, n_deciles=5)
        assert monotonic is False, "Top-decile underperformance should give monotonic=False"

    def test_bug4_score_decile_table_too_few_trades(self):
        """Bug 4: _compute_score_decile_table returns empty list when too few trades."""
        from train.v5_train import _compute_score_decile_table
        scores = np.array([0.1, 0.5, 0.9])
        r = np.array([0.1, 0.2, 0.3])
        rows, monotonic = _compute_score_decile_table(scores, r, n_deciles=10)
        assert rows == [] and monotonic is False

    def test_finding5_entry_lag_reduces_r(self):
        """Finding 5: entry_lag_atr_fraction > 0 should reduce avg realized R vs lag=0."""
        from data.common import generate_v5_sweep_outcomes
        import pandas as pd
        rng = np.random.RandomState(42)
        n = 500
        close = np.cumprod(1 + rng.randn(n) * 0.01) * 100
        df = pd.DataFrame({
            'open':  close,
            'high':  close * 1.01,
            'low':   close * 0.99,
            'close': close,
            'volume': np.ones(n) * 1000,
        })
        res_no_lag = generate_v5_sweep_outcomes(df, entry_lag_atr_fraction=0.0)
        res_with_lag = generate_v5_sweep_outcomes(df, entry_lag_atr_fraction=0.5)

        finite_no_lag = res_no_lag['r_long'][np.isfinite(res_no_lag['r_long'])]
        finite_with_lag = res_with_lag['r_long'][np.isfinite(res_with_lag['r_long'])]

        assert np.mean(finite_with_lag) < np.mean(finite_no_lag), (
            "entry_lag should reduce avg realized R for LONG trades "
            f"(no_lag={np.mean(finite_no_lag):.4f}, with_lag={np.mean(finite_with_lag):.4f})"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
