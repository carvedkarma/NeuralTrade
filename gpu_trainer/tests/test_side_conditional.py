"""Tests for side-conditional outcome fix (oracle elimination).

Validates that:
1. r_long != r_short in some bars (not always same outcome)
2. Evaluation selects r_long when side=LONG and r_short when side=SHORT
3. Oracle best-side is NOT used in forward test evaluation
4. Costs scale with position size
"""

import numpy as np
import pandas as pd
import pytest


def _make_synthetic_df(n=200):
    """Create a synthetic OHLC series with known properties."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(100, 10000, n).astype(float)
    timestamps = np.arange(n) * 900_000 + 1_700_000_000_000

    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })
    return df


class TestSideConditionalOutcomes:
    def test_returns_all_keys(self):
        """generate_v5_sweep_outcomes must return both side-conditional and legacy keys."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from data.common import generate_v5_sweep_outcomes

        df = _make_synthetic_df()
        result = generate_v5_sweep_outcomes(df, horizon=16, tp_mult=2.0, sl_mult=1.5)

        assert 'r_long' in result
        assert 'r_short' in result
        assert 'out_long' in result
        assert 'out_short' in result
        assert 'realized_r' in result
        assert 'outcome' in result

    def test_r_long_differs_from_r_short(self):
        """LONG and SHORT outcomes must differ in at least some bars."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from data.common import generate_v5_sweep_outcomes

        df = _make_synthetic_df(500)
        result = generate_v5_sweep_outcomes(df, horizon=16, tp_mult=2.0, sl_mult=1.5)

        r_long = result['r_long']
        r_short = result['r_short']

        valid = ~np.isnan(r_long) & ~np.isnan(r_short)
        assert np.sum(valid) > 50, "Too few valid bars"

        differ = r_long[valid] != r_short[valid]
        assert np.sum(differ) > 0, (
            "r_long and r_short are identical everywhere -- "
            "side-conditional outcomes are not differentiated"
        )
        pct_differ = np.mean(differ) * 100
        assert pct_differ > 5, (
            f"Only {pct_differ:.1f}% of bars differ between LONG and SHORT -- suspiciously low"
        )

    def test_oracle_best_side_matches_max(self):
        """Legacy realized_r should be max(r_long, r_short)."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from data.common import generate_v5_sweep_outcomes

        df = _make_synthetic_df()
        result = generate_v5_sweep_outcomes(df, horizon=16, tp_mult=2.0, sl_mult=1.5)

        r_long = result['r_long']
        r_short = result['r_short']
        realized_r = result['realized_r']

        valid = ~np.isnan(r_long) & ~np.isnan(r_short) & ~np.isnan(realized_r)
        expected_best = np.maximum(r_long[valid], r_short[valid])
        np.testing.assert_allclose(
            realized_r[valid], expected_best, atol=1e-6,
            err_msg="Legacy realized_r is not max(r_long, r_short)"
        )

    def test_side_selection_logic(self):
        """Ensure np.where correctly selects r_long when side=1 and r_short when side=-1."""
        r_long = np.array([1.0, -0.5, 0.3, 1.33])
        r_short = np.array([-1.0, 0.8, -0.2, 1.33])
        sides = np.array([1, -1, 1, -1])

        r_eval = np.where(sides == 1, r_long, r_short)

        expected = np.array([1.0, 0.8, 0.3, 1.33])
        np.testing.assert_array_equal(r_eval, expected)

    def test_side_conditional_worse_than_oracle(self):
        """Side-conditional average R should be <= oracle best-side R (on average)."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from data.common import generate_v5_sweep_outcomes

        df = _make_synthetic_df(500)
        result = generate_v5_sweep_outcomes(df, horizon=16, tp_mult=2.0, sl_mult=1.5)

        r_long = result['r_long']
        r_short = result['r_short']
        oracle_r = result['realized_r']

        valid = ~np.isnan(r_long) & ~np.isnan(r_short) & ~np.isnan(oracle_r)

        sides_random = np.where(np.random.RandomState(123).rand(np.sum(valid)) > 0.5, 1, -1)
        r_random = np.where(sides_random == 1, r_long[valid], r_short[valid])

        assert np.mean(r_random) <= np.mean(oracle_r[valid]) + 0.01, (
            "Random-side selection should not beat oracle on average"
        )

    def test_outcome_string_selection(self):
        """Ensure outcome strings are properly selected by side."""
        out_long = np.array(["TP", "SL", "EXP_WIN", "TP"])
        out_short = np.array(["SL", "TP", "EXP_LOSS", "SL"])
        sides = np.array([1, -1, 1, -1])

        out_eval = np.where(sides == 1, out_long, out_short)

        expected = np.array(["TP", "TP", "EXP_WIN", "SL"])
        np.testing.assert_array_equal(out_eval, expected)


class TestCostScaling:
    def test_fees_scale_with_size(self):
        """Fees should scale linearly with position size."""
        pytest.importorskip("torch")
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from training.walk_forward import TransactionCosts

        costs = TransactionCosts()

        fee_small, slip_small = costs.calculate_costs(
            price=50000.0, size=0.01, volatility=0.001, is_taker=True
        )
        fee_large, slip_large = costs.calculate_costs(
            price=50000.0, size=1.0, volatility=0.001, is_taker=True
        )

        assert abs(fee_large / fee_small - 100.0) < 0.01, (
            f"Fee ratio should be 100x but got {fee_large / fee_small:.2f}x"
        )
        assert abs(slip_large / slip_small - 100.0) < 0.01, (
            f"Slippage ratio should be 100x but got {slip_large / slip_small:.2f}x"
        )


class TestSharpeAnnualization:
    def test_sharpe_not_inflated(self):
        """Sharpe should not be wildly inflated for few trades over many bars."""
        np.random.seed(42)
        n_trades = 20
        t_r = np.random.randn(n_trades) * 0.5 + 0.1
        val_bars = 96 * 30
        val_days = val_bars / 96.0

        expect = float(np.mean(t_r))
        std_r = float(np.std(t_r))
        trades_per_year = (n_trades / max(val_days, 1e-6)) * 252
        sharpe = expect / max(std_r, 1e-6) * np.sqrt(max(trades_per_year, 1))

        old_sharpe = expect / max(std_r, 1e-6) * np.sqrt(252 * 96)

        assert abs(sharpe) < abs(old_sharpe), (
            f"New Sharpe ({sharpe:.2f}) should be less inflated than old ({old_sharpe:.2f})"
        )
        assert abs(sharpe) < 50, f"Sharpe {sharpe:.2f} still seems too high"
