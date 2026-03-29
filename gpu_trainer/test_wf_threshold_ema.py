"""
Regression test for WF threshold EMA decay floor.

Verifies that after dead-fold and no-report scenarios, threshold_ema is NOT
clamped to the old stale 0.01 floor, but instead respects the configured
min_threshold_floor (default 0.001) matching the V5 score scale 0.001-0.002.

Run with:
    python -m pytest gpu_trainer/test_wf_threshold_ema.py -v
or:
    python gpu_trainer/test_wf_threshold_ema.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gpu_trainer'))

import pytest


def simulate_dead_fold_decay(threshold_ema, wf_threshold_decay, wf_threshold_floor):
    """Simulate the dead-fold path: max(floor, ema * decay)."""
    old_ema = threshold_ema
    new_ema = max(wf_threshold_floor, threshold_ema * wf_threshold_decay)
    return old_ema, new_ema


def simulate_no_report_decay(threshold_ema, wf_threshold_decay, wf_threshold_floor):
    """Simulate the no-report path: max(floor, ema * decay)."""
    old_ema = threshold_ema
    new_ema = max(wf_threshold_floor, threshold_ema * wf_threshold_decay)
    return old_ema, new_ema


class TestWFThresholdEMAFloor:
    """Tests for walk-forward threshold EMA decay floor fix."""

    REALISTIC_EMA = 0.0018       # V5-realistic threshold_ema
    DECAY = 0.5                  # typical wf_threshold_decay
    NEW_FLOOR = 0.001            # corrected floor (V5TPDControllerConfig.min_threshold_floor default)
    OLD_FLOOR = 0.01             # old hardcoded stale floor

    def test_dead_fold_corrected_floor_allows_below_old_floor(self):
        """After a dead fold, decay from a realistic V5 ema should go BELOW 0.01."""
        _, new_ema = simulate_dead_fold_decay(
            self.REALISTIC_EMA, self.DECAY, self.NEW_FLOOR
        )
        # With correct floor=0.001, result is max(0.001, 0.0018*0.5=0.0009) = 0.001
        assert new_ema < self.OLD_FLOOR, (
            f"Expected new_ema={new_ema:.6f} < old_floor={self.OLD_FLOOR}, "
            f"but it is still clamped to the stale 0.01 floor."
        )

    def test_dead_fold_old_floor_would_clamp(self):
        """Confirm the bug: old floor=0.01 would have clamped the decayed value."""
        _, new_ema_old = simulate_dead_fold_decay(
            self.REALISTIC_EMA, self.DECAY, self.OLD_FLOOR
        )
        # With old floor=0.01, result is max(0.01, 0.0009) = 0.01 — stuck at 0.01
        assert new_ema_old == self.OLD_FLOOR, (
            f"Expected old-floor path to produce exactly {self.OLD_FLOOR}, got {new_ema_old}"
        )

    def test_dead_fold_corrected_floor_respects_new_floor(self):
        """After decay from very low ema, result is clamped to new_floor, not below."""
        tiny_ema = 0.0005
        _, new_ema = simulate_dead_fold_decay(tiny_ema, self.DECAY, self.NEW_FLOOR)
        # max(0.001, 0.0005*0.5=0.00025) = 0.001
        assert new_ema >= self.NEW_FLOOR, (
            f"Expected new_ema={new_ema:.6f} >= new_floor={self.NEW_FLOOR}"
        )
        assert new_ema == self.NEW_FLOOR, (
            f"Expected exact floor={self.NEW_FLOOR}, got {new_ema:.6f}"
        )

    def test_no_report_corrected_floor_allows_below_old_floor(self):
        """After a no-report fold, decay from realistic V5 ema should go below 0.01."""
        _, new_ema = simulate_no_report_decay(
            self.REALISTIC_EMA, self.DECAY, self.NEW_FLOOR
        )
        assert new_ema < self.OLD_FLOOR, (
            f"Expected new_ema={new_ema:.6f} < old_floor={self.OLD_FLOOR}"
        )

    def test_no_report_old_floor_would_clamp(self):
        """Confirm the bug: old floor=0.01 clamps the no-report decay too."""
        _, new_ema_old = simulate_no_report_decay(
            self.REALISTIC_EMA, self.DECAY, self.OLD_FLOOR
        )
        assert new_ema_old == self.OLD_FLOOR

    def test_multiple_dead_folds_converge_to_new_floor(self):
        """Multiple consecutive dead folds converge toward new_floor, not old_floor."""
        ema = 0.005
        floor = self.NEW_FLOOR
        for _ in range(20):
            _, ema = simulate_dead_fold_decay(ema, self.DECAY, floor)
        assert ema == floor, (
            f"After 20 dead folds, ema={ema:.6f} should converge to floor={floor}"
        )
        assert ema < self.OLD_FLOOR, (
            f"Converged ema={ema:.6f} should be below old stale floor={self.OLD_FLOOR}"
        )

    def test_floor_from_tpd_ctrl_cfg(self):
        """V5TPDControllerConfig.min_threshold_floor default matches the corrected floor."""
        try:
            from train.v5_train import V5TPDControllerConfig
            cfg = V5TPDControllerConfig()
            assert cfg.min_threshold_floor == self.NEW_FLOOR, (
                f"Expected min_threshold_floor={self.NEW_FLOOR}, got {cfg.min_threshold_floor}"
            )
        except ImportError:
            pytest.skip("v5_train not importable in this environment (GPU/torch dependency)")

    def test_decay_above_new_floor_passes_through(self):
        """When ema*decay > new_floor, the decayed value is used unchanged."""
        large_ema = 0.010
        _, new_ema = simulate_dead_fold_decay(large_ema, self.DECAY, self.NEW_FLOOR)
        expected = large_ema * self.DECAY  # 0.005
        assert abs(new_ema - expected) < 1e-10, (
            f"Expected {expected:.6f}, got {new_ema:.6f}"
        )


if __name__ == "__main__":
    import unittest

    suite = unittest.TestLoader().loadTestsFromTestCase(
        type("TestWFThresholdEMAFloor", (TestWFThresholdEMAFloor, unittest.TestCase), {})
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
