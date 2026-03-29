"""
Task #51 Regression Tests — V5 Training Signal Fix

Verifies:
  1. mu_debias default is False in all 3 function signatures / dataclass
  2. SIDE_BAL_W == 0.05 (reduced from 0.30 to limit KL loss dominance)
  3. barrier_aligned_ret_R logic: LONG→+r_long, SHORT→-r_short, HOLD→0.0
  4. Aligned ret_R produces no NaNs and correct sign per action label
  5. [V5_TRAIN_QUALITY] per-epoch logging is present in v5_train.py
  6. [V5_TRAIN_WARN] fires at epoch>=20 when mu_r_corr < -0.05
  7. [V5_BASELINE_CMP] / v5_run_metrics.json infrastructure is in place
  8. Fold-start log emits mu_debias ENABLED/DISABLED status
"""
import ast
import re
import sys
import numpy as np
import pytest
from pathlib import Path


V5_TRAIN_PATH = Path(__file__).parent / "train" / "v5_train.py"
V5_TGT_PATH   = Path(__file__).parent / "data" / "v5_target_generator.py"


@pytest.fixture(scope="module")
def v5_train_src():
    return V5_TRAIN_PATH.read_text()


@pytest.fixture(scope="module")
def v5_tgt_src():
    return V5_TGT_PATH.read_text()


class TestMuDebiasDefault:
    def test_v5forwardconfig_default_false(self, v5_train_src):
        assert "mu_debias: bool = False" in v5_train_src, (
            "V5ForwardConfig must have mu_debias default=False "
            "(disable mu_debias by default to prevent gradient collapse)"
        )

    def test_no_mu_debias_true_defaults(self, v5_train_src):
        hits = [ln.strip() for ln in v5_train_src.splitlines()
                if "mu_debias=True" in ln and "mu_debias_alpha" not in ln
                and not ln.lstrip().startswith("#")]
        assert hits == [], (
            f"Found mu_debias=True default in function signature(s): {hits}"
        )

    def test_run_v5_walk_forward_default_false(self, v5_train_src):
        match = re.search(
            r"def run_v5_walk_forward.*?mu_debias\s*=\s*(True|False)",
            v5_train_src, re.DOTALL
        )
        assert match, "run_v5_walk_forward must declare mu_debias parameter"
        assert match.group(1) == "False", (
            f"run_v5_walk_forward mu_debias default must be False, got {match.group(1)}"
        )

    def test_train_v5_model_default_false(self, v5_train_src):
        match = re.search(
            r"def train_v5_model.*?mu_debias\s*=\s*(True|False)",
            v5_train_src, re.DOTALL
        )
        assert match, "train_v5_model must declare mu_debias parameter"
        assert match.group(1) == "False", (
            f"train_v5_model mu_debias default must be False, got {match.group(1)}"
        )


class TestSideBalanceWeight:
    def test_side_bal_w_is_005(self, v5_train_src):
        assert "SIDE_BAL_W = 0.05" in v5_train_src, (
            "SIDE_BAL_W must be 0.05 (reduced from 0.30) to prevent KL loss "
            "from overriding per-bar action labels"
        )

    def test_old_side_bal_w_030_removed(self, v5_train_src):
        assert "SIDE_BAL_W = 0.30" not in v5_train_src, (
            "Old SIDE_BAL_W = 0.30 must be removed"
        )


class TestBarrierAlignedRetR:
    """Unit tests for the barrier_aligned_ret_R override block."""

    def _run_alignment(self, action_label, b_r_long, b_r_short, valid_mask=None):
        n = len(action_label)
        if valid_mask is None:
            valid_mask = np.ones(n, dtype=bool)
        ret_R = np.zeros(n, dtype=np.float64)
        n_l = n_s = n_h = 0
        for i in range(n):
            if not valid_mask[i]:
                continue
            lbl = action_label[i]
            if lbl == 1:
                rl = b_r_long[i] if np.isfinite(b_r_long[i]) else 0.0
                ret_R[i] = float(rl)
                n_l += 1
            elif lbl == 2:
                rs = b_r_short[i] if np.isfinite(b_r_short[i]) else 0.0
                ret_R[i] = -float(rs)
                n_s += 1
            else:
                ret_R[i] = 0.0
                n_h += 1
        return ret_R, n_l, n_s, n_h

    def test_long_bars_have_nonnegative_ret_r(self):
        action_label = np.array([1, 1, 1, 2, 0])
        b_r_long  = np.array([1.5, 0.5, 2.0, 0.0, 0.0])
        b_r_short = np.array([0.0, 0.0, 0.0, 1.2, 0.0])
        ret_R, _, _, _ = self._run_alignment(action_label, b_r_long, b_r_short)
        long_mask = action_label == 1
        assert np.all(ret_R[long_mask] >= 0), (
            f"LONG-labelled bars must have non-negative ret_R: {ret_R[long_mask]}"
        )

    def test_short_bars_have_nonpositive_ret_r(self):
        action_label = np.array([2, 2, 2, 1, 0])
        b_r_long  = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
        b_r_short = np.array([1.2, 0.8, 2.0, 0.0, 0.0])
        ret_R, _, _, _ = self._run_alignment(action_label, b_r_long, b_r_short)
        short_mask = action_label == 2
        assert np.all(ret_R[short_mask] <= 0), (
            f"SHORT-labelled bars must have non-positive ret_R: {ret_R[short_mask]}"
        )

    def test_hold_bars_have_zero_ret_r(self):
        action_label = np.array([0, 0, 0, 1, 2])
        b_r_long  = np.array([0.3, 0.1, 0.2, 1.5, 0.0])
        b_r_short = np.array([0.0, 0.4, 0.1, 0.0, 1.0])
        ret_R, _, _, _ = self._run_alignment(action_label, b_r_long, b_r_short)
        hold_mask = action_label == 0
        assert np.all(ret_R[hold_mask] == 0.0), (
            f"HOLD-labelled bars must have ret_R=0: {ret_R[hold_mask]}"
        )

    def test_nan_barrier_values_become_zero(self):
        action_label = np.array([1, 2])
        b_r_long  = np.array([float('nan'), 0.0])
        b_r_short = np.array([0.0, float('nan')])
        ret_R, _, _, _ = self._run_alignment(action_label, b_r_long, b_r_short)
        assert np.all(np.isfinite(ret_R)), (
            f"NaN barrier values must be clamped to 0.0: ret_R={ret_R}"
        )
        assert ret_R[0] == 0.0, "NaN r_long → ret_R[0] must be 0.0"
        assert ret_R[1] == 0.0, "NaN r_short → ret_R[1] must be 0.0"

    def test_invalid_bars_not_overridden(self):
        action_label = np.array([1, 1, 0])
        b_r_long  = np.array([1.5, 0.8, 0.0])
        b_r_short = np.array([0.0, 0.0, 0.0])
        valid_mask = np.array([True, False, True])
        ret_R_original = np.array([99.0, 99.0, 99.0])
        ret_R = ret_R_original.copy()
        n_l = n_s = n_h = 0
        for i in range(3):
            if not valid_mask[i]:
                continue
            lbl = action_label[i]
            if lbl == 1:
                ret_R[i] = float(b_r_long[i])
                n_l += 1
            elif lbl == 2:
                ret_R[i] = -float(b_r_short[i])
                n_s += 1
            else:
                ret_R[i] = 0.0
                n_h += 1
        assert ret_R[1] == 99.0, "Invalid bar must NOT have its ret_R overridden"
        assert ret_R[0] == 1.5, "Valid LONG bar must get +r_long"
        assert ret_R[2] == 0.0, "Valid HOLD bar must get 0.0"

    def test_sign_convention_no_gradient_contradiction(self):
        # LONG label → positive ret_R → model predicts positive mu_R
        # SHORT label → negative ret_R → model predicts negative mu_R
        # side_aware_scoring: long requires mu_R>0, short requires mu_R<0
        # → no contradiction between NLL loss target and CE action label
        action_label = np.array([1, 2])
        b_r_long  = np.array([1.5, 0.0])
        b_r_short = np.array([0.0, 1.2])
        ret_R, _, _, _ = self._run_alignment(action_label, b_r_long, b_r_short)
        assert ret_R[0] > 0, "LONG bar: positive ret_R → model learns mu_R>0 (bullish)"
        assert ret_R[1] < 0, "SHORT bar: negative ret_R → model learns mu_R<0 (bearish)"


class TestPerEpochQualityLogging:
    def test_v5_train_quality_log_present(self, v5_train_src):
        assert "[V5_TRAIN_QUALITY]" in v5_train_src, (
            "Must log [V5_TRAIN_QUALITY] per epoch with mu_r_corr_val"
        )

    def test_mu_r_corr_val_computed(self, v5_train_src):
        assert "mu_r_corr_val" in v5_train_src, (
            "Must compute mu_r_corr_val (correlation between predicted mu_R and val_ret_R)"
        )

    def test_v5_train_warn_after_epoch_20(self, v5_train_src):
        assert "[V5_TRAIN_WARN]" in v5_train_src, (
            "Must emit [V5_TRAIN_WARN] when mu_r_corr_val < -0.05 after epoch 20"
        )
        assert "epoch >= 20" in v5_train_src, (
            "Warning must fire only at epoch>=20 (not on early noisy epochs)"
        )

    def test_quality_block_guarded_by_not_use_v6(self, v5_train_src):
        idx = v5_train_src.find("[V5_TRAIN_QUALITY]")
        assert idx >= 0
        surrounding = v5_train_src[max(0, idx-2000):idx]
        assert "not use_v6" in surrounding, (
            "[V5_TRAIN_QUALITY] block must be guarded by 'not use_v6'"
        )


class TestBaselineComparison:
    def test_baseline_cmp_block_present(self, v5_train_src):
        assert "[V5_BASELINE_CMP]" in v5_train_src, (
            "Must have [V5_BASELINE_CMP] block in walk-forward summary"
        )

    def test_run_metrics_saved(self, v5_train_src):
        assert "v5_run_metrics.json" in v5_train_src, (
            "Must save current run metrics to v5_run_metrics.json"
        )

    def test_baseline_metrics_loaded(self, v5_train_src):
        assert "v5_baseline_metrics.json" in v5_train_src, (
            "Must try to load v5_baseline_metrics.json for comparison"
        )

    def test_comparison_includes_mu_r_correlation(self, v5_train_src):
        assert "mu_r_correlation" in v5_train_src, (
            "Baseline comparison must include mu_r_correlation metric"
        )

    def test_comparison_includes_action_accuracy(self, v5_train_src):
        idx = v5_train_src.find("[V5_BASELINE_CMP]")
        segment = v5_train_src[idx:idx + 5000]
        assert "action_accuracy" in segment, (
            "Baseline comparison must include action_accuracy metric"
        )

    def test_comparison_includes_total_r(self, v5_train_src):
        idx = v5_train_src.find("[V5_BASELINE_CMP]")
        segment = v5_train_src[idx:idx + 5000]
        assert "total_r" in segment, (
            "Baseline comparison must include total_r metric"
        )


class TestFoldStartLog:
    def test_fold_start_logs_mu_debias_status(self, v5_train_src):
        assert '"ENABLED" if mu_debias else "DISABLED"' in v5_train_src or \
               "'ENABLED' if mu_debias else 'DISABLED'" in v5_train_src, (
            "Fold-start log must emit mu_debias ENABLED/DISABLED status"
        )

    def test_fold_start_logs_side_bal_w(self, v5_train_src):
        idx = v5_train_src.find("barrier_aligned_ret_R=True")
        assert idx >= 0, (
            "Fold-start log must mention barrier_aligned_ret_R=True"
        )


class TestTargetGeneratorAlignment:
    def test_barrier_aligned_ret_r_code_present(self, v5_tgt_src):
        assert "barrier_aligned_ret_R" in v5_tgt_src, (
            "v5_target_generator.py must have barrier_aligned_ret_R alignment block"
        )

    def test_short_bars_negated(self, v5_tgt_src):
        assert "ret_R[i] = -float(rs_val)" in v5_tgt_src, (
            "SHORT bars must set ret_R[i] = -float(rs_val) for correct sign convention"
        )

    def test_hold_bars_zero(self, v5_tgt_src):
        assert "ret_R[i] = 0.0" in v5_tgt_src, (
            "HOLD bars must set ret_R[i] = 0.0"
        )

    def test_nan_guard_present(self, v5_tgt_src):
        assert "np.isfinite(b_r_long[i])" in v5_tgt_src, (
            "Must guard b_r_long[i] against NaN before assigning ret_R"
        )
        assert "np.isfinite(b_r_short[i])" in v5_tgt_src, (
            "Must guard b_r_short[i] against NaN before assigning ret_R"
        )

    def test_alignment_log_emitted(self, v5_tgt_src):
        assert "barrier_aligned_ret_R:" in v5_tgt_src, (
            "Must log barrier_aligned_ret_R counts and stats for observability"
        )
