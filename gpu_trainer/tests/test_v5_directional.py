"""Tests for v5.3.1 directional balance fixes: penalty debiasing, side-balance loss, mu_R debiasing, diagnostics.

T001: Directional penalty scales with edge magnitude, independent of mu_R sign
T002: Side-balance regularization has gradients
T003: mu_R debiasing via EMA centers constant streams
T004: Diagnostics produce correct side distributions
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


class TestT003MuDebias:

    def test_ema_centers_constant_stream(self):
        ema = 0.0
        alpha = 0.01
        for _ in range(5000):
            mu = 0.5
            ema = (1 - alpha) * ema + alpha * mu
            mu_debiased = mu - ema
        assert abs(mu_debiased) < 0.05, f"Debiased should be near zero: {mu_debiased}"

    def test_ema_centers_negative_stream(self):
        ema = 0.0
        alpha = 0.01
        for _ in range(5000):
            mu = -0.3
            ema = (1 - alpha) * ema + alpha * mu
            mu_debiased = mu - ema
        assert abs(mu_debiased) < 0.05, f"Debiased should be near zero: {mu_debiased}"

    def test_ema_preserves_variation(self):
        np.random.seed(42)
        ema = 0.0
        alpha = 0.01
        raw = np.random.randn(1000) * 0.1 + 0.5
        debiased = np.zeros(1000)
        for i in range(1000):
            ema = (1 - alpha) * ema + alpha * raw[i]
            debiased[i] = raw[i] - ema

        assert np.std(debiased) > 0.05, "Debiasing should preserve signal variation"
        assert abs(np.mean(debiased[-200:])) < abs(np.mean(raw[-200:])), \
            "Debiased tail mean should be closer to zero than raw"

    def test_per_symbol_debiasing(self):
        n = 2000
        mu_R = np.zeros(n)
        sym_ids = np.zeros(n, dtype=int)
        mu_R[:1000] = 0.5
        sym_ids[:1000] = 0
        mu_R[1000:] = -0.3
        sym_ids[1000:] = 1

        alpha = 0.01
        debiased = mu_R.copy()
        for sym_id in [0, 1]:
            mask = sym_ids == sym_id
            indices = np.where(mask)[0]
            ema_val = 0.0
            for i in indices:
                ema_val = (1 - alpha) * ema_val + alpha * mu_R[i]
                debiased[i] = mu_R[i] - ema_val

        sym0_mean = np.mean(debiased[sym_ids == 0][-100:])
        sym1_mean = np.mean(debiased[sym_ids == 1][-100:])
        assert abs(sym0_mean) < 0.1, f"Symbol 0 should be debiased: {sym0_mean}"
        assert abs(sym1_mean) < 0.1, f"Symbol 1 should be debiased: {sym1_mean}"

    def test_zero_stream_unchanged(self):
        ema = 0.0
        alpha = 0.01
        for _ in range(100):
            mu = 0.0
            ema = (1 - alpha) * ema + alpha * mu
            mu_debiased = mu - ema
        assert abs(mu_debiased) < 1e-10, f"Zero stream should stay zero: {mu_debiased}"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestT001DirectionalPenalty:

    def test_penalty_independent_of_mu_sign_action_head(self):
        from train.v5_train import compute_v5_scores
        n = 100
        arrays_pos = {
            'mu_R': np.full(n, 0.5),
            'mae': np.full(n, 0.3),
            'mfe': np.full(n, 0.8),
            'p_long': np.full(n, 0.7),
            'p_short': np.full(n, 0.3),
        }
        arrays_neg = {
            'mu_R': np.full(n, -0.5),
            'mae': np.full(n, 0.3),
            'mfe': np.full(n, 0.8),
            'p_long': np.full(n, 0.7),
            'p_short': np.full(n, 0.3),
        }
        scores_pos, sides_pos, _ = compute_v5_scores(
            None, _arrays=arrays_pos, side_mode='action_head', score_lambda=1.0)
        scores_neg, sides_neg, _ = compute_v5_scores(
            None, _arrays=arrays_neg, side_mode='action_head', score_lambda=1.0)

        assert np.allclose(scores_pos, scores_neg, atol=1e-6), \
            f"Scores differ for pos/neg mu_R: {scores_pos[0]:.6f} vs {scores_neg[0]:.6f}"

    def test_penalty_equal_for_long_and_short_with_same_confidence(self):
        from train.v5_train import compute_v5_scores
        n = 50
        arrays_long = {
            'mu_R': np.full(n, 0.4),
            'mae': np.full(n, 0.2),
            'mfe': np.full(n, 0.6),
            'p_long': np.full(n, 0.8),
            'p_short': np.full(n, 0.2),
        }
        arrays_short = {
            'mu_R': np.full(n, -0.4),
            'mae': np.full(n, 0.2),
            'mfe': np.full(n, 0.6),
            'p_long': np.full(n, 0.2),
            'p_short': np.full(n, 0.8),
        }
        scores_long, sides_long, _ = compute_v5_scores(
            None, _arrays=arrays_long, side_mode='action_head', score_lambda=0.5)
        scores_short, sides_short, _ = compute_v5_scores(
            None, _arrays=arrays_short, side_mode='action_head', score_lambda=0.5)

        assert np.all(sides_long == 1)
        assert np.all(sides_short == -1)
        assert np.allclose(scores_long, scores_short, atol=1e-6), \
            f"Scores differ for equal-confidence L/S: {scores_long[0]:.6f} vs {scores_short[0]:.6f}"

    def test_higher_conviction_gets_lower_penalty(self):
        from train.v5_train import compute_v5_scores
        n = 50
        arrays_high = {
            'mu_R': np.full(n, 0.3),
            'mae': np.full(n, 0.2),
            'mfe': np.full(n, 0.5),
            'p_long': np.full(n, 0.9),
            'p_short': np.full(n, 0.1),
        }
        arrays_low = {
            'mu_R': np.full(n, 0.3),
            'mae': np.full(n, 0.2),
            'mfe': np.full(n, 0.5),
            'p_long': np.full(n, 0.55),
            'p_short': np.full(n, 0.45),
        }
        scores_high, _, _ = compute_v5_scores(
            None, _arrays=arrays_high, side_mode='action_head', score_lambda=1.0)
        scores_low, _, _ = compute_v5_scores(
            None, _arrays=arrays_low, side_mode='action_head', score_lambda=1.0)

        assert float(np.mean(scores_high)) > float(np.mean(scores_low)), \
            "High conviction should get higher score"

    def test_penalty_scales_with_mu_over_risk(self):
        from train.v5_train import compute_v5_scores
        n = 50
        arrays_big = {
            'mu_R': np.full(n, 1.0),
            'mae': np.full(n, 0.5),
            'mfe': np.full(n, 1.0),
            'p_long': np.full(n, 0.6),
            'p_short': np.full(n, 0.4),
        }
        arrays_small = {
            'mu_R': np.full(n, 0.1),
            'mae': np.full(n, 0.05),
            'mfe': np.full(n, 0.1),
            'p_long': np.full(n, 0.6),
            'p_short': np.full(n, 0.4),
        }
        scores_big, _, diag_big = compute_v5_scores(
            None, _arrays=arrays_big, side_mode='action_head', score_lambda=0.5)
        scores_small, _, diag_small = compute_v5_scores(
            None, _arrays=arrays_small, side_mode='action_head', score_lambda=0.5)

        ratio = scores_big[0] / scores_small[0]
        assert abs(ratio - 1.0) < 0.01, \
            f"Score ratio should be ~1.0 (same mu/risk ratio): {ratio:.4f}"

    def test_score_positive_when_p_side_above_breakeven(self):
        from train.v5_train import compute_v5_scores
        n = 50
        lam = 0.5
        breakeven = lam / (1.0 + lam)
        p_side_val = breakeven + 0.1
        arrays = {
            'mu_R': np.full(n, 0.3),
            'mae': np.full(n, 0.2),
            'mfe': np.full(n, 0.5),
            'p_long': np.full(n, p_side_val),
            'p_short': np.full(n, 1.0 - p_side_val),
        }
        scores, sides, _ = compute_v5_scores(
            None, _arrays=arrays, side_mode='action_head', score_lambda=lam)

        assert np.all(sides == 1)
        assert np.all(scores > 0), \
            f"Score should be positive when p_side={p_side_val:.3f} > breakeven={breakeven:.3f}: {scores[0]:.6f}"

    def test_score_negative_when_p_side_below_breakeven(self):
        from train.v5_train import compute_v5_scores
        n = 50
        lam = 0.5
        breakeven = lam / (1.0 + lam)
        p_side_val = breakeven - 0.1
        arrays = {
            'mu_R': np.full(n, 0.3),
            'mae': np.full(n, 0.2),
            'mfe': np.full(n, 0.5),
            'p_long': np.full(n, p_side_val),
            'p_short': np.full(n, 1.0 - p_side_val),
        }
        scores, sides, _ = compute_v5_scores(
            None, _arrays=arrays, side_mode='action_head', score_lambda=lam)

        assert np.all(scores < 0), \
            f"Score should be negative when p_side={p_side_val:.3f} < breakeven={breakeven:.3f}: {scores[0]:.6f}"

    def test_score_zero_when_mu_zero(self):
        from train.v5_train import compute_v5_scores
        n = 50
        arrays = {
            'mu_R': np.full(n, 0.0),
            'mae': np.full(n, 0.3),
            'mfe': np.full(n, 0.5),
            'p_long': np.full(n, 0.7),
            'p_short': np.full(n, 0.3),
        }
        scores, _, _ = compute_v5_scores(
            None, _arrays=arrays, side_mode='action_head', score_lambda=0.5)

        finite_scores = scores[np.isfinite(scores)]
        if len(finite_scores) > 0:
            assert np.allclose(finite_scores, 0.0, atol=1e-6), \
                f"Score should be zero when mu_R=0: {finite_scores[0]:.6f}"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestT002SideBalanceLoss:

    def test_side_balance_loss_has_gradient(self):
        import torch.nn.functional as F

        logits = torch.randn(64, 3, requires_grad=True)
        labels = torch.randint(0, 3, (64,))

        LONG_IDX, SHORT_IDX = 1, 2
        action_probs = F.softmax(logits, dim=-1)
        p_long_mean = action_probs[:, LONG_IDX].mean()
        p_short_mean = action_probs[:, SHORT_IDX].mean()

        is_long = (labels == LONG_IDX)
        is_short = (labels == SHORT_IDX)
        target_long = is_long.float().mean()
        target_short = is_short.float().mean()

        eps = 1e-8
        target_sum = target_long + target_short
        target_dist = torch.stack([target_long, target_short]) / (target_sum + eps)
        pred_dist = torch.stack([p_long_mean, p_short_mean])
        pred_dist = pred_dist / (pred_dist.sum() + eps)

        loss = F.kl_div(
            (pred_dist + eps).log(),
            target_dist.detach(),
            reduction="batchmean"
        )
        loss = 0.1 * loss
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.abs().sum().item() > 0

    def test_side_balance_loss_nonzero_when_collapsed(self):
        import torch.nn.functional as F

        logits = torch.zeros(64, 3)
        logits[:, 1] = 5.0
        logits[:, 2] = -5.0
        labels = torch.zeros(64, dtype=torch.long)
        labels[:32] = 1
        labels[32:] = 2

        LONG_IDX, SHORT_IDX = 1, 2
        action_probs = F.softmax(logits, dim=-1)
        p_long_mean = action_probs[:, LONG_IDX].mean()
        p_short_mean = action_probs[:, SHORT_IDX].mean()

        is_long = (labels == LONG_IDX)
        is_short = (labels == SHORT_IDX)
        target_long = is_long.float().mean()
        target_short = is_short.float().mean()

        eps = 1e-8
        target_sum = target_long + target_short
        target_dist = torch.stack([target_long, target_short]) / (target_sum + eps)
        pred_dist = torch.stack([p_long_mean, p_short_mean])
        pred_dist = pred_dist / (pred_dist.sum() + eps)

        loss = F.kl_div(
            (pred_dist + eps).log(),
            target_dist.detach(),
            reduction="batchmean"
        )

        assert loss.item() > 0.01, f"Loss should be significant when predictions are collapsed: {loss.item()}"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestT004Diagnostics:

    def test_compute_side_distribution_balanced(self):
        from train.v5_train import _compute_side_distribution
        sides = np.array([1, 1, -1, -1, 1])
        indices = [0, 1, 2, 3, 4]
        dist = _compute_side_distribution(sides, indices)
        assert dist['long'] == 3
        assert dist['short'] == 2
        assert abs(dist['long_pct'] - 60.0) < 0.1
        assert abs(dist['short_pct'] - 40.0) < 0.1

    def test_compute_side_distribution_empty(self):
        from train.v5_train import _compute_side_distribution
        sides = np.array([1, -1])
        indices = []
        dist = _compute_side_distribution(sides, indices)
        assert dist['long'] == 0
        assert dist['short'] == 0

    def test_compute_side_distribution_all_long(self):
        from train.v5_train import _compute_side_distribution
        sides = np.array([1, 1, 1, 1])
        indices = [0, 1, 2, 3]
        dist = _compute_side_distribution(sides, indices)
        assert dist['long'] == 4
        assert dist['short'] == 0
        assert abs(dist['long_pct'] - 100.0) < 0.1


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestT005MinTradesGate:

    def test_min_trades_default_is_20(self):
        from train.v5_train import V5ForwardTestConfig
        cfg = V5ForwardTestConfig()
        assert cfg.min_trades == 20

    def test_min_trades_configurable(self):
        from train.v5_train import V5ForwardTestConfig
        cfg = V5ForwardTestConfig(min_trades=50)
        assert cfg.min_trades == 50

    def test_min_trades_zero_disables_gate(self):
        from train.v5_train import V5ForwardTestConfig
        cfg = V5ForwardTestConfig(min_trades=0)
        assert cfg.min_trades == 0


class TestT006ThresholdEMA:

    def test_ema_blending_formula(self):
        alpha = 0.5
        prev_ema = 0.20
        new_sweep = 0.02
        blended = alpha * new_sweep + (1 - alpha) * prev_ema
        assert abs(blended - 0.11) < 1e-6

    def test_ema_converges_to_stable_threshold(self):
        alpha = 0.5
        ema = None
        thresholds = [0.15, 0.15, 0.15, 0.15, 0.15]
        for t in thresholds:
            if ema is None:
                ema = t
            else:
                ema = alpha * t + (1 - alpha) * ema
        assert abs(ema - 0.15) < 1e-6

    def test_ema_smooths_outlier(self):
        alpha = 0.5
        ema = 0.20
        outlier = 0.02
        blended = alpha * outlier + (1 - alpha) * ema
        assert blended > outlier
        assert blended < ema
        assert abs(blended - 0.11) < 1e-6

    def test_ema_alpha_1_uses_only_new(self):
        alpha = 1.0
        ema = 0.20
        new_val = 0.05
        blended = alpha * new_val + (1 - alpha) * ema
        assert abs(blended - 0.05) < 1e-6

    def test_ema_alpha_0_uses_only_old(self):
        alpha = 0.0
        ema = 0.20
        new_val = 0.05
        blended = alpha * new_val + (1 - alpha) * ema
        assert abs(blended - 0.20) < 1e-6

    def test_ema_sequence_dampens_oscillation(self):
        alpha = 0.5
        ema = None
        thresholds = [0.30, 0.02, 0.30, 0.02, 0.30, 0.02]
        for t in thresholds:
            if ema is None:
                ema = t
            else:
                ema = alpha * t + (1 - alpha) * ema
        assert 0.10 < ema < 0.22
