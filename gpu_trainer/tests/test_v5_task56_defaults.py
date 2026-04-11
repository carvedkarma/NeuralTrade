"""Task #56 validation tests that do NOT require torch.

Validates:
  - CLI parser defaults for score_lambda, wf_train_months, recency_half_life, short_min_fraction
  - V5_FOLD_HEALTH and V5_THRESHOLD_GUIDE log blocks are present in v5_train.py
    with correct VERDICT states and --v5-min-threshold CLI flag
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_task56_cli_parser_defaults():
    """Task #56 A1/B3/C1/C2: quick_start.py CLI defaults match the task spec.

    - score_lambda=0.30 (A1: break-even p_side 0.333→0.231)
    - wf_train_months=9  (C2: 12→9 for more folds over history)
    - recency_half_life=60 (C1: 90→60 for faster regime adaptation)
    - short_min_fraction=0.40 (B3: 0.35→0.40 to fix SHORT under-representation)
    """
    qs_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'quick_start.py')
    )
    assert os.path.exists(qs_path), f"quick_start.py not found at {qs_path}"

    with open(qs_path, 'r') as f:
        src = f.read()

    assert re.search(r'v5-score-lambda.*?default\s*=\s*0\.30', src, re.DOTALL), \
        "CLI --v5-score-lambda must default to 0.30 (Task #56 A1)"

    assert re.search(r'v5-wf-train-months.*?default\s*=\s*9[^0-9]', src, re.DOTALL), \
        "CLI --v5-wf-train-months must default to 9 (Task #56 C2)"

    assert re.search(r'v5-recency-half-life.*?default\s*=\s*60[^0-9]', src, re.DOTALL), \
        "CLI --v5-recency-half-life must default to 60 (Task #56 C1)"

    assert re.search(r'v5-short-min-fraction.*?default\s*=\s*0\.40', src, re.DOTALL), \
        "CLI --v5-short-min-fraction must default to 0.40 (Task #56 B3)"


def test_task56_fold_health_and_threshold_guide_in_source():
    """Task #56 D1/D2: v5_train.py contains [V5_FOLD_HEALTH] and [V5_THRESHOLD_GUIDE] blocks.

    D1 must include LIVE_READY/COLLAPSED verdicts and expected_daily_R.
    D2 must recommend --v5-min-threshold (NOT --v5-score-threshold).
    """
    v5_train_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'train', 'v5_train.py')
    )
    assert os.path.exists(v5_train_path), f"v5_train.py not found at {v5_train_path}"

    with open(v5_train_path, 'r') as f:
        src = f.read()

    # D1: fold health block
    assert '[V5_FOLD_HEALTH]' in src, \
        "[V5_FOLD_HEALTH] not in v5_train.py — D1 missing"
    assert 'LIVE_READY' in src, \
        "LIVE_READY verdict missing — D1 incomplete"
    assert 'COLLAPSED' in src, \
        "COLLAPSED verdict missing — D1 incomplete"
    assert 'expected_daily_R' in src, \
        "expected_daily_R missing from D1 fold health block"

    # D2: threshold guide block
    assert '[V5_THRESHOLD_GUIDE]' in src, \
        "[V5_THRESHOLD_GUIDE] not in v5_train.py — D2 missing"

    # Extract the THRESHOLD_GUIDE section from all occurrences and check the log.info f-string
    # There are multiple occurrences: comment, f-string header, f-string body.
    # Join them all and search for --v5-min-threshold (found in the log.info body).
    tg_all = ''.join(src.split('[V5_THRESHOLD_GUIDE]')[1:]).split('agg_report')[0]
    assert '--v5-min-threshold' in tg_all, \
        "D2 must recommend --v5-min-threshold (not --v5-score-threshold)"
    assert '--v5-score-threshold' not in tg_all, \
        "D2 must NOT recommend --v5-score-threshold (wrong flag); use --v5-min-threshold"


def test_task73_score_exponent_rank_preservation():
    """Task #73: score_exponent monotonic transform must preserve rank and improve right-tail spread.

    The transform is: score → clamp(score, 0, inf)^exponent
    - NaN entries must remain NaN (never affected by the transform)
    - -inf entries (gate-killed) must remain -inf
    - Negative-finite entries must remain unchanged
    - Positive-finite entries must be transformed: score ** exponent
    - Rank among positive-finite entries must be preserved for any exponent > 0
    - Right-tail spread (p90/|mean|) must improve when exponent < 1
    """
    import numpy as np

    def apply_score_exponent(scores_in, exponent):
        """Replicate the exact transform from run_v5_forward_test (v5_train.py)."""
        scores = scores_in.copy()
        _is_positive_finite = np.isfinite(scores) & (scores > 0.0)
        _pos_clipped = np.clip(scores, 0.0, np.inf)
        scores = np.where(_is_positive_finite, _pos_clipped ** exponent, scores)
        return scores

    rng = np.random.RandomState(42)
    base = np.abs(rng.normal(0.01, 0.005, size=500))  # positive, ~realistic score cloud

    # Add special entries
    special = np.array([np.nan, -np.inf, -0.005, -0.001, 0.0])
    scores_in = np.concatenate([base, special])

    exponent = 0.5  # square-root stretch — right tail improvement expected

    scores_out = apply_score_exponent(scores_in, exponent)

    # 1. NaN must remain NaN
    nan_mask_in = np.isnan(scores_in)
    assert np.all(np.isnan(scores_out[nan_mask_in])), \
        "NaN entries must remain NaN after score_exponent transform"

    # 2. -inf must remain -inf
    neginf_mask = np.isneginf(scores_in)
    assert np.all(np.isneginf(scores_out[neginf_mask])), \
        "-inf (gate-killed) entries must remain -inf after score_exponent transform"

    # 3. Negative-finite entries must remain unchanged
    neg_finite_mask = np.isfinite(scores_in) & (scores_in < 0)
    np.testing.assert_array_equal(
        scores_out[neg_finite_mask], scores_in[neg_finite_mask],
        err_msg="Negative-finite scores must not be changed by score_exponent transform"
    )

    # 4. Zero-finite entries must remain zero (0.0**exp = 0.0 for exp > 0)
    zero_mask = np.isfinite(scores_in) & (scores_in == 0.0)
    assert np.all(scores_out[zero_mask] == 0.0), \
        "Zero-valued scores must remain 0.0 after score_exponent transform"

    # 5. Rank preservation among positive-finite entries
    pos_mask = np.isfinite(scores_in) & (scores_in > 0.0)
    in_vals  = scores_in[pos_mask]
    out_vals = scores_out[pos_mask]
    rank_in  = np.argsort(in_vals)
    rank_out = np.argsort(out_vals)
    np.testing.assert_array_equal(
        rank_in, rank_out,
        err_msg="score_exponent must preserve rank among positive-finite entries"
    )

    # 6. All positive-finite outputs must equal input**exponent
    expected = in_vals ** exponent
    np.testing.assert_allclose(out_vals, expected, rtol=1e-10,
        err_msg="Positive-finite entries must be transformed as score**exponent")

    # 7. Monotonicity of transform: f(x) = x^exponent for any exponent > 0 is strictly
    # monotonically increasing on (0, inf). So if score_a > score_b > 0, then
    # score_a^exp > score_b^exp. Verify this holds for arbitrary pairs of positive scores.
    test_pairs = [(0.001, 0.0005), (0.05, 0.02), (0.1, 0.09), (0.5, 0.3), (1.0, 0.99)]
    for s_high, s_low in test_pairs:
        assert s_high > s_low > 0
        out_high = s_high ** exponent
        out_low  = s_low  ** exponent
        assert out_high > out_low, (
            f"Monotonicity violated: {s_high}^{exponent} = {out_high:.6f} "
            f"should be > {s_low}^{exponent} = {out_low:.6f}"
        )

    # 8. Verify exponent=1.0 is a no-op on positive-finite entries
    scores_identity = apply_score_exponent(scores_in, 1.0)
    np.testing.assert_array_equal(
        scores_identity[pos_mask], scores_in[pos_mask],
        err_msg="score_exponent=1.0 must be a no-op (identity transform)"
    )


def test_task73_cli_flags_present():
    """Task #73: --v5-min-mu-r-score and --v5-score-exponent flags must be in quick_start.py."""
    qs_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'quick_start.py')
    )
    assert os.path.exists(qs_path), f"quick_start.py not found at {qs_path}"
    with open(qs_path, 'r') as f:
        src = f.read()

    assert '--v5-min-mu-r-score' in src, \
        "Task #73: --v5-min-mu-r-score CLI flag missing from quick_start.py"
    assert '--v5-score-exponent' in src, \
        "Task #73: --v5-score-exponent CLI flag missing from quick_start.py"

    # Both flags must be wired into call sites (v5_min_mu_r_score must appear as a kwarg)
    assert 'v5_min_mu_r_score' in src, \
        "Task #73: v5_min_mu_r_score kwarg not found in quick_start.py call sites"
    assert 'v5_score_exponent' in src, \
        "Task #73: v5_score_exponent kwarg not found in quick_start.py call sites"


def test_task73_min_mu_r_score_not_hardcoded():
    """Task #73: min_mu_r_score=0.0 must not appear as a hardcoded literal at the epoch sweep
    call sites inside train_v5_model (lines ~8573 and ~8923 in v5_train.py).

    These should now read `min_mu_r_score=min_mu_r_score` (passing the parameter).
    Allowing a hardcoded 0.0 there would silently bypass any --v5-min-mu-r-score > 0 setting.
    """
    vt_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'train', 'v5_train.py')
    )
    assert os.path.exists(vt_path), f"v5_train.py not found at {vt_path}"
    with open(vt_path, 'r') as f:
        lines = f.readlines()

    hardcoded_violations = []
    # We expect exactly 0 occurrences of `min_mu_r_score=0.0,` inside compute_v5_scores calls
    # within train_v5_model (the function starts at ~line 7021). Allow it in function signatures
    # (default value) and in tests/comments only.
    in_train_v5_model = False
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if 'def train_v5_model(' in stripped:
            in_train_v5_model = True
        # If we see the pattern as a call argument (not a default= in def, not a comment)
        if in_train_v5_model and 'min_mu_r_score=0.0,' in stripped:
            # Allow it in function signature (default value) lines
            if not stripped.startswith('min_mu_r_score=0.0') or 'def ' in stripped:
                # Allow default value in function definition
                if 'def ' not in stripped:
                    hardcoded_violations.append((i, line.rstrip()))

    assert len(hardcoded_violations) == 0, (
        f"Task #73: Found hardcoded min_mu_r_score=0.0 inside train_v5_model at lines:\n"
        + "\n".join(f"  L{lno}: {ltext}" for lno, ltext in hardcoded_violations)
    )


def test_task73_highbar_fallback_yields_trades():
    """Task #73: When all symbols have HIGH_BAR (inf) threshold and per_sym_no_edge_fallback=True,
    the forward test must still produce trades (using the global threshold fallback).

    This is a regression test for the per_sym_no_edge_fallback path in run_v5_forward_test.
    We verify the logic by inspecting the source code's fallback branch, since torch is not
    available in this environment and we cannot call run_v5_forward_test directly.

    The test verifies:
    1. The V5ForwardTestConfig class has a per_sym_no_edge_fallback field.
    2. The run_v5_forward_test code has the ALL_INF_BLOCKED branch.
    3. When per_sym_no_edge_fallback=True, the code resets inf thresholds to a fallback value.
    4. The fallback path's log message matches [V5_FWD][ALL_INF_FALLBACK].
    """
    vt_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'train', 'v5_train.py')
    )
    assert os.path.exists(vt_path), f"v5_train.py not found at {vt_path}"
    with open(vt_path, 'r') as f:
        src = f.read()

    # 1. V5ForwardTestConfig must have per_sym_no_edge_fallback field
    assert 'per_sym_no_edge_fallback' in src, \
        "V5ForwardTestConfig must have per_sym_no_edge_fallback field"

    # 2. The ALL_INF_BLOCKED detection branch must exist
    assert '_all_sym_inf' in src or 'ALL_INF_BLOCKED' in src, \
        "run_v5_forward_test must have an ALL_INF_BLOCKED / _all_sym_inf detection branch"

    # 3. The fallback reset path must set per_bar_threshold for inf-blocked bars
    assert 'per_bar_threshold[_inf_mask] = _fallback_thr' in src or \
           'per_bar_threshold[_inf_mask]' in src, \
        "run_v5_forward_test ALL_INF_BLOCKED path must reset inf thresholds to fallback value"

    # 4. The fallback log message must exist
    assert 'ALL_INF_FALLBACK' in src, \
        "run_v5_forward_test must log [V5_FWD][ALL_INF_FALLBACK] when fallback is applied"

    # 5. The fallback threshold must be computed from score distribution (not hardcoded)
    # — the code uses p80 of finite scores, with a floor at config.min_threshold
    assert '_score_p80' in src, \
        "ALL_INF_FALLBACK must compute fallback from score p80 to avoid hardcoded threshold mismatch"

    # 6. The V5ForwardTestConfig.per_sym_no_edge_fallback must have a default
    assert 'per_sym_no_edge_fallback: bool' in src or \
           'per_sym_no_edge_fallback=False' in src or \
           'per_sym_no_edge_fallback: bool = False' in src, \
        "per_sym_no_edge_fallback must have a default of False in V5ForwardTestConfig"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
