from gpu_trainer.research_cli import evaluate_promotion, parse_symbols
from gpu_trainer.research_config import DEFAULT_RESEARCH_SYMBOLS, build_research_profile


def test_parse_symbols_normalizes_and_filters() -> None:
    assert parse_symbols(" btcusdt, ETHUSDT ,, solusdt ") == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_default_research_profile_uses_ten_symbols() -> None:
    profile = build_research_profile()
    assert profile.symbols == DEFAULT_RESEARCH_SYMBOLS
    assert len(profile.symbols) == 10


def test_evaluate_promotion_passes_absolute_gates() -> None:
    profile = build_research_profile()
    metrics = {
        "total_r": 1.5,
        "avg_expectancy_r": 0.05,
        "total_trades": 80,
        "active_folds": 3,
        "mean_action_accuracy": 0.57,
        "mean_mu_r_correlation": 0.12,
        "mean_score_disc_p90p50": 4.2,
        "long_pct": 55.0,
    }
    baseline = {
        "total_r": 1.4,
        "avg_expectancy_r": 0.04,
    }

    promoted, reasons = evaluate_promotion(metrics, profile, baseline)

    assert promoted is True
    assert reasons == []


def test_evaluate_promotion_rejects_regression_vs_baseline() -> None:
    profile = build_research_profile()
    metrics = {
        "total_r": 0.7,
        "avg_expectancy_r": 0.01,
        "total_trades": 80,
        "active_folds": 3,
        "mean_action_accuracy": 0.57,
        "mean_mu_r_correlation": 0.12,
        "mean_score_disc_p90p50": 4.2,
        "long_pct": 55.0,
    }
    baseline = {
        "total_r": 1.4,
        "avg_expectancy_r": 0.04,
    }

    promoted, reasons = evaluate_promotion(metrics, profile, baseline)

    assert promoted is False
    assert any("regressed vs baseline" in reason for reason in reasons)


def test_evaluate_promotion_rejects_missing_baseline_when_required() -> None:
    profile = build_research_profile()
    metrics = {
        "total_r": 1.5,
        "avg_expectancy_r": 0.05,
        "total_trades": 80,
        "active_folds": 3,
        "mean_action_accuracy": 0.57,
        "mean_mu_r_correlation": 0.12,
        "mean_score_disc_p90p50": 4.2,
        "long_pct": 55.0,
    }

    promoted, reasons = evaluate_promotion(metrics, profile, baseline=None)

    assert promoted is False
    assert "baseline metrics missing" in reasons
