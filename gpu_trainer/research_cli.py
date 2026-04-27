from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import torch
except ImportError:  # pragma: no cover - exercised on non-GPU test hosts
    torch = None

try:  # Support both "python research_cli.py" and "python -m gpu_trainer.research_cli"
    from .research_config import DEFAULT_RESEARCH_SYMBOLS, ResearchProfile, build_research_profile
except ImportError:  # pragma: no cover - direct script fallback
    from research_config import DEFAULT_RESEARCH_SYMBOLS, ResearchProfile, build_research_profile

log = logging.getLogger("research_cli")

CHECKPOINT_DIR = Path("checkpoints")
DEPLOYED_DIR = CHECKPOINT_DIR / "deployed"
RESEARCH_RUNS_DIR = CHECKPOINT_DIR / "research_runs"


def _import_bulk_download():
    try:
        return import_module("gpu_trainer.bulk_download")
    except ImportError:
        return import_module("bulk_download")


def _import_learning():
    try:
        return import_module("gpu_trainer.learning")
    except ImportError:
        return import_module("learning")


def _import_v5_train():
    try:
        return import_module("gpu_trainer.train.v5_train")
    except ImportError:
        return import_module("train.v5_train")


def _bulk_download_helpers():
    mod = _import_bulk_download()
    return Path(mod.DATA_DIR), mod.download_symbols, mod.print_summary


def _v5_training_helpers():
    mod = _import_v5_train()
    return mod.run_v5_walk_forward, mod.train_v5_model


def _learning_modules():
    mod = _import_learning()
    return mod.LearningConfig, mod.LearningManager, mod.RetrainResult


def data_dir() -> Path:
    return Path(_import_bulk_download().DATA_DIR)


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def detect_device(prefer_cpu: bool = False) -> str:
    if prefer_cpu:
        return "cpu"
    return "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"


def parse_symbols(raw: str) -> List[str]:
    return [sym.strip().upper() for sym in raw.split(",") if sym.strip()]


def profile_from_args(args: argparse.Namespace) -> ResearchProfile:
    symbols = parse_symbols(args.symbols)
    return build_research_profile(
        symbols=symbols,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_candles=args.min_candles,
        train_months=args.train_months,
        test_months=args.test_months,
        walk_forward_folds=args.walk_forward_folds,
    )


def ensure_data(
    profile: ResearchProfile,
    *,
    force: bool = False,
    refresh: bool = False,
    dashboard_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    data_dir_path, download_symbols, print_summary = _bulk_download_helpers()
    log.info(
        "[DATA] Ensuring %d symbols have at least %d candles",
        len(profile.symbols),
        profile.min_candles,
    )
    results = download_symbols(
        profile.symbols,
        force=force,
        refresh=refresh,
        data_dir=data_dir_path,
        min_candles=profile.min_candles,
        dashboard_url=dashboard_url,
    )
    ok = print_summary(results, min_candles=profile.min_candles)
    if not ok:
        raise RuntimeError("One or more symbols have insufficient data")
    return results


def load_json(path: Path) -> Dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)


def latest_v5_metrics() -> Dict[str, Any]:
    metrics_path = CHECKPOINT_DIR / "v5_run_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected metrics file not found: {metrics_path}")
    return load_json(metrics_path)


def latest_walk_forward_report() -> Dict[str, Any]:
    report_path = CHECKPOINT_DIR / "v5_walkforward_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Expected walk-forward report not found: {report_path}")
    return load_json(report_path)


def export_best_policy_from_walk_forward(report: Dict[str, Any], profile: ResearchProfile) -> Dict[str, Any]:
    folds = report.get("folds", [])
    if not folds:
        raise ValueError("Walk-forward report has no folds")

    best_fold = max(
        folds,
        key=lambda fold: (
            float(fold.get("total_r", 0.0)),
            float(fold.get("expectancy_r", 0.0)),
            int(fold.get("total_trades", 0)),
        ),
    )
    threshold = best_fold.get("threshold_ema", best_fold.get("score_threshold", profile.shared_v5_kwargs()["min_threshold"]))
    policy = {
        "version": profile.name,
        "source": "research_cli.walk_forward",
        "policy_type": "threshold",
        "policy_value": float(threshold),
        "threshold": float(threshold),
        "tp_mult": profile.tp_mult,
        "sl_mult": profile.sl_mult,
        "cooldown": profile.cooldown,
        "horizon": profile.horizon,
        "metrics": {
            "selected_fold": best_fold.get("fold"),
            "selected_fold_total_r": best_fold.get("total_r"),
            "selected_fold_expectancy_r": best_fold.get("expectancy_r"),
            "selected_fold_trades": best_fold.get("total_trades"),
            "aggregate_total_r": report.get("aggregate", {}).get("total_r"),
            "aggregate_expectancy_r": report.get("aggregate", {}).get("avg_expectancy_r"),
            "aggregate_total_trades": report.get("aggregate", {}).get("total_trades"),
        },
    }
    out_path = CHECKPOINT_DIR / "best_policy.json"
    write_json(out_path, policy)
    log.info("[POLICY] Saved %s", out_path)
    return policy


def evaluate_promotion(metrics: Dict[str, Any], profile: ResearchProfile, baseline: Optional[Dict[str, Any]]) -> tuple[bool, List[str]]:
    gates = profile.promotion_gates
    reasons: List[str] = []

    total_r = float(metrics.get("total_r") or 0.0)
    expectancy = float(metrics.get("avg_expectancy_r") or 0.0)
    total_trades = int(metrics.get("total_trades") or 0)
    active_folds = int(metrics.get("active_folds") or 0)
    action_accuracy = float(metrics.get("mean_action_accuracy") or 0.0)
    mu_corr = float(metrics.get("mean_mu_r_correlation") or 0.0)
    score_disc = float(metrics.get("mean_score_disc_p90p50") or 0.0)
    long_pct = float(metrics.get("long_pct") or 0.0)

    if total_r < gates.min_total_r:
        reasons.append(f"total_r {total_r:.4f} < {gates.min_total_r:.4f}")
    if expectancy < gates.min_expectancy_r:
        reasons.append(f"avg_expectancy_r {expectancy:.4f} < {gates.min_expectancy_r:.4f}")
    if total_trades < gates.min_total_trades:
        reasons.append(f"total_trades {total_trades} < {gates.min_total_trades}")
    if active_folds < gates.min_active_folds:
        reasons.append(f"active_folds {active_folds} < {gates.min_active_folds}")
    if action_accuracy < gates.min_action_accuracy:
        reasons.append(f"mean_action_accuracy {action_accuracy:.4f} < {gates.min_action_accuracy:.4f}")
    if mu_corr < gates.min_mu_r_correlation:
        reasons.append(f"mean_mu_r_correlation {mu_corr:.4f} < {gates.min_mu_r_correlation:.4f}")
    if score_disc < gates.min_score_disc_p90p50:
        reasons.append(f"mean_score_disc_p90p50 {score_disc:.2f} < {gates.min_score_disc_p90p50:.2f}")
    if long_pct > gates.max_long_pct:
        reasons.append(f"long_pct {long_pct:.2f} > {gates.max_long_pct:.2f}")

    if gates.require_vs_baseline:
        if baseline is None:
            reasons.append("baseline metrics missing")
        else:
            baseline_total_r = float(baseline.get("total_r") or 0.0)
            baseline_expectancy = float(baseline.get("avg_expectancy_r") or 0.0)
            if total_r < baseline_total_r - gates.allow_total_r_regression:
                reasons.append(
                    f"total_r {total_r:.4f} regressed vs baseline {baseline_total_r:.4f}"
                )
            if expectancy < baseline_expectancy - gates.allow_expectancy_regression:
                reasons.append(
                    f"avg_expectancy_r {expectancy:.4f} regressed vs baseline {baseline_expectancy:.4f}"
                )

    return len(reasons) == 0, reasons


def copy_promoted_artifacts(symbols: Iterable[str]) -> None:
    DEPLOYED_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ("best_v5_expectancy.pt", "best_v5_loss.pt"):
        src = CHECKPOINT_DIR / filename
        if not src.exists():
            continue
        shutil.copy2(src, DEPLOYED_DIR / filename)
        for symbol in symbols:
            target_dir = DEPLOYED_DIR / symbol
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target_dir / filename)
    policy_path = CHECKPOINT_DIR / "best_policy.json"
    if policy_path.exists():
        shutil.copy2(policy_path, DEPLOYED_DIR / "best_policy.json")


def set_baseline(metrics: Dict[str, Any]) -> None:
    baseline_path = CHECKPOINT_DIR / "v5_baseline_metrics.json"
    gate_baseline_path = CHECKPOINT_DIR / "v5_gate_baseline_metrics.json"
    write_json(baseline_path, metrics)
    write_json(gate_baseline_path, metrics)
    log.info("[PROMOTION] Baseline updated")


def run_training(profile: ResearchProfile, *, device: str) -> Dict[str, Any]:
    data_dir_path, _, _ = _bulk_download_helpers()
    _, train_v5_model = _v5_training_helpers()
    log.info(
        "[TRAIN] Training V5 on %d symbols with epochs=%d batch=%d lr=%g",
        len(profile.symbols),
        profile.epochs,
        profile.batch_size,
        profile.lr,
    )
    data_path = data_dir_path / f"{profile.symbols[0]}_{profile.interval}.parquet"
    train_v5_model(
        data_path,
        device,
        profile.epochs,
        profile.batch_size,
        profile.lr,
        symbols=profile.symbols,
        **profile.train_v5_kwargs(),
    )
    metrics = latest_v5_metrics()
    write_json(RESEARCH_RUNS_DIR / "latest_train_metrics.json", metrics)
    return metrics


def run_walk_forward(profile: ResearchProfile, *, device: str) -> Dict[str, Any]:
    data_dir_path, _, _ = _bulk_download_helpers()
    run_v5_walk_forward, _ = _v5_training_helpers()
    log.info(
        "[EVAL] Running walk-forward with %d folds over %d symbols",
        profile.walk_forward_folds,
        len(profile.symbols),
    )
    report = run_v5_walk_forward(
        data_dir=data_dir_path,
        device=device,
        symbols=profile.symbols,
        epochs=profile.epochs,
        batch_size=profile.batch_size,
        lr=profile.lr,
        **profile.walk_forward_kwargs(),
    )
    if report is None:
        raise RuntimeError("Walk-forward did not return a report")
    write_json(RESEARCH_RUNS_DIR / "latest_walk_forward_report.json", report)
    export_best_policy_from_walk_forward(report, profile)
    return report


def current_baseline() -> Optional[Dict[str, Any]]:
    baseline_path = CHECKPOINT_DIR / "v5_baseline_metrics.json"
    if baseline_path.exists():
        return load_json(baseline_path)
    return None


def save_run_bundle(profile: ResearchProfile, metrics: Dict[str, Any], report: Dict[str, Any], promoted: bool, reasons: List[str]) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESEARCH_RUNS_DIR / f"research_run_{ts}.json"
    payload = {
        "timestamp": ts,
        "profile": asdict(profile),
        "metrics": metrics,
        "walk_forward_report_path": str(CHECKPOINT_DIR / "v5_walkforward_report.json"),
        "report_summary": report.get("aggregate", {}),
        "promoted": promoted,
        "promotion_reasons": reasons,
    }
    write_json(out, payload)
    return out


def run_pipeline(
    profile: ResearchProfile,
    *,
    device: str,
    ensure_data_first: bool,
    force_download: bool,
    refresh: bool,
    dashboard_url: Optional[str],
    set_baseline_on_first_success: bool,
) -> Dict[str, Any]:
    if ensure_data_first:
        ensure_data(profile, force=force_download, refresh=refresh, dashboard_url=dashboard_url)

    train_metrics = run_training(profile, device=device)
    wf_report = run_walk_forward(profile, device=device)
    metrics = latest_v5_metrics()
    baseline = current_baseline()
    baseline_for_eval = baseline
    if baseline is None and set_baseline_on_first_success:
        baseline_for_eval = dict(metrics)
        log.info("[PROMOTION] No baseline found; evaluating first successful run against absolute gates only")
    promoted, reasons = evaluate_promotion(metrics, profile, baseline_for_eval)

    if promoted:
        copy_promoted_artifacts(profile.symbols)
        if baseline is not None or set_baseline_on_first_success:
            set_baseline(metrics)
        log.info("[PROMOTION] Candidate promoted")
    else:
        log.warning("[PROMOTION] Candidate rejected: %s", "; ".join(reasons))

    run_bundle = save_run_bundle(profile, metrics, wf_report, promoted, reasons)
    return {
        "train_metrics": train_metrics,
        "metrics": metrics,
        "walk_forward_report": wf_report,
        "promoted": promoted,
        "reasons": reasons,
        "run_bundle": str(run_bundle),
    }


def make_local_learning_manager():
    LearningConfig, LearningManager, RetrainResult = _learning_modules()
    data_dir_path, _, _ = _bulk_download_helpers()

    class LocalLearningManager(LearningManager):
        def __init__(self, device: str, symbols: List[str], config: Optional[LearningConfig] = None):
            super().__init__(replit_url="", device=device, symbols=symbols, config=config or LearningConfig())

        def _fetch_full_history(self, symbol: str) -> List[Dict]:
            import pandas as pd

            parquet_path = data_dir_path / f"{symbol}_15m.parquet"
            if not parquet_path.exists():
                return []
            df = pd.read_parquet(parquet_path)
            return df.sort_values("timestamp").to_dict("records")

        def _push_learning_stats(
            self,
            symbol: str,
            result: RetrainResult,
            promoted: bool,
            reason: str,
            prev_stats: Optional[Dict] = None,
        ):
            payload = {
                "symbol": symbol,
                "promoted": promoted,
                "reason": reason,
                "pf_net": result.pf_net,
                "e_net": result.e_net,
                "trades_per_day": result.trades_per_day,
                "profitable_regimes": result.profitable_regimes,
            }
            stats_dir = RESEARCH_RUNS_DIR / "learning_stats"
            stats_dir.mkdir(parents=True, exist_ok=True)
            write_json(stats_dir / f"{symbol}.json", payload)
            if promoted:
                self.deployed_stats[symbol] = {
                    "pf_net": result.pf_net,
                    "e_net": result.e_net,
                    "profitable_regimes": result.profitable_regimes,
                    "trades_per_day": result.trades_per_day,
                }

    return LearningConfig, LocalLearningManager


def run_self_training_loop(
    profile: ResearchProfile,
    *,
    device: str,
    cycles: int,
    sleep_seconds: int,
    dashboard_url: Optional[str],
) -> None:
    LearningConfig, LocalLearningManager = make_local_learning_manager()
    learning_cfg = LearningConfig(
        retrain_hour_utc=0,
        retrain_interval_hours=0,
        min_new_bars=profile.min_candles,
        training_epochs=max(1, min(profile.epochs, 20)),
        gate_pf_net=1.02,
        gate_profitable_regimes=2,
        gate_enet=0.0,
        gate_maxdd_r=8.0,
        min_tpd=0.1,
        max_tpd=10.0,
        auto_promote=True,
        geometry_sweep_on_retrain=True,
        tp_mult=profile.tp_mult,
        sl_mult=profile.sl_mult,
        horizon=profile.horizon,
    )
    manager = LocalLearningManager(device=device, symbols=profile.symbols, config=learning_cfg)

    for cycle in range(1, cycles + 1):
        log.info("[SELF] Cycle %d/%d", cycle, cycles)
        ensure_data(profile, refresh=True, dashboard_url=dashboard_url)
        for symbol in profile.symbols:
            result = manager.retrain_symbol(symbol)
            if result.success and result.best_policy:
                symbol_policy = {
                    "version": profile.name,
                    "source": "research_cli.self_train",
                    "symbol": symbol,
                    "threshold": result.best_policy.get("threshold"),
                    "cooldown": result.best_policy.get("cooldown"),
                    "tp_mult": result.best_policy.get("tp_mult"),
                    "sl_mult": result.best_policy.get("sl_mult"),
                }
                write_json(RESEARCH_RUNS_DIR / "learning_stats" / f"{symbol}_policy.json", symbol_policy)
        if cycle < cycles:
            time.sleep(sleep_seconds)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local-first GPU crypto research CLI",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(DEFAULT_RESEARCH_SYMBOLS),
        help="Comma-separated research universe (default: 10 liquid symbols)",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-candles", type=int, default=20_000, dest="min_candles")
    parser.add_argument("--train-months", type=int, default=9, dest="train_months")
    parser.add_argument("--test-months", type=int, default=1, dest="test_months")
    parser.add_argument("--walk-forward-folds", type=int, default=3, dest="walk_forward_folds")
    parser.add_argument("--dashboard-url", type=str, default="", help="Optional dashboard source; empty means Binance-only")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare-data", help="Download or refresh local data for the research universe")
    sub.add_parser("train", help="Train the multi-symbol V5 model")
    sub.add_parser("evaluate", help="Run walk-forward evaluation and export best policy")

    pipeline = sub.add_parser("run", help="Run full prepare -> train -> evaluate -> promote pipeline")
    pipeline.add_argument("--skip-data", action="store_true", help="Skip data refresh before running the pipeline")
    pipeline.add_argument("--set-baseline-on-first-success", action="store_true", help="Seed the baseline when none exists and the run passes")

    baseline = sub.add_parser("set-baseline", help="Promote latest metrics to the comparison baseline")
    baseline.add_argument("--from-file", type=str, default="", help="Optional metrics JSON file to use instead of checkpoints/v5_run_metrics.json")

    self_train = sub.add_parser("self-train", help="Periodic local retraining loop over the 10-symbol universe")
    self_train.add_argument("--cycles", type=int, default=1, help="Number of retraining cycles to execute")
    self_train.add_argument("--sleep-seconds", type=int, default=300, help="Sleep time between cycles")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    device = detect_device(prefer_cpu=args.cpu)
    profile = profile_from_args(args)
    dashboard_url = args.dashboard_url or None

    log.info("[INIT] Device=%s symbols=%s", device, profile.symbols)
    RESEARCH_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if args.command == "prepare-data":
        ensure_data(profile, force=args.force_download, refresh=args.refresh, dashboard_url=dashboard_url)
        return 0

    if args.command == "train":
        run_training(profile, device=device)
        return 0

    if args.command == "evaluate":
        report = run_walk_forward(profile, device=device)
        metrics = latest_v5_metrics()
        promoted, reasons = evaluate_promotion(metrics, profile, current_baseline())
        bundle = save_run_bundle(profile, metrics, report, promoted, reasons)
        log.info("[EVAL] Run bundle saved to %s", bundle)
        return 0 if promoted else 1

    if args.command == "run":
        result = run_pipeline(
            profile,
            device=device,
            ensure_data_first=not args.skip_data,
            force_download=args.force_download,
            refresh=args.refresh,
            dashboard_url=dashboard_url,
            set_baseline_on_first_success=args.set_baseline_on_first_success,
        )
        log.info("[RUN] promoted=%s bundle=%s", result["promoted"], result["run_bundle"])
        if not result["promoted"]:
            for reason in result["reasons"]:
                log.warning("[RUN] %s", reason)
        return 0 if result["promoted"] else 1

    if args.command == "set-baseline":
        metrics = load_json(Path(args.from_file)) if args.from_file else latest_v5_metrics()
        set_baseline(metrics)
        return 0

    if args.command == "self-train":
        run_self_training_loop(
            profile,
            device=device,
            cycles=args.cycles,
            sleep_seconds=args.sleep_seconds,
            dashboard_url=dashboard_url,
        )
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
