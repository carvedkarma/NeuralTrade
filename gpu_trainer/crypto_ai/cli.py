from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .backtest import backtest_model
from .config import DEFAULT_SYMBOLS, SystemConfig
from .data import build_data_bundle
from .paper import paper_trade_replay
from .training import train_model


def _run_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def _load_config(path: str | None) -> SystemConfig:
    if path is None:
        return SystemConfig()
    cfg_path = Path(path)
    raw = json.loads(cfg_path.read_text())
    return SystemConfig.from_dict(raw)


def _save_config(config: SystemConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2))


def cmd_init_config(args: argparse.Namespace) -> int:
    out = Path(args.output)
    config = SystemConfig()
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if len(symbols) != 10:
            raise ValueError(
                f"Expected exactly 10 symbols, got {len(symbols)}. "
                "Example: BTCUSDT,ETHUSDT,... (10 symbols total)."
            )
        config.symbols = symbols
    _save_config(config, out)
    print(f"[ok] wrote config -> {out}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    bundle = build_data_bundle(config, horizon=args.horizon, refresh=args.refresh)
    print(
        "[ok] fetched",
        f"symbols={len(bundle.symbols)}",
        f"rows={bundle.features.shape[0]}",
        f"features={bundle.features.shape[2]}",
    )
    return 0


def _resolve_checkpoint(explicit: str | None, run_name: str | None, config: SystemConfig) -> Path:
    if explicit:
        return Path(explicit)
    if run_name:
        path = config.artifact_dir / run_name / "best_model.pt"
        return path
    raise ValueError("Provide --checkpoint or --run-name.")


def cmd_train(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.device is not None:
        config.device = args.device
    if args.history_days is not None:
        config.history_days = args.history_days
    if args.interval is not None:
        config.interval = args.interval

    run_name = args.run_name or _run_name("train")
    bundle = build_data_bundle(config, horizon=args.horizon, refresh=args.refresh_data)
    artifacts = train_model(config=config, bundle=bundle, run_name=run_name)
    print(f"[ok] trained run={run_name}")
    print(f"checkpoint: {artifacts.checkpoint_path}")
    print(f"metrics:    {artifacts.metrics_path}")
    print(f"best_val_loss: {artifacts.best_val_loss:.6f}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    if args.device is not None:
        config.device = args.device
    checkpoint = _resolve_checkpoint(args.checkpoint, args.run_name, config)
    bundle = build_data_bundle(config, horizon=args.horizon, refresh=args.refresh_data)
    report = backtest_model(
        config=config,
        bundle=bundle,
        checkpoint_path=checkpoint,
        report_name=args.report_name or _run_name("backtest"),
    )
    print(f"[ok] backtest report: {report.report_path}")
    print(
        "metrics:",
        f"return={report.total_return:.4f}",
        f"ann={report.annualized_return:.4f}",
        f"sharpe={report.sharpe:.3f}",
        f"max_dd={report.max_drawdown:.4f}",
        f"win_rate={report.win_rate:.3f}",
        f"turnover={report.turnover:.4f}",
    )
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    if args.device is not None:
        config.device = args.device
    checkpoint = _resolve_checkpoint(args.checkpoint, args.run_name, config)
    bundle = build_data_bundle(config, horizon=args.horizon, refresh=args.refresh_data)
    report = paper_trade_replay(
        config=config,
        bundle=bundle,
        checkpoint_path=checkpoint,
        report_name=args.report_name or _run_name("paper"),
        steps=args.steps,
    )
    print(f"[ok] paper replay report: {report.report_path}")
    print(
        "metrics:",
        f"return={report.cumulative_return:.4f}",
        f"max_dd={report.max_drawdown:.4f}",
        f"avg_turnover={report.avg_turnover:.4f}",
        f"steps={report.n_steps}",
    )
    return 0


def cmd_self_train(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    if args.device is not None:
        config.device = args.device
    cycles = args.cycles if args.cycles is not None else config.self_train_cycles
    sleep_seconds = (
        args.sleep_seconds if args.sleep_seconds is not None else config.self_train_sleep_seconds
    )
    best_sharpe = float("-inf")
    best_checkpoint: Path | None = None
    best_report: Path | None = None

    for cycle in range(1, cycles + 1):
        run_name = _run_name(f"self_train_cycle{cycle}")
        bundle = build_data_bundle(config, horizon=args.horizon, refresh=True)
        artifacts = train_model(config=config, bundle=bundle, run_name=run_name)
        report = backtest_model(
            config=config,
            bundle=bundle,
            checkpoint_path=artifacts.checkpoint_path,
            report_name=f"{run_name}_bt",
        )
        print(
            f"[cycle {cycle}/{cycles}]",
            f"checkpoint={artifacts.checkpoint_path}",
            f"sharpe={report.sharpe:.3f}",
            f"return={report.total_return:.4f}",
            f"max_dd={report.max_drawdown:.4f}",
        )
        if report.sharpe > best_sharpe:
            best_sharpe = report.sharpe
            best_checkpoint = artifacts.checkpoint_path
            best_report = report.report_path
        if cycle < cycles:
            time.sleep(max(0, sleep_seconds))

    if best_checkpoint is not None:
        print("[ok] self-train complete")
        print(f"best_checkpoint: {best_checkpoint}")
        print(f"best_report:     {best_report}")
        print(f"best_sharpe:     {best_sharpe:.3f}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crypto_ai",
        description=(
            "GPU-first autonomous crypto trading AI for 10 symbols. "
            "Includes data fetch, training, backtesting, paper replay, and self-training loop."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file. If omitted, built-in defaults are used.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_cfg = sub.add_parser("init-config", help="Write a starter JSON config.")
    init_cfg.add_argument("--output", type=str, default="gpu_trainer/crypto_ai_config.json")
    init_cfg.add_argument(
        "--symbols",
        type=str,
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated 10-symbol list, e.g. BTCUSDT,ETHUSDT,...",
    )
    init_cfg.set_defaults(func=cmd_init_config)

    fetch = sub.add_parser("fetch", help="Download/update market data and build aligned bundle.")
    fetch.add_argument("--horizon", type=int, default=6, help="Forward horizon in candles.")
    fetch.add_argument("--refresh", action="store_true", help="Force re-download data from Binance.")
    fetch.set_defaults(func=cmd_fetch)

    train = sub.add_parser("train", help="Train the multi-symbol transformer.")
    train.add_argument("--run-name", type=str, default=None)
    train.add_argument("--horizon", type=int, default=6)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default=None)
    train.add_argument("--history-days", type=int, default=None)
    train.add_argument("--interval", type=str, default=None)
    train.add_argument("--refresh-data", action="store_true")
    train.set_defaults(func=cmd_train)

    bt = sub.add_parser("backtest", help="Backtest a trained checkpoint on holdout data.")
    bt.add_argument("--checkpoint", type=str, default=None)
    bt.add_argument("--run-name", type=str, default=None)
    bt.add_argument("--report-name", type=str, default=None)
    bt.add_argument("--horizon", type=int, default=6)
    bt.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default=None)
    bt.add_argument("--refresh-data", action="store_true")
    bt.set_defaults(func=cmd_backtest)

    paper = sub.add_parser("paper", help="Replay paper trading over recent data.")
    paper.add_argument("--checkpoint", type=str, default=None)
    paper.add_argument("--run-name", type=str, default=None)
    paper.add_argument("--report-name", type=str, default=None)
    paper.add_argument("--steps", type=int, default=240)
    paper.add_argument("--horizon", type=int, default=6)
    paper.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default=None)
    paper.add_argument("--refresh-data", action="store_true")
    paper.set_defaults(func=cmd_paper)

    st = sub.add_parser("self-train", help="Run repeated retrain+backtest cycles automatically.")
    st.add_argument("--cycles", type=int, default=None)
    st.add_argument("--sleep-seconds", type=int, default=None)
    st.add_argument("--horizon", type=int, default=6)
    st.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default=None)
    st.set_defaults(func=cmd_self_train)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
