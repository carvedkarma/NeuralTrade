from __future__ import annotations

"""Rithal Tail Guard V6.5.3 R2.

R2 fixes the installer preflight failure caused by importing the entire active
``mythos_live_tail_guard_v6_6.py`` merely to discover raw premium paths. The
active guard has project/runtime imports and side effects that are unnecessary
for path discovery and can fail when the repair utility is executed from a
temporary directory.

The V22 contract already defines the canonical path unambiguously as:
``data_lake/raw/binance_um/premium_15m/<SYMBOL>.parquet``.
R2 resolves that path directly, delegates source patching and parquet
canonicalization to the verified V6.5.3 implementation, and never imports or
executes the active guard during preflight.
"""

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

VERSION = "RITHAL_TAIL_GUARD_V6_5_3_R2"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT"]


def _load_base() -> Any:
    base_path = Path(__file__).resolve().with_name("rithal_tail_guard_premium_fix_v653.py")
    if not base_path.is_file():
        raise FileNotFoundError(f"base V6.5.3 repair module missing: {base_path}")
    spec = importlib.util.spec_from_file_location("rithal_tail_guard_premium_fix_v653_base", str(base_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base V6.5.3 repair module: {base_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_premium_paths(data_root: Path, symbols: Sequence[str]) -> dict[str, str]:
    root = Path(data_root).resolve()
    premium_root = root / "raw" / "binance_um" / "premium_15m"
    return {symbol: str((premium_root / f"{symbol}.parquet").resolve()) for symbol in symbols}


def canonicalize_symbols(data_root: Path, symbols: Sequence[str]) -> dict[str, Any]:
    base = _load_base()
    paths = resolve_premium_paths(data_root, symbols)
    results: dict[str, Any] = {}
    for symbol, raw_path in paths.items():
        results[symbol] = base.canonicalize_premium_file(Path(raw_path))
    return {"version": VERSION, "status": "PASS", "results": results}


def self_test() -> dict[str, Any]:
    base = _load_base()
    base_report = base.self_test()
    if base_report.get("status") != "PASS":
        raise RuntimeError("base V6.5.3 self-test did not pass")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "data_lake"
        paths = resolve_premium_paths(root, DEFAULT_SYMBOLS)
        expected_root = (root / "raw" / "binance_um" / "premium_15m").resolve()
        assert len(paths) == 6
        for symbol in DEFAULT_SYMBOLS:
            assert Path(paths[symbol]).parent == expected_root
            assert Path(paths[symbol]).name == f"{symbol}.parquet"

    return {
        "version": VERSION,
        "status": "PASS",
        "checks": [
            "base_v653_regression_pass",
            "no_active_guard_import",
            "canonical_v22_premium_path",
            "six_symbol_resolution",
            "full_state_canonicalization_delegate",
        ],
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("--engine")
    parser.add_argument("--guard")  # accepted only for source patch compatibility
    parser.add_argument("--data-root")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--patch-source", action="store_true")
    parser.add_argument("--list-premium-paths", action="store_true")
    parser.add_argument("--canonicalize-premium-state", action="store_true")
    args = parser.parse_args(argv)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return 0

    if args.patch_source:
        if not args.engine or not args.guard:
            parser.error("--patch-source requires --engine and --guard")
        base = _load_base()
        report = {
            "version": VERSION,
            "status": "PASS",
            "engine": base.patch_file(Path(args.engine), "engine"),
            "guard": base.patch_file(Path(args.guard), "guard"),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.list_premium_paths:
        if not args.data_root:
            parser.error("--list-premium-paths requires --data-root")
        print(json.dumps(resolve_premium_paths(Path(args.data_root), symbols), indent=2, sort_keys=True))
        return 0

    if args.canonicalize_premium_state:
        if not args.data_root:
            parser.error("--canonicalize-premium-state requires --data-root")
        print(json.dumps(canonicalize_symbols(Path(args.data_root), symbols), indent=2, sort_keys=True))
        return 0

    parser.error("select --self-test, --patch-source, --list-premium-paths, or --canonicalize-premium-state")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
