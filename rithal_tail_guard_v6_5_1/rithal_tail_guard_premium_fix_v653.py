from __future__ import annotations

"""Rithal Tail Guard V6.5.3 premium-state canonicalization repair.

V6.5.2 proved that a four-bar fetch warm-up reduces the bounded/full mismatch,
but exact-project logs still showed a residual failure in
``premium_index_change_1h``. The close series itself passed parity. Therefore
older derivative values already persisted in the raw premium parquet were the
remaining inconsistent state.

V6.5.3 keeps the frozen V22 engine contract, keeps the four hidden predecessor
bars for future incremental fetches, and performs one transactional full-state
recalculation of only ``premium_index_change_1h`` from the existing sorted
``premium_index_close`` series. Other columns, model files, enriched archives,
ledgers, thresholds and execution behavior are untouched.
"""

import argparse
import importlib.util
import json
import py_compile
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

VERSION = "RITHAL_TAIL_GUARD_V6_5_3"
ENGINE_MARKER_OLD = "RITHAL_TAIL_GUARD_V6_5_1_PREMIUM_ALIGNMENT"
ENGINE_MARKER_V652 = "RITHAL_TAIL_GUARD_V6_5_2_CANONICAL_PREMIUM"
ENGINE_MARKER = "RITHAL_TAIL_GUARD_V6_5_3_CANONICAL_PREMIUM"
GUARD_MARKER_V652 = "RITHAL_TAIL_GUARD_V6_5_2_PREMIUM_WARMUP"
GUARD_MARKER = "RITHAL_TAIL_GUARD_V6_5_3_PREMIUM_WARMUP"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT"]


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _paren_block_end(lines: list[str], start: int) -> int:
    balance = 0
    saw_open = False
    for index in range(start, len(lines)):
        balance += lines[index].count("(") - lines[index].count(")")
        saw_open = saw_open or "(" in lines[index]
        if saw_open and balance <= 0:
            return index
        if not saw_open:
            return index
    raise RuntimeError(f"unterminated expression at line {start + 1}")


def _canonical_engine_lines(indent: str) -> list[str]:
    child = indent + "    "
    return [
        f"{indent}# {ENGINE_MARKER}",
        f"{indent}# Frozen V22 semantics: consume the derivative already computed",
        f"{indent}# in the raw premium frame; do not redefine it after alignment.",
        f'{indent}for col in ["premium_index_close", "premium_index_change_1h"]:',
        f"{child}df[col] = _numeric(df, col, 0.0)",
        f'{indent}df["premium_index_available"] = (df["premium_index_close"].abs() > 0).astype(float)',
        f'{indent}df["premium_index_z_7d"] = _z(df["premium_index_close"].replace(0.0, np.nan).ffill().fillna(0.0), 7 * 96)',
    ]


def patch_engine_text(text: str) -> tuple[str, bool]:
    lines = text.splitlines()
    if any(ENGINE_MARKER in line for line in lines):
        return text, False

    marker_index = next(
        (
            i
            for i, line in enumerate(lines)
            if ENGINE_MARKER_OLD in line or ENGINE_MARKER_V652 in line
        ),
        None,
    )
    if marker_index is not None:
        z_index = next(
            (i for i in range(marker_index + 1, len(lines)) if "premium_index_z_7d" in lines[i] and "_z(" in lines[i]),
            None,
        )
        if z_index is None:
            raise RuntimeError("existing premium z-score block was not found")
        end = _paren_block_end(lines, z_index)
        lines[marker_index : end + 1] = _canonical_engine_lines(_indent(lines[marker_index]))
    else:
        start = next(
            (
                i
                for i, line in enumerate(lines)
                if "for col in" in line
                and "premium_index_close" in line
                and "premium_index_change_1h" in line
            ),
            None,
        )
        if start is None:
            raise RuntimeError("canonical V22 premium block was not found")
        z_index = next(
            (i for i in range(start, min(len(lines), start + 16)) if "premium_index_z_7d" in lines[i]),
            None,
        )
        if z_index is None:
            raise RuntimeError("canonical V22 premium z-score line was not found")
        end = _paren_block_end(lines, z_index)
        lines[start : end + 1] = _canonical_engine_lines(_indent(lines[start]))

    updated = "\n".join(lines) + ("\n" if text.endswith(("\n", "\r")) else "")
    if updated.count(ENGINE_MARKER) != 1:
        raise RuntimeError("V6.5.3 engine marker count is not exactly one")
    if ENGINE_MARKER_OLD in updated or ENGINE_MARKER_V652 in updated:
        raise RuntimeError("superseded engine marker remains")
    return updated, updated != text


def _function_bounds(lines: list[str], name: str) -> tuple[int, int]:
    start = next((i for i, line in enumerate(lines) if line.startswith(f"def {name}(")), None)
    if start is None:
        raise RuntimeError(f"{name} was not found")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("def ")), len(lines))
    return start, end


def patch_guard_text(text: str) -> tuple[str, bool]:
    lines = text.splitlines()
    changed = False

    if any(GUARD_MARKER_V652 in line for line in lines):
        lines = [line.replace(GUARD_MARKER_V652, GUARD_MARKER) for line in lines]
        changed = True
    elif not any(GUARD_MARKER in line for line in lines):
        start, end = _function_bounds(lines, "fetch_binance_premium")
        cursor = next((i for i in range(start, end) if lines[i].strip() == "cursor = int(start_ms)"), None)
        if cursor is None:
            raise RuntimeError("premium fetch cursor=start_ms anchor was not found")
        indent = _indent(lines[cursor])
        lines[cursor : cursor + 1] = [
            f"{indent}# {GUARD_MARKER}",
            f"{indent}# Four hidden bars make a bounded 1h derivative identical",
            f"{indent}# to the full raw-premium derivation.",
            f"{indent}_requested_start_ms = int(start_ms)",
            f"{indent}_premium_warmup_start_ms = max(0, _requested_start_ms - 4 * BAR_15M_MS)",
            f"{indent}cursor = _premium_warmup_start_ms",
        ]
        changed = True

        start, end = _function_bounds(lines, "fetch_binance_premium")
        derivative = next(
            (i for i in range(start, end) if "premium_index_change_1h" in lines[i] and "=" in lines[i]),
            None,
        )
        if derivative is None:
            raise RuntimeError("premium derivative assignment was not found")
        derivative_end = _paren_block_end(lines, derivative)
        dindent = _indent(lines[derivative])
        lines[derivative : derivative_end + 1] = [
            f'{dindent}out["premium_index_change_1h"] = (',
            f'{dindent}    out["premium_index_close"].pct_change(periods=4, fill_method=None)',
            f"{dindent}    .replace([np.inf, -np.inf], 0.0)",
            f"{dindent}    .fillna(0.0)",
            f"{dindent})",
            f'{dindent}out = out.loc[out["timestamp"] >= _requested_start_ms].reset_index(drop=True)',
        ]

    updated = "\n".join(lines) + ("\n" if text.endswith(("\n", "\r")) else "")
    if updated.count(GUARD_MARKER) != 1:
        raise RuntimeError("V6.5.3 guard marker count is not exactly one")
    for token in (
        "_requested_start_ms - 4 * BAR_15M_MS",
        'out = out.loc[out["timestamp"] >= _requested_start_ms]',
        "pct_change(periods=4, fill_method=None)",
    ):
        if token not in updated:
            raise RuntimeError(f"guard contract missing: {token}")
    return updated, changed or updated != text


def patch_file(path: Path, kind: str) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    before = path.read_text(encoding="utf-8-sig")
    after, changed = patch_engine_text(before) if kind == "engine" else patch_guard_text(before)
    if changed:
        tmp = path.with_suffix(path.suffix + ".v653.tmp")
        tmp.write_text(after, encoding="utf-8", newline="\n")
        tmp.replace(path)
    return {"path": str(path), "kind": kind, "changed": changed}


def canonicalize_premium_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        raise RuntimeError("premium parquet is empty")
    for column in ("timestamp", "premium_index_close"):
        if column not in frame.columns:
            raise RuntimeError(f"premium parquet missing required column: {column}")

    original_rows = int(len(frame))
    work = frame.copy()
    work["timestamp"] = pd.to_numeric(work["timestamp"], errors="coerce")
    if bool(work["timestamp"].isna().any()):
        raise RuntimeError("premium parquet contains invalid timestamps")
    work["timestamp"] = work["timestamp"].astype(np.int64)
    work = work.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    close_before = pd.to_numeric(work["premium_index_close"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    old_change = (
        pd.to_numeric(work["premium_index_change_1h"], errors="coerce")
        if "premium_index_change_1h" in work.columns
        else pd.Series(np.nan, index=work.index, dtype=float)
    ).replace([np.inf, -np.inf], np.nan)
    new_change = (
        close_before.pct_change(periods=4, fill_method=None)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .astype(float)
    )

    diff = (old_change.fillna(0.0) - new_change).abs()
    work["premium_index_change_1h"] = new_change

    close_after = pd.to_numeric(work["premium_index_close"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not close_before.equals(close_after):
        raise RuntimeError("premium close changed during derivative canonicalization")

    return work, {
        "rows_before": original_rows,
        "rows_after": int(len(work)),
        "duplicates_removed": int(original_rows - len(work)),
        "changed_derivative_rows": int((diff > 1e-15).sum()),
        "max_abs_derivative_change": float(diff.max()) if len(diff) else 0.0,
        "latest_timestamp": int(work["timestamp"].iloc[-1]),
    }


def canonicalize_premium_file(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    canonical, stats = canonicalize_premium_frame(frame)
    tmp = path.with_suffix(path.suffix + ".v653.tmp")
    canonical.to_parquet(tmp, index=False)
    verify = pd.read_parquet(tmp)
    if int(len(verify)) != int(len(canonical)):
        tmp.unlink(missing_ok=True)
        raise RuntimeError("temporary premium parquet row-count verification failed")
    tmp.replace(path)
    return {"path": str(path), **stats}


def load_guard_module(guard_path: Path) -> Any:
    guard_path = Path(guard_path).resolve()
    spec = importlib.util.spec_from_file_location("rithal_tail_guard_v653_active", str(guard_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import active guard: {guard_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "raw_premium_path"):
        raise RuntimeError("active guard has no raw_premium_path function")
    return module


def resolve_premium_paths(guard_path: Path, data_root: Path, symbols: Sequence[str]) -> dict[str, str]:
    module = load_guard_module(guard_path)
    root = Path(data_root).resolve()
    return {symbol: str(Path(module.raw_premium_path(root, symbol)).resolve()) for symbol in symbols}


def canonicalize_symbols(guard_path: Path, data_root: Path, symbols: Sequence[str]) -> dict[str, Any]:
    paths = resolve_premium_paths(guard_path, data_root, symbols)
    results: dict[str, Any] = {}
    for symbol, raw_path in paths.items():
        results[symbol] = canonicalize_premium_file(Path(raw_path))
    return {"version": VERSION, "status": "PASS", "results": results}


def self_test() -> dict[str, Any]:
    timestamps = np.arange(64, dtype=np.int64) * 15 * 60 * 1000
    close = pd.Series(np.linspace(0.001, 0.003, 64) + np.sin(np.arange(64)) * 0.0002)
    canonical = close.pct_change(periods=4, fill_method=None).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    stale = canonical.copy()
    for start in range(0, 64, 12):
        chunk = close.iloc[start : start + 12]
        stale.iloc[start : start + 12] = chunk.pct_change(periods=4, fill_method=None).replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy()
    assert not np.array_equal(stale.to_numpy(), canonical.to_numpy())

    frame = pd.DataFrame({
        "timestamp": timestamps,
        "premium_index_close": close,
        "premium_source_covered": np.ones(64),
        "premium_index_change_1h": stale,
    })
    repaired, stats = canonicalize_premium_frame(frame)
    assert np.array_equal(repaired["timestamp"].to_numpy(), timestamps)
    assert np.array_equal(repaired["premium_index_close"].to_numpy(), close.to_numpy())
    assert np.array_equal(repaired["premium_source_covered"].to_numpy(), np.ones(64))
    assert np.array_equal(repaired["premium_index_change_1h"].to_numpy(), canonical.to_numpy())
    assert stats["changed_derivative_rows"] > 0

    engine_fixture = '''        # RITHAL_TAIL_GUARD_V6_5_1_PREMIUM_ALIGNMENT
        _premium_close_raw = (
            pd.to_numeric(df["premium_index_close"], errors="coerce")
            if "premium_index_close" in df.columns
            else pd.Series(np.nan, index=df.index, dtype=float)
        ).replace([np.inf, -np.inf], np.nan)
        df["premium_index_close"] = _premium_close_raw.fillna(0.0)
        df["premium_index_change_1h"] = (
            _premium_close_raw.pct_change(periods=4, fill_method=None)
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
        )
        df["premium_index_z_7d"] = _z(
            _premium_close_raw.ffill().fillna(0.0), 7 * 96
        )
'''
    patched_engine, changed = patch_engine_text(engine_fixture)
    assert changed and ENGINE_MARKER in patched_engine
    patched_engine2, changed2 = patch_engine_text(patched_engine)
    assert not changed2 and patched_engine2 == patched_engine

    guard_fixture = '''from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
BAR_15M_MS=900000
def fetch_binance_premium(client, symbol, start_ms, end_open_ms) -> pd.DataFrame:
    cursor = int(start_ms)
    rows=[]
    out=pd.DataFrame(rows)
    out["premium_index_change_1h"] = (
        out["premium_index_close"].pct_change(periods=4, fill_method=None)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )
    return out

def raw_premium_path(root, symbol):
    return Path(root) / f"{symbol}.parquet"

def main():
    pass
'''
    patched_guard, guard_changed = patch_guard_text(guard_fixture)
    assert guard_changed and GUARD_MARKER in patched_guard
    patched_guard2, guard_changed2 = patch_guard_text(patched_guard)
    assert not guard_changed2 and patched_guard2 == patched_guard

    synthetic = (
        "import numpy as np\nimport pandas as pd\n"
        "def _numeric(df,col,default=0.0): return pd.to_numeric(df[col],errors='coerce').fillna(default)\n"
        "def _z(s,n): return s*0.0\n"
        "class Engine:\n"
        "    def build(self,df):\n" + patched_engine + "        return df\n"
    )
    with tempfile.TemporaryDirectory() as td:
        source = Path(td) / "engine.py"
        source.write_text(synthetic, encoding="utf-8")
        py_compile.compile(str(source), doraise=True)

    return {
        "version": VERSION,
        "status": "PASS",
        "checks": [
            "stale_chunk_boundaries_reproduced",
            "full_raw_state_derivative_repaired",
            "timestamp_close_coverage_unchanged",
            "future_fetch_four_bar_warmup",
            "engine_contract_restored",
            "source_patch_idempotent",
            "synthetic_compile",
        ],
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("--engine")
    parser.add_argument("--guard")
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
        report = {
            "version": VERSION,
            "status": "PASS",
            "engine": patch_file(Path(args.engine), "engine"),
            "guard": patch_file(Path(args.guard), "guard"),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.list_premium_paths:
        if not args.guard or not args.data_root:
            parser.error("--list-premium-paths requires --guard and --data-root")
        print(json.dumps(resolve_premium_paths(Path(args.guard), Path(args.data_root), symbols), indent=2, sort_keys=True))
        return 0
    if args.canonicalize_premium_state:
        if not args.guard or not args.data_root:
            parser.error("--canonicalize-premium-state requires --guard and --data-root")
        print(json.dumps(canonicalize_symbols(Path(args.guard), Path(args.data_root), symbols), indent=2, sort_keys=True))
        return 0
    parser.error("select --self-test, --patch-source, --list-premium-paths, or --canonicalize-premium-state")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
