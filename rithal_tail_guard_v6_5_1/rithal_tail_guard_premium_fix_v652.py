from __future__ import annotations

"""Rithal Tail Guard V6.5.2 canonical premium parity repair.

The frozen V22 contract computes ``premium_index_change_1h`` in the raw premium
frame before merging it into the enriched frame. V6.5.1 changed that definition
by deriving it after final alignment. V6.5.2 restores the original engine
semantics and fixes the real bounded-window problem: fetch four hidden 15-minute
predecessor bars, derive the one-hour change, then trim to the requested window.
"""

import argparse
import json
import py_compile
import tempfile
from pathlib import Path
from typing import Optional

VERSION = "RITHAL_TAIL_GUARD_V6_5_2"
ENGINE_MARKER_OLD = "RITHAL_TAIL_GUARD_V6_5_1_PREMIUM_ALIGNMENT"
ENGINE_MARKER = "RITHAL_TAIL_GUARD_V6_5_2_CANONICAL_PREMIUM"
GUARD_MARKER = "RITHAL_TAIL_GUARD_V6_5_2_PREMIUM_WARMUP"


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
    raise RuntimeError(f"unterminated expression starting at line {start + 1}")


def canonical_engine_lines(indent: str) -> list[str]:
    child = indent + "    "
    return [
        f"{indent}# {ENGINE_MARKER}",
        f"{indent}# Frozen V22 semantics: derive change in the raw premium frame,",
        f"{indent}# then carry close and change through the same merge contract.",
        f'{indent}for col in ["premium_index_close", "premium_index_change_1h"]:',
        f"{child}df[col] = _numeric(df, col, 0.0)",
        f'{indent}df["premium_index_available"] = (df["premium_index_close"].abs() > 0).astype(float)',
        f'{indent}df["premium_index_z_7d"] = _z(df["premium_index_close"].replace(0.0, np.nan).ffill().fillna(0.0), 7 * 96)',
    ]


def patch_engine_text(text: str) -> tuple[str, bool]:
    lines = text.splitlines()
    if any(ENGINE_MARKER in line for line in lines) and not any(
        ENGINE_MARKER_OLD in line for line in lines
    ):
        return text, False

    old_index = next((i for i, line in enumerate(lines) if ENGINE_MARKER_OLD in line), None)
    if old_index is not None:
        z_index = next(
            (
                i
                for i in range(old_index + 1, len(lines))
                if 'df["premium_index_z_7d"] = _z(' in lines[i]
                or "df['premium_index_z_7d'] = _z(" in lines[i]
            ),
            None,
        )
        if z_index is None:
            raise RuntimeError("V6.5.1 z-score block was not found")
        end = _paren_block_end(lines, z_index)
        lines[old_index : end + 1] = canonical_engine_lines(_indent(lines[old_index]))
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
            (i for i in range(start, min(len(lines), start + 12)) if "premium_index_z_7d" in lines[i]),
            None,
        )
        if z_index is None:
            raise RuntimeError("canonical V22 premium z-score line was not found")
        end = _paren_block_end(lines, z_index)
        lines[start : end + 1] = canonical_engine_lines(_indent(lines[start]))

    updated = "\n".join(lines) + ("\n" if text.endswith(("\n", "\r")) else "")
    if updated.count(ENGINE_MARKER) != 1:
        raise RuntimeError("V6.5.2 engine marker count is not exactly one")
    if ENGINE_MARKER_OLD in updated:
        raise RuntimeError("V6.5.1 premium semantics remain")
    return updated, updated != text


def _function_bounds(lines: list[str]) -> tuple[int, int]:
    start = next(
        (i for i, line in enumerate(lines) if line.startswith("def fetch_binance_premium(")),
        None,
    )
    if start is None:
        raise RuntimeError("fetch_binance_premium was not found")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("def ")), len(lines))
    return start, end


def patch_guard_text(text: str) -> tuple[str, bool]:
    lines = text.splitlines()
    if any(GUARD_MARKER in line for line in lines):
        return text, False

    start, end = _function_bounds(lines)
    cursor = next(
        (i for i in range(start, end) if lines[i].strip() == "cursor = int(start_ms)"),
        None,
    )
    if cursor is None:
        raise RuntimeError("premium fetch cursor=start_ms anchor was not found")
    indent = _indent(lines[cursor])
    lines[cursor : cursor + 1] = [
        f"{indent}# {GUARD_MARKER}",
        f"{indent}# Four hidden bars are required because V22 derives a 1h",
        f"{indent}# pct_change in the raw premium frame before publication trim.",
        f"{indent}_requested_start_ms = int(start_ms)",
        f"{indent}_premium_warmup_start_ms = max(0, _requested_start_ms - 4 * BAR_15M_MS)",
        f"{indent}cursor = _premium_warmup_start_ms",
    ]

    start, end = _function_bounds(lines)
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
        raise RuntimeError("V6.5.2 guard marker count is not exactly one")
    for token in (
        "_requested_start_ms - 4 * BAR_15M_MS",
        'out = out.loc[out["timestamp"] >= _requested_start_ms]',
        "pct_change(periods=4, fill_method=None)",
    ):
        if token not in updated:
            raise RuntimeError(f"guard contract missing: {token}")
    return updated, updated != text


def patch_file(path: Path, kind: str) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    before = path.read_text(encoding="utf-8-sig")
    after, changed = patch_engine_text(before) if kind == "engine" else patch_guard_text(before)
    if changed:
        tmp = path.with_suffix(path.suffix + ".v652.tmp")
        tmp.write_text(after, encoding="utf-8", newline="\n")
        tmp.replace(path)
    return {
        "path": str(path),
        "kind": kind,
        "changed": changed,
        "marker": ENGINE_MARKER if kind == "engine" else GUARD_MARKER,
    }


def self_test() -> dict:
    import numpy as np
    import pandas as pd

    timestamps = pd.Series(np.arange(20, dtype=np.int64) * 15 * 60 * 1000)
    closes = pd.Series(
        [0.0010, 0.0012, 0.0011, 0.0013, 0.0014, 0.0011, 0.0015, 0.0012,
         0.0016, 0.0017, 0.0013, 0.0018, 0.0019, 0.0014, 0.0020, 0.0016,
         0.0021, 0.0022, 0.0018, 0.0023], dtype=float,
    )
    full = pd.DataFrame({"timestamp": timestamps, "premium_index_close": closes})
    full["premium_index_change_1h"] = (
        full["premium_index_close"].pct_change(periods=4, fill_method=None)
        .replace([np.inf, -np.inf], 0.0).fillna(0.0)
    )
    requested_start = int(full.loc[8, "timestamp"])
    expected = full.loc[full["timestamp"] >= requested_start, "premium_index_change_1h"].reset_index(drop=True)

    broken = full.loc[full["timestamp"] >= requested_start, ["timestamp", "premium_index_close"]].copy()
    broken["premium_index_change_1h"] = (
        broken["premium_index_close"].pct_change(periods=4, fill_method=None)
        .replace([np.inf, -np.inf], 0.0).fillna(0.0)
    )
    assert not np.array_equal(broken["premium_index_change_1h"].to_numpy(), expected.to_numpy())

    warmup_start = requested_start - 4 * 15 * 60 * 1000
    repaired = full.loc[full["timestamp"] >= warmup_start, ["timestamp", "premium_index_close"]].copy()
    repaired["premium_index_change_1h"] = (
        repaired["premium_index_close"].pct_change(periods=4, fill_method=None)
        .replace([np.inf, -np.inf], 0.0).fillna(0.0)
    )
    repaired = repaired.loc[repaired["timestamp"] >= requested_start].reset_index(drop=True)
    assert np.array_equal(repaired["premium_index_change_1h"].to_numpy(), expected.to_numpy())

    v651 = '''        # RITHAL_TAIL_GUARD_V6_5_1_PREMIUM_ALIGNMENT
        _premium_close_raw = (
            pd.to_numeric(df["premium_index_close"], errors="coerce")
            if "premium_index_close" in df.columns
            else pd.Series(np.nan, index=df.index, dtype=float)
        ).replace([np.inf, -np.inf], np.nan)
        _premium_covered_raw = (
            pd.to_numeric(df["premium_source_covered"], errors="coerce")
            if "premium_source_covered" in df.columns
            else pd.Series(np.nan, index=df.index, dtype=float)
        )
        df["premium_index_close"] = _premium_close_raw.fillna(0.0)
        df["premium_index_change_1h"] = (
            _premium_close_raw.pct_change(periods=4, fill_method=None)
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
        )
        if bool(_premium_covered_raw.notna().any()):
            df["premium_index_available"] = (
                _premium_covered_raw.fillna(0.0) >= 0.5
            ).astype(float)
        else:
            df["premium_index_available"] = _premium_close_raw.notna().astype(float)
        df["premium_index_z_7d"] = _z(
            _premium_close_raw.ffill().fillna(0.0), 7 * 96
        )
'''
    restored, changed = patch_engine_text(v651)
    assert changed and ENGINE_MARKER in restored and ENGINE_MARKER_OLD not in restored
    restored2, changed2 = patch_engine_text(restored)
    assert not changed2 and restored2 == restored

    guard = '''def fetch_binance_premium(client, symbol, start_ms, end_open_ms) -> pd.DataFrame:
    end_time = int(end_open_ms + BAR_15M_MS - 1)
    cursor = int(start_ms)
    rows = []
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    out["premium_index_change_1h"] = (
        out["premium_index_close"].pct_change(periods=4, fill_method=None)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )
    return out


def next_function():
    pass
'''
    guard_patched, guard_changed = patch_guard_text(guard)
    assert guard_changed and GUARD_MARKER in guard_patched
    guard2, guard_changed2 = patch_guard_text(guard_patched)
    assert not guard_changed2 and guard2 == guard_patched

    synthetic = (
        "import numpy as np\nimport pandas as pd\n\n"
        "def _numeric(df, col, default=0.0):\n"
        "    return pd.to_numeric(df[col], errors='coerce').fillna(default) if col in df else pd.Series(default, index=df.index)\n\n"
        "def _z(s, n): return s * 0.0\n\n"
        "class Engine:\n"
        "    def build(self, df):\n" + restored + "        return df\n"
    )
    with tempfile.TemporaryDirectory() as td:
        source = Path(td) / "synthetic_v22.py"
        source.write_text(synthetic, encoding="utf-8")
        py_compile.compile(str(source), doraise=True)

    return {
        "version": VERSION,
        "status": "PASS",
        "checks": [
            "canonical_v22_engine_semantics_restored",
            "four_bar_hidden_warmup",
            "trim_after_raw_derivative",
            "bounded_equals_full_exactly",
            "engine_patch_idempotent",
            "guard_patch_idempotent",
            "synthetic_compile",
        ],
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("--engine")
    parser.add_argument("--guard")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return 0
    if not args.engine or not args.guard:
        parser.error("--engine and --guard are required unless --self-test is used")
    print(json.dumps({
        "version": VERSION,
        "status": "PASS",
        "engine": patch_file(Path(args.engine), "engine"),
        "guard": patch_file(Path(args.guard), "guard"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
