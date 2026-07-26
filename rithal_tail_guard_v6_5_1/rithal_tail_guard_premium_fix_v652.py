from __future__ import annotations

"""Rithal Tail Guard V6.5.2 canonical premium parity repair.

The frozen V22 archive/checkpoint contract computes ``premium_index_change_1h``
in the raw premium frame and then merges that derived column into the enriched
frame. V6.5.1 incorrectly moved the derivative after final alignment, changing
feature semantics relative to the immutable five-year archive.

V6.5.2 restores the exact V22 engine block and fixes the actual bounded-window
bug: the premium fetch obtains four hidden 15-minute warm-up rows, computes the
one-hour derivative on that expanded frame, and only then trims to the requested
publication window. This makes bounded and full-history derivations identical
without weakening parity validation or rebuilding the canonical archive.
"""

import argparse
import json
import py_compile
import re
import tempfile
from pathlib import Path
from typing import Optional

VERSION = "RITHAL_TAIL_GUARD_V6_5_2"
ENGINE_MARKER_OLD = "RITHAL_TAIL_GUARD_V6_5_1_PREMIUM_ALIGNMENT"
ENGINE_MARKER = "RITHAL_TAIL_GUARD_V6_5_2_CANONICAL_PREMIUM"
GUARD_MARKER = "RITHAL_TAIL_GUARD_V6_5_2_PREMIUM_WARMUP"
BAR_EXPR = "BAR_15M_MS"

# The V6.5.1 replacement starts at its marker and ends after the z-score block.
V651_ENGINE_BLOCK = re.compile(
    r"(?P<indent>^[ \t]+)#\s+RITHAL_TAIL_GUARD_V6_5_1_PREMIUM_ALIGNMENT\s*\r?\n"
    r".*?"
    r"(?P=indent)df\[[\"']premium_index_z_7d[\"']\]\s*=\s*_z\(\s*\r?\n"
    r".*?"
    r"(?P=indent)\)\s*(?:\r?\n|$)",
    re.MULTILINE | re.DOTALL,
)

CANONICAL_ENGINE_PATTERN = re.compile(
    r"(?P<indent>^[ \t]+)for\s+col\s+in\s+\[\s*[\"']premium_index_close[\"']\s*,\s*[\"']premium_index_change_1h[\"']\s*\]\s*:\s*\r?\n"
    r"(?P=indent)[ \t]+df\[col\]\s*=\s*_numeric\(df,\s*col,\s*0\.0\)\s*\r?\n"
    r"(?P=indent)df\[[\"']premium_index_available[\"']\]\s*=\s*\(df\[[\"']premium_index_close[\"']\]\.abs\(\)\s*>\s*0\)\.astype\(float\)\s*\r?\n"
    r"(?P=indent)df\[[\"']premium_index_z_7d[\"']\]\s*=\s*_z\(df\[[\"']premium_index_close[\"']\]\.replace\(0\.0,\s*np\.nan\)\.ffill\(\)\.fillna\(0\.0\),\s*7\s*\*\s*96\)",
    re.MULTILINE,
)


def canonical_engine_block(indent: str) -> str:
    i = indent
    j = indent + "    "
    return "\n".join(
        [
            f"{i}# {ENGINE_MARKER}",
            f"{i}# Frozen V22 semantics: derive change in the raw premium frame,",
            f"{i}# then carry both close and change through the same merge contract.",
            f'{i}for col in ["premium_index_close", "premium_index_change_1h"]:',
            f"{j}df[col] = _numeric(df, col, 0.0)",
            f'{i}df["premium_index_available"] = (df["premium_index_close"].abs() > 0).astype(float)',
            f'{i}df["premium_index_z_7d"] = _z(df["premium_index_close"].replace(0.0, np.nan).ffill().fillna(0.0), 7 * 96)',
        ]
    )


def patch_engine_text(text: str) -> tuple[str, bool]:
    if ENGINE_MARKER in text and ENGINE_MARKER_OLD not in text:
        return text, False

    old = V651_ENGINE_BLOCK.search(text)
    if old:
        updated = V651_ENGINE_BLOCK.sub(
            canonical_engine_block(old.group("indent")), text, count=1
        )
    else:
        canonical = CANONICAL_ENGINE_PATTERN.search(text)
        if not canonical:
            raise RuntimeError(
                "neither the V6.5.1 premium block nor the canonical V22 block was found"
            )
        updated = CANONICAL_ENGINE_PATTERN.sub(
            canonical_engine_block(canonical.group("indent")), text, count=1
        )

    if updated.count(ENGINE_MARKER) != 1:
        raise RuntimeError("V6.5.2 engine marker count is not exactly one")
    if ENGINE_MARKER_OLD in updated:
        raise RuntimeError("V6.5.1 engine marker remains after canonical restoration")
    return updated, updated != text


def _function_window(text: str) -> tuple[int, int, str]:
    match = re.search(
        r"(?m)^def\s+fetch_binance_premium\([^\n]*\)\s*->\s*pd\.DataFrame\s*:\s*$",
        text,
    )
    if not match:
        raise RuntimeError("fetch_binance_premium function was not found")
    next_def = re.search(r"(?m)^def\s+", text[match.end():])
    end = match.end() + (next_def.start() if next_def else len(text) - match.end())
    return match.start(), end, text[match.start():end]


def patch_guard_text(text: str) -> tuple[str, bool]:
    if GUARD_MARKER in text:
        return text, False

    start, end, fn = _function_window(text)

    cursor_pattern = re.compile(r"(?m)^(?P<indent>[ \t]+)cursor\s*=\s*int\(start_ms\)\s*$")
    cursor = cursor_pattern.search(fn)
    if not cursor:
        raise RuntimeError("premium fetch cursor=start_ms anchor was not found")
    indent = cursor.group("indent")
    cursor_replacement = "\n".join(
        [
            f"{indent}# {GUARD_MARKER}",
            f"{indent}# Four hidden bars are required because V22 derives a 1h",
            f"{indent}# pct_change in the raw premium frame before publication trim.",
            f"{indent}_requested_start_ms = int(start_ms)",
            f"{indent}_premium_warmup_start_ms = max(0, _requested_start_ms - 4 * {BAR_EXPR})",
            f"{indent}cursor = _premium_warmup_start_ms",
        ]
    )
    fn = cursor_pattern.sub(cursor_replacement, fn, count=1)

    # Accept either the original compact expression or the explicit V6.5.1 form.
    derivative_pattern = re.compile(
        r"(?ms)^(?P<indent>[ \t]+)out\[[\"']premium_index_change_1h[\"']\]\s*=\s*\(?.*?pct_change\((?:periods\s*=\s*)?4(?:\s*,\s*fill_method\s*=\s*None)?\).*?fillna\(0\.0\)\s*\)?\s*$"
    )
    derivative = derivative_pattern.search(fn)
    if not derivative:
        raise RuntimeError("premium derivative expression was not found")
    dindent = derivative.group("indent")
    canonical_derivative = "\n".join(
        [
            f'{dindent}out["premium_index_change_1h"] = (',
            f'{dindent}    out["premium_index_close"].pct_change(periods=4, fill_method=None)',
            f"{dindent}    .replace([np.inf, -np.inf], 0.0)",
            f"{dindent}    .fillna(0.0)",
            f"{dindent})",
            f'{dindent}out = out.loc[out["timestamp"] >= _requested_start_ms].reset_index(drop=True)',
        ]
    )
    fn = derivative_pattern.sub(canonical_derivative, fn, count=1)

    updated = text[:start] + fn + text[end:]
    if updated.count(GUARD_MARKER) != 1:
        raise RuntimeError("V6.5.2 guard marker count is not exactly one")
    if "_premium_warmup_start_ms" not in updated:
        raise RuntimeError("premium warmup cursor was not installed")
    if 'out = out.loc[out["timestamp"] >= _requested_start_ms]' not in updated:
        raise RuntimeError("premium publication trim was not installed")
    return updated, updated != text


def patch_file(path: Path, kind: str) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    before = path.read_text(encoding="utf-8-sig")
    if kind == "engine":
        after, changed = patch_engine_text(before)
    elif kind == "guard":
        after, changed = patch_guard_text(before)
    else:
        raise ValueError(kind)
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
         0.0021, 0.0022, 0.0018, 0.0023],
        dtype=float,
    )
    full = pd.DataFrame({"timestamp": timestamps, "premium_index_close": closes})
    full["premium_index_change_1h"] = (
        full["premium_index_close"]
        .pct_change(periods=4, fill_method=None)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )

    requested_start = int(full.loc[8, "timestamp"])
    broken = full.loc[full["timestamp"] >= requested_start, ["timestamp", "premium_index_close"]].copy()
    broken["premium_index_change_1h"] = (
        broken["premium_index_close"]
        .pct_change(periods=4, fill_method=None)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )
    expected = full.loc[full["timestamp"] >= requested_start, "premium_index_change_1h"].reset_index(drop=True)
    assert not np.allclose(broken["premium_index_change_1h"].to_numpy(), expected.to_numpy())

    warmup_start = requested_start - 4 * 15 * 60 * 1000
    repaired = full.loc[full["timestamp"] >= warmup_start, ["timestamp", "premium_index_close"]].copy()
    repaired["premium_index_change_1h"] = (
        repaired["premium_index_close"]
        .pct_change(periods=4, fill_method=None)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )
    repaired = repaired.loc[repaired["timestamp"] >= requested_start].reset_index(drop=True)
    assert np.allclose(repaired["premium_index_change_1h"].to_numpy(), expected.to_numpy(), atol=0.0, rtol=0.0)

    v651 = '''        # RITHAL_TAIL_GUARD_V6_5_1_PREMIUM_ALIGNMENT
        # Derive every premium field from the final engine-aligned 15m close.
        # A zero premium is a valid covered observation, not a missing feed.
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
    assert changed
    assert ENGINE_MARKER in restored and ENGINE_MARKER_OLD not in restored
    assert 'for col in ["premium_index_close", "premium_index_change_1h"]' in restored
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
    assert guard_changed
    assert GUARD_MARKER in guard_patched
    assert "_requested_start_ms - 4 * BAR_15M_MS" in guard_patched
    assert 'out = out.loc[out["timestamp"] >= _requested_start_ms]' in guard_patched
    guard2, guard_changed2 = patch_guard_text(guard_patched)
    assert not guard_changed2 and guard2 == guard_patched

    synthetic = (
        "import numpy as np\nimport pandas as pd\n\n"
        "def _numeric(df, col, default=0.0):\n"
        "    return pd.to_numeric(df[col], errors='coerce').fillna(default) if col in df else pd.Series(default, index=df.index)\n\n"
        "def _z(s, n): return s * 0.0\n\n"
        "class Engine:\n"
        "    def build(self, df):\n"
        + restored
        + "        return df\n"
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
    report = {
        "version": VERSION,
        "status": "PASS",
        "engine": patch_file(Path(args.engine), "engine"),
        "guard": patch_file(Path(args.guard), "guard"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
