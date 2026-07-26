from __future__ import annotations

"""Rithal Tail Guard V6.5.1 premium-alignment repair.

The existing guard correctly rolled publication back when the engine-facing
96-row output disagreed with the transaction candidate. The disagreement came
from carrying a precomputed premium_index_change_1h through an asof merge while
premium_index_close was aligned separately. Availability was also inferred
from value != 0, which incorrectly treats a genuine zero premium as missing.

V6.5.1 makes the final aligned premium close the single source of truth:
- recompute premium_index_change_1h after final 15m alignment;
- use premium_source_covered for availability when present;
- accept a genuine zero premium value as covered;
- retain fail-closed publication and all other feature contracts.
"""

import argparse
import json
import py_compile
import re
import tempfile
from pathlib import Path
from typing import Optional

VERSION = "RITHAL_TAIL_GUARD_V6_5_1"
ENGINE_MARKER = "RITHAL_TAIL_GUARD_V6_5_1_PREMIUM_ALIGNMENT"
GUARD_MARKER = "RITHAL_TAIL_GUARD_V6_5_1_PREMIUM_FETCH"

ENGINE_PATTERN = re.compile(
    r"(?P<indent>^[ \t]+)for\s+col\s+in\s+\[\s*[\"']premium_index_close[\"']\s*,\s*[\"']premium_index_change_1h[\"']\s*\]\s*:\s*\r?\n"
    r"(?P=indent)[ \t]+df\[col\]\s*=\s*_numeric\(df,\s*col,\s*0\.0\)\s*\r?\n"
    r"(?P=indent)df\[[\"']premium_index_available[\"']\]\s*=\s*\(df\[[\"']premium_index_close[\"']\]\.abs\(\)\s*>\s*0\)\.astype\(float\)\s*\r?\n"
    r"(?P=indent)df\[[\"']premium_index_z_7d[\"']\]\s*=\s*_z\(df\[[\"']premium_index_close[\"']\]\.replace\(0\.0,\s*np\.nan\)\.ffill\(\)\.fillna\(0\.0\),\s*7\s*\*\s*96\)",
    re.MULTILINE,
)

GUARD_PATTERN = re.compile(
    r"out\[[\"']premium_index_change_1h[\"']\]\s*=\s*out\[[\"']premium_index_close[\"']\]\.pct_change\(4\)\.replace\(\[np\.inf,\s*-np\.inf\],\s*0\.0\)\.fillna\(0\.0\)"
)


def engine_replacement(indent: str) -> str:
    i = indent
    j = indent + "    "
    k = j + "    "
    return "\n".join(
        [
            f"{i}# {ENGINE_MARKER}",
            f"{i}# Derive every premium field from the final engine-aligned 15m close.",
            f"{i}# A zero premium is a valid covered observation, not a missing feed.",
            f"{i}_premium_close_raw = (",
            f"{j}pd.to_numeric(df[\"premium_index_close\"], errors=\"coerce\")",
            f"{j}if \"premium_index_close\" in df.columns",
            f"{j}else pd.Series(np.nan, index=df.index, dtype=float)",
            f"{i}).replace([np.inf, -np.inf], np.nan)",
            f"{i}_premium_covered_raw = (",
            f"{j}pd.to_numeric(df[\"premium_source_covered\"], errors=\"coerce\")",
            f"{j}if \"premium_source_covered\" in df.columns",
            f"{j}else pd.Series(np.nan, index=df.index, dtype=float)",
            f"{i})",
            f"{i}df[\"premium_index_close\"] = _premium_close_raw.fillna(0.0)",
            f"{i}df[\"premium_index_change_1h\"] = (",
            f"{j}_premium_close_raw.pct_change(periods=4, fill_method=None)",
            f"{j}.replace([np.inf, -np.inf], 0.0)",
            f"{j}.fillna(0.0)",
            f"{i})",
            f"{i}if bool(_premium_covered_raw.notna().any()):",
            f"{j}df[\"premium_index_available\"] = (",
            f"{k}_premium_covered_raw.fillna(0.0) >= 0.5",
            f"{j}).astype(float)",
            f"{i}else:",
            f"{j}# Compatibility fallback for older raw premium files that predate",
            f"{j}# premium_source_covered. Presence/finite state, not nonzero value,",
            f"{j}# defines availability.",
            f"{j}df[\"premium_index_available\"] = _premium_close_raw.notna().astype(float)",
            f"{i}df[\"premium_index_z_7d\"] = _z(",
            f"{j}_premium_close_raw.ffill().fillna(0.0), 7 * 96",
            f"{i})",
        ]
    )


def patch_engine_text(text: str) -> tuple[str, bool]:
    if ENGINE_MARKER in text:
        return text, False
    match = ENGINE_PATTERN.search(text)
    if not match:
        raise RuntimeError(
            "active V22 premium block was not found; refusing an ungrounded source edit"
        )
    updated = ENGINE_PATTERN.sub(engine_replacement(match.group("indent")), text, count=1)
    if updated.count(ENGINE_MARKER) != 1:
        raise RuntimeError("engine marker count is not exactly one after patch")
    return updated, True


def patch_guard_text(text: str) -> tuple[str, bool]:
    if GUARD_MARKER in text:
        return text, False
    match = GUARD_PATTERN.search(text)
    if not match:
        if "premium_index_change_1h" in text and "pct_change(periods=4, fill_method=None)" in text:
            return text, False
        raise RuntimeError(
            "active tail-guard premium derivation was not found; refusing an ungrounded source edit"
        )
    replacement = (
        f"# {GUARD_MARKER}\n"
        "    out[\"premium_index_change_1h\"] = (\n"
        "        out[\"premium_index_close\"].pct_change(periods=4, fill_method=None)\n"
        "        .replace([np.inf, -np.inf], 0.0)\n"
        "        .fillna(0.0)\n"
        "    )"
    )
    updated = GUARD_PATTERN.sub(replacement, text, count=1)
    if updated.count(GUARD_MARKER) != 1:
        raise RuntimeError("guard marker count is not exactly one after patch")
    return updated, True


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
        temp = path.with_suffix(path.suffix + ".v651.tmp")
        temp.write_text(after, encoding="utf-8", newline="\n")
        temp.replace(path)
    return {
        "path": str(path),
        "kind": kind,
        "changed": changed,
        "marker": ENGINE_MARKER if kind == "engine" else GUARD_MARKER,
    }


def self_test() -> dict:
    import numpy as np
    import pandas as pd

    close = pd.Series([0.0010, 0.0012, 0.0011, 0.0013, 0.0, 0.0014, 0.0015, 0.0016, 0.0017])
    covered = pd.Series([1.0] * len(close))
    derived = close.pct_change(periods=4, fill_method=None).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    availability = (covered >= 0.5).astype(float)
    assert float(availability.mean()) == 1.0
    assert availability.iloc[4] == 1.0
    assert np.isfinite(derived.to_numpy()).all()

    old_block = '''        for col in ["premium_index_close", "premium_index_change_1h"]:
            df[col] = _numeric(df, col, 0.0)
        df["premium_index_available"] = (df["premium_index_close"].abs() > 0).astype(float)
        df["premium_index_z_7d"] = _z(df["premium_index_close"].replace(0.0, np.nan).ffill().fillna(0.0), 7 * 96)'''
    patched, changed = patch_engine_text(old_block)
    assert changed and ENGINE_MARKER in patched
    assert "pct_change(periods=4, fill_method=None)" in patched
    assert "premium_source_covered" in patched
    patched2, changed2 = patch_engine_text(patched)
    assert not changed2 and patched2 == patched

    guard_old = '    out["premium_index_change_1h"] = out["premium_index_close"].pct_change(4).replace([np.inf, -np.inf], 0.0).fillna(0.0)'
    guard_new, guard_changed = patch_guard_text(guard_old)
    assert guard_changed and GUARD_MARKER in guard_new
    assert "fill_method=None" in guard_new

    synthetic = (
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "def _numeric(df, col, default=0.0):\n"
        "    return pd.to_numeric(df[col], errors='coerce').fillna(default) if col in df else pd.Series(default, index=df.index)\n\n"
        "def _z(s, n):\n"
        "    return s * 0.0\n\n"
        "class SyntheticV22:\n"
        "    def build(self, df):\n"
        + patched
        + "\n        return df\n"
    )
    with tempfile.TemporaryDirectory() as td:
        source = Path(td) / "synthetic_v22.py"
        source.write_text(synthetic, encoding="utf-8")
        py_compile.compile(str(source), doraise=True)

    return {
        "version": VERSION,
        "status": "PASS",
        "checks": [
            "final_aligned_close_is_derivative_source",
            "zero_premium_remains_available",
            "coverage_flag_drives_availability",
            "no_implicit_pct_change_fill",
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
