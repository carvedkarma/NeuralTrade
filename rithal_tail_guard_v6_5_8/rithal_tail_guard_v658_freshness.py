from __future__ import annotations

"""Rithal Tail Guard V6.5.8 anchored freshness correction.

A build is anchored to one immutable ``anchor_now_ms``. Production showed that
all six outputs could commit with 100% coverage and exact anchored timestamps,
then fail because ``latest_bar_age_min`` was recalculated from wall-clock time at
the end of a long run. Crossing one 15-minute boundary turned a valid anchored
publication into a false 35-minute age failure.

This delta keeps wall-clock age as telemetry, but validates the Python build with
the same immutable anchor used to select its latest candle. The existing
PowerShell one-bar rollover handshake remains responsible for wall-clock catch-up.
"""

import argparse
import json
import py_compile
import tempfile
from pathlib import Path
from typing import Optional

VERSION = "RITHAL_TAIL_GUARD_V6_5_8"
MARKER = "RITHAL_TAIL_GUARD_V6_5_8_ANCHORED_FRESHNESS"

TAIL_STATUS_OLD = '''    report["tail_status"] = {symbol: tail_ratios(live_enriched_dir, symbol, n=96) for symbol in symbols}
    report["preferred_tail_contract_ok"] = bool(all('''

TAIL_STATUS_NEW = '''    report["tail_status"] = {symbol: tail_ratios(live_enriched_dir, symbol, n=96) for symbol in symbols}
    # RITHAL_TAIL_GUARD_V6_5_8_ANCHORED_FRESHNESS
    # The feature transaction is built against one immutable run anchor.
    # Keep wall-clock age for diagnostics, but judge the anchored publication
    # using age measured from that same anchor. A later one-bar rollover is a
    # separate scheduler concern handled by Start-RithalTailWatch.ps1.
    for _symbol, _status in report["tail_status"].items():
        _latest = int(_status.get("latest_timestamp", -1))
        _status["wall_clock_latest_bar_age_min"] = float(
            _status.get("latest_bar_age_min", 1e9)
        )
        _status["anchored_latest_bar_age_min"] = round(
            max(0.0, (int(anchor_now_ms) - _latest) / 60_000.0), 2
        )
    report["preferred_tail_contract_ok"] = bool(all('''

AGE_OLD = '''        and float(status.get("latest_bar_age_min", 1e9)) <= float(args.max_latest_age_minutes)'''
AGE_NEW = '''        and float(status.get("anchored_latest_bar_age_min", 1e9)) <= float(args.max_latest_age_minutes)'''


def patch_guard_text(text: str) -> tuple[str, bool]:
    if MARKER in text:
        return text, False
    if TAIL_STATUS_OLD not in text:
        raise RuntimeError("tail_status anchor insertion point not found")
    count = text.count(AGE_OLD)
    if count != 1:
        raise RuntimeError(f"expected exactly one wall-clock age gate, found {count}")
    updated = text.replace(TAIL_STATUS_OLD, TAIL_STATUS_NEW, 1)
    updated = updated.replace(AGE_OLD, AGE_NEW, 1)
    updated = updated.replace('"guard_version": "6.5.7"', '"guard_version": "6.5.8"', 1)
    if MARKER not in updated:
        raise RuntimeError("V6.5.8 marker missing after patch")
    if AGE_OLD in updated:
        raise RuntimeError("wall-clock age remains authoritative")
    if AGE_NEW not in updated:
        raise RuntimeError("anchored age gate missing")
    return updated, updated != text


def patch_file(path: Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    before = path.read_text(encoding="utf-8-sig")
    after, changed = patch_guard_text(before)
    if changed:
        tmp = path.with_suffix(path.suffix + ".v658.tmp")
        tmp.write_text(after, encoding="utf-8", newline="\n")
        tmp.replace(path)
    return {"path": str(path), "changed": changed}


def anchored_age_minutes(anchor_now_ms: int, latest_open_ms: int) -> float:
    return max(0.0, (int(anchor_now_ms) - int(latest_open_ms)) / 60_000.0)


def wall_expected_open(now_ms: int, step_ms: int = 900_000) -> int:
    now_ms = int(now_ms)
    return now_ms - (now_ms % step_ms) - step_ms


def self_test() -> dict:
    step = 900_000
    # Synthetic reproduction of the exact production shape:
    # anchor 20:11, newest anchored candle opens 19:45, finish 20:26.
    latest = 19 * 60 * 60_000 + 45 * 60_000
    anchor = 20 * 60 * 60_000 + 11 * 60_000
    finish = 20 * 60 * 60_000 + 26 * 60_000
    anchored_age = anchored_age_minutes(anchor, latest)
    wall_age = anchored_age_minutes(finish, latest)
    assert anchored_age == 26.0
    assert wall_age == 41.0
    assert anchored_age <= 35.0
    assert wall_age > 35.0
    assert wall_expected_open(finish) - latest == step

    fixture = '''from __future__ import annotations
class Args:
    max_latest_age_minutes = 35.0
args = Args()
anchor_now_ms = 72060000
symbols = []
live_enriched_dir = None
def tail_ratios(*args, **kwargs):
    return {}
def check():
    report = {"guard_version": "6.5.7"}
    status = {}
    report["tail_status"] = {symbol: tail_ratios(live_enriched_dir, symbol, n=96) for symbol in symbols}
    report["preferred_tail_contract_ok"] = bool(all(
        True for status in report["tail_status"].values()
    ))
    ok = (
        True
        and float(status.get("latest_bar_age_min", 1e9)) <= float(args.max_latest_age_minutes)
    )
    return report, ok
'''
    patched, changed = patch_guard_text(fixture)
    assert changed
    assert MARKER in patched
    assert "anchored_latest_bar_age_min" in patched
    assert AGE_OLD not in patched
    patched2, changed2 = patch_guard_text(patched)
    assert not changed2 and patched2 == patched

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "guard.py"
        path.write_text(patched, encoding="utf-8")
        py_compile.compile(str(path), doraise=True)

    return {
        "version": VERSION,
        "status": "PASS",
        "simulation": {
            "anchored_age_min": anchored_age,
            "wall_clock_age_min": wall_age,
            "wall_clock_delta_bars": 1,
            "old_wall_clock_contract_passed": False,
            "new_anchored_contract_passed": True,
        },
        "checks": [
            "one_bar_rollover_reproduced",
            "wall_clock_35m_false_negative_reproduced",
            "anchored_35m_contract_passes",
            "wall_clock_age_retained_as_telemetry",
            "patch_compiles",
            "patch_idempotent",
        ],
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--guard")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return 0
    if args.patch:
        if not args.guard:
            parser.error("--guard is required with --patch")
        result = patch_file(Path(args.guard))
        print(json.dumps({"version": VERSION, "status": "PASS", "result": result}, indent=2))
        return 0
    parser.error("choose --self-test or --patch")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
