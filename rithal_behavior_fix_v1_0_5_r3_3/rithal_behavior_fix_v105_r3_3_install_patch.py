from __future__ import annotations

"""Line-ending-safe, exact-scope installer patch for Rithal V1.0.5 R3.3."""

import argparse
import json
import py_compile
import re
import tempfile
from pathlib import Path
from typing import Optional

VERSION = "RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3_INSTALL_PATCH"
LOGGER_MARKER = "RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3_SAFE_REGIME_LOG"
ACTIVATION_START = "RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3_ACTIVATION_START"
ACTIVATION_END = "RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3_ACTIVATION_END"

LOGGER_PATTERN = re.compile(
    r'(?m)^(?P<i>[ \t]+)_rp\s*=\s*self\._get_regime_probs\(sym,\s*pred\)\s*\r?\n'
    r'(?P=i)L\("  │ REGIME-P:   p_trend=%\.3f  p_chop=%\.3f  p_breakout=%\.3f   '
    r'\(chop gate %\.2f / panic gate %\.2f\)",\s*\r?\n'
    r'(?P=i)[ \t]+_rp\.get\("p_trend",\s*0\.0\),\s*_rp\.get\("p_chop",\s*0\.0\),\s*'
    r'_rp\.get\("p_panic",\s*0\.0\),\s*\r?\n'
    r'(?P=i)[ \t]+self\.CHOP_GATE_THRESHOLD,\s*self\.PANIC_GATE_THRESHOLD\)'
)

ACTIVATION_BLOCK = '''

# RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3_ACTIVATION_START
try:
    from .rithal_behavior_fix_v105_r3_3 import apply_live_patch as _rithal_v105_r33_apply_live_patch
except ImportError:
    from rithal_behavior_fix_v105_r3_3 import apply_live_patch as _rithal_v105_r33_apply_live_patch
_rithal_v105_r33_apply_live_patch(globals())
# RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3_ACTIVATION_END
'''


def _logger_replacement(indent: str, newline: str) -> str:
    lines = [
        f'{indent}_rp = self._get_regime_probs(sym, pred) or {{}}',
        f'{indent}# {LOGGER_MARKER}',
        f'{indent}def _r33_regime_log_number(name):',
        f'{indent}    try:',
        f'{indent}        value = float(_rp.get(name, float("nan")))',
        f'{indent}        return value if np.isfinite(value) else float("nan")',
        f'{indent}    except Exception:',
        f'{indent}        return float("nan")',
        f'{indent}L("  │ REGIME-P:   p_trend=%.3f  p_chop=%.3f  p_breakout=%.3f   (chop gate %.2f / panic gate %.2f)",',
        f'{indent}  _r33_regime_log_number("p_trend"), _r33_regime_log_number("p_chop"),',
        f'{indent}  _r33_regime_log_number("p_breakout"),',
        f'{indent}  self.CHOP_GATE_THRESHOLD, self.PANIC_GATE_THRESHOLD)',
    ]
    return newline.join(lines)


def patch_text(text: str) -> tuple[str, dict]:
    original = text
    newline = "\r\n" if text.count("\r\n") > text.count("\n") / 2 else "\n"

    logger_existing = text.count(LOGGER_MARKER)
    if logger_existing > 1:
        raise RuntimeError(f"safe logger marker appears {logger_existing} times")
    logger_changed = False
    if logger_existing == 0:
        matches = list(LOGGER_PATTERN.finditer(text))
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one legacy regime logger block, found {len(matches)}")
        match = matches[0]
        replacement = _logger_replacement(match.group("i"), newline)
        text = text[: match.start()] + replacement + text[match.end() :]
        logger_changed = True

    activation_count = text.count(ACTIVATION_START)
    if activation_count > 1:
        raise RuntimeError(f"activation marker appears {activation_count} times")
    activation_changed = False
    if activation_count == 0:
        block = ACTIVATION_BLOCK.replace("\n", newline)
        text = text.rstrip() + block + newline
        activation_changed = True

    if text.count(LOGGER_MARKER) != 1:
        raise RuntimeError("safe logger marker contract failed")
    if text.count(ACTIVATION_START) != 1 or text.count(ACTIVATION_END) != 1:
        raise RuntimeError("activation marker contract failed")
    if '_rp.get("p_panic", 0.0)' in text:
        raise RuntimeError("legacy panic-as-breakout logger remains")
    if '_r33_regime_log_number("p_breakout")' not in text:
        raise RuntimeError("breakout logger source missing")

    return text, {
        "changed": text != original,
        "logger_changed": logger_changed,
        "activation_changed": activation_changed,
        "newline": "CRLF" if newline == "\r\n" else "LF",
    }


def patch_file(path: Path) -> dict:
    path = Path(path).resolve()
    before = path.read_text(encoding="utf-8-sig")
    after, report = patch_text(before)
    if report["changed"]:
        temp = path.with_suffix(path.suffix + ".r33.tmp")
        temp.write_text(after, encoding="utf-8", newline="")
        temp.replace(path)
    return {"path": str(path), **report}


def self_test() -> dict:
    body = '''import numpy as np
class NeuralV2Model: pass
class LiveEngine:
    CHOP_GATE_THRESHOLD=0.60
    PANIC_GATE_THRESHOLD=0.50
    def _get_regime_probs(self, sym, pred): return pred
    def _log_rich_bar(self, sym, df, last_bar, pred):
        def L(*args): return args
        try:
            _rp = self._get_regime_probs(sym, pred)
            L("  │ REGIME-P:   p_trend=%.3f  p_chop=%.3f  p_breakout=%.3f   (chop gate %.2f / panic gate %.2f)",
              _rp.get("p_trend", 0.0), _rp.get("p_chop", 0.0), _rp.get("p_panic", 0.0),
              self.CHOP_GATE_THRESHOLD, self.PANIC_GATE_THRESHOLD)
        except Exception:
            pass
'''
    checks = []
    for name, fixture in (("LF", body), ("CRLF", body.replace("\n", "\r\n"))):
        patched, report = patch_text(fixture)
        assert report["logger_changed"] and report["activation_changed"]
        assert patched.count(LOGGER_MARKER) == 1
        assert patched.count(ACTIVATION_START) == 1
        patched2, report2 = patch_text(patched)
        assert patched2 == patched and not report2["changed"]
        # The appended activation imports a project module unavailable in this
        # isolated fixture, so compile only the logger-bearing class segment.
        compile_segment = patched.split(f"# {ACTIVATION_START}", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "live.py"
            source.write_text(compile_segment, encoding="utf-8", newline="")
            py_compile.compile(str(source), doraise=True)
        checks.extend([f"{name.lower()}_patch", f"{name.lower()}_idempotent", f"{name.lower()}_compile"])

    return {"version": VERSION, "status": "PASS", "checks": checks}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--live")
    args = parser.parse_args(argv)
    if args.self_test:
        result = self_test()
    elif args.patch:
        if not args.live:
            parser.error("--live is required with --patch")
        result = {"version": VERSION, "status": "PASS", "result": patch_file(Path(args.live))}
    else:
        parser.error("choose --self-test or --patch")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
