from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

VERSION = "RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3"


def main() -> int:
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    neural = root / "mythos" / "neural"
    live_path = neural / "live.py"
    model_path = neural / "model.py"
    module_path = neural / "rithal_behavior_fix_v105_r3_3.py"

    required = (live_path, model_path, module_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required source: " + ", ".join(missing))

    for path in required:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    live_text = live_path.read_text(encoding="utf-8-sig")
    model_text = model_path.read_text(encoding="utf-8-sig")
    module_text = module_path.read_text(encoding="utf-8-sig")

    checks = {
        "model_regime_head_five": "self.regime_head     = mlp(5)" in model_text or "self.regime_head = mlp(5)" in model_text,
        "model_regime_gate_five": "n_regimes=5" in model_text,
        "module_exact_five_contract": "REGIME_CLASS_COUNT = len(REGIME_LABELS)" in module_text and '"PANIC"' in module_text,
        "module_preserves_infer": "RITHAL_V105_R3_3_INFER_MUTATED_DURING_INSTALL" in module_text,
        "module_preserves_prewarm": "RITHAL_V105_R3_3_PREWARM_MUTATED_DURING_INSTALL" in module_text,
        "module_preserves_process": "RITHAL_V105_R3_3_PROCESS_MUTATED_DURING_INSTALL" in module_text,
        "live_activation_block": "# RITHAL_BEHAVIOR_V105_R3_3_START" in live_text and "_rithal_v105_r33_apply_live_patch(globals())" in live_text,
        "logger_breakout_field": '_rp.get("p_breakout", 0.0)' in live_text,
        "startup_probe_five": "probe_logits = _np.asarray([1.0, 0.2, -0.4, 0.8, -0.3], dtype=float)" in live_text,
    }

    failed = [name for name, ok in checks.items() if not ok]
    report = {
        "version": VERSION,
        "status": "FAIL" if failed else "PASS",
        "project_root": str(root),
        "checks": checks,
        "failed": failed,
    }

    if not args.source_only and not failed:
        sys.path.insert(0, str(root))
        from mythos.neural import live
        from mythos.neural import rithal_behavior_fix_v105_r3_3 as patch

        self_test = patch.self_test()
        report["module_self_test"] = self_test
        runtime_checks = {
            "activation_marker": bool(getattr(live, "_RITHAL_BEHAVIOR_V105_R3_3_APPLIED", False)),
            "active_score_owner": getattr(live.NeuralV2Model._score_model_output, "__name__", "") == "score_r33",
            "active_regime_owner": getattr(live.LiveEngine._get_regime_probs, "__name__", "") == "regime_r33",
            "regime_contract_count": int((getattr(live, "RITHAL_REGIME_CONTRACT", {}) or {}).get("count", 0)) == 5,
        }
        report["runtime_checks"] = runtime_checks
        runtime_failed = [name for name, ok in runtime_checks.items() if not ok]
        if self_test.get("status") != "PASS" or runtime_failed:
            report["status"] = "FAIL"
            report["failed"].extend(runtime_failed or ["module_self_test"])

    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
