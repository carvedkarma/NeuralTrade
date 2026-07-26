from __future__ import annotations

"""Rithal V1.0.5 R3.1 final compatibility wrapper.

R3.1 preserves every R3 correction and additionally propagates the expanded
feature family as the actual forced-neutralization argument.  This guarantees
that family companions are neutralized deliberately rather than merely being
eligible through a temporary global contract.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

try:
    from . import rithal_behavior_fix_v105 as _base
    from . import rithal_behavior_fix_v105_r3 as _r3
except ImportError:
    import rithal_behavior_fix_v105 as _base
    import rithal_behavior_fix_v105_r3 as _r3

VERSION = "RITHAL_BEHAVIOR_FIX_V1_0_5_R3_1"
INSTANCE_ID = _r3.INSTANCE_ID
score_values = _r3.score_values
_thesis_status = _r3._thesis_status
_manager_policy = _r3._manager_policy
authority_contract_self_test = _r3.authority_contract_self_test


def apply_live_patch(ns: dict) -> None:
    if ns.get("_RITHAL_BEHAVIOR_V105_R3_1_APPLIED"):
        return
    _r3.apply_live_patch(ns)
    Model = ns["NeuralV2Model"]
    prepared_r3 = Model._prepare_model_inputs

    def prepare_r31(self, df, *, context, force_neutralize_nonstationary=None):
        effective_force = force_neutralize_nonstationary
        if force_neutralize_nonstationary:
            if force_neutralize_nonstationary is True:
                requested = [
                    name
                    for members in _base.FEATURE_FAMILIES.values()
                    for name in members
                ]
            else:
                requested = [str(name) for name in force_neutralize_nonstationary]
            effective_force, _ = _base.expand_repair_families(
                requested,
                getattr(self, "feature_names", ()),
            )
        return prepared_r3(
            self,
            df,
            context=context,
            force_neutralize_nonstationary=effective_force,
        )

    Model._prepare_model_inputs = prepare_r31
    ns["_RITHAL_BEHAVIOR_V105_R3_1_APPLIED"] = True
    ns["RITHAL_BEHAVIOR_FIX_VERSION"] = VERSION


def apply_trade_manager_patch(module) -> None:
    if getattr(module, "_RITHAL_BEHAVIOR_V105_R3_1_TM_APPLIED", False):
        return
    _r3.apply_trade_manager_patch(module)
    module._RITHAL_BEHAVIOR_V105_R3_1_TM_APPLIED = True
    module.RITHAL_BEHAVIOR_FIX_VERSION = VERSION


def configure_project(project_root: Path, instance_id: str = INSTANCE_ID, manager_mode: str = "PAPER_CONTROL") -> dict:
    report = dict(_r3.configure_project(
        Path(project_root),
        instance_id=instance_id,
        manager_mode=manager_mode,
    ))
    report["version"] = VERSION
    report["forced_family_propagation"] = True
    return report


def self_test() -> dict:
    report = dict(_r3.self_test())

    class SyntheticModel:
        feature_names = ("open_interest", "open_interest_delta")
        seq_len = 2

        def _prepare_model_inputs(self, df, *, context, force_neutralize_nonstationary=None):
            return None, [], {
                "blocked": False,
                "top_ood_features": [],
                "received_force": list(force_neutralize_nonstationary or []),
            }

        def _run_scored_sequence(self, sequence):
            return {}

    class SyntheticEngine:
        def _get_regime_probs(self, sym, pred): return {}
        def _trade_manager_entry_context(self, sym, pos, pred, threshold_used, bar_timestamp_ms): return {}
        def _arm_trade_manager_position(self, sym, pos, pred, threshold_used, bar_timestamp_ms): return None
        def _process(self, sym, *args, **kwargs): return None

    import numpy as np
    namespace = {
        "NeuralV2Model": SyntheticModel,
        "LiveEngine": SyntheticEngine,
        "np": np,
        "RANK_WINDOW": 100,
        "RANK_MIN_HISTORY": 10,
        "RITHAL_ABSOLUTE_OOD_NEUTRALIZE_FEATURES": ("open_interest",),
    }
    apply_live_patch(namespace)
    _, _, health = SyntheticModel()._prepare_model_inputs(
        None,
        context="r31_forced_family",
        force_neutralize_nonstationary=["open_interest"],
    )
    assert health["received_force"] == ["open_interest", "open_interest_delta"], health
    report["version"] = VERSION
    report["checks"] = list(report.get("checks") or []) + ["expanded_family_passed_as_actual_force_argument"]
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--authority-self-test", action="store_true")
    parser.add_argument("--configure", action="store_true")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--instance-id", default=INSTANCE_ID)
    parser.add_argument("--manager-mode", choices=("PAPER_CONTROL", "SHADOW_ONLY"), default="PAPER_CONTROL")
    args = parser.parse_args(argv)
    if args.authority_self_test:
        result = dict(authority_contract_self_test(Path(args.project_root)))
        result["version"] = VERSION
    elif args.configure:
        result = configure_project(Path(args.project_root), args.instance_id, args.manager_mode)
    else:
        result = self_test()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
