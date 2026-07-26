from __future__ import annotations

"""Rithal V1.0.5 R3.2 hotfix.

R3.2 removes the R3/R3.1 raw regime-head reimplementation from the live rank
prewarm path. The active project already has one canonical `_score_model_output`
that publishes the correct double-softmax true/rank regime contract. R3.2 keeps
that scorer and its prewarm implementation, then adds only the risk-adjusted
side overlay plus the R3.1 family-repair isolation.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    from . import rithal_behavior_fix_v105_r3 as _r3
    from . import rithal_behavior_fix_v105_r3_1 as _r31
except ImportError:
    import rithal_behavior_fix_v105_r3 as _r3
    import rithal_behavior_fix_v105_r3_1 as _r31

VERSION = "RITHAL_BEHAVIOR_FIX_V1_0_5_R3_2"
INSTANCE_ID = _r3.INSTANCE_ID


def _overlay_risk_adjusted_side(scored: Mapping[str, Any]) -> dict:
    result = dict(scored or {})
    side_overlay = _r3.score_values({
        "p_long_win": result.get("p_long_win"),
        "p_short_win": result.get("p_short_win"),
        "expected_r_long": result.get("expected_r_long"),
        "expected_r_short": result.get("expected_r_short"),
        "mae_long": result.get("mae_long", 1.0),
        "mae_short": result.get("mae_short", 1.0),
        "calibration_score": result.get("calibration_score", 1.0),
        "router_confidence": result.get("router_confidence", 1.0),
        "p_no_trade": result.get("p_no_trade", 0.0),
        "direction_confidence": result.get("direction_confidence", 0.5),
        "quantile_spread": result.get("quantile_spread", 1.0),
        "specialist_conf": result.get("specialist_conf", 1.0),
        # Deliberately omitted: regime_logits. The active scorer owns regime truth.
    })
    for key in (
        "side", "expected_r_side", "side_policy", "side_disagreement",
        "long_edge", "short_edge", "risk_adjusted_long_edge",
        "risk_adjusted_short_edge", "edge", "edge_margin",
        "chosen_expected_r", "chosen_win_probability",
    ):
        result[key] = side_overlay.get(key)
    result["side_policy"] = "RISK_ADJUSTED_EDGE_V105_R3_2_ACTIVE_SCORER_PRESERVED"
    result["regime_contract_source"] = "ACTIVE_SHARED_SCORE_MODEL_OUTPUT"
    return result


def apply_live_patch(ns: dict) -> None:
    if ns.get("_RITHAL_BEHAVIOR_V105_R3_2_APPLIED"):
        return
    Model = ns.get("NeuralV2Model")
    if Model is None:
        raise RuntimeError("RITHAL_V105_R3_2_MODEL_CLASS_MISSING")

    # Capture the active, already parity-correct shared scorer and prewarm before
    # R3/R3.1 install their raw-head replacement.
    active_score = Model._score_model_output
    active_prewarm = Model.prewarm_rank_history

    _r31.apply_live_patch(ns)
    prepared_r31 = Model._prepare_model_inputs

    def score_r32(self, out):
        return _overlay_risk_adjusted_side(active_score(self, out))

    def prewarm_r32(self, df, max_bars=None):
        # The active prewarm calls self._score_model_output, which now means the
        # preserved canonical scorer plus the risk-adjusted side overlay.
        return active_prewarm(self, df, max_bars=max_bars)

    def prepare_r32(self, df, *, context, force_neutralize_nonstationary=None):
        return prepared_r31(
            self,
            df,
            context=context,
            force_neutralize_nonstationary=force_neutralize_nonstationary,
        )

    Model._score_model_output = score_r32
    Model.prewarm_rank_history = prewarm_r32
    Model._prepare_model_inputs = prepare_r32
    ns["_RITHAL_BEHAVIOR_V105_R3_2_APPLIED"] = True
    ns["RITHAL_BEHAVIOR_FIX_VERSION"] = VERSION


def apply_trade_manager_patch(module) -> None:
    if getattr(module, "_RITHAL_BEHAVIOR_V105_R3_2_TM_APPLIED", False):
        return
    _r31.apply_trade_manager_patch(module)
    module._RITHAL_BEHAVIOR_V105_R3_2_TM_APPLIED = True
    module.RITHAL_BEHAVIOR_FIX_VERSION = VERSION


def configure_project(project_root: Path, instance_id: str = INSTANCE_ID, manager_mode: str = "PAPER_CONTROL") -> dict:
    report = dict(_r31.configure_project(Path(project_root), instance_id, manager_mode))
    report.update({
        "version": VERSION,
        "rank_prewarm": "ACTIVE_SHARED_SCORER_PRESERVED",
        "raw_regime_head_reimplementation": False,
    })
    return report


def authority_contract_self_test(project_root: Path) -> dict:
    result = dict(_r3.authority_contract_self_test(Path(project_root)))
    result["version"] = VERSION
    return result


def self_test() -> dict:
    class SyntheticModel:
        seq_len = 2
        feature_names = ("open_interest", "open_interest_delta")

        def _prepare_model_inputs(self, df, *, context, force_neutralize_nonstationary=None):
            return None, [], {"blocked": False, "top_ood_features": []}

        def _run_scored_sequence(self, sequence):
            return {}

        def _score_model_output(self, out):
            # Active scorer can publish a valid regime without exposing raw
            # `regime_logits` to the overlay.
            return {
                "p_long_win": 0.55,
                "p_short_win": 0.72,
                "expected_r_long": 1.4,
                "expected_r_short": 1.1,
                "mae_long": 2.0,
                "mae_short": 0.8,
                "calibration_score": 0.9,
                "router_confidence": 0.9,
                "p_no_trade": 0.1,
                "direction_confidence": 0.5,
                "quantile_spread": 0.5,
                "specialist_conf": 0.9,
                "regime_valid": True,
                "regime_state": "BREAKOUT",
                "true_regime_probs": [0.1, 0.1, 0.2, 0.6],
                "rank_regime_probs": [0.2, 0.2, 0.2, 0.4],
                "p_breakout": 0.6,
                "raw_composite": 0.5,
            }

        def prewarm_rank_history(self, df, max_bars=None):
            scored = self._score_model_output({"some_other_regime_head_key": [1, 2, 3, 4]})
            self._rank_prewarm_status = {
                "ok": bool(scored.get("regime_valid")),
                "mode": "ACTIVE_SHARED_SCORER",
                "count": 100,
            }
            return bool(scored.get("regime_valid"))

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
    model = SyntheticModel()
    assert model.prewarm_rank_history(None) is True
    assert model._rank_prewarm_status["mode"] == "ACTIVE_SHARED_SCORER"
    scored = model._score_model_output({})
    assert scored["regime_valid"] is True
    assert scored["p_breakout"] == 0.6
    assert scored["true_regime_probs"] == [0.1, 0.1, 0.2, 0.6]
    assert scored["rank_regime_probs"] == [0.2, 0.2, 0.2, 0.4]
    assert scored["side"] == -1
    assert scored["expected_r_side"] == 1
    assert scored["side_disagreement"] is True
    return {
        "version": VERSION,
        "status": "PASS",
        "checks": [
            "active_shared_scorer_preserved",
            "rank_prewarm_without_raw_regime_logits",
            "regime_truth_unchanged",
            "risk_adjusted_side_overlay",
            "r3_1_family_repair_retained",
        ],
    }


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
        result = authority_contract_self_test(Path(args.project_root))
    elif args.configure:
        result = configure_project(Path(args.project_root), args.instance_id, args.manager_mode)
    else:
        result = self_test()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
