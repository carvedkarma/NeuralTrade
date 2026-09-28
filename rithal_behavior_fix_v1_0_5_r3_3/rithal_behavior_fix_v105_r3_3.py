from __future__ import annotations

"""Rithal V1.0.5 R3.3 scorer/inference compatibility repair.

R3.2 correctly restored the pre-existing canonical scorer and rank prewarm, but
left the V1.0.5 family-repair ``infer`` wrapper in place.  That wrapper requires
``regime_valid`` and scalar regime telemetry, while the canonical scorer publishes
only ``true_regime_probs`` and ``rank_regime_probs``.  Every otherwise valid model
response was therefore marked ``regime_unknown_or_invalid_four_class_posterior``.

R3.3 keeps every intended layer:

* canonical model heads, raw composite, true/rank regime probabilities and rank
  prewarm are unchanged;
* R3.1 coherent family neutralisation remains active;
* the risk-adjusted side policy remains active, but it reuses the canonical
  long/short edges calculated with the real MAE heads instead of silently falling
  back to MAE=1.0;
* the V1.0.5 fail-closed inference, thesis, sizing and PAPER execution wrappers
  remain active;
* no checkpoint, feature order, threshold, fee, TP/SL, position, ledger or
  authority setting is changed.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

VERSION = "RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3"
INSTANCE_ID = "rithal-1-0-contract-locked"
MARKER = "RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3_SCORER_INFER_BRIDGE"

_DEPENDENCY_OVERRIDE: Optional[tuple[Any, Any]] = None


def _dependencies() -> tuple[Any, Any]:
    if _DEPENDENCY_OVERRIDE is not None:
        return _DEPENDENCY_OVERRIDE
    try:
        from . import rithal_behavior_fix_v105_r3_2 as r32
        from . import rithal_behavior_fix_v105_r3 as r3
    except ImportError:
        import rithal_behavior_fix_v105_r3_2 as r32
        import rithal_behavior_fix_v105_r3 as r3
    return r32, r3


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _vector4(value: Any) -> Optional[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        return None
    if sum(result) <= 0.0:
        return None
    return result


def _raw_scalar(scored: Mapping[str, Any], name: str, default: Optional[float] = None) -> Optional[float]:
    if _finite(scored.get(name)):
        return float(scored[name])
    raw = scored.get("_raw_model_output")
    if not isinstance(raw, Mapping) or name not in raw:
        return default
    value = raw[name]
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        if hasattr(value, "reshape"):
            value = value.reshape(-1)[0]
        elif isinstance(value, (list, tuple)):
            value = value[0]
        elif hasattr(value, "item"):
            value = value.item()
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def _regime_contract(result: dict) -> dict:
    """Bridge canonical posterior arrays to the V1.0.5 inference contract.

    The posterior arrays themselves are never recalculated or replaced.  They are
    validated and projected into scalar telemetry required by the existing
    fail-closed inference and engine gates.
    """

    true_probs = _vector4(result.get("true_regime_probs"))
    rank_probs = _vector4(result.get("rank_regime_probs"))
    valid = true_probs is not None and rank_probs is not None
    result["regime_valid"] = bool(valid)
    result["regime_contract_source"] = "ACTIVE_SHARED_SCORE_MODEL_OUTPUT"
    result["regime_class_contract"] = "trend_up,trend_down,chop,breakout"

    if not valid:
        result["regime_state"] = "REGIME_UNKNOWN"
        result["p_trend"] = None
        result["p_chop"] = None
        result["p_breakout"] = None
        result["rank_p_trend"] = None
        result["rank_p_chop"] = None
        result["regime_entropy"] = None
        result["regime_contract_error"] = "missing_or_invalid_true_or_rank_four_class_posterior"
        return result

    labels = ("TREND_UP", "TREND_DOWN", "CHOP", "BREAKOUT")
    result["regime_state"] = labels[max(range(4), key=true_probs.__getitem__)]
    result["p_trend"] = float(true_probs[0] + true_probs[1])
    result["p_chop"] = float(true_probs[2])
    result["p_breakout"] = float(true_probs[3])
    result["rank_p_trend"] = float(rank_probs[0] + rank_probs[1])
    result["rank_p_chop"] = float(rank_probs[2])
    result["regime_entropy"] = float(
        -sum(probability * math.log(max(probability, 1e-12)) for probability in true_probs)
    )
    result.pop("regime_contract_error", None)
    return result


def _risk_adjusted_side(result: dict) -> dict:
    """Apply only the intended side overlay without rebuilding model scoring."""

    p_long = _raw_scalar(result, "p_long_win", 0.0) or 0.0
    p_short = _raw_scalar(result, "p_short_win", 0.0) or 0.0
    er_long = _raw_scalar(result, "expected_r_long", 0.0) or 0.0
    er_short = _raw_scalar(result, "expected_r_short", 0.0) or 0.0
    mae_long = _raw_scalar(result, "mae_long", 1.0) or 1.0
    mae_short = _raw_scalar(result, "mae_short", 1.0) or 1.0
    direction_confidence = _raw_scalar(result, "direction_confidence", 0.5)
    direction_confidence = 0.5 if direction_confidence is None else min(1.0, max(0.0, direction_confidence))

    # The canonical scorer already calculated these with the real MAE heads.  Use
    # those exact values.  Recalculate only as a fail-safe for an older scorer that
    # did not publish edge telemetry.
    if _finite(result.get("long_edge")):
        long_edge = float(result["long_edge"])
    else:
        long_edge = p_long * max(er_long, 0.0) - (1.0 - p_long) * max(mae_long, 1.0) * 0.28
    if _finite(result.get("short_edge")):
        short_edge = float(result["short_edge"])
    else:
        short_edge = p_short * max(er_short, 0.0) - (1.0 - p_short) * max(mae_short, 1.0) * 0.28

    direction_bias = (direction_confidence - 0.5) * 0.05
    adjusted_long = long_edge + direction_bias
    adjusted_short = short_edge - direction_bias
    side = 1 if adjusted_long >= adjusted_short else -1
    expected_r_side = 1 if er_long >= er_short else -1

    result.update({
        "side": int(side),
        "expected_r_side": int(expected_r_side),
        "side_policy": "RISK_ADJUSTED_EDGE_V105_R3_3_CANONICAL_SCORE_PRESERVED",
        "side_disagreement": bool(side != expected_r_side),
        "long_edge": float(long_edge),
        "short_edge": float(short_edge),
        "risk_adjusted_long_edge": float(adjusted_long),
        "risk_adjusted_short_edge": float(adjusted_short),
        "edge": float(long_edge if side == 1 else short_edge),
        "edge_margin": float(abs(adjusted_long - adjusted_short)),
        "chosen_expected_r": float(er_long if side == 1 else er_short),
        "chosen_win_probability": float(p_long if side == 1 else p_short),
        # Preserve real head values for immutable entry-thesis and manager audit.
        "mae_long": float(mae_long),
        "mae_short": float(mae_short),
        "direction_confidence": float(direction_confidence),
    })
    quantile_spread = _raw_scalar(result, "quantile_spread")
    if quantile_spread is not None:
        result["quantile_spread"] = float(quantile_spread)
    return result


def adapt_scored_contract(scored: Mapping[str, Any]) -> dict:
    """Idempotent, non-mutating adapter used by live inference and tests."""

    result = dict(scored or {})
    protected = {
        key: result.get(key)
        for key in (
            "raw_composite", "true_regime_probs", "rank_regime_probs",
            "calibration_score", "router_confidence", "p_no_trade",
            "p_long_win", "p_short_win", "expected_r_long", "expected_r_short",
            "specialist_conf", "chosen_specialist_idx", "risk_score",
            "saturation_state", "regime_idx", "_raw_model_output",
        )
    }
    result = _regime_contract(result)
    result = _risk_adjusted_side(result)
    # These fields belong exclusively to the canonical scorer/model and must not
    # be changed by the compatibility bridge.
    for key, value in protected.items():
        if key in scored:
            result[key] = value
    result["behavior_contract"] = VERSION
    result["scorer_infer_bridge"] = MARKER
    return result


def apply_live_patch(ns: dict) -> None:
    if ns.get("_RITHAL_BEHAVIOR_V105_R3_3_APPLIED"):
        return
    model_class = ns.get("NeuralV2Model")
    if model_class is None:
        raise RuntimeError("RITHAL_V105_R3_3_MODEL_CLASS_MISSING")

    r32, _r3 = _dependencies()
    # score_r32 resolves this module global at call time.  Replacing it repairs an
    # already-installed R3.2 scorer as well as a fresh installation.
    r32._overlay_risk_adjusted_side = adapt_scored_contract
    r32.apply_live_patch(ns)

    current_score = model_class._score_model_output

    def score_r33(self, out):
        return adapt_scored_contract(current_score(self, out))

    model_class._score_model_output = score_r33
    ns["_RITHAL_BEHAVIOR_V105_R3_3_APPLIED"] = True
    ns["RITHAL_BEHAVIOR_FIX_VERSION"] = VERSION
    log = ns.get("log")
    if log:
        log.warning(
            "[%s] installed: canonical score/rank preserved, four-class scorer-infer bridge active, real-head side overlay active",
            VERSION,
        )


def apply_trade_manager_patch(module) -> None:
    if getattr(module, "_RITHAL_BEHAVIOR_V105_R3_3_TM_APPLIED", False):
        return
    r32, _r3 = _dependencies()
    r32.apply_trade_manager_patch(module)
    module._RITHAL_BEHAVIOR_V105_R3_3_TM_APPLIED = True
    module.RITHAL_BEHAVIOR_FIX_VERSION = VERSION


def configure_project(project_root: Path, instance_id: str = INSTANCE_ID, manager_mode: str = "PAPER_CONTROL") -> dict:
    r32, _r3 = _dependencies()
    report = dict(r32.configure_project(Path(project_root), instance_id, manager_mode))
    report.update({
        "version": VERSION,
        "canonical_raw_composite_preserved": True,
        "canonical_true_rank_posteriors_preserved": True,
        "scorer_infer_contract_bridged": True,
        "real_mae_edges_preserved": True,
    })
    return report


def self_test() -> dict:
    class FakeTensor:
        def __init__(self, value):
            self.value = value
        def detach(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return self
        def reshape(self, *_args):
            return [self.value]

    raw = {
        "mae_long": FakeTensor(4.0),
        "mae_short": FakeTensor(0.8),
        "direction_confidence": FakeTensor(0.4),
        "quantile_spread": FakeTensor(0.7),
    }
    canonical = {
        "p_long_win": 0.55,
        "p_short_win": 0.72,
        "expected_r_long": 1.4,
        "expected_r_short": 1.1,
        "long_edge": 0.55 * 1.4 - 0.45 * 4.0 * 0.28,
        "short_edge": 0.72 * 1.1 - 0.28 * 1.0 * 0.28,
        "calibration_score": 0.91,
        "router_confidence": 0.92,
        "p_no_trade": 0.08,
        "specialist_conf": 0.83,
        "chosen_specialist_idx": 2,
        "risk_score": 0.4,
        "raw_composite": 0.73123456789,
        "saturation_state": "NORMAL",
        "true_regime_probs": [0.10, 0.15, 0.20, 0.55],
        "rank_regime_probs": [0.22, 0.23, 0.24, 0.31],
        "regime_idx": 3,
        "_raw_model_output": raw,
    }
    adapted = adapt_scored_contract(canonical)
    assert adapted["raw_composite"] == canonical["raw_composite"]
    assert adapted["true_regime_probs"] is canonical["true_regime_probs"]
    assert adapted["rank_regime_probs"] is canonical["rank_regime_probs"]
    assert adapted["regime_valid"] is True
    assert adapted["p_breakout"] == 0.55
    assert adapted["p_chop"] == 0.20
    assert adapted["rank_p_chop"] == 0.24
    assert adapted["mae_long"] == 4.0 and adapted["mae_short"] == 0.8
    assert adapted["long_edge"] == canonical["long_edge"]
    assert adapted["short_edge"] == canonical["short_edge"]
    assert adapted["side"] == -1
    assert adapted["expected_r_side"] == 1
    assert adapted["side_disagreement"] is True
    again = adapt_scored_contract(adapted)
    assert again["side"] == adapted["side"]
    assert again["edge_margin"] == adapted["edge_margin"]
    assert again["raw_composite"] == adapted["raw_composite"]

    invalid = dict(canonical)
    invalid["true_regime_probs"] = [0.2, 0.3, 0.5]
    invalid_adapted = adapt_scored_contract(invalid)
    assert invalid_adapted["regime_valid"] is False
    assert invalid_adapted["raw_composite"] == canonical["raw_composite"]

    # Integration simulation: R3.2 scorer emits canonical fields but no
    # regime_valid.  R3.3 must make the existing V1.0.5 infer contract tradeable.
    class SyntheticModel:
        def _score_model_output(self, _out):
            return dict(canonical)
        def infer_contract_probe(self):
            scored = self._score_model_output({})
            return not bool(scored.get("regime_valid"))

    class FakeR32:
        _overlay_risk_adjusted_side = staticmethod(lambda scored: dict(scored))
        @classmethod
        def apply_live_patch(cls, ns):
            model = ns["NeuralV2Model"]
            active = model._score_model_output
            def score_r32(self, out):
                return cls._overlay_risk_adjusted_side(active(self, out))
            model._score_model_output = score_r32
        @staticmethod
        def apply_trade_manager_patch(_module):
            return None
        @staticmethod
        def configure_project(*_args, **_kwargs):
            return {"status": "PASS"}

    class FakeR3:
        INSTANCE_ID = INSTANCE_ID

    global _DEPENDENCY_OVERRIDE
    prior = _DEPENDENCY_OVERRIDE
    try:
        _DEPENDENCY_OVERRIDE = (FakeR32, FakeR3)
        namespace = {"NeuralV2Model": SyntheticModel, "log": None}
        apply_live_patch(namespace)
        assert SyntheticModel().infer_contract_probe() is False
        scored = SyntheticModel()._score_model_output({})
        assert scored["regime_valid"] is True
        assert scored["raw_composite"] == canonical["raw_composite"]
    finally:
        _DEPENDENCY_OVERRIDE = prior

    return {
        "version": VERSION,
        "status": "PASS",
        "checks": [
            "canonical_raw_composite_unchanged",
            "canonical_true_posterior_unchanged",
            "canonical_rank_posterior_unchanged",
            "four_class_contract_projected",
            "real_mae_heads_recovered",
            "canonical_edges_reused",
            "risk_adjusted_side_preserved",
            "invalid_posterior_fails_closed",
            "adapter_idempotent",
            "v105_infer_contract_integration",
        ],
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--configure", action="store_true")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--instance-id", default=INSTANCE_ID)
    parser.add_argument("--manager-mode", choices=("PAPER_CONTROL", "SHADOW_ONLY"), default="PAPER_CONTROL")
    args = parser.parse_args(argv)
    if args.configure:
        result = configure_project(Path(args.project_root), args.instance_id, args.manager_mode)
    else:
        result = self_test()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
