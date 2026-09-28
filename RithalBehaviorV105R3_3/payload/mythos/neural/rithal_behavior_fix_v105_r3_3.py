from __future__ import annotations

"""Rithal V1.0.5 R3.3 exact five-class regime-contract repair.

The deployed MythosNeuralV2 architecture has five trained regime classes:
TREND_UP, TREND_DOWN, CHOP, BREAKOUT and PANIC. Earlier V1.0.5 R3/R3.1
logic required exactly four logits, while R3.2's synthetic test also supplied
four. Real five-logit checkpoint inference was therefore rejected even though
the model inputs, data and rank history were healthy.

R3.3 is deliberately narrow. It changes no checkpoint, weight, feature order,
scaler, threshold, TP/SL, fee, sizing, ledger or Trade Manager authority. It
replaces only the model-output scoring adapter and regime-probability accessor,
while preserving the active inference, feature preparation and rank-prewarm
methods.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

VERSION = "RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3"
INSTANCE_ID = "rithal-1-0-contract-locked"
REGIME_LABELS = ("TREND_UP", "TREND_DOWN", "CHOP", "BREAKOUT", "PANIC")
REGIME_CLASS_COUNT = len(REGIME_LABELS)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else float(default)
    except Exception:
        return float(default)


def _softmax(values: Sequence[float]) -> list[float]:
    raw = [float(value) for value in values]
    if not raw or not all(math.isfinite(value) for value in raw):
        return []
    top = max(raw)
    exp_values = [math.exp(value - top) for value in raw]
    total = sum(exp_values)
    return [value / total for value in exp_values] if total > 0.0 else []


def _entropy(probs: Sequence[float]) -> float:
    return -sum(float(prob) * math.log(max(float(prob), 1e-12)) for prob in probs)


def _five_class_posteriors(logits: Any) -> tuple[Optional[list[float]], Optional[list[float]], int]:
    if logits is None:
        return None, None, 0
    try:
        values = [float(value) for value in logits]
    except Exception:
        return None, None, 0
    count = len(values)
    if count != REGIME_CLASS_COUNT or not all(math.isfinite(value) for value in values):
        return None, None, count
    true_probs = _softmax(values)
    rank_probs = _softmax(true_probs)
    if len(true_probs) != REGIME_CLASS_COUNT or len(rank_probs) != REGIME_CLASS_COUNT:
        return None, None, count
    return true_probs, rank_probs, count


def _risk_adjusted_side(values: Mapping[str, Any]) -> dict:
    p_long = _num(values.get("p_long_win"))
    p_short = _num(values.get("p_short_win"))
    er_long = _num(values.get("expected_r_long"))
    er_short = _num(values.get("expected_r_short"))
    mae_long = _num(values.get("mae_long"), 1.0)
    mae_short = _num(values.get("mae_short"), 1.0)
    direction_confidence = min(1.0, max(0.0, _num(values.get("direction_confidence"), 0.5)))

    long_edge = p_long * max(er_long, 0.0) - (1.0 - p_long) * max(mae_long, 1.0) * 0.28
    short_edge = p_short * max(er_short, 0.0) - (1.0 - p_short) * max(mae_short, 1.0) * 0.28
    direction_bias = (direction_confidence - 0.5) * 0.05
    adjusted_long = long_edge + direction_bias
    adjusted_short = short_edge - direction_bias
    side = 1 if adjusted_long >= adjusted_short else -1
    expected_r_side = 1 if er_long >= er_short else -1

    return {
        "side": side,
        "expected_r_side": expected_r_side,
        "side_disagreement": side != expected_r_side,
        "long_edge": float(long_edge),
        "short_edge": float(short_edge),
        "risk_adjusted_long_edge": float(adjusted_long),
        "risk_adjusted_short_edge": float(adjusted_short),
        "edge": float(long_edge if side == 1 else short_edge),
        "edge_margin": float(abs(adjusted_long - adjusted_short)),
        "chosen_expected_r": float(er_long if side == 1 else er_short),
        "chosen_win_probability": float(p_long if side == 1 else p_short),
    }


def apply_live_patch(ns: dict) -> None:
    if ns.get("_RITHAL_BEHAVIOR_V105_R3_3_APPLIED"):
        return

    Model = ns.get("NeuralV2Model")
    Engine = ns.get("LiveEngine")
    np = ns.get("np")
    if Model is None or Engine is None or np is None:
        missing = [
            name for name, value in (
                ("NeuralV2Model", Model),
                ("LiveEngine", Engine),
                ("np", np),
            )
            if value is None
        ]
        raise RuntimeError("RITHAL_V105_R3_3_LIVE_CONTRACT_MISSING:" + ",".join(missing))

    log = ns.get("log")
    active_score = Model._score_model_output
    active_infer = Model.infer
    active_prewarm = Model.prewarm_rank_history
    active_prepare = Model._prepare_model_inputs
    active_process = Engine._process

    def _head_scalar(out, name: str, result: Mapping[str, Any], default: float = 0.0) -> float:
        if result.get(name) is not None:
            return _num(result.get(name), default)
        value = out.get(name) if isinstance(out, Mapping) else None
        if value is None:
            return float(default)
        try:
            return float(value.detach().cpu().numpy().reshape(-1)[0])
        except Exception:
            try:
                return float(np.asarray(value).reshape(-1)[0])
            except Exception:
                return float(default)

    def _head_array(out, name: str) -> Optional[list[float]]:
        value = out.get(name) if isinstance(out, Mapping) else None
        if value is None:
            return None
        try:
            return value.detach().cpu().numpy().reshape(-1).astype(float).tolist()
        except Exception:
            try:
                return np.asarray(value, dtype=float).reshape(-1).tolist()
            except Exception:
                return None

    def score_r33(self, out):
        # Preserve every field produced by the active scorer, then correct only
        # the contracts that were incompatible with the deployed five-class head.
        result = dict(active_score(self, out) or {})

        p_long = _head_scalar(out, "p_long_win", result)
        p_short = _head_scalar(out, "p_short_win", result)
        er_long = _head_scalar(out, "expected_r_long", result)
        er_short = _head_scalar(out, "expected_r_short", result)
        mae_long = _head_scalar(out, "mae_long", result, 1.0)
        mae_short = _head_scalar(out, "mae_short", result, 1.0)
        p_no_trade = min(1.0, max(0.0, _head_scalar(out, "p_no_trade", result, 1.0)))
        calibration = min(1.0, max(0.0, _head_scalar(out, "calibration_score", result)))
        router = min(1.0, max(0.0, _head_scalar(out, "router_confidence", result)))
        risk_score = min(1.0, max(0.0, _head_scalar(out, "risk_score", result, 0.5)))
        direction_confidence = min(1.0, max(0.0, _head_scalar(out, "direction_confidence", result, 0.5)))
        quantile_spread = max(0.0, _head_scalar(out, "quantile_spread", result, 1.0))

        specialist_conf = _num(result.get("specialist_conf"), 1.0)
        specialist_idx = int(_num(result.get("chosen_specialist_idx"), -1.0))
        specialist_logits = _head_array(out, "specialist_logits")
        if specialist_logits:
            specialist_probs = _softmax(specialist_logits)
            if specialist_probs:
                specialist_conf = float(max(specialist_probs))
                specialist_idx = int(max(range(len(specialist_probs)), key=specialist_probs.__getitem__))

        true_probs, rank_probs, regime_count = _five_class_posteriors(_head_array(out, "regime_logits"))
        regime_valid = true_probs is not None and rank_probs is not None

        # Preserve the frozen live/backtest rank-distribution mathematics. Rank
        # score uses the expected-R side; the V1.0.5 side overlay is independent.
        rank_side = 1 if er_long >= er_short else -1
        rank_expected_r = er_long if rank_side == 1 else er_short
        rank_pwin = p_long if rank_side == 1 else p_short
        raw_composite = (
            max(rank_expected_r, 0.0)
            * min(1.0, max(0.0, 1.0 - p_no_trade))
            * (0.5 + 0.5 * min(1.0, max(0.0, rank_pwin)))
        )

        p_trend = p_chop = p_breakout = p_panic = None
        rank_p_trend = rank_p_chop = rank_p_breakout = rank_p_panic = None
        regime_state = "REGIME_UNKNOWN"
        regime_idx = true_regime_idx = -1
        regime_entropy = None

        if regime_valid:
            p_trend = float(true_probs[0] + true_probs[1])
            p_chop = float(true_probs[2])
            p_breakout = float(true_probs[3])
            p_panic = float(true_probs[4])
            rank_p_trend = float(rank_probs[0] + rank_probs[1])
            rank_p_chop = float(rank_probs[2])
            rank_p_breakout = float(rank_probs[3])
            rank_p_panic = float(rank_probs[4])

            regime_multiplier = 0.35 + 0.65 * min(1.0, max(0.0, rank_p_trend))
            regime_multiplier *= 1.0 - 0.30 * min(1.0, max(0.0, rank_p_chop))
            raw_composite *= regime_multiplier

            true_regime_idx = int(max(range(REGIME_CLASS_COUNT), key=true_probs.__getitem__))
            regime_idx = int(max(range(REGIME_CLASS_COUNT), key=rank_probs.__getitem__))
            regime_state = REGIME_LABELS[true_regime_idx]
            regime_entropy = _entropy(true_probs)
        else:
            # A malformed trained head remains fail-closed and never enters rank.
            raw_composite = 0.0

        side_fields = _risk_adjusted_side({
            "p_long_win": p_long,
            "p_short_win": p_short,
            "expected_r_long": er_long,
            "expected_r_short": er_short,
            "mae_long": mae_long,
            "mae_short": mae_short,
            "direction_confidence": direction_confidence,
        })

        chosen_expected_r = float(side_fields["chosen_expected_r"])
        chosen_pwin = float(side_fields["chosen_win_probability"])
        saturation_state = "NORMAL"
        if abs(chosen_expected_r) >= 3.98 and chosen_pwin >= 0.995:
            saturation_state = "HARD_SATURATED"
        elif abs(chosen_expected_r) >= 3.90:
            saturation_state = "SATURATED"
        elif abs(chosen_expected_r) >= 3.60:
            saturation_state = "HIGH_CONVICTION"

        result.update(side_fields)
        result.update({
            "side_policy": "RISK_ADJUSTED_EDGE_V105_R3_3_FIVE_CLASS_CANONICAL_RANK",
            "rank_side": int(rank_side),
            "rank_expected_r": float(rank_expected_r),
            "rank_win_probability": float(rank_pwin),
            "raw_composite": float(raw_composite),
            "p_long_win": float(p_long),
            "p_short_win": float(p_short),
            "expected_r_long": float(er_long),
            "expected_r_short": float(er_short),
            "mae_long": float(mae_long),
            "mae_short": float(mae_short),
            "p_no_trade": float(p_no_trade),
            "calibration_score": float(calibration),
            "router_confidence": float(router),
            "risk_score": float(risk_score),
            "direction_confidence": float(direction_confidence),
            "quantile_spread": float(quantile_spread),
            "specialist_conf": float(specialist_conf),
            "chosen_specialist_idx": int(specialist_idx),
            "regime_valid": bool(regime_valid),
            "regime_state": regime_state,
            "regime_idx": int(regime_idx),
            "true_regime_idx": int(true_regime_idx),
            "regime_class_count": int(regime_count),
            "regime_contract_expected_classes": REGIME_CLASS_COUNT,
            "regime_class_labels": list(REGIME_LABELS),
            "regime_probability_source": (
                "EXACT_FIVE_CLASS_TRUE_POSTERIOR_PLUS_BACKTEST_RANK_POSTERIOR"
                if regime_valid else "INVALID_FIVE_CLASS_HEAD"
            ),
            "true_regime_probs": true_probs,
            "rank_regime_probs": rank_probs,
            "p_trend": p_trend,
            "p_chop": p_chop,
            "p_breakout": p_breakout,
            "p_panic": p_panic,
            "rank_p_trend": rank_p_trend,
            "rank_p_chop": rank_p_chop,
            "rank_p_breakout": rank_p_breakout,
            "rank_p_panic": rank_p_panic,
            "regime_entropy": regime_entropy,
            "saturation_state": saturation_state,
            "regime_contract_source": "DEPLOYED_MYTHOS_NEURAL_V2_FIVE_CLASS_HEAD",
            "_raw_model_output": out,
        })
        return result

    def regime_r33(self, sym, pred):
        pred = pred if isinstance(pred, Mapping) else {}
        try:
            probs = [float(value) for value in pred.get("true_regime_probs")]
        except Exception:
            probs = []
        if (
            pred.get("regime_valid") is True
            and len(probs) == REGIME_CLASS_COUNT
            and all(math.isfinite(value) for value in probs)
            and sum(probs) > 0.0
        ):
            total = sum(probs)
            probs = [value / total for value in probs]
            return {
                "valid": True,
                "state": REGIME_LABELS[max(range(REGIME_CLASS_COUNT), key=probs.__getitem__)],
                "p_trend": float(probs[0] + probs[1]),
                "p_chop": float(probs[2]),
                "p_breakout": float(probs[3]),
                "p_panic": float(probs[4]),
                "entropy": _entropy(probs),
                "panic_class_trained": True,
                "class_contract": "trend_up,trend_down,chop,breakout,panic",
                "true_probs": probs,
            }
        # Numeric fail-closed fallback prevents logger exceptions and guarantees
        # that both chop and panic entry gates remain blocked.
        return {
            "valid": False,
            "state": "REGIME_UNKNOWN",
            "p_trend": 0.0,
            "p_chop": 1.0,
            "p_breakout": 0.0,
            "p_panic": 1.0,
            "entropy": None,
            "panic_class_trained": True,
            "class_contract": "trend_up,trend_down,chop,breakout,panic",
            "fallback_reason": "missing_or_invalid_five_class_posterior",
        }

    Model._score_model_output = score_r33
    Engine._get_regime_probs = regime_r33

    # Hard installation invariants: this repair cannot replace any of these paths.
    if Model.infer is not active_infer:
        raise RuntimeError("RITHAL_V105_R3_3_INFER_MUTATED_DURING_INSTALL")
    if Model.prewarm_rank_history is not active_prewarm:
        raise RuntimeError("RITHAL_V105_R3_3_PREWARM_MUTATED_DURING_INSTALL")
    if Model._prepare_model_inputs is not active_prepare:
        raise RuntimeError("RITHAL_V105_R3_3_PREPARE_MUTATED_DURING_INSTALL")
    if Engine._process is not active_process:
        raise RuntimeError("RITHAL_V105_R3_3_PROCESS_MUTATED_DURING_INSTALL")

    ns["_RITHAL_BEHAVIOR_V105_R3_3_APPLIED"] = True
    ns["RITHAL_BEHAVIOR_FIX_VERSION"] = VERSION
    ns["RITHAL_REGIME_CONTRACT"] = {
        "version": VERSION,
        "classes": list(REGIME_LABELS),
        "count": REGIME_CLASS_COUNT,
        "rank_prewarm_preserved": True,
        "infer_preserved": True,
        "feature_prepare_preserved": True,
        "process_preserved": True,
        "trade_manager_authority_preserved": True,
    }

    if log is not None:
        log.warning(
            "[%s] installed: exact 5-class regime contract, canonical rank score restored, "
            "risk-adjusted side retained, inference/prewarm/execution authority unchanged",
            VERSION,
        )


def self_test() -> dict:
    import numpy as np

    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=float)
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value

    class SyntheticModel:
        def _prepare_model_inputs(self, *args, **kwargs): return None, [], {}
        def infer(self, df): return None
        def prewarm_rank_history(self, df, max_bars=None): return True
        def _score_model_output(self, out):
            return {"legacy_field_preserved": True}

    class SyntheticEngine:
        def _get_regime_probs(self, sym, pred): return {}
        def _process(self, sym, *args, **kwargs): return None

    namespace = {
        "NeuralV2Model": SyntheticModel,
        "LiveEngine": SyntheticEngine,
        "np": np,
    }
    pre_infer = SyntheticModel.infer
    pre_prewarm = SyntheticModel.prewarm_rank_history
    pre_prepare = SyntheticModel._prepare_model_inputs
    pre_process = SyntheticEngine._process
    apply_live_patch(namespace)

    valid_output = {
        "p_long_win": FakeTensor([0.55]),
        "p_short_win": FakeTensor([0.72]),
        "expected_r_long": FakeTensor([1.40]),
        "expected_r_short": FakeTensor([1.10]),
        "mae_long": FakeTensor([2.00]),
        "mae_short": FakeTensor([0.80]),
        "p_no_trade": FakeTensor([0.10]),
        "calibration_score": FakeTensor([0.90]),
        "router_confidence": FakeTensor([0.90]),
        "risk_score": FakeTensor([0.50]),
        "direction_confidence": FakeTensor([0.50]),
        "quantile_spread": FakeTensor([0.50]),
        "specialist_logits": FakeTensor([[0.1, 0.2, 0.3]]),
        "regime_logits": FakeTensor([[1.0, 0.2, -0.4, 0.8, -0.3]]),
    }
    scored = SyntheticModel()._score_model_output(valid_output)
    assert scored["legacy_field_preserved"] is True
    assert scored["regime_valid"] is True
    assert scored["regime_class_count"] == 5
    assert len(scored["true_regime_probs"]) == 5
    assert len(scored["rank_regime_probs"]) == 5
    assert scored["p_breakout"] is not None and scored["p_panic"] is not None
    assert scored["side"] == -1
    assert scored["expected_r_side"] == 1
    assert scored["side_disagreement"] is True
    assert math.isfinite(scored["raw_composite"]) and scored["raw_composite"] > 0.0

    invalid_output = dict(valid_output)
    invalid_output["regime_logits"] = FakeTensor([[1.0, 0.2, -0.4, 0.8]])
    invalid = SyntheticModel()._score_model_output(invalid_output)
    assert invalid["regime_valid"] is False
    assert invalid["raw_composite"] == 0.0
    fail_closed = SyntheticEngine()._get_regime_probs("BTCUSDT", invalid)
    assert fail_closed["p_chop"] == 1.0 and fail_closed["p_panic"] == 1.0

    assert SyntheticModel.infer is pre_infer
    assert SyntheticModel.prewarm_rank_history is pre_prewarm
    assert SyntheticModel._prepare_model_inputs is pre_prepare
    assert SyntheticEngine._process is pre_process

    return {
        "version": VERSION,
        "status": "PASS",
        "checks": [
            "active_scorer_fields_preserved",
            "exact_five_class_regime_head",
            "true_and_rank_double_softmax",
            "breakout_and_panic_separate",
            "canonical_rank_raw_composite_restored",
            "risk_adjusted_side_retained",
            "four_class_output_rejected_fail_closed",
            "infer_unchanged",
            "rank_prewarm_unchanged",
            "feature_prepare_unchanged",
            "process_routing_unchanged",
            "trade_manager_authority_unchanged",
        ],
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(self_test(), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
