from __future__ import annotations

"""Rithal V1.0.5 R1 safety wrapper.

R1 preserves the base V1.0.5 behavioral changes while tightening two contracts:

1. Derived feature-family members are neutralized only during an explicitly
   requested family-repair inference. They are never added permanently to the
   normal automatic-neutralization list.
2. The 15m thesis handoff always creates a dictionary before persisting the
   immutable entry snapshot, including restored legacy manager states.
"""

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Mapping

try:
    from . import rithal_behavior_fix_v105 as _base
except ImportError:
    import rithal_behavior_fix_v105 as _base

VERSION = "RITHAL_BEHAVIOR_FIX_V1_0_5_R1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else float(default)
    except Exception:
        return float(default)


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items() if not str(k).startswith("_raw_model_output")}
    if isinstance(value, (list, tuple, set)):
        return [_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _safe(value.item())
        except Exception:
            pass
    return str(value)


def apply_live_patch(ns: dict) -> None:
    if ns.get("_RITHAL_BEHAVIOR_V105_R1_APPLIED"):
        return
    original_repairable = tuple(ns.get("RITHAL_ABSOLUTE_OOD_NEUTRALIZE_FEATURES") or ())
    _base.apply_live_patch(ns)
    Model = ns["NeuralV2Model"]
    prepared_by_base = Model._prepare_model_inputs

    # Restore the normal input contract. Family expansion is scoped to one forced
    # repair call below, preventing stationary companions from being neutralized
    # merely because they crossed a generic absolute-feature threshold.
    ns["RITHAL_ABSOLUTE_OOD_NEUTRALIZE_FEATURES"] = original_repairable

    def prepare_r1(self, df, *, context, force_neutralize_nonstationary=None):
        if not force_neutralize_nonstationary:
            ns["RITHAL_ABSOLUTE_OOD_NEUTRALIZE_FEATURES"] = original_repairable
            return prepared_by_base(
                self, df, context=context,
                force_neutralize_nonstationary=force_neutralize_nonstationary,
            )
        requested = set(original_repairable)
        if force_neutralize_nonstationary is True:
            requested.update(str(x) for members in _base.FEATURE_FAMILIES.values() for x in members)
        else:
            requested.update(str(x) for x in force_neutralize_nonstationary)
        available = {str(x) for x in getattr(self, "feature_names", ())}
        temporary_contract = tuple(sorted(requested & available))
        ns["RITHAL_ABSOLUTE_OOD_NEUTRALIZE_FEATURES"] = temporary_contract
        try:
            return prepared_by_base(
                self, df, context=context,
                force_neutralize_nonstationary=force_neutralize_nonstationary,
            )
        finally:
            ns["RITHAL_ABSOLUTE_OOD_NEUTRALIZE_FEATURES"] = original_repairable

    Model._prepare_model_inputs = prepare_r1
    ns["_RITHAL_BEHAVIOR_V105_R1_APPLIED"] = True
    ns["RITHAL_BEHAVIOR_FIX_VERSION"] = VERSION
    log = ns.get("log")
    if log:
        log.warning(
            "[%s] safety wrapper installed: normal repair contract=%s; family expansion=forced-call-only",
            VERSION, ",".join(original_repairable),
        )


def _thesis_status(pred: Mapping[str, Any], entry: Mapping[str, Any]) -> tuple[str, list[str]]:
    pred = pred if isinstance(pred, Mapping) else {}
    entry = entry if isinstance(entry, Mapping) else {}
    health = pred.get("feature_health") if isinstance(pred.get("feature_health"), Mapping) else {}
    if health.get("blocked") or pred.get("regime_valid") is False:
        return "UNAVAILABLE", ["model_context_unavailable_or_blocked"]
    entry_side = int(_num(entry.get("side"), 0.0))
    new_side = int(_num(pred.get("side"), 0.0))
    opposite = bool(entry_side and new_side and entry_side != new_side)
    threshold = _num(pred.get("threshold", pred.get("effective_threshold", entry.get("threshold"))), 0.0)
    score = _num(pred.get("decision_score"))
    weak = bool(
        score < threshold
        or _num(pred.get("edge_margin")) < 0.08
        or _num(pred.get("router_confidence")) < 0.55
        or _num(pred.get("p_no_trade"), 1.0) > 0.55
    )
    if opposite and weak:
        return "BROKEN", ["qualified_opposite_side", "decision_quality_deteriorated"]
    if opposite or weak:
        return "WEAKENING", ["opposite_side" if opposite else "decision_quality_deteriorated"]
    return "INTACT", ["side_and_quality_preserved"]


def apply_trade_manager_patch(module) -> None:
    if getattr(module, "_RITHAL_BEHAVIOR_V105_R1_TM_APPLIED", False):
        return
    TradeManager = getattr(module, "TradeManager", None)
    if TradeManager is None:
        raise RuntimeError("RITHAL_V105_R1_TRADE_MANAGER_CLASS_MISSING")

    # Capture the verified pre-V1.0.5 15m review. The base patch is then applied
    # for entry context, 5m persistence, mark arbitration and public state.
    pre_v105_review_15m = TradeManager.review_15m_context
    _base.apply_trade_manager_patch(module)

    def review_15m_r1(self, sym, pos, pred, last_bar):
        result = pre_v105_review_15m(self, sym, pos, pred, last_bar)
        trade_id = self._trade_id(pos)
        with self._lock:
            state = self._trades.get(trade_id)
            if not isinstance(state, dict):
                return result
            entry_context = state.get("entry_context")
            if not isinstance(entry_context, dict):
                entry_context = {}
                state["entry_context"] = entry_context
            entry_context.setdefault("immutable_entry_snapshot", _safe(dict(entry_context)))
            status, reasons = _thesis_status(pred if isinstance(pred, Mapping) else {}, entry_context)
            bar_id = None
            for source in (
                last_bar if isinstance(last_bar, Mapping) else {},
                pred if isinstance(pred, Mapping) else {},
            ):
                for key in ("timestamp", "bar_open_ms", "open_time", "time", "datetime", "closed_15m"):
                    if source.get(key) is not None:
                        bar_id = str(source.get(key))
                        break
                if bar_id:
                    break
            snapshot_keys = (
                "side", "side_policy", "decision_score", "threshold",
                "expected_r_long", "expected_r_short", "p_long_win", "p_short_win",
                "mae_long", "mae_short", "long_edge", "short_edge", "edge",
                "edge_margin", "router_confidence", "calibration_score", "p_no_trade",
                "true_regime_probs", "regime_valid", "regime_state", "feature_health",
                "dual_inference", "sequence_fingerprint",
            )
            snapshot = {key: pred.get(key) for key in snapshot_keys} if isinstance(pred, Mapping) else {}
            thesis = {
                "version": VERSION,
                "status": status,
                "reasons": reasons,
                "bar_id": bar_id,
                "reviewed_at": _utc_now(),
                "snapshot": _safe(snapshot),
            }
            thesis["snapshot_hash"] = hashlib.sha256(
                json.dumps(thesis["snapshot"], sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            state["thesis_v105"] = thesis
            self._persist()
            output = dict(result or {})
            output["thesis_v105"] = dict(thesis)
            return output

    TradeManager.review_15m_context = review_15m_r1
    module._RITHAL_BEHAVIOR_V105_R1_TM_APPLIED = True
    module.RITHAL_BEHAVIOR_FIX_VERSION = VERSION


def self_test() -> dict:
    base = _base.self_test()
    assert base.get("status") == "PASS"
    status, reasons = _thesis_status(
        {"side": -1, "decision_score": 0.4, "threshold": 0.8, "edge_margin": 0.02,
         "router_confidence": 0.4, "p_no_trade": 0.7, "regime_valid": True,
         "feature_health": {"blocked": False}},
        {"side": 1},
    )
    assert status == "BROKEN" and "qualified_opposite_side" in reasons
    return {
        "version": VERSION,
        "status": "PASS",
        "base": base,
        "checks": [
            "base_behavior_contract",
            "forced_call_only_family_neutralization",
            "restored_state_entry_context_safety",
            "thesis_status_contract",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2, sort_keys=True))
