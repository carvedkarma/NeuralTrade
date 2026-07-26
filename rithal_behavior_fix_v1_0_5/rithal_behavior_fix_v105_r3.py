from __future__ import annotations

"""Rithal V1.0.5 R3 behavior safety wrapper.

R3 supersedes the earlier R1/R2 candidate.  It retains the intended PAPER-only
behavior changes while correcting active-source signature compatibility,
regime-rank parity, fail-closed thesis handling, restored-position context, and
concurrent feature-repair isolation.
"""

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

try:
    from . import rithal_behavior_fix_v105 as _base
except ImportError:
    import rithal_behavior_fix_v105 as _base

VERSION = "RITHAL_BEHAVIOR_FIX_V1_0_5_R3"
INSTANCE_ID = "rithal-1-0-contract-locked"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else float(default)
    except Exception:
        return float(default)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _safe(item)
            for key, item in value.items()
            if not str(key).startswith("_raw_model_output")
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _safe(value.item())
        except Exception:
            pass
    return str(value)


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    top = max(float(value) for value in values)
    exp_values = [math.exp(float(value) - top) for value in values]
    total = sum(exp_values)
    return [value / total for value in exp_values] if total > 0 else []


def _entropy(probs: Sequence[float]) -> float:
    return -sum(float(prob) * math.log(max(float(prob), 1e-12)) for prob in probs)


def _regime_probabilities(logits: Any) -> tuple[Optional[list[float]], Optional[list[float]]]:
    if not isinstance(logits, (list, tuple)) or len(logits) != 4:
        return None, None
    if not all(_finite(value) for value in logits):
        return None, None
    true_probs = _softmax([float(value) for value in logits])
    if len(true_probs) != 4 or not all(_finite(value) for value in true_probs):
        return None, None
    # The deployed rank history/backtest contract applies a second softmax to the
    # true posterior.  Publish both; never mislabel rank probabilities as truth.
    rank_probs = _softmax(true_probs)
    return true_probs, rank_probs


def score_values(values: Mapping[str, Any]) -> dict:
    """Canonical live/prewarm scorer with separate true and rank posteriors."""
    p_long = _num(values.get("p_long_win"))
    p_short = _num(values.get("p_short_win"))
    er_long = _num(values.get("expected_r_long"))
    er_short = _num(values.get("expected_r_short"))
    mae_long = _num(values.get("mae_long"), 1.0)
    mae_short = _num(values.get("mae_short"), 1.0)
    calibration = min(1.0, max(0.0, _num(values.get("calibration_score"))))
    router = min(1.0, max(0.0, _num(values.get("router_confidence"))))
    p_no_trade = min(1.0, max(0.0, _num(values.get("p_no_trade"), 1.0)))
    direction_confidence = min(1.0, max(0.0, _num(values.get("direction_confidence"), 0.5)))
    quantile_spread = max(0.0, _num(values.get("quantile_spread"), 1.0))
    specialist_conf = min(1.0, max(0.0, _num(values.get("specialist_conf"), 1.0)))

    long_edge = p_long * max(er_long, 0.0) - (1.0 - p_long) * max(mae_long, 1.0) * 0.28
    short_edge = p_short * max(er_short, 0.0) - (1.0 - p_short) * max(mae_short, 1.0) * 0.28
    direction_bias = (direction_confidence - 0.5) * 0.05
    adjusted_long = long_edge + direction_bias
    adjusted_short = short_edge - direction_bias
    side = 1 if adjusted_long >= adjusted_short else -1
    expected_r_side = 1 if er_long >= er_short else -1
    edge = long_edge if side == 1 else short_edge
    edge_margin = abs(adjusted_long - adjusted_short)
    chosen_expected_r = er_long if side == 1 else er_short
    chosen_pwin = p_long if side == 1 else p_short

    true_probs, rank_probs = _regime_probabilities(values.get("regime_logits"))
    regime_valid = true_probs is not None and rank_probs is not None
    p_chop_true = true_probs[2] if regime_valid else None
    p_breakout_true = true_probs[3] if regime_valid else None
    p_trend_true = true_probs[0] + true_probs[1] if regime_valid else None
    p_chop_rank = rank_probs[2] if regime_valid else None
    p_trend_rank = rank_probs[0] + rank_probs[1] if regime_valid else None

    regime_mult = 0.0
    if regime_valid:
        regime_mult = 0.35 + 0.65 * min(1.0, max(0.0, p_trend_rank))
        regime_mult *= 1.0 - 0.30 * min(1.0, max(0.0, p_chop_rank))

    spread_conf = min(1.0, max(0.1, 1.0 - (quantile_spread - 0.5) / 2.0))
    raw_composite = max(edge, 0.0) * router * calibration * (1.0 - p_no_trade)
    raw_composite *= specialist_conf * spread_conf * regime_mult

    saturation_state = "NORMAL"
    if abs(chosen_expected_r) >= 3.98 and chosen_pwin >= 0.995:
        saturation_state = "HARD_SATURATED"
    elif abs(chosen_expected_r) >= 3.90:
        saturation_state = "SATURATED"
    elif abs(chosen_expected_r) >= 3.60:
        saturation_state = "HIGH_CONVICTION"

    labels = ("TREND_UP", "TREND_DOWN", "CHOP", "BREAKOUT")
    return {
        "side": side,
        "expected_r_side": expected_r_side,
        "side_policy": "RISK_ADJUSTED_EDGE_V105_R3",
        "side_disagreement": side != expected_r_side,
        "long_edge": float(long_edge),
        "short_edge": float(short_edge),
        "risk_adjusted_long_edge": float(adjusted_long),
        "risk_adjusted_short_edge": float(adjusted_short),
        "edge": float(edge),
        "edge_margin": float(edge_margin),
        "chosen_expected_r": float(chosen_expected_r),
        "chosen_win_probability": float(chosen_pwin),
        "raw_composite": float(raw_composite),
        "regime_valid": bool(regime_valid),
        "regime_state": labels[max(range(4), key=true_probs.__getitem__)] if regime_valid else "REGIME_UNKNOWN",
        "true_regime_probs": true_probs,
        "rank_regime_probs": rank_probs,
        "regime_probability_source": "TRUE_POSTERIOR_PLUS_BACKTEST_RANK_POSTERIOR" if regime_valid else "UNAVAILABLE",
        "p_trend": p_trend_true,
        "p_chop": p_chop_true,
        "p_breakout": p_breakout_true,
        "rank_p_trend": p_trend_rank,
        "rank_p_chop": p_chop_rank,
        "regime_entropy": _entropy(true_probs) if regime_valid else None,
        "specialist_conf": float(specialist_conf),
        "spread_confidence": float(spread_conf),
        "saturation_state": saturation_state,
    }


def _thesis_status(pred: Mapping[str, Any], entry: Mapping[str, Any]) -> tuple[str, list[str]]:
    pred = pred if isinstance(pred, Mapping) else {}
    entry = entry if isinstance(entry, Mapping) else {}
    health = pred.get("feature_health") if isinstance(pred.get("feature_health"), Mapping) else {}
    if not pred:
        return "UNAVAILABLE", ["current_15m_prediction_missing"]
    if health.get("blocked"):
        return "UNAVAILABLE", ["current_15m_feature_health_blocked"]
    if pred.get("regime_valid") is not True:
        return "UNAVAILABLE", ["current_15m_regime_missing_or_invalid"]

    entry_side = int(_num(entry.get("side"), 0.0))
    new_side = int(_num(pred.get("side"), 0.0))
    if entry_side not in {-1, 1}:
        return "UNAVAILABLE", ["immutable_entry_side_unavailable"]
    if new_side not in {-1, 1}:
        return "UNAVAILABLE", ["current_15m_side_unavailable"]

    threshold_value = pred.get("threshold", pred.get("effective_threshold", entry.get("threshold", entry.get("threshold_used"))))
    required = {
        "decision_score": pred.get("decision_score"),
        "edge_margin": pred.get("edge_margin"),
        "router_confidence": pred.get("router_confidence"),
        "p_no_trade": pred.get("p_no_trade"),
        "threshold": threshold_value,
    }
    missing = [name for name, value in required.items() if not _finite(value)]
    if missing:
        return "UNAVAILABLE", ["current_15m_quality_fields_missing:" + ",".join(sorted(missing))]

    opposite = entry_side != new_side
    weak = bool(
        float(required["decision_score"]) < float(required["threshold"])
        or float(required["edge_margin"]) < 0.08
        or float(required["router_confidence"]) < 0.55
        or float(required["p_no_trade"]) > 0.55
    )
    if opposite and weak:
        return "BROKEN", ["qualified_opposite_side", "decision_quality_deteriorated"]
    if opposite or weak:
        return "WEAKENING", ["opposite_side" if opposite else "decision_quality_deteriorated"]
    return "INTACT", ["side_and_quality_preserved"]


def _manager_policy(
    metrics: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    thesis_status: str,
    existing_loss_state: str,
    existing_loss_branch: str,
) -> dict:
    current_r = _num(metrics.get("current_r"))
    mfe_r = _num(metrics.get("mfe_r"))
    giveback_r = max(0.0, _num(metrics.get("giveback_r"), mfe_r - current_r))
    age = _num(metrics.get("age_minutes"))
    target_r = max(1e-9, _num(metrics.get("target_r"), 1.0))
    current_progress = _num(metrics.get("current_progress"), current_r / target_r)
    no_new_mfe = int(lifecycle.get("no_new_mfe_reviews") or 0)
    giveback_reviews = int(lifecycle.get("giveback_reviews") or 0)
    adverse_reviews = int(lifecycle.get("adverse_reviews") or 0)
    recovery_fail_reviews = int(lifecycle.get("recovery_fail_reviews") or 0)
    thesis_status = str(thesis_status or "UNAVAILABLE").upper()
    thesis_available = thesis_status in {"INTACT", "WEAKENING", "BROKEN"}
    loss_confirmed = str(existing_loss_state or "").startswith("SHADOW_EXIT_CONFIRMED")
    branch = str(existing_loss_branch or "NONE").upper()

    p80 = bool(
        lifecycle.get("profit_zone_latched")
        and current_r > 0.10
        and giveback_r >= 0.20
        and no_new_mfe >= 2
        and giveback_reviews >= 2
        and recovery_fail_reviews >= 2
    )
    mature = bool(
        thesis_available
        and age >= 70.0
        and current_progress >= 0.55
        and giveback_r >= 0.20
        and no_new_mfe >= 3
        and giveback_reviews >= 2
        and (thesis_status in {"WEAKENING", "BROKEN"} or _num(lifecycle.get("continuation_score"), 0.5) < 0.50)
    )
    earlier_mid = bool(
        thesis_status in {"WEAKENING", "BROKEN"}
        and mfe_r >= 0.45
        and current_r <= -0.15
        and giveback_r >= 0.65
        and no_new_mfe >= 2
        and adverse_reviews >= 2
    )

    confirmed_loss = False
    if loss_confirmed:
        if branch == "EXTREME_REVERSAL":
            confirmed_loss = current_r <= -0.25 and adverse_reviews >= 2
        elif branch == "EARLY_FAILURE":
            confirmed_loss = current_r <= -0.15 and (
                adverse_reviews >= 2 or thesis_status in {"WEAKENING", "BROKEN"}
            )
        elif branch == "MID_MFE_REVERSAL":
            confirmed_loss = earlier_mid or (current_r <= -0.30 and adverse_reviews >= 2)
        elif branch == "PERSISTENT_THESIS_INVALIDATION":
            confirmed_loss = thesis_status == "BROKEN" and adverse_reviews >= 2
        else:
            confirmed_loss = adverse_reviews >= 2
    elif earlier_mid:
        confirmed_loss = True
        branch = "MID_MFE_REVERSAL"

    if p80:
        return {
            "action": "P80_PARTIAL_70_PROTECTED_30",
            "state": "PROFIT_EROSION_CONFIRMED",
            "branch": "P80_PERSISTENT_EROSION",
            "actionable": True,
        }
    if mature:
        return {
            "action": "M70_P55_FULL_EXIT",
            "state": "MATURE_PROFIT_EXIT_CONFIRMED",
            "branch": "MATURE_STAGNATION",
            "actionable": True,
        }
    if confirmed_loss:
        return {
            "action": "V3_2_SHADOW_EXIT_CONFIRMED",
            "state": "SHADOW_EXIT_CONFIRMED_ACTIVE",
            "branch": branch,
            "actionable": True,
        }
    return {
        "action": "HOLD",
        "state": "HOLD_PERSISTENCE_REQUIRED",
        "branch": branch,
        "actionable": False,
    }


def apply_live_patch(ns: dict) -> None:
    if ns.get("_RITHAL_BEHAVIOR_V105_R3_APPLIED"):
        return
    required = ("NeuralV2Model", "LiveEngine", "np", "RANK_WINDOW", "RANK_MIN_HISTORY")
    missing = [name for name in required if name not in ns]
    if missing:
        raise RuntimeError("RITHAL_V105_R3_LIVE_CONTRACT_MISSING:" + ",".join(missing))

    Model = ns["NeuralV2Model"]
    Engine = ns["LiveEngine"]
    np = ns["np"]
    rank_window = int(ns["RANK_WINDOW"])
    rank_min = int(ns["RANK_MIN_HISTORY"])
    log = ns.get("log")

    original_repairable = tuple(ns.get("RITHAL_ABSOLUTE_OOD_NEUTRALIZE_FEATURES") or ())
    original_prepare = Model._prepare_model_inputs
    original_context = Engine._trade_manager_entry_context
    original_arm = Engine._arm_trade_manager_position

    # Install base economic behavior first, then replace every contract that needed
    # cross-verification corrections.  No base method is invoked between these steps.
    _base.apply_live_patch(ns)
    repair_lock = ns.get("_RITHAL_BEHAVIOR_V105_R3_REPAIR_LOCK")
    if repair_lock is None:
        repair_lock = threading.RLock()
        ns["_RITHAL_BEHAVIOR_V105_R3_REPAIR_LOCK"] = repair_lock
    ns["RITHAL_ABSOLUTE_OOD_NEUTRALIZE_FEATURES"] = original_repairable

    def prepare_r3(self, df, *, context, force_neutralize_nonstationary=None):
        with repair_lock:
            if force_neutralize_nonstationary:
                if force_neutralize_nonstationary is True:
                    requested = [
                        name
                        for members in _base.FEATURE_FAMILIES.values()
                        for name in members
                    ]
                else:
                    requested = [str(name) for name in force_neutralize_nonstationary]
                expanded, _ = _base.expand_repair_families(
                    requested,
                    getattr(self, "feature_names", ()),
                )
                available = {str(name) for name in getattr(self, "feature_names", ())}
                temporary_contract = tuple(sorted((set(original_repairable) | set(expanded)) & available))
            else:
                temporary_contract = original_repairable
            ns["RITHAL_ABSOLUTE_OOD_NEUTRALIZE_FEATURES"] = temporary_contract
            try:
                return original_prepare(
                    self,
                    df,
                    context=context,
                    force_neutralize_nonstationary=force_neutralize_nonstationary,
                )
            finally:
                ns["RITHAL_ABSOLUTE_OOD_NEUTRALIZE_FEATURES"] = original_repairable

    def _head_scalar(out, name, default=0.0):
        if name not in out:
            return float(default)
        return float(out[name].detach().cpu().numpy().flat[0])

    def score_r3(self, out):
        specialist_conf = 1.0
        specialist_idx = -1
        if "specialist_logits" in out:
            values = out["specialist_logits"].detach().cpu().numpy().reshape(-1)
            if values.size:
                probs = np.exp(values - values.max())
                probs = probs / max(float(probs.sum()), 1e-12)
                specialist_conf = float(probs.max())
                specialist_idx = int(np.argmax(probs))
        regime_logits = (
            out["regime_logits"].detach().cpu().numpy().reshape(-1).tolist()
            if "regime_logits" in out
            else None
        )
        values = {
            "p_long_win": _head_scalar(out, "p_long_win"),
            "p_short_win": _head_scalar(out, "p_short_win"),
            "expected_r_long": _head_scalar(out, "expected_r_long"),
            "expected_r_short": _head_scalar(out, "expected_r_short"),
            "mae_long": _head_scalar(out, "mae_long", 1.0),
            "mae_short": _head_scalar(out, "mae_short", 1.0),
            "calibration_score": _head_scalar(out, "calibration_score"),
            "router_confidence": _head_scalar(out, "router_confidence"),
            "p_no_trade": _head_scalar(out, "p_no_trade", 1.0),
            "risk_score": _head_scalar(out, "risk_score", 0.5),
            "direction_confidence": _head_scalar(out, "direction_confidence", 0.5),
            "quantile_spread": _head_scalar(out, "quantile_spread", 1.0),
            "specialist_conf": specialist_conf,
            "regime_logits": regime_logits,
        }
        result = score_values(values)
        result.update({
            "calibration_score": values["calibration_score"],
            "router_confidence": values["router_confidence"],
            "p_no_trade": values["p_no_trade"],
            "p_long_win": values["p_long_win"],
            "p_short_win": values["p_short_win"],
            "expected_r_long": values["expected_r_long"],
            "expected_r_short": values["expected_r_short"],
            "mae_long": values["mae_long"],
            "mae_short": values["mae_short"],
            "risk_score": values["risk_score"],
            "chosen_specialist_idx": specialist_idx,
            "_raw_model_output": out,
        })
        result["regime_idx"] = (
            int(np.argmax(result["rank_regime_probs"]))
            if result.get("regime_valid")
            else -1
        )
        return result

    def prewarm_r3(self, df, max_bars=None):
        import torch

        if max_bars is None:
            max_bars = rank_window
        self._rank_prewarm_last_attempt_monotonic = time.monotonic()
        attempted = _utc_now()
        _, matrix, health = self._prepare_model_inputs(
            df,
            context="rank_prewarm_original_v105_r3",
        )
        top = list((health or {}).get("top_ood_features") or [])
        original_set = set(original_repairable)
        median_gate = _num(ns.get("RITHAL_SATURATION_OOD_MEDIAN_Z"), 8.0)
        clip_gate = _num(ns.get("RITHAL_SATURATION_OOD_CLIP_FRACTION"), 0.5)
        raw_candidates = [
            str(item.get("feature"))
            for item in top
            if str(item.get("feature")) in original_set
            and (
                _num(item.get("median_abs_z")) >= median_gate
                or _num(item.get("clip_fraction")) >= clip_gate
            )
        ]
        repair_names, repair_families = _base.expand_repair_families(
            raw_candidates,
            getattr(self, "feature_names", ()),
        )
        mode = "ORIGINAL"
        if health.get("blocked") and repair_names:
            _, repaired, repaired_health = self._prepare_model_inputs(
                df,
                context="rank_prewarm_family_v105_r3",
                force_neutralize_nonstationary=repair_names,
            )
            if not repaired_health.get("blocked"):
                matrix, health, mode = repaired, repaired_health, "REPAIRED_FAMILY"
        if health.get("blocked"):
            self._raw_hist = deque(maxlen=rank_window)
            self._rank_prewarm_status = {
                "ok": False,
                "mode": "QUARANTINED",
                "at": attempted,
                "symbol": getattr(self, "symbol", None),
                "count": 0,
                "reasons": list(health.get("blocks") or []),
                "repair_candidates": repair_names,
                "repair_families": repair_families,
            }
            return False

        seq = self.seq_len
        endpoints = list(range(max(seq, len(matrix) - int(max_bars)), len(matrix)))
        if not endpoints:
            self._raw_hist = deque(maxlen=rank_window)
            return False

        composites: list[float] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(endpoints), 96):
                batch_ends = endpoints[start:start + 96]
                batch = np.stack([matrix[index-seq:index] for index in batch_ends])
                tensor = torch.tensor(batch, dtype=torch.float32, device=self.device)
                output = self.model(
                    tensor,
                    symbol_ids=self._semantic_ids(tensor.shape[0], tensor.device),
                )
                for row_index in range(len(batch_ends)):
                    def scalar(name, default=0.0):
                        if name not in output:
                            return float(default)
                        array = output[name].detach().cpu().numpy()
                        return float(array.reshape(len(batch_ends), -1)[row_index, 0])

                    specialist_conf = 1.0
                    if "specialist_logits" in output:
                        logits = output["specialist_logits"].detach().cpu().numpy()[row_index].reshape(-1)
                        probs = np.exp(logits - logits.max())
                        probs = probs / max(float(probs.sum()), 1e-12)
                        specialist_conf = float(probs.max())
                    regime_logits = (
                        output["regime_logits"].detach().cpu().numpy()[row_index].reshape(-1).tolist()
                        if "regime_logits" in output
                        else None
                    )
                    scored = score_values({
                        "p_long_win": scalar("p_long_win"),
                        "p_short_win": scalar("p_short_win"),
                        "expected_r_long": scalar("expected_r_long"),
                        "expected_r_short": scalar("expected_r_short"),
                        "mae_long": scalar("mae_long", 1.0),
                        "mae_short": scalar("mae_short", 1.0),
                        "calibration_score": scalar("calibration_score"),
                        "router_confidence": scalar("router_confidence"),
                        "p_no_trade": scalar("p_no_trade", 1.0),
                        "direction_confidence": scalar("direction_confidence", 0.5),
                        "quantile_spread": scalar("quantile_spread", 1.0),
                        "specialist_conf": specialist_conf,
                        "regime_logits": regime_logits,
                    })
                    if not scored.get("regime_valid"):
                        self._raw_hist = deque(maxlen=rank_window)
                        self._rank_prewarm_status = {
                            "ok": False,
                            "mode": "QUARANTINED_REGIME",
                            "at": attempted,
                            "count": 0,
                            "reasons": ["regime_head_not_valid_four_class"],
                        }
                        return False
                    value = _num(scored.get("raw_composite"), float("nan"))
                    if math.isfinite(value):
                        composites.append(value)

        self._raw_hist = deque(composites, maxlen=rank_window)
        ok = len(self._raw_hist) >= rank_min
        self._rank_prewarm_status = {
            "ok": ok,
            "mode": mode,
            "at": attempted,
            "symbol": getattr(self, "symbol", None),
            "count": len(self._raw_hist),
            "reasons": [] if ok else [f"shallow_history:{len(self._raw_hist)}/{rank_min}"],
            "repair_candidates": repair_names,
            "repair_families": repair_families,
            "scorer": VERSION,
        }
        return ok

    def context_r3(self, sym, pos, pred, threshold_used, bar_timestamp_ms):
        base = dict(
            original_context(
                self,
                sym,
                pos,
                pred,
                threshold_used,
                bar_timestamp_ms,
            )
            or {}
        )
        pred = pred if isinstance(pred, Mapping) else {}
        pos = pos if isinstance(pos, Mapping) else {}
        full = {
            "behavior_contract": VERSION,
            "decision_score": pred.get("decision_score"),
            "threshold": float(threshold_used or 0.0),
            "side": pred.get("side", pos.get("side")),
            "side_policy": pred.get("side_policy"),
            "expected_r_side": pred.get("expected_r_side"),
            "side_disagreement": pred.get("side_disagreement"),
            "expected_r_long": pred.get("expected_r_long"),
            "expected_r_short": pred.get("expected_r_short"),
            "p_long_win": pred.get("p_long_win"),
            "p_short_win": pred.get("p_short_win"),
            "mae_long": pred.get("mae_long"),
            "mae_short": pred.get("mae_short"),
            "long_edge": pred.get("long_edge"),
            "short_edge": pred.get("short_edge"),
            "risk_adjusted_long_edge": pred.get("risk_adjusted_long_edge"),
            "risk_adjusted_short_edge": pred.get("risk_adjusted_short_edge"),
            "edge": pred.get("edge"),
            "edge_margin": pred.get("edge_margin"),
            "router_confidence": pred.get("router_confidence"),
            "calibration_score": pred.get("calibration_score"),
            "p_no_trade": pred.get("p_no_trade"),
            "specialist_conf": pred.get("specialist_conf"),
            "chosen_specialist_idx": pred.get("chosen_specialist_idx"),
            "regime_valid": pred.get("regime_valid"),
            "regime_state": pred.get("regime_state"),
            "true_regime_probs": pred.get("true_regime_probs"),
            "rank_regime_probs": pred.get("rank_regime_probs"),
            "p_trend": pred.get("p_trend"),
            "p_chop": pred.get("p_chop"),
            "p_breakout": pred.get("p_breakout"),
            "rank_p_trend": pred.get("rank_p_trend"),
            "rank_p_chop": pred.get("rank_p_chop"),
            "regime_entropy": pred.get("regime_entropy"),
            "feature_health": _safe(pred.get("feature_health")),
            "dual_inference": _safe(pred.get("dual_inference")),
            "inference_selection": pred.get("inference_selection"),
            "sequence_fingerprint": pred.get("sequence_fingerprint"),
            "original_sequence_fingerprint": pred.get("original_sequence_fingerprint"),
            "semantic_contract_status": pred.get("semantic_contract_status"),
            "semantic_contract_verified": pred.get("semantic_contract_verified"),
            "semantic_economic_ready": pred.get("semantic_economic_ready"),
            "controller_decision": _safe(pos.get("controller_decision")),
            "controller_proposed_multiplier": pos.get("controller_proposed_size_multiplier"),
            "controller_effective_multiplier": pos.get(
                "controller_effective_size_multiplier",
                pos.get("controller_size_multiplier"),
            ),
            "applied_multiplier": pos.get(
                "applied_size_multiplier",
                pos.get("controller_size_multiplier"),
            ),
            "quantity": pos.get("qty"),
            "risk_usd": pos.get("risk_usd"),
            "margin_usd": pos.get("margin_required_est", pos.get("requested_margin_usd")),
            "notional_usd": pos.get("entry_notional_usd"),
            "leverage": pos.get("leverage"),
            "entry_bar_timestamp": bar_timestamp_ms,
            "created_at": _utc_now(),
        }
        base.update(full)
        canonical = json.dumps(_safe(base), sort_keys=True, separators=(",", ":"))
        base["context_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return base

    def arm_r3(self, sym, pos, pred, threshold_used, bar_timestamp_ms):
        if isinstance(pos, dict):
            applied = _num(
                pos.get(
                    "controller_effective_size_multiplier",
                    pos.get("controller_size_multiplier", 1.0),
                ),
                1.0,
            )
            pos["applied_size_multiplier"] = applied
            pos["sizing_truth"] = {
                "proposed_multiplier": pos.get("controller_proposed_size_multiplier"),
                "effective_multiplier": pos.get(
                    "controller_effective_size_multiplier",
                    pos.get("controller_size_multiplier"),
                ),
                "applied_multiplier": applied,
                "quantity_source": "effective_entry_authority_multiplier",
            }
            pos["entry_thesis_snapshot"] = context_r3(
                self,
                sym,
                pos,
                pred,
                threshold_used,
                bar_timestamp_ms,
            )
        return original_arm(
            self,
            sym,
            pos,
            pred,
            threshold_used,
            bar_timestamp_ms,
        )

    Model._prepare_model_inputs = prepare_r3
    Model._score_model_output = score_r3
    Model.prewarm_rank_history = prewarm_r3
    Engine._trade_manager_entry_context = context_r3
    Engine._arm_trade_manager_position = arm_r3
    ns["RITHAL_ABSOLUTE_OOD_NEUTRALIZE_FEATURES"] = original_repairable
    ns["_RITHAL_BEHAVIOR_V105_R3_APPLIED"] = True
    ns["RITHAL_BEHAVIOR_FIX_VERSION"] = VERSION

    if log:
        log.warning(
            "[%s] installed: active signatures verified, repair lock enabled, true/rank regime parity separated",
            VERSION,
        )


def apply_trade_manager_patch(module) -> None:
    if getattr(module, "_RITHAL_BEHAVIOR_V105_R3_TM_APPLIED", False):
        return
    TradeManager = getattr(module, "TradeManager", None)
    if TradeManager is None:
        raise RuntimeError("RITHAL_V105_R3_TRADE_MANAGER_CLASS_MISSING")

    pre_entry_context = TradeManager._entry_context
    pre_review_15m = TradeManager.review_15m_context
    pre_decision = TradeManager._decision
    pre_observe = TradeManager.observe_mark
    pre_public = TradeManager.public_trade_state
    pre_status = TradeManager.status

    _base.apply_trade_manager_patch(module)
    enriched_entry_context = TradeManager._entry_context

    def entry_context_r3(self, pos):
        base = dict(enriched_entry_context(self, pos) or {})
        base["behavior_contract"] = VERSION
        base["context_hash"] = hashlib.sha256(
            json.dumps(_safe(base), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return base

    def review_15m_r3(self, sym, pos, pred, last_bar):
        result = pre_review_15m(self, sym, pos, pred, last_bar)
        trade_id = self._trade_id(pos)
        with self._lock:
            state = self._trades.get(trade_id)
            if not isinstance(state, dict):
                return result

            entry_context = state.get("entry_context")
            if not isinstance(entry_context, dict) or int(_num(entry_context.get("side"), 0.0)) not in {-1, 1}:
                rebuilt = entry_context_r3(self, pos if isinstance(pos, dict) else {})
                if isinstance(rebuilt, dict) and int(_num(rebuilt.get("side"), 0.0)) in {-1, 1}:
                    entry_context = rebuilt
                    state["entry_context"] = entry_context
                else:
                    entry_context = {
                        "behavior_contract": VERSION,
                        "context_status": "UNAVAILABLE",
                        "context_reason": "restored_position_entry_thesis_could_not_be_reconstructed",
                    }
                    state["entry_context"] = entry_context

            if "immutable_entry_snapshot" not in entry_context:
                immutable_source = {
                    key: value
                    for key, value in entry_context.items()
                    if key != "immutable_entry_snapshot"
                }
                entry_context["immutable_entry_snapshot"] = _safe(immutable_source)

            status, reasons = _thesis_status(
                pred if isinstance(pred, Mapping) else {},
                entry_context,
            )
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
                "effective_threshold", "expected_r_long", "expected_r_short",
                "p_long_win", "p_short_win", "mae_long", "mae_short",
                "long_edge", "short_edge", "risk_adjusted_long_edge",
                "risk_adjusted_short_edge", "edge", "edge_margin",
                "router_confidence", "calibration_score", "p_no_trade",
                "true_regime_probs", "rank_regime_probs", "regime_valid",
                "regime_state", "p_breakout", "feature_health",
                "dual_inference", "sequence_fingerprint",
            )
            snapshot = {
                key: pred.get(key)
                for key in snapshot_keys
            } if isinstance(pred, Mapping) else {}
            thesis = {
                "version": VERSION,
                "status": status,
                "reasons": reasons,
                "bar_id": bar_id,
                "reviewed_at": _utc_now(),
                "entry_context_status": entry_context.get("context_status", "AVAILABLE"),
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

    def _block(state):
        block = state.get("trade_manager_v105")
        if not isinstance(block, dict):
            block = {}
            state["trade_manager_v105"] = block
        block.setdefault("version", VERSION)
        block.setdefault("last_review_key", None)
        block.setdefault("last_mfe_r", None)
        block.setdefault("last_current_r", None)
        block.setdefault("no_new_mfe_reviews", 0)
        block.setdefault("giveback_reviews", 0)
        block.setdefault("adverse_reviews", 0)
        block.setdefault("recovery_fail_reviews", 0)
        block.setdefault("active_action", "HOLD")
        block.setdefault("active_state", "HOLD_PERSISTENCE_REQUIRED")
        block.setdefault("last", {})
        return block

    def decision_r3(self, state, review, *, bar_open_ms, hard_flip=None, source="CLOSED_5M"):
        base = pre_decision(
            self,
            state,
            review,
            bar_open_ms=bar_open_ms,
            hard_flip=hard_flip,
            source=source,
        )
        block = _block(state)
        v32 = base.get("trade_manager_v3_1") if isinstance(base.get("trade_manager_v3_1"), dict) else {}
        v33 = base.get("trade_manager_v3_3") if isinstance(base.get("trade_manager_v3_3"), dict) else {}
        metrics32 = v32.get("metrics") if isinstance(v32.get("metrics"), dict) else {}
        metrics33 = v33.get("metrics") if isinstance(v33.get("metrics"), dict) else {}
        current_r = _num(metrics32.get("gross_price_r"), _num(state.get("current_r")))
        mfe_r = _num(metrics32.get("mfe_price_r"), _num(state.get("mfe_r")))
        giveback = max(0.0, _num(metrics32.get("giveback_price_r"), mfe_r - current_r))
        age = _num(v32.get("age_minutes"), _num(metrics33.get("age_minutes")))
        target_r = max(1e-9, _num(metrics33.get("target_r"), _num(state.get("target_r"), 1.0)))
        continuation = _num(
            v32.get("continuation_score"),
            _num((state.get("last_decision") or {}).get("continuation_score"), 0.5),
        )
        review_key = str(
            bar_open_ms
            if bar_open_ms is not None
            else (review or {}).get("bar_open_ms")
            if isinstance(review, dict)
            else "NO_BAR"
        )
        new_review = review_key != str(block.get("last_review_key"))
        if new_review:
            previous_mfe = block.get("last_mfe_r")
            previous_current = block.get("last_current_r")
            expanded = previous_mfe is None or mfe_r >= _num(previous_mfe) + 0.05
            block["no_new_mfe_reviews"] = 0 if expanded else int(block.get("no_new_mfe_reviews") or 0) + 1
            block["giveback_reviews"] = int(block.get("giveback_reviews") or 0) + 1 if giveback >= 0.20 else 0
            adverse = (
                current_r < 0.0
                and previous_current is not None
                and current_r <= _num(previous_current) - 0.05
            )
            block["adverse_reviews"] = int(block.get("adverse_reviews") or 0) + 1 if adverse else 0
            recovered = previous_current is not None and current_r >= _num(previous_current) + 0.10
            block["recovery_fail_reviews"] = (
                0
                if recovered or expanded
                else int(block.get("recovery_fail_reviews") or 0) + 1
            )
            block["last_review_key"] = review_key
            block["last_mfe_r"] = mfe_r
            block["last_current_r"] = current_r

        sentinel = v33.get("intrabar_profit_sentinel") if isinstance(v33.get("intrabar_profit_sentinel"), dict) else {}
        thesis = state.get("thesis_v105") if isinstance(state.get("thesis_v105"), dict) else {}
        lifecycle = {
            **block,
            "profit_zone_latched": bool(
                sentinel.get("profit_zone_latched")
                or (state.get("trade_manager_v3_3") or {}).get("profit_zone_latched")
            ),
            "continuation_score": continuation,
        }
        metrics = {
            "current_r": current_r,
            "mfe_r": mfe_r,
            "giveback_r": giveback,
            "age_minutes": age,
            "target_r": target_r,
            "current_progress": current_r / target_r,
        }
        policy = _manager_policy(
            metrics,
            lifecycle,
            thesis.get("status", "UNAVAILABLE"),
            v32.get("manager_state"),
            v32.get("branch"),
        )
        block["active_action"] = policy["action"]
        block["active_state"] = policy["state"]
        block["last"] = {
            "version": VERSION,
            "at": _utc_now(),
            "review_key": review_key,
            "new_review": new_review,
            "thesis_status": thesis.get("status", "UNAVAILABLE"),
            "metrics": metrics,
            "counters": {
                key: block.get(key)
                for key in (
                    "no_new_mfe_reviews",
                    "giveback_reviews",
                    "adverse_reviews",
                    "recovery_fail_reviews",
                )
            },
            "policy": policy,
            "actual_execution": "ROUTED_BY_EXISTING_PAPER_AUTHORITY_ONLY",
        }

        v33 = dict(v33)
        if policy["actionable"]:
            if policy["action"] == "P80_PARTIAL_70_PROTECTED_30":
                v33["final_manager_state"] = "PROFIT_EROSION_CONFIRMED"
                v33["profit_state"] = "PROFIT_EROSION_CONFIRMED"
                v33["final_shadow_recommendation"] = "P80_PARTIAL_70_PROTECTED_30"
                v33["profit_reason_codes"] = [
                    "profit_zone_latched",
                    "persistent_closed_5m_erosion",
                    "no_new_mfe_two_reviews",
                    "recovery_failed_two_reviews",
                ]
                sentinel = dict(v33.get("intrabar_profit_sentinel") or {})
                sentinel["profit_zone_latched"] = True
                sentinel["v105_persistence_confirmed"] = True
                v33["intrabar_profit_sentinel"] = sentinel
            elif policy["action"] == "M70_P55_FULL_EXIT":
                v33["final_manager_state"] = "MATURE_PROFIT_EXIT_LATCHED"
                v33["profit_state"] = "MATURE_PROFIT_EXIT_LATCHED"
                v33["final_shadow_recommendation"] = "M70_P55_FULL_EXIT"
                v33["profit_reason_codes"] = [
                    "age_gte_70",
                    "current_progress_gte_55_percent",
                    "no_new_mfe_three_reviews",
                    "persistent_giveback",
                    "thesis_or_continuation_deteriorated",
                ]
                sentinel = dict(v33.get("intrabar_profit_sentinel") or {})
                sentinel["mature_profit_exit_latched"] = True
                sentinel["v105_persistence_confirmed"] = True
                v33["intrabar_profit_sentinel"] = sentinel
            else:
                v32 = dict(v32)
                v32["manager_state"] = "SHADOW_EXIT_CONFIRMED_ACTIVE"
                v32["branch"] = policy["branch"]
                v32["severity"] = "ACTIONABLE_SHADOW_ACTIVE"
                v32["reason_codes"] = list(v32.get("reason_codes") or []) + [
                    "v105_r3_persistent_evidence_confirmed"
                ]
                base["trade_manager_v3_1"] = v32
                v33["final_manager_state"] = "SHADOW_EXIT_CONFIRMED_ACTIVE"
                v33["final_shadow_recommendation"] = "V3_2_SHADOW_EXIT_CONFIRMED"
                v33["loss_recovery_layer"] = {
                    "version": VERSION,
                    "manager_state": "SHADOW_EXIT_CONFIRMED_ACTIVE",
                    "branch": policy["branch"],
                    "severity": "ACTIONABLE_SHADOW_ACTIVE",
                    "reason_codes": [
                        "persistent_closed_5m_evidence",
                        "thesis_aware_loss_confirmation",
                    ],
                }
        else:
            v33["final_manager_state"] = "HOLD_PERSISTENCE_REQUIRED"
            v33["final_shadow_recommendation"] = "HOLD"
            v33["profit_state"] = "HOLD_PERSISTENCE_REQUIRED"
            v33["profit_reason_codes"] = ["closed_5m_persistence_not_yet_confirmed"]
            sentinel = dict(v33.get("intrabar_profit_sentinel") or {})
            sentinel["v105_persistence_confirmed"] = False
            sentinel["mature_profit_exit_latched"] = False
            v33["intrabar_profit_sentinel"] = sentinel

        v33["behavior_v105"] = dict(block["last"])
        v33["actual_execution"] = "ROUTED_BY_EXISTING_PAPER_AUTHORITY_ONLY"
        base["trade_manager_v3_3"] = v33
        base["trade_manager_v105"] = dict(block["last"])
        base["unified_manager_state"] = v33.get("final_manager_state")
        base["unified_shadow_recommendation"] = v33.get("final_shadow_recommendation")
        base["selected_action"] = "HOLD"
        base["authority"] = "SHADOW"
        base["plan"] = {
            "kind": "NONE",
            "paper_only": True,
            "requires_paper_full_exit": False,
        }
        return base

    def observe_r3(self, sym, pos, current_price):
        output = dict(pre_observe(self, sym, pos, current_price) or {})
        trade_id = self._trade_id(pos)
        with self._lock:
            state = self._trades.get(trade_id)
            block = _block(state) if isinstance(state, dict) else {}
            active = str(block.get("active_action") or "HOLD")
            detail = dict(block.get("last") or {})
        v33 = output.get("trade_manager_v3_3") if isinstance(output.get("trade_manager_v3_3"), dict) else {}
        v33 = dict(v33)
        if active == "M70_P55_FULL_EXIT":
            v33.update({
                "final_manager_state": "MATURE_PROFIT_EXIT_LATCHED",
                "profit_state": "MATURE_PROFIT_EXIT_LATCHED",
                "final_shadow_recommendation": active,
            })
        elif active == "P80_PARTIAL_70_PROTECTED_30":
            v33.update({
                "final_manager_state": "PROFIT_EROSION_CONFIRMED",
                "profit_state": "PROFIT_EROSION_CONFIRMED",
                "final_shadow_recommendation": active,
            })
        elif active == "V3_2_SHADOW_EXIT_CONFIRMED":
            v33.update({
                "final_manager_state": "SHADOW_EXIT_CONFIRMED_ACTIVE",
                "final_shadow_recommendation": active,
            })
        else:
            v33.update({
                "final_manager_state": "HOLD_PERSISTENCE_REQUIRED",
                "profit_state": "HOLD_PERSISTENCE_REQUIRED",
                "final_shadow_recommendation": "HOLD",
                "authority_suppressed_by_v105_r3": True,
            })
        v33["behavior_v105"] = detail
        output["trade_manager_v3_3"] = v33
        output["trade_manager_v105"] = detail
        output["unified_manager_state"] = v33.get("final_manager_state")
        output["unified_shadow_recommendation"] = v33.get("final_shadow_recommendation")
        output["selected_action"] = "HOLD"
        output["authority"] = "SHADOW"
        return output

    def public_r3(self, trade_id):
        output = dict(pre_public(self, trade_id) or {})
        with self._lock:
            state = self._trades.get(str(trade_id))
            if isinstance(state, dict):
                output["trade_manager_v105"] = dict(_block(state).get("last") or {})
                output["thesis_v105"] = dict(state.get("thesis_v105") or {})
        return output

    def status_r3(self):
        output = dict(pre_status(self) or {})
        output["behavior_v105"] = {
            "version": VERSION,
            "single_canonical_manager": True,
            "closed_5m_persistence_required": True,
            "age_only_mature_exit_disabled": True,
            "first_touch_p80_execution_disabled": True,
            "thesis_aware_loss_confirmation": True,
            "missing_thesis_fails_closed": True,
            "restored_entry_context_reconstructed": True,
            "paper_authority_adapter_reused": True,
            "live_exchange_control": False,
        }
        return output

    TradeManager._entry_context = entry_context_r3
    TradeManager.review_15m_context = review_15m_r3
    TradeManager._decision = decision_r3
    TradeManager.observe_mark = observe_r3
    TradeManager.public_trade_state = public_r3
    TradeManager.status = status_r3
    module._RITHAL_BEHAVIOR_V105_R3_TM_APPLIED = True
    module.RITHAL_BEHAVIOR_FIX_VERSION = VERSION


def configure_project(
    project_root: Path,
    instance_id: str = INSTANCE_ID,
    manager_mode: str = "PAPER_CONTROL",
) -> dict:
    mode = str(manager_mode).upper()
    if mode not in {"PAPER_CONTROL", "SHADOW_ONLY"}:
        raise ValueError(f"unsupported manager mode: {manager_mode}")
    report = _base.configure_project(
        Path(project_root),
        instance_id=instance_id,
        manager_mode=mode,
    )
    root = Path(project_root).resolve()
    control_path = root / "mythos_5m_execution_control.json"
    control: dict = {}
    if control_path.is_file():
        try:
            payload = json.loads(control_path.read_text(encoding="utf-8-sig"))
            control = payload if isinstance(payload, dict) else {}
        except Exception:
            control = {}
    if mode == "PAPER_CONTROL":
        control.update({
            "mode": "PAPER_AUTOMANAGE",
            "execution_enabled": True,
            "scope": "PAPER_ONLY",
        })
    else:
        control.update({
            "mode": "SHADOW_ONLY",
            "execution_enabled": False,
            "scope": "PAPER_ONLY",
        })
    control.update({"updated_at": _utc_now(), "updated_by": VERSION})
    temp = control_path.with_suffix(".tmp")
    temp.write_text(json.dumps(control, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, control_path)
    return {
        "version": VERSION,
        "status": "PASS",
        "manager_mode": mode,
        "control_mode": control["mode"],
        "execution_enabled": control["execution_enabled"],
        "base_report": report,
    }


def authority_contract_self_test(project_root: Path) -> dict:
    root = Path(project_root).resolve()
    neural = root / "mythos" / "neural"
    sys.path.insert(0, str(neural))
    try:
        from rithal_manager_authority_contract import classify_exact_instruction
    finally:
        if sys.path and sys.path[0] == str(neural):
            sys.path.pop(0)

    settings = {
        "trade_manager": {
            "mode": "PAPER_CONTROL",
            "apply_to_existing_positions": True,
            "loss_protection_enabled": True,
            "early_failure_exit_enabled": True,
            "extreme_reversal_exit_enabled": True,
            "mid_mfe_reversal_exit_enabled": True,
            "mature_profit_exit_enabled": True,
            "p80_partial_harvest_enabled": True,
            "hard_profit_cap_enabled": False,
            "parity_lock_required": True,
        },
        "safety": {
            "paper_only": True,
            "live_exchange_control": False,
            "manager_can_resize_positions": True,
            "manager_resize_scope": "P80_REDUCE_ONLY_70_PERCENT",
        },
    }
    position = {
        "trade_id": "r3-authority-test",
        "symbol": "BTCUSDT",
        "side": 1,
        "paper": True,
        "entry": 100.0,
        "tp": 103.0,
        "sl": 99.0,
        "qty": 1.0,
        "risk_usd": 100.0,
        "last_mark": 101.0,
    }

    def classify(instruction):
        return classify_exact_instruction(
            copy.deepcopy(instruction),
            dict(position),
            copy.deepcopy(settings),
            current_price=101.0,
            source="RITHAL_V105_R3_INSTALL_PREFLIGHT",
        )

    common = {
        "selected_action": "HOLD",
        "authority": "SHADOW",
        "plan": {"kind": "NONE", "paper_only": True},
        "source": VERSION,
    }
    hold = classify({
        **common,
        "trade_manager_v3_3": {
            "final_manager_state": "HOLD_PERSISTENCE_REQUIRED",
            "profit_state": "HOLD_PERSISTENCE_REQUIRED",
            "final_shadow_recommendation": "HOLD",
        },
    })
    mature = classify({
        **common,
        "trade_manager_v3_3": {
            "final_manager_state": "MATURE_PROFIT_EXIT_LATCHED",
            "profit_state": "MATURE_PROFIT_EXIT_LATCHED",
            "final_shadow_recommendation": "M70_P55_FULL_EXIT",
            "intrabar_profit_sentinel": {
                "mature_profit_exit_latched": True,
                "v105_persistence_confirmed": True,
            },
        },
    })
    p80 = classify({
        **common,
        "trade_manager_v3_3": {
            "final_manager_state": "PROFIT_EROSION_CONFIRMED",
            "profit_state": "PROFIT_EROSION_CONFIRMED",
            "final_shadow_recommendation": "P80_PARTIAL_70_PROTECTED_30",
            "intrabar_profit_sentinel": {
                "profit_zone_latched": True,
                "v105_persistence_confirmed": True,
            },
        },
    })
    loss = classify({
        **common,
        "trade_manager_v3_1": {
            "manager_state": "SHADOW_EXIT_CONFIRMED_ACTIVE",
            "branch": "EARLY_FAILURE",
            "severity": "ACTIONABLE_SHADOW_ACTIVE",
        },
        "trade_manager_v3_3": {
            "final_manager_state": "SHADOW_EXIT_CONFIRMED_ACTIVE",
            "final_shadow_recommendation": "V3_2_SHADOW_EXIT_CONFIRMED",
            "loss_recovery_layer": {
                "manager_state": "SHADOW_EXIT_CONFIRMED_ACTIVE",
                "branch": "EARLY_FAILURE",
                "severity": "ACTIONABLE_SHADOW_ACTIVE",
            },
        },
    })

    observed = {
        "hold": hold.get("candidate_type"),
        "mature": mature.get("candidate_type"),
        "p80": p80.get("candidate_type"),
        "p80_fraction": p80.get("close_fraction"),
        "loss": loss.get("candidate_type"),
        "loss_branch": loss.get("loss_branch"),
    }
    expected = {
        "hold": "NONE",
        "mature": "MATURE_PROFIT_EXIT",
        "p80": "PROFIT_HARVEST_PARTIAL_70",
        "loss": "LOSS_EXIT",
    }
    errors = []
    for key, value in expected.items():
        if observed.get(key) != value:
            errors.append(f"{key}:{observed.get(key)}!={value}")
    if observed.get("p80_fraction") is None or abs(float(observed["p80_fraction"]) - 0.70) > 1e-9:
        errors.append(f"p80_fraction:{observed.get('p80_fraction')}!=0.7")
    if str(observed.get("loss_branch") or "").upper() != "EARLY_FAILURE":
        errors.append(f"loss_branch:{observed.get('loss_branch')}!=EARLY_FAILURE")
    if errors:
        raise RuntimeError("RITHAL_V105_R3_AUTHORITY_CONTRACT_MISMATCH:" + ";".join(errors))
    return {"version": VERSION, "status": "PASS", "observed": observed}


def self_test() -> dict:
    scored = score_values({
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
        "regime_logits": [1.0, 0.2, -0.4, 0.8],
    })
    assert scored["side"] == -1
    assert scored["expected_r_side"] == 1
    assert scored["side_disagreement"] is True
    assert scored["regime_valid"] is True
    assert scored["true_regime_probs"] != scored["rank_regime_probs"]
    assert abs(sum(scored["true_regime_probs"]) - 1.0) < 1e-9
    assert abs(sum(scored["rank_regime_probs"]) - 1.0) < 1e-9

    invalid = score_values({"regime_logits": [1.0, 2.0, 3.0]})
    assert invalid["regime_valid"] is False
    assert invalid["regime_state"] == "REGIME_UNKNOWN"

    unavailable, _ = _thesis_status({}, {"side": 1})
    assert unavailable == "UNAVAILABLE"
    unavailable_regime, _ = _thesis_status(
        {
            "side": -1,
            "decision_score": 0.9,
            "threshold": 0.8,
            "edge_margin": 0.2,
            "router_confidence": 0.9,
            "p_no_trade": 0.1,
        },
        {"side": 1},
    )
    assert unavailable_regime == "UNAVAILABLE"

    mature_missing = _manager_policy(
        {
            "current_r": 1.8,
            "mfe_r": 2.2,
            "giveback_r": 0.4,
            "age_minutes": 90,
            "target_r": 3.0,
            "current_progress": 0.60,
        },
        {
            "no_new_mfe_reviews": 4,
            "giveback_reviews": 3,
            "continuation_score": 0.2,
        },
        "UNAVAILABLE",
        "HOLD_NORMAL",
        "NONE",
    )
    assert mature_missing["action"] == "HOLD"

    p80 = _manager_policy(
        {
            "current_r": 1.5,
            "mfe_r": 2.4,
            "giveback_r": 0.9,
            "age_minutes": 50,
            "target_r": 3.0,
            "current_progress": 0.5,
        },
        {
            "profit_zone_latched": True,
            "no_new_mfe_reviews": 2,
            "giveback_reviews": 2,
            "recovery_fail_reviews": 2,
        },
        "UNAVAILABLE",
        "HOLD_NORMAL",
        "NONE",
    )
    assert p80["action"] == "P80_PARTIAL_70_PROTECTED_30"

    expanded, families = _base.expand_repair_families(
        ["open_interest"],
        ["open_interest", "open_interest_delta", "liq_total_usd_15m"],
    )
    assert expanded == ["open_interest", "open_interest_delta"]
    assert families == ["OPEN_INTEREST"]

    # Exact active-source method contract: threshold and timestamp are separate.
    class SyntheticModel:
        feature_names = ("open_interest", "open_interest_delta")
        seq_len = 2

        def _prepare_model_inputs(self, df, *, context, force_neutralize_nonstationary=None):
            contract = tuple(namespace["RITHAL_ABSOLUTE_OOD_NEUTRALIZE_FEATURES"])
            return None, [], {"contract": contract, "blocked": False, "top_ood_features": []}

        def _run_scored_sequence(self, sequence):
            return {}

    class SyntheticEngine:
        def _get_regime_probs(self, sym, pred):
            return {}

        def _trade_manager_entry_context(self, sym, pos, pred, threshold_used, bar_timestamp_ms):
            return {"threshold_used": threshold_used, "entry_bar_timestamp": bar_timestamp_ms}

        def _arm_trade_manager_position(self, sym, pos, pred, threshold_used, bar_timestamp_ms):
            pos["armed_context"] = self._trade_manager_entry_context(
                sym, pos, pred, threshold_used, bar_timestamp_ms
            )
            return None

        def _process(self, sym, *args, **kwargs):
            return None

    class NPStub:
        pass

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
    engine = SyntheticEngine()
    position = {"side": 1, "controller_size_multiplier": 0.5}
    engine._arm_trade_manager_position(
        "BTCUSDT",
        position,
        {"side": 1, "decision_score": 0.9, "regime_valid": True},
        0.88,
        123456,
    )
    assert position["armed_context"]["threshold"] == 0.88
    assert position["armed_context"]["entry_bar_timestamp"] == 123456
    assert position["applied_size_multiplier"] == 0.5
    signature = inspect.signature(SyntheticEngine._arm_trade_manager_position)
    assert list(signature.parameters) == [
        "self", "sym", "pos", "pred", "threshold_used", "bar_timestamp_ms"
    ]

    model = SyntheticModel()
    _, _, forced = model._prepare_model_inputs(
        None,
        context="forced",
        force_neutralize_nonstationary=["open_interest"],
    )
    _, _, normal = model._prepare_model_inputs(None, context="normal")
    assert "open_interest_delta" in forced["contract"]
    assert normal["contract"] == ("open_interest",)
    assert namespace["RITHAL_ABSOLUTE_OOD_NEUTRALIZE_FEATURES"] == ("open_interest",)

    return {
        "version": VERSION,
        "status": "PASS",
        "checks": [
            "risk_adjusted_side",
            "true_and_rank_regime_separation",
            "invalid_regime_fail_closed",
            "missing_thesis_fail_closed",
            "mature_exit_missing_thesis_blocked",
            "p80_persistence_contract",
            "family_expansion_scope",
            "active_live_method_signature",
            "forced_repair_isolation",
            "applied_sizing_truth",
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
        print(json.dumps(authority_contract_self_test(Path(args.project_root)), indent=2, sort_keys=True))
        return 0
    if args.configure:
        print(json.dumps(configure_project(
            Path(args.project_root),
            instance_id=args.instance_id,
            manager_mode=args.manager_mode,
        ), indent=2, sort_keys=True, default=str))
        return 0
    # Default execution is intentionally a deterministic self-test.
    print(json.dumps(self_test(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
