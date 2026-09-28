from __future__ import annotations

"""Rithal V1.0.5 behavior repair overlay.

This module is an incremental runtime overlay for the existing Rithal engine. It
never edits neural checkpoints, the 168-feature order, static deployment score
thresholds, fees, or the original TP/SL geometry. It changes PAPER behavior in
five tightly scoped places:

* risk-adjusted side selection and one canonical scorer;
* coherent feature-family repair with fail-closed regime handling;
* complete immutable 15m thesis handoff to the single 5m TradeManager;
* conservative persistent-evidence manager arbitration;
* explicit proposed/effective/applied sizing truth and PAPER monthly-cap bypass.

LIVE exchange authority remains disabled. Manager economic effects are routed
only through the pre-existing parity-locked PAPER authority adapter.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

VERSION = "RITHAL_BEHAVIOR_FIX_V1_0_5"
INSTANCE_ID = "rithal-1-0-contract-locked"

FEATURE_FAMILIES = {
    "OPEN_INTEREST": (
        "open_interest", "binance_oi_usd_15m", "open_interest_usd",
        "open_interest_delta", "open_interest_delta_pct", "oi_delta",
        "oi_delta_pct", "oi_change", "oi_change_pct", "oi_z", "oi_z_7d",
        "oi_change_z", "cross_oi", "cross_oi_delta", "cross_oi_z",
    ),
    "LIQUIDATION": (
        "liq_total_usd_15m", "long_liq_usd_15m", "short_liq_usd_15m",
        "liq_long_usd_15m", "liq_short_usd_15m", "liq_imbalance",
        "liq_imbalance_z", "liq_spike", "liq_spike_z",
    ),
    "AGG_TRADE": (
        "agg_trade_notional_15m", "agg_buy_notional_15m",
        "agg_sell_notional_15m", "agg_trade_imbalance",
        "agg_trade_delta", "agg_trade_delta_z", "vpin", "vpin_z",
    ),
}


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
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return _safe(value.item())
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items() if not str(k).startswith("_raw_model_output")}
    if isinstance(value, (list, tuple, set)):
        return [_safe(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return _safe(value.to_dict())
        except Exception:
            pass
    return str(value)


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    top = max(float(x) for x in values)
    exps = [math.exp(float(x) - top) for x in values]
    total = sum(exps)
    return [x / total for x in exps] if total > 0 else []


def _entropy(probs: Sequence[float]) -> float:
    return -sum(float(p) * math.log(max(float(p), 1e-12)) for p in probs)


def _family_for_feature(name: str) -> Optional[str]:
    name = str(name)
    for family, members in FEATURE_FAMILIES.items():
        if name in members:
            return family
    return None


def expand_repair_families(candidates: Iterable[str], available: Iterable[str]) -> tuple[list[str], list[str]]:
    available_set = {str(x) for x in available}
    requested = {str(x) for x in candidates}
    families = {family for item in requested if (family := _family_for_feature(item))}
    expanded = set(requested)
    for family in families:
        expanded.update(name for name in FEATURE_FAMILIES[family] if name in available_set)
    return sorted(expanded & available_set), sorted(families)


def score_values(values: Mapping[str, Any]) -> dict:
    """Pure canonical scorer used by runtime code and installation tests."""
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

    logits = values.get("regime_logits")
    regime_valid = isinstance(logits, (list, tuple)) and len(logits) == 4 and all(math.isfinite(_num(x, float("nan"))) for x in logits)
    true_probs = _softmax([float(x) for x in logits]) if regime_valid else []
    if len(true_probs) != 4:
        regime_valid = False
        true_probs = []
    p_chop = true_probs[2] if regime_valid else None
    p_breakout = true_probs[3] if regime_valid else None
    p_trend = true_probs[0] + true_probs[1] if regime_valid else None
    regime_mult = 0.0
    if regime_valid:
        regime_mult = (0.35 + 0.65 * min(1.0, max(0.0, p_trend)))
        regime_mult *= 1.0 - 0.30 * min(1.0, max(0.0, p_chop))

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

    return {
        "side": side,
        "expected_r_side": expected_r_side,
        "side_policy": "RISK_ADJUSTED_EDGE_V105",
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
        "regime_state": ("TREND_UP", "TREND_DOWN", "CHOP", "BREAKOUT")[max(range(4), key=true_probs.__getitem__)] if regime_valid else "REGIME_UNKNOWN",
        "true_regime_probs": true_probs or None,
        "rank_regime_probs": true_probs or None,
        "regime_probability_source": "TRUE_POSTERIOR_NO_SECOND_SOFTMAX" if regime_valid else "UNAVAILABLE",
        "p_trend": p_trend,
        "p_chop": p_chop,
        "p_breakout": p_breakout,
        "regime_entropy": _entropy(true_probs) if regime_valid else None,
        "specialist_conf": float(specialist_conf),
        "spread_confidence": float(spread_conf),
        "saturation_state": saturation_state,
    }


def _manager_policy(metrics: Mapping[str, Any], lifecycle: Mapping[str, Any], thesis_status: str, existing_loss_state: str, existing_loss_branch: str) -> dict:
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
        age >= 70.0
        and current_progress >= 0.55
        and giveback_r >= 0.20
        and no_new_mfe >= 3
        and giveback_reviews >= 2
        and (thesis_status in {"WEAKENING", "BROKEN"} or _num(lifecycle.get("continuation_score"), 0.5) < 0.50)
    )
    earlier_mid = bool(
        mfe_r >= 0.45
        and current_r <= -0.15
        and giveback_r >= 0.65
        and no_new_mfe >= 2
        and adverse_reviews >= 2
        and thesis_status in {"WEAKENING", "BROKEN"}
    )
    confirmed_loss = False
    if loss_confirmed:
        if branch == "EXTREME_REVERSAL":
            confirmed_loss = current_r <= -0.25
        elif branch == "EARLY_FAILURE":
            confirmed_loss = current_r <= -0.15 and (adverse_reviews >= 2 or thesis_status in {"WEAKENING", "BROKEN"})
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
        return {"action": "P80_PARTIAL_70_PROTECTED_30", "state": "PROFIT_EROSION_CONFIRMED", "branch": "P80_PERSISTENT_EROSION", "actionable": True}
    if mature:
        return {"action": "M70_P55_FULL_EXIT", "state": "MATURE_PROFIT_EXIT_CONFIRMED", "branch": "MATURE_STAGNATION", "actionable": True}
    if confirmed_loss:
        return {"action": "V3_2_SHADOW_EXIT_CONFIRMED", "state": "SHADOW_EXIT_CONFIRMED_ACTIVE", "branch": branch, "actionable": True}
    return {"action": "HOLD", "state": "HOLD_PERSISTENCE_REQUIRED", "branch": branch, "actionable": False}


def apply_live_patch(ns: dict) -> None:
    if ns.get("_RITHAL_BEHAVIOR_V105_APPLIED"):
        return
    required = ("NeuralV2Model", "LiveEngine", "np", "RANK_WINDOW", "RANK_MIN_HISTORY")
    missing = [name for name in required if name not in ns]
    if missing:
        raise RuntimeError("RITHAL_V105_LIVE_CONTRACT_MISSING:" + ",".join(missing))
    ns["_RITHAL_BEHAVIOR_V105_APPLIED"] = True
    np = ns["np"]
    Model = ns["NeuralV2Model"]
    Engine = ns["LiveEngine"]
    log = ns.get("log")
    rank_window = int(ns["RANK_WINDOW"])
    rank_min = int(ns["RANK_MIN_HISTORY"])

    # Expand only the list of fields eligible for a forced repair. Feature order is
    # untouched; absent names are ignored by the existing input-preparation loop.
    existing_repairable = tuple(ns.get("RITHAL_ABSOLUTE_OOD_NEUTRALIZE_FEATURES") or ())
    expanded_repairable = list(existing_repairable)
    for members in FEATURE_FAMILIES.values():
        for name in members:
            if name not in expanded_repairable:
                expanded_repairable.append(name)
    ns["RITHAL_ABSOLUTE_OOD_NEUTRALIZE_FEATURES"] = tuple(expanded_repairable)

    def _head_scalar(out, name, default=0.0):
        if name not in out:
            return float(default)
        return float(out[name].detach().cpu().numpy().flat[0])

    def score_v105(self, out):
        spec_conf = 1.0
        spec_idx = -1
        if "specialist_logits" in out:
            arr = out["specialist_logits"].detach().cpu().numpy().reshape(-1)
            if arr.size:
                probs = np.exp(arr - arr.max())
                probs = probs / max(float(probs.sum()), 1e-12)
                spec_conf = float(probs.max())
                spec_idx = int(np.argmax(probs))
        rlog = out["regime_logits"].detach().cpu().numpy().reshape(-1).tolist() if "regime_logits" in out else None
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
            "specialist_conf": spec_conf,
            "regime_logits": rlog,
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
            "chosen_specialist_idx": spec_idx,
            "_raw_model_output": out,
        })
        result["regime_idx"] = int(np.argmax(result["true_regime_probs"])) if result.get("regime_valid") else -1
        return result

    def _repair_candidates(self, health):
        top = list((health or {}).get("top_ood_features") or [])
        eligible = set(ns["RITHAL_ABSOLUTE_OOD_NEUTRALIZE_FEATURES"])
        median_gate = _num(ns.get("RITHAL_SATURATION_OOD_MEDIAN_Z"), 8.0)
        clip_gate = _num(ns.get("RITHAL_SATURATION_OOD_CLIP_FRACTION"), 0.5)
        raw = [str(item.get("feature")) for item in top if str(item.get("feature")) in eligible and (_num(item.get("median_abs_z")) >= median_gate or _num(item.get("clip_fraction")) >= clip_gate)]
        return expand_repair_families(raw, getattr(self, "feature_names", ()))

    def infer_v105(self, df):
        try:
            _, original_matrix, original_health = self._prepare_model_inputs(df, context="infer_original_v105")
        except Exception as exc:
            if log:
                log.error("[%s] V1.0.5 feature error: %s", getattr(self, "symbol", "?"), exc)
            return None
        if len(original_matrix) < self.seq_len:
            return None
        original_sequence = np.ascontiguousarray(original_matrix[-self.seq_len:], dtype=np.float32)
        original_fp = hashlib.sha256(original_sequence.tobytes()).hexdigest()
        original_score = self._run_scored_sequence(original_sequence)
        repair_names, repair_families = _repair_candidates(self, original_health)
        original_repairable_block = any("_ood:" in str(x) or "severe_feature_ood:" in str(x) for x in list(original_health.get("blocks") or []))
        should_repair = bool(repair_names and (original_repairable_block or original_score.get("saturation_state") == "HARD_SATURATED"))
        selected_score = original_score
        selected_health = dict(original_health)
        selected_fp = original_fp
        dual = {
            "version": VERSION,
            "attempted": should_repair,
            "selected": "ORIGINAL",
            "reason": "repair_not_required",
            "repair_candidates": repair_names,
            "repair_families": repair_families,
            "original": {"side": original_score.get("side"), "expected_r_side": original_score.get("expected_r_side"), "raw_composite": original_score.get("raw_composite"), "regime_valid": original_score.get("regime_valid"), "saturation_state": original_score.get("saturation_state"), "feature_status": original_health.get("status"), "fingerprint": original_fp},
        }
        if should_repair:
            try:
                _, repaired_matrix, repaired_health = self._prepare_model_inputs(df, context="infer_repaired_family_v105", force_neutralize_nonstationary=repair_names)
                repaired_sequence = np.ascontiguousarray(repaired_matrix[-self.seq_len:], dtype=np.float32)
                repaired_fp = hashlib.sha256(repaired_sequence.tobytes()).hexdigest()
                repaired_score = self._run_scored_sequence(repaired_sequence)
                p_chop = _num(repaired_score.get("p_chop"), 1.0)
                min_margin = _num(getattr(self, "config", {}).get("min_edge_margin", 0.08), 0.08)
                repaired_safe = bool(
                    not repaired_health.get("blocked", False)
                    and repaired_score.get("regime_valid")
                    and p_chop <= _num(getattr(self, "CHOP_GATE_THRESHOLD", 0.60), 0.60)
                    and _num(repaired_score.get("edge")) > 0.0
                    and _num(repaired_score.get("edge_margin")) >= min_margin
                    and repaired_score.get("saturation_state") != "HARD_SATURATED"
                )
                dual["direction_changed"] = repaired_score.get("side") != original_score.get("side")
                dual["repaired"] = {"side": repaired_score.get("side"), "expected_r_side": repaired_score.get("expected_r_side"), "raw_composite": repaired_score.get("raw_composite"), "regime_valid": repaired_score.get("regime_valid"), "p_chop": repaired_score.get("p_chop"), "edge": repaired_score.get("edge"), "edge_margin": repaired_score.get("edge_margin"), "saturation_state": repaired_score.get("saturation_state"), "feature_status": repaired_health.get("status"), "fingerprint": repaired_fp, "neutralized": list(repaired_health.get("neutralized_ood_features") or [])}
                if repaired_safe:
                    selected_score = repaired_score
                    selected_health = dict(repaired_health)
                    selected_health["status"] = "REPAIRED_FAMILY_REINFER"
                    selected_health["blocked"] = False
                    selected_health["blocks"] = [x for x in list(selected_health.get("blocks") or []) if "hard_saturation_on_ood_inputs" not in str(x)]
                    selected_fp = repaired_fp
                    dual["selected"] = "REPAIRED"
                    dual["reason"] = "coherent_family_repair_healthy_direction_change_allowed"
                else:
                    dual["reason"] = "family_repair_failed_health_regime_edge_or_chop"
            except Exception as exc:
                dual["reason"] = f"family_repair_error:{type(exc).__name__}:{exc}"

        if not selected_score.get("regime_valid"):
            selected_health.setdefault("blocks", []).append("regime_unknown_or_invalid_four_class_posterior")
            selected_health["blocked"] = True
            selected_health["status"] = "BLOCK"
        if selected_score.get("saturation_state") == "HARD_SATURATED" and dual.get("selected") != "REPAIRED":
            selected_health.setdefault("blocks", []).append("unresolved_hard_saturation_after_family_repair")
            selected_health["blocked"] = True
            selected_health["status"] = "BLOCK"
        selected_health["dual_inference"] = dual
        selected_health["repair_families"] = repair_families

        if not hasattr(self, "_raw_hist"):
            self._raw_hist = deque(maxlen=rank_window)
        hist = self._raw_hist
        if not selected_health.get("blocked") and len(hist) < rank_min:
            now_mono = time.monotonic()
            last_attempt = _num(getattr(self, "_rank_prewarm_last_attempt_monotonic", 0.0))
            if now_mono - last_attempt >= _num(ns.get("RITHAL_RANK_PREWARM_RETRY_SECONDS"), 900.0):
                try:
                    self.prewarm_rank_history(df)
                except Exception:
                    pass
                hist = self._raw_hist
        raw = _num(selected_score.get("raw_composite"))
        if selected_health.get("blocked"):
            decision_score = 0.0
        elif len(hist) >= rank_min:
            arr = np.fromiter(hist, dtype=float)
            decision_score = float((arr <= raw).mean())
            hist.append(raw)
        else:
            decision_score = 0.0
            hist.append(raw)

        hi = df["high"].values[-20:].astype(float)
        lo = df["low"].values[-20:].astype(float)
        cl = df["close"].values[-20:].astype(float)
        tr = np.maximum(hi[1:] - lo[1:], np.maximum(abs(hi[1:] - cl[:-1]), abs(lo[1:] - cl[:-1])))
        atr = float(np.mean(tr[:14]))
        for value in tr[14:]:
            atr = (atr * 13.0 + float(value)) / 14.0
        result = dict(selected_score)
        result.update({
            "decision_score": decision_score,
            "sequence_fingerprint": selected_fp,
            "original_sequence_fingerprint": original_fp,
            "inference_selection": dual.get("selected"),
            "dual_inference": dual,
            "feature_health": selected_health,
            "feature_health_status": selected_health.get("status", "UNKNOWN"),
            "feature_health_tradeable": not bool(selected_health.get("blocked")),
            "semantic_mode": getattr(self, "semantic_mode", None),
            "semantic_behavior": getattr(self, "semantic_behavior", None),
            "semantic_symbol_id": int(getattr(self, "active_symbol_id", -1)),
            "semantic_recovered_symbol_id": int(getattr(self, "semantic_symbol_id", -1)),
            "semantic_contract_status": getattr(self, "semantic_contract_status", None),
            "semantic_contract_claimed_verified": bool(getattr(self, "semantic_contract_claimed_verified", False)),
            "semantic_contract_verified": bool(getattr(self, "semantic_contract_verified", False)),
            "semantic_economic_ready": bool(getattr(self, "semantic_economic_ready", False)),
            "semantic_validation_errors": list(getattr(self, "semantic_validation_errors", []) or []),
            "semantic_checkpoint_sha256": getattr(self, "semantic_checkpoint_sha256", None),
            "semantic_sidecar_sha256": getattr(self, "semantic_sidecar_sha256", None),
            "atr": atr,
            "close": float(df["close"].iloc[-1]),
            "_raw_model_output_original": original_score.get("_raw_model_output"),
        })
        return result

    def prewarm_v105(self, df, max_bars=None):
        import torch
        if max_bars is None:
            max_bars = rank_window
        self._rank_prewarm_last_attempt_monotonic = time.monotonic()
        attempted = _utc_now()
        _, matrix, health = self._prepare_model_inputs(df, context="rank_prewarm_original_v105")
        repair_names, repair_families = _repair_candidates(self, health)
        mode = "ORIGINAL"
        if health.get("blocked") and repair_names:
            _, repaired, repaired_health = self._prepare_model_inputs(df, context="rank_prewarm_family_v105", force_neutralize_nonstationary=repair_names)
            if not repaired_health.get("blocked"):
                matrix, health, mode = repaired, repaired_health, "REPAIRED_FAMILY"
        if health.get("blocked"):
            self._raw_hist = deque(maxlen=rank_window)
            self._rank_prewarm_status = {"ok": False, "mode": "QUARANTINED", "at": attempted, "symbol": getattr(self, "symbol", None), "count": 0, "reasons": list(health.get("blocks") or []), "repair_candidates": repair_names, "repair_families": repair_families}
            return False
        seq = self.seq_len
        endpoints = list(range(max(seq, len(matrix) - int(max_bars)), len(matrix)))
        if not endpoints:
            self._raw_hist = deque(maxlen=rank_window)
            return False
        composites = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(endpoints), 96):
                batch_ends = endpoints[start:start + 96]
                batch = np.stack([matrix[i-seq:i] for i in batch_ends])
                tensor = torch.tensor(batch, dtype=torch.float32, device=self.device)
                out = self.model(tensor, symbol_ids=self._semantic_ids(tensor.shape[0], tensor.device))
                size = len(batch_ends)
                def flat(name, default=0.0):
                    if name not in out:
                        return np.full(size, default, dtype=float)
                    return out[name].detach().cpu().numpy().reshape(size, -1)[:, 0]
                p_l, p_s = flat("p_long_win"), flat("p_short_win")
                er_l, er_s = flat("expected_r_long"), flat("expected_r_short")
                mae_l, mae_s = flat("mae_long", 1.0), flat("mae_short", 1.0)
                router, calib, nt = flat("router_confidence"), flat("calibration_score"), flat("p_no_trade", 1.0)
                direction = flat("direction_confidence", 0.5)
                spread = flat("quantile_spread", 1.0)
                long_edge = p_l * np.maximum(er_l, 0.0) - (1.0-p_l) * np.maximum(mae_l, 1.0) * .28
                short_edge = p_s * np.maximum(er_s, 0.0) - (1.0-p_s) * np.maximum(mae_s, 1.0) * .28
                choose_long = (long_edge + (direction-.5)*.05) >= (short_edge - (direction-.5)*.05)
                edge = np.where(choose_long, long_edge, short_edge)
                spec_conf = np.ones(size)
                if "specialist_logits" in out:
                    logits = out["specialist_logits"].detach().cpu().numpy()
                    logits = logits - logits.max(axis=-1, keepdims=True)
                    probs = np.exp(logits); probs /= np.clip(probs.sum(axis=-1, keepdims=True), 1e-12, None)
                    spec_conf = probs.max(axis=-1).reshape(-1)
                spread_conf = np.clip(1.0-(spread-.5)/2.0, .1, 1.0)
                if "regime_logits" not in out or out["regime_logits"].shape[-1] != 4:
                    self._raw_hist = deque(maxlen=rank_window)
                    self._rank_prewarm_status = {"ok": False, "mode": "QUARANTINED_REGIME", "at": attempted, "count": 0, "reasons": ["regime_head_not_four_class"]}
                    return False
                logits = out["regime_logits"].detach().cpu().numpy()
                logits = logits - logits.max(axis=1, keepdims=True)
                probs = np.exp(logits); probs /= np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
                p_trend = probs[:,0] + probs[:,1]
                p_chop = probs[:,2]
                regime_mult = (0.35 + 0.65*np.clip(p_trend,0,1)) * (1.0 - 0.30*np.clip(p_chop,0,1))
                comp = np.maximum(edge,0) * np.clip(router,0,1) * np.clip(calib,0,1) * np.clip(1-nt,0,1) * np.clip(spec_conf,0,1) * spread_conf * regime_mult
                composites.extend(comp[np.isfinite(comp)].tolist())
        self._raw_hist = deque(composites, maxlen=rank_window)
        ok = len(self._raw_hist) >= rank_min
        self._rank_prewarm_status = {"ok": ok, "mode": mode, "at": attempted, "symbol": getattr(self, "symbol", None), "count": len(self._raw_hist), "reasons": [] if ok else [f"shallow_history:{len(self._raw_hist)}/{rank_min}"], "repair_candidates": repair_names, "repair_families": repair_families, "scorer": VERSION}
        return ok

    base_regime = Engine._get_regime_probs
    def regime_v105(self, sym, pred):
        probs = list((pred or {}).get("true_regime_probs") or [])
        if len(probs) != 4 or not all(math.isfinite(_num(x, float("nan"))) for x in probs):
            return {"valid": False, "state": "REGIME_UNKNOWN", "p_chop": None, "p_breakout": None, "p_panic": 0.0, "p_trend": None, "entropy": None, "panic_class_trained": False, "class_contract": "trend_up,trend_down,chop,breakout", "fallback_reason": "missing_or_invalid_four_class_posterior"}
        total = sum(float(x) for x in probs)
        if total <= 0:
            return {"valid": False, "state": "REGIME_UNKNOWN", "p_chop": None, "p_breakout": None, "p_panic": 0.0, "p_trend": None, "entropy": None, "panic_class_trained": False, "class_contract": "trend_up,trend_down,chop,breakout", "fallback_reason": "nonpositive_posterior_sum"}
        probs = [float(x)/total for x in probs]
        labels = ("TREND_UP", "TREND_DOWN", "CHOP", "BREAKOUT")
        return {"valid": True, "state": labels[max(range(4), key=probs.__getitem__)], "p_chop": probs[2], "p_breakout": probs[3], "p_panic": 0.0, "p_trend": probs[0]+probs[1], "entropy": _entropy(probs), "panic_class_trained": False, "class_contract": "trend_up,trend_down,chop,breakout", "true_probs": probs}

    base_context = Engine._trade_manager_entry_context
    def context_v105(self, sym, pos, pred, last_bar):
        base = dict(base_context(self, sym, pos, pred, last_bar) or {})
        pred = pred if isinstance(pred, dict) else {}
        pos = pos if isinstance(pos, dict) else {}
        full = {
            "behavior_contract": VERSION,
            "decision_score": pred.get("decision_score"), "threshold": pred.get("effective_threshold", pred.get("threshold")),
            "side": pred.get("side", pos.get("side")), "side_policy": pred.get("side_policy"), "expected_r_side": pred.get("expected_r_side"), "side_disagreement": pred.get("side_disagreement"),
            "expected_r_long": pred.get("expected_r_long"), "expected_r_short": pred.get("expected_r_short"),
            "p_long_win": pred.get("p_long_win"), "p_short_win": pred.get("p_short_win"),
            "mae_long": pred.get("mae_long"), "mae_short": pred.get("mae_short"),
            "long_edge": pred.get("long_edge"), "short_edge": pred.get("short_edge"), "risk_adjusted_long_edge": pred.get("risk_adjusted_long_edge"), "risk_adjusted_short_edge": pred.get("risk_adjusted_short_edge"), "edge": pred.get("edge"), "edge_margin": pred.get("edge_margin"),
            "router_confidence": pred.get("router_confidence"), "calibration_score": pred.get("calibration_score"), "p_no_trade": pred.get("p_no_trade"), "specialist_conf": pred.get("specialist_conf"), "chosen_specialist_idx": pred.get("chosen_specialist_idx"),
            "regime_valid": pred.get("regime_valid"), "regime_state": pred.get("regime_state"), "true_regime_probs": pred.get("true_regime_probs"), "rank_regime_probs": pred.get("rank_regime_probs"), "p_trend": pred.get("p_trend"), "p_chop": pred.get("p_chop"), "p_breakout": pred.get("p_breakout"), "regime_entropy": pred.get("regime_entropy"),
            "feature_health": _safe(pred.get("feature_health")), "dual_inference": _safe(pred.get("dual_inference")), "inference_selection": pred.get("inference_selection"),
            "sequence_fingerprint": pred.get("sequence_fingerprint"), "original_sequence_fingerprint": pred.get("original_sequence_fingerprint"),
            "semantic_contract_status": pred.get("semantic_contract_status"), "semantic_contract_verified": pred.get("semantic_contract_verified"), "semantic_economic_ready": pred.get("semantic_economic_ready"),
            "controller_decision": _safe(pos.get("controller_decision")), "controller_proposed_multiplier": pos.get("controller_proposed_size_multiplier"), "controller_effective_multiplier": pos.get("controller_effective_size_multiplier", pos.get("controller_size_multiplier")), "applied_multiplier": pos.get("applied_size_multiplier", pos.get("controller_size_multiplier")),
            "quantity": pos.get("qty"), "risk_usd": pos.get("risk_usd"), "margin_usd": pos.get("margin_required_est", pos.get("requested_margin_usd")), "notional_usd": pos.get("entry_notional_usd"), "leverage": pos.get("leverage"),
            "entry_bar": _safe(last_bar), "created_at": _utc_now(),
        }
        base.update(full)
        canonical = json.dumps(_safe(base), sort_keys=True, separators=(",", ":"))
        base["context_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return base

    base_arm = Engine._arm_trade_manager_position
    def arm_v105(self, sym, pos, pred, last_bar):
        if isinstance(pos, dict):
            pos["applied_size_multiplier"] = _num(pos.get("controller_size_multiplier"), 1.0)
            pos["sizing_truth"] = {
                "proposed_multiplier": pos.get("controller_proposed_size_multiplier"),
                "effective_multiplier": pos.get("controller_effective_size_multiplier", pos.get("controller_size_multiplier")),
                "applied_multiplier": pos.get("applied_size_multiplier"),
                "quantity_source": "applied_multiplier",
            }
            pos["entry_thesis_snapshot"] = context_v105(self, sym, pos, pred, last_bar)
        return base_arm(self, sym, pos, pred, last_bar)

    base_process = Engine._process
    def process_v105(self, sym, *args, **kwargs):
        if not bool(getattr(self, "paper", False)):
            return base_process(self, sym, *args, **kwargs)
        original_cap = getattr(self, "MONTHLY_TRADE_CAP", None)
        try:
            # PAPER trade count remains recorded; only the count cap is disabled.
            self.MONTHLY_TRADE_CAP = max(int(original_cap or 0), int(getattr(self, "monthly_trade_count", 0)) + 1_000_000)
            return base_process(self, sym, *args, **kwargs)
        finally:
            if original_cap is not None:
                self.MONTHLY_TRADE_CAP = original_cap

    Model._score_model_output = score_v105
    Model.infer = infer_v105
    Model.prewarm_rank_history = prewarm_v105
    Engine._get_regime_probs = regime_v105
    Engine._trade_manager_entry_context = context_v105
    Engine._arm_trade_manager_position = arm_v105
    Engine._process = process_v105

    try:
        from .rithal_control_brain_v1 import RithalControlBrainV1, ControlDecision
    except Exception:
        try:
            from rithal_control_brain_v1 import RithalControlBrainV1, ControlDecision
        except Exception:
            RithalControlBrainV1 = ControlDecision = None
    if RithalControlBrainV1 is not None and not getattr(RithalControlBrainV1, "_v105_applied", False):
        original_decide = RithalControlBrainV1.decide
        def decide_v105(self, *args, **kwargs):
            decision = original_decide(self, *args, **kwargs)
            effective = max(0.0, _num(getattr(decision, "effective_size_multiplier", getattr(decision, "size_multiplier", 1.0)), 1.0))
            decision.size_multiplier = effective
            decision.applied_size_multiplier = effective
            return decision
        RithalControlBrainV1.decide = decide_v105
        RithalControlBrainV1._v105_applied = True
        if ControlDecision is not None:
            original_to_dict = ControlDecision.to_dict
            def to_dict_v105(self):
                data = dict(original_to_dict(self))
                data["applied_size_multiplier"] = _num(getattr(self, "applied_size_multiplier", data.get("effective_size_multiplier", data.get("size_multiplier", 1.0))), 1.0)
                data["quantity_source"] = "applied_size_multiplier"
                return data
            ControlDecision.to_dict = to_dict_v105

    if log:
        log.warning("[%s] installed: risk-adjusted side, family repair, fail-closed regime, full thesis, PAPER count-cap bypass", VERSION)


def apply_trade_manager_patch(module) -> None:
    if getattr(module, "_RITHAL_BEHAVIOR_V105_TM_APPLIED", False):
        return
    TradeManager = getattr(module, "TradeManager", None)
    if TradeManager is None:
        raise RuntimeError("RITHAL_V105_TRADE_MANAGER_CLASS_MISSING")
    module._RITHAL_BEHAVIOR_V105_TM_APPLIED = True
    base_entry_context = TradeManager._entry_context
    base_review_15m = TradeManager.review_15m_context
    base_decision = TradeManager._decision
    base_observe = TradeManager.observe_mark
    base_public = TradeManager.public_trade_state
    base_status = TradeManager.status

    def entry_context_v105(self, pos):
        base = dict(base_entry_context(self, pos) or {})
        raw = pos.get("entry_thesis_snapshot") if isinstance(pos, dict) and isinstance(pos.get("entry_thesis_snapshot"), dict) else pos.get("entry_context") if isinstance(pos, dict) and isinstance(pos.get("entry_context"), dict) else {}
        allowed = {
            "behavior_contract", "decision_score", "threshold", "side", "side_policy", "expected_r_side", "side_disagreement",
            "expected_r_long", "expected_r_short", "p_long_win", "p_short_win", "mae_long", "mae_short",
            "long_edge", "short_edge", "risk_adjusted_long_edge", "risk_adjusted_short_edge", "edge", "edge_margin",
            "router_confidence", "calibration_score", "p_no_trade", "specialist_conf", "chosen_specialist_idx",
            "regime_valid", "regime_state", "true_regime_probs", "rank_regime_probs", "p_trend", "p_chop", "p_breakout", "regime_entropy",
            "feature_health", "dual_inference", "inference_selection", "sequence_fingerprint", "original_sequence_fingerprint",
            "semantic_contract_status", "semantic_contract_verified", "semantic_economic_ready", "controller_decision",
            "controller_proposed_multiplier", "controller_effective_multiplier", "applied_multiplier", "quantity", "risk_usd", "margin_usd", "notional_usd", "leverage", "entry_bar", "created_at",
        }
        for key in allowed:
            if key in raw:
                base[key] = _safe(raw[key])
        base["behavior_contract"] = VERSION
        base["context_hash"] = hashlib.sha256(json.dumps(_safe(base), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return base

    def _thesis_status(pred, entry):
        pred = pred if isinstance(pred, dict) else {}
        health = pred.get("feature_health") if isinstance(pred.get("feature_health"), dict) else {}
        if health.get("blocked") or pred.get("regime_valid") is False:
            return "UNAVAILABLE", ["model_context_unavailable_or_blocked"]
        entry_side = int(_num(entry.get("side"), 0.0))
        new_side = int(_num(pred.get("side"), 0.0))
        opposite = bool(entry_side and new_side and entry_side != new_side)
        threshold = _num(pred.get("threshold", pred.get("effective_threshold", entry.get("threshold"))), 0.0)
        score = _num(pred.get("decision_score"))
        weak = bool(score < threshold or _num(pred.get("edge_margin")) < 0.08 or _num(pred.get("router_confidence")) < 0.55 or _num(pred.get("p_no_trade"), 1.0) > 0.55)
        if opposite and weak:
            return "BROKEN", ["qualified_opposite_side", "decision_quality_deteriorated"]
        if opposite or weak:
            return "WEAKENING", ["opposite_side" if opposite else "decision_quality_deteriorated"]
        return "INTACT", ["side_and_quality_preserved"]

    def review_15m_v105(self, sym, pos, pred, last_bar):
        result = base_review_15m(self, sym, pos, pred, last_bar)
        trade_id = self._trade_id(pos)
        with self._lock:
            state = self._trades.get(trade_id)
            if isinstance(state, dict):
                entry = state.get("entry_context") if isinstance(state.get("entry_context"), dict) else {}
                status, reasons = _thesis_status(pred, entry)
                bar_id = None
                for source in (last_bar if isinstance(last_bar, dict) else {}, pred if isinstance(pred, dict) else {}):
                    for key in ("timestamp", "bar_open_ms", "open_time", "time", "datetime", "closed_15m"):
                        if source.get(key) is not None:
                            bar_id = str(source.get(key)); break
                    if bar_id:
                        break
                state["thesis_v105"] = {
                    "version": VERSION, "status": status, "reasons": reasons, "bar_id": bar_id,
                    "reviewed_at": _utc_now(), "snapshot": _safe({k: pred.get(k) for k in (
                        "side", "side_policy", "decision_score", "threshold", "expected_r_long", "expected_r_short",
                        "p_long_win", "p_short_win", "mae_long", "mae_short", "long_edge", "short_edge", "edge", "edge_margin",
                        "router_confidence", "calibration_score", "p_no_trade", "true_regime_probs", "regime_valid", "regime_state",
                        "feature_health", "dual_inference", "sequence_fingerprint")}),
                }
                state["entry_context"].setdefault("immutable_entry_snapshot", _safe(entry))
                self._persist()
                result = dict(result or {})
                result["thesis_v105"] = dict(state["thesis_v105"])
        return result

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

    def decision_v105(self, state, review, *, bar_open_ms, hard_flip=None, source="CLOSED_5M"):
        base = base_decision(self, state, review, bar_open_ms=bar_open_ms, hard_flip=hard_flip, source=source)
        block = _block(state)
        v32 = base.get("trade_manager_v3_1") if isinstance(base.get("trade_manager_v3_1"), dict) else {}
        v33 = base.get("trade_manager_v3_3") if isinstance(base.get("trade_manager_v3_3"), dict) else {}
        metrics32 = v32.get("metrics") if isinstance(v32.get("metrics"), dict) else {}
        metrics33 = v33.get("metrics") if isinstance(v33.get("metrics"), dict) else {}
        current_r = _num(metrics32.get("gross_price_r"), _num(state.get("current_r")))
        mfe_r = _num(metrics32.get("mfe_price_r"), _num(state.get("mfe_r")))
        giveback = max(0.0, _num(metrics32.get("giveback_price_r"), mfe_r-current_r))
        age = _num(v32.get("age_minutes"), _num(metrics33.get("age_minutes")))
        target_r = max(1e-9, _num(metrics33.get("target_r"), _num(state.get("target_r"), 1.0)))
        continuation = _num(v32.get("continuation_score"), _num((state.get("last_decision") or {}).get("continuation_score"), 0.5))
        review_key = str(bar_open_ms if bar_open_ms is not None else (review or {}).get("bar_open_ms") if isinstance(review, dict) else "NO_BAR")
        new_review = review_key != str(block.get("last_review_key"))
        if new_review:
            previous_mfe = block.get("last_mfe_r")
            previous_current = block.get("last_current_r")
            expanded = previous_mfe is None or mfe_r >= _num(previous_mfe) + 0.05
            block["no_new_mfe_reviews"] = 0 if expanded else int(block.get("no_new_mfe_reviews") or 0) + 1
            block["giveback_reviews"] = int(block.get("giveback_reviews") or 0) + 1 if giveback >= 0.20 else 0
            adverse = current_r < 0.0 and previous_current is not None and current_r <= _num(previous_current) - 0.05
            block["adverse_reviews"] = int(block.get("adverse_reviews") or 0) + 1 if adverse else 0
            recovered = previous_current is not None and current_r >= _num(previous_current) + 0.10
            block["recovery_fail_reviews"] = 0 if recovered or expanded else int(block.get("recovery_fail_reviews") or 0) + 1
            block["last_review_key"] = review_key
            block["last_mfe_r"] = mfe_r
            block["last_current_r"] = current_r
        profit_sentinel = v33.get("intrabar_profit_sentinel") if isinstance(v33.get("intrabar_profit_sentinel"), dict) else {}
        thesis = state.get("thesis_v105") if isinstance(state.get("thesis_v105"), dict) else {}
        lifecycle = {
            **block,
            "profit_zone_latched": bool(profit_sentinel.get("profit_zone_latched") or (state.get("trade_manager_v3_3") or {}).get("profit_zone_latched")),
            "continuation_score": continuation,
        }
        metrics = {"current_r": current_r, "mfe_r": mfe_r, "giveback_r": giveback, "age_minutes": age, "target_r": target_r, "current_progress": current_r/target_r}
        policy = _manager_policy(metrics, lifecycle, thesis.get("status", "UNAVAILABLE"), v32.get("manager_state"), v32.get("branch"))
        block["active_action"] = policy["action"]
        block["active_state"] = policy["state"]
        block["last"] = {"version": VERSION, "at": _utc_now(), "review_key": review_key, "new_review": new_review, "thesis_status": thesis.get("status", "UNAVAILABLE"), "metrics": metrics, "counters": {k: block.get(k) for k in ("no_new_mfe_reviews", "giveback_reviews", "adverse_reviews", "recovery_fail_reviews")}, "policy": policy, "actual_execution": "ROUTED_BY_EXISTING_PAPER_AUTHORITY_ONLY"}

        v33 = dict(v33)
        if policy["actionable"]:
            if policy["action"] == "P80_PARTIAL_70_PROTECTED_30":
                v33["final_manager_state"] = "PROFIT_EROSION_CONFIRMED"
                v33["profit_state"] = "PROFIT_EROSION_CONFIRMED"
                v33["final_shadow_recommendation"] = "P80_PARTIAL_70_PROTECTED_30"
                v33["profit_reason_codes"] = ["profit_zone_latched", "persistent_closed_5m_erosion", "no_new_mfe_two_reviews", "recovery_failed_two_reviews"]
            elif policy["action"] == "M70_P55_FULL_EXIT":
                v33["final_manager_state"] = "MATURE_PROFIT_EXIT_LATCHED"
                v33["profit_state"] = "MATURE_PROFIT_EXIT_LATCHED"
                v33["final_shadow_recommendation"] = "M70_P55_FULL_EXIT"
                v33["profit_reason_codes"] = ["age_gte_70", "current_progress_gte_55_percent", "no_new_mfe_three_reviews", "persistent_giveback", "thesis_or_continuation_deteriorated"]
                sentinel = dict(v33.get("intrabar_profit_sentinel") or {})
                sentinel["mature_profit_exit_latched"] = True
                sentinel["v105_persistence_confirmed"] = True
                v33["intrabar_profit_sentinel"] = sentinel
            else:
                v32 = dict(v32)
                v32["manager_state"] = "SHADOW_EXIT_CONFIRMED_ACTIVE"
                v32["branch"] = policy["branch"]
                v32["severity"] = "ACTIONABLE_SHADOW_ACTIVE"
                v32["reason_codes"] = list(v32.get("reason_codes") or []) + ["v105_persistent_evidence_confirmed"]
                base["trade_manager_v3_1"] = v32
                v33["final_manager_state"] = "SHADOW_EXIT_CONFIRMED_ACTIVE"
                v33["final_shadow_recommendation"] = "V3_2_SHADOW_EXIT_CONFIRMED"
                v33["loss_recovery_layer"] = {"version": VERSION, "manager_state": "SHADOW_EXIT_CONFIRMED_ACTIVE", "branch": policy["branch"], "severity": "ACTIONABLE_SHADOW_ACTIVE", "reason_codes": ["persistent_closed_5m_evidence", "thesis_aware_loss_confirmation"]}
        else:
            # Suppress the old age-only mature latch and first-touch P80 policy.
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
        base["plan"] = {"kind": "NONE", "paper_only": True, "requires_paper_full_exit": False}
        return base

    def observe_v105(self, sym, pos, current_price):
        output = dict(base_observe(self, sym, pos, current_price) or {})
        trade_id = self._trade_id(pos)
        with self._lock:
            state = self._trades.get(trade_id)
            block = _block(state) if isinstance(state, dict) else {}
            active = str(block.get("active_action") or "HOLD")
            detail = dict(block.get("last") or {})
        v33 = output.get("trade_manager_v3_3") if isinstance(output.get("trade_manager_v3_3"), dict) else {}
        v33 = dict(v33)
        if active == "M70_P55_FULL_EXIT":
            v33.update({"final_manager_state": "MATURE_PROFIT_EXIT_LATCHED", "profit_state": "MATURE_PROFIT_EXIT_LATCHED", "final_shadow_recommendation": active})
        elif active == "P80_PARTIAL_70_PROTECTED_30":
            v33.update({"final_manager_state": "PROFIT_EROSION_CONFIRMED", "profit_state": "PROFIT_EROSION_CONFIRMED", "final_shadow_recommendation": active})
        elif active == "V3_2_SHADOW_EXIT_CONFIRMED":
            v33.update({"final_manager_state": "SHADOW_EXIT_CONFIRMED_ACTIVE", "final_shadow_recommendation": active})
        else:
            v33.update({"final_manager_state": "HOLD_PERSISTENCE_REQUIRED", "profit_state": "HOLD_PERSISTENCE_REQUIRED", "final_shadow_recommendation": "HOLD", "authority_suppressed_by_v105": True})
        v33["behavior_v105"] = detail
        output["trade_manager_v3_3"] = v33
        output["trade_manager_v105"] = detail
        output["unified_manager_state"] = v33.get("final_manager_state")
        output["unified_shadow_recommendation"] = v33.get("final_shadow_recommendation")
        output["selected_action"] = "HOLD"
        output["authority"] = "SHADOW"
        return output

    def public_v105(self, trade_id):
        output = dict(base_public(self, trade_id) or {})
        with self._lock:
            state = self._trades.get(str(trade_id))
            if isinstance(state, dict):
                output["trade_manager_v105"] = dict(_block(state).get("last") or {})
                output["thesis_v105"] = dict(state.get("thesis_v105") or {})
        return output

    def status_v105(self):
        output = dict(base_status(self) or {})
        output["behavior_v105"] = {"version": VERSION, "single_canonical_manager": True, "closed_5m_persistence_required": True, "age_only_mature_exit_disabled": True, "first_touch_p80_execution_disabled": True, "thesis_aware_loss_confirmation": True, "paper_authority_adapter_reused": True, "live_exchange_control": False}
        return output

    TradeManager._entry_context = entry_context_v105
    TradeManager.review_15m_context = review_15m_v105
    TradeManager._decision = decision_v105
    TradeManager.observe_mark = observe_v105
    TradeManager.public_trade_state = public_v105
    TradeManager.status = status_v105


def configure_project(project_root: Path, instance_id: str = INSTANCE_ID, manager_mode: str = "PAPER_CONTROL") -> dict:
    root = Path(project_root).resolve()
    neural = root / "mythos" / "neural"
    if not neural.is_dir():
        raise RuntimeError(f"invalid project root: {root}")
    sys.path.insert(0, str(neural))
    from rithal_runtime_settings import load_settings, save_settings, preflight_settings

    before = load_settings(root, instance_id)
    candidate = copy.deepcopy(before)
    candidate.setdefault("capital", {}).update({
        "starting_equity": 18000.0,
        "per_trade_margin_pct": 0.10,
        "leverage": 10,
        "max_portfolio_margin_pct": 0.60,
        "max_portfolio_heat_pct": 0.03,
        "entry_fee_bps": 4.0,
        "exit_fee_bps": 4.0,
        "reset_equity_on_next_start": False,
    })
    candidate.setdefault("runtime", {})["new_entries_enabled"] = True
    candidate.setdefault("trade_manager", {}).update({
        "mode": str(manager_mode).upper(),
        "apply_to_existing_positions": True,
        "loss_protection_enabled": True,
        "early_failure_exit_enabled": True,
        "extreme_reversal_exit_enabled": True,
        "mid_mfe_reversal_exit_enabled": True,
        "mature_profit_exit_enabled": True,
        "p80_partial_harvest_enabled": True,
        "hard_profit_cap_enabled": False,
        "parity_lock_required": True,
    })
    candidate.setdefault("safety", {}).update({
        "paper_only": True,
        "live_exchange_control": False,
        "remote_write_blocked": True,
        "authority_bridge_fail_closed": True,
        "manager_can_open_positions": False,
        "manager_can_reverse_positions": False,
        "manager_can_resize_positions": True,
        "manager_resize_scope": "P80_REDUCE_ONLY_70_PERCENT",
        "model_scoring_editable": False,
        "combined_partial_runner_accounting": True,
    })
    report = preflight_settings(before, candidate, instance_id=instance_id)
    validated = report.get("candidate") if isinstance(report, dict) and isinstance(report.get("candidate"), dict) else candidate
    saved, save_report = save_settings(root, instance_id, validated, actor=VERSION, remote_addr="127.0.0.1", expected_revision=int(before.get("revision") or 0))

    control_path = root / "mythos_5m_execution_control.json"
    control = {}
    if control_path.is_file():
        try:
            control = json.loads(control_path.read_text(encoding="utf-8-sig"))
        except Exception:
            control = {}
    control.update({"mode": "PAPER_AUTOMANAGE", "execution_enabled": True, "scope": "PAPER_ONLY", "updated_at": _utc_now(), "updated_by": VERSION})
    temp = control_path.with_suffix(".tmp")
    temp.write_text(json.dumps(control, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, control_path)

    policy_path = neural / "mythos_paper_policy.json"
    policy = {"starting_equity": 18000.0, "per_trade_margin_pct": 0.10, "leverage": 10, "leverage_mode": "fixed", "max_portfolio_margin_pct": 0.60}
    temp = policy_path.with_suffix(".tmp")
    temp.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, policy_path)

    instance_dir = root / "mythos_model_instances" / instance_id
    controller_path = instance_dir / "rithal_controller_config.json"
    if controller_path.is_file():
        try:
            controller = json.loads(controller_path.read_text(encoding="utf-8-sig"))
        except Exception:
            controller = {}
        controller.update({"audit_only": True, "adaptive_sizing_enabled": False, "micro_live_enabled": False, "full_margin_multiplier": 1.0, "half_margin_multiplier": 0.5, "safety": {**dict(controller.get("safety") or {}), "paper_only": True, "never_touch_model_files": True, "fail_open_to_original_engine_when_controller_errors": False}})
        temp = controller_path.with_suffix(".tmp")
        temp.write_text(json.dumps(controller, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, controller_path)

    return {"version": VERSION, "instance_id": instance_id, "settings_saved": saved, "save_report": save_report, "control": control, "paper_policy": policy, "restart_required": True, "live_exchange_control": False}


def self_test() -> dict:
    risk_side = score_values({"p_long_win": .52, "p_short_win": .70, "expected_r_long": 2.0, "expected_r_short": 1.8, "mae_long": 4.0, "mae_short": 1.0, "calibration_score": .9, "router_confidence": .9, "p_no_trade": .05, "direction_confidence": .5, "quantile_spread": 1.0, "specialist_conf": .9, "regime_logits": [1.0, 2.0, .2, .1]})
    assert risk_side["expected_r_side"] == 1 and risk_side["side"] == -1
    invalid = score_values({"p_long_win": .9, "p_short_win": .1, "expected_r_long": 3, "expected_r_short": 0, "regime_logits": [1, 2, 3]})
    assert invalid["regime_valid"] is False and invalid["raw_composite"] == 0.0
    expanded, families = expand_repair_families(["open_interest"], ["open_interest", "oi_delta", "oi_z", "ret_1"])
    assert expanded == ["oi_delta", "oi_z", "open_interest"] and families == ["OPEN_INTEREST"]
    p80 = _manager_policy({"current_r": .45, "mfe_r": .85, "giveback_r": .40, "age_minutes": 60, "target_r": 1.0}, {"profit_zone_latched": True, "no_new_mfe_reviews": 2, "giveback_reviews": 2, "recovery_fail_reviews": 2, "adverse_reviews": 0}, "INTACT", "HOLD", "NONE")
    assert p80["action"] == "P80_PARTIAL_70_PROTECTED_30"
    first_touch = _manager_policy({"current_r": .80, "mfe_r": .80, "giveback_r": 0, "age_minutes": 30, "target_r": 1.0}, {"profit_zone_latched": True, "no_new_mfe_reviews": 0, "giveback_reviews": 0, "recovery_fail_reviews": 0, "adverse_reviews": 0}, "INTACT", "HOLD", "NONE")
    assert first_touch["action"] == "HOLD"
    mature = _manager_policy({"current_r": .60, "mfe_r": .90, "giveback_r": .30, "age_minutes": 80, "target_r": 1.0}, {"no_new_mfe_reviews": 3, "giveback_reviews": 2, "recovery_fail_reviews": 2, "adverse_reviews": 0, "continuation_score": .4}, "WEAKENING", "HOLD", "NONE")
    assert mature["action"] == "M70_P55_FULL_EXIT"
    age_only = _manager_policy({"current_r": .60, "mfe_r": .60, "giveback_r": 0, "age_minutes": 80, "target_r": 1.0}, {"no_new_mfe_reviews": 0, "giveback_reviews": 0, "recovery_fail_reviews": 0, "adverse_reviews": 0, "continuation_score": .8}, "INTACT", "HOLD", "NONE")
    assert age_only["action"] == "HOLD"
    return {"version": VERSION, "status": "PASS", "checks": ["risk_adjusted_side", "invalid_regime_fail_closed", "family_repair_expansion", "p80_persistence", "first_touch_suppressed", "mature_stagnation", "age_only_suppressed"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--configure", action="store_true")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--instance-id", default=INSTANCE_ID)
    parser.add_argument("--manager-mode", choices=("SHADOW_ONLY", "PAPER_CONTROL"), default="PAPER_CONTROL")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
    if args.configure:
        print(json.dumps(configure_project(Path(args.project_root), args.instance_id, args.manager_mode), indent=2, sort_keys=True, default=str))
    if not args.self_test and not args.configure:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
