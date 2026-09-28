from __future__ import annotations

"""Rithal V1.0.4.7 canonical layer-truth and counterfactual runtime.

This sidecar is deliberately non-executing. It does not change model weights,
thresholds, TP/SL geometry, orders, or exchange authority. It reads the state
already produced by Rithal, resolves contradictory layer outputs into one
canonical truth record, and writes shadow-only counterfactual diagnostics.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

VERSION = "RITHAL_LAYER_TRUTH_V1_0_4_7"
SCHEMA_VERSION = 1
EXPECTED_PROFILE = {
    "instance_id": "rithal-1-0-contract-locked",
    "paper_equity_usd": 18000.0,
    "fixed_margin_pct": 0.10,
    "leverage": 10.0,
    "allocation_cap_pct": 0.60,
    "heat_cap_pct": 0.03,
    "max_open_positions": 2,
    "manager_mode": "SHADOW_ONLY",
}
SOURCE_PRIORITY = {
    "RITHAL_TM_V3_3": 400,
    "RITHAL_TM_V3_2": 300,
    "RITHAL_EXIT_BRAIN": 200,
    "LEGACY_MANAGER": 100,
    "UNAVAILABLE": 0,
}
ACTIONABLE_EXIT_STATES = {
    "SHADOW_EXIT_CONFIRMED",
    "SHADOW_EXIT_CONFIRMED_ACTIVE",
    "MATURE_PROFIT_EXIT_LATCHED",
    "P80_PARTIAL_70_PROTECTED_30",
    "CLOSE_FULL",
    "CLOSE_PARTIAL",
    "EXIT",
}
STATE_FILES = {
    "positions": "mythos_open_positions.json",
    "manager": "mythos_trade_manager_state.json",
    "runtime": "mythos_runtime_state.json",
    "account": "mythos_account_state.json",
    "interactions": "mythos_model_interactions.json",
    "prices": "mythos_live_prices.json",
    "control": "mythos_5m_execution_control.json",
    "settings": "settings_override_v1.json",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    number = safe_float(value)
    return int(number) if number is not None else None


def parse_time(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        raw = float(value)
        if raw > 10_000_000_000:
            raw /= 1000.0
        try:
            return datetime.fromtimestamp(raw, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def read_json(path: Path, retries: int = 3) -> Any:
    if not path.exists():
        return None
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            text = path.read_text(encoding="utf-8-sig")
            if not text.strip():
                return None
            return json.loads(text)
        except (PermissionError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    return {"_read_error": f"{type(last_error).__name__}: {last_error}"}


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    temp.write_text(text + "\n", encoding="utf-8")
    os.replace(temp, path)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def iter_dicts(value: Any, path: str = "$") -> Iterable[Tuple[str, Dict[str, Any]]]:
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            yield from iter_dicts(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_dicts(child, f"{path}[{index}]")


def first_value(mapping: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def recursive_first(value: Any, keys: Iterable[str]) -> Any:
    wanted = tuple(keys)
    for _, mapping in iter_dicts(value):
        found = first_value(mapping, wanted)
        if found not in (None, ""):
            return found
    return None


def normalize_symbol(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).upper().strip()
    return text if text.endswith("USDT") else None


def extract_symbol(mapping: Dict[str, Any]) -> Optional[str]:
    return normalize_symbol(first_value(mapping, ("symbol", "asset", "ticker")))


def extract_trade_id(mapping: Dict[str, Any]) -> Optional[str]:
    value = first_value(
        mapping,
        (
            "trade_id",
            "baseline_trade",
            "baseline_trade_id",
            "position_id",
            "ticket_trade_id",
            "core_trade_id",
        ),
    )
    if value is None:
        nested = mapping.get("ticket")
        if isinstance(nested, dict):
            value = first_value(nested, ("trade_id", "baseline_trade", "baseline_trade_id"))
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def extract_ticket_id(mapping: Dict[str, Any]) -> Optional[str]:
    value = first_value(mapping, ("ticket_id", "ticket", "manager_ticket_id", "five_minute_ticket_id"))
    if isinstance(value, dict):
        value = first_value(value, ("id", "ticket_id"))
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def extract_timestamp(mapping: Dict[str, Any]) -> Optional[datetime]:
    value = first_value(
        mapping,
        (
            "reviewed_at",
            "review_time",
            "updated_at",
            "at",
            "timestamp",
            "closed_5m",
            "decision_at",
            "created_at",
            "entry_time",
        ),
    )
    return parse_time(value)


def text_blob(mapping: Dict[str, Any]) -> str:
    try:
        return json.dumps(mapping, sort_keys=True, default=str).upper()
    except Exception:
        return str(mapping).upper()


def source_name(mapping: Dict[str, Any], path: str = "") -> str:
    blob = f"{path} {text_blob(mapping)}"
    if "RITHAL_TM_V3_3" in blob or "PROFIT_STATE" in blob or "P80_PARTIAL" in blob or "M70_P55" in blob:
        return "RITHAL_TM_V3_3"
    if "RITHAL_TM_V3_2" in blob or "LOSS_RECOVERY_LAYER" in blob or "MID_MFE_REVERSAL" in blob or "EARLY_FAILURE" in blob:
        return "RITHAL_TM_V3_2"
    if "RITHAL_EXIT_BRAIN" in blob or "EXIT_BRAIN" in blob:
        return "RITHAL_EXIT_BRAIN"
    if "MANAGER" in blob or any(key in mapping for key in ("manager_action", "manager_state", "recommended_action")):
        return "LEGACY_MANAGER"
    return "UNAVAILABLE"


def extract_state(mapping: Dict[str, Any]) -> Optional[str]:
    value = first_value(
        mapping,
        (
            "state",
            "manager_state",
            "profit_state",
            "candidate_state",
            "manager_candidate_state",
            "action",
            "recommendation",
            "rec",
        ),
    )
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def recommended_action(state: Optional[str], mapping: Dict[str, Any]) -> str:
    explicit = first_value(mapping, ("recommended_action", "recommendation", "rec"))
    combined = " ".join(str(x).upper() for x in (state, explicit) if x not in (None, ""))
    if "P80_PARTIAL" in combined or "CLOSE_PARTIAL" in combined or "PARTIAL" in combined:
        return "CLOSE_PARTIAL"
    if any(token in combined for token in ("SHADOW_EXIT_CONFIRMED", "MATURE_PROFIT_EXIT", "CLOSE_FULL", "FULL_EXIT")):
        return "CLOSE_FULL"
    if "WATCH" in combined or "WARNING" in combined:
        return "WATCH"
    if "DATA_GATED" in combined or "UNAVAILABLE" in combined:
        return "HOLD_DATA_GATED"
    if "HOLD" in combined or not combined:
        return "HOLD"
    return combined.split()[0]


def control_mode(control: Any, settings: Any) -> str:
    for payload in (control, settings):
        raw = recursive_first(payload, ("manager_mode", "mode", "authority_mode", "trade_manager_mode"))
        if raw is not None:
            text = str(raw).strip().upper()
            if text in {"OFF", "SHADOW_ONLY", "PAPER_CONTROL", "FULL_CONTROL"}:
                return text
    return "UNAVAILABLE"


def actual_execution(mapping: Dict[str, Any]) -> str:
    raw = first_value(mapping, ("actual_execution", "execution", "actual", "execution_result"))
    if raw is None:
        return "UNAVAILABLE"
    text = str(raw).strip().upper()
    return text or "UNAVAILABLE"


def choose_manager_record(
    manager_state: Any,
    position: Dict[str, Any],
    stale_after_seconds: int,
) -> Dict[str, Any]:
    trade_id = extract_trade_id(position)
    ticket_id = extract_ticket_id(position)
    symbol = extract_symbol(position)
    candidates: List[Dict[str, Any]] = []
    for path, mapping in iter_dicts(manager_state):
        src = source_name(mapping, path)
        if src == "UNAVAILABLE":
            continue
        cand_trade_id = extract_trade_id(mapping)
        cand_ticket_id = extract_ticket_id(mapping)
        cand_symbol = extract_symbol(mapping)
        exact_trade = bool(trade_id and cand_trade_id and trade_id == cand_trade_id)
        exact_ticket = bool(ticket_id and cand_ticket_id and ticket_id == cand_ticket_id)
        mismatch = bool(trade_id and cand_trade_id and trade_id != cand_trade_id)
        if mismatch:
            continue
        ts = extract_timestamp(mapping)
        candidates.append(
            {
                "path": path,
                "record": mapping,
                "source": src,
                "priority": SOURCE_PRIORITY[src],
                "exact_trade": exact_trade,
                "exact_ticket": exact_ticket,
                "symbol_match_only": bool(symbol and cand_symbol and symbol == cand_symbol and not exact_trade and not exact_ticket),
                "timestamp": ts,
            }
        )
    candidates.sort(
        key=lambda item: (
            1 if item["exact_trade"] else 0,
            1 if item["exact_ticket"] else 0,
            item["priority"],
            item["timestamp"].timestamp() if item["timestamp"] else 0.0,
        ),
        reverse=True,
    )
    chosen = candidates[0] if candidates else None
    if chosen is None:
        return {
            "binding": "UNAVAILABLE",
            "source": "UNAVAILABLE",
            "advisory_state": "UNAVAILABLE",
            "recommended_action": "UNAVAILABLE",
            "effective_authority": "UNAVAILABLE",
            "actual_execution": "UNAVAILABLE",
            "reviewed_at": None,
            "review_age_seconds": None,
            "source_path": None,
            "reason": "no_manager_record_found",
        }
    timestamp = chosen["timestamp"]
    age_seconds = max(0.0, (utc_now() - timestamp).total_seconds()) if timestamp else None
    if chosen["exact_trade"] or chosen["exact_ticket"]:
        binding = "LINKED"
    else:
        binding = "UNBOUND"
    if age_seconds is not None and age_seconds > stale_after_seconds:
        binding = "STALE"
    record = chosen["record"]
    state = extract_state(record) or "UNAVAILABLE"
    return {
        "binding": binding,
        "source": chosen["source"],
        "advisory_state": state,
        "recommended_action": recommended_action(state, record),
        "effective_authority": "UNAVAILABLE",
        "actual_execution": actual_execution(record),
        "reviewed_at": timestamp.isoformat() if timestamp else None,
        "review_age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
        "source_path": chosen["path"],
        "reason": first_value(record, ("reason", "reason_codes", "branch", "severity")),
        "raw_trade_id": extract_trade_id(record),
        "raw_ticket_id": extract_ticket_id(record),
        "symbol_match_only": chosen["symbol_match_only"],
    }


def normalize_probs(raw: Any) -> Optional[List[float]]:
    if isinstance(raw, dict):
        ordered = []
        for key in ("trend_up", "trend_down", "chop", "breakout"):
            value = safe_float(raw.get(key))
            if value is None:
                return None
            ordered.append(value)
        raw = ordered
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    values = [safe_float(value) for value in raw[:4]]
    if any(value is None for value in values):
        return None
    total = sum(values)  # type: ignore[arg-type]
    if total <= 0:
        return None
    return [float(value) / total for value in values]  # type: ignore[arg-type]


def entropy(probs: Optional[List[float]]) -> Optional[float]:
    if not probs:
        return None
    return -sum(p * math.log(max(p, 1e-12)) for p in probs)


def regime_truth(payload: Any) -> Dict[str, Any]:
    true_raw = recursive_first(payload, ("true_regime_probs", "regime_probs", "regime_posterior"))
    rank_raw = recursive_first(payload, ("rank_regime_probs", "ranking_regime_probs"))
    true_probs = normalize_probs(true_raw)
    rank_probs = normalize_probs(rank_raw)
    if true_probs is None:
        named = {
            "trend_up": safe_float(recursive_first(payload, ("p_trend_up",))),
            "trend_down": safe_float(recursive_first(payload, ("p_trend_down",))),
            "chop": safe_float(recursive_first(payload, ("p_chop",))),
            "breakout": safe_float(recursive_first(payload, ("p_breakout",))),
        }
        if all(value is not None for value in named.values()):
            true_probs = normalize_probs(list(named.values()))
    if true_probs is None:
        return {
            "valid": False,
            "state": "REGIME_UNKNOWN",
            "fallback_reason": "missing_or_invalid_regime_output",
            "class_order": ["TREND_UP", "TREND_DOWN", "CHOP", "BREAKOUT"],
            "true_probs": None,
            "rank_probs": rank_probs,
            "p_trend": None,
            "p_chop": None,
            "p_breakout": None,
            "entropy": None,
        }
    labels = ["TREND_UP", "TREND_DOWN", "CHOP", "BREAKOUT"]
    state = labels[max(range(4), key=true_probs.__getitem__)]
    return {
        "valid": True,
        "state": state,
        "fallback_reason": None,
        "class_order": labels,
        "true_probs": [round(value, 8) for value in true_probs],
        "rank_probs": [round(value, 8) for value in rank_probs] if rank_probs else None,
        "p_trend": round(true_probs[0] + true_probs[1], 8),
        "p_chop": round(true_probs[2], 8),
        "p_breakout": round(true_probs[3], 8),
        "entropy": round(entropy(true_probs) or 0.0, 8),
    }


def extract_metrics(payload: Any) -> Dict[str, Optional[float]]:
    aliases = {
        "gross_r": ("gross_r", "current_r", "unrealized_r"),
        "net_r": ("net_r", "net_est_r", "estimated_net_r"),
        "mfe_r": ("mfe", "mfe_r", "max_favorable_r"),
        "mae_r": ("mae", "mae_r", "max_adverse_r"),
        "giveback_r": ("giveback", "giveback_r"),
        "age_minutes": ("age_minutes", "age_min", "duration_min", "age"),
        "current_progress_pct": ("current_progress", "current_progress_pct"),
        "mfe_progress_pct": ("mfe_progress", "mfe_progress_pct"),
    }
    result: Dict[str, Optional[float]] = {}
    for name, keys in aliases.items():
        result[name] = safe_float(recursive_first(payload, keys))
    return result


def side_shadow(payload: Any) -> Dict[str, Any]:
    canonical = recursive_first(payload, ("side", "canonical_side", "selected_side"))
    canonical_text = str(canonical).upper() if canonical is not None else "UNAVAILABLE"
    long_edge = safe_float(recursive_first(payload, ("long_edge", "risk_adjusted_long_edge")))
    short_edge = safe_float(recursive_first(payload, ("short_edge", "risk_adjusted_short_edge")))
    if long_edge is None or short_edge is None:
        shadow = "UNAVAILABLE"
    else:
        shadow = "LONG" if long_edge >= short_edge else "SHORT"
    return {
        "canonical_side": canonical_text,
        "risk_adjusted_shadow_side": shadow,
        "long_edge": long_edge,
        "short_edge": short_edge,
        "agreement": None if "UNAVAILABLE" in (canonical_text, shadow) else canonical_text == shadow,
        "execution_changed": False,
        "mode": "SHADOW_ONLY",
    }


def sizing_truth(position: Dict[str, Any]) -> Dict[str, Any]:
    controller = recursive_first(position, ("controller_decision",))
    if not isinstance(controller, dict):
        controller = position.get("controller") if isinstance(position.get("controller"), dict) else {}
    proposed = safe_float(first_value(controller, ("proposed_size_multiplier", "proposed_multiplier")))
    effective = safe_float(first_value(controller, ("effective_size_multiplier", "effective_multiplier", "size_multiplier")))
    applied = safe_float(first_value(position, ("applied_size_multiplier", "lane_mult", "controller_size_multiplier")))
    base_margin = safe_float(first_value(position, ("base_margin", "base_margin_usd")))
    requested_margin = safe_float(first_value(position, ("requested_margin", "margin", "margin_usd")))
    if applied is None and base_margin and requested_margin is not None and base_margin > 0:
        applied = requested_margin / base_margin
    contradictions: List[str] = []
    if effective is not None and applied is not None and abs(effective - applied) > 1e-9:
        contradictions.append("effective_multiplier_differs_from_applied_multiplier")
    if proposed is not None and effective is None:
        contradictions.append("proposal_present_but_effective_missing")
    return {
        "proposed_multiplier": proposed,
        "effective_multiplier": effective,
        "applied_multiplier": applied,
        "base_margin_usd": base_margin,
        "requested_margin_usd": requested_margin,
        "contradictions": contradictions,
        "canonical_quantity_source": "applied_multiplier",
    }


def profile_value(payloads: Iterable[Any], aliases: Iterable[str]) -> Any:
    for payload in payloads:
        value = recursive_first(payload, aliases)
        if value not in (None, ""):
            return value
    return None


def profile_truth(settings: Any, runtime: Any, account: Any, control: Any) -> Dict[str, Any]:
    payloads = (settings, runtime, account, control)
    actual = {
        "instance_id": profile_value(payloads, ("model_instance_id", "instance_id")),
        "paper_equity_usd": safe_float(profile_value(payloads, ("paper_equity", "paper_equity_usd", "paper_amount", "equity"))),
        "fixed_margin_pct": safe_float(profile_value(payloads, ("fixed_margin_pct", "margin_pct", "paper_fixed_margin_pct"))),
        "leverage": safe_float(profile_value(payloads, ("leverage", "live_leverage", "paper_leverage"))),
        "allocation_cap_pct": safe_float(profile_value(payloads, ("allocation_cap", "allocation_cap_pct"))),
        "heat_cap_pct": safe_float(profile_value(payloads, ("heat_cap", "heat_cap_pct", "portfolio_heat_cap"))),
        "max_open_positions": safe_int(profile_value(payloads, ("max_open_positions", "max_positions"))),
        "manager_mode": control_mode(control, settings),
    }
    mismatches: List[str] = []
    tolerances = {"paper_equity_usd": 0.01, "fixed_margin_pct": 1e-9, "leverage": 1e-9, "allocation_cap_pct": 1e-9, "heat_cap_pct": 1e-9}
    for key, expected in EXPECTED_PROFILE.items():
        value = actual.get(key)
        if value is None or value == "UNAVAILABLE":
            mismatches.append(f"{key}:UNAVAILABLE")
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if abs(float(value) - float(expected)) > tolerances.get(key, 0.0):
                mismatches.append(f"{key}:{value}!={expected}")
        elif str(value).lower() != str(expected).lower():
            mismatches.append(f"{key}:{value}!={expected}")
    return {"expected": EXPECTED_PROFILE, "actual": actual, "mismatches": mismatches, "aligned": not mismatches}


@dataclass
class ShadowLifecycleState:
    last_mfe: Optional[float] = None
    no_new_mfe_reviews: int = 0
    giveback_reviews: int = 0


def shadow_policy(trade_id: str, metrics: Dict[str, Optional[float]], state: ShadowLifecycleState) -> Dict[str, Any]:
    gross = metrics.get("gross_r")
    mfe = metrics.get("mfe_r")
    giveback = metrics.get("giveback_r")
    age = metrics.get("age_minutes")
    current_progress = metrics.get("current_progress_pct")
    mfe_progress = metrics.get("mfe_progress_pct")
    if mfe is not None:
        if state.last_mfe is None or mfe > state.last_mfe + 0.02:
            state.no_new_mfe_reviews = 0
            state.last_mfe = mfe
        else:
            state.no_new_mfe_reviews += 1
    if giveback is not None and giveback >= 0.35:
        state.giveback_reviews += 1
    else:
        state.giveback_reviews = 0
    earlier_loss = bool(
        mfe is not None
        and gross is not None
        and giveback is not None
        and mfe >= 0.45
        and gross <= -0.15
        and giveback >= 0.65
        and state.giveback_reviews >= 2
    )
    mature_exit = bool(
        age is not None
        and current_progress is not None
        and age >= 70.0
        and current_progress >= 55.0
        and state.no_new_mfe_reviews >= 3
        and state.giveback_reviews >= 2
    )
    p80_partial = bool(
        mfe_progress is not None
        and mfe_progress >= 80.0
        and state.no_new_mfe_reviews >= 2
        and state.giveback_reviews >= 2
    )
    if earlier_loss:
        action = "SHADOW_CLOSE_FULL_EARLIER_LOSS"
    elif mature_exit:
        action = "SHADOW_CLOSE_FULL_MATURE_STAGNATION"
    elif p80_partial:
        action = "SHADOW_CLOSE_PARTIAL_P80_PERSISTENT_EROSION"
    else:
        action = "SHADOW_HOLD"
    return {
        "trade_id": trade_id,
        "action": action,
        "execution_changed": False,
        "mode": "SHADOW_ONLY",
        "no_new_mfe_reviews": state.no_new_mfe_reviews,
        "persistent_giveback_reviews": state.giveback_reviews,
        "conditions": {
            "earlier_loss": earlier_loss,
            "mature_exit_with_stagnation": mature_exit,
            "p80_partial_with_persistent_erosion": p80_partial,
        },
    }


def list_positions(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("positions", "open_positions", "items", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            records = []
            for symbol, item in value.items():
                if isinstance(item, dict):
                    copied = dict(item)
                    copied.setdefault("symbol", symbol)
                    records.append(copied)
            if records:
                return records
    records = []
    for symbol, item in payload.items():
        if isinstance(item, dict) and normalize_symbol(symbol):
            copied = dict(item)
            copied.setdefault("symbol", symbol)
            records.append(copied)
    return records


def exact_thesis(interactions: Any, trade_id: Optional[str], ticket_id: Optional[str]) -> Dict[str, Any]:
    matches: List[Tuple[Optional[datetime], Dict[str, Any], str]] = []
    for path, mapping in iter_dicts(interactions):
        mapped_trade = extract_trade_id(mapping)
        mapped_ticket = extract_ticket_id(mapping)
        if trade_id and mapped_trade == trade_id:
            matches.append((extract_timestamp(mapping), mapping, path))
        elif ticket_id and mapped_ticket == ticket_id:
            matches.append((extract_timestamp(mapping), mapping, path))
    if not matches:
        return {"binding": "UNAVAILABLE", "source_path": None, "snapshot": None}
    matches.sort(key=lambda item: item[0].timestamp() if item[0] else 0.0, reverse=True)
    _, mapping, path = matches[0]
    keys = (
        "decision_score",
        "threshold",
        "expected_r_long",
        "expected_r_short",
        "p_win_long",
        "p_win_short",
        "mae_long",
        "mae_short",
        "long_edge",
        "short_edge",
        "edge_margin",
        "router_confidence",
        "calibration_score",
        "specialist_confidence",
        "specialist",
        "feature_health",
        "model_input_health",
        "repaired_features",
        "ood_features",
        "raw_composite",
        "sequence_fingerprint",
    )
    snapshot = {key: recursive_first(mapping, (key,)) for key in keys}
    snapshot["regime"] = regime_truth(mapping)
    snapshot["side_shadow"] = side_shadow(mapping)
    return {"binding": "LINKED", "source_path": path, "snapshot": snapshot}


class LayerTruthRuntime:
    def __init__(self, root: Path, stale_after_seconds: int = 650):
        self.root = root.resolve()
        self.stale_after_seconds = stale_after_seconds
        self.output = self.root / "mythos_rithal_intelligence_truth.json"
        self.counterfactual = self.root / "mythos_rithal_counterfactual.jsonl"
        self.health = self.root / "mythos_rithal_layer_health.json"
        self._last_digest: Optional[str] = None
        self._shadow_states: Dict[str, ShadowLifecycleState] = {}

    def load(self) -> Dict[str, Any]:
        return {name: read_json(self.root / filename) for name, filename in STATE_FILES.items()}

    def build(self) -> Dict[str, Any]:
        data = self.load()
        positions = list_positions(data["positions"])
        mode = control_mode(data["control"], data["settings"])
        canonical_positions: List[Dict[str, Any]] = []
        for position in positions:
            symbol = extract_symbol(position) or "UNAVAILABLE"
            trade_id = extract_trade_id(position)
            ticket_id = extract_ticket_id(position)
            manager = choose_manager_record(data["manager"], position, self.stale_after_seconds)
            manager["effective_authority"] = mode
            if manager["actual_execution"] == "UNAVAILABLE":
                manager["actual_execution"] = "UNCHANGED_SHADOW_ONLY" if mode == "SHADOW_ONLY" else "UNAVAILABLE"
            thesis = exact_thesis(data["interactions"], trade_id, ticket_id)
            metrics = extract_metrics(position)
            if manager.get("source_path"):
                for path, mapping in iter_dicts(data["manager"]):
                    if path == manager["source_path"]:
                        manager_metrics = extract_metrics(mapping)
                        metrics = {key: manager_metrics.get(key) if manager_metrics.get(key) is not None else value for key, value in metrics.items()}
                        break
            state_key = trade_id or ticket_id or f"UNBOUND:{symbol}"
            lifecycle_state = self._shadow_states.setdefault(state_key, ShadowLifecycleState())
            shadow = shadow_policy(state_key, metrics, lifecycle_state)
            canonical_positions.append(
                {
                    "symbol": symbol,
                    "trade_id": trade_id,
                    "ticket_id": ticket_id,
                    "binding": manager["binding"],
                    "manager": manager,
                    "metrics": metrics,
                    "entry_thesis": thesis,
                    "regime": regime_truth(thesis.get("snapshot") if isinstance(thesis, dict) else position),
                    "side_comparison": side_shadow(thesis.get("snapshot") if isinstance(thesis, dict) and thesis.get("snapshot") else position),
                    "sizing": sizing_truth(position),
                    "shadow_manager_v1047": shadow,
                }
            )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "version": VERSION,
            "generated_at": iso_now(),
            "execution_policy": {
                "model_weights_changed": False,
                "thresholds_changed": False,
                "tp_sl_changed": False,
                "exchange_authority_changed": False,
                "behavioral_improvements": "SHADOW_ONLY",
            },
            "profile": profile_truth(data["settings"], data["runtime"], data["account"], data["control"]),
            "positions": canonical_positions,
            "source_health": {
                name: {
                    "path": str(self.root / filename),
                    "available": (self.root / filename).exists(),
                    "read_error": value.get("_read_error") if isinstance(value, dict) else None,
                }
                for (name, filename), value in zip(STATE_FILES.items(), data.values())
            },
        }
        return payload

    def publish(self) -> Dict[str, Any]:
        payload = self.build()
        canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        payload["content_sha256"] = digest
        atomic_write_json(self.output, payload)
        health = {
            "version": VERSION,
            "generated_at": payload["generated_at"],
            "status": "PASS" if not payload["profile"]["mismatches"] else "WARN",
            "profile_aligned": payload["profile"]["aligned"],
            "profile_mismatches": payload["profile"]["mismatches"],
            "open_positions": len(payload["positions"]),
            "linked_positions": sum(1 for item in payload["positions"] if item["binding"] == "LINKED"),
            "stale_positions": sum(1 for item in payload["positions"] if item["binding"] == "STALE"),
            "unbound_positions": sum(1 for item in payload["positions"] if item["binding"] == "UNBOUND"),
            "truth_file": str(self.output),
        }
        atomic_write_json(self.health, health)
        if digest != self._last_digest:
            append_jsonl(
                self.counterfactual,
                {
                    "version": VERSION,
                    "generated_at": payload["generated_at"],
                    "profile_aligned": payload["profile"]["aligned"],
                    "positions": [
                        {
                            "symbol": item["symbol"],
                            "trade_id": item["trade_id"],
                            "binding": item["binding"],
                            "recommended_action": item["manager"]["recommended_action"],
                            "effective_authority": item["manager"]["effective_authority"],
                            "actual_execution": item["manager"]["actual_execution"],
                            "side_comparison": item["side_comparison"],
                            "shadow_manager_v1047": item["shadow_manager_v1047"],
                        }
                        for item in payload["positions"]
                    ],
                },
            )
            self._last_digest = digest
        return health


def verify(root: Path) -> int:
    runtime = LayerTruthRuntime(root)
    try:
        health = runtime.publish()
    except Exception as exc:
        print(f"[{VERSION}] VERIFICATION FAIL: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(health, indent=2, sort_keys=True))
    if not runtime.output.exists() or not runtime.health.exists():
        print(f"[{VERSION}] VERIFICATION FAIL: output files missing")
        return 1
    print(f"[{VERSION}] VERIFICATION PASS")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve()
    if not (root / "mythos" / "neural").is_dir():
        print(f"[{VERSION}] project root invalid: {root}")
        return 2
    if args.verify:
        return verify(root)
    runtime = LayerTruthRuntime(root)
    if args.once:
        print(json.dumps(runtime.publish(), indent=2, sort_keys=True))
        return 0
    print(f"[{VERSION}] started root={root} interval={args.interval:.2f}s")
    while True:
        try:
            health = runtime.publish()
            print(
                f"[{VERSION}] status={health['status']} open={health['open_positions']} "
                f"linked={health['linked_positions']} stale={health['stale_positions']} "
                f"unbound={health['unbound_positions']}"
            )
        except KeyboardInterrupt:
            print(f"[{VERSION}] stopped")
            return 0
        except Exception as exc:
            print(f"[{VERSION}] cycle error: {type(exc).__name__}: {exc}", file=sys.stderr)
        time.sleep(max(args.interval, 0.5))


if __name__ == "__main__":
    raise SystemExit(main())
