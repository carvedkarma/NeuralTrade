from __future__ import annotations

"""Rithal V1.0.4.7 R3 canonical layer-truth runtime.

Read-only / shadow-only. It never changes model weights, thresholds, TP/SL,
positions, orders, ledgers, or exchange authority.
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

VERSION = "RITHAL_LAYER_TRUTH_V1_0_4_7_R3"
INSTANCE_ID = "rithal-1-0-contract-locked"
EXPECTED_PROFILE = {
    "starting_equity_usd": 18000.0,
    "fixed_margin_pct": 0.10,
    "leverage": 10.0,
    "allocation_cap_pct": 0.60,
    "heat_cap_pct": 0.03,
    "max_open_positions": 6,
    "manager_mode": "SHADOW_ONLY",
}
SOURCE_PRIORITY = {
    "RITHAL_TM_V3_3": 400,
    "RITHAL_TM_V3_2": 300,
    "RITHAL_EXIT_BRAIN": 200,
    "LEGACY_MANAGER": 100,
}
REQUIRED_SOURCES = ("positions", "manager", "runtime", "account", "interactions", "control", "settings")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return now_utc().isoformat()


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    value = safe_float(value)
    return int(value) if value is not None else None


def parse_time(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 10_000_000_000:
            number /= 1000.0
        try:
            return datetime.fromtimestamp(number, timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        result = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return (result if result.tzinfo else result.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    except ValueError:
        return None


def read_json(path: Path, retries: int = 4) -> Any:
    if not path.is_file():
        return None
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            raw = path.read_text(encoding="utf-8-sig")
            return json.loads(raw) if raw.strip() else None
        except (PermissionError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    return {"_read_error": f"{type(last_error).__name__}: {last_error}"}


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
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


def first(mapping: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def recursive_first(value: Any, keys: Iterable[str]) -> Any:
    keys = tuple(keys)
    for _, mapping in iter_dicts(value):
        result = first(mapping, keys)
        if result not in (None, ""):
            return result
    return None


def symbol_of(mapping: Dict[str, Any]) -> Optional[str]:
    value = first(mapping, ("symbol", "sym", "asset", "ticker"))
    if value is None:
        return None
    text = str(value).strip().upper()
    return text if text.endswith("USDT") else None


def trade_id_of(mapping: Dict[str, Any]) -> Optional[str]:
    value = first(mapping, ("trade_id", "baseline_trade", "baseline_trade_id", "position_id", "core_trade_id"))
    if value is None and isinstance(mapping.get("ticket"), dict):
        value = first(mapping["ticket"], ("trade_id", "baseline_trade", "baseline_trade_id"))
    text = str(value).strip() if value is not None else ""
    return text or None


def ticket_id_of(mapping: Dict[str, Any]) -> Optional[str]:
    value = first(mapping, ("ticket_id", "manager_ticket_id", "five_minute_ticket_id", "ticket"))
    if isinstance(value, dict):
        value = first(value, ("id", "ticket_id"))
    text = str(value).strip() if value is not None else ""
    return text or None


def timestamp_of(mapping: Dict[str, Any]) -> Optional[datetime]:
    return parse_time(first(mapping, ("reviewed_at", "review_time", "updated_at", "at", "timestamp", "closed_5m", "decision_at", "created_at")))


def source_of(mapping: Dict[str, Any], path: str) -> Optional[str]:
    own = " ".join(str(mapping.get(key) or "") for key in ("module", "version", "source", "branch", "state", "profit_state", "rec")).upper()
    own = f"{path.upper()} {own}"
    if any(token in own for token in ("RITHAL_TM_V3_3", "PROFIT_STATE", "M70_P55", "P80_PARTIAL")):
        return "RITHAL_TM_V3_3"
    if any(token in own for token in ("RITHAL_TM_V3_2", "LOSS_RECOVERY_LAYER", "MID_MFE_REVERSAL", "EARLY_FAILURE", "EXTREME_REVERSAL")):
        return "RITHAL_TM_V3_2"
    if any(token in own for token in ("RITHAL_EXIT_BRAIN", "EXIT_BRAIN")):
        return "RITHAL_EXIT_BRAIN"
    if any(key in mapping for key in ("manager_action", "manager_state", "recommended_action")):
        return "LEGACY_MANAGER"
    return None


def state_of(mapping: Dict[str, Any]) -> str:
    value = first(mapping, ("state", "manager_state", "profit_state", "candidate_state", "manager_candidate_state", "action", "recommendation", "rec"))
    return str(value).strip().upper() if value is not None else "UNAVAILABLE"


def action_of(state: str, mapping: Dict[str, Any]) -> str:
    explicit = first(mapping, ("recommended_action", "recommendation", "rec"))
    combined = f"{state} {str(explicit or '').upper()}"
    if any(token in combined for token in ("P80_PARTIAL", "CLOSE_PARTIAL")):
        return "CLOSE_PARTIAL"
    if any(token in combined for token in ("SHADOW_EXIT_CONFIRMED", "MATURE_PROFIT_EXIT", "CLOSE_FULL", "FULL_EXIT")):
        return "CLOSE_FULL"
    if any(token in combined for token in ("WATCH", "WARNING")):
        return "WATCH"
    if "DATA_GATED" in combined:
        return "HOLD_DATA_GATED"
    return "HOLD"


def actionable_of(state: str, action: str, mapping: Dict[str, Any]) -> bool:
    if "SHADOW_EXIT_CONFIRMED" in state or action in {"CLOSE_FULL", "CLOSE_PARTIAL"} and bool(mapping.get("triggered")):
        return True
    return action in {"CLOSE_FULL", "CLOSE_PARTIAL"} and state in {"CLOSE_FULL", "CLOSE_PARTIAL", "EXIT"}


def action_rank(state: str, action: str, mapping: Dict[str, Any]) -> int:
    if actionable_of(state, action, mapping):
        return 500 if action == "CLOSE_FULL" else 450
    if action == "WATCH":
        return 300
    if action in {"CLOSE_FULL", "CLOSE_PARTIAL"}:
        return 200
    if action == "HOLD_DATA_GATED":
        return 50
    return 100


def actual_execution_of(mapping: Dict[str, Any]) -> str:
    value = first(mapping, ("actual_execution", "execution", "actual", "execution_result"))
    text = str(value).strip().upper() if value is not None else ""
    return text or "UNAVAILABLE"


def control_mode(control: Any, settings: Any = None) -> str:
    for payload in (control, settings):
        value = recursive_first(payload, ("manager_mode_effective", "manager_mode", "mode", "authority_mode", "trade_manager_mode"))
        if value is not None:
            text = str(value).strip().upper()
            if text in {"OFF", "SHADOW_ONLY", "PAPER_CONTROL", "FULL_CONTROL"}:
                return text
    return "UNAVAILABLE"


def manager_truth(manager_state: Any, position: Dict[str, Any], stale_seconds: int) -> Dict[str, Any]:
    trade_id = trade_id_of(position)
    ticket_id = ticket_id_of(position)
    symbol = symbol_of(position)
    exact: List[Dict[str, Any]] = []
    unbound_symbol_records = 0
    for path, mapping in iter_dicts(manager_state):
        source = source_of(mapping, path)
        if source is None:
            continue
        candidate_trade = trade_id_of(mapping) or (trade_id if trade_id and trade_id in path else None)
        candidate_ticket = ticket_id_of(mapping) or (ticket_id if ticket_id and ticket_id in path else None)
        trade_match = bool(trade_id and candidate_trade == trade_id)
        ticket_match = bool(ticket_id and candidate_ticket == ticket_id)
        if not (trade_match or ticket_match):
            if symbol and symbol_of(mapping) == symbol:
                unbound_symbol_records += 1
            continue
        state = state_of(mapping)
        action = action_of(state, mapping)
        reviewed_at = timestamp_of(mapping)
        exact.append({
            "path": path,
            "record": mapping,
            "source": source,
            "state": state,
            "action": action,
            "actionable": actionable_of(state, action, mapping),
            "rank": action_rank(state, action, mapping),
            "source_rank": SOURCE_PRIORITY[source],
            "reviewed_at": reviewed_at,
            "trade_match": trade_match,
            "ticket_match": ticket_match,
        })
    if not exact:
        return {
            "binding": "UNBOUND" if unbound_symbol_records else "UNAVAILABLE",
            "source": "UNAVAILABLE",
            "advisory_state": "UNAVAILABLE",
            "recommended_action": "UNAVAILABLE",
            "actionable": False,
            "effective_authority": "UNAVAILABLE",
            "actual_execution": "UNAVAILABLE",
            "reviewed_at": None,
            "review_age_seconds": None,
            "source_path": None,
            "reason": "exact_trade_or_ticket_record_not_found",
        }
    exact.sort(key=lambda row: (row["rank"], row["source_rank"], row["reviewed_at"].timestamp() if row["reviewed_at"] else 0.0), reverse=True)
    chosen = exact[0]
    age = max(0.0, (now_utc() - chosen["reviewed_at"]).total_seconds()) if chosen["reviewed_at"] else None
    binding = "STALE" if age is not None and age > stale_seconds else "LINKED"
    record = chosen["record"]
    return {
        "binding": binding,
        "source": chosen["source"],
        "advisory_state": chosen["state"],
        "recommended_action": chosen["action"],
        "actionable": chosen["actionable"],
        "effective_authority": "UNAVAILABLE",
        "actual_execution": actual_execution_of(record),
        "reviewed_at": chosen["reviewed_at"].isoformat() if chosen["reviewed_at"] else None,
        "review_age_seconds": round(age, 3) if age is not None else None,
        "source_path": chosen["path"],
        "reason": first(record, ("reason", "reason_codes", "branch", "severity")),
        "arbitration": "EXACT_IDENTITY_THEN_ACTIONABILITY_THEN_V3_3_V3_2_EXIT_BRAIN_LEGACY",
    }


def normalize_probs(value: Any) -> Optional[List[float]]:
    if isinstance(value, dict):
        aliases = (
            ("trend_up", "uptrend", "bull_trend"),
            ("trend_down", "downtrend", "bear_trend"),
            ("chop", "range", "ranging"),
            ("breakout", "break_out"),
        )
        values = []
        for group in aliases:
            found = next((safe_float(value.get(key)) for key in group if safe_float(value.get(key)) is not None), None)
            if found is None:
                return None
            values.append(found)
    elif isinstance(value, (list, tuple)) and len(value) >= 4:
        values = [safe_float(item) for item in value[:4]]
        if any(item is None for item in values):
            return None
    else:
        return None
    total = sum(float(item) for item in values)
    return [float(item) / total for item in values] if total > 0 else None


def regime_truth(payload: Any) -> Dict[str, Any]:
    true_probs = normalize_probs(recursive_first(payload, ("true_regime_probs", "regime_probs", "regime_posterior")))
    rank_probs = normalize_probs(recursive_first(payload, ("rank_regime_probs", "ranking_regime_probs")))
    if true_probs is None:
        named = [
            safe_float(recursive_first(payload, ("p_trend_up",))),
            safe_float(recursive_first(payload, ("p_trend_down",))),
            safe_float(recursive_first(payload, ("p_chop",))),
            safe_float(recursive_first(payload, ("p_breakout",))),
        ]
        if all(item is not None for item in named):
            true_probs = normalize_probs(named)
    labels = ["TREND_UP", "TREND_DOWN", "CHOP", "BREAKOUT"]
    if true_probs is None:
        return {"valid": False, "state": "REGIME_UNKNOWN", "fallback_reason": "missing_or_invalid_regime_output", "class_order": labels, "true_probs": None, "rank_probs": rank_probs, "p_trend": None, "p_chop": None, "p_breakout": None, "entropy": None}
    entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in true_probs)
    return {
        "valid": True,
        "state": labels[max(range(4), key=true_probs.__getitem__)],
        "fallback_reason": None,
        "class_order": labels,
        "true_probs": [round(item, 8) for item in true_probs],
        "rank_probs": [round(item, 8) for item in rank_probs] if rank_probs else None,
        "p_trend": round(true_probs[0] + true_probs[1], 8),
        "p_chop": round(true_probs[2], 8),
        "p_breakout": round(true_probs[3], 8),
        "entropy": round(entropy, 8),
    }


def side_truth(payload: Any) -> Dict[str, Any]:
    canonical = recursive_first(payload, ("canonical_side", "selected_side", "side"))
    canonical = str(canonical).strip().upper() if canonical is not None else "UNAVAILABLE"
    long_edge = safe_float(recursive_first(payload, ("risk_adjusted_long_edge", "long_edge")))
    short_edge = safe_float(recursive_first(payload, ("risk_adjusted_short_edge", "short_edge")))
    shadow = "UNAVAILABLE" if long_edge is None or short_edge is None else ("LONG" if long_edge >= short_edge else "SHORT")
    return {"canonical_side": canonical, "risk_adjusted_shadow_side": shadow, "long_edge": long_edge, "short_edge": short_edge, "agreement": None if "UNAVAILABLE" in {canonical, shadow} else canonical == shadow, "mode": "SHADOW_ONLY", "execution_changed": False}


def sizing_truth(position: Dict[str, Any]) -> Dict[str, Any]:
    controller = recursive_first(position, ("controller_decision",))
    controller = controller if isinstance(controller, dict) else {}
    proposed = safe_float(first(controller, ("proposed_size_multiplier", "proposed_multiplier")))
    effective = safe_float(first(controller, ("effective_size_multiplier", "effective_multiplier", "size_multiplier")))
    applied = safe_float(first(position, ("applied_size_multiplier", "lane_mult", "controller_size_multiplier")))
    base_margin = safe_float(first(position, ("base_margin", "base_margin_usd")))
    requested_margin = safe_float(first(position, ("requested_margin", "requested_margin_usd", "margin", "margin_usd")))
    if applied is None and base_margin and requested_margin is not None and base_margin > 0:
        applied = requested_margin / base_margin
    contradictions = []
    if effective is not None and applied is not None and abs(effective - applied) > 1e-9:
        contradictions.append("effective_multiplier_differs_from_applied_multiplier")
    if proposed is not None and effective is None:
        contradictions.append("proposal_present_but_effective_missing")
    return {"proposed_multiplier": proposed, "effective_multiplier": effective, "applied_multiplier": applied, "base_margin_usd": base_margin, "requested_margin_usd": requested_margin, "contradictions": contradictions, "canonical_quantity_source": "applied_multiplier"}


def metrics_of(payload: Any) -> Dict[str, Optional[float]]:
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
    return {name: safe_float(recursive_first(payload, keys)) for name, keys in aliases.items()}


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
            rows = []
            for symbol, item in value.items():
                if isinstance(item, dict):
                    row = dict(item)
                    row.setdefault("symbol", symbol)
                    rows.append(row)
            if rows:
                return rows
    return []


def exact_thesis(interactions: Any, trade_id: Optional[str], ticket_id: Optional[str]) -> Dict[str, Any]:
    matches = []
    for path, mapping in iter_dicts(interactions):
        mapped_trade = trade_id_of(mapping) or (trade_id if trade_id and trade_id in path else None)
        mapped_ticket = ticket_id_of(mapping) or (ticket_id if ticket_id and ticket_id in path else None)
        if (trade_id and mapped_trade == trade_id) or (ticket_id and mapped_ticket == ticket_id):
            matches.append((timestamp_of(mapping), mapping, path))
    if not matches:
        return {"binding": "UNAVAILABLE", "source_path": None, "snapshot": None}
    matches.sort(key=lambda row: row[0].timestamp() if row[0] else 0.0, reverse=True)
    _, mapping, path = matches[0]
    keys = ("side", "decision_score", "threshold", "expected_r_long", "expected_r_short", "p_win_long", "p_win_short", "mae_long", "mae_short", "long_edge", "short_edge", "edge_margin", "router_confidence", "calibration_score", "specialist_confidence", "specialist", "feature_health", "model_input_health", "repaired_features", "ood_features", "raw_composite", "sequence_fingerprint")
    snapshot = {key: recursive_first(mapping, (key,)) for key in keys}
    snapshot["regime"] = regime_truth(mapping)
    snapshot["side_comparison"] = side_truth(mapping)
    return {"binding": "LINKED", "source_path": path, "snapshot": snapshot}


def resolve_state_paths(root: Path) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    instance_dir = root / "mythos_model_instances" / INSTANCE_ID
    manifest_path = instance_dir / "manifest.json"
    manifest = read_json(manifest_path)
    manifest = manifest if isinstance(manifest, dict) else {}
    manifest_id = str(manifest.get("model_instance_id") or manifest.get("instance_id") or "").strip()
    if manifest_id and manifest_id != INSTANCE_ID:
        raise RuntimeError(f"manifest instance mismatch: {manifest_id}")

    def candidate(manifest_key: str, *fallbacks: Path) -> Path:
        rows: List[Path] = []
        raw = manifest.get(manifest_key)
        if raw:
            path = Path(str(raw))
            rows.append(path if path.is_absolute() else root / path)
        rows.extend(fallbacks)
        for path in rows:
            if path.is_file():
                payload = read_json(path)
                identity = recursive_first(payload, ("model_instance_id", "instance_id"))
                if identity and str(identity).strip() != INSTANCE_ID:
                    continue
                return path.resolve()
        return rows[0].resolve()

    paths = {
        "positions": candidate("positions_file", instance_dir / "open_positions.json", root / "mythos_open_positions.json"),
        "manager": candidate("trade_manager_state_file", instance_dir / "trade_manager_state.json", root / "mythos_trade_manager_state.json"),
        "runtime": candidate("runtime_file", instance_dir / "runtime_state.json", root / "mythos_runtime_state.json"),
        "account": candidate("account_file", instance_dir / "account_state.json", instance_dir / "paper_ledger_state.json", root / "mythos_account_state.json"),
        "interactions": candidate("interaction_file", instance_dir / "interactions.json", root / "mythos_model_interactions.json"),
        "prices": candidate("prices_file", instance_dir / "live_prices.json", root / "mythos_live_prices.json"),
        "control": root / "mythos_5m_execution_control.json",
        "settings": root / "settings_override_v1.json",
    }
    return paths, {"path": str(manifest_path), "available": manifest_path.is_file(), "instance_id": manifest_id or INSTANCE_ID}


def profile_truth(runtime: Any, account: Any, control: Any, settings: Any, manifest: Dict[str, Any]) -> Dict[str, Any]:
    desired = recursive_first(settings, ("rithal_layer_contract_v1047",))
    desired = desired if isinstance(desired, dict) else EXPECTED_PROFILE
    actual_payloads = (runtime, account, control)

    def value(keys: Iterable[str]) -> Any:
        for payload in actual_payloads:
            found = recursive_first(payload, keys)
            if found not in (None, ""):
                return found
        return None

    actual = {
        "instance_id": str(value(("model_instance_id", "instance_id")) or manifest.get("instance_id") or "UNAVAILABLE"),
        "starting_equity_usd": safe_float(value(("starting_equity_usd", "initial_equity", "paper_starting_equity", "paper_equity_baseline"))),
        "current_equity_usd": safe_float(value(("current_equity_usd", "paper_equity", "equity", "balance"))),
        "fixed_margin_pct": safe_float(value(("fixed_margin_pct", "paper_fixed_margin_pct", "margin_pct"))),
        "leverage": safe_float(value(("effective_leverage", "live_leverage", "paper_leverage", "leverage"))),
        "allocation_cap_pct": safe_float(value(("allocation_cap_pct", "allocation_cap", "max_portfolio_margin_pct"))),
        "heat_cap_pct": safe_float(value(("heat_cap_pct", "heat_cap", "portfolio_heat_cap"))),
        "max_open_positions": safe_int(value(("max_open_positions", "max_positions", "max_concurrent_positions"))),
        "manager_mode": control_mode(control),
    }
    mismatches = []
    for key, expected in EXPECTED_PROFILE.items():
        current = actual.get(key)
        if current is None or current == "UNAVAILABLE":
            mismatches.append(f"{key}:UNAVAILABLE")
        elif isinstance(expected, (int, float)):
            if abs(float(current) - float(expected)) > (0.01 if key == "starting_equity_usd" else 1e-9):
                mismatches.append(f"{key}:{current}!={expected}")
        elif str(current).upper() != str(expected).upper():
            mismatches.append(f"{key}:{current}!={expected}")
    if actual["instance_id"] != INSTANCE_ID:
        mismatches.append(f"instance_id:{actual['instance_id']}!={INSTANCE_ID}")
    return {"expected": EXPECTED_PROFILE, "desired_contract": desired, "runtime_actual": actual, "mismatches": mismatches, "aligned": not mismatches, "note": "current_equity_usd is telemetry and is not compared with the $18,000 starting-equity contract"}


@dataclass
class LifecycleState:
    last_review_key: Optional[str] = None
    last_mfe: Optional[float] = None
    no_new_mfe_reviews: int = 0
    giveback_reviews: int = 0


def shadow_policy(trade_key: str, metrics: Dict[str, Optional[float]], state: LifecycleState, review_key: Optional[str]) -> Dict[str, Any]:
    review_advanced = bool(review_key and review_key != state.last_review_key)
    if review_advanced:
        state.last_review_key = review_key
        mfe = metrics.get("mfe_r")
        if mfe is not None:
            if state.last_mfe is None or mfe > state.last_mfe + 0.02:
                state.last_mfe = mfe
                state.no_new_mfe_reviews = 0
            else:
                state.no_new_mfe_reviews += 1
        giveback = metrics.get("giveback_r")
        state.giveback_reviews = state.giveback_reviews + 1 if giveback is not None and giveback >= 0.35 else 0
    gross = metrics.get("gross_r")
    mfe = metrics.get("mfe_r")
    giveback = metrics.get("giveback_r")
    age = metrics.get("age_minutes")
    current_progress = metrics.get("current_progress_pct")
    mfe_progress = metrics.get("mfe_progress_pct")
    earlier_loss = bool(mfe is not None and gross is not None and giveback is not None and mfe >= 0.45 and gross <= -0.15 and giveback >= 0.65 and state.giveback_reviews >= 2)
    mature = bool(age is not None and current_progress is not None and age >= 70 and current_progress >= 55 and state.no_new_mfe_reviews >= 3 and state.giveback_reviews >= 2)
    p80 = bool(mfe_progress is not None and mfe_progress >= 80 and state.no_new_mfe_reviews >= 2 and state.giveback_reviews >= 2)
    action = "SHADOW_CLOSE_FULL_EARLIER_LOSS" if earlier_loss else "SHADOW_CLOSE_FULL_MATURE_STAGNATION" if mature else "SHADOW_CLOSE_PARTIAL_P80_PERSISTENT_EROSION" if p80 else "SHADOW_HOLD"
    return {"trade_id": trade_key, "action": action, "mode": "SHADOW_ONLY", "execution_changed": False, "review_advanced": review_advanced, "no_new_mfe_reviews": state.no_new_mfe_reviews, "persistent_giveback_reviews": state.giveback_reviews, "conditions": {"earlier_loss": earlier_loss, "mature_exit_with_stagnation": mature, "p80_partial_with_persistent_erosion": p80}}


class LayerTruthRuntime:
    def __init__(self, root: Path, stale_seconds: int = 650):
        self.root = root.resolve()
        self.stale_seconds = stale_seconds
        self.paths, self.manifest = resolve_state_paths(self.root)
        self.output = self.root / "mythos_rithal_intelligence_truth.json"
        self.health_file = self.root / "mythos_rithal_layer_health.json"
        self.counterfactual = self.root / "mythos_rithal_counterfactual.jsonl"
        self._last_digest: Optional[str] = None
        self._lifecycle: Dict[str, LifecycleState] = {}

    def load(self) -> Dict[str, Any]:
        return {name: read_json(path) for name, path in self.paths.items()}

    def build(self) -> Dict[str, Any]:
        data = self.load()
        mode = control_mode(data["control"], data["settings"])
        rows = []
        for position in list_positions(data["positions"]):
            trade_id = trade_id_of(position)
            ticket_id = ticket_id_of(position)
            manager = manager_truth(data["manager"], position, self.stale_seconds)
            manager["effective_authority"] = mode
            if manager["actual_execution"] == "UNAVAILABLE" and mode == "SHADOW_ONLY":
                manager["actual_execution"] = "UNCHANGED_SHADOW_ONLY"
            thesis = exact_thesis(data["interactions"], trade_id, ticket_id)
            metrics = metrics_of(position)
            if manager.get("source_path"):
                for path, mapping in iter_dicts(data["manager"]):
                    if path == manager["source_path"]:
                        manager_metrics = metrics_of(mapping)
                        metrics = {key: manager_metrics[key] if manager_metrics[key] is not None else value for key, value in metrics.items()}
                        break
            key = trade_id or ticket_id or f"UNBOUND:{symbol_of(position) or 'UNAVAILABLE'}"
            lifecycle = self._lifecycle.setdefault(key, LifecycleState())
            snapshot = thesis.get("snapshot") if isinstance(thesis, dict) else None
            rows.append({
                "symbol": symbol_of(position) or "UNAVAILABLE",
                "trade_id": trade_id,
                "ticket_id": ticket_id,
                "binding": manager["binding"],
                "manager": manager,
                "metrics": metrics,
                "entry_thesis": thesis,
                "regime": snapshot.get("regime") if isinstance(snapshot, dict) and isinstance(snapshot.get("regime"), dict) else regime_truth(position),
                "side_comparison": snapshot.get("side_comparison") if isinstance(snapshot, dict) and isinstance(snapshot.get("side_comparison"), dict) else side_truth(position),
                "sizing": sizing_truth(position),
                "shadow_manager_v1047_r3": shadow_policy(key, metrics, lifecycle, manager.get("reviewed_at")),
            })
        source_health = {name: {"path": str(path), "available": path.is_file(), "read_error": data[name].get("_read_error") if isinstance(data[name], dict) else None} for name, path in self.paths.items()}
        return {
            "schema_version": 3,
            "version": VERSION,
            "generated_at": now_iso(),
            "manifest": self.manifest,
            "execution_policy": {"model_weights_changed": False, "thresholds_changed": False, "tp_sl_changed": False, "exchange_authority_changed": False, "behavioral_improvements": "SHADOW_ONLY"},
            "profile": profile_truth(data["runtime"], data["account"], data["control"], data["settings"], self.manifest),
            "positions": rows,
            "source_health": source_health,
        }

    def publish(self) -> Dict[str, Any]:
        payload = self.build()
        semantic = dict(payload)
        semantic.pop("generated_at", None)
        digest = hashlib.sha256(json.dumps(semantic, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        payload["content_sha256"] = digest
        atomic_write_json(self.output, payload)
        missing = [name for name in REQUIRED_SOURCES if not payload["source_health"][name]["available"]]
        read_errors = [name for name, item in payload["source_health"].items() if item["read_error"]]
        unlinked = sum(1 for item in payload["positions"] if item["binding"] != "LINKED")
        status = "FAIL" if read_errors or payload["profile"]["runtime_actual"]["manager_mode"] not in {"SHADOW_ONLY", "OFF"} else "WARN" if missing or payload["profile"]["mismatches"] or unlinked else "PASS"
        health = {"version": VERSION, "generated_at": payload["generated_at"], "status": status, "profile_aligned": payload["profile"]["aligned"], "profile_mismatches": payload["profile"]["mismatches"], "missing_sources": missing, "read_errors": read_errors, "open_positions": len(payload["positions"]), "linked_positions": sum(1 for item in payload["positions"] if item["binding"] == "LINKED"), "stale_positions": sum(1 for item in payload["positions"] if item["binding"] == "STALE"), "unbound_positions": sum(1 for item in payload["positions"] if item["binding"] == "UNBOUND"), "truth_file": str(self.output)}
        atomic_write_json(self.health_file, health)
        if digest != self._last_digest:
            append_jsonl(self.counterfactual, {"version": VERSION, "generated_at": payload["generated_at"], "content_sha256": digest, "positions": [{"symbol": row["symbol"], "trade_id": row["trade_id"], "binding": row["binding"], "recommended_action": row["manager"]["recommended_action"], "effective_authority": row["manager"]["effective_authority"], "actual_execution": row["manager"]["actual_execution"], "side_comparison": row["side_comparison"], "shadow_manager": row["shadow_manager_v1047_r3"]} for row in payload["positions"]]})
            self._last_digest = digest
        return health


def verify(root: Path) -> int:
    try:
        runtime = LayerTruthRuntime(root)
        health = runtime.publish()
    except Exception as exc:
        print(f"[{VERSION}] VERIFICATION FAIL: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(health, indent=2, sort_keys=True))
    if not runtime.output.is_file() or not runtime.health_file.is_file() or health["status"] == "FAIL":
        print(f"[{VERSION}] VERIFICATION FAIL")
        return 1
    if health["status"] == "WARN":
        print(f"[{VERSION}] VERIFICATION PASS WITH WARNINGS")
    else:
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
            print(f"[{VERSION}] status={health['status']} open={health['open_positions']} linked={health['linked_positions']} stale={health['stale_positions']} unbound={health['unbound_positions']}")
        except KeyboardInterrupt:
            print(f"[{VERSION}] stopped")
            return 0
        except Exception as exc:
            print(f"[{VERSION}] cycle error: {type(exc).__name__}: {exc}", file=sys.stderr)
        time.sleep(max(args.interval, 0.5))


if __name__ == "__main__":
    raise SystemExit(main())
