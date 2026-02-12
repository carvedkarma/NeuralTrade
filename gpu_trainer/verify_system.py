#!/usr/bin/env python3
"""
System Verification Module (v4.3.x)
====================================
Runs N cycles of assertions to prove that the Triple-Lane Aggression Engine,
CROSS correlation blocking, quota controller, and payload completeness are
all wired correctly end-to-end.

Usage:
    python quick_start.py --paper --verify-system --cycles 30 --url <URL>
"""

import logging
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

log = logging.getLogger("Verify")


@dataclass
class VerifyFailure:
    cycle: int
    category: str
    message: str


@dataclass
class VerifyStats:
    total_cycles: int = 0
    lane_decisions: Dict[str, int] = field(default_factory=lambda: {"CORE": 0, "FLOW": 0, "SCALP": 0, "HOLD": 0, "COOLDOWN": 0})
    cross_blocks: int = 0
    cross_allows_opposite: int = 0
    cross_checks_total: int = 0
    quota_steps_seen: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0, 3: 0})
    flow_thr_values: List[float] = field(default_factory=list)
    core_thr_values: List[float] = field(default_factory=list)
    payload_checks: int = 0
    payload_passes: int = 0
    net_check_passes: int = 0
    net_check_violations: int = 0
    failures: List[VerifyFailure] = field(default_factory=list)


REQUIRED_CYCLE_PAYLOAD_KEYS = [
    "lane_selected", "htf_score", "hold_reason",
    "quota_step", "core_thr", "flow_thr", "scalp_thr",
    "lane_size_mult", "lane_budget_remaining_r",
]


class SystemVerifier:
    """Hooks into the LiveRunner to verify system invariants."""

    def __init__(self, max_cycles: int = 30):
        self.max_cycles = max_cycles
        self.stats = VerifyStats()
        self.cycle_payloads: List[dict] = []

    def verify_lane_routing(self, cycle: int, lane_result: dict, p_enter: float,
                            htf_score: int, htf: dict) -> List[VerifyFailure]:
        """Assert lane routing priority: CORE > FLOW > SCALP > HOLD."""
        failures = []
        lane = lane_result.get('lane_selected', 'UNKNOWN')
        self.stats.lane_decisions[lane] = self.stats.lane_decisions.get(lane, 0) + 1

        range_ok = htf.get('range_ok', False)
        momentum_ok = lane_result.get('momentum_ok', False)
        volatility_ok = lane_result.get('volatility_ok', False)
        core_thr = lane_result.get('core_thr')
        flow_thr = lane_result.get('flow_thr')
        scalp_thr = lane_result.get('scalp_thr')

        if core_thr is not None:
            self.stats.core_thr_values.append(core_thr)
        if flow_thr is not None:
            self.stats.flow_thr_values.append(flow_thr)

        if lane == 'HOLD':
            hold_reason = lane_result.get('hold_reason', '')
            if not hold_reason:
                failures.append(VerifyFailure(
                    cycle, "LANE_ROUTING",
                    f"HOLD decision has empty hold_reason"
                ))

        if lane == 'FLOW':
            if htf_score >= 3 and range_ok and core_thr is not None and p_enter >= core_thr:
                failures.append(VerifyFailure(
                    cycle, "LANE_PRIORITY",
                    f"FLOW selected but CORE conditions met: htf={htf_score} range_ok={range_ok} "
                    f"p_enter={p_enter:.4f} >= core_thr={core_thr:.4f}"
                ))

        if lane == 'SCALP':
            if htf_score >= 3 and range_ok and core_thr is not None and p_enter >= core_thr:
                failures.append(VerifyFailure(
                    cycle, "LANE_PRIORITY",
                    f"SCALP selected but CORE conditions met: htf={htf_score} range_ok={range_ok} "
                    f"p_enter={p_enter:.4f} >= core_thr={core_thr:.4f}"
                ))
            if htf_score >= 2 and flow_thr is not None and p_enter >= flow_thr:
                failures.append(VerifyFailure(
                    cycle, "LANE_PRIORITY",
                    f"SCALP selected but FLOW conditions met: htf={htf_score} "
                    f"p_enter={p_enter:.4f} >= flow_thr={flow_thr:.4f}"
                ))

        return failures

    def verify_cross_blocking(self, cycle: int, symbol: str, side: str,
                              blocked: bool, blocked_by: Optional[str],
                              existing_positions: Dict) -> List[VerifyFailure]:
        """Assert CROSS blocks only same-direction correlated exposure."""
        failures = []
        self.stats.cross_checks_total += 1

        if blocked and blocked_by:
            self.stats.cross_blocks += 1
            if blocked_by in existing_positions:
                existing_side = existing_positions[blocked_by].side
                if existing_side != side:
                    failures.append(VerifyFailure(
                        cycle, "CROSS_BLOCK",
                        f"CROSS blocked {symbol} {side} with {blocked_by} {existing_side} -- "
                        f"opposite-direction should NOT be blocked!"
                    ))

        if not blocked:
            for pos_sym, pos in existing_positions.items():
                if pos.side != side:
                    self.stats.cross_allows_opposite += 1

        return failures

    def verify_payload(self, cycle: int, payload: dict) -> List[VerifyFailure]:
        """Assert all required keys exist in cycle payload (the actual POST payload dict).
        
        For HOLD decisions, thresholds/size_mult/budget fields may legitimately be None
        (e.g., NEUTRAL_SIDE hold). For ENTER decisions, all fields must be populated.
        """
        failures = []
        self.stats.payload_checks += 1

        lane = payload.get('lane_selected')
        non_hold_keys = ["core_thr", "flow_thr", "scalp_thr", "lane_size_mult", "lane_budget_remaining_r"]
        always_required = [k for k in REQUIRED_CYCLE_PAYLOAD_KEYS if k not in non_hold_keys]

        if lane in ('HOLD', None):
            actual_missing = [k for k in always_required if k not in payload or payload[k] is None]
        else:
            actual_missing = [k for k in REQUIRED_CYCLE_PAYLOAD_KEYS if k not in payload or payload[k] is None]

        if actual_missing:
            failures.append(VerifyFailure(
                cycle, "PAYLOAD_MISSING",
                f"Cycle payload missing keys: {actual_missing} (lane={payload.get('lane_selected')})"
            ))
        else:
            self.stats.payload_passes += 1

        self.cycle_payloads.append(payload)
        return failures

    def verify_net_r(self, cycle: int, gross_r: float, cost_r: float, net_r: float) -> List[VerifyFailure]:
        """Assert net_r == gross_r - cost_r within tolerance."""
        failures = []
        expected = gross_r - cost_r
        diff = abs(net_r - expected)
        if diff > 1e-6:
            self.stats.net_check_violations += 1
            failures.append(VerifyFailure(
                cycle, "NET_CHECK",
                f"net_r={net_r:.6f} != gross_r={gross_r:.6f} - cost_r={cost_r:.6f} (diff={diff:.8f})"
            ))
        else:
            self.stats.net_check_passes += 1
        return failures

    def verify_flow_threshold_stepping(self) -> List[VerifyFailure]:
        """Assert FLOW threshold differs from CORE threshold at least once."""
        failures = []
        if len(self.stats.core_thr_values) > 0 and len(self.stats.flow_thr_values) > 0:
            min_len = min(len(self.stats.core_thr_values), len(self.stats.flow_thr_values))
            any_different = any(
                abs(self.stats.flow_thr_values[i] - self.stats.core_thr_values[i]) > 1e-6
                for i in range(min_len)
            )
            if not any_different:
                failures.append(VerifyFailure(
                    0, "FLOW_STEPPING",
                    f"flow_thr NEVER differed from core_thr across {min_len} matched cycles. "
                    f"FLOW quota stepping may not be working."
                ))
        return failures

    def record_quota_step(self, quota_step: int):
        """Track quota step distribution."""
        self.stats.quota_steps_seen[quota_step] = self.stats.quota_steps_seen.get(quota_step, 0) + 1

    def add_failure(self, failure: VerifyFailure):
        self.stats.failures.append(failure)

    def add_failures(self, failures: List[VerifyFailure]):
        self.stats.failures.extend(failures)

    def generate_report(self) -> str:
        """Generate the final verification report."""
        s = self.stats
        all_failures = s.failures.copy()
        all_failures.extend(self.verify_flow_threshold_stepping())

        passed = len(all_failures) == 0
        status = "PASSED" if passed else "FAILED"

        lines = []
        lines.append("# System Verification Report (v4.3.x)")
        lines.append(f"")
        lines.append(f"## Overall Status: **VERIFICATION {status}**")
        lines.append(f"")
        lines.append(f"Cycles run: {s.total_cycles}")
        lines.append(f"Total failures: {len(all_failures)}")
        lines.append(f"")

        lines.append("## Summary Table")
        lines.append("")
        lines.append("| Check | Status |")
        lines.append("|-------|--------|")

        lane_ok = not any(f.category == "LANE_PRIORITY" for f in all_failures)
        lines.append(f"| Lane Router Priority (CORE>FLOW>SCALP>HOLD) | {'PASS' if lane_ok else 'FAIL'} |")

        hold_ok = not any(f.category == "LANE_ROUTING" for f in all_failures)
        lines.append(f"| HOLD includes hold_reason | {'PASS' if hold_ok else 'FAIL'} |")

        cross_ok = not any(f.category == "CROSS_BLOCK" for f in all_failures)
        lines.append(f"| CROSS blocks same-dir only | {'PASS' if cross_ok else 'FAIL'} |")

        flow_step_ok = not any(f.category == "FLOW_STEPPING" for f in all_failures)
        lines.append(f"| FLOW threshold stepping | {'PASS' if flow_step_ok else 'FAIL'} |")

        payload_ok = not any(f.category == "PAYLOAD_MISSING" for f in all_failures)
        lines.append(f"| Cycle payload completeness | {'PASS' if payload_ok else 'FAIL'} |")

        net_ok = not any(f.category == "NET_CHECK" for f in all_failures)
        lines.append(f"| net_r accounting invariant | {'PASS' if net_ok else 'FAIL'} |")

        lines.append(f"")
        lines.append("## Lane Decision Distribution")
        lines.append(f"")
        for lane, count in sorted(s.lane_decisions.items()):
            lines.append(f"- {lane}: {count}")
        lines.append(f"")

        lines.append("## CROSS Blocking Stats")
        lines.append(f"- Total CROSS checks: {s.cross_checks_total}")
        lines.append(f"- Same-direction blocks: {s.cross_blocks}")
        lines.append(f"- Opposite-direction allows: {s.cross_allows_opposite}")
        lines.append(f"")

        lines.append("## Quota Step Distribution")
        lines.append(f"")
        for step, count in sorted(s.quota_steps_seen.items()):
            lines.append(f"- Step {step}: {count} cycles")
        lines.append(f"")

        lines.append("## Payload Checks")
        lines.append(f"- Total checks: {s.payload_checks}")
        lines.append(f"- Passes: {s.payload_passes}")
        lines.append(f"- Required keys: {REQUIRED_CYCLE_PAYLOAD_KEYS}")
        lines.append(f"")

        lines.append("## Net R Accounting")
        lines.append(f"- Passes: {s.net_check_passes}")
        lines.append(f"- Violations: {s.net_check_violations}")
        lines.append(f"")

        if s.flow_thr_values and s.core_thr_values:
            lines.append("## Threshold Samples (first 5)")
            lines.append(f"")
            for i in range(min(5, len(s.flow_thr_values))):
                ct = s.core_thr_values[i] if i < len(s.core_thr_values) else "?"
                ft = s.flow_thr_values[i]
                lines.append(f"- Cycle ~{i}: core_thr={ct} flow_thr={ft}")
            lines.append(f"")

        if all_failures:
            lines.append("## Failures (first 10)")
            lines.append(f"")
            for f in all_failures[:10]:
                lines.append(f"- [Cycle {f.cycle}] **{f.category}**: {f.message}")
            lines.append(f"")

        lines.append("---")
        lines.append(f"Report generated at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

        return "\n".join(lines)

    def save_report(self, path: str = "verify_report.md"):
        report = self.generate_report()
        with open(path, 'w') as f:
            f.write(report)
        log.info(f"Verification report saved to {path}")
        print("\n" + report)
        return report


def run_static_audit() -> str:
    """Phase 1: Static code audit -- check key functions exist and are wired correctly."""
    lines = []
    lines.append("## Phase 1: Static Code Audit")
    lines.append("")

    import inspect
    from live_runner import LiveRunner, _compute_htf_score, _compute_momentum_ok, _compute_volatility_ok
    from live_runner import LANE_BUDGET, FLOW_QUOTA_STEPS, SCALP_HORIZON, SCALP_TP_R, SCALP_SL_R, SCALP_SIZE_MULT
    from portfolio import PortfolioManager, Position

    lines.append("### Lane Router (`_select_lane` in live_runner.py)")
    src = inspect.getsource(LiveRunner._select_lane)
    has_core_check = "htf_score >= 3" in src and "range_ok" in src
    has_flow_check = "htf_score >= 2" in src
    has_scalp_check = "htf_score >= 1" in src and "volatility_ok" in src and "momentum_ok" in src
    has_ordered_routing = src.index("CORE") < src.index("FLOW") < src.index("SCALP")
    lines.append(f"- CORE check (htf>=3 + range_ok): {'FOUND' if has_core_check else 'MISSING'}")
    lines.append(f"- FLOW check (htf>=2): {'FOUND' if has_flow_check else 'MISSING'}")
    lines.append(f"- SCALP check (htf>=1 + vol + mom): {'FOUND' if has_scalp_check else 'MISSING'}")
    lines.append(f"- Ordered routing CORE>FLOW>SCALP: {'CORRECT' if has_ordered_routing else 'WRONG'}")
    lines.append(f"- Uses distinct thresholds per lane: {'YES' if 'core_thr' in src and 'flow_thr' in src and 'scalp_thr' in src else 'NO'}")
    lines.append("")

    lines.append("### HTF Score (`_compute_htf_score` in live_runner.py)")
    htf_src = inspect.getsource(_compute_htf_score)
    lines.append(f"- Scores h1_trend match: {'YES' if 'h1' in htf_src else 'NO'}")
    lines.append(f"- Scores h4_trend match: {'YES' if 'h4' in htf_src else 'NO'}")
    lines.append(f"- Scores slope_ok: {'YES' if 'slope_ok' in htf_src else 'NO'}")
    lines.append("")

    lines.append("### CROSS Blocking (`_correlated_block` + `can_enter` in portfolio.py)")
    cross_src = inspect.getsource(PortfolioManager._correlated_block)
    can_enter_src = inspect.getsource(PortfolioManager.can_enter)
    lines.append(f"- Checks same-direction only: {'YES' if 'self.open_positions[partner].side == side' in cross_src else 'NO'}")
    lines.append(f"- Does NOT block opposite: {'CORRECT' if 'side == side' in cross_src else 'NEEDS CHECK'}")
    lines.append(f"- Checks BEFORE placing: {'YES' if 'blocked_by' in can_enter_src else 'NO'}")
    lines.append(f"- Logs specific reason: {'YES' if 'CROSS_BLOCK' in can_enter_src else 'NO'}")
    lines.append("")

    lines.append("### Quota Controller (in `_select_lane`)")
    lines.append(f"- quota_step computed: {'YES' if 'quota_step' in src else 'NO'}")
    lines.append(f"- FLOW_QUOTA_STEPS defined: {json.dumps({k: v for k, v in FLOW_QUOTA_STEPS.items()})}")
    lines.append(f"- Flow percentile stepping: {'YES' if 'flow_pct' in src else 'NO'}")
    lines.append(f"- Daily budget reset: {'YES' if '_reset_daily_budget_if_needed' in inspect.getsource(LiveRunner._process_symbol) else 'NO'}")
    lines.append("")

    lines.append("### SCALP Geometry Constants")
    lines.append(f"- SCALP_HORIZON: {SCALP_HORIZON}")
    lines.append(f"- SCALP_TP_R: {SCALP_TP_R}")
    lines.append(f"- SCALP_SL_R: {SCALP_SL_R}")
    lines.append(f"- SCALP_SIZE_MULT: {SCALP_SIZE_MULT}")
    lines.append("")

    lines.append("### Daily R Budgets")
    lines.append(f"- LANE_BUDGET: {json.dumps(LANE_BUDGET)}")
    lines.append("")

    lines.append("### Position Close Callback")
    close_src = inspect.getsource(PortfolioManager.close_position)
    lines.append(f"- on_close_callback called: {'YES' if 'on_close_callback' in close_src else 'NO'}")
    lines.append(f"- dashboard_trade_id checked: {'YES' if 'dashboard_trade_id' in close_src else 'NO'}")
    lines.append("")

    lines.append("### SCALP Time-Stop")
    exit_src = inspect.getsource(Position.check_exit)
    lines.append(f"- TIME_EXIT for SCALP: {'YES' if 'TIME_EXIT' in exit_src else 'NO'}")
    lines.append(f"- horizon check: {'YES' if 'self.horizon' in exit_src else 'NO'}")
    lines.append("")

    return "\n".join(lines)
