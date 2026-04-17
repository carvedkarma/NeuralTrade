"""
Append the V11 walk-forward verdict block to v5-static-postmortem-verdict.md.

Reads:
    reports/walkforward_rule_A_h*.json   (V11 specialist A)
    reports/walkforward_rule_B_h*.json   (V11 specialist B)
    reports/diversification_*.json       (optional)
    reports/xgb_baseline.json            (REQUIRED for honest baseline-beat check;
                                          produced by gpu_trainer/eval/honest_walkforward.py
                                          and copied/symlinked into V11 reports/)

Per the locked contract (README §"Stop criterion"), V11 PASSES only if:
    1. avg PF >= 1.3 across 6 test folds
    2. all per-fold trade counts >= 500
    3. no fold PF < 1.0
    4. AND beats the XGBoost Phase-1 baseline avg PF (read from xgb_baseline.json)

The `passed` field on each per-rule walk-forward JSON only enforces (1)-(3);
this script enforces (4) by combining with the baseline file. If the
baseline file is missing, the verdict explicitly says INCOMPLETE rather
than silently approving.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "gpu_trainer_v11" / "reports"
VERDICT_PATH = REPO_ROOT / ".local" / "tasks" / "v5-static-postmortem-verdict.md"


def _load_walkforwards():
    return [json.loads(p.read_text()) for p in sorted(REPORT_DIR.glob("walkforward_rule_*.json"))]


def _load_diversifications():
    return [json.loads(p.read_text()) for p in sorted(REPORT_DIR.glob("diversification_*.json"))]


def _load_baseline() -> dict | None:
    p = REPORT_DIR / "xgb_baseline.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _baseline_avg_pf(baseline: dict | None) -> float | None:
    """Extract a single 'avg PF' number from the XGBoost Phase 1 baseline JSON.
    Honest_walkforward writes per-horizon JSON with `avg_pf`; we accept either
    a top-level `avg_pf` (single horizon) or a list `horizons` of per-horizon dicts."""
    if baseline is None:
        return None
    if "avg_pf" in baseline:
        return float(baseline["avg_pf"])
    if "horizons" in baseline and baseline["horizons"]:
        # take the BEST baseline avg_pf — V11 must beat the strongest baseline horizon
        return max(float(h.get("avg_pf", 0.0)) for h in baseline["horizons"])
    return None


def _format_section(walks: list[dict], divs: list[dict], baseline: dict | None) -> str:
    base_pf = _baseline_avg_pf(baseline)
    lines = []
    lines.append("\n## V11 — Honest Walk-Forward Verdict\n")
    lines.append(f"_Generated {datetime.utcnow().isoformat()} on V11 green-field brain "
                 f"(causal Transformer meta-classifier on dollar bars, two specialists, "
                 f"6-fold walk-forward, bagged ensemble N=5, three-way split: "
                 f"train→val→cal with horizon-purge at every boundary, Mondrian conformal "
                 f"per regime bucket). All metrics net of 6 bps round-trip slippage. "
                 f"Single-shot evaluation._\n")
    if base_pf is not None:
        lines.append(f"_XGBoost Phase-1 baseline avg PF for beat-check: **{base_pf:.2f}**_\n")
    else:
        lines.append("_**INCOMPLETE — XGBoost Phase-1 baseline file missing at "
                     "reports/xgb_baseline.json. PASS verdict cannot be issued without it.**_\n")

    any_full_pass = False
    for rep in walks:
        floor_pass = bool(rep.get("passed"))
        beats_baseline = (base_pf is not None) and (float(rep.get("avg_pf", 0.0)) > base_pf)
        full_pass = floor_pass and beats_baseline
        any_full_pass = any_full_pass or full_pass
        if base_pf is None:
            verdict = "**INCOMPLETE (no baseline)**"
        elif full_pass:
            verdict = "**PASS**"
        elif floor_pass and not beats_baseline:
            verdict = f"**FAIL (does not beat baseline PF {base_pf:.2f})**"
        else:
            verdict = "**FAIL**"
        lines.append(f"\n### Rule `{rep.get('rule')}` h={rep.get('horizon_bars')} bars on {rep.get('symbol')} — {verdict}\n")
        if not floor_pass:
            lines.append(f"_Floor failure reasons: {'; '.join(rep.get('fail_reasons', []))}_\n")
        lines.append("\n| Fold | Train | Test | n_test_seq | n_trades | WR | exp R | PF | maxDD R | adv AUC |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for f in rep.get("folds", []):
            lines.append(
                f"| {f.get('fold_num')} | {f.get('train_start')}→{f.get('train_end')} | "
                f"{f.get('test_start')}→{f.get('test_end')} | {f.get('n_test_eligible')} | "
                f"{f.get('n_trades')} | {f.get('win_rate', 0):.3f} | "
                f"{f.get('expectancy_R', 0):+.3f} | {f.get('pf', 0):.2f} | "
                f"{f.get('max_dd_R', 0):.2f} | {f.get('adversarial_auc', float('nan')):.3f} |"
            )
        lines.append(f"\n_Avg PF {rep.get('avg_pf', 0):.2f}  min PF {rep.get('min_pf', 0):.2f}  "
                     f"avg trades {rep.get('avg_trades', 0):.0f}  "
                     f"baseline beat: {'yes' if beats_baseline else 'no'}_\n")

    if divs:
        lines.append("\n### Diversification probe (post-2024, non-pool symbols)\n")
        lines.append("\n| Rule | Symbol | n_eligible | n_trades | PF | exp R | win rate |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for d in divs:
            for r in d.get("rows", []):
                lines.append(
                    f"| {d.get('rule')} | {r.get('symbol')} | {r.get('n_eligible')} | "
                    f"{r.get('n_trades')} | {r.get('pf', 0):.2f} | "
                    f"{r.get('expectancy_R', 0):+.3f} | {r.get('win_rate', 0):.3f} |"
                )

    lines.append("\n### Conclusion\n")
    if base_pf is None:
        lines.append("Verdict INCOMPLETE — the XGBoost Phase-1 baseline must be regenerated "
                     "and copied to `gpu_trainer_v11/reports/xgb_baseline.json` before any "
                     "specialist can be declared a PASS. The contract requires beating it.\n")
    elif any_full_pass:
        passed = [f"{r.get('rule')}@h{r.get('horizon_bars')}" for r in walks
                  if r.get('passed') and float(r.get('avg_pf', 0.0)) > base_pf]
        lines.append(f"At least one specialist PASSED the locked stop criterion AND beat the "
                     f"XGBoost baseline ({', '.join(passed)}). Wire passing specialist(s) into "
                     f"the live signal dashboard. Anti-tuning policy holds: do NOT modify "
                     f"hyperparameters now that results are known.\n")
    else:
        lines.append("All specialists failed the locked stop criterion (floor and/or baseline-beat). "
                     "V11 dies clean. Next move: write a post-mortem follow-up task — "
                     "do NOT rerun training with tweaked hyperparameters.\n")
    return "\n".join(lines) + "\n"


def main():
    walks = _load_walkforwards()
    divs = _load_diversifications()
    baseline = _load_baseline()
    if not walks:
        print("No walk-forward reports found. Run train_walkforward first.")
        return
    section = _format_section(walks, divs, baseline)
    if VERDICT_PATH.exists():
        with open(VERDICT_PATH, "a") as f:
            f.write(section)
        print(f"Appended V11 verdict to {VERDICT_PATH}")
    else:
        VERDICT_PATH.parent.mkdir(parents=True, exist_ok=True)
        VERDICT_PATH.write_text("# V5 Static Post-Mortem — Verdict\n" + section)
        print(f"Wrote new verdict file at {VERDICT_PATH}")


if __name__ == "__main__":
    main()
