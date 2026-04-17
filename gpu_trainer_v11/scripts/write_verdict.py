"""
Append the V11 walk-forward verdict block to v5-static-postmortem-verdict.md.

Reads:
    reports/walkforward_rule_A_h*.json   (V11 specialist A)
    reports/walkforward_rule_B_h*.json   (V11 specialist B)
    reports/diversification_*.json       (optional)
    reports/xgb_baseline.json            (REQUIRED — produced by
                                          gpu_trainer_v11/eval/honest_walkforward.py)
    reports/v5_baseline.json             (REQUIRED — V5 forward summary;
                                          {"avg_pf": <float>, "label": <string>,
                                           "source": <string>}. Lift from the
                                          last passing fold of the V5 forward
                                          report cited in
                                          .local/tasks/v5-static-postmortem-verdict.md)

Per the locked contract (README §"Stop criterion"), V11 PASSES only if:
    1. avg PF >= 1.3 across 6 test folds
    2. all per-fold trade counts >= 500
    3. no fold PF < 1.0
    4. AND beats the XGBoost Phase-1 baseline avg PF
    5. AND beats the V5 forward report avg PF
        (any specialist that beats only one baseline is FAIL.)

Side-by-side rows in the appended verdict block show V11 / XGBoost-baseline /
V5-baseline so the beat-check is auditable. If either baseline file is
missing the verdict says INCOMPLETE rather than silently approving.
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


def _load_baseline(name: str) -> dict | None:
    p = REPORT_DIR / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _baseline_avg_pf(baseline: dict | None) -> float | None:
    """Extract a single 'avg PF' number from a baseline JSON.
    Accepts either a top-level `avg_pf` or a list `horizons` of per-horizon
    dicts (then takes the BEST so V11 must beat the strongest)."""
    if baseline is None:
        return None
    if "avg_pf" in baseline:
        return float(baseline["avg_pf"])
    if "horizons" in baseline and baseline["horizons"]:
        return max(float(h.get("avg_pf", 0.0)) for h in baseline["horizons"])
    return None


def _format_section(walks: list[dict], divs: list[dict],
                    xgb_baseline: dict | None, v5_baseline: dict | None) -> str:
    xgb_pf = _baseline_avg_pf(xgb_baseline)
    v5_pf = _baseline_avg_pf(v5_baseline)
    v5_label = (v5_baseline or {}).get("label", "V5 forward")
    xgb_label = (xgb_baseline or {}).get("label", "XGBoost Phase-1")
    lines = []
    lines.append("\n## V11 — Honest Walk-Forward Verdict\n")
    lines.append(f"_Generated {datetime.utcnow().isoformat()} on V11 green-field brain "
                 f"(causal Transformer meta-classifier on dollar bars, two specialists, "
                 f"6-fold walk-forward, bagged ensemble N=5, three-way split: "
                 f"train→val→cal with horizon-purge at every boundary, Mondrian conformal "
                 f"per regime bucket). All metrics net of 6 bps round-trip slippage. "
                 f"Single-shot evaluation._\n")
    lines.append(f"_Beat-check baselines:  {xgb_label} avg PF = "
                 f"{'**' + format(xgb_pf, '.2f') + '**' if xgb_pf is not None else '**MISSING**'}  ;  "
                 f"{v5_label} avg PF = "
                 f"{'**' + format(v5_pf, '.2f') + '**' if v5_pf is not None else '**MISSING**'}_\n")
    if xgb_pf is None:
        lines.append("_**INCOMPLETE — reports/xgb_baseline.json missing.**_\n")
    if v5_pf is None:
        lines.append("_**INCOMPLETE — reports/v5_baseline.json missing.**_\n")

    any_full_pass = False
    # Side-by-side comparison header
    lines.append("\n### Side-by-side beat-check\n")
    lines.append("\n| Rule | H | Symbol | V11 avg PF | V11 min PF | V11 avg trades | "
                 f"{xgb_label} PF | beats XGB | {v5_label} PF | beats V5 | floor | verdict |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---|---:|---|---|---|")
    for rep in walks:
        floor_pass = bool(rep.get("passed"))
        v11_pf = float(rep.get("avg_pf", 0.0))
        beats_xgb = (xgb_pf is not None) and (v11_pf > xgb_pf)
        beats_v5  = (v5_pf  is not None) and (v11_pf > v5_pf)
        full_pass = floor_pass and beats_xgb and beats_v5
        any_full_pass = any_full_pass or full_pass
        if xgb_pf is None or v5_pf is None:
            verdict = "INCOMPLETE"
        elif full_pass:
            verdict = "**PASS**"
        else:
            verdict = "FAIL"
        lines.append(
            f"| {rep.get('rule')} | {rep.get('horizon_bars')} | {rep.get('symbol')} | "
            f"{v11_pf:.2f} | {rep.get('min_pf', 0):.2f} | {rep.get('avg_trades', 0):.0f} | "
            f"{('%.2f' % xgb_pf) if xgb_pf is not None else '—'} | "
            f"{'yes' if beats_xgb else 'no' if xgb_pf is not None else '—'} | "
            f"{('%.2f' % v5_pf) if v5_pf is not None else '—'} | "
            f"{'yes' if beats_v5 else 'no' if v5_pf is not None else '—'} | "
            f"{'pass' if floor_pass else 'fail'} | {verdict} |"
        )

    for rep in walks:
        floor_pass = bool(rep.get("passed"))
        v11_pf = float(rep.get("avg_pf", 0.0))
        beats_xgb = (xgb_pf is not None) and (v11_pf > xgb_pf)
        beats_v5  = (v5_pf  is not None) and (v11_pf > v5_pf)
        full_pass = floor_pass and beats_xgb and beats_v5
        if xgb_pf is None or v5_pf is None:
            verdict = "**INCOMPLETE (a baseline is missing)**"
        elif full_pass:
            verdict = "**PASS**"
        elif floor_pass and not beats_xgb:
            verdict = f"**FAIL (does not beat XGBoost baseline PF {xgb_pf:.2f})**"
        elif floor_pass and not beats_v5:
            verdict = f"**FAIL (does not beat V5 baseline PF {v5_pf:.2f})**"
        else:
            verdict = "**FAIL**"
        lines.append(f"\n### Rule `{rep.get('rule')}` h={rep.get('horizon_bars')} bars on "
                     f"{rep.get('symbol')} — {verdict}\n")
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
        lines.append(f"\n_Avg PF {v11_pf:.2f}  min PF {rep.get('min_pf', 0):.2f}  "
                     f"avg trades {rep.get('avg_trades', 0):.0f}  "
                     f"beats XGB: {'yes' if beats_xgb else 'no'}  "
                     f"beats V5: {'yes' if beats_v5 else 'no'}_\n")

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
    if xgb_pf is None or v5_pf is None:
        lines.append("Verdict INCOMPLETE — both baselines (XGBoost Phase-1 and V5 forward) "
                     "must be present in reports/ before any specialist can be declared a "
                     "PASS. The contract requires beating BOTH.\n")
    elif any_full_pass:
        passed = [f"{r.get('rule')}@h{r.get('horizon_bars')}" for r in walks
                  if r.get('passed')
                  and float(r.get('avg_pf', 0.0)) > xgb_pf
                  and float(r.get('avg_pf', 0.0)) > v5_pf]
        lines.append(f"At least one specialist PASSED the locked stop criterion AND beat BOTH "
                     f"baselines ({', '.join(passed)}). Wire passing specialist(s) into the "
                     f"live signal dashboard. Anti-tuning policy holds: do NOT modify "
                     f"hyperparameters now that results are known.\n")
    else:
        lines.append("All specialists failed the locked stop criterion (floor and/or "
                     "either baseline-beat). V11 dies clean. Next move: write a post-mortem "
                     "follow-up task — do NOT rerun training with tweaked hyperparameters.\n")
    return "\n".join(lines) + "\n"


def main():
    walks = _load_walkforwards()
    divs = _load_diversifications()
    xgb_baseline = _load_baseline("xgb_baseline.json")
    v5_baseline = _load_baseline("v5_baseline.json")
    if not walks:
        print("No walk-forward reports found. Run train_walkforward first.")
        return
    section = _format_section(walks, divs, xgb_baseline, v5_baseline)
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
