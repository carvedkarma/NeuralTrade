"""
Train + walk-forward one specialist on BTCUSDT (training base symbol).

Pre-flight must have passed for `--rule` before running this.

Hyperparameters are LOCKED by the README contract and intentionally
NOT exposed as CLI flags here. Editing them post-hoc to make a fold
pass is a contract violation. Bug fixes only.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from gpu_trainer_v11.eval.walkforward import run_walk_forward

REPO_ROOT = Path(__file__).resolve().parents[2]
DOLLAR_DIR = REPO_ROOT / "gpu_trainer_v11" / "data_cache_dollar"
REPORT_DIR = REPO_ROOT / "gpu_trainer_v11" / "reports"

# LOCKED hyperparameters — do not edit to chase a passing fold.
LOCKED_N_BAG = 5
LOCKED_PRETRAIN_EPOCHS = 20
LOCKED_FINETUNE_EPOCHS = 30
LOCKED_TOP_K_FEATURES = 64


def _require_preflight_pass(rule: str):
    """Refuse to train if pre-flight did not record a PASS for this rule."""
    p = REPORT_DIR / "preflight.json"
    if not p.exists():
        raise SystemExit(
            f"FATAL: preflight report missing at {p}. Run scripts/preflight.py first.")
    rep = json.loads(p.read_text())
    decision = rep.get("decisions", {}).get(rule, {})
    if not decision.get("pass"):
        raise SystemExit(
            f"FATAL: pre-flight FAILED for rule {rule} ({decision}). "
            f"Per the locked contract, do NOT proceed to training.")


def main():
    ap = argparse.ArgumentParser(
        description="Walk-forward one specialist. Hyperparameters are LOCKED.")
    ap.add_argument("--rule", choices=["A", "B"], required=True)
    ap.add_argument("--horizon", type=int, default=32,
                    help="Locked horizons set: {16, 32, 96}.")
    ap.add_argument("--symbol", default="BTCUSDT",
                    help="Training base symbol; pool is BTC/ETH/SOL but per-fold "
                         "training is single-symbol per the as-built contract.")
    ap.add_argument("--allow-without-preflight", action="store_true",
                    help="ESCAPE HATCH for code-review smoke runs only. "
                         "Production runs MUST satisfy pre-flight.")
    args = ap.parse_args()

    if args.horizon not in (16, 32, 96):
        raise SystemExit(f"horizon must be in {{16, 32, 96}} (locked); got {args.horizon}")
    if not args.allow_without_preflight:
        _require_preflight_pass(args.rule)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    bars = pd.read_parquet(DOLLAR_DIR / f"{args.symbol}_dollar.parquet")
    btc_path = DOLLAR_DIR / "BTCUSDT_dollar.parquet"
    btc = pd.read_parquet(btc_path) if btc_path.exists() else None

    pretrain_ckpt = REPORT_DIR / "pretrained_trunk.pt"
    if not args.allow_without_preflight and not pretrain_ckpt.exists():
        raise SystemExit(
            f"FATAL: pooled pretrained trunk missing at {pretrain_ckpt}. "
            f"Run scripts/pretrain.py first (locked contract requires pooled pretrain).")

    rep = run_walk_forward(
        bars, btc, args.symbol, args.rule, args.horizon,
        n_bag=LOCKED_N_BAG,
        pretrain_epochs=LOCKED_PRETRAIN_EPOCHS,
        finetune_epochs=LOCKED_FINETUNE_EPOCHS,
        top_k_features=LOCKED_TOP_K_FEATURES,
        pretrain_checkpoint=pretrain_ckpt if pretrain_ckpt.exists() else None,
    )

    out = {
        "rule": rep.rule,
        "horizon_bars": rep.horizon_bars,
        "symbol": args.symbol,
        "passed": rep.passed,
        "fail_reasons": rep.fail_reasons,
        "avg_pf": rep.avg_pf,
        "min_pf": rep.min_pf,
        "avg_trades": rep.avg_trades,
        "folds": [asdict(f) for f in rep.folds],
        "pretrain_source": rep.pretrain_source,
        "last_fold_artifact": rep.last_fold_artifact,
        "locked_hyperparams": {
            "n_bag": LOCKED_N_BAG,
            "pretrain_epochs": LOCKED_PRETRAIN_EPOCHS,
            "finetune_epochs": LOCKED_FINETUNE_EPOCHS,
            "top_k_features": LOCKED_TOP_K_FEATURES,
        },
        "generated_at": datetime.utcnow().isoformat(),
    }
    out_path = REPORT_DIR / f"walkforward_rule_{rep.rule}_h{rep.horizon_bars}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n{'PASS' if rep.passed else 'FAIL'}  avg PF {rep.avg_pf:.2f}  "
          f"min PF {rep.min_pf:.2f}  avg trades {rep.avg_trades:.0f}")
    print(f"Wrote {out_path}")

    _refresh_canonical_outputs()


def _refresh_canonical_outputs():
    """Aggregate every per-rule walkforward_rule_*.json into the canonical
    `reports/walkforward.json`, and write `reports/drift.md` summarising
    per-fold adversarial-validation AUC across all rule × horizon runs."""
    rule_files = sorted(REPORT_DIR.glob("walkforward_rule_*.json"))
    if not rule_files:
        return
    rules = {}
    for p in rule_files:
        d = json.loads(p.read_text())
        rules.setdefault(d["rule"], []).append(d)
    canonical = {
        "generated_at": datetime.utcnow().isoformat(),
        "rules": rules,
        "aggregate_passed": all(r["passed"] for runs in rules.values() for r in runs),
    }
    (REPORT_DIR / "walkforward.json").write_text(json.dumps(canonical, indent=2, default=str))

    drift_lines = ["# V11 — Adversarial-Validation Drift Report\n",
                   "Per-fold AUC of a classifier asked to distinguish train-window "
                   "feature distributions from test-window feature distributions. "
                   "Values near 0.5 ⇒ no detectable drift; values ≫ 0.5 ⇒ regime shift "
                   "during fold (informational; does not gate verdict).\n",
                   "| Rule | H | Fold | Train start | Test start | adv_auc | n_trades | PF |",
                   "|---|---:|---:|---|---|---:|---:|---:|"]
    for rule, runs in sorted(rules.items()):
        for d in runs:
            for f in d["folds"]:
                drift_lines.append(
                    f"| {rule} | {d['horizon_bars']} | {f['fold_num']} | "
                    f"{f['train_start']} | {f['test_start']} | "
                    f"{f.get('adversarial_auc', float('nan')):.3f} | "
                    f"{f.get('n_trades', 0)} | {f.get('pf', 0.0):.2f} |"
                )
    (REPORT_DIR / "drift.md").write_text("\n".join(drift_lines) + "\n")
    print(f"Wrote {REPORT_DIR / 'walkforward.json'} and {REPORT_DIR / 'drift.md'}")


if __name__ == "__main__":
    main()
