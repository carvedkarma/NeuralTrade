"""
Diversification probe runner.

Note: this script depends on bagged models + per-fold conformal calibrators
that were created during walk-forward training. We persist the LAST fold's
ensemble to disk during walk-forward (see future enhancement), and reload
it here. For now this script wires the pieces together and is intended to
be run after a future patch that pickles the last-fold artifacts.

Until the persistence patch lands, the run_walk_forward harness can be
invoked with --keep-last-fold to keep the trained models in memory and
this script's logic can be invoked from a notebook.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "gpu_trainer_v11" / "reports"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=["A", "B"], required=True)
    ap.add_argument("--horizon", type=int, default=32)
    args = ap.parse_args()
    print("Diversification probe requires last-fold artifacts to be persisted.")
    print("This script is a stub — the v0 path is to call probe_symbols(...)")
    print("from a Python REPL after running train_walkforward.")
    out = {"rule": args.rule, "horizon": args.horizon, "rows": []}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"diversification_{args.rule}_h{args.horizon}.json").write_text(
        json.dumps(out, indent=2)
    )


if __name__ == "__main__":
    main()
