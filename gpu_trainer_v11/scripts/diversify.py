"""
Per-symbol diversification probe.

Loads the last fold's persisted artifacts (`reports/last_fold_<rule>_h<h>.pt`,
written by the walk-forward harness) and scores them on the post-2024
windows of every symbol NOT in the training pool. Same conformal
thresholds, same feature columns, no retraining, no threshold re-search.

Outputs:
    reports/diversification_<rule>_h<h>.json
    reports/diversification.md  (combined across all rule/horizon files)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from gpu_trainer_v11.eval.diversification import probe_symbols
from gpu_trainer_v11.models.causal_transformer import CausalTransformer, V11ModelConfig
from gpu_trainer_v11.models.conformal import MondrianCalibrator

REPO_ROOT = Path(__file__).resolve().parents[2]
DOLLAR_DIR = REPO_ROOT / "gpu_trainer_v11" / "data_cache_dollar"
REPORT_DIR = REPO_ROOT / "gpu_trainer_v11" / "reports"

TRAINING_POOL = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
START_2024_MS = int(pd.Timestamp("2024-01-01", tz="UTC").value // 10**6)


def _all_symbols() -> list[str]:
    return sorted(p.stem.replace("_dollar", "") for p in DOLLAR_DIR.glob("*_dollar.parquet"))


def _load_artifacts(rule: str, horizon: int) -> dict:
    p = REPORT_DIR / f"last_fold_{rule}_h{horizon}.pt"
    if not p.exists():
        raise SystemExit(
            f"FATAL: last-fold artifact missing at {p}. "
            f"Run scripts/train_walkforward.py --rule {rule} --horizon {horizon} first.")
    return torch.load(p, map_location="cpu", weights_only=False)


def _models_from_artifact(blob: dict) -> list[CausalTransformer]:
    cfg_d = blob["cfg"]
    cfg = V11ModelConfig(n_features=cfg_d["n_features"], seq_len=cfg_d["seq_len"])
    models = []
    for sd in blob["model_state_dicts"]:
        m = CausalTransformer(cfg)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    return models


def _conformal_from_artifact(blob: dict) -> MondrianCalibrator:
    cal = MondrianCalibrator()
    cal.thresholds = {int(k): v for k, v in blob["conformal_thresholds"].items()}
    cal.diag = {int(k): v for k, v in blob.get("conformal_diag", {}).items()}
    return cal


def _maybe_load_btc() -> pd.DataFrame | None:
    p = DOLLAR_DIR / "BTCUSDT_dollar.parquet"
    return pd.read_parquet(p) if p.exists() else None


def run_one(rule: str, horizon: int) -> dict:
    blob = _load_artifacts(rule, horizon)
    feature_cols = blob["feature_cols"]
    keep_mask = blob.get("keep_mask")
    if keep_mask is None:
        # Backward-compat fallback: if an older artifact lacks the gate,
        # default to all-features-on (matches old behavior). New artifacts
        # always carry the per-fold transfer-entropy gate.
        keep_mask = [True] * len(feature_cols)
    import numpy as _np
    keep_mask_arr = _np.asarray(keep_mask, dtype=bool)
    models = _models_from_artifact(blob)
    cal = _conformal_from_artifact(blob)
    btc = _maybe_load_btc()

    target_symbols = [s for s in _all_symbols() if s not in TRAINING_POOL]
    print(f"  rule={rule} h={horizon}: probing {len(target_symbols)} non-pool symbols "
          f"from 2024-01-01  (gate keeps {int(keep_mask_arr.sum())}/"
          f"{len(feature_cols)} feature cols)")
    rows = probe_symbols(
        symbols=target_symbols, rule=rule, horizon_bars=horizon,
        feature_cols=feature_cols, keep_mask=keep_mask_arr,
        bagged_models=models, conformal=cal,
        btc_bars=btc, seq_len=blob["cfg"]["seq_len"], start_ts_ms=START_2024_MS,
    )
    out = {
        "rule": rule, "horizon": horizon,
        "from_artifact": str(REPORT_DIR / f"last_fold_{rule}_h{horizon}.pt"),
        "n_features_kept": int(keep_mask_arr.sum()),
        "rows": [asdict(r) for r in rows],
    }
    out_path = REPORT_DIR / f"diversification_{rule}_h{horizon}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  wrote {out_path}  ({len(rows)} rows)")
    return out


def write_combined_md():
    rows = []
    for p in sorted(REPORT_DIR.glob("diversification_*.json")):
        d = json.loads(p.read_text())
        for r in d.get("rows", []):
            rows.append({**r, "rule": d["rule"], "horizon": d["horizon"]})
    if not rows:
        return
    md = ["# V11 — Per-Symbol Diversification Probe\n",
          "Read-only application of the last-fold bagged ensemble + conformal thresholds "
          "to non-pool symbols' post-2024 windows.\n",
          "| Rule | H | Symbol | n_eligible | n_trades | PF | exp R | win rate |",
          "|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(
            f"| {r['rule']} | {r['horizon']} | {r['symbol']} | {r['n_eligible']} | "
            f"{r['n_trades']} | {r['pf']:.2f} | {r['expectancy_R']:+.3f} | {r['win_rate']:.3f} |"
        )
    (REPORT_DIR / "diversification.md").write_text("\n".join(md) + "\n")
    print(f"  wrote {REPORT_DIR / 'diversification.md'}")


def main():
    ap = argparse.ArgumentParser(description="Per-symbol diversification probe.")
    ap.add_argument("--rule", choices=["A", "B", "all"], default="all")
    ap.add_argument("--horizon", type=int, default=None,
                    help="If omitted, probes every artifact found.")
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.rule == "all" and args.horizon is None:
        artifacts = sorted(REPORT_DIR.glob("last_fold_*.pt"))
        if not artifacts:
            raise SystemExit("FATAL: no last-fold artifacts found. Run train_walkforward first.")
        for p in artifacts:
            stem = p.stem  # last_fold_A_h32
            parts = stem.split("_")
            rule = parts[2]; horizon = int(parts[3].lstrip("h"))
            run_one(rule, horizon)
    else:
        rules = ["A", "B"] if args.rule == "all" else [args.rule]
        if args.horizon is None:
            raise SystemExit("--horizon is required unless probing all artifacts.")
        for r in rules:
            run_one(r, args.horizon)
    write_combined_md()


if __name__ == "__main__":
    main()
