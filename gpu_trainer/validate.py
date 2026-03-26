"""validate.py — V5 fast validation harness.

Modes
-----
unit_audit    : Run the existing pytest precision-audit suite only (no torch).
smoke_wf      : 2-symbol, 2-fold walk-forward smoke test (BTCUSDT + ETHUSDT).
canary_wf     : 4-symbol, 3-fold walk-forward (BTC/ETH/SOL/BNB).
candidate_diff: Full 20-symbol WF on last N folds, capturing per-candidate gate
                breakdown to show oracle R available vs actually taken.
compare       : Load two validate_runs JSON files and diff their metric tables.

Usage
-----
python validate.py unit_audit
python validate.py smoke_wf
python validate.py canary_wf [--folds 3]
python validate.py candidate_diff [--folds 2]
python validate.py compare runs/smoke_wf_A.json runs/smoke_wf_B.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validate")

RUNS_DIR = Path(__file__).parent / "validate_runs"
DATA_DIR = Path(__file__).parent / "data_cache"

SMOKE_SYMBOLS   = ["BTCUSDT", "ETHUSDT"]
CANARY_SYMBOLS  = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
ALL_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "AVAXUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT", "LTCUSDT",
    "NEARUSDT", "PEPEUSDT", "SUIUSDT", "AAVEUSDT", "ARBUSDT",
    "DOTUSDT", "MATICUSDT", "FILUSDT", "APTUSDT", "OPUSDT",
]


# ─────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────

def _compute_metrics(records: List[Dict]) -> Dict[str, Any]:
    """Compute trade-level metrics from a list of taken-trade records.

    Each record must have at least: oracle_r (float), side (int 1/-1),
    symbol (str), blocked_by (str), taken (bool).
    """
    taken = [r for r in records if r.get("taken")]
    blocked = [r for r in records if not r.get("taken")]

    n_trades = len(taken)
    if n_trades == 0:
        return {
            "trades": 0,
            "oracle_r_total": 0.0,
            "expectancy": 0.0,
            "win_rate": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "long_expectancy": 0.0,
            "short_expectancy": 0.0,
            "top_decile_avg_r": 0.0,
            "bottom_decile_avg_r": 0.0,
            "monotonic_score_r": "N/A",
            "gate_block_pct": {},
            "n_candidates": len(records),
            "n_blocked": len(blocked),
        }

    rs = [r["oracle_r"] for r in taken]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]

    n_total_cands = len(records)
    long_rs  = [r["oracle_r"] for r in taken if r.get("side") == 1]
    short_rs = [r["oracle_r"] for r in taken if r.get("side") == -1]

    sorted_by_score = sorted(taken, key=lambda r: r.get("score_work", r.get("score", 0)), reverse=True)
    decile = max(1, n_trades // 10)
    top_decile  = [r["oracle_r"] for r in sorted_by_score[:decile]]
    bot_decile  = [r["oracle_r"] for r in sorted_by_score[-decile:]]

    monotonic = _check_monotonic_score_r(taken)

    gate_counts: Dict[str, int] = defaultdict(int)
    for r in blocked:
        gate = r.get("blocked_by") or "unknown"
        gate_counts[gate] += 1
    n_blocked = len(blocked)
    gate_pct = {g: round(c / max(n_blocked, 1) * 100, 1) for g, c in sorted(gate_counts.items(), key=lambda x: -x[1])}

    return {
        "trades": n_trades,
        "oracle_r_total": round(sum(rs), 3),
        "expectancy": round(sum(rs) / n_trades, 4),
        "win_rate": round(len(wins) / n_trades, 4),
        "avg_win_r": round(sum(wins) / max(len(wins), 1), 4),
        "avg_loss_r": round(sum(losses) / max(len(losses), 1), 4),
        "long_trades": len(long_rs),
        "short_trades": len(short_rs),
        "long_expectancy": round(sum(long_rs) / max(len(long_rs), 1), 4),
        "short_expectancy": round(sum(short_rs) / max(len(short_rs), 1), 4),
        "top_decile_avg_r": round(sum(top_decile) / max(len(top_decile), 1), 4),
        "bottom_decile_avg_r": round(sum(bot_decile) / max(len(bot_decile), 1), 4),
        "monotonic_score_r": "PASS" if monotonic else "FAIL",
        "gate_block_pct": gate_pct,
        "n_candidates": n_total_cands,
        "n_blocked": n_blocked,
    }


def _check_monotonic_score_r(taken: List[Dict], n_buckets: int = 5) -> bool:
    """Compute average oracle_r per score quintile and check monotonic direction."""
    if len(taken) < n_buckets * 2:
        return True
    sorted_t = sorted(taken, key=lambda r: r.get("score_work", r.get("score", 0)))
    bucket_size = len(sorted_t) // n_buckets
    bucket_means = []
    for i in range(n_buckets):
        bucket = sorted_t[i * bucket_size:(i + 1) * bucket_size]
        if bucket:
            bucket_means.append(sum(r["oracle_r"] for r in bucket) / len(bucket))
    increasing = all(bucket_means[i] <= bucket_means[i + 1] for i in range(len(bucket_means) - 1))
    return increasing


# ─────────────────────────────────────────────
# ASCII table printer
# ─────────────────────────────────────────────

def _print_table(title: str, metrics: Dict[str, Any]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)
    core_keys = [
        ("trades",              "Trades"),
        ("n_candidates",        "Candidates"),
        ("n_blocked",           "Blocked"),
        ("oracle_r_total",      "Total oracle R"),
        ("expectancy",          "Expectancy (R/trade)"),
        ("win_rate",            "Win rate"),
        ("avg_win_r",           "Avg win (R)"),
        ("avg_loss_r",          "Avg loss (R)"),
        ("long_trades",         "Long trades"),
        ("short_trades",        "Short trades"),
        ("long_expectancy",     "Long E[R]"),
        ("short_expectancy",    "Short E[R]"),
        ("top_decile_avg_r",    "Top-decile avg R"),
        ("bottom_decile_avg_r", "Bottom-decile avg R"),
        ("monotonic_score_r",   "Monotonic score→R"),
    ]
    for key, label in core_keys:
        val = metrics.get(key, "—")
        if isinstance(val, float):
            val = f"{val:.4f}"
        print(f"  {label:<28} {val}")

    gate_pct = metrics.get("gate_block_pct", {})
    if gate_pct:
        print(f"\n  {'Gate block breakdown':}")
        for gate, pct in list(gate_pct.items())[:10]:
            print(f"    {gate:<24} {pct:>5.1f}%")
    print()


def _compare_tables(label_a: str, label_b: str,
                    m_a: Dict[str, Any], m_b: Dict[str, Any]) -> None:
    numeric_keys = [
        ("trades",              "Trades"),
        ("oracle_r_total",      "Total oracle R"),
        ("expectancy",          "Expectancy (R/trade)"),
        ("win_rate",            "Win rate"),
        ("avg_win_r",           "Avg win (R)"),
        ("avg_loss_r",          "Avg loss (R)"),
        ("top_decile_avg_r",    "Top-decile avg R"),
        ("bottom_decile_avg_r", "Bottom-decile avg R"),
    ]
    print(f"\n{'=' * 72}")
    print(f"  COMPARE  {label_a}  vs  {label_b}")
    print(f"{'=' * 72}")
    print(f"  {'Metric':<28} {'A':>12} {'B':>12} {'Delta':>12}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*12}")
    for key, label in numeric_keys:
        va = m_a.get(key, 0) or 0
        vb = m_b.get(key, 0) or 0
        delta = vb - va if isinstance(va, (int, float)) else "—"
        arrow = ("↑" if delta > 0 else "↓") if isinstance(delta, float) else ""
        print(f"  {label:<28} {str(va):>12} {str(vb):>12} {f'{delta:+.4f}{arrow}':>12}")
    print(f"\n  Monotonic A: {m_a.get('monotonic_score_r', '—')}  "
          f"Monotonic B: {m_b.get('monotonic_score_r', '—')}")
    print()


# ─────────────────────────────────────────────
# Save / load JSON results
# ─────────────────────────────────────────────

def _save_run(mode: str, payload: Dict) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RUNS_DIR / f"{mode}_{ts}.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log.info("[validate] Results saved → %s", out)
    return out


def _load_run(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Walk-forward runner helper
# ─────────────────────────────────────────────

def _run_wf(
    symbols: List[str],
    epochs: int,
    batch_size: int,
    lr: float,
    train_months: int,
    test_months: int,
    max_folds: Optional[int],
    seed: int = 42,
    device: str = "cpu",
) -> List[Dict]:
    """Run walk-forward and collect all candidate records via candidate_logger."""
    try:
        import torch  # noqa: F401
    except ImportError:
        log.error("[validate] torch not available — WF modes require GPU machine.")
        sys.exit(1)

    try:
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        dev = "cpu"

    sys.path.insert(0, str(Path(__file__).parent))
    from train.v5_train import run_v5_walk_forward

    candidates: List[Dict] = []

    def _logger(rec: Dict) -> None:
        candidates.append(rec)

    torch.manual_seed(seed)

    run_v5_walk_forward(
        data_dir=DATA_DIR,
        device=dev,
        symbols=symbols,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        train_months=train_months,
        test_months=test_months,
        max_folds=max_folds,
        candidate_logger=_logger,
        slippage_base_bps=6.0,
        score_lambda=0.5,
        balanced_sampling=True,
        calibration_monitor=True,
    )

    return candidates


# ─────────────────────────────────────────────
# Mode: unit_audit
# ─────────────────────────────────────────────

def mode_unit_audit(args: argparse.Namespace) -> None:
    log.info("[validate] mode=unit_audit — running pytest precision audit suite")
    import subprocess
    tests_path = Path(__file__).parent / "tests" / "test_v5_precision_audit.py"
    if not tests_path.exists():
        log.error("Test file not found: %s", tests_path)
        sys.exit(1)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(tests_path), "-v", "--tb=short"],
        cwd=str(Path(__file__).parent),
    )
    if result.returncode != 0:
        log.error("[validate] unit_audit FAILED (exit=%d)", result.returncode)
        sys.exit(result.returncode)
    log.info("[validate] unit_audit PASSED")
    _save_run("unit_audit", {"mode": "unit_audit", "exit_code": 0, "status": "PASSED"})


# ─────────────────────────────────────────────
# Mode: smoke_wf
# ─────────────────────────────────────────────

def mode_smoke_wf(args: argparse.Namespace) -> None:
    log.info("[validate] mode=smoke_wf  symbols=%s  epochs=15  folds=2", SMOKE_SYMBOLS)
    t0 = time.time()
    candidates = _run_wf(
        symbols=SMOKE_SYMBOLS,
        epochs=15,
        batch_size=128,
        lr=3e-4,
        train_months=3,
        test_months=1,
        max_folds=2,
        seed=42,
    )
    elapsed = time.time() - t0
    metrics = _compute_metrics(candidates)
    _print_table("smoke_wf results", metrics)
    payload = {
        "mode": "smoke_wf",
        "elapsed_s": round(elapsed, 1),
        "symbols": SMOKE_SYMBOLS,
        "metrics": metrics,
        "n_candidate_records": len(candidates),
    }
    out = _save_run("smoke_wf", payload)
    print(f"Run saved: {out}")


# ─────────────────────────────────────────────
# Mode: canary_wf
# ─────────────────────────────────────────────

def mode_canary_wf(args: argparse.Namespace) -> None:
    folds = getattr(args, "folds", 3)
    log.info("[validate] mode=canary_wf  symbols=%s  epochs=30  folds=%d", CANARY_SYMBOLS, folds)
    t0 = time.time()
    candidates = _run_wf(
        symbols=CANARY_SYMBOLS,
        epochs=30,
        batch_size=128,
        lr=3e-4,
        train_months=6,
        test_months=1,
        max_folds=folds,
        seed=42,
    )
    elapsed = time.time() - t0
    metrics = _compute_metrics(candidates)
    _print_table("canary_wf results", metrics)
    payload = {
        "mode": "canary_wf",
        "elapsed_s": round(elapsed, 1),
        "symbols": CANARY_SYMBOLS,
        "max_folds": folds,
        "metrics": metrics,
        "n_candidate_records": len(candidates),
    }
    out = _save_run("canary_wf", payload)
    print(f"Run saved: {out}")


# ─────────────────────────────────────────────
# Mode: candidate_diff
# ─────────────────────────────────────────────

def mode_candidate_diff(args: argparse.Namespace) -> None:
    folds = getattr(args, "folds", 2)
    log.info("[validate] mode=candidate_diff  symbols=ALL20  folds=%d", folds)
    t0 = time.time()
    candidates = _run_wf(
        symbols=ALL_SYMBOLS,
        epochs=30,
        batch_size=128,
        lr=3e-4,
        train_months=6,
        test_months=1,
        max_folds=folds,
        seed=42,
    )
    elapsed = time.time() - t0

    taken = [r for r in candidates if r.get("taken")]
    blocked = [r for r in candidates if not r.get("taken")]
    all_r = [r["oracle_r"] for r in candidates]
    taken_r = [r["oracle_r"] for r in taken]

    oracle_total = sum(all_r)
    taken_total = sum(taken_r)
    capture_rate = taken_total / oracle_total if oracle_total != 0 else 0.0

    metrics = _compute_metrics(candidates)

    gate_oracle: Dict[str, float] = defaultdict(float)
    gate_count: Dict[str, int] = defaultdict(int)
    for r in blocked:
        g = r.get("blocked_by") or "unknown"
        gate_oracle[g] += r.get("oracle_r", 0)
        gate_count[g] += 1

    print(f"\n{'=' * 60}")
    print("  CANDIDATE DIFF — Gate oracle R left on the table")
    print('=' * 60)
    print(f"  All candidates : {len(candidates):5d}   oracle_R={oracle_total:.2f}")
    print(f"  Taken          : {len(taken):5d}   oracle_R={taken_total:.2f}")
    print(f"  Blocked        : {len(blocked):5d}   oracle_R={sum(r['oracle_r'] for r in blocked):.2f}")
    print(f"  Capture rate   : {capture_rate:.1%}")
    print(f"\n  {'Gate':<24} {'Blocked':>8} {'Oracle R':>10} {'Avg R':>10}")
    print(f"  {'-'*24} {'-'*8} {'-'*10} {'-'*10}")
    for gate, count in sorted(gate_count.items(), key=lambda x: -abs(gate_oracle[x[0]])):
        avg = gate_oracle[gate] / count if count else 0
        print(f"  {gate:<24} {count:>8} {gate_oracle[gate]:>10.3f} {avg:>10.4f}")
    print()

    _print_table("candidate_diff trade metrics", metrics)

    payload = {
        "mode": "candidate_diff",
        "elapsed_s": round(elapsed, 1),
        "symbols": "ALL20",
        "max_folds": folds,
        "metrics": metrics,
        "oracle_total_all": round(oracle_total, 3),
        "oracle_total_taken": round(taken_total, 3),
        "capture_rate": round(capture_rate, 4),
        "n_candidates": len(candidates),
        "gate_oracle_r": {g: round(v, 3) for g, v in gate_oracle.items()},
        "gate_block_count": dict(gate_count),
    }
    out = _save_run("candidate_diff", payload)
    print(f"Run saved: {out}")


# ─────────────────────────────────────────────
# Mode: compare
# ─────────────────────────────────────────────

def mode_compare(args: argparse.Namespace) -> None:
    path_a, path_b = args.files
    log.info("[validate] mode=compare  A=%s  B=%s", path_a, path_b)
    run_a = _load_run(path_a)
    run_b = _load_run(path_b)
    m_a = run_a.get("metrics", run_a)
    m_b = run_b.get("metrics", run_b)
    label_a = f"{run_a.get('mode', '?')} [{Path(path_a).stem}]"
    label_b = f"{run_b.get('mode', '?')} [{Path(path_b).stem}]"
    _compare_tables(label_a, label_b, m_a, m_b)

    diff_payload = {
        "mode": "compare",
        "file_a": path_a,
        "file_b": path_b,
        "metrics_a": m_a,
        "metrics_b": m_b,
    }
    out = _save_run("compare", diff_payload)
    print(f"Run saved: {out}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="V5 fast validation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    sub.add_parser("unit_audit", help="Run pytest precision audit suite (no torch needed)")

    sub.add_parser("smoke_wf", help="2-symbol, 2-fold walk-forward smoke test")

    canary = sub.add_parser("canary_wf", help="4-symbol, N-fold walk-forward")
    canary.add_argument("--folds", type=int, default=3, help="Number of folds (default: 3)")

    diff = sub.add_parser("candidate_diff", help="20-symbol WF gate oracle R breakdown")
    diff.add_argument("--folds", type=int, default=2, help="Number of folds (default: 2)")

    cmp = sub.add_parser("compare", help="Diff two validate_runs JSON files")
    cmp.add_argument("files", nargs=2, metavar="FILE", help="Two JSON run files to compare")

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dispatch: Dict[str, Callable] = {
        "unit_audit":     mode_unit_audit,
        "smoke_wf":       mode_smoke_wf,
        "canary_wf":      mode_canary_wf,
        "candidate_diff": mode_candidate_diff,
        "compare":        mode_compare,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
