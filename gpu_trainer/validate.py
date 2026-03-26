"""validate.py — V5 fast validation harness.

Modes
-----
unit_audit    : Pytest precision-audit suite + direct torch-free config checks.
smoke_wf      : 2-symbol, 2-fold walk-forward smoke test (BTC + ETH).
canary_wf     : 4-symbol, 3-fold walk-forward (BTC/ETH/SOL/BNB).
candidate_diff: Run WF and dump per-candidate CSV.  Optionally compare two CSVs
                to surface top-20 decision changes between runs.
compare       : Load two validate_runs JSON files and diff all metric tables
                including gate block %.

Usage
-----
python validate.py unit_audit
python validate.py smoke_wf
python validate.py canary_wf [--folds 3]
python validate.py candidate_diff [--folds 2] [--csv out.csv]
python validate.py candidate_diff --run baseline.csv [--vs new.csv]
python validate.py compare runs/smoke_wf_A.json runs/smoke_wf_B.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validate")

RUNS_DIR = Path(__file__).parent / "validate_runs"
DATA_DIR = Path(__file__).parent / "data_cache"

SMOKE_SYMBOLS  = ["BTCUSDT", "ETHUSDT"]
CANARY_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
ALL_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "AVAXUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT", "LTCUSDT",
    "NEARUSDT", "PEPEUSDT", "SUIUSDT", "AAVEUSDT", "ARBUSDT",
    "DOTUSDT", "MATICUSDT", "FILUSDT", "APTUSDT", "OPUSDT",
]

CSV_COLUMNS = [
    "bar_idx", "timestamp", "symbol", "side",
    "raw_score", "final_score", "threshold",
    "taken", "block_reason",
    "mu_R", "p_trade", "adx_val", "regime_label", "corr_blocked",
    "oracle_r",
]


# ─────────────────────────────────────────────
# Metric computation helpers
# ─────────────────────────────────────────────

def _oracle_r(r: Dict) -> float:
    return float(r.get("oracle_r", 0.0) or 0.0)


def _compute_metrics(records: List[Dict]) -> Dict[str, Any]:
    """Compute all required trade-level metrics from a list of candidate records."""
    taken = [r for r in records if r.get("taken")]
    blocked = [r for r in records if not r.get("taken")]
    n_trades = len(taken)

    gate_counts: Dict[str, int] = defaultdict(int)
    for r in blocked:
        gate = r.get("block_reason") or "unknown"
        gate_counts[gate] += 1
    n_blocked = len(blocked)
    gate_pct = {
        g: round(c / max(n_blocked, 1) * 100, 1)
        for g, c in sorted(gate_counts.items(), key=lambda x: -x[1])
    }

    if n_trades == 0:
        return {
            "trades": 0, "oracle_r_total": 0.0, "expectancy": 0.0,
            "win_rate": 0.0, "avg_win_r": 0.0, "avg_loss_r": 0.0,
            "long_trades": 0, "short_trades": 0,
            "long_expectancy": 0.0, "short_expectancy": 0.0,
            "top_decile_avg_r": 0.0, "bottom_decile_avg_r": 0.0,
            "monotonic_score_r": "N/A",
            "gate_block_pct": gate_pct,
            "n_candidates": len(records), "n_blocked": n_blocked,
        }

    rs = [_oracle_r(r) for r in taken]
    wins   = [v for v in rs if v > 0]
    losses = [v for v in rs if v <= 0]
    long_rs  = [_oracle_r(r) for r in taken if r.get("side") == 1]
    short_rs = [_oracle_r(r) for r in taken if r.get("side") == -1]

    sorted_by_score = sorted(taken, key=lambda r: r.get("final_score", r.get("raw_score", 0.0)), reverse=True)
    decile = max(1, n_trades // 10)
    top_decile = [_oracle_r(r) for r in sorted_by_score[:decile]]
    bot_decile = [_oracle_r(r) for r in sorted_by_score[-decile:]]

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
        "monotonic_score_r": "PASS" if _check_monotonic(taken) else "FAIL",
        "gate_block_pct": gate_pct,
        "n_candidates": len(records),
        "n_blocked": n_blocked,
    }


def _check_monotonic(taken: List[Dict], n_buckets: int = 5) -> bool:
    if len(taken) < n_buckets * 2:
        return True
    sorted_t = sorted(taken, key=lambda r: r.get("final_score", r.get("raw_score", 0.0)))
    bsz = len(sorted_t) // n_buckets
    means = [sum(_oracle_r(r) for r in sorted_t[i*bsz:(i+1)*bsz]) / bsz
             for i in range(n_buckets) if sorted_t[i*bsz:(i+1)*bsz]]
    return all(means[i] <= means[i+1] for i in range(len(means) - 1))


# ─────────────────────────────────────────────
# ASCII tables
# ─────────────────────────────────────────────

CORE_METRIC_ROWS = [
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


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _print_metrics_table(title: str, m: Dict[str, Any]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)
    for key, label in CORE_METRIC_ROWS:
        print(f"  {label:<28} {_fmt(m.get(key, '—'))}")
    gate_pct = m.get("gate_block_pct", {})
    if gate_pct:
        print(f"\n  Gate block breakdown (% of blocked):")
        for gate, pct in list(gate_pct.items())[:12]:
            print(f"    {gate:<24} {pct:>5.1f}%")
    print()


def _print_compare_table(label_a: str, label_b: str,
                         m_a: Dict[str, Any], m_b: Dict[str, Any]) -> None:
    numeric_keys = [
        ("trades",              "Trades"),
        ("oracle_r_total",      "Total oracle R"),
        ("expectancy",          "Expectancy (R/trade)"),
        ("win_rate",            "Win rate"),
        ("avg_win_r",           "Avg win (R)"),
        ("avg_loss_r",          "Avg loss (R)"),
        ("long_expectancy",     "Long E[R]"),
        ("short_expectancy",    "Short E[R]"),
        ("top_decile_avg_r",    "Top-decile avg R"),
        ("bottom_decile_avg_r", "Bottom-decile avg R"),
        ("n_candidates",        "Candidates"),
        ("n_blocked",           "Blocked"),
    ]
    print(f"\n{'=' * 72}")
    print(f"  COMPARE  {label_a}  vs  {label_b}")
    print(f"{'=' * 72}")
    print(f"  {'Metric':<28} {'A':>12} {'B':>12} {'Delta':>14}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*14}")
    for key, label in numeric_keys:
        va = m_a.get(key, 0) or 0
        vb = m_b.get(key, 0) or 0
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = vb - va
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
            print(f"  {label:<28} {_fmt(va):>12} {_fmt(vb):>12} {f'{delta:+.4f}{arrow}':>14}")
        else:
            print(f"  {label:<28} {str(va):>12} {str(vb):>12} {'—':>14}")

    print(f"\n  Monotonic  A: {m_a.get('monotonic_score_r', '—')}   "
          f"B: {m_b.get('monotonic_score_r', '—')}")

    gp_a = m_a.get("gate_block_pct", {})
    gp_b = m_b.get("gate_block_pct", {})
    all_gates = sorted(set(list(gp_a.keys()) + list(gp_b.keys())))
    if all_gates:
        print(f"\n  Gate block % comparison:")
        print(f"  {'Gate':<24} {'A%':>8} {'B%':>8} {'Delta%':>10}")
        print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*10}")
        for g in all_gates:
            pa = gp_a.get(g, 0.0)
            pb = gp_b.get(g, 0.0)
            dg = pb - pa
            arrow = "↑" if dg > 0 else ("↓" if dg < 0 else "=")
            print(f"  {g:<24} {pa:>7.1f}% {pb:>7.1f}% {f'{dg:+.1f}%{arrow}':>10}")
    print()


# ─────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────

def _save_csv(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    log.info("[validate] CSV saved → %s  (%d rows)", path, len(records))


def _load_csv(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec: Dict[str, Any] = {}
            for col in CSV_COLUMNS:
                v = row.get(col, "")
                if col in ("bar_idx", "timestamp", "side"):
                    try:
                        rec[col] = int(v) if v not in ("", "nan") else 0
                    except ValueError:
                        rec[col] = 0
                elif col in ("raw_score", "final_score", "threshold",
                             "mu_R", "p_trade", "adx_val", "oracle_r"):
                    try:
                        rec[col] = float(v) if v not in ("", "nan") else float("nan")
                    except ValueError:
                        rec[col] = float("nan")
                elif col in ("taken", "corr_blocked"):
                    rec[col] = str(v).lower() in ("true", "1", "yes")
                else:
                    rec[col] = v
            records.append(rec)
    log.info("[validate] Loaded %d records from %s", len(records), path)
    return records


# ─────────────────────────────────────────────
# JSON run helpers
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
# Walk-forward runner
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
) -> List[Dict]:
    """Run walk-forward and collect all candidate records via candidate_logger."""
    try:
        import torch
    except ImportError:
        log.error("[validate] torch not available — WF modes require the GPU machine.")
        sys.exit(1)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("[validate] Device: %s", dev)

    sys.path.insert(0, str(Path(__file__).parent))
    from train.v5_train import run_v5_walk_forward

    records: List[Dict] = []

    def _logger(rec: Dict) -> None:
        records.append(rec)

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
    return records


# ─────────────────────────────────────────────
# Mode: unit_audit
# ─────────────────────────────────────────────

def _torch_free_config_checks() -> List[Tuple[str, str, bool]]:
    """Direct torch-free checks of V5 config defaults.

    Loads shared_v5_trade_config.py directly via importlib to avoid the
    config/__init__.py which imports torch.

    Returns list of (check_name, detail, passed) tuples.
    """
    results: List[Tuple[str, str, bool]] = []
    try:
        import importlib.util
        cfg_path = Path(__file__).parent / "config" / "shared_v5_trade_config.py"
        spec = importlib.util.spec_from_file_location("_shared_cfg_direct", str(cfg_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        cfg = mod.V5TradeDefaults()

        checks = [
            ("score_threshold == 0.02",   cfg.score_threshold == 0.02,   f"got {cfg.score_threshold}"),
            ("slippage_base_bps == 6.0",  cfg.slippage_base_bps == 6.0,  f"got {cfg.slippage_base_bps}"),
            ("score_lambda == 0.5",       cfg.score_lambda == 0.5,        f"got {cfg.score_lambda}"),
            ("size_floor == 0.5",         cfg.size_floor == 0.5,          f"got {cfg.size_floor}"),
            ("cooldown_bars >= 4",        cfg.cooldown_bars >= 4,         f"got {cfg.cooldown_bars}"),
        ]
        for name, passed, detail in checks:
            results.append((name, detail, passed))
    except Exception as e:
        results.append(("config_import", str(e), False))
    return results


def mode_unit_audit(args: argparse.Namespace) -> None:
    log.info("[validate] mode=unit_audit")
    import subprocess
    tests_path = Path(__file__).parent / "tests" / "test_v5_precision_audit.py"
    if not tests_path.exists():
        log.error("Test file not found: %s", tests_path)
        sys.exit(1)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(tests_path), "-v", "--tb=short"],
        cwd=str(Path(__file__).parent),
    )
    pytest_ok = result.returncode == 0

    config_checks = _torch_free_config_checks()
    config_ok = all(passed for _, _, passed in config_checks)

    print(f"\n{'=' * 60}")
    print("  unit_audit — torch-free config checks")
    print('=' * 60)
    for name, detail, passed in config_checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}  ({detail})")
    print()

    overall_ok = pytest_ok and config_ok
    status = "PASSED" if overall_ok else "FAILED"
    log.info("[validate] unit_audit %s", status)

    payload = {
        "mode": "unit_audit",
        "pytest_exit_code": result.returncode,
        "pytest_ok": pytest_ok,
        "config_checks": [{"name": n, "detail": d, "passed": p} for n, d, p in config_checks],
        "config_ok": config_ok,
        "status": status,
    }
    _save_run("unit_audit", payload)
    if not overall_ok:
        sys.exit(1)


# ─────────────────────────────────────────────
# Fold-level breakdown helper
# ─────────────────────────────────────────────

def _fold_breakdown(records: List[Dict]) -> List[Dict]:
    """Group taken-trade records by approximate fold (distinct timestamp ranges).

    Since candidate_logger emits records for all folds sequentially, we detect
    fold boundaries by looking for timestamp resets (fold start < previous fold end).
    """
    taken = [r for r in records if r.get("taken")]
    if not taken:
        return []
    taken_sorted = sorted(taken, key=lambda r: r.get("timestamp", 0))

    folds: List[List[Dict]] = []
    current_fold: List[Dict] = [taken_sorted[0]]
    for r in taken_sorted[1:]:
        ts = r.get("timestamp", 0)
        prev_ts = current_fold[-1].get("timestamp", 0)
        if ts < prev_ts:
            folds.append(current_fold)
            current_fold = [r]
        else:
            current_fold.append(r)
    if current_fold:
        folds.append(current_fold)

    fold_summaries = []
    for i, fold_records in enumerate(folds):
        rs = [_oracle_r(r) for r in fold_records]
        wins = [v for v in rs if v > 0]
        fold_summaries.append({
            "fold": i + 1,
            "trades": len(fold_records),
            "total_r": round(sum(rs), 3),
            "expectancy": round(sum(rs) / max(len(rs), 1), 4),
            "win_rate": round(len(wins) / max(len(rs), 1), 4),
        })
    return fold_summaries


def _print_fold_table(fold_summaries: List[Dict]) -> None:
    if not fold_summaries:
        print("  (no fold breakdown available)")
        return
    print(f"\n  {'Fold':<6} {'Trades':>8} {'Total R':>10} {'Exp':>10} {'WR':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for f in fold_summaries:
        print(f"  {f['fold']:<6} {f['trades']:>8} {f['total_r']:>10.3f} "
              f"{f['expectancy']:>10.4f} {f['win_rate']:>7.1%}")
    print()


# ─────────────────────────────────────────────
# Mode: smoke_wf
# ─────────────────────────────────────────────

def mode_smoke_wf(args: argparse.Namespace) -> None:
    log.info("[validate] mode=smoke_wf  symbols=%s  epochs=15  folds=2  test_months=1",
             SMOKE_SYMBOLS)
    t0 = time.time()
    records = _run_wf(
        symbols=SMOKE_SYMBOLS, epochs=15, batch_size=128, lr=3e-4,
        train_months=3, test_months=1, max_folds=2, seed=42,
    )
    elapsed = time.time() - t0
    metrics = _compute_metrics(records)
    fold_summary = _fold_breakdown(records)
    _print_metrics_table("smoke_wf results", metrics)
    print("  Fold-level breakdown:")
    _print_fold_table(fold_summary)
    payload = {
        "mode": "smoke_wf", "elapsed_s": round(elapsed, 1),
        "symbols": SMOKE_SYMBOLS,
        "metrics": metrics,
        "fold_summary": fold_summary,
        "n_candidate_records": len(records),
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
    records = _run_wf(
        symbols=CANARY_SYMBOLS, epochs=30, batch_size=128, lr=3e-4,
        train_months=6, test_months=1, max_folds=folds, seed=42,
    )
    elapsed = time.time() - t0
    metrics = _compute_metrics(records)
    fold_summary = _fold_breakdown(records)
    _print_metrics_table("canary_wf results", metrics)
    print("  Fold-level breakdown:")
    _print_fold_table(fold_summary)
    payload = {
        "mode": "canary_wf", "elapsed_s": round(elapsed, 1),
        "symbols": CANARY_SYMBOLS, "max_folds": folds,
        "metrics": metrics,
        "fold_summary": fold_summary,
        "n_candidate_records": len(records),
    }
    out = _save_run("canary_wf", payload)
    print(f"Run saved: {out}")


# ─────────────────────────────────────────────
# Mode: candidate_diff
# ─────────────────────────────────────────────

def _top20_decision_changes(base: List[Dict], new: List[Dict]) -> None:
    """Print top-20 bars where taken/block_reason changed between two runs."""
    base_map = {(r.get("bar_idx", 0), r.get("symbol", ""), r.get("timestamp", 0)): r
                for r in base}
    new_map  = {(r.get("bar_idx", 0), r.get("symbol", ""), r.get("timestamp", 0)): r
                for r in new}
    changes = []
    for key, r_b in base_map.items():
        r_n = new_map.get(key)
        if r_n is None:
            continue
        taken_b = bool(r_b.get("taken"))
        taken_n = bool(r_n.get("taken"))
        reason_b = r_b.get("block_reason", "")
        reason_n = r_n.get("block_reason", "")
        if taken_b != taken_n or reason_b != reason_n:
            oracle = _oracle_r(r_b)
            changes.append({
                "bar_idx": key[0], "symbol": key[1],
                "old_taken": taken_b, "new_taken": taken_n,
                "old_reason": reason_b or "taken",
                "new_reason": reason_n or "taken",
                "oracle_r": oracle,
                "impact": abs(oracle),
            })
    changes.sort(key=lambda x: -x["impact"])
    top = changes[:20]
    print(f"\n{'=' * 72}")
    print(f"  TOP-20 DECISION CHANGES  (baseline → new)  total_changes={len(changes)}")
    print('=' * 72)
    print(f"  {'bar_idx':>8} {'sym':<12} {'old':>20} {'new':>20} {'oracle_R':>10}")
    print(f"  {'-'*8} {'-'*12} {'-'*20} {'-'*20} {'-'*10}")
    for c in top:
        old_str = "TAKEN" if c["old_taken"] else f"BLOCKED({c['old_reason']})"
        new_str = "TAKEN" if c["new_taken"] else f"BLOCKED({c['new_reason']})"
        print(f"  {c['bar_idx']:>8} {c['symbol']:<12} {old_str:>20} {new_str:>20} {c['oracle_r']:>10.4f}")
    print()


def mode_candidate_diff(args: argparse.Namespace) -> None:
    folds     = getattr(args, "folds", 2)
    run_path  = getattr(args, "run", None)
    vs_path   = getattr(args, "vs", None)
    csv_out   = getattr(args, "csv", None)

    if run_path:
        log.info("[validate] mode=candidate_diff  loading from CSV: %s", run_path)
        records = _load_csv(run_path)
    else:
        log.info("[validate] mode=candidate_diff  symbols=ALL20  folds=%d", folds)
        t0 = time.time()
        records = _run_wf(
            symbols=ALL_SYMBOLS, epochs=30, batch_size=128, lr=3e-4,
            train_months=6, test_months=1, max_folds=folds, seed=42,
        )
        log.info("[validate] WF complete in %.1fs  total_records=%d",
                 time.time() - t0, len(records))

    if csv_out:
        _save_csv(records, Path(csv_out))
    elif not run_path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        auto_path = RUNS_DIR / f"candidate_diff_{ts}.csv"
        _save_csv(records, auto_path)
        print(f"CSV saved: {auto_path}")

    taken  = [r for r in records if r.get("taken")]
    blocked = [r for r in records if not r.get("taken")]
    all_r   = [_oracle_r(r) for r in records if "oracle_r" in r]
    taken_r = [_oracle_r(r) for r in taken if "oracle_r" in r]

    oracle_total = sum(all_r)
    taken_total  = sum(taken_r)
    blocked_total = sum(_oracle_r(r) for r in blocked if "oracle_r" in r)
    capture = taken_total / oracle_total if oracle_total != 0 else 0.0

    gate_oracle: Dict[str, float] = defaultdict(float)
    gate_count:  Dict[str, int]   = defaultdict(int)
    for r in blocked:
        g = r.get("block_reason") or "unknown"
        gate_oracle[g] += _oracle_r(r)
        gate_count[g]  += 1

    print(f"\n{'=' * 60}")
    print("  CANDIDATE DIFF — Gate oracle R left on the table")
    print('=' * 60)
    print(f"  All candidates : {len(records):5d}   oracle_R={oracle_total:.2f}")
    print(f"  Taken          : {len(taken):5d}   oracle_R={taken_total:.2f}")
    print(f"  Blocked        : {len(blocked):5d}   oracle_R={blocked_total:.2f}")
    print(f"  Capture rate   : {capture:.1%}")
    print(f"\n  {'Gate':<24} {'Blocked':>8} {'Oracle R':>10} {'Avg R':>10}")
    print(f"  {'-'*24} {'-'*8} {'-'*10} {'-'*10}")
    for gate, cnt in sorted(gate_count.items(), key=lambda x: -abs(gate_oracle[x[0]])):
        avg = gate_oracle[gate] / cnt if cnt else 0
        print(f"  {gate:<24} {cnt:>8} {gate_oracle[gate]:>10.3f} {avg:>10.4f}")
    print()

    metrics = _compute_metrics(records)
    _print_metrics_table("candidate_diff trade metrics", metrics)

    if vs_path:
        new_records = _load_csv(vs_path)
        _top20_decision_changes(records, new_records)

    payload = {
        "mode": "candidate_diff",
        "symbols": "ALL20" if not run_path else f"from_csv:{run_path}",
        "max_folds": folds,
        "metrics": metrics,
        "oracle_total_all": round(oracle_total, 3),
        "oracle_total_taken": round(taken_total, 3),
        "capture_rate": round(capture, 4),
        "n_candidates": len(records),
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
    _print_compare_table(label_a, label_b, m_a, m_b)
    out = _save_run("compare", {
        "mode": "compare", "file_a": path_a, "file_b": path_b,
        "metrics_a": m_a, "metrics_b": m_b,
    })
    print(f"Run saved: {out}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    epilog = (
        "Modes:\n"
        "  unit_audit     -- Pytest precision audit + torch-free config checks\n"
        "  smoke_wf       -- 2-symbol 2-fold walk-forward (BTC/ETH)\n"
        "  canary_wf      -- 4-symbol N-fold walk-forward (BTC/ETH/SOL/BNB)\n"
        "  candidate_diff -- Per-candidate gate oracle R breakdown (CSV output)\n"
        "  compare        -- Diff two validate_runs JSON files\n"
    )
    p = argparse.ArgumentParser(
        description="V5 fast validation harness",
        epilog=epilog,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    sub.add_parser("unit_audit",
                   help="Pytest precision audit + torch-free config checks")

    sub.add_parser("smoke_wf",
                   help="2-symbol 2-fold walk-forward smoke test (BTC/ETH)")

    canary = sub.add_parser("canary_wf",
                            help="4-symbol N-fold walk-forward (BTC/ETH/SOL/BNB)")
    canary.add_argument("--folds", type=int, default=3,
                        help="Number of folds (default: 3)")

    diff = sub.add_parser("candidate_diff",
                          help="Per-candidate gate oracle R breakdown with optional CSV compare")
    diff.add_argument("--folds", type=int, default=2,
                      help="Number of folds to run (default: 2)")
    diff.add_argument("--csv", metavar="PATH",
                      help="Save all candidate records to this CSV path")
    diff.add_argument("--run", metavar="BASELINE_CSV",
                      help="Load a previously saved candidate CSV as baseline")
    diff.add_argument("--vs", metavar="NEW_CSV",
                      help="Load a second CSV and print top-20 decision changes vs --run")

    cmp = sub.add_parser("compare",
                         help="Diff two validate_runs JSON files (including gate block pct)")
    cmp.add_argument("files", nargs=2, metavar="FILE",
                     help="Two JSON run files to compare")

    return p


def main() -> None:
    parser = _build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    {
        "unit_audit":     mode_unit_audit,
        "smoke_wf":       mode_smoke_wf,
        "canary_wf":      mode_canary_wf,
        "candidate_diff": mode_candidate_diff,
        "compare":        mode_compare,
    }[args.mode](args)


if __name__ == "__main__":
    main()
