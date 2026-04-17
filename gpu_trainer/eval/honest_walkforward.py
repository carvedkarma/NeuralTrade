"""
V10 Phase 1 — Honest 6-fold walk-forward evaluator for the XGBoost meta-label
baseline. BTCUSDT only.

Locked Phase-1 fold scheme (do NOT tune):
    train_months = 24
    test_months  =  6
    max_folds    =  6
    purge_bars   = horizon_bars   # drop forward-leaking labels at train/test boundary

Per-fold pipeline:
    1. compute features for [train_start - warmup, test_end]
    2. compute meta-labels for the same window
    3. drop last `horizon_bars` rows from train (label leakage purge)
    4. baselines.xgboost_meta.train_fold (internally holds out last 20%
       of train as calibration; selects T_fold per the locked rule)
    5. score test, admit trades where p_calibrated >= T_fold,
       compute trade-level metrics

Stop criterion:
    PASS iff (a) fold-average PF >= 1.3, (b) per-fold trades >= 500,
    (c) no fold has PF < 1.0. All three checked on test windows only.

Run:
    cd gpu_trainer
    python -m eval.honest_walkforward --horizon 4h
    python -m eval.honest_walkforward --all-horizons
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
GPU_TRAINER_DIR = THIS_DIR.parent
if str(GPU_TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(GPU_TRAINER_DIR))

from data.pipeline import FeatureEngineer  # noqa: E402
from labels.meta_label import HORIZONS_BARS, compute_meta_labels  # noqa: E402
from baselines.xgboost_meta import (  # noqa: E402
    train_fold, predict_calibrated, _profit_factor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("honest_wf")

DATA_CACHE = GPU_TRAINER_DIR / "data_cache"
REPORT_DIR = GPU_TRAINER_DIR / "reports" / "v10_phase1"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
VERDICT_PATH = Path(__file__).resolve().parents[2] / ".local" / "tasks" / "v5-static-postmortem-verdict.md"

TRAIN_MONTHS_LOCKED = 24
TEST_MONTHS_LOCKED = 6
MAX_FOLDS_LOCKED = 6

PASS_FOLD_PF_MIN = 1.3
PASS_FOLD_TRADES_MIN = 500
PASS_FOLD_FLOOR = 1.0


@dataclass
class FoldResult:
    fold_num: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    threshold: float | None
    cal_trades: int
    cal_pf: float
    n_eligible_test: int
    n_trades: int
    win_rate: float
    expectancy_R: float
    pf: float
    max_dd_R: float
    auc: float
    calibration_slope: float
    notes: str = ""


@dataclass
class HorizonReport:
    horizon: str
    horizon_bars: int
    folds: list = field(default_factory=list)
    avg_pf: float = 0.0
    min_pf: float = 0.0
    avg_trades: float = 0.0
    passed: bool = False
    fail_reasons: list = field(default_factory=list)


def _load_btc() -> pd.DataFrame:
    p = DATA_CACHE / "BTCUSDT_15m.parquet"
    df = pd.read_parquet(p).sort_values("timestamp").reset_index(drop=True)
    log.info("BTC loaded: %d bars  range %s -> %s", len(df),
             datetime.utcfromtimestamp(df["timestamp"].iloc[0] / 1000),
             datetime.utcfromtimestamp(df["timestamp"].iloc[-1] / 1000))
    return df


def _build_folds(ts_ms: np.ndarray) -> list:
    from dateutil.relativedelta import relativedelta
    data_start = datetime.utcfromtimestamp(ts_ms[0] / 1000)
    data_end = datetime.utcfromtimestamp(ts_ms[-1] / 1000)
    first_test_start = data_start + relativedelta(months=TRAIN_MONTHS_LOCKED)
    folds = []
    cur = first_test_start
    while cur < data_end and len(folds) < MAX_FOLDS_LOCKED:
        train_end = cur
        train_start = train_end - relativedelta(months=TRAIN_MONTHS_LOCKED)
        if train_start < data_start:
            train_start = data_start
        test_end = cur + relativedelta(months=TEST_MONTHS_LOCKED)
        if test_end > data_end:
            test_end = data_end
        if (test_end - cur).total_seconds() < 7 * 86400:
            break  # skip stub trailing folds
        folds.append((train_start, train_end, cur, test_end))
        cur = test_end
    return folds


def _slice_idx(ts_dt: pd.Series, start: datetime, end: datetime) -> tuple[int, int]:
    a = int(ts_dt.searchsorted(pd.Timestamp(start), side="left"))
    b = int(ts_dt.searchsorted(pd.Timestamp(end), side="left"))
    return a, b


def _max_drawdown_R(R: np.ndarray) -> float:
    if len(R) == 0:
        return 0.0
    eq = np.cumsum(R)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    return float(dd.min())


def _calibration_slope(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Slope of mean-actual vs mean-predicted in `bins` quantile bins.
    Perfect calibration -> slope = 1.0."""
    if len(p) < bins * 5:
        return float("nan")
    qs = np.quantile(p, np.linspace(0, 1, bins + 1))
    qs[0] = -np.inf
    qs[-1] = np.inf
    xs, ys = [], []
    for i in range(bins):
        m = (p >= qs[i]) & (p < qs[i + 1])
        if m.sum() < 5:
            continue
        xs.append(p[m].mean())
        ys.append(y[m].mean())
    if len(xs) < 3:
        return float("nan")
    xs = np.array(xs); ys = np.array(ys)
    cov = np.cov(xs, ys, ddof=0)[0, 1]
    var = np.var(xs)
    return float(cov / var) if var > 1e-12 else float("nan")


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def run_horizon(horizon: str, df: pd.DataFrame, feats: pd.DataFrame) -> HorizonReport:
    if horizon not in HORIZONS_BARS:
        raise ValueError(f"unknown horizon {horizon}")
    hbars = HORIZONS_BARS[horizon]
    log.info("=" * 88)
    log.info("HORIZON %s  (%d bars)", horizon, hbars)
    log.info("=" * 88)

    meta = compute_meta_labels(df, horizon_bars=hbars)
    ts_dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    feature_names = list(feats.columns)
    X_all = feats.to_numpy(dtype=np.float32)
    y_all = meta["meta_label"].to_numpy()
    R_all = meta["R_net"].to_numpy()
    elig_all = meta["eligible"].to_numpy()

    finite_rows = np.isfinite(X_all).all(axis=1)
    usable = elig_all & finite_rows
    log.info("Eligible rows %d / %d  (finite features: %d)",
             elig_all.sum(), len(df), finite_rows.sum())

    folds = _build_folds(df["timestamp"].to_numpy())
    log.info("Built %d folds (train_months=%d, test_months=%d, max=%d)",
             len(folds), TRAIN_MONTHS_LOCKED, TEST_MONTHS_LOCKED, MAX_FOLDS_LOCKED)

    report = HorizonReport(horizon=horizon, horizon_bars=hbars)
    pfs, trades_list = [], []

    for fold_num, (tr_start, tr_end, te_start, te_end) in enumerate(folds, start=1):
        i_tr_a, i_tr_b = _slice_idx(ts_dt, tr_start, tr_end)
        i_te_a, i_te_b = _slice_idx(ts_dt, te_start, te_end)

        # Purge: drop last `hbars` rows of train (forward-looking labels).
        i_tr_b_purged = max(i_tr_a, i_tr_b - hbars)

        train_mask = np.zeros(len(df), dtype=bool)
        train_mask[i_tr_a:i_tr_b_purged] = True
        train_mask &= usable
        test_mask = np.zeros(len(df), dtype=bool)
        test_mask[i_te_a:i_te_b] = True
        test_mask &= usable

        X_tr = X_all[train_mask]
        y_tr = y_all[train_mask]
        R_tr = R_all[train_mask]
        X_te = X_all[test_mask]
        y_te = y_all[test_mask]
        R_te = R_all[test_mask]

        log.info("Fold %d: train %s..%s  N_train=%d  test %s..%s  N_test=%d",
                 fold_num, tr_start.date(), tr_end.date(), len(X_tr),
                 te_start.date(), te_end.date(), len(X_te))

        if len(X_tr) < 1000 or len(X_te) < 100:
            report.folds.append(FoldResult(
                fold_num=fold_num,
                train_start=str(tr_start.date()), train_end=str(tr_end.date()),
                test_start=str(te_start.date()), test_end=str(te_end.date()),
                threshold=None, cal_trades=0, cal_pf=0.0,
                n_eligible_test=len(X_te), n_trades=0, win_rate=0.0,
                expectancy_R=0.0, pf=0.0, max_dd_R=0.0, auc=float("nan"),
                calibration_slope=float("nan"),
                notes="insufficient data",
            ))
            pfs.append(0.0); trades_list.append(0)
            continue

        model = train_fold(X_tr, y_tr, R_tr, feature_names)
        thr = model.threshold_info
        log.info("  best_iter=%d  cal: T=%s  trades=%d  PF=%.2f  reason=%s",
                 model.best_iteration,
                 f"{thr.threshold:.4f}" if thr.threshold is not None else "None",
                 thr.cal_trades, thr.cal_pf, thr.reason)

        p_te = predict_calibrated(model, X_te)

        if thr.threshold is None:
            n_trades = 0
            R_sel = np.array([], dtype=np.float64)
        else:
            sel = p_te >= thr.threshold
            R_sel = R_te[sel]
            n_trades = int(sel.sum())

        if n_trades > 0:
            wr = float((R_sel > 0).mean())
            exp_R = float(R_sel.mean())
            pf = _profit_factor(R_sel)
            mdd = _max_drawdown_R(R_sel)
        else:
            wr = exp_R = pf = mdd = 0.0

        auc = _auc(p_te, y_te)
        cal_slope = _calibration_slope(p_te, y_te)

        log.info("  TEST trades=%d  WR=%.3f  expR=%+.4f  PF=%.2f  maxDD=%.2fR  AUC=%.3f  calslope=%.2f",
                 n_trades, wr, exp_R, pf, mdd, auc, cal_slope)

        report.folds.append(FoldResult(
            fold_num=fold_num,
            train_start=str(tr_start.date()), train_end=str(tr_end.date()),
            test_start=str(te_start.date()), test_end=str(te_end.date()),
            threshold=thr.threshold, cal_trades=thr.cal_trades, cal_pf=thr.cal_pf,
            n_eligible_test=len(X_te), n_trades=n_trades, win_rate=wr,
            expectancy_R=exp_R, pf=pf, max_dd_R=mdd, auc=auc,
            calibration_slope=cal_slope,
            notes=thr.reason if thr.threshold is None else "",
        ))
        pfs.append(pf); trades_list.append(n_trades)

    report.avg_pf = float(np.mean(pfs)) if pfs else 0.0
    report.min_pf = float(np.min(pfs)) if pfs else 0.0
    report.avg_trades = float(np.mean(trades_list)) if trades_list else 0.0

    fail_reasons = []
    if report.avg_pf < PASS_FOLD_PF_MIN:
        fail_reasons.append(f"avg PF {report.avg_pf:.2f} < {PASS_FOLD_PF_MIN}")
    if any(t < PASS_FOLD_TRADES_MIN for t in trades_list):
        bad = [f"f{i+1}={t}" for i, t in enumerate(trades_list) if t < PASS_FOLD_TRADES_MIN]
        fail_reasons.append(f"trades<{PASS_FOLD_TRADES_MIN} in folds: {bad}")
    if any(p < PASS_FOLD_FLOOR for p in pfs):
        bad = [f"f{i+1}={p:.2f}" for i, p in enumerate(pfs) if p < PASS_FOLD_FLOOR]
        fail_reasons.append(f"PF<{PASS_FOLD_FLOOR} in folds: {bad}")
    report.fail_reasons = fail_reasons
    report.passed = len(fail_reasons) == 0

    log.info("HORIZON %s  avg PF=%.2f  min PF=%.2f  avg trades=%.0f  -> %s",
             horizon, report.avg_pf, report.min_pf, report.avg_trades,
             "PASS" if report.passed else f"FAIL ({'; '.join(fail_reasons)})")

    out_path = REPORT_DIR / f"walkforward_{horizon}.json"
    with open(out_path, "w") as f:
        json.dump({
            "horizon": horizon, "horizon_bars": hbars,
            "train_months": TRAIN_MONTHS_LOCKED, "test_months": TEST_MONTHS_LOCKED,
            "max_folds": MAX_FOLDS_LOCKED,
            "passed": report.passed, "fail_reasons": fail_reasons,
            "avg_pf": report.avg_pf, "min_pf": report.min_pf,
            "avg_trades": report.avg_trades,
            "folds": [asdict(fr) for fr in report.folds],
            "generated_at": datetime.utcnow().isoformat(),
        }, f, indent=2)
    log.info("Wrote %s", out_path)
    return report


def _format_verdict_section(reports: list) -> str:
    lines = []
    lines.append("\n## V10 Phase 1 — Honest Walk-Forward Verdict\n")
    lines.append(f"_Generated {datetime.utcnow().isoformat()} on BTCUSDT, "
                 f"6-fold walk-forward (24 mo train / 6 mo test), "
                 f"XGBoost meta-label, ±1.5×ATR(14) barriers, "
                 f"isotonic calibration, threshold rule = smallest cal-prob "
                 f"with cal_trades≥100 AND cal_PF≥1.4. "
                 f"All metrics net of 6 bps round-trip slippage._\n")

    any_pass = False
    for rep in reports:
        verdict = "**PASS**" if rep.passed else "**FAIL**"
        lines.append(f"\n### Horizon `{rep.horizon}` — {verdict}\n")
        if not rep.passed:
            lines.append(f"_Failure reasons: {'; '.join(rep.fail_reasons)}_\n")
        lines.append("\n| Fold | Train window | Test window | T | n_trades | WR | exp R | PF | maxDD R | AUC | cal slope |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for f in rep.folds:
            T_disp = f"{f.threshold:.4f}" if f.threshold is not None else "—"
            lines.append(
                f"| {f.fold_num} | {f.train_start}→{f.train_end} | "
                f"{f.test_start}→{f.test_end} | {T_disp} | {f.n_trades} | "
                f"{f.win_rate:.3f} | {f.expectancy_R:+.3f} | {f.pf:.2f} | "
                f"{f.max_dd_R:.2f} | {f.auc:.3f} | {f.calibration_slope:.2f} |"
            )
        lines.append(f"\n_Avg PF {rep.avg_pf:.2f}  min PF {rep.min_pf:.2f}  "
                     f"avg trades {rep.avg_trades:.0f}_\n")
        if rep.passed:
            any_pass = True

    lines.append("\n### Conclusion\n")
    if any_pass:
        passed = [r.horizon for r in reports if r.passed]
        lines.append(f"At least one horizon passed the locked stop criterion "
                     f"({', '.join(passed)}). Draft Phase 2 task — neural V10 "
                     f"on the winning horizon(s) — citing the measured baseline "
                     f"PF as the floor any neural rebuild must beat.\n")
    else:
        lines.append("All three horizons failed the locked stop criterion. "
                     "V10 Phase 2 (neural) is not justified by the data at the "
                     "tested horizons. Next task: Phase 1b feature audit / "
                     "rules-based pivot, planned with this evidence in hand.\n")
    return "\n".join(lines) + "\n"


def write_verdict(reports: list):
    section = _format_verdict_section(reports)
    if VERDICT_PATH.exists():
        with open(VERDICT_PATH, "a") as f:
            f.write(section)
        log.info("Appended verdict section to %s", VERDICT_PATH)
    else:
        log.warning("Verdict file %s missing; writing standalone", VERDICT_PATH)
        with open(VERDICT_PATH, "w") as f:
            f.write("# V5 Static Post-Mortem — Verdict\n")
            f.write(section)


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--horizon", choices=list(HORIZONS_BARS.keys()))
    g.add_argument("--all-horizons", action="store_true")
    p.add_argument("--skip-verdict", action="store_true",
                   help="Do not append to v5-static-postmortem-verdict.md "
                        "(useful when running individual horizons).")
    args = p.parse_args()

    df = _load_btc()
    eng = FeatureEngineer()
    feats = eng.compute_all_features(df)
    log.info("Features: %d columns x %d rows", feats.shape[1], feats.shape[0])

    horizons = list(HORIZONS_BARS.keys()) if args.all_horizons else [args.horizon]
    reports = [run_horizon(h, df, feats) for h in horizons]

    if args.all_horizons and not args.skip_verdict:
        write_verdict(reports)


if __name__ == "__main__":
    main()
