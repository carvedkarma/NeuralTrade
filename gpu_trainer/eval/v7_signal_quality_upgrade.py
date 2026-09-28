"""V7 signal-quality upgrade via two-stage walk-forward modeling.

Approach:
  1) Train a direction classifier (P(up)) and magnitude regressor (|ret_60m|)
     on each walk-forward fold.
  2) Build an expected-edge score per bar:
       edge = (2 * P(up) - 1) * E[|ret_60m|]
  3) Search policy filters (symbols, top-% edge selectivity, p-threshold,
     long-only/bi-directional, vol floor, session) for robust monthly R.

Run:
  python -m gpu_trainer.eval.v7_signal_quality_upgrade
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from gpu_trainer.eval.v7_signal_audit_augmented import (
    build_features,
    build_targets,
    load_symbol,
    walk_forward_indices,
)


COST_BPS = 8.0
COST_FRAC = COST_BPS / 1e4
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT", "XRPUSDT"]
CACHE_PATH = Path(".local/cache/v7_signal_upgrade/oos_preds.parquet")
OUT_JSON = Path(".local/reports/v7_signal_quality_upgrade.json")
OUT_MD = Path(".local/reports/v7_signal_quality_upgrade.md")


def _clf() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=220,
        max_depth=4,
        learning_rate=0.04,
        min_samples_leaf=250,
        random_state=0,
    )


def _reg() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=220,
        max_depth=4,
        learning_rate=0.04,
        min_samples_leaf=250,
        random_state=0,
        early_stopping=False,
    )


def _session_from_ts(ts_ms: np.ndarray) -> np.ndarray:
    hour = pd.to_datetime(ts_ms, unit="ms", utc=True).hour.to_numpy()
    out = np.full(len(hour), "Late", dtype=object)
    out[(hour >= 0) & (hour < 7)] = "Asia"
    out[(hour >= 7) & (hour < 14)] = "EU"
    out[(hour >= 14) & (hour < 21)] = "US"
    return out


def collect_oos_predictions(symbol: str) -> pd.DataFrame:
    df = load_symbol(symbol)
    if df.empty or len(df) < 20000:
        return pd.DataFrame()

    feats = build_features(df)
    targs = build_targets(df)
    y_ret = targs["ret_60m"].to_numpy()
    y_sign = (y_ret > 0).astype(int)

    feat_warmup = feats.notna().sum(axis=1)
    valid_from = max(int((feat_warmup > 8).idxmax()), 200)

    X = feats.iloc[valid_from:].to_numpy(dtype="float64")
    y_ret = y_ret[valid_from:]
    y_sign = y_sign[valid_from:]
    ts = df["timestamp"].to_numpy()[valid_from:]

    close = df["close"].astype("float64").to_numpy()[valid_from:]
    logret = np.log(pd.Series(close)).diff()
    vol_16 = logret.rolling(16).std().to_numpy()
    trend_16 = (np.log(pd.Series(close)) - np.log(pd.Series(close).shift(16))).to_numpy()

    folds = walk_forward_indices(ts)
    rows: list[dict] = []
    for fold, (tlo, thi, slo, shi) in enumerate(folds):
        X_tr, X_te = X[tlo:thi], X[slo:shi]
        yret_tr, yret_te = y_ret[tlo:thi], y_ret[slo:shi]
        ysgn_tr = y_sign[tlo:thi]

        good = np.isfinite(yret_tr)
        if good.sum() < 800:
            continue

        clf = _clf()
        clf.fit(X_tr[good], ysgn_tr[good])

        reg = _reg()
        reg.fit(X_tr[good], np.abs(yret_tr[good]))

        p_up = clf.predict_proba(X_te)[:, 1]
        mag = np.clip(reg.predict(X_te), 0.0, None)
        edge = (2.0 * p_up - 1.0) * mag

        vol = vol_16[slo:shi]
        sess = _session_from_ts(ts[slo:shi])
        side = np.sign(edge)
        sign_trend = np.sign(trend_16[slo:shi])
        regime = np.where(sign_trend == 0, "FLAT", np.where(sign_trend == side, "WITH", "COUNTER"))

        good_te = (
            np.isfinite(yret_te)
            & np.isfinite(p_up)
            & np.isfinite(mag)
            & np.isfinite(edge)
            & np.isfinite(vol)
            & (vol > 1e-8)
        )
        idx = np.where(good_te)[0]
        for i in idx:
            rows.append(
                {
                    "symbol": symbol,
                    "fold": int(fold),
                    "ts": int(ts[slo + i]),
                    "ret_60m": float(yret_te[i]),
                    "vol_16": float(vol[i]),
                    "risk_bps": float(vol[i] * 1e4),
                    "p_up": float(p_up[i]),
                    "pred_mag": float(mag[i]),
                    "edge": float(edge[i]),
                    "abs_edge": float(abs(edge[i])),
                    "session": str(sess[i]),
                    "regime": str(regime[i]),
                }
            )
    out = pd.DataFrame(rows)
    return out


def ensure_oos_cache(symbols: list[str], rebuild_cache: bool = False) -> pd.DataFrame:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cached = pd.DataFrame()
    if CACHE_PATH.exists() and not rebuild_cache:
        cached = pd.read_parquet(CACHE_PATH)
    cached_syms = set(cached["symbol"].unique()) if not cached.empty else set()
    missing = [s for s in symbols if s not in cached_syms]

    frames = [cached] if not cached.empty else []
    for sym in missing:
        s = collect_oos_predictions(sym)
        if not s.empty:
            frames.append(s)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[out["symbol"].isin(symbols)].copy()
    out["month"] = pd.to_datetime(out["ts"], unit="ms", utc=True).dt.to_period("M").astype(str)
    out["rank_pct"] = out.groupby(["symbol", "fold"])["abs_edge"].rank(pct=True, method="average")
    out.to_parquet(CACHE_PATH, index=False)
    return out


@dataclass
class PolicyResult:
    top_pct: float
    symbols: str
    session: str
    regime: str
    p_min: float
    long_only: bool
    risk_min_bps: float
    n: int
    months: int
    mean_net_bps: float
    avg_r_net: float
    mean_monthly_total_r: float
    median_monthly_total_r: float
    win_month_pct: float
    fold_min_avg_r: float
    fold_pos: int
    folds: int


def eval_policy(
    d: pd.DataFrame,
    top_pct: float,
    symbols: tuple[str, ...],
    session: str | None,
    regime: str | None,
    p_min: float,
    long_only: bool,
    risk_min_bps: float,
) -> PolicyResult | None:
    x = d[d["symbol"].isin(symbols)]
    if x.empty:
        return None

    q_cut = 1.0 - (top_pct / 100.0)
    x = x[x["rank_pct"] >= q_cut]
    if session is not None:
        x = x[x["session"] == session]
    if regime is not None:
        x = x[x["regime"] == regime]
    x = x[x["risk_bps"] >= risk_min_bps]

    if long_only:
        x = x[(x["edge"] > 0) & (x["p_up"] >= p_min)]
        side = np.ones(len(x), dtype=float)
    else:
        long_mask = (x["edge"] > 0) & (x["p_up"] >= p_min)
        short_mask = (x["edge"] < 0) & (x["p_up"] <= (1.0 - p_min))
        x = x[long_mask | short_mask]
        side = np.where(x["edge"].to_numpy() > 0, 1.0, -1.0)

    if len(x) < 80:
        return None

    gross = x["ret_60m"].to_numpy() * side
    net = gross - COST_FRAC
    r = net / x["vol_16"].to_numpy()
    months = x.groupby("month").size()
    if len(months) < 8:
        return None

    msum = pd.DataFrame({"month": x["month"].to_numpy(), "R": r}).groupby("month")["R"].sum()
    fold_avg = pd.DataFrame({"fold": x["fold"].to_numpy(), "R": r}).groupby("fold")["R"].mean()
    return PolicyResult(
        top_pct=float(top_pct),
        symbols=",".join(symbols),
        session=session or "ALL",
        regime=regime or "ALL",
        p_min=float(p_min),
        long_only=bool(long_only),
        risk_min_bps=float(risk_min_bps),
        n=int(len(x)),
        months=int(len(msum)),
        mean_net_bps=float(np.mean(net) * 1e4),
        avg_r_net=float(np.mean(r)),
        mean_monthly_total_r=float(msum.mean()),
        median_monthly_total_r=float(msum.median()),
        win_month_pct=float((msum > 0).mean() * 100),
        fold_min_avg_r=float(fold_avg.min()),
        fold_pos=int((fold_avg > 0).sum()),
        folds=int(len(fold_avg)),
    )


def search_policies(d: pd.DataFrame) -> pd.DataFrame:
    syms = sorted(d["symbol"].unique())
    symbol_sets = []
    for r in range(1, len(syms) + 1):
        symbol_sets.extend(itertools.combinations(syms, r))

    results: list[dict] = []
    for top_pct in (1.0, 2.0, 3.0, 5.0, 8.0, 10.0):
        for symbols in symbol_sets:
            for session in (None, "Asia", "EU", "US", "Late"):
                for regime in (None, "WITH", "COUNTER"):
                    for p_min in (0.52, 0.55, 0.58, 0.60, 0.62, 0.65):
                        for long_only in (True, False):
                            for risk_min in (0, 40, 60, 80, 100, 120, 150):
                                r = eval_policy(
                                    d=d,
                                    top_pct=top_pct,
                                    symbols=symbols,
                                    session=session,
                                    regime=regime,
                                    p_min=p_min,
                                    long_only=long_only,
                                    risk_min_bps=risk_min,
                                )
                                if r is None:
                                    continue
                                results.append(asdict(r))
    if not results:
        return pd.DataFrame()
    out = pd.DataFrame(results)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V7 two-stage signal quality WF search")
    p.add_argument("--symbols", nargs="+", default=SYMBOLS, help="Symbol list to evaluate")
    p.add_argument("--top-pcts", nargs="+", type=float,
                   default=[1.0, 2.0, 3.0, 5.0, 8.0, 10.0])
    p.add_argument("--p-mins", nargs="+", type=float,
                   default=[0.52, 0.55, 0.58, 0.60, 0.62, 0.65])
    p.add_argument("--risk-min-bps", nargs="+", type=float,
                   default=[0, 40, 60, 80, 100, 120, 150])
    p.add_argument("--sessions", nargs="+", default=["ALL", "Asia", "EU", "US", "Late"])
    p.add_argument("--regimes", nargs="+", default=["ALL", "WITH", "COUNTER"])
    p.add_argument("--long-only-only", action="store_true",
                   help="Only evaluate long-only policies")
    p.add_argument("--max-symbol-set-size", type=int, default=0,
                   help="Max symbols in subset; 0 means no limit")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Force rebuild OOS cache for selected symbols")
    return p.parse_args()


def search_policies_with_args(d: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    syms = sorted(d["symbol"].unique())
    symbol_sets = []
    max_size = args.max_symbol_set_size if args.max_symbol_set_size > 0 else len(syms)
    for r in range(1, min(len(syms), max_size) + 1):
        symbol_sets.extend(itertools.combinations(syms, r))

    sessions: list[str | None] = [None if s == "ALL" else s for s in args.sessions]
    regimes: list[str | None] = [None if r == "ALL" else r for r in args.regimes]
    long_flags = [True] if args.long_only_only else [True, False]

    results: list[dict] = []
    for top_pct in args.top_pcts:
        for symbols in symbol_sets:
            for session in sessions:
                for regime in regimes:
                    for p_min in args.p_mins:
                        for long_only in long_flags:
                            for risk_min in args.risk_min_bps:
                                r = eval_policy(
                                    d=d,
                                    top_pct=top_pct,
                                    symbols=symbols,
                                    session=session,
                                    regime=regime,
                                    p_min=p_min,
                                    long_only=long_only,
                                    risk_min_bps=risk_min,
                                )
                                if r is None:
                                    continue
                                results.append(asdict(r))
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


def main() -> None:
    args = parse_args()
    oos = ensure_oos_cache(symbols=args.symbols, rebuild_cache=args.rebuild_cache)
    if oos.empty:
        raise RuntimeError("no OOS predictions")

    policies = search_policies_with_args(oos, args)
    if policies.empty:
        raise RuntimeError("no valid policies evaluated")

    robust = policies[
        (policies["months"] >= 12)
        & (policies["n"] >= 300)
        & (policies["fold_min_avg_r"] > 0)
    ].copy()
    robust["robust_score"] = (
        robust["mean_monthly_total_r"]
        + 1.5 * robust["avg_r_net"]
        + 0.02 * robust["win_month_pct"]
    )
    robust = robust.sort_values(
        ["robust_score", "mean_monthly_total_r", "avg_r_net"],
        ascending=False,
    )

    best_overall = policies.sort_values("mean_monthly_total_r", ascending=False).head(25)
    best_robust = robust.head(25)

    ceiling = float(best_overall["mean_monthly_total_r"].max())
    robust_ceiling = float(best_robust["mean_monthly_total_r"].max()) if not best_robust.empty else 0.0

    payload = {
        "cost_bps": COST_BPS,
        "oos_rows": int(len(oos)),
        "symbols": sorted(oos["symbol"].unique().tolist()),
        "total_policies": int(len(policies)),
        "best_overall_monthly_r": ceiling,
        "best_robust_monthly_r": robust_ceiling,
        "best_overall": best_overall.to_dict("records"),
        "best_robust": best_robust.to_dict("records"),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    lines = [
        "# V7 Signal Quality Upgrade",
        "",
        f"- OOS rows: **{len(oos):,}**",
        f"- Policies tested: **{len(policies):,}**",
        f"- Best monthly R (overall): **{ceiling:+.2f}**",
        f"- Best monthly R (robust): **{robust_ceiling:+.2f}**",
        "",
        "## Top robust policies",
        "",
        "| top% | symbols | long_only | p_min | risk>=bps | session | regime | n | months | avg_R | mean monthly R | win-month% | fold min R |",
        "|---:|---|:---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in best_robust.iterrows():
        lines.append(
            f"| {r['top_pct']:.1f} | {r['symbols']} | "
            f"{'Y' if bool(r['long_only']) else 'N'} | {r['p_min']:.2f} | "
            f"{r['risk_min_bps']:.0f} | {r['session']} | {r['regime']} | "
            f"{int(r['n']):,} | {int(r['months'])} | {r['avg_r_net']:+.3f} | "
            f"{r['mean_monthly_total_r']:+.2f} | {r['win_month_pct']:.1f} | {r['fold_min_avg_r']:+.3f} |"
        )
    OUT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
