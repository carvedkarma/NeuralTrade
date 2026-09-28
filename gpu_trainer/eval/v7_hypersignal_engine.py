"""V7 HyperSignal Engine (stacked, uncertainty-aware, walk-forward).

This script pushes beyond simple thresholding by using:
  1) Base models per fold:
       - P(up) classifier
       - |ret_60m| regressor
       - signed-return ensemble (two regressors for disagreement)
  2) Uncertainty-aware quality score:
       quality = pred_signed / (abs(pred_a - pred_b) + entropy + eps)
  3) Fold-local meta model:
       predicts tradeability (net positive after cost) from base signals.
  4) Portfolio selector:
       per-timestamp top-K by quality to avoid low-conviction crowding.

Run:
  python -m gpu_trainer.eval.v7_hypersignal_engine
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
DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "XRPUSDT",
]
CACHE_DIR = Path(".local/cache/v7_hypersignal")
OUT_JSON = Path(".local/reports/v7_hypersignal.json")
OUT_MD = Path(".local/reports/v7_hypersignal.md")


def _clf(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=260,
        max_depth=5,
        learning_rate=0.04,
        min_samples_leaf=240,
        random_state=seed,
    )


def _reg(seed: int = 0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=260,
        max_depth=5,
        learning_rate=0.04,
        min_samples_leaf=240,
        random_state=seed,
        early_stopping=False,
    )


def _session_from_ts(ts_ms: np.ndarray) -> np.ndarray:
    h = pd.to_datetime(ts_ms, unit="ms", utc=True).hour.to_numpy()
    out = np.full(len(h), "Late", dtype=object)
    out[(h >= 0) & (h < 7)] = "Asia"
    out[(h >= 7) & (h < 14)] = "EU"
    out[(h >= 14) & (h < 21)] = "US"
    return out


def _entropy_binary(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _enc_session(s: np.ndarray) -> np.ndarray:
    m = {"Asia": 0.0, "EU": 1.0, "US": 2.0, "Late": 3.0}
    return np.array([m.get(str(x), 3.0) for x in s], dtype=float)


def _enc_regime(r: np.ndarray) -> np.ndarray:
    m = {"COUNTER": 0.0, "FLAT": 1.0, "WITH": 2.0}
    return np.array([m.get(str(x), 1.0) for x in r], dtype=float)


def _meta_features(
    p_up: np.ndarray,
    pred_mag: np.ndarray,
    pred_signed: np.ndarray,
    pred_disagree: np.ndarray,
    entropy: np.ndarray,
    risk_bps: np.ndarray,
    sess: np.ndarray,
    regime: np.ndarray,
) -> np.ndarray:
    quality = pred_signed / (pred_disagree + entropy + 1e-6)
    return np.column_stack([
        p_up,
        pred_mag,
        pred_signed,
        pred_disagree,
        entropy,
        risk_bps,
        quality,
        _enc_session(sess),
        _enc_regime(regime),
    ])


def collect_oos(symbol: str) -> pd.DataFrame:
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
    trend_16 = (
        np.log(pd.Series(close)) - np.log(pd.Series(close).shift(16))
    ).to_numpy()

    folds = walk_forward_indices(ts)
    rows: list[dict] = []
    for fold, (tlo, thi, slo, shi) in enumerate(folds):
        X_tr, X_te = X[tlo:thi], X[slo:shi]
        yret_tr, yret_te = y_ret[tlo:thi], y_ret[slo:shi]
        ysgn_tr = y_sign[tlo:thi]

        good_tr = np.isfinite(yret_tr)
        if good_tr.sum() < 1200:
            continue

        c = _clf(0)
        c.fit(X_tr[good_tr], ysgn_tr[good_tr])
        p_up_te = c.predict_proba(X_te)[:, 1]
        p_up_tr = c.predict_proba(X_tr)[:, 1]

        r_abs = _reg(0)
        r_abs.fit(X_tr[good_tr], np.abs(yret_tr[good_tr]))
        mag_te = np.clip(r_abs.predict(X_te), 0.0, None)
        mag_tr = np.clip(r_abs.predict(X_tr), 0.0, None)

        r_sig_a = _reg(11)
        r_sig_b = _reg(29)
        r_sig_a.fit(X_tr[good_tr], yret_tr[good_tr])
        r_sig_b.fit(X_tr[good_tr], yret_tr[good_tr])
        sig_a_te = r_sig_a.predict(X_te)
        sig_b_te = r_sig_b.predict(X_te)
        sig_a_tr = r_sig_a.predict(X_tr)
        sig_b_tr = r_sig_b.predict(X_tr)
        pred_signed_te = 0.5 * (sig_a_te + sig_b_te)
        pred_signed_tr = 0.5 * (sig_a_tr + sig_b_tr)

        disagree_te = np.abs(sig_a_te - sig_b_te)
        disagree_tr = np.abs(sig_a_tr - sig_b_tr)
        ent_te = _entropy_binary(p_up_te)
        ent_tr = _entropy_binary(p_up_tr)

        vol_tr = vol_16[tlo:thi]
        vol_te = vol_16[slo:shi]
        sess_tr = _session_from_ts(ts[tlo:thi])
        sess_te = _session_from_ts(ts[slo:shi])
        sign_trend_tr = np.sign(trend_16[tlo:thi])
        sign_trend_te = np.sign(trend_16[slo:shi])

        side_tr = np.sign(pred_signed_tr)
        side_te = np.sign(pred_signed_te)
        reg_tr = np.where(
            sign_trend_tr == 0,
            "FLAT",
            np.where(sign_trend_tr == side_tr, "WITH", "COUNTER"),
        )
        reg_te = np.where(
            sign_trend_te == 0,
            "FLAT",
            np.where(sign_trend_te == side_te, "WITH", "COUNTER"),
        )

        # Meta-label from train rows: was predicted direction net-positive after cost?
        pred_side_tr = np.sign(pred_signed_tr)
        pred_side_tr[pred_side_tr == 0] = 1
        net_pred_tr = yret_tr * pred_side_tr - COST_FRAC
        y_meta_tr = (net_pred_tr > 0).astype(int)
        meta_good = (
            np.isfinite(yret_tr)
            & np.isfinite(vol_tr)
            & (vol_tr > 1e-8)
            & np.isfinite(pred_signed_tr)
            & np.isfinite(disagree_tr)
            & np.isfinite(ent_tr)
            & np.isfinite(mag_tr)
        )
        meta_te = np.full(len(X_te), 0.5, dtype=float)
        if meta_good.sum() > 1500:
            X_meta_tr = _meta_features(
                p_up=p_up_tr[meta_good],
                pred_mag=mag_tr[meta_good],
                pred_signed=pred_signed_tr[meta_good],
                pred_disagree=disagree_tr[meta_good],
                entropy=ent_tr[meta_good],
                risk_bps=vol_tr[meta_good] * 1e4,
                sess=sess_tr[meta_good],
                regime=reg_tr[meta_good],
            )
            mclf = _clf(77)
            mclf.fit(X_meta_tr, y_meta_tr[meta_good])

            te_good_for_meta = (
                np.isfinite(vol_te)
                & (vol_te > 1e-8)
                & np.isfinite(pred_signed_te)
                & np.isfinite(disagree_te)
                & np.isfinite(ent_te)
                & np.isfinite(mag_te)
            )
            X_meta_te = _meta_features(
                p_up=p_up_te[te_good_for_meta],
                pred_mag=mag_te[te_good_for_meta],
                pred_signed=pred_signed_te[te_good_for_meta],
                pred_disagree=disagree_te[te_good_for_meta],
                entropy=ent_te[te_good_for_meta],
                risk_bps=vol_te[te_good_for_meta] * 1e4,
                sess=sess_te[te_good_for_meta],
                regime=reg_te[te_good_for_meta],
            )
            meta_te[te_good_for_meta] = mclf.predict_proba(X_meta_te)[:, 1]

        good_te = (
            np.isfinite(yret_te)
            & np.isfinite(vol_te)
            & (vol_te > 1e-8)
            & np.isfinite(pred_signed_te)
            & np.isfinite(disagree_te)
            & np.isfinite(ent_te)
            & np.isfinite(mag_te)
            & np.isfinite(meta_te)
        )
        idx = np.where(good_te)[0]
        quality_te = pred_signed_te / (disagree_te + ent_te + 1e-6)
        for i in idx:
            rows.append({
                "symbol": symbol,
                "fold": int(fold),
                "ts": int(ts[slo + i]),
                "ret_60m": float(yret_te[i]),
                "vol_16": float(vol_te[i]),
                "risk_bps": float(vol_te[i] * 1e4),
                "p_up": float(p_up_te[i]),
                "pred_mag": float(mag_te[i]),
                "pred_signed": float(pred_signed_te[i]),
                "pred_disagree": float(disagree_te[i]),
                "entropy": float(ent_te[i]),
                "meta_p": float(meta_te[i]),
                "quality": float(quality_te[i]),
                "abs_quality": float(abs(quality_te[i])),
                "session": str(sess_te[i]),
                "regime": str(reg_te[i]),
            })
    return pd.DataFrame(rows)


def cache_path_for(symbols: list[str]) -> Path:
    key = "_".join(sorted(symbols))
    return CACHE_DIR / f"oos_{key}.parquet"


def ensure_oos(symbols: list[str], rebuild: bool) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = cache_path_for(symbols)
    if p.exists() and not rebuild:
        return pd.read_parquet(p)
    frames = []
    for sym in symbols:
        s = collect_oos(sym)
        if not s.empty:
            frames.append(s)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["month"] = pd.to_datetime(out["ts"], unit="ms", utc=True).dt.to_period("M").astype(str)
    out["rank_pct"] = out.groupby(["symbol", "fold"])["abs_quality"].rank(pct=True, method="average")
    out.to_parquet(p, index=False)
    return out


@dataclass
class Policy:
    top_pct: float
    symbols: str
    session: str
    regime: str
    p_min: float
    meta_min: float
    long_only: bool
    risk_min_bps: float
    max_per_ts: int
    n: int
    months: int
    mean_net_bps: float
    avg_r_net: float
    mean_monthly_r: float
    median_monthly_r: float
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
    meta_min: float,
    long_only: bool,
    risk_min_bps: float,
    max_per_ts: int,
) -> Policy | None:
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
    x = x[x["meta_p"] >= meta_min]

    if long_only:
        x = x[(x["pred_signed"] > 0) & (x["p_up"] >= p_min)]
        x = x.copy()
        x["side"] = 1.0
    else:
        long_mask = (x["pred_signed"] > 0) & (x["p_up"] >= p_min)
        short_mask = (x["pred_signed"] < 0) & (x["p_up"] <= (1.0 - p_min))
        x = x[long_mask | short_mask].copy()
        x["side"] = np.where(x["pred_signed"] > 0, 1.0, -1.0)
    if x.empty:
        return None

    if max_per_ts > 0:
        x = (
            x.sort_values(["ts", "abs_quality"], ascending=[True, False])
            .groupby("ts", as_index=False)
            .head(max_per_ts)
        )

    if len(x) < 150:
        return None

    gross = x["ret_60m"].to_numpy() * x["side"].to_numpy()
    net = gross - COST_FRAC
    r = net / x["vol_16"].to_numpy()
    months = pd.DataFrame({"month": x["month"].to_numpy(), "R": r}).groupby("month")["R"].sum()
    if len(months) < 10:
        return None
    fold = pd.DataFrame({"fold": x["fold"].to_numpy(), "R": r}).groupby("fold")["R"].mean()

    return Policy(
        top_pct=float(top_pct),
        symbols=",".join(symbols),
        session=session or "ALL",
        regime=regime or "ALL",
        p_min=float(p_min),
        meta_min=float(meta_min),
        long_only=bool(long_only),
        risk_min_bps=float(risk_min_bps),
        max_per_ts=int(max_per_ts),
        n=int(len(x)),
        months=int(len(months)),
        mean_net_bps=float(np.mean(net) * 1e4),
        avg_r_net=float(np.mean(r)),
        mean_monthly_r=float(months.mean()),
        median_monthly_r=float(months.median()),
        win_month_pct=float((months > 0).mean() * 100),
        fold_min_avg_r=float(fold.min()),
        fold_pos=int((fold > 0).sum()),
        folds=int(len(fold)),
    )


def search(
    d: pd.DataFrame,
    top_pcts: list[float],
    p_mins: list[float],
    meta_mins: list[float],
    risk_mins: list[float],
    sessions: list[str],
    regimes: list[str],
    long_only_only: bool,
    max_symbol_set_size: int,
    max_per_ts_vals: list[int],
) -> pd.DataFrame:
    syms = sorted(d["symbol"].unique())
    sets: list[tuple[str, ...]] = []
    for r in range(1, min(len(syms), max_symbol_set_size) + 1):
        sets.extend(itertools.combinations(syms, r))
    results: list[dict] = []
    long_opts = [True] if long_only_only else [True, False]

    norm_session = [None if s.upper() == "ALL" else s for s in sessions]
    norm_regime = [None if r.upper() == "ALL" else r for r in regimes]

    for top in top_pcts:
        for sset in sets:
            for sess in norm_session:
                for reg in norm_regime:
                    for p in p_mins:
                        for mp in meta_mins:
                            for lo in long_opts:
                                for rm in risk_mins:
                                    for k in max_per_ts_vals:
                                        out = eval_policy(
                                            d=d,
                                            top_pct=top,
                                            symbols=sset,
                                            session=sess,
                                            regime=reg,
                                            p_min=p,
                                            meta_min=mp,
                                            long_only=lo,
                                            risk_min_bps=rm,
                                            max_per_ts=k,
                                        )
                                        if out is None:
                                            continue
                                        results.append(asdict(out))
    return pd.DataFrame(results)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V7 HyperSignal stacked WF search")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--top-pcts", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0])
    p.add_argument("--p-mins", nargs="+", type=float, default=[0.52, 0.55, 0.58, 0.60, 0.65])
    p.add_argument("--meta-mins", nargs="+", type=float, default=[0.50, 0.55, 0.60, 0.65])
    p.add_argument("--risk-min-bps", nargs="+", type=float, default=[0, 20, 40, 60, 80, 100, 120])
    p.add_argument("--sessions", nargs="+", default=["ALL", "Asia", "EU", "US", "Late"])
    p.add_argument("--regimes", nargs="+", default=["ALL", "WITH", "COUNTER", "FLAT"])
    p.add_argument("--max-per-ts", nargs="+", type=int, default=[1, 2, 3, 5])
    p.add_argument("--long-only-only", action="store_true")
    p.add_argument("--max-symbol-set-size", type=int, default=7)
    p.add_argument("--rebuild-cache", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    oos = ensure_oos(args.symbols, rebuild=args.rebuild_cache)
    if oos.empty:
        raise RuntimeError("no OOS rows produced")

    pol = search(
        d=oos,
        top_pcts=args.top_pcts,
        p_mins=args.p_mins,
        meta_mins=args.meta_mins,
        risk_mins=args.risk_min_bps,
        sessions=args.sessions,
        regimes=args.regimes,
        long_only_only=args.long_only_only,
        max_symbol_set_size=args.max_symbol_set_size,
        max_per_ts_vals=args.max_per_ts,
    )
    if pol.empty:
        raise RuntimeError("no valid policies")

    robust = pol[
        (pol["months"] >= 12)
        & (pol["n"] >= 300)
        & (pol["fold_min_avg_r"] > 0)
    ].copy()
    robust["robust_score"] = (
        robust["mean_monthly_r"]
        + 2.0 * robust["avg_r_net"]
        + 0.02 * robust["win_month_pct"]
    )
    robust = robust.sort_values(
        ["robust_score", "mean_monthly_r", "avg_r_net"],
        ascending=False,
    )

    best_all = pol.sort_values("mean_monthly_r", ascending=False).head(30)
    best_rob = robust.head(30)
    payload = {
        "cost_bps": COST_BPS,
        "oos_rows": int(len(oos)),
        "symbols": sorted(oos["symbol"].unique().tolist()),
        "total_policies": int(len(pol)),
        "best_overall_monthly_r": float(best_all["mean_monthly_r"].max()),
        "best_robust_monthly_r": float(best_rob["mean_monthly_r"].max()) if not best_rob.empty else 0.0,
        "best_overall": best_all.to_dict("records"),
        "best_robust": best_rob.to_dict("records"),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    lines = [
        "# V7 HyperSignal Engine",
        "",
        f"- OOS rows: **{len(oos):,}**",
        f"- Policies tested: **{len(pol):,}**",
        f"- Best monthly R (overall): **{payload['best_overall_monthly_r']:+.2f}**",
        f"- Best monthly R (robust): **{payload['best_robust_monthly_r']:+.2f}**",
        "",
        "## Top robust policies",
        "",
        "| top% | symbols | long_only | p_min | meta_min | risk>=bps | max/ts | session | regime | n | months | avg_R | mean monthly R | win-month% | fold min R |",
        "|---:|---|:---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in best_rob.iterrows():
        lines.append(
            f"| {r['top_pct']:.2f} | {r['symbols']} | "
            f"{'Y' if bool(r['long_only']) else 'N'} | {r['p_min']:.2f} | {r['meta_min']:.2f} | "
            f"{r['risk_min_bps']:.0f} | {int(r['max_per_ts'])} | {r['session']} | {r['regime']} | "
            f"{int(r['n']):,} | {int(r['months'])} | {r['avg_r_net']:+.3f} | "
            f"{r['mean_monthly_r']:+.2f} | {r['win_month_pct']:.1f} | {r['fold_min_avg_r']:+.3f} |"
        )
    OUT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()

