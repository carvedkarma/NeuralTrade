"""V7 SQ2.1: macro-context + portfolio allocator walk-forward simulation.

SQ2.1 extends SQ2 with:
  1) Broader macro/context features:
     - BTC-vs-alt dominance proxy
     - Market breadth/dispersion/risk-on state
     - Shock z-scores and regime context
  2) Fold-local meta tradeability gate:
     - learns P(trade is net-positive after costs)
  3) Portfolio allocator:
     - per-timestamp top-K allocation with score-normalized weights

Run:
  python -m gpu_trainer.eval.v7_signal_quality_sq21 --rebuild-cache
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from gpu_trainer.eval.v7_signal_audit_augmented import (
    DATA_CACHE_DIR,
    build_features,
    build_targets,
    load_symbol,
    walk_forward_indices,
)


COST_BPS = 8.0
COST_FRAC = COST_BPS / 1e4
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT", "XRPUSDT"]
CACHE_ROOT = Path(".local/cache/v7_signal_upgrade_sq21")
OUT_JSON = Path(".local/reports/v7_signal_quality_sq21.json")
OUT_MD = Path(".local/reports/v7_signal_quality_sq21.md")


def _clf(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=260,
        max_depth=5,
        learning_rate=0.035,
        min_samples_leaf=220,
        random_state=seed,
    )


def _reg(seed: int = 0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=260,
        max_depth=5,
        learning_rate=0.035,
        min_samples_leaf=220,
        random_state=seed,
        early_stopping=False,
    )


def _session_from_ts(ts_ms: np.ndarray) -> np.ndarray:
    hour = pd.to_datetime(ts_ms, unit="ms", utc=True).hour.to_numpy()
    out = np.full(len(hour), "Late", dtype=object)
    out[(hour >= 0) & (hour < 7)] = "Asia"
    out[(hour >= 7) & (hour < 14)] = "EU"
    out[(hour >= 14) & (hour < 21)] = "US"
    return out


def _zscore(s: pd.Series, win: int) -> pd.Series:
    mu = s.rolling(win, min_periods=max(8, win // 4)).mean()
    sd = s.rolling(win, min_periods=max(8, win // 4)).std()
    return (s - mu) / sd.replace(0, np.nan)


def _discover_cached_symbols() -> list[str]:
    if not DATA_CACHE_DIR.exists():
        return []
    out = []
    for p in DATA_CACHE_DIR.glob("*_15m.parquet"):
        name = p.name
        if name.endswith("_15m.parquet"):
            out.append(name.replace("_15m.parquet", ""))
    return sorted(set(out))


def _cache_path(symbols: list[str], context_symbols: list[str]) -> Path:
    token = "|".join(
        [
            ",".join(sorted(set(symbols))),
            ",".join(sorted(set(context_symbols))),
            f"cost={COST_BPS:.2f}",
            "sq21",
        ]
    )
    key = hashlib.sha1(token.encode("utf-8")).hexdigest()[:16]
    return CACHE_ROOT / f"oos_{key}.parquet"


def load_symbol_universe(symbols: list[str], min_bars: int) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym in sorted(set(symbols)):
        df = load_symbol(sym)
        if df.empty or len(df) < min_bars:
            continue
        out[sym] = df
    return out


def build_market_context(raw_by_symbol: dict[str, pd.DataFrame], context_symbols: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sym in context_symbols:
        df = raw_by_symbol.get(sym)
        if df is None or df.empty:
            continue
        c = df["close"].astype("float64")
        logc = np.log(c.replace(0, np.nan))
        r1 = logc.diff()
        r4 = logc.diff(4)
        r16 = logc.diff(16)
        v16 = r1.rolling(16).std()
        frame = pd.DataFrame(
            {
                "timestamp": df["timestamp"].to_numpy(),
                "symbol": sym,
                "ret_1": r1.to_numpy(),
                "ret_4": r4.to_numpy(),
                "ret_16": r16.to_numpy(),
                "vol_16": v16.to_numpy(),
            }
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    long_df = pd.concat(frames, ignore_index=True)
    grp = long_df.groupby("timestamp")
    ctx = grp.agg(
        mkt_ret1_mean=("ret_1", "mean"),
        mkt_ret1_std=("ret_1", "std"),
        mkt_ret4_mean=("ret_4", "mean"),
        mkt_ret16_mean=("ret_16", "mean"),
        mkt_vol16_mean=("vol_16", "mean"),
        mkt_n_syms=("symbol", "count"),
    )
    ctx["mkt_breadth_up_1"] = grp["ret_1"].apply(lambda x: float(np.mean(x > 0)))
    ctx["mkt_breadth_up_4"] = grp["ret_4"].apply(lambda x: float(np.mean(x > 0)))
    ctx["mkt_dispersion_z"] = _zscore(ctx["mkt_ret1_std"], 96 * 5)
    ctx["mkt_trend_strength"] = ctx["mkt_ret16_mean"] / ctx["mkt_vol16_mean"].replace(0, np.nan)
    ctx["mkt_shock_z"] = _zscore(ctx["mkt_ret1_mean"].abs(), 96 * 5)
    ctx["risk_on_score"] = ctx["mkt_ret4_mean"] / ctx["mkt_ret1_std"].replace(0, np.nan)

    for anchor in ("BTCUSDT", "ETHUSDT"):
        df = raw_by_symbol.get(anchor)
        if df is None or df.empty:
            continue
        c = df["close"].astype("float64")
        logc = np.log(c.replace(0, np.nan))
        r1 = logc.diff()
        r4 = logc.diff(4)
        col = anchor.lower()
        anchor_df = pd.DataFrame(
            {
                f"{col}_ret1": r1.to_numpy(),
                f"{col}_ret4": r4.to_numpy(),
            },
            index=df["timestamp"].to_numpy(),
        )
        ctx = ctx.join(anchor_df, how="left")

    alt_df = long_df[long_df["symbol"] != "BTCUSDT"]
    if not alt_df.empty:
        alt_grp = alt_df.groupby("timestamp")
        ctx["alts_ret1_mean"] = alt_grp["ret_1"].mean()
        ctx["alts_ret4_mean"] = alt_grp["ret_4"].mean()
        ctx["alts_breadth_up_1"] = alt_grp["ret_1"].apply(lambda x: float(np.mean(x > 0)))
    else:
        ctx["alts_ret1_mean"] = np.nan
        ctx["alts_ret4_mean"] = np.nan
        ctx["alts_breadth_up_1"] = np.nan

    if "btcusdt_ret1" in ctx.columns:
        ctx["btc_dominance_ret1"] = ctx["btcusdt_ret1"] - ctx["alts_ret1_mean"]
    else:
        ctx["btc_dominance_ret1"] = np.nan
    if "btcusdt_ret4" in ctx.columns:
        ctx["btc_dominance_ret4"] = ctx["btcusdt_ret4"] - ctx["alts_ret4_mean"]
    else:
        ctx["btc_dominance_ret4"] = np.nan

    lag_cols = [
        c for c in ctx.columns
        if c.endswith(("ret1", "ret4", "mean", "std", "strength", "score", "z"))
    ]
    for col in lag_cols:
        ctx[f"{col}_lag1"] = ctx[col].shift(1)
        ctx[f"{col}_lag2"] = ctx[col].shift(2)
    return ctx.sort_index()


def build_sq21_features(symbol: str, df: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    out = build_features(df).copy()
    ts = df["timestamp"].to_numpy()
    ctx = context.reindex(ts)

    c = df["close"].astype("float64")
    logc = np.log(c.replace(0, np.nan))
    ret1 = logc.diff()
    ret4 = logc.diff(4)
    ret16 = logc.diff(16)
    vol16 = ret1.rolling(16).std()
    dt = pd.to_datetime(ts, unit="ms", utc=True)

    out["ret1_minus_mkt"] = ret1.to_numpy() - ctx["mkt_ret1_mean"].to_numpy()
    out["ret4_minus_mkt"] = ret4.to_numpy() - ctx["mkt_ret4_mean"].to_numpy()
    out["ret16_minus_mkt"] = ret16.to_numpy() - ctx["mkt_ret16_mean"].to_numpy()
    out["vol16_ratio_mkt"] = vol16.to_numpy() / ctx["mkt_vol16_mean"].to_numpy()
    out["breadth_contra_1"] = -(np.sign(ret1.to_numpy()) * (ctx["mkt_breadth_up_1"].to_numpy() - 0.5))
    out["mkt_dispersion_z"] = ctx["mkt_dispersion_z"].to_numpy()
    out["mkt_trend_strength"] = ctx["mkt_trend_strength"].to_numpy()
    out["mkt_shock_z"] = ctx["mkt_shock_z"].to_numpy()
    out["risk_on_score"] = ctx["risk_on_score"].to_numpy()
    out["alts_breadth_up_1"] = ctx["alts_breadth_up_1"].to_numpy()
    out["btc_dom_ret1"] = ctx["btc_dominance_ret1"].to_numpy()
    out["btc_dom_ret4"] = ctx["btc_dominance_ret4"].to_numpy()

    mkt_ret1 = pd.Series(ctx["mkt_ret1_mean"].to_numpy(), index=df.index)
    beta_cov = ret1.rolling(96).cov(mkt_ret1)
    beta_var = mkt_ret1.rolling(96).var()
    out["beta_mkt_96"] = (beta_cov / beta_var.replace(0, np.nan)).to_numpy()

    for anchor in ("btcusdt", "ethusdt"):
        c1 = f"{anchor}_ret1"
        c4 = f"{anchor}_ret4"
        if c1 in ctx.columns:
            out[f"{anchor}_ret1"] = ctx[c1].to_numpy()
            out[f"{anchor}_ret1_lag1"] = ctx[f"{c1}_lag1"].to_numpy()
            out[f"{anchor}_ret1_lag2"] = ctx[f"{c1}_lag2"].to_numpy()
            out[f"spread_{anchor}_ret1"] = ret1.to_numpy() - ctx[c1].to_numpy()
        if c4 in ctx.columns:
            out[f"{anchor}_ret4"] = ctx[c4].to_numpy()
            out[f"{anchor}_ret4_lag1"] = ctx[f"{c4}_lag1"].to_numpy()

    hour = dt.hour + dt.minute / 60.0
    dow = dt.dayofweek
    out["tod_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["tod_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    if "cvd_delta_z_96" in out.columns:
        out["flow_relmom_interaction"] = out["cvd_delta_z_96"] * out["ret1_minus_mkt"]
    if "aggressor_z_96" in out.columns:
        out["aggr_dispersion_interaction"] = out["aggressor_z_96"] * out["mkt_dispersion_z"]
    if "funding_z" in out.columns and "oi_z_96" in out.columns:
        out["crowding_pressure"] = out["funding_z"] * out["oi_z_96"]
    if "atr_z_96" in out.columns:
        out["atr_vs_dispersion"] = out["atr_z_96"] - out["mkt_dispersion_z"]
    out["dom_shock_interaction"] = out["btc_dom_ret1"] * out["mkt_shock_z"]

    sym_hash = int(hashlib.sha1(symbol.encode("utf-8")).hexdigest()[:8], 16)
    out["symbol_hash"] = float(sym_hash % 997) / 997.0
    return out.replace([np.inf, -np.inf], np.nan)


def _meta_features(
    p_up: np.ndarray,
    pred_mag: np.ndarray,
    edge: np.ndarray,
    risk_bps: np.ndarray,
    mkt_disp: np.ndarray,
    risk_on: np.ndarray,
    dom_ret1: np.ndarray,
) -> np.ndarray:
    return np.column_stack([p_up, pred_mag, edge, np.abs(edge), risk_bps, mkt_disp, risk_on, dom_ret1])


def collect_oos_predictions(symbol: str, df: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    feats = build_sq21_features(symbol, df, context)
    targs = build_targets(df)
    y_ret = targs["ret_60m"].to_numpy()
    y_sign = (y_ret > 0).astype(int)

    feat_warmup = feats.notna().sum(axis=1)
    valid_from = max(int((feat_warmup > 14).idxmax()), 320)

    X = feats.iloc[valid_from:].to_numpy(dtype="float64")
    y_ret = y_ret[valid_from:]
    y_sign = y_sign[valid_from:]
    ts = df["timestamp"].to_numpy()[valid_from:]

    close = df["close"].astype("float64").to_numpy()[valid_from:]
    logret = np.log(pd.Series(close)).diff()
    vol_16 = logret.rolling(16).std().to_numpy()
    trend_16 = (np.log(pd.Series(close)) - np.log(pd.Series(close).shift(16))).to_numpy()

    ctx_slice = context.reindex(ts)
    mkt_disp = ctx_slice["mkt_ret1_std"].to_numpy()
    risk_on = ctx_slice["risk_on_score"].to_numpy()
    dom_ret1 = ctx_slice["btc_dominance_ret1"].to_numpy()

    folds = walk_forward_indices(ts)
    rows: list[dict] = []
    for fold, (tlo, thi, slo, shi) in enumerate(folds):
        X_tr, X_te = X[tlo:thi], X[slo:shi]
        yret_tr, yret_te = y_ret[tlo:thi], y_ret[slo:shi]
        ysgn_tr = y_sign[tlo:thi]

        good = np.isfinite(yret_tr)
        if good.sum() < 1200:
            continue

        clf = _clf(0)
        clf.fit(X_tr[good], ysgn_tr[good])
        reg = _reg(0)
        reg.fit(X_tr[good], np.abs(yret_tr[good]))

        p_up_tr = clf.predict_proba(X_tr)[:, 1]
        p_up_te = clf.predict_proba(X_te)[:, 1]
        mag_tr = np.clip(reg.predict(X_tr), 0.0, None)
        mag_te = np.clip(reg.predict(X_te), 0.0, None)
        edge_tr = (2.0 * p_up_tr - 1.0) * mag_tr
        edge_te = (2.0 * p_up_te - 1.0) * mag_te

        vol_tr = vol_16[tlo:thi]
        vol_te = vol_16[slo:shi]
        disp_tr = mkt_disp[tlo:thi]
        disp_te = mkt_disp[slo:shi]
        risk_on_tr = risk_on[tlo:thi]
        risk_on_te = risk_on[slo:shi]
        dom_tr = dom_ret1[tlo:thi]
        dom_te = dom_ret1[slo:shi]

        side_tr = np.where(edge_tr >= 0, 1.0, -1.0)
        net_pred_tr = side_tr * yret_tr - COST_FRAC
        y_meta_tr = (net_pred_tr > 0).astype(int)

        meta_good = (
            np.isfinite(yret_tr)
            & np.isfinite(edge_tr)
            & np.isfinite(vol_tr)
            & (vol_tr > 1e-8)
            & np.isfinite(disp_tr)
            & np.isfinite(risk_on_tr)
            & np.isfinite(dom_tr)
        )
        meta_te = np.full(len(X_te), 0.5, dtype=float)
        if meta_good.sum() > 1500:
            X_meta_tr = _meta_features(
                p_up=p_up_tr[meta_good],
                pred_mag=mag_tr[meta_good],
                edge=edge_tr[meta_good],
                risk_bps=vol_tr[meta_good] * 1e4,
                mkt_disp=disp_tr[meta_good],
                risk_on=risk_on_tr[meta_good],
                dom_ret1=dom_tr[meta_good],
            )
            mclf = _clf(41)
            mclf.fit(X_meta_tr, y_meta_tr[meta_good])

            te_good_for_meta = (
                np.isfinite(vol_te)
                & (vol_te > 1e-8)
                & np.isfinite(edge_te)
                & np.isfinite(disp_te)
                & np.isfinite(risk_on_te)
                & np.isfinite(dom_te)
            )
            X_meta_te = _meta_features(
                p_up=p_up_te[te_good_for_meta],
                pred_mag=mag_te[te_good_for_meta],
                edge=edge_te[te_good_for_meta],
                risk_bps=vol_te[te_good_for_meta] * 1e4,
                mkt_disp=disp_te[te_good_for_meta],
                risk_on=risk_on_te[te_good_for_meta],
                dom_ret1=dom_te[te_good_for_meta],
            )
            meta_te[te_good_for_meta] = mclf.predict_proba(X_meta_te)[:, 1]

        sess = _session_from_ts(ts[slo:shi])
        side = np.sign(edge_te)
        sign_trend = np.sign(trend_16[slo:shi])
        regime = np.where(sign_trend == 0, "FLAT", np.where(sign_trend == side, "WITH", "COUNTER"))

        good_te = (
            np.isfinite(yret_te)
            & np.isfinite(p_up_te)
            & np.isfinite(mag_te)
            & np.isfinite(edge_te)
            & np.isfinite(meta_te)
            & np.isfinite(vol_te)
            & (vol_te > 1e-8)
            & np.isfinite(disp_te)
            & np.isfinite(risk_on_te)
            & np.isfinite(dom_te)
        )
        idx = np.where(good_te)[0]
        for i in idx:
            rows.append(
                {
                    "symbol": symbol,
                    "fold": int(fold),
                    "ts": int(ts[slo + i]),
                    "ret_60m": float(yret_te[i]),
                    "vol_16": float(vol_te[i]),
                    "risk_bps": float(vol_te[i] * 1e4),
                    "p_up": float(p_up_te[i]),
                    "pred_mag": float(mag_te[i]),
                    "edge": float(edge_te[i]),
                    "abs_edge": float(abs(edge_te[i])),
                    "meta_p": float(meta_te[i]),
                    "mkt_dispersion": float(disp_te[i]),
                    "risk_on_score": float(risk_on_te[i]),
                    "btc_dom_ret1": float(dom_te[i]),
                    "session": str(sess[i]),
                    "regime": str(regime[i]),
                }
            )
    return pd.DataFrame(rows)


def ensure_oos_cache(
    symbols: list[str],
    context_symbols: list[str],
    rebuild_cache: bool = False,
    min_bars: int = 20000,
) -> tuple[pd.DataFrame, list[str]]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(symbols, context_symbols)
    if cache_path.exists() and not rebuild_cache:
        return pd.read_parquet(cache_path), sorted(set(context_symbols))

    universe = sorted(set(symbols) | set(context_symbols))
    raw_by_symbol = load_symbol_universe(universe, min_bars=min_bars)
    usable_symbols = [s for s in symbols if s in raw_by_symbol]
    usable_context = [s for s in context_symbols if s in raw_by_symbol]
    if not usable_symbols or not usable_context:
        return pd.DataFrame(), []

    context = build_market_context(raw_by_symbol, usable_context)
    if context.empty:
        return pd.DataFrame(), []

    frames = []
    for sym in usable_symbols:
        s = collect_oos_predictions(sym, raw_by_symbol[sym], context)
        if not s.empty:
            frames.append(s)
    if not frames:
        return pd.DataFrame(), usable_context

    out = pd.concat(frames, ignore_index=True)
    out["month"] = pd.to_datetime(out["ts"], unit="ms", utc=True).dt.to_period("M").astype(str)
    out["rank_pct"] = out.groupby(["symbol", "fold"])["abs_edge"].rank(pct=True, method="average")
    out.to_parquet(cache_path, index=False)
    return out, usable_context


@dataclass
class PolicyResult:
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
    trade_win_rate_pct: float
    weighted_win_rate_pct: float
    mean_net_bps: float
    avg_r_net: float
    mean_monthly_total_r: float
    median_monthly_total_r: float
    total_r: float
    win_month_pct: float
    fold_min_avg_r: float
    fold_pos: int
    folds: int


def _apply_policy_and_allocate(
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
) -> pd.DataFrame:
    x = d[d["symbol"].isin(symbols)]
    if x.empty:
        return pd.DataFrame()

    q_cut = 1.0 - (top_pct / 100.0)
    x = x[x["rank_pct"] >= q_cut]
    if session is not None:
        x = x[x["session"] == session]
    if regime is not None:
        x = x[x["regime"] == regime]
    x = x[(x["risk_bps"] >= risk_min_bps) & (x["meta_p"] >= meta_min)].copy()
    if x.empty:
        return pd.DataFrame()

    if long_only:
        x = x[(x["edge"] > 0) & (x["p_up"] >= p_min)].copy()
        x["side"] = 1.0
    else:
        long_mask = (x["edge"] > 0) & (x["p_up"] >= p_min)
        short_mask = (x["edge"] < 0) & (x["p_up"] <= (1.0 - p_min))
        x = x[long_mask | short_mask].copy()
        x["side"] = np.where(x["edge"].to_numpy() > 0, 1.0, -1.0)
    if x.empty:
        return pd.DataFrame()

    x["allocator_score"] = (x["abs_edge"] * np.clip(x["meta_p"], 0.0, 1.0)).astype("float64")
    x = x.sort_values(["ts", "allocator_score"], ascending=[True, False])
    if max_per_ts > 0:
        x = x.groupby("ts", as_index=False).head(max_per_ts).copy()

    score_sum = x.groupby("ts")["allocator_score"].transform("sum")
    x["weight"] = np.where(score_sum > 1e-12, x["allocator_score"] / score_sum, 1.0)
    ts_count = x.groupby("ts")["symbol"].transform("count").replace(0, np.nan)
    x["weight"] = np.where(np.isfinite(x["weight"]), x["weight"], 1.0 / ts_count)

    x["gross"] = x["ret_60m"] * x["side"]
    x["net"] = x["gross"] - COST_FRAC
    x["r_net"] = x["net"] / x["vol_16"]
    x["r_weighted"] = x["r_net"] * x["weight"]
    x["net_bps_weighted"] = x["net"] * x["weight"] * 1e4
    return x


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
) -> PolicyResult | None:
    x = _apply_policy_and_allocate(
        d=d,
        top_pct=top_pct,
        symbols=symbols,
        session=session,
        regime=regime,
        p_min=p_min,
        meta_min=meta_min,
        long_only=long_only,
        risk_min_bps=risk_min_bps,
        max_per_ts=max_per_ts,
    )
    if len(x) < 120:
        return None

    msum = x.groupby("month")["r_weighted"].sum()
    if len(msum) < 10:
        return None
    fold_avg = x.groupby("fold")["r_weighted"].mean()
    weighted_win = x.loc[x["net"] > 0, "weight"].sum() / max(1e-12, x["weight"].sum())
    return PolicyResult(
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
        months=int(len(msum)),
        trade_win_rate_pct=float((x["net"] > 0).mean() * 100),
        weighted_win_rate_pct=float(weighted_win * 100),
        mean_net_bps=float(x["net_bps_weighted"].mean()),
        avg_r_net=float(x["r_weighted"].mean()),
        mean_monthly_total_r=float(msum.mean()),
        median_monthly_total_r=float(msum.median()),
        total_r=float(msum.sum()),
        win_month_pct=float((msum > 0).mean() * 100),
        fold_min_avg_r=float(fold_avg.min()),
        fold_pos=int((fold_avg > 0).sum()),
        folds=int(len(fold_avg)),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V7 SQ2.1 macro + allocator WF search")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="Symbol list to evaluate")
    p.add_argument("--context-symbols", nargs="+", default=[],
                   help="Symbols used to build market context (default: same as --symbols)")
    p.add_argument("--auto-symbols-from-cache", action="store_true",
                   help="Union cached parquet symbols into --symbols and context symbols")
    p.add_argument("--min-bars", type=int, default=20000,
                   help="Minimum bars required per symbol")
    p.add_argument("--top-pcts", nargs="+", type=float, default=[1.0, 2.0])
    p.add_argument("--p-mins", nargs="+", type=float, default=[0.55, 0.58, 0.60])
    p.add_argument("--meta-mins", nargs="+", type=float, default=[0.50, 0.55, 0.60])
    p.add_argument("--risk-min-bps", nargs="+", type=float, default=[40, 60, 80])
    p.add_argument("--max-per-ts", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--sessions", nargs="+", default=["ALL", "EU", "US"])
    p.add_argument("--regimes", nargs="+", default=["ALL", "WITH", "COUNTER"])
    p.add_argument("--long-only-only", action="store_true",
                   help="Only evaluate long-only policies")
    p.add_argument("--max-symbol-set-size", type=int, default=5,
                   help="Max symbols in subset")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Force rebuild OOS cache for selected symbols/context")
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
                        for meta_min in args.meta_mins:
                            for long_only in long_flags:
                                for risk_min in args.risk_min_bps:
                                    for max_per_ts in args.max_per_ts:
                                        r = eval_policy(
                                            d=d,
                                            top_pct=top_pct,
                                            symbols=symbols,
                                            session=session,
                                            regime=regime,
                                            p_min=p_min,
                                            meta_min=meta_min,
                                            long_only=long_only,
                                            risk_min_bps=risk_min,
                                            max_per_ts=max_per_ts,
                                        )
                                        if r is None:
                                            continue
                                        results.append(asdict(r))
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


def _monthly_breakdown_for_policy(d: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    symbols = tuple(str(row["symbols"]).split(","))
    session = None if row["session"] == "ALL" else str(row["session"])
    regime = None if row["regime"] == "ALL" else str(row["regime"])
    x = _apply_policy_and_allocate(
        d=d,
        top_pct=float(row["top_pct"]),
        symbols=symbols,
        session=session,
        regime=regime,
        p_min=float(row["p_min"]),
        meta_min=float(row["meta_min"]),
        long_only=bool(row["long_only"]),
        risk_min_bps=float(row["risk_min_bps"]),
        max_per_ts=int(row["max_per_ts"]),
    )
    if x.empty:
        return pd.DataFrame()
    out = x.groupby("month").agg(
        trades=("symbol", "size"),
        month_r=("r_weighted", "sum"),
        trade_win_rate_pct=("net", lambda s: float((s > 0).mean() * 100)),
        mean_weighted_r=("r_weighted", "mean"),
    )
    out = out.reset_index().sort_values("month")
    return out


def main() -> None:
    args = parse_args()

    symbols = list(dict.fromkeys(args.symbols))
    context_symbols = list(dict.fromkeys(args.context_symbols)) or symbols.copy()
    if args.auto_symbols_from_cache:
        cached_symbols = _discover_cached_symbols()
        symbols = sorted(set(symbols) | set(cached_symbols))
        context_symbols = sorted(set(context_symbols) | set(cached_symbols))

    oos, usable_context = ensure_oos_cache(
        symbols=symbols,
        context_symbols=context_symbols,
        rebuild_cache=args.rebuild_cache,
        min_bars=args.min_bars,
    )
    if oos.empty:
        raise RuntimeError("no OOS predictions")

    policies = search_policies_with_args(oos, args)
    if policies.empty:
        raise RuntimeError("no valid policies evaluated")

    robust = policies[
        (policies["months"] >= 10)
        & (policies["n"] >= 300)
        & (policies["fold_min_avg_r"] > 0)
    ].copy()
    robust["robust_score"] = (
        robust["mean_monthly_total_r"]
        + 1.7 * robust["avg_r_net"]
        + 0.025 * robust["win_month_pct"]
        + 0.01 * robust["weighted_win_rate_pct"]
    )
    robust = robust.sort_values(
        ["robust_score", "mean_monthly_total_r", "avg_r_net"],
        ascending=False,
    )

    best_overall = policies.sort_values("mean_monthly_total_r", ascending=False).head(25)
    best_robust = robust.head(25)
    chosen = best_robust.iloc[0] if not best_robust.empty else best_overall.iloc[0]
    monthly = _monthly_breakdown_for_policy(oos, chosen)

    if len(monthly) >= 10:
        last10 = monthly.tail(10)
        last10_total_r = float(last10["month_r"].sum())
        last10_win_month_pct = float((last10["month_r"] > 0).mean() * 100)
    else:
        last10_total_r = 0.0
        last10_win_month_pct = 0.0

    ceiling = float(best_overall["mean_monthly_total_r"].max())
    robust_ceiling = float(best_robust["mean_monthly_total_r"].max()) if not best_robust.empty else 0.0
    payload = {
        "feature_pack": "sq21-macro-context-meta-allocator",
        "cost_bps": COST_BPS,
        "oos_rows": int(len(oos)),
        "symbols": sorted(oos["symbol"].unique().tolist()),
        "context_symbols": usable_context,
        "total_policies": int(len(policies)),
        "best_overall_monthly_r": ceiling,
        "best_robust_monthly_r": robust_ceiling,
        "best_overall": best_overall.to_dict("records"),
        "best_robust": best_robust.to_dict("records"),
        "selected_policy_for_monthly_view": chosen.to_dict(),
        "monthly_breakdown_selected_policy": monthly.to_dict("records"),
        "last10_months_total_r": last10_total_r,
        "last10_months_win_month_pct": last10_win_month_pct,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    lines = [
        "# V7 SQ2.1 Signal Quality Upgrade",
        "",
        "- Feature pack: **macro context + meta tradeability gate + top-K allocator**",
        f"- OOS rows: **{len(oos):,}**",
        f"- Active symbols: **{', '.join(sorted(oos['symbol'].unique()))}**",
        f"- Context symbols: **{', '.join(usable_context)}**",
        f"- Policies tested: **{len(policies):,}**",
        f"- Best monthly R (overall): **{ceiling:+.2f}**",
        f"- Best monthly R (robust): **{robust_ceiling:+.2f}**",
        "",
        "## Top robust policies",
        "",
        "| top% | symbols | long_only | p_min | meta_min | risk>=bps | max/ts | n | months | trade win% | weighted win% | avg_R | mean monthly R | total R | win-month% | fold min R |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in best_robust.iterrows():
        lines.append(
            f"| {r['top_pct']:.1f} | {r['symbols']} | "
            f"{'Y' if bool(r['long_only']) else 'N'} | {r['p_min']:.2f} | {r['meta_min']:.2f} | "
            f"{r['risk_min_bps']:.0f} | {int(r['max_per_ts'])} | "
            f"{int(r['n']):,} | {int(r['months'])} | "
            f"{r['trade_win_rate_pct']:.1f} | {r['weighted_win_rate_pct']:.1f} | "
            f"{r['avg_r_net']:+.3f} | {r['mean_monthly_total_r']:+.2f} | {r['total_r']:+.2f} | "
            f"{r['win_month_pct']:.1f} | {r['fold_min_avg_r']:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## Monthly breakdown (selected best robust policy)",
            "",
            "| month | trades | month R | month trade win% | mean weighted R |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, r in monthly.iterrows():
        lines.append(
            f"| {r['month']} | {int(r['trades']):,} | {r['month_r']:+.2f} | "
            f"{r['trade_win_rate_pct']:.1f} | {r['mean_weighted_r']:+.3f} |"
        )
    if len(monthly) >= 10:
        lines.extend(
            [
                "",
                f"- Last 10 months total R: **{last10_total_r:+.2f}**",
                f"- Last 10 months positive-month rate: **{last10_win_month_pct:.1f}%**",
            ]
        )
    OUT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
