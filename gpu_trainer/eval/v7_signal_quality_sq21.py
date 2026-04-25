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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

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


def _horizon_to_bars(h: str) -> int:
    key = str(h).strip().lower()
    if key == "15m":
        return 1
    if key == "30m":
        return 2
    return 4


def _forward_return_from_close(close: pd.Series, bars: int) -> pd.Series:
    logc = np.log(pd.to_numeric(close, errors="coerce").replace(0, np.nan))
    return logc.shift(-int(max(1, bars))) - logc


def _parse_symbol_regime_pairs(values: list[str]) -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, str]] = []
    for raw in values:
        txt = str(raw).strip()
        if ":" not in txt:
            continue
        sym, reg = txt.split(":", 1)
        sym_u = sym.strip().upper()
        reg_u = reg.strip().upper()
        if sym_u and reg_u:
            out.append((sym_u, reg_u))
    return tuple(sorted(set(out)))


def _normalize_horizon(h: str) -> str:
    key = str(h).strip().lower()
    if key in {"15m", "30m", "60m"}:
        return key
    return "60m"


def _tail_sample_weights(
    y: np.ndarray,
    tail_weight: float,
    shock: np.ndarray | None = None,
    dispersion: np.ndarray | None = None,
) -> np.ndarray:
    w = np.ones(len(y), dtype="float64")
    tw = float(max(0.0, tail_weight))
    if tw <= 1e-9 or len(y) == 0:
        return w
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype="float64")
    finite = np.isfinite(y)
    if not finite.any():
        return w
    scale = float(np.nanpercentile(np.abs(y[finite]), 75))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    downside = np.clip((-y) / scale, 0.0, 3.0)
    downside = np.where(np.isfinite(downside), downside, 0.0)

    tail_state = np.zeros(len(y), dtype="float64")
    if shock is not None:
        sh = pd.to_numeric(pd.Series(shock), errors="coerce").abs().to_numpy(dtype="float64")
        if np.isfinite(sh).any():
            sh_cut = float(np.nanpercentile(sh[np.isfinite(sh)], 70))
            tail_state += np.where(np.isfinite(sh) & (sh >= sh_cut), 1.0, 0.0)
    if dispersion is not None:
        dsp = pd.to_numeric(pd.Series(dispersion), errors="coerce").to_numpy(dtype="float64")
        if np.isfinite(dsp).any():
            dsp_cut = float(np.nanpercentile(dsp[np.isfinite(dsp)], 70))
            tail_state += np.where(np.isfinite(dsp) & (dsp >= dsp_cut), 1.0, 0.0)
    tail_state = np.clip(tail_state, 0.0, 2.0)
    return w + tw * (0.65 * downside + 0.35 * tail_state)


def _utility_asymmetric_weights(y: np.ndarray, downside_mult: float = 1.0) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype="float64")
    w = np.ones(len(arr), dtype="float64")
    dm = float(max(1.0, downside_mult))
    if dm <= 1.000001:
        return w
    neg = np.isfinite(arr) & (arr < 0.0)
    # Emphasize downside utility outcomes to reduce false-positive profitability.
    w[neg] = dm
    return w


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


def _cache_path(
    symbols: list[str],
    context_symbols: list[str],
    model_family: str,
    sq3_seq_weight: float,
    sq3_seq_window: int,
    wf_train_months: int,
    wf_test_months: int,
    wf_folds: int,
    fast_oos_mode: bool,
    primary_horizon: str,
    secondary_horizon: str,
    secondary_horizon_weight: float,
    tail_objective_weight: float,
    utility_label_mode: str = "net_r",
    utility_downside_mult: float = 1.0,
) -> Path:
    token = "|".join(
        [
            ",".join(sorted(set(symbols))),
            ",".join(sorted(set(context_symbols))),
            f"cost={COST_BPS:.2f}",
            f"model={model_family}",
            f"sq3w={float(sq3_seq_weight):.3f}",
            f"sq3win={int(sq3_seq_window)}",
            f"wftr={int(wf_train_months)}",
            f"wfte={int(wf_test_months)}",
            f"wff={int(wf_folds)}",
            f"fast={int(bool(fast_oos_mode))}",
            f"h1={str(primary_horizon).lower()}",
            f"h2={str(secondary_horizon).lower()}",
            f"h2w={float(secondary_horizon_weight):.3f}",
            f"tail={float(tail_objective_weight):.3f}",
            f"util={str(utility_label_mode).lower()}",
            f"udm={float(max(1.0, utility_downside_mult)):.2f}",
            "sq21-intel-moe-v2",
        ]
    )
    key = hashlib.sha1(token.encode("utf-8")).hexdigest()[:16]
    return CACHE_ROOT / f"oos_{key}.parquet"


def _resolve_target_col(primary_horizon: str) -> str:
    _ = _normalize_horizon(primary_horizon)
    # ret_h1 is always generated from the configured primary horizon.
    return "ret_h1"


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
    expert_dispersion: np.ndarray,
    router_conf: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            p_up,
            pred_mag,
            edge,
            np.abs(edge),
            risk_bps,
            mkt_disp,
            risk_on,
            dom_ret1,
            expert_dispersion,
            router_conf,
        ]
    )


def _utility_features(
    p_up: np.ndarray,
    pred_mag: np.ndarray,
    edge: np.ndarray,
    meta_p: np.ndarray,
    risk_bps: np.ndarray,
    mkt_disp: np.ndarray,
    risk_on: np.ndarray,
    dom_ret1: np.ndarray,
    expert_dispersion: np.ndarray,
    router_conf: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            p_up,
            pred_mag,
            edge,
            np.abs(edge),
            np.clip(meta_p, 0.0, 1.0),
            risk_bps,
            mkt_disp,
            risk_on,
            dom_ret1,
            expert_dispersion,
            router_conf,
        ]
    )


def _regime_labels_from_context(risk_on: np.ndarray, shock_z: np.ndarray) -> np.ndarray:
    labels = np.full(len(risk_on), -1, dtype=np.int8)
    valid = np.isfinite(risk_on) & np.isfinite(shock_z)
    if not valid.any():
        return labels

    risk_abs = np.abs(risk_on)
    risk_cut = np.nanquantile(risk_abs[valid], 0.65)
    shock_abs = np.abs(shock_z)

    labels[valid & (shock_abs >= 1.8)] = 2  # shock regime
    labels[valid & (shock_abs < 1.8) & (risk_abs >= risk_cut)] = 1  # directional/trending
    labels[valid & (shock_abs < 1.8) & (risk_abs < risk_cut)] = 0  # calm/chop
    return labels


def _predict_regime_moe(
    X_in: np.ndarray,
    base_clf: HistGradientBoostingClassifier,
    base_reg: HistGradientBoostingRegressor,
    experts: dict[int, tuple[HistGradientBoostingClassifier, HistGradientBoostingRegressor]],
    router: HistGradientBoostingClassifier | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p_out = base_clf.predict_proba(X_in)[:, 1]
    mag_out = np.clip(base_reg.predict(X_in), 0.0, None)
    router_conf = np.full(len(X_in), 0.5, dtype="float64")

    if not experts:
        return p_out, mag_out, np.zeros(len(X_in), dtype="float64"), router_conf

    p_stack: list[np.ndarray] = []
    if router is not None:
        router_proba = router.predict_proba(X_in)
        cls_to_idx = {int(c): i for i, c in enumerate(router.classes_)}
        router_conf = np.max(router_proba, axis=1)

        p_blend = np.zeros(len(X_in), dtype="float64")
        m_blend = np.zeros(len(X_in), dtype="float64")
        wsum = np.zeros(len(X_in), dtype="float64")

        for reg_id, (eclf, ereg) in experts.items():
            p_e = eclf.predict_proba(X_in)[:, 1]
            m_e = np.clip(ereg.predict(X_in), 0.0, None)
            p_stack.append(p_e)
            idx = cls_to_idx.get(int(reg_id))
            if idx is None:
                continue
            w = router_proba[:, idx]
            p_blend += w * p_e
            m_blend += w * m_e
            wsum += w

        use = wsum > 1e-6
        p_out[use] = p_blend[use] / wsum[use]
        mag_out[use] = m_blend[use] / wsum[use]
    else:
        for _, (eclf, _) in experts.items():
            p_stack.append(eclf.predict_proba(X_in)[:, 1])

    if len(p_stack) >= 2:
        p_mat = np.column_stack(p_stack)
        expert_disp = np.std(p_mat, axis=1)
    else:
        expert_disp = np.zeros(len(X_in), dtype="float64")

    return p_out, mag_out, expert_disp, router_conf


def _seq_expand(X: np.ndarray, seq_window: int) -> np.ndarray:
    w = max(1, int(seq_window))
    if w <= 1:
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    base = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    lags = sorted({1, 2, 4, 8, w})
    parts = [base]
    for lag in lags:
        if lag <= 0:
            continue
        shifted = np.vstack([np.zeros((lag, base.shape[1])), base[:-lag]])
        parts.append(shifted)
        parts.append(base - shifted)
    out = np.column_stack(parts)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def collect_oos_predictions(
    symbol: str,
    df: pd.DataFrame,
    context: pd.DataFrame,
    use_sq3: bool = False,
    seq_weight: float = 0.55,
    seq_window: int = 12,
    wf_train_months: int = 24,
    wf_test_months: int = 6,
    wf_folds: int = 5,
    fast_oos_mode: bool = False,
    fast_max_bars: int = 20000,
    primary_horizon: str = "15m",
    secondary_horizon: str = "30m",
    secondary_horizon_weight: float = 0.35,
    tail_objective_weight: float = 0.0,
    utility_label_mode: str = "net_r",
    utility_downside_mult: float = 1.0,
) -> pd.DataFrame:
    feats = build_sq21_features(symbol, df, context)
    h1_bars = _horizon_to_bars(primary_horizon)
    h2_bars = _horizon_to_bars(secondary_horizon)
    close_s = df["close"].astype("float64")
    y_ret_1 = _forward_return_from_close(close_s, h1_bars)
    y_ret_2 = _forward_return_from_close(close_s, h2_bars)
    sec_w = float(np.clip(secondary_horizon_weight, 0.0, 1.0))
    y_ret = ((1.0 - sec_w) * y_ret_1 + sec_w * y_ret_2).to_numpy(dtype="float64")
    y_sign = (y_ret > 0).astype(int)

    feat_warmup = feats.notna().sum(axis=1)
    valid_from = max(int((feat_warmup > 14).idxmax()), 320)

    X_base = feats.iloc[valid_from:].to_numpy(dtype="float64")
    y_ret = y_ret[valid_from:]
    y_sign = y_sign[valid_from:]
    ts = df["timestamp"].to_numpy()[valid_from:]
    if fast_oos_mode and int(fast_max_bars) > 0 and len(ts) > int(fast_max_bars):
        # Keep fast mode from truncating below the configured WF horizon.
        wf_min_bars = int((int(wf_train_months) + int(wf_test_months)) * 30 * 24 * 4 + 512)
        keep = max(int(fast_max_bars), wf_min_bars)
        keep = min(len(ts), keep)
        X_base = X_base[-keep:]
        y_ret = y_ret[-keep:]
        y_sign = y_sign[-keep:]
        ts = ts[-keep:]
    X_seq = _seq_expand(X_base, seq_window=seq_window) if use_sq3 else np.empty((len(X_base), 0))
    X = X_base

    close = df["close"].astype("float64").to_numpy()[valid_from:]
    logret = np.log(pd.Series(close)).diff()
    vol_16 = logret.rolling(16).std().to_numpy()
    trend_16 = (np.log(pd.Series(close)) - np.log(pd.Series(close).shift(16))).to_numpy()

    ctx_slice = context.reindex(ts)
    mkt_disp = ctx_slice["mkt_ret1_std"].to_numpy()
    risk_on = ctx_slice["risk_on_score"].to_numpy()
    shock_z = ctx_slice["mkt_shock_z"].to_numpy()
    dom_ret1 = ctx_slice["btc_dominance_ret1"].to_numpy()
    regime_lbl = _regime_labels_from_context(risk_on=risk_on, shock_z=shock_z)

    folds = walk_forward_indices(
        ts,
        train_months=int(wf_train_months),
        test_months=int(wf_test_months),
        n_folds=int(wf_folds),
    )
    rows: list[dict] = []
    for fold, (tlo, thi, slo, shi) in enumerate(folds):
        X_tr, X_te = X[tlo:thi], X[slo:shi]
        Xs_tr = X_seq[tlo:thi] if use_sq3 else np.empty((thi - tlo, 0))
        Xs_te = X_seq[slo:shi] if use_sq3 else np.empty((shi - slo, 0))
        yret_tr, yret_te = y_ret[tlo:thi], y_ret[slo:shi]
        ysgn_tr = y_sign[tlo:thi]

        good = np.isfinite(yret_tr)
        if good.sum() < 1200:
            continue

        if fast_oos_mode:
            # Fast all-symbol path: linear learners drastically reduce OOS build time.
            good_idx = np.where(good)[0]
            if good_idx.size > 7000:
                take = np.random.default_rng(41 + fold).choice(good_idx, size=7000, replace=False)
            else:
                take = good_idx
            X_fit = X_tr[take]
            y_fit_cls = ysgn_tr[take]
            y_fit_reg = np.abs(yret_tr[take])
            med = np.nanmedian(X_fit, axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            X_fit_i = np.where(np.isfinite(X_fit), X_fit, med)
            X_tr_i = np.where(np.isfinite(X_tr), X_tr, med)
            X_te_i = np.where(np.isfinite(X_te), X_te, med)
            sd = np.nanstd(X_fit_i, axis=0)
            sd = np.where(sd > 1e-9, sd, 1.0)
            lo = med - 8.0 * sd
            hi = med + 8.0 * sd
            X_fit_i = np.clip(X_fit_i, lo, hi)
            X_tr_i = np.clip(X_tr_i, lo, hi)
            X_te_i = np.clip(X_te_i, lo, hi)
            scaler = StandardScaler()
            X_fit_s = scaler.fit_transform(X_fit_i)
            X_tr_s = scaler.transform(X_tr_i)
            X_te_s = scaler.transform(X_te_i)

            p_base = float(np.clip(y_fit_cls.mean(), 0.01, 0.99))
            p_up_tr = np.full(len(X_tr), p_base, dtype="float64")
            p_up_te = np.full(len(X_te), p_base, dtype="float64")
            if np.unique(y_fit_cls).size >= 2:
                clf_fast = LogisticRegression(C=0.8, max_iter=350, solver="lbfgs")
                fit_sw = _tail_sample_weights(yret_tr[take], tail_objective_weight)
                clf_fast.fit(X_fit_s, y_fit_cls, sample_weight=fit_sw)
                p_up_tr = clf_fast.predict_proba(X_tr_s)[:, 1]
                p_up_te = clf_fast.predict_proba(X_te_s)[:, 1]

            reg_fast = Ridge(alpha=2.0)
            fit_sw = _tail_sample_weights(yret_tr[take], tail_objective_weight)
            reg_fast.fit(X_fit_s, y_fit_reg, sample_weight=fit_sw)
            mag_tr = np.clip(reg_fast.predict(X_tr_s), 0.0, None)
            mag_te = np.clip(reg_fast.predict(X_te_s), 0.0, None)
            p_disp_tr = np.zeros(len(X_tr), dtype="float64")
            p_disp_te = np.zeros(len(X_te), dtype="float64")
            router_conf_tr = np.clip(np.abs(p_up_tr - 0.5) * 2.0, 0.0, 1.0)
            router_conf_te = np.clip(np.abs(p_up_te - 0.5) * 2.0, 0.0, 1.0)
        else:
            clf = _clf(11)
            tr_sw = _tail_sample_weights(yret_tr[good], tail_objective_weight)
            clf.fit(X_tr[good], ysgn_tr[good], sample_weight=tr_sw)
            reg = _reg(11)
            reg.fit(X_tr[good], np.abs(yret_tr[good]), sample_weight=tr_sw)

            reg_tr = regime_lbl[tlo:thi]
            experts: dict[int, tuple[HistGradientBoostingClassifier, HistGradientBoostingRegressor]] = {}
            for reg_id in (0, 1, 2):
                rmask = good & (reg_tr == reg_id)
                if int(rmask.sum()) < 700:
                    continue
                eclf = _clf(101 + reg_id)
                reg_sw = _tail_sample_weights(yret_tr[rmask], tail_objective_weight)
                eclf.fit(X_tr[rmask], ysgn_tr[rmask], sample_weight=reg_sw)
                ereg = _reg(101 + reg_id)
                ereg.fit(X_tr[rmask], np.abs(yret_tr[rmask]), sample_weight=reg_sw)
                experts[reg_id] = (eclf, ereg)

            router: HistGradientBoostingClassifier | None = None
            router_mask = good & np.isin(reg_tr, np.array([0, 1, 2], dtype=np.int8))
            if int(router_mask.sum()) > 1800 and np.unique(reg_tr[router_mask]).size >= 2:
                router = HistGradientBoostingClassifier(
                    max_iter=180,
                    max_depth=4,
                    learning_rate=0.04,
                    min_samples_leaf=240,
                    random_state=73,
                )
                router.fit(X_tr[router_mask], reg_tr[router_mask])

            p_up_tr, mag_tr, p_disp_tr, router_conf_tr = _predict_regime_moe(
                X_in=X_tr,
                base_clf=clf,
                base_reg=reg,
                experts=experts,
                router=router,
            )
            p_up_te, mag_te, p_disp_te, router_conf_te = _predict_regime_moe(
                X_in=X_te,
                base_clf=clf,
                base_reg=reg,
                experts=experts,
                router=router,
            )
        if use_sq3 and Xs_tr.shape[1] > 0:
            seq_good = good.copy()
            p_seq_tr = p_up_tr.copy()
            p_seq_te = p_up_te.copy()
            m_seq_tr = mag_tr.copy()
            m_seq_te = mag_te.copy()
            if (
                int(seq_good.sum()) > 1800
                and np.unique(ysgn_tr[seq_good]).size >= 2
            ):
                try:
                    scaler = StandardScaler()
                    Xn_tr = scaler.fit_transform(Xs_tr[seq_good])
                    Xn_te = scaler.transform(Xs_te)

                    s_clf = LogisticRegression(
                        C=0.7,
                        max_iter=500,
                        solver="lbfgs",
                    )
                    seq_sw = _tail_sample_weights(yret_tr[seq_good], tail_objective_weight)
                    s_clf.fit(Xn_tr, ysgn_tr[seq_good], sample_weight=seq_sw)
                    p_seq_tr[seq_good] = s_clf.predict_proba(Xn_tr)[:, 1]
                    p_seq_te = s_clf.predict_proba(Xn_te)[:, 1]

                    s_reg = Ridge(alpha=2.0)
                    s_reg.fit(Xn_tr, np.abs(yret_tr[seq_good]), sample_weight=seq_sw)
                    m_seq_tr[seq_good] = np.clip(s_reg.predict(Xn_tr), 0.0, None)
                    m_seq_te = np.clip(s_reg.predict(Xn_te), 0.0, None)
                except Exception:
                    p_seq_tr = p_up_tr.copy()
                    p_seq_te = p_up_te.copy()
                    m_seq_tr = mag_tr.copy()
                    m_seq_te = mag_te.copy()

            w = float(np.clip(seq_weight, 0.0, 0.90))
            base_ptr = p_up_tr.copy()
            base_pte = p_up_te.copy()
            p_up_tr = ((1.0 - w) * p_up_tr) + (w * p_seq_tr)
            p_up_te = ((1.0 - w) * p_up_te) + (w * p_seq_te)
            mag_tr = np.clip(((1.0 - w) * mag_tr) + (w * m_seq_tr), 0.0, None)
            mag_te = np.clip(((1.0 - w) * mag_te) + (w * m_seq_te), 0.0, None)

            seq_conf_tr = np.clip(np.abs(p_seq_tr - 0.5) * 2.0, 0.0, 1.0)
            seq_conf_te = np.clip(np.abs(p_seq_te - 0.5) * 2.0, 0.0, 1.0)
            router_conf_tr = np.clip(0.70 * router_conf_tr + 0.30 * seq_conf_tr, 0.0, 1.0)
            router_conf_te = np.clip(0.70 * router_conf_te + 0.30 * seq_conf_te, 0.0, 1.0)
            p_disp_tr = np.sqrt(np.clip((p_disp_tr ** 2) + ((base_ptr - p_seq_tr) ** 2), 0.0, None))
            p_disp_te = np.sqrt(np.clip((p_disp_te ** 2) + ((base_pte - p_seq_te) ** 2), 0.0, None))

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
        util_mode = str(utility_label_mode).strip().lower()
        if util_mode == "downside_penalized":
            dm = float(max(1.0, utility_downside_mult))
            y_utility_tr = np.where(net_pred_tr < 0.0, net_pred_tr * dm, net_pred_tr)
        else:
            y_utility_tr = net_pred_tr

        meta_good = (
            np.isfinite(yret_tr)
            & np.isfinite(edge_tr)
            & np.isfinite(vol_tr)
            & (vol_tr > 1e-8)
            & np.isfinite(disp_tr)
            & np.isfinite(risk_on_tr)
            & np.isfinite(dom_tr)
            & np.isfinite(p_disp_tr)
            & np.isfinite(router_conf_tr)
        )
        meta_te = np.full(len(X_te), 0.5, dtype=float)
        util_te = np.zeros(len(X_te), dtype=float)
        util_win_te = np.full(len(X_te), 0.5, dtype=float)
        if meta_good.sum() > 1500:
            X_meta_tr = _meta_features(
                p_up=p_up_tr[meta_good],
                pred_mag=mag_tr[meta_good],
                edge=edge_tr[meta_good],
                risk_bps=vol_tr[meta_good] * 1e4,
                mkt_disp=disp_tr[meta_good],
                risk_on=risk_on_tr[meta_good],
                dom_ret1=dom_tr[meta_good],
                expert_dispersion=p_disp_tr[meta_good],
                router_conf=router_conf_tr[meta_good],
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
                & np.isfinite(p_disp_te)
                & np.isfinite(router_conf_te)
            )
            X_meta_te = _meta_features(
                p_up=p_up_te[te_good_for_meta],
                pred_mag=mag_te[te_good_for_meta],
                edge=edge_te[te_good_for_meta],
                risk_bps=vol_te[te_good_for_meta] * 1e4,
                mkt_disp=disp_te[te_good_for_meta],
                risk_on=risk_on_te[te_good_for_meta],
                dom_ret1=dom_te[te_good_for_meta],
                expert_dispersion=p_disp_te[te_good_for_meta],
                router_conf=router_conf_te[te_good_for_meta],
            )
            meta_te[te_good_for_meta] = mclf.predict_proba(X_meta_te)[:, 1]

            # Two-layer utility router:
            #  1) stage-1 mclf estimates tradeability (meta_te),
            #  2) stage-2 utility heads estimate payoff on tradeability-aware features.
            meta_tr_hat = np.clip(mclf.predict_proba(X_meta_tr)[:, 1], 0.0, 1.0)
            X_util_tr = _utility_features(
                p_up=p_up_tr[meta_good],
                pred_mag=mag_tr[meta_good],
                edge=edge_tr[meta_good],
                meta_p=meta_tr_hat,
                risk_bps=vol_tr[meta_good] * 1e4,
                mkt_disp=disp_tr[meta_good],
                risk_on=risk_on_tr[meta_good],
                dom_ret1=dom_tr[meta_good],
                expert_dispersion=p_disp_tr[meta_good],
                router_conf=router_conf_tr[meta_good],
            )
            X_util_te = _utility_features(
                p_up=p_up_te[te_good_for_meta],
                pred_mag=mag_te[te_good_for_meta],
                edge=edge_te[te_good_for_meta],
                meta_p=meta_te[te_good_for_meta],
                risk_bps=vol_te[te_good_for_meta] * 1e4,
                mkt_disp=disp_te[te_good_for_meta],
                risk_on=risk_on_te[te_good_for_meta],
                dom_ret1=dom_te[te_good_for_meta],
                expert_dispersion=p_disp_te[te_good_for_meta],
                router_conf=router_conf_te[te_good_for_meta],
            )
            tradeable_mask = meta_tr_hat >= float(np.nanquantile(meta_tr_hat, 0.35))
            if int(tradeable_mask.sum()) < 900:
                tradeable_mask = np.ones_like(meta_tr_hat, dtype=bool)
            utility_sw = _tail_sample_weights(y_utility_tr[meta_good], tail_objective_weight)
            utility_sw = utility_sw * _utility_asymmetric_weights(
                y_utility_tr[meta_good],
                downside_mult=float(max(1.0, utility_downside_mult)),
            )

            ureg = _reg(57)
            ureg.fit(
                X_util_tr[tradeable_mask],
                y_utility_tr[meta_good][tradeable_mask],
                sample_weight=utility_sw[tradeable_mask],
            )
            util_te[te_good_for_meta] = ureg.predict(X_util_te)

            y_ucls = (net_pred_tr[meta_good] > 0).astype(int)
            if np.unique(y_ucls).size >= 2:
                uclf = _clf(59)
                uclf.fit(X_util_tr, y_ucls, sample_weight=utility_sw)
                util_win_te[te_good_for_meta] = uclf.predict_proba(X_util_te)[:, 1]

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
            & np.isfinite(p_disp_te)
            & np.isfinite(router_conf_te)
        )
        idx = np.where(good_te)[0]
        for i in idx:
            rows.append(
                {
                    "symbol": symbol,
                    "fold": int(fold),
                    "ts": int(ts[slo + i]),
                    "ret_60m": float(yret_te[i]),
                    "ret_h1": float((y_ret_1.to_numpy(dtype="float64")[valid_from:])[slo + i]),
                    "ret_h2": float((y_ret_2.to_numpy(dtype="float64")[valid_from:])[slo + i]),
                    "vol_16": float(vol_te[i]),
                    "risk_bps": float(vol_te[i] * 1e4),
                    "p_up": float(p_up_te[i]),
                    "pred_mag": float(mag_te[i]),
                    "edge": float(edge_te[i]),
                    "abs_edge": float(abs(edge_te[i])),
                    "meta_p": float(meta_te[i]),
                    "utility_r_hat": float(util_te[i]),
                    "utility_win_p": float(util_win_te[i]),
                    "expert_dispersion": float(p_disp_te[i]),
                    "router_conf": float(router_conf_te[i]),
                    "mkt_dispersion": float(disp_te[i]),
                    "risk_on_score": float(risk_on_te[i]),
                    "btc_dom_ret1": float(dom_te[i]),
                    "session": str(sess[i]),
                    "regime": str(regime[i]),
                }
            )
    return pd.DataFrame(rows)


def collect_oos_predictions_sq3(
    symbol: str,
    df: pd.DataFrame,
    context: pd.DataFrame,
    seq_weight: float = 0.55,
    seq_window: int = 12,
    wf_train_months: int = 24,
    wf_test_months: int = 6,
    wf_folds: int = 5,
    fast_oos_mode: bool = False,
    primary_horizon: str = "15m",
    secondary_horizon: str = "30m",
    secondary_horizon_weight: float = 0.35,
    tail_objective_weight: float = 0.0,
    utility_label_mode: str = "net_r",
    utility_downside_mult: float = 1.0,
) -> pd.DataFrame:
    return collect_oos_predictions(
        symbol=symbol,
        df=df,
        context=context,
        use_sq3=True,
        seq_weight=seq_weight,
        seq_window=seq_window,
        wf_train_months=wf_train_months,
        wf_test_months=wf_test_months,
        wf_folds=wf_folds,
        fast_oos_mode=fast_oos_mode,
        primary_horizon=primary_horizon,
        secondary_horizon=secondary_horizon,
        secondary_horizon_weight=secondary_horizon_weight,
        tail_objective_weight=tail_objective_weight,
        utility_label_mode=utility_label_mode,
        utility_downside_mult=utility_downside_mult,
    )


def ensure_oos_cache(
    symbols: list[str],
    context_symbols: list[str],
    rebuild_cache: bool = False,
    min_bars: int = 20000,
    model_family: str = "sq21",
    sq3_seq_weight: float = 0.55,
    sq3_seq_window: int = 12,
    wf_train_months: int = 24,
    wf_test_months: int = 6,
    wf_folds: int = 5,
    fast_oos_mode: bool = False,
    primary_horizon: str = "15m",
    secondary_horizon: str = "60m",
    secondary_horizon_weight: float = 0.25,
    tail_objective_weight: float = 0.5,
    utility_label_mode: str = "net_r",
    utility_downside_mult: float = 1.0,
) -> tuple[pd.DataFrame, list[str]]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(
        symbols=symbols,
        context_symbols=context_symbols,
        model_family=model_family,
        sq3_seq_weight=sq3_seq_weight,
        sq3_seq_window=sq3_seq_window,
        wf_train_months=wf_train_months,
        wf_test_months=wf_test_months,
        wf_folds=wf_folds,
        fast_oos_mode=bool(fast_oos_mode),
        primary_horizon=str(primary_horizon),
        secondary_horizon=str(secondary_horizon),
        secondary_horizon_weight=float(secondary_horizon_weight),
        tail_objective_weight=float(tail_objective_weight),
        utility_label_mode=str(utility_label_mode),
        utility_downside_mult=float(max(1.0, utility_downside_mult)),
    )
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

    use_sq3 = str(model_family).lower().startswith("sq3")
    frames = []
    for sym in usable_symbols:
        if use_sq3:
            s = collect_oos_predictions_sq3(
                sym,
                raw_by_symbol[sym],
                context,
                seq_weight=sq3_seq_weight,
                seq_window=sq3_seq_window,
                wf_train_months=wf_train_months,
                wf_test_months=wf_test_months,
                wf_folds=wf_folds,
                fast_oos_mode=bool(fast_oos_mode),
                primary_horizon=str(primary_horizon),
                secondary_horizon=str(secondary_horizon),
                secondary_horizon_weight=float(secondary_horizon_weight),
                tail_objective_weight=float(tail_objective_weight),
                utility_label_mode=str(utility_label_mode),
                utility_downside_mult=float(max(1.0, utility_downside_mult)),
            )
        else:
            s = collect_oos_predictions(
                sym,
                raw_by_symbol[sym],
                context,
                wf_train_months=wf_train_months,
                wf_test_months=wf_test_months,
                wf_folds=wf_folds,
                fast_oos_mode=bool(fast_oos_mode),
                primary_horizon=str(primary_horizon),
                secondary_horizon=str(secondary_horizon),
                secondary_horizon_weight=float(secondary_horizon_weight),
                tail_objective_weight=float(tail_objective_weight),
                utility_label_mode=str(utility_label_mode),
                utility_downside_mult=float(max(1.0, utility_downside_mult)),
            )
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
    shock_z_cut: float
    shock_p_boost: float
    shock_meta_boost: float
    quality_floor: float
    allocator_quality_blend: float
    allocator_quality_power: float
    allocator_utility_weight: float
    allocator_utility_power: float
    topup_score_scale: float
    min_router_conf: float
    max_expert_disp: float
    month_loss_cap_r: float
    reentry_drawdown_r: float
    conf_weight_power: float
    leverage_max: float
    leverage_quality_floor: float
    leverage_quality_power: float
    post_cap_scale: float
    long_only: bool
    risk_min_bps: float
    max_per_ts: int
    n: int
    months: int
    mean_monthly_trades: float
    median_monthly_trades: float
    min_monthly_trades: int
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
    worst_month_r: float
    last10_total_r: float
    last10_win_month_pct: float


def _effective_thresholds(
    frame: pd.DataFrame,
    p_min: float,
    meta_min: float,
    shock_z_cut: float,
    shock_p_boost: float,
    shock_meta_boost: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(frame)
    p_req = np.full(n, float(p_min), dtype="float64")
    meta_req = np.full(n, float(meta_min), dtype="float64")
    if shock_z_cut <= 0:
        return p_req, meta_req
    shock_col: str | None = None
    if "mkt_shock_z" in frame.columns:
        shock_col = "mkt_shock_z"
    elif "mkt_shock_z_l1" in frame.columns:
        shock_col = "mkt_shock_z_l1"
    if shock_col is None:
        return p_req, meta_req

    shock_abs = pd.to_numeric(frame[shock_col], errors="coerce").abs().to_numpy(dtype="float64")
    shock_mask = np.isfinite(shock_abs) & (shock_abs >= float(shock_z_cut))
    if not shock_mask.any():
        return p_req, meta_req
    p_req[shock_mask] = np.clip(p_req[shock_mask] + float(shock_p_boost), 0.5, 0.999)
    meta_req[shock_mask] = np.clip(meta_req[shock_mask] + float(shock_meta_boost), 0.0, 0.999)
    return p_req, meta_req


def _confidence_quality(frame: pd.DataFrame) -> np.ndarray:
    n = len(frame)
    quality = np.ones(n, dtype="float64")
    if n == 0:
        return quality

    if "router_conf" in frame.columns:
        rc = pd.to_numeric(frame["router_conf"], errors="coerce").to_numpy(dtype="float64")
        rc = np.where(np.isfinite(rc), np.clip(rc, 0.0, 1.0), 0.5)
    else:
        rc = np.full(n, 0.5, dtype="float64")

    if "expert_dispersion" in frame.columns:
        ed = pd.to_numeric(frame["expert_dispersion"], errors="coerce").to_numpy(dtype="float64")
        ed = np.where(np.isfinite(ed), np.clip(ed, 0.0, 0.6), 0.15)
        # Strongly penalize elevated expert disagreement.
        disp_pen = np.clip(1.0 - (ed / 0.35), 0.1, 1.0)
    else:
        disp_pen = np.ones(n, dtype="float64")

    if "p_up" in frame.columns:
        p_up = pd.to_numeric(frame["p_up"], errors="coerce").to_numpy(dtype="float64")
        p_up = np.where(np.isfinite(p_up), np.clip(p_up, 0.0, 1.0), 0.5)
    else:
        p_up = np.full(n, 0.5, dtype="float64")
    p_margin = np.clip(np.abs(p_up - 0.5) * 2.0, 0.0, 1.0)

    if "meta_p" in frame.columns:
        meta = pd.to_numeric(frame["meta_p"], errors="coerce").to_numpy(dtype="float64")
        meta = np.where(np.isfinite(meta), np.clip(meta, 0.0, 1.0), 0.5)
    else:
        meta = np.full(n, 0.5, dtype="float64")
    util_p_col = "utility_win_p" if "utility_win_p" in frame.columns else "utility_p"
    if util_p_col in frame.columns:
        util_p = pd.to_numeric(frame[util_p_col], errors="coerce").to_numpy(dtype="float64")
        util_p = np.where(np.isfinite(util_p), np.clip(util_p, 0.0, 1.0), 0.5)
    else:
        util_p = meta.copy()
    if "utility_r_hat" in frame.columns:
        util_r = pd.to_numeric(frame["utility_r_hat"], errors="coerce").to_numpy(dtype="float64")
        if np.isfinite(util_r).any():
            u_scale = float(np.nanpercentile(np.abs(util_r[np.isfinite(util_r)]), 80))
            if not np.isfinite(u_scale) or u_scale < 1e-8:
                u_scale = 1.0
            util_r_sig = np.clip((util_r / u_scale + 1.0) * 0.5, 0.0, 1.0)
            util_r_sig = np.where(np.isfinite(util_r_sig), util_r_sig, 0.5)
        else:
            util_r_sig = np.full(n, 0.5, dtype="float64")
    else:
        util_r_sig = np.full(n, 0.5, dtype="float64")

    if "abs_edge" in frame.columns:
        abs_edge = pd.to_numeric(frame["abs_edge"], errors="coerce").to_numpy(dtype="float64")
    elif "edge" in frame.columns:
        abs_edge = np.abs(pd.to_numeric(frame["edge"], errors="coerce").to_numpy(dtype="float64"))
    else:
        abs_edge = np.full(n, 0.0, dtype="float64")
    abs_edge = np.where(np.isfinite(abs_edge), np.clip(abs_edge, 0.0, None), 0.0)

    if "risk_bps" in frame.columns:
        risk_bps = pd.to_numeric(frame["risk_bps"], errors="coerce").to_numpy(dtype="float64")
    else:
        risk_bps = np.full(n, np.nan, dtype="float64")
    risk_frac = np.where(np.isfinite(risk_bps), np.clip(risk_bps, 1.0, None) / 1e4, np.nan)
    edge_eff = np.where(np.isfinite(risk_frac), abs_edge / (abs_edge + risk_frac), 0.5)
    edge_eff = np.where(np.isfinite(edge_eff), np.clip(edge_eff, 0.0, 1.0), 0.5)

    if "regime" in frame.columns:
        reg = frame["regime"].astype(str).str.upper().to_numpy()
        regime_align = np.where(reg == "WITH", 1.0, np.where(reg == "FLAT", 0.75, 0.55))
    else:
        regime_align = np.full(n, 0.75, dtype="float64")

    shock_col: str | None = None
    if "mkt_shock_z" in frame.columns:
        shock_col = "mkt_shock_z"
    elif "mkt_shock_z_l1" in frame.columns:
        shock_col = "mkt_shock_z_l1"
    if shock_col is not None:
        shock = pd.to_numeric(frame[shock_col], errors="coerce").abs().to_numpy(dtype="float64")
        shock = np.where(np.isfinite(shock), shock, 0.0)
        shock_pen = np.clip(1.0 - np.maximum(shock - 1.2, 0.0) / 2.5, 0.45, 1.0)
    else:
        shock_pen = np.ones(n, dtype="float64")

    quality = (
        0.34 * rc
        + 0.17 * disp_pen
        + 0.13 * p_margin
        + 0.10 * meta
        + 0.14 * util_p
        + 0.14 * util_r_sig
        + 0.08 * edge_eff
        + 0.04 * regime_align
    )
    quality = np.clip(quality * shock_pen, 0.05, 1.0)
    return quality.astype("float64", copy=False)


def _utility_score(frame: pd.DataFrame) -> np.ndarray:
    n = len(frame)
    if n == 0:
        return np.zeros(0, dtype="float64")
    win_col = "utility_win_p" if "utility_win_p" in frame.columns else "utility_p"
    if win_col in frame.columns:
        up = pd.to_numeric(frame[win_col], errors="coerce").to_numpy(dtype="float64")
        up = np.where(np.isfinite(up), np.clip(up, 0.0, 1.0), 0.5)
    else:
        up = np.full(n, 0.5, dtype="float64")

    if "utility_r_hat" in frame.columns:
        ur = pd.to_numeric(frame["utility_r_hat"], errors="coerce").to_numpy(dtype="float64")
        if np.isfinite(ur).any():
            sc = float(np.nanpercentile(np.abs(ur[np.isfinite(ur)]), 80))
            if not np.isfinite(sc) or sc < 1e-8:
                sc = 1.0
            ur_sig = np.clip((ur / sc + 1.0) * 0.5, 0.0, 1.0)
            ur_sig = np.where(np.isfinite(ur_sig), ur_sig, 0.5)
        else:
            ur_sig = np.full(n, 0.5, dtype="float64")
    else:
        ur_sig = np.full(n, 0.5, dtype="float64")

    score = 0.58 * up + 0.42 * ur_sig
    return np.where(np.isfinite(score), np.clip(score, 0.0, 1.0), 0.5)


def _allocator_score(
    frame: pd.DataFrame,
    quality_blend: float = 0.0,
    quality_power: float = 1.0,
    utility_weight: float = 0.0,
    utility_power: float = 1.0,
) -> np.ndarray:
    n = len(frame)
    if n == 0:
        return np.zeros(0, dtype="float64")
    edge = np.abs(pd.to_numeric(frame["edge"], errors="coerce").to_numpy(dtype="float64"))
    edge = np.where(np.isfinite(edge), edge, 0.0)
    meta = np.clip(pd.to_numeric(frame["meta_p"], errors="coerce").to_numpy(dtype="float64"), 0.0, 1.0)
    meta = np.where(np.isfinite(meta), meta, 0.0)
    p_up = np.clip(pd.to_numeric(frame["p_up"], errors="coerce").to_numpy(dtype="float64"), 0.0, 1.0)
    p_up = np.where(np.isfinite(p_up), p_up, 0.5)

    base = edge * meta
    p_margin = np.abs(p_up - 0.5) * 2.0
    base = base * (0.85 + 0.15 * np.clip(p_margin, 0.0, 1.0))
    if "utility_score" in frame.columns:
        util = pd.to_numeric(frame["utility_score"], errors="coerce").to_numpy(dtype="float64")
        util = np.where(np.isfinite(util), np.clip(util, 0.0, 1.0), 0.0)
        uw = float(np.clip(utility_weight, 0.0, 1.0))
        if uw > 1e-12:
            up = np.power(util, float(max(0.1, utility_power)))
            base = base * ((1.0 - uw) + uw * up)
    blend = float(np.clip(quality_blend, 0.0, 1.0))
    if blend <= 1e-12:
        return np.where(np.isfinite(base), base, 0.0)

    q = _confidence_quality(frame)
    qpow = np.power(np.clip(q, 0.0, 1.0), float(max(0.1, quality_power)))
    score = base * ((1.0 - blend) + blend * qpow)
    return np.where(np.isfinite(score), score, 0.0)


def _normalize_utility_grid(values: list[float], default: list[float]) -> list[float]:
    src = values if values else default
    out = [float(np.clip(v, 0.0, 1.0)) for v in src]
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(out))


def _ensure_utility_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "utility_win_p" not in out.columns:
        out["utility_win_p"] = pd.to_numeric(out.get("meta_p", 0.5), errors="coerce").fillna(0.5)
    if "utility_r_hat" not in out.columns:
        out["utility_r_hat"] = pd.to_numeric(out.get("edge", 0.0), errors="coerce").fillna(0.0)
    up = pd.to_numeric(out["utility_win_p"], errors="coerce").to_numpy(dtype="float64")
    up = np.where(np.isfinite(up), np.clip(up, 0.0, 1.0), 0.5)
    ur = pd.to_numeric(out["utility_r_hat"], errors="coerce").to_numpy(dtype="float64")
    ur = np.where(np.isfinite(ur), ur, 0.0)
    ur_pos = np.clip(ur, 0.0, None)
    if np.isfinite(ur_pos).any():
        sc = float(np.nanpercentile(ur_pos[np.isfinite(ur_pos)], 85))
        if not np.isfinite(sc) or sc < 1e-8:
            sc = 1.0
    else:
        sc = 1.0
    ur_sig = np.clip(ur_pos / sc, 0.0, 1.5)
    util_score = np.clip(0.65 * up + 0.35 * ur_sig, 0.0, 1.0)
    out["utility_win_p"] = up
    out["utility_r_hat"] = ur
    out["utility_score"] = util_score
    return out


def _derive_symbol_regime_abstentions(
    frame: pd.DataFrame,
    min_samples: int = 120,
    min_mean_r: float = -0.03,
    min_win_rate: float = 0.38,
    min_mean_resid: float = -0.01,
    max_overconf_rate: float = 0.58,
    target_col: str = "ret_h1",
    top_k: int = 8,
) -> tuple[tuple[str, str], ...]:
    """Learn hard symbol-regime blocks from utility/payoff residual underperformance."""
    if frame.empty:
        return ()
    cols = {"symbol", "regime", "utility_r_hat", "utility_win_p"}
    if not cols.issubset(frame.columns):
        return ()

    x = frame.copy()
    x["symbol"] = x["symbol"].astype(str).str.upper()
    x["regime"] = x["regime"].astype(str).str.upper()
    x["utility_r_hat"] = pd.to_numeric(x["utility_r_hat"], errors="coerce")
    x["utility_win_p"] = pd.to_numeric(x["utility_win_p"], errors="coerce")
    ret_col = str(target_col) if str(target_col) in x.columns else ("ret_h1" if "ret_h1" in x.columns else "ret_60m")
    if ret_col not in x.columns:
        return ()
    ret = pd.to_numeric(x[ret_col], errors="coerce").to_numpy(dtype="float64")
    side = np.where(pd.to_numeric(x.get("edge", 0.0), errors="coerce").to_numpy(dtype="float64") >= 0.0, 1.0, -1.0)
    realized_net = side * ret - COST_FRAC
    x["realized_net"] = realized_net
    x["utility_resid"] = x["realized_net"] - x["utility_r_hat"]
    x["realized_win"] = (x["realized_net"] > 0.0).astype("float64")
    x["overconf"] = (x["utility_resid"] <= float(min_mean_resid)).astype("float64")
    x = x.dropna(
        subset=["symbol", "regime", "utility_r_hat", "utility_win_p", "realized_net", "utility_resid"]
    )
    if x.empty:
        return ()

    grp = x.groupby(["symbol", "regime"], dropna=False).agg(
        n=("utility_r_hat", "size"),
        mean_ur=("utility_r_hat", "mean"),
        mean_up=("utility_win_p", "mean"),
        mean_realized=("realized_net", "mean"),
        realized_win=("realized_win", "mean"),
        mean_resid=("utility_resid", "mean"),
        overconf_rate=("overconf", "mean"),
    )
    bad = grp[
        (grp["n"] >= int(max(1, min_samples)))
        & (
            (grp["mean_realized"] <= float(min_mean_r))
            | (grp["realized_win"] <= float(min_win_rate))
            | (grp["mean_resid"] <= float(min_mean_resid))
            | (grp["overconf_rate"] >= float(np.clip(max_overconf_rate, 0.0, 1.0)))
        )
    ].copy()
    if bad.empty:
        return ()
    # Lower residuals / lower realized edge and higher overconfidence are worse.
    bad["severity"] = (
        np.minimum(0.0, bad["mean_realized"])
        + np.minimum(0.0, bad["mean_resid"])
        + 0.75 * (bad["realized_win"] - float(min_win_rate))
        + 0.50 * (float(np.clip(max_overconf_rate, 0.0, 1.0)) - bad["overconf_rate"])
    )
    bad = bad.sort_values(["severity", "n"], ascending=[True, False]).head(int(max(1, top_k)))
    out = tuple((str(sym), str(reg)) for sym, reg in bad.index.to_list())
    return tuple(sorted(set(out)))


def _merge_symbol_regime_abstentions(
    base: tuple[tuple[str, str], ...],
    learned: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    if not base and not learned:
        return ()
    out: set[tuple[str, str]] = set()
    for sym, reg in list(base) + list(learned):
        out.add((str(sym).upper(), str(reg).upper()))
    return tuple(sorted(out))


def _apply_dynamic_leverage(
    x: pd.DataFrame,
    leverage_max: float = 1.0,
    leverage_quality_floor: float = 1.0,
    leverage_quality_power: float = 1.0,
) -> pd.DataFrame:
    if x.empty:
        return x
    lev_max = float(max(1.0, leverage_max))
    if lev_max <= 1.000001:
        x = x.copy()
        x["leverage"] = 1.0
        return x

    floor = float(np.clip(leverage_quality_floor, 0.0, 0.999))
    q = _confidence_quality(x)
    qn = np.clip((q - floor) / max(1e-6, 1.0 - floor), 0.0, 1.0)
    curve = np.power(qn, float(max(0.1, leverage_quality_power)))
    lev = 1.0 + (lev_max - 1.0) * curve

    out = x.copy()
    out["leverage"] = lev.astype("float64")
    lev_arr = out["leverage"].to_numpy(dtype="float64", copy=False)
    for col in ("gross", "net", "r_net", "r_weighted", "net_bps_weighted"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype="float64") * lev_arr
    return out


def _apply_hard_abstentions(
    frame: pd.DataFrame,
    abstain_symbols: tuple[str, ...] = (),
    abstain_regimes: tuple[str, ...] = (),
    abstain_symbol_regimes: tuple[tuple[str, str], ...] = (),
) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame
    if abstain_symbols:
        sym_set = {str(s).upper() for s in abstain_symbols}
        sym_arr = out["symbol"].astype(str).str.upper().to_numpy()
        out = out[~np.isin(sym_arr, list(sym_set))]
    if abstain_regimes:
        reg_set = {str(r).upper() for r in abstain_regimes}
        reg_arr = out["regime"].astype(str).str.upper().to_numpy()
        out = out[~np.isin(reg_arr, list(reg_set))]
    if out.empty or not abstain_symbol_regimes:
        return out
    pair_set = {(str(s).upper(), str(r).upper()) for s, r in abstain_symbol_regimes}
    sym_arr = out["symbol"].astype(str).str.upper().to_numpy()
    reg_arr = out["regime"].astype(str).str.upper().to_numpy()
    block = np.fromiter(((s, r) in pair_set for s, r in zip(sym_arr, reg_arr)), dtype=bool, count=len(out))
    return out[~block]


def _apply_monthly_loss_cap(x: pd.DataFrame, month_loss_cap_r: float) -> pd.DataFrame:
    cap = float(month_loss_cap_r)
    if x.empty or cap <= 0:
        return x

    keep_rows: list[pd.DataFrame] = []
    for _, g_month in x.sort_values(["month", "ts"]).groupby("month", sort=False):
        running = 0.0
        chunks: list[pd.DataFrame] = []
        for _, g_ts in g_month.groupby("ts", sort=True):
            chunks.append(g_ts)
            batch_r = float(pd.to_numeric(g_ts["r_weighted"], errors="coerce").sum())
            if np.isfinite(batch_r):
                running += batch_r
            if running <= -cap:
                break
        if chunks:
            keep_rows.append(pd.concat(chunks, ignore_index=False))

    if not keep_rows:
        return pd.DataFrame(columns=x.columns)
    return pd.concat(keep_rows, ignore_index=False).sort_values(["ts", "symbol"])


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
    min_trades_per_month_hard: int = 0,
    shock_z_cut: float = 0.0,
    shock_p_boost: float = 0.0,
    shock_meta_boost: float = 0.0,
    hard_abstain_symbols: tuple[str, ...] = (),
    hard_abstain_regimes: tuple[str, ...] = (),
    hard_abstain_symbol_regimes: tuple[tuple[str, str], ...] = (),
    quality_floor: float = 0.0,
    allocator_quality_blend: float = 0.25,
    allocator_quality_power: float = 1.2,
    allocator_utility_weight: float = 0.35,
    allocator_utility_power: float = 1.2,
    leverage_max: float = 1.0,
    leverage_quality_floor: float = 0.8,
    leverage_quality_power: float = 1.5,
    target_col: str = "ret_60m",
) -> pd.DataFrame:
    d = _ensure_utility_columns(d)
    x = d[d["symbol"].isin(symbols)]
    if x.empty:
        return pd.DataFrame()

    q_cut = 1.0 - (top_pct / 100.0)
    if session is not None:
        x = x[x["session"] == session]
    if regime is not None:
        x = x[x["regime"] == regime]
    x = _apply_hard_abstentions(
        x,
        abstain_symbols=hard_abstain_symbols,
        abstain_regimes=hard_abstain_regimes,
        abstain_symbol_regimes=hard_abstain_symbol_regimes,
    )
    if x.empty:
        return pd.DataFrame()
    # Shock-aware threshold modulation: in elevated shock bars,
    # require stronger probability/meta quality.
    p_req, meta_req = _effective_thresholds(
        frame=x,
        p_min=p_min,
        meta_min=meta_min,
        shock_z_cut=shock_z_cut,
        shock_p_boost=shock_p_boost,
        shock_meta_boost=shock_meta_boost,
    )

    edge_arr = pd.to_numeric(x["edge"], errors="coerce").to_numpy(dtype="float64")
    p_up_arr = pd.to_numeric(x["p_up"], errors="coerce").to_numpy(dtype="float64")
    meta_arr = pd.to_numeric(x["meta_p"], errors="coerce").to_numpy(dtype="float64")
    risk_arr = pd.to_numeric(x["risk_bps"], errors="coerce").to_numpy(dtype="float64")

    base_mask = np.isfinite(edge_arr) & np.isfinite(p_up_arr) & np.isfinite(meta_arr) & np.isfinite(risk_arr)
    base_mask &= (risk_arr >= float(risk_min_bps)) & (meta_arr >= meta_req)

    if long_only:
        dir_mask = (edge_arr > 0) & (p_up_arr >= p_req)
        keep_mask = base_mask & dir_mask
        x = x[keep_mask].copy()
        x["side"] = 1.0
    else:
        long_mask = (edge_arr > 0) & (p_up_arr >= p_req)
        short_mask = (edge_arr < 0) & (p_up_arr <= (1.0 - p_req))
        keep_mask = base_mask & (long_mask | short_mask)
        x = x[keep_mask].copy()
        x["side"] = np.where(x["edge"].to_numpy(dtype="float64", copy=False) > 0, 1.0, -1.0)
    if x.empty:
        return pd.DataFrame()

    qv = _confidence_quality(x)
    if float(quality_floor) > 0.0:
        x = x[qv >= float(quality_floor)].copy()
        if x.empty:
            return pd.DataFrame()

    # Start with quality-selective subset; optionally top-up month cadence.
    # The hard cadence mode is cap-aware (max_per_ts) to avoid adding trades
    # that will later be dropped by timestamp capacity constraints.
    x["allocator_score"] = _allocator_score(
        x,
        quality_blend=float(allocator_quality_blend),
        quality_power=float(allocator_quality_power),
        utility_weight=float(allocator_utility_weight),
        utility_power=float(allocator_utility_power),
    ).astype("float64")
    selected = x[x["rank_pct"] >= q_cut].copy()
    if min_trades_per_month_hard > 0:
        frames: list[pd.DataFrame] = []
        for _, g in x.groupby("month", sort=False):
            g_cap = g.sort_values("allocator_score", ascending=False).copy()
            if max_per_ts > 0:
                g_cap["_ts_rank_cap"] = g_cap.groupby("ts").cumcount()
                g_cap = g_cap[g_cap["_ts_rank_cap"] < max_per_ts].drop(columns=["_ts_rank_cap"])

            g_sel = g_cap[g_cap["rank_pct"] >= q_cut].copy()
            need = int(min_trades_per_month_hard - len(g_sel))
            if need > 0:
                extra = g_cap[g_cap["rank_pct"] < q_cut].head(need)
                if not extra.empty:
                    g_sel = pd.concat([g_sel, extra], ignore_index=False)
            if not g_sel.empty:
                frames.append(g_sel)
        if frames:
            selected = pd.concat(frames, ignore_index=False).drop_duplicates(subset=["ts", "symbol"])
    x = selected.copy()
    if x.empty:
        return pd.DataFrame()

    x = x.sort_values(["ts", "allocator_score"], ascending=[True, False])
    if max_per_ts > 0:
        x = x.groupby("ts", as_index=False).head(max_per_ts).copy()

    x["allocator_score"] = pd.to_numeric(x["allocator_score"], errors="coerce").astype("float64")
    score_sum = x.groupby("ts")["allocator_score"].transform("sum").astype("float64")
    x["weight"] = np.where(score_sum > 1e-12, x["allocator_score"] / score_sum, 1.0)
    x["weight"] = pd.to_numeric(x["weight"], errors="coerce").astype("float64")
    ts_count = x.groupby("ts")["symbol"].transform("count").replace(0, np.nan)
    x["weight"] = np.where(np.isfinite(x["weight"]), x["weight"], 1.0 / ts_count)

    ret_col = str(target_col)
    if ret_col not in x.columns:
        ret_col = "ret_60m"
    ret_arr = pd.to_numeric(x[ret_col], errors="coerce").to_numpy(dtype="float64")
    side_arr = pd.to_numeric(x["side"], errors="coerce").to_numpy(dtype="float64")
    vol_arr = pd.to_numeric(x["vol_16"], errors="coerce").to_numpy(dtype="float64")
    weight_arr = pd.to_numeric(x["weight"], errors="coerce").to_numpy(dtype="float64")
    x["gross"] = ret_arr * side_arr
    x["net"] = x["gross"] - COST_FRAC
    x["r_net"] = np.where(np.isfinite(vol_arr) & (vol_arr > 1e-12), x["net"] / vol_arr, np.nan)
    x["r_weighted"] = x["r_net"] * weight_arr
    x["net_bps_weighted"] = x["net"] * weight_arr * 1e4
    x = _apply_dynamic_leverage(
        x,
        leverage_max=float(leverage_max),
        leverage_quality_floor=float(leverage_quality_floor),
        leverage_quality_power=float(leverage_quality_power),
    )
    return x


def _apply_policy_with_adaptive_pacing(
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
    min_trades_per_month_hard: int,
    shock_z_cut: float = 0.0,
    shock_p_boost: float = 0.0,
    shock_meta_boost: float = 0.0,
    topup_weight_scale: float = 1.0,
    expert_disp_max: float = 1.0,
    router_conf_min: float = 0.0,
    hard_abstain_symbols: tuple[str, ...] = (),
    hard_abstain_regimes: tuple[str, ...] = (),
    hard_abstain_symbol_regimes: tuple[tuple[str, str], ...] = (),
    cadence_quality_floor: float = 0.0,
    quality_floor: float = 0.0,
    allocator_quality_blend: float = 0.25,
    allocator_quality_power: float = 1.2,
    allocator_utility_weight: float = 0.35,
    allocator_utility_power: float = 1.2,
    leverage_max: float = 1.0,
    leverage_quality_floor: float = 0.8,
    leverage_quality_power: float = 1.5,
    target_col: str = "ret_60m",
) -> pd.DataFrame:
    d = _ensure_utility_columns(d)
    if router_conf_min > 0.0 or expert_disp_max < 0.999:
        d = d.copy()
        if "router_conf" in d.columns and router_conf_min > 0.0:
            d = d[pd.to_numeric(d["router_conf"], errors="coerce") >= float(router_conf_min)]
        if "expert_dispersion" in d.columns and expert_disp_max < 0.999:
            d = d[pd.to_numeric(d["expert_dispersion"], errors="coerce") <= float(expert_disp_max)]
        if d.empty:
            return pd.DataFrame()
    if min_trades_per_month_hard <= 0:
        return _apply_policy_and_allocate(
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
            min_trades_per_month_hard=0,
            shock_z_cut=shock_z_cut,
            shock_p_boost=shock_p_boost,
            shock_meta_boost=shock_meta_boost,
            hard_abstain_symbols=hard_abstain_symbols,
            hard_abstain_regimes=hard_abstain_regimes,
            hard_abstain_symbol_regimes=hard_abstain_symbol_regimes,
            quality_floor=quality_floor,
            allocator_quality_blend=allocator_quality_blend,
            allocator_quality_power=allocator_quality_power,
            leverage_max=leverage_max,
            leverage_quality_floor=leverage_quality_floor,
            leverage_quality_power=leverage_quality_power,
            target_col=target_col,
        )

    # Build base candidate universe without any rank cutoff.
    base = d[d["symbol"].isin(symbols)]
    if base.empty:
        return pd.DataFrame()
    if session is not None:
        base = base[base["session"] == session]
    if regime is not None:
        base = base[base["regime"] == regime]
    base = _apply_hard_abstentions(
        base,
        abstain_symbols=hard_abstain_symbols,
        abstain_regimes=hard_abstain_regimes,
        abstain_symbol_regimes=hard_abstain_symbol_regimes,
    )
    if base.empty:
        return pd.DataFrame()
    p_req, meta_req = _effective_thresholds(
        frame=base,
        p_min=p_min,
        meta_min=meta_min,
        shock_z_cut=shock_z_cut,
        shock_p_boost=shock_p_boost,
        shock_meta_boost=shock_meta_boost,
    )
    edge_arr = pd.to_numeric(base["edge"], errors="coerce").to_numpy(dtype="float64")
    p_up_arr = pd.to_numeric(base["p_up"], errors="coerce").to_numpy(dtype="float64")
    meta_arr = pd.to_numeric(base["meta_p"], errors="coerce").to_numpy(dtype="float64")
    risk_arr = pd.to_numeric(base["risk_bps"], errors="coerce").to_numpy(dtype="float64")

    valid = np.isfinite(edge_arr) & np.isfinite(p_up_arr) & np.isfinite(meta_arr) & np.isfinite(risk_arr)
    valid &= (risk_arr >= float(risk_min_bps)) & (meta_arr >= meta_req)

    if long_only:
        dir_mask = (edge_arr > 0) & (p_up_arr >= p_req)
        keep_mask = valid & dir_mask
        base = base[keep_mask].copy()
        base["side"] = 1.0
    else:
        long_mask = (edge_arr > 0) & (p_up_arr >= p_req)
        short_mask = (edge_arr < 0) & (p_up_arr <= (1.0 - p_req))
        keep_mask = valid & (long_mask | short_mask)
        base = base[keep_mask].copy()
        base["side"] = np.where(base["edge"].to_numpy(dtype="float64", copy=False) > 0, 1.0, -1.0)
    if base.empty:
        return pd.DataFrame()

    qv_base = _confidence_quality(base)
    if float(quality_floor) > 0.0:
        base = base[qv_base >= float(quality_floor)].copy()
        if base.empty:
            return pd.DataFrame()

    q_cut = 1.0 - (top_pct / 100.0)
    base["allocator_score"] = _allocator_score(
        base,
        quality_blend=float(allocator_quality_blend),
        quality_power=float(allocator_quality_power),
        utility_weight=float(allocator_utility_weight),
        utility_power=float(allocator_utility_power),
    ).astype("float64")

    selected_chunks: list[pd.DataFrame] = []
    for _, g in base.groupby("month", sort=False):
        if g.empty:
            continue
        g = g.sort_values(["ts", "allocator_score"], ascending=[True, False]).copy()
        per_ts_cap = int(max(1, max_per_ts))

        # Base selective set by quality.
        g_sel = g[g["rank_pct"] >= q_cut].copy()
        if not g_sel.empty:
            g_sel = g_sel.groupby("ts", as_index=False).head(per_ts_cap).copy()

        current_n = int(len(g_sel))
        target_n = int(min_trades_per_month_hard)
        if current_n < target_n:
            need = target_n - current_n
            chosen_pairs = set(zip(g_sel["ts"], g_sel["symbol"])) if not g_sel.empty else set()
            ts_selected_counts = (
                g_sel.groupby("ts")["symbol"].size().to_dict() if not g_sel.empty else {}
            )
            extra_pool = g[g["rank_pct"] < q_cut].copy()
            if float(cadence_quality_floor) > 0.0:
                qv = _confidence_quality(extra_pool)
                extra_pool = extra_pool[qv >= float(cadence_quality_floor)]
            # Top-up from non-selected candidates, respecting per-ts cap.
            for _, row in extra_pool.sort_values("allocator_score", ascending=False).iterrows():
                key = (row["ts"], row["symbol"])
                if key in chosen_pairs:
                    continue
                ts = row["ts"]
                if int(ts_selected_counts.get(ts, 0)) >= per_ts_cap:
                    continue
                chosen_pairs.add(key)
                ts_selected_counts[ts] = int(ts_selected_counts.get(ts, 0)) + 1
                g_sel = pd.concat([g_sel, row.to_frame().T], ignore_index=False)
                need -= 1
                if need <= 0:
                    break

        if not g_sel.empty:
            g_sel["is_topup"] = False
            if target_n > current_n:
                # rows added beyond initial quality-selective set are adaptive top-up rows
                # and can be down-weighted to preserve portfolio quality.
                g_sel = g_sel.copy()
                g_sel.iloc[current_n:, g_sel.columns.get_loc("is_topup")] = True
            selected_chunks.append(g_sel.drop_duplicates(subset=["ts", "symbol"]))

    if not selected_chunks:
        return pd.DataFrame()

    x = pd.concat(selected_chunks, ignore_index=False)
    x = x.sort_values(["ts", "allocator_score"], ascending=[True, False])
    if max_per_ts > 0:
        x = x.groupby("ts", as_index=False).head(max_per_ts).copy()

    score_sum = pd.to_numeric(x.groupby("ts")["allocator_score"].transform("sum"), errors="coerce")
    raw_weight = np.where(score_sum > 1e-12, x["allocator_score"] / score_sum, 1.0)
    x["weight"] = pd.to_numeric(raw_weight, errors="coerce")
    if "is_topup" in x.columns and float(topup_weight_scale) < 1.0:
        scale = np.where(x["is_topup"].astype(bool).to_numpy(), float(topup_weight_scale), 1.0)
        x["weight"] = x["weight"] * scale
        # Renormalize by timestamp after top-up scaling.
        wsum = x.groupby("ts")["weight"].transform("sum")
        x["weight"] = np.where(wsum > 1e-12, x["weight"] / wsum, x["weight"])
    ts_count = pd.to_numeric(x.groupby("ts")["symbol"].transform("count"), errors="coerce").replace(0, np.nan)
    fallback = 1.0 / ts_count
    weight_arr = x["weight"].to_numpy(dtype="float64", copy=False)
    x["weight"] = np.where(np.isfinite(weight_arr), weight_arr, fallback)

    ret_col = str(target_col)
    if ret_col not in x.columns:
        ret_col = "ret_60m"
    ret_arr = pd.to_numeric(x[ret_col], errors="coerce").to_numpy(dtype="float64")
    side_arr = pd.to_numeric(x["side"], errors="coerce").to_numpy(dtype="float64")
    vol_arr = pd.to_numeric(x["vol_16"], errors="coerce").to_numpy(dtype="float64")
    weight_arr = pd.to_numeric(x["weight"], errors="coerce").to_numpy(dtype="float64")
    x["gross"] = ret_arr * side_arr
    x["net"] = x["gross"] - COST_FRAC
    x["r_net"] = np.where(np.isfinite(vol_arr) & (vol_arr > 1e-12), x["net"] / vol_arr, np.nan)
    x["r_weighted"] = x["r_net"] * weight_arr
    x["net_bps_weighted"] = x["net"] * weight_arr * 1e4
    x = _apply_dynamic_leverage(
        x,
        leverage_max=float(leverage_max),
        leverage_quality_floor=float(leverage_quality_floor),
        leverage_quality_power=float(leverage_quality_power),
    )
    return x


def _apply_monthly_loss_governor(
    x: pd.DataFrame,
    monthly_loss_cap_r: float,
    reentry_drawdown_r: float,
    post_cap_scale: float = 0.25,
) -> pd.DataFrame:
    if x.empty or monthly_loss_cap_r <= 0:
        return x
    if reentry_drawdown_r < 0:
        reentry_drawdown_r = 0.0
    post_cap_scale = float(np.clip(post_cap_scale, 0.0, 1.0))

    out_chunks: list[pd.DataFrame] = []
    for _, g in x.sort_values(["month", "ts"]).groupby("month", sort=False):
        if g.empty:
            continue
        cum = 0.0
        derisk = False
        month_rows: list[pd.DataFrame] = []
        by_ts = g.groupby("ts", sort=True)
        for _, tblock in by_ts:
            block = tblock.copy()
            scale = post_cap_scale if derisk else 1.0
            if scale < 0.999:
                block["r_weighted"] = pd.to_numeric(block["r_weighted"], errors="coerce") * scale
                block["net_bps_weighted"] = pd.to_numeric(block["net_bps_weighted"], errors="coerce") * scale
            block["governor_scale"] = float(scale)
            t_r = float(pd.to_numeric(block["r_weighted"], errors="coerce").sum())

            month_rows.append(block)
            if np.isfinite(t_r):
                cum += t_r
            if (not derisk) and cum <= -float(monthly_loss_cap_r):
                # Keep trading to preserve cadence, but at scaled exposure.
                derisk = True
            elif derisk and cum >= (-(monthly_loss_cap_r) + reentry_drawdown_r):
                derisk = False

        if month_rows:
            out_chunks.append(pd.concat(month_rows, ignore_index=False))

    if not out_chunks:
        return pd.DataFrame(columns=x.columns)
    return pd.concat(out_chunks, ignore_index=False)


def _full_months_from_ts(ts: pd.Series, min_unique_days: int = 20) -> pd.Index:
    if ts.empty:
        return pd.Index([], dtype="object")
    # ts can be an object-typed series; normalize through numeric coercion first.
    ts_num = pd.to_numeric(ts, errors="coerce")
    dt = pd.to_datetime(ts_num, unit="ms", utc=True, errors="coerce")
    if isinstance(dt, pd.DatetimeIndex):
        if dt.tz is not None:
            dt = dt.tz_localize(None)
        month = dt.to_period("M").astype(str)
        day = dt.floor("D")
    else:
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_localize(None)
        month = dt.dt.to_period("M").astype(str)
        day = dt.dt.floor("D")
    by_month_days = pd.DataFrame({"month": month, "day": day}).dropna()
    if by_month_days.empty:
        return pd.Index([], dtype="object")
    day_counts = by_month_days.groupby("month")["day"].nunique()
    return day_counts[day_counts >= int(max(1, min_unique_days))].index


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
    min_trades_per_month_hard: int = 0,
    shock_z_cut: float = 0.0,
    shock_p_boost: float = 0.0,
    shock_meta_boost: float = 0.0,
    topup_score_scale: float = 1.0,
    router_conf_min: float = 0.0,
    expert_disp_max: float = 1.0,
    monthly_loss_cap_r: float = 0.0,
    reentry_drawdown_r: float = 0.0,
    conf_weight_power: float = 1.0,
    post_cap_scale: float = 0.25,
    hard_abstain_symbols: tuple[str, ...] = (),
    hard_abstain_regimes: tuple[str, ...] = (),
    hard_abstain_symbol_regimes: tuple[tuple[str, str], ...] = (),
    adaptive_floor_quality: float = 0.0,
    quality_floor: float = 0.0,
    allocator_quality_blend: float = 0.25,
    allocator_quality_power: float = 1.2,
    allocator_utility_weight: float = 0.35,
    allocator_utility_power: float = 1.2,
    leverage_max: float = 1.0,
    leverage_quality_floor: float = 0.8,
    leverage_quality_power: float = 1.5,
    target_col: str = "ret_60m",
) -> PolicyResult | None:
    if router_conf_min > 0.0 or expert_disp_max < 0.999:
        d = d.copy()
        if "router_conf" in d.columns and router_conf_min > 0.0:
            d = d[pd.to_numeric(d["router_conf"], errors="coerce") >= float(router_conf_min)]
        if "expert_dispersion" in d.columns and expert_disp_max < 0.999:
            d = d[pd.to_numeric(d["expert_dispersion"], errors="coerce") <= float(expert_disp_max)]
        if d.empty:
            return None
    if min_trades_per_month_hard > 0:
        # Hard monthly floor is treated as a true constraint when enabled.
        x = _apply_policy_with_adaptive_pacing(
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
            min_trades_per_month_hard=min_trades_per_month_hard,
            shock_z_cut=shock_z_cut,
            shock_p_boost=shock_p_boost,
            shock_meta_boost=shock_meta_boost,
            topup_weight_scale=topup_score_scale,
            hard_abstain_symbols=hard_abstain_symbols,
            hard_abstain_regimes=hard_abstain_regimes,
            hard_abstain_symbol_regimes=hard_abstain_symbol_regimes,
            cadence_quality_floor=adaptive_floor_quality,
            quality_floor=quality_floor,
            allocator_quality_blend=allocator_quality_blend,
            allocator_quality_power=allocator_quality_power,
            allocator_utility_weight=allocator_utility_weight,
            allocator_utility_power=allocator_utility_power,
            leverage_max=leverage_max,
            leverage_quality_floor=leverage_quality_floor,
            leverage_quality_power=leverage_quality_power,
            target_col=target_col,
        )
    else:
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
            min_trades_per_month_hard=0,
            shock_z_cut=shock_z_cut,
            shock_p_boost=shock_p_boost,
            shock_meta_boost=shock_meta_boost,
            hard_abstain_symbols=hard_abstain_symbols,
            hard_abstain_regimes=hard_abstain_regimes,
            hard_abstain_symbol_regimes=hard_abstain_symbol_regimes,
            quality_floor=quality_floor,
            allocator_quality_blend=allocator_quality_blend,
            allocator_quality_power=allocator_quality_power,
            allocator_utility_weight=allocator_utility_weight,
            allocator_utility_power=allocator_utility_power,
            leverage_max=leverage_max,
            leverage_quality_floor=leverage_quality_floor,
            leverage_quality_power=leverage_quality_power,
            target_col=target_col,
        )
    if x.empty:
        return None
    if float(conf_weight_power) != 1.0:
        conf = np.clip(pd.to_numeric(x["meta_p"], errors="coerce").to_numpy(dtype="float64"), 0.0, 1.0)
        scale = np.power(conf, float(max(0.1, conf_weight_power)))
        x["weight"] = pd.to_numeric(x["weight"], errors="coerce").to_numpy(dtype="float64") * scale
        wsum = x.groupby("ts")["weight"].transform("sum")
        x["weight"] = np.where(wsum > 1e-12, x["weight"] / wsum, x["weight"])
        x["r_weighted"] = x["r_net"] * x["weight"]
        x["net_bps_weighted"] = x["net"] * x["weight"] * 1e4

    x = _apply_monthly_loss_governor(
        x=x,
        monthly_loss_cap_r=float(monthly_loss_cap_r),
        reentry_drawdown_r=float(reentry_drawdown_r),
        post_cap_scale=float(post_cap_scale),
    )
    if len(x) < 120:
        return None

    msum = x.groupby("month")["r_weighted"].sum()
    mtrades = x.groupby("month")["symbol"].size()
    if len(msum) < 10:
        return None
    if min_trades_per_month_hard > 0:
        # Do not enforce hard trade floor on partial boundary months.
        full_months = _full_months_from_ts(x["ts"], min_unique_days=20)
        mcheck = mtrades.loc[mtrades.index.intersection(full_months)]
        if mcheck.empty:
            return None
        if int(mcheck.min()) < int(min_trades_per_month_hard):
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
        shock_z_cut=float(shock_z_cut),
        shock_p_boost=float(shock_p_boost),
        shock_meta_boost=float(shock_meta_boost),
        quality_floor=float(quality_floor),
        allocator_quality_blend=float(allocator_quality_blend),
        allocator_quality_power=float(allocator_quality_power),
        allocator_utility_weight=float(allocator_utility_weight),
        allocator_utility_power=float(allocator_utility_power),
        topup_score_scale=float(topup_score_scale),
        min_router_conf=float(router_conf_min),
        max_expert_disp=float(expert_disp_max),
        month_loss_cap_r=float(monthly_loss_cap_r),
        reentry_drawdown_r=float(reentry_drawdown_r),
        conf_weight_power=float(conf_weight_power),
        leverage_max=float(leverage_max),
        leverage_quality_floor=float(leverage_quality_floor),
        leverage_quality_power=float(leverage_quality_power),
        post_cap_scale=float(post_cap_scale),
        long_only=bool(long_only),
        risk_min_bps=float(risk_min_bps),
        max_per_ts=int(max_per_ts),
        n=int(len(x)),
        months=int(len(msum)),
        mean_monthly_trades=float(mtrades.mean()),
        median_monthly_trades=float(mtrades.median()),
        min_monthly_trades=int(mtrades.min()),
        trade_win_rate_pct=float((x["net"] > 0).mean() * 100),
        weighted_win_rate_pct=float(weighted_win * 100),
        mean_net_bps=float(x["net_bps_weighted"].mean()),
        avg_r_net=float(x["r_weighted"].mean()),
        mean_monthly_total_r=float(msum.mean()),
        median_monthly_total_r=float(msum.median()),
        total_r=float(msum.sum()),
        win_month_pct=float((msum > 0).mean() * 100),
        worst_month_r=float(msum.min()),
        last10_total_r=float(msum.tail(10).sum()) if len(msum) >= 10 else float(msum.sum()),
        last10_win_month_pct=float((msum.tail(10) > 0).mean() * 100) if len(msum) >= 10 else float((msum > 0).mean() * 100),
        fold_min_avg_r=float(fold_avg.min()),
        fold_pos=int((fold_avg > 0).sum()),
        folds=int(len(fold_avg)),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V7 SQ2.1 macro + allocator WF search")
    p.add_argument("--model-family", type=str, default="sq21", choices=["sq21", "sq3"],
                   help="Model family for OOS generation: sq21 baseline or sq3 sequence blend")
    p.add_argument("--fast-all-symbol-mode", action="store_true",
                   help="Use faster learners/router path to complete very large all-symbol sweeps")
    p.add_argument("--sq3-seq-weight", type=float, default=0.55,
                   help="SQ3 only: blend weight for sequence model predictions (0..0.9)")
    p.add_argument("--sq3-seq-window", type=int, default=12,
                   help="SQ3 only: lag window used for sequence feature expansion")
    p.add_argument("--primary-horizon", type=str, default="15m", choices=["15m", "30m", "60m"],
                   help="Primary execution horizon used for target return labeling")
    p.add_argument("--secondary-horizon", type=str, default="60m", choices=["15m", "30m", "60m"],
                   help="Secondary horizon blended into execution labels")
    p.add_argument("--secondary-horizon-weight", type=float, default=0.25,
                   help="Blend weight for secondary horizon return in target construction")
    p.add_argument("--tail-objective-weight", type=float, default=0.5,
                   help="Downside-sensitive sample weighting during model/meta fitting")
    p.add_argument("--utility-label-mode", type=str, default="downside_penalized",
                   choices=["net_r", "downside_penalized"],
                   help="Utility regression label mode: raw net-R or downside-penalized net-R")
    p.add_argument("--utility-downside-mult", type=float, default=1.8,
                   help="Multiplier for negative utility labels/weights (>=1) to penalize false positives")
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
    p.add_argument("--abstain-symbols", nargs="+", default=[],
                   help="Hard abstain symbols globally (e.g. DOGEUSDT)")
    p.add_argument("--abstain-regimes", nargs="+", default=[],
                   help="Hard abstain these regimes globally (e.g. COUNTER FLAT)")
    p.add_argument("--abstain-symbol-regimes", nargs="+", default=[],
                   help="Hard abstain symbol:regime pairs (e.g. SOLUSDT:COUNTER)")
    p.add_argument("--long-only-only", action="store_true",
                   help="Only evaluate long-only policies")
    p.add_argument("--max-symbol-set-size", type=int, default=5,
                   help="Max symbols in subset")
    p.add_argument("--full-symbol-set-only", action="store_true",
                   help="Only evaluate the full active symbol set as one basket")
    p.add_argument("--min-monthly-trades", type=float, default=0.0,
                   help="Require mean monthly trades >= this value for robust policies")
    p.add_argument("--min-monthly-trades-hard", type=int, default=0,
                   help="Adaptive pacing: attempt >= this many trades per month via quality top-up")
    p.add_argument("--auto-abstain-symbol-regimes", action="store_true",
                   help="Learn hard symbol:regime abstentions from utility underperformance")
    p.add_argument("--abstain-min-samples", type=int, default=120,
                   help="Min samples needed for learned symbol:regime abstention")
    p.add_argument("--abstain-min-mean-r", type=float, default=-0.03,
                   help="Learn abstention when mean utility R is below this threshold")
    p.add_argument("--abstain-min-win-p", type=float, default=0.38,
                   help="Learn abstention when utility win-prob is below this threshold")
    p.add_argument("--abstain-top-k", type=int, default=8,
                   help="Max learned symbol:regime abstention pairs to apply")
    p.add_argument("--cadence-quality-floor", type=float, default=0.0,
                   help="For adaptive pacing, top-up rows must satisfy confidence quality floor (0..1)")
    p.add_argument("--quality-floors", nargs="+", type=float, default=[0.0],
                   help="Hard quality floor for all selected trades (0..1)")
    p.add_argument("--allocator-quality-blends", nargs="+", type=float, default=[0.25, 0.40],
                   help="Blend quality into allocator score (0=edge/meta only, 1=quality-dominated)")
    p.add_argument("--allocator-quality-powers", nargs="+", type=float, default=[1.0, 1.2],
                   help="Exponent on quality term used in allocator score")
    p.add_argument("--allocator-utility-weights", nargs="+", type=float, default=[0.35, 0.55],
                   help="Blend utility score into allocator score (0=edge/meta only, 1=utility-dominated)")
    p.add_argument("--allocator-utility-powers", nargs="+", type=float, default=[1.0, 1.2],
                   help="Exponent on utility score in allocator")
    p.add_argument("--shock-z-cut", type=float, default=0.0,
                   help="Apply shock-aware threshold boosts when |mkt_shock_z| >= this cutoff (0 disables)")
    p.add_argument("--shock-p-boost", type=float, default=0.0,
                   help="Additive boost to p_min during shock bars")
    p.add_argument("--shock-meta-boost", type=float, default=0.0,
                   help="Additive boost to meta_min during shock bars")
    p.add_argument("--topup-score-scale", type=float, default=1.0,
                   help="Weight scale applied to adaptive top-up rows (0..1)")
    p.add_argument("--router-conf-mins", nargs="+", type=float, default=[0.0],
                   help="Minimum router confidence gating value(s)")
    p.add_argument("--expert-disp-maxs", nargs="+", type=float, default=[1.0],
                   help="Maximum expert dispersion gating value(s)")
    p.add_argument("--monthly-loss-cap-rs", nargs="+", type=float, default=[0.0],
                   help="Monthly loss-cap in weighted R; 0 disables")
    p.add_argument("--reentry-drawdown-rs", nargs="+", type=float, default=[0.0],
                   help="Re-enable trading after capped month when recovery >= this R from cap trough")
    p.add_argument("--conf-weight-powers", nargs="+", type=float, default=[1.0],
                   help="Power on meta confidence for position weight scaling (1=off)")
    p.add_argument("--post-cap-scales", nargs="+", type=float, default=[0.15, 0.25, 0.35],
                   help="After monthly loss cap is hit, keep trading with this exposure scale (0..1)")
    p.add_argument("--leverage-maxs", nargs="+", type=float, default=[1.0, 1.2],
                   help="Max leverage multiplier for top-quality trades (>=1)")
    p.add_argument("--leverage-quality-floors", nargs="+", type=float, default=[0.80],
                   help="Quality floor above which dynamic leverage starts scaling up")
    p.add_argument("--leverage-quality-powers", nargs="+", type=float, default=[1.5],
                   help="Shape of quality->leverage curve (higher=more selective leverage)")
    p.add_argument("--wf-train-months", type=int, default=24,
                   help="Walk-forward train window in months for OOS generation")
    p.add_argument("--wf-test-months", type=int, default=6,
                   help="Walk-forward test window in months for OOS generation")
    p.add_argument("--wf-folds", type=int, default=5,
                   help="Number of walk-forward folds for OOS generation")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Force rebuild OOS cache for selected symbols/context")
    return p.parse_args()


def search_policies_with_args(d: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    d = _ensure_utility_columns(d)
    syms = sorted(d["symbol"].unique())
    abstain_symbols = tuple(str(v).upper() for v in getattr(args, "abstain_symbols", []))
    abstain_regimes = tuple(str(v).upper() for v in args.abstain_regimes)
    abstain_symbol_regimes = _parse_symbol_regime_pairs(list(args.abstain_symbol_regimes))
    if bool(getattr(args, "auto_abstain_symbol_regimes", False)):
        learned_pairs = _derive_symbol_regime_abstentions(
            d,
            min_samples=int(getattr(args, "abstain_min_samples", 120)),
            min_mean_r=float(getattr(args, "abstain_min_mean_r", -0.03)),
            min_win_rate=float(getattr(args, "abstain_min_win_p", 0.38)),
            top_k=int(getattr(args, "abstain_top_k", 8)),
        )
        abstain_symbol_regimes = tuple(sorted(set(abstain_symbol_regimes) | set(learned_pairs)))
    target_col = _resolve_target_col(str(args.primary_horizon))
    quality_floors = [float(np.clip(v, 0.0, 0.99)) for v in getattr(args, "quality_floors", [0.0])]
    allocator_quality_blends = [float(np.clip(v, 0.0, 1.0)) for v in getattr(args, "allocator_quality_blends", [0.25])]
    allocator_quality_powers = [float(max(0.1, v)) for v in getattr(args, "allocator_quality_powers", [1.2])]
    allocator_utility_weights = [float(np.clip(v, 0.0, 1.0)) for v in getattr(args, "allocator_utility_weights", [0.35])]
    allocator_utility_powers = [float(max(0.1, v)) for v in getattr(args, "allocator_utility_powers", [1.2])]
    leverage_maxs = [float(max(1.0, v)) for v in getattr(args, "leverage_maxs", [1.0])]
    leverage_quality_floors = [float(np.clip(v, 0.0, 0.99)) for v in getattr(args, "leverage_quality_floors", [0.8])]
    leverage_quality_powers = [float(max(0.1, v)) for v in getattr(args, "leverage_quality_powers", [1.5])]
    if args.full_symbol_set_only:
        symbol_sets = [tuple(syms)]
    else:
        symbol_sets = []
        max_size = args.max_symbol_set_size if args.max_symbol_set_size > 0 else len(syms)
        for r in range(1, min(len(syms), max_size) + 1):
            symbol_sets.extend(itertools.combinations(syms, r))

    sessions: list[str | None] = [None if s == "ALL" else s for s in args.sessions]
    regimes: list[str | None] = [None if r == "ALL" else r for r in args.regimes]
    long_flags = [True] if args.long_only_only else [True, False]

    results: list[dict] = []
    grid_iter = itertools.product(
        args.top_pcts,
        symbol_sets,
        sessions,
        regimes,
        args.p_mins,
        args.meta_mins,
        long_flags,
        args.risk_min_bps,
        args.max_per_ts,
        args.router_conf_mins,
        args.expert_disp_maxs,
        args.monthly_loss_cap_rs,
        args.reentry_drawdown_rs,
        args.conf_weight_powers,
        args.post_cap_scales,
        quality_floors,
        allocator_quality_blends,
        allocator_quality_powers,
        allocator_utility_weights,
        allocator_utility_powers,
        leverage_maxs,
        leverage_quality_floors,
        leverage_quality_powers,
    )
    for (
        top_pct,
        symbols,
        session,
        regime,
        p_min,
        meta_min,
        long_only,
        risk_min,
        max_per_ts,
        router_conf_min,
        expert_disp_max,
        month_loss_cap_r,
        reentry_dd_r,
        conf_w_pow,
        post_cap_scale,
        quality_floor,
        allocator_quality_blend,
        allocator_quality_power,
        allocator_utility_weight,
        allocator_utility_power,
        leverage_max,
        leverage_quality_floor,
        leverage_quality_power,
    ) in grid_iter:
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
            min_trades_per_month_hard=int(args.min_monthly_trades_hard),
            shock_z_cut=float(args.shock_z_cut),
            shock_p_boost=float(args.shock_p_boost),
            shock_meta_boost=float(args.shock_meta_boost),
            topup_score_scale=float(args.topup_score_scale),
            router_conf_min=float(router_conf_min),
            expert_disp_max=float(expert_disp_max),
            monthly_loss_cap_r=float(month_loss_cap_r),
            reentry_drawdown_r=float(reentry_dd_r),
            conf_weight_power=float(conf_w_pow),
            post_cap_scale=float(post_cap_scale),
            hard_abstain_symbols=abstain_symbols,
            hard_abstain_regimes=abstain_regimes,
            hard_abstain_symbol_regimes=abstain_symbol_regimes,
            adaptive_floor_quality=float(args.cadence_quality_floor),
            quality_floor=float(quality_floor),
            allocator_quality_blend=float(allocator_quality_blend),
            allocator_quality_power=float(allocator_quality_power),
            allocator_utility_weight=float(allocator_utility_weight),
            allocator_utility_power=float(allocator_utility_power),
            leverage_max=float(leverage_max),
            leverage_quality_floor=float(leverage_quality_floor),
            leverage_quality_power=float(leverage_quality_power),
            target_col=target_col,
        )
        if r is None:
            continue
        results.append(asdict(r))
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


def _monthly_breakdown_for_policy(
    d: pd.DataFrame,
    row: pd.Series,
    min_trades_per_month_hard: int = 0,
    shock_z_cut: float = 0.0,
    shock_p_boost: float = 0.0,
    shock_meta_boost: float = 0.0,
    topup_score_scale: float = 1.0,
    expert_disp_max: float = 1.0,
    router_conf_min: float = 0.0,
    monthly_loss_cap_r: float = 0.0,
    reentry_drawdown_r: float = 0.0,
    conf_weight_power: float = 1.0,
    post_cap_scale: float = 0.25,
    hard_abstain_symbols: tuple[str, ...] = (),
    hard_abstain_regimes: tuple[str, ...] = (),
    hard_abstain_symbol_regimes: tuple[tuple[str, str], ...] = (),
    cadence_quality_floor: float = 0.0,
    quality_floor: float = 0.0,
    allocator_quality_blend: float = 0.25,
    allocator_quality_power: float = 1.2,
    allocator_utility_weight: float = 0.35,
    allocator_utility_power: float = 1.2,
    leverage_max: float = 1.0,
    leverage_quality_floor: float = 0.8,
    leverage_quality_power: float = 1.5,
    target_col: str = "ret_60m",
) -> pd.DataFrame:
    symbols = tuple(str(row["symbols"]).split(","))
    session = None if row["session"] == "ALL" else str(row["session"])
    regime = None if row["regime"] == "ALL" else str(row["regime"])
    if min_trades_per_month_hard > 0:
        x = _apply_policy_with_adaptive_pacing(
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
            min_trades_per_month_hard=min_trades_per_month_hard,
            shock_z_cut=shock_z_cut,
            shock_p_boost=shock_p_boost,
            shock_meta_boost=shock_meta_boost,
            topup_weight_scale=topup_score_scale,
            expert_disp_max=expert_disp_max,
            router_conf_min=router_conf_min,
            hard_abstain_symbols=hard_abstain_symbols,
            hard_abstain_regimes=hard_abstain_regimes,
            hard_abstain_symbol_regimes=hard_abstain_symbol_regimes,
            cadence_quality_floor=cadence_quality_floor,
            quality_floor=quality_floor,
            allocator_quality_blend=allocator_quality_blend,
            allocator_quality_power=allocator_quality_power,
            allocator_utility_weight=allocator_utility_weight,
            allocator_utility_power=allocator_utility_power,
            leverage_max=leverage_max,
            leverage_quality_floor=leverage_quality_floor,
            leverage_quality_power=leverage_quality_power,
            target_col=target_col,
        )
    else:
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
            min_trades_per_month_hard=0,
            shock_z_cut=shock_z_cut,
            shock_p_boost=shock_p_boost,
            shock_meta_boost=shock_meta_boost,
            hard_abstain_symbols=hard_abstain_symbols,
            hard_abstain_regimes=hard_abstain_regimes,
            hard_abstain_symbol_regimes=hard_abstain_symbol_regimes,
            quality_floor=quality_floor,
            allocator_quality_blend=allocator_quality_blend,
            allocator_quality_power=allocator_quality_power,
            allocator_utility_weight=allocator_utility_weight,
            allocator_utility_power=allocator_utility_power,
            leverage_max=leverage_max,
            leverage_quality_floor=leverage_quality_floor,
            leverage_quality_power=leverage_quality_power,
            target_col=target_col,
        )
    if x.empty:
        return pd.DataFrame()
    if float(conf_weight_power) != 1.0:
        conf = np.clip(pd.to_numeric(x["meta_p"], errors="coerce").to_numpy(dtype="float64"), 0.0, 1.0)
        scale = np.power(conf, float(max(0.1, conf_weight_power)))
        x["weight"] = pd.to_numeric(x["weight"], errors="coerce").to_numpy(dtype="float64") * scale
        wsum = x.groupby("ts")["weight"].transform("sum")
        x["weight"] = np.where(wsum > 1e-12, x["weight"] / wsum, x["weight"])
        x["r_weighted"] = x["r_net"] * x["weight"]
        x["net_bps_weighted"] = x["net"] * x["weight"] * 1e4
    x = _apply_monthly_loss_governor(
        x=x,
        monthly_loss_cap_r=float(monthly_loss_cap_r),
        reentry_drawdown_r=float(reentry_drawdown_r),
        post_cap_scale=float(post_cap_scale),
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


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        f = float(v)
        if np.isfinite(f):
            return f
    except Exception:
        pass
    return float(default)


def _safe_int(v: object, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def main() -> None:
    args = parse_args()

    symbols = list(dict.fromkeys(args.symbols))
    context_symbols = list(dict.fromkeys(args.context_symbols)) or symbols.copy()
    abstain_symbols = tuple(str(v).upper() for v in getattr(args, "abstain_symbols", []))
    abstain_regimes = tuple(str(v).upper() for v in args.abstain_regimes)
    abstain_symbol_regimes = _parse_symbol_regime_pairs(list(args.abstain_symbol_regimes))
    target_col = _resolve_target_col(str(args.primary_horizon))
    if args.auto_symbols_from_cache:
        cached_symbols = _discover_cached_symbols()
        symbols = sorted(set(symbols) | set(cached_symbols))
        context_symbols = sorted(set(context_symbols) | set(cached_symbols))

    oos, usable_context = ensure_oos_cache(
        symbols=symbols,
        context_symbols=context_symbols,
        rebuild_cache=args.rebuild_cache,
        min_bars=args.min_bars,
        model_family=str(args.model_family),
        sq3_seq_weight=float(args.sq3_seq_weight),
        sq3_seq_window=int(args.sq3_seq_window),
        wf_train_months=int(args.wf_train_months),
        wf_test_months=int(args.wf_test_months),
        wf_folds=int(args.wf_folds),
        fast_oos_mode=bool(args.fast_all_symbol_mode),
        primary_horizon=str(args.primary_horizon),
        secondary_horizon=str(args.secondary_horizon),
        secondary_horizon_weight=float(args.secondary_horizon_weight),
        tail_objective_weight=float(args.tail_objective_weight),
        utility_label_mode=str(args.utility_label_mode),
        utility_downside_mult=float(max(1.0, args.utility_downside_mult)),
    )
    if oos.empty:
        raise RuntimeError("no OOS predictions")

    if bool(getattr(args, "auto_abstain_symbol_regimes", False)):
        learned_pairs = _derive_symbol_regime_abstentions(
            _ensure_utility_columns(oos),
            min_samples=int(getattr(args, "abstain_min_samples", 120)),
            min_mean_r=float(getattr(args, "abstain_min_mean_r", -0.03)),
            min_win_rate=float(getattr(args, "abstain_min_win_p", 0.38)),
            top_k=int(getattr(args, "abstain_top_k", 8)),
        )
        if learned_pairs:
            abstain_symbol_regimes = tuple(sorted(set(abstain_symbol_regimes) | set(learned_pairs)))

    policies = search_policies_with_args(oos, args)
    if policies.empty:
        raise RuntimeError("no valid policies evaluated")

    robust = policies[
        (policies["months"] >= 10)
        & (policies["n"] >= 300)
        & (policies["fold_min_avg_r"] > 0)
        & (policies["mean_monthly_trades"] >= float(args.min_monthly_trades))
    ].copy()
    robust["robust_score"] = (
        robust["mean_monthly_total_r"]
        + 1.7 * robust["avg_r_net"]
        + 0.025 * robust["win_month_pct"]
        + 0.01 * robust["weighted_win_rate_pct"]
        + 0.004 * robust["mean_monthly_trades"]
    )
    robust = robust.sort_values(
        ["robust_score", "mean_monthly_total_r", "avg_r_net"],
        ascending=False,
    )

    best_overall = policies.sort_values("mean_monthly_total_r", ascending=False).head(25)
    best_tail = policies.sort_values(
        ["last10_total_r", "worst_month_r", "mean_monthly_total_r"],
        ascending=False,
    ).head(25)
    best_robust = robust.head(25)
    # If robust set is empty, prefer strongest tail profile over pure headline monthly R.
    chosen_from = "robust" if not best_robust.empty else "tail"
    chosen = best_robust.iloc[0] if not best_robust.empty else best_tail.iloc[0]
    monthly = _monthly_breakdown_for_policy(
        oos,
        chosen,
        min_trades_per_month_hard=int(args.min_monthly_trades_hard),
        shock_z_cut=float(args.shock_z_cut),
        shock_p_boost=float(args.shock_p_boost),
        shock_meta_boost=float(args.shock_meta_boost),
        topup_score_scale=float(args.topup_score_scale),
        expert_disp_max=float(args.expert_disp_maxs[0]),
        router_conf_min=float(args.router_conf_mins[0]),
        monthly_loss_cap_r=float(chosen.get("month_loss_cap_r", 0.0)),
        reentry_drawdown_r=float(chosen.get("reentry_drawdown_r", 0.0)),
        conf_weight_power=float(chosen.get("conf_weight_power", 1.0)),
        post_cap_scale=float(chosen.get("post_cap_scale", 0.25)),
        hard_abstain_symbols=abstain_symbols,
        hard_abstain_regimes=abstain_regimes,
        hard_abstain_symbol_regimes=abstain_symbol_regimes,
        cadence_quality_floor=float(args.cadence_quality_floor),
        quality_floor=float(chosen.get("quality_floor", 0.0)),
        allocator_quality_blend=float(chosen.get("allocator_quality_blend", 0.25)),
        allocator_quality_power=float(chosen.get("allocator_quality_power", 1.2)),
        leverage_max=float(chosen.get("leverage_max", 1.0)),
        leverage_quality_floor=float(chosen.get("leverage_quality_floor", 0.8)),
        leverage_quality_power=float(chosen.get("leverage_quality_power", 1.5)),
        target_col=target_col,
    )

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
        "feature_pack": "sq3-seq-moe-macro-allocator" if str(args.model_family).lower().startswith("sq3")
        else "sq21-macro-context-meta-allocator",
        "model_family": str(args.model_family),
        "sq3_seq_weight": float(args.sq3_seq_weight),
        "sq3_seq_window": int(args.sq3_seq_window),
        "primary_horizon": str(args.primary_horizon),
        "secondary_horizon": str(args.secondary_horizon),
        "secondary_horizon_weight": float(args.secondary_horizon_weight),
        "tail_objective_weight": float(args.tail_objective_weight),
        "utility_label_mode": str(args.utility_label_mode),
        "utility_downside_mult": float(max(1.0, args.utility_downside_mult)),
        "target_col": str(target_col),
        "wf_train_months": int(args.wf_train_months),
        "wf_test_months": int(args.wf_test_months),
        "wf_folds": int(args.wf_folds),
        "cost_bps": COST_BPS,
        "oos_rows": int(len(oos)),
        "symbols": sorted(oos["symbol"].unique().tolist()),
        "context_symbols": usable_context,
        "total_policies": int(len(policies)),
        "best_overall_monthly_r": ceiling,
        "best_robust_monthly_r": robust_ceiling,
        "min_monthly_trades_constraint": float(args.min_monthly_trades),
        "min_monthly_trades_hard": int(args.min_monthly_trades_hard),
        "shock_z_cut": float(args.shock_z_cut),
        "shock_p_boost": float(args.shock_p_boost),
        "shock_meta_boost": float(args.shock_meta_boost),
        "topup_score_scale": float(args.topup_score_scale),
        "cadence_quality_floor": float(args.cadence_quality_floor),
        "abstain_symbols": list(abstain_symbols),
        "abstain_regimes": list(abstain_regimes),
        "abstain_symbol_regimes": [f"{s}:{r}" for s, r in abstain_symbol_regimes],
        "router_conf_mins": [float(v) for v in args.router_conf_mins],
        "expert_disp_maxs": [float(v) for v in args.expert_disp_maxs],
        "monthly_loss_cap_rs": [float(v) for v in args.monthly_loss_cap_rs],
        "reentry_drawdown_rs": [float(v) for v in args.reentry_drawdown_rs],
        "conf_weight_powers": [float(v) for v in args.conf_weight_powers],
        "post_cap_scales": [float(v) for v in args.post_cap_scales],
        "quality_floor": float(args.quality_floors[0]) if args.quality_floors else 0.0,
        "allocator_quality_blends": [float(v) for v in args.allocator_quality_blends],
        "allocator_quality_powers": [float(v) for v in args.allocator_quality_powers],
        "allocator_utility_weights": [float(v) for v in args.allocator_utility_weights],
        "allocator_utility_powers": [float(v) for v in args.allocator_utility_powers],
        "auto_abstain_symbol_regimes": bool(args.auto_abstain_symbol_regimes),
        "abstain_min_samples": int(args.abstain_min_samples),
        "abstain_min_mean_r": float(args.abstain_min_mean_r),
        "abstain_min_win_p": float(args.abstain_min_win_p),
        "abstain_top_k": int(args.abstain_top_k),
        "leverage_maxs": [float(v) for v in args.leverage_maxs],
        "leverage_quality_floors": [float(v) for v in args.leverage_quality_floors],
        "leverage_quality_powers": [float(v) for v in args.leverage_quality_powers],
        "best_overall": best_overall.to_dict("records"),
        "best_tail": best_tail.to_dict("records"),
        "best_robust": best_robust.to_dict("records"),
        "selected_policy_source": chosen_from,
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
        f"- Model family: **{str(args.model_family)}**",
        f"- Feature pack: **{'SQ3 sequence blend + macro context + meta gate + top-K allocator' if str(args.model_family).lower().startswith('sq3') else 'macro context + meta tradeability gate + top-K allocator'}**",
        f"- SQ3 sequence weight: **{float(args.sq3_seq_weight):.2f}**",
        f"- SQ3 sequence window: **{int(args.sq3_seq_window)}**",
        f"- Primary/secondary horizon: **{str(args.primary_horizon)}/{str(args.secondary_horizon)}** (secondary weight **{float(args.secondary_horizon_weight):.2f}**)",
        f"- Tail objective weight: **{float(args.tail_objective_weight):.2f}**",
        f"- Target return column: **{target_col}**",
        f"- Walk-forward train/test/folds: **{int(args.wf_train_months)}/{int(args.wf_test_months)}/{int(args.wf_folds)}**",
        f"- OOS rows: **{len(oos):,}**",
        f"- Active symbols: **{', '.join(sorted(oos['symbol'].unique()))}**",
        f"- Context symbols: **{', '.join(usable_context)}**",
        f"- Policies tested: **{len(policies):,}**",
        f"- Min monthly trades constraint (robust): **{float(args.min_monthly_trades):.1f}**",
        f"- Adaptive hard pacing target (per month): **{int(args.min_monthly_trades_hard)}**",
        f"- Adaptive cadence quality floor: **{float(args.cadence_quality_floor):.2f}**",
        f"- Global quality floor grid: **{', '.join(f'{float(v):.2f}' for v in args.quality_floors)}**",
        f"- Allocator quality blend grid: **{', '.join(f'{float(v):.2f}' for v in args.allocator_quality_blends)}**",
        f"- Allocator quality power grid: **{', '.join(f'{float(v):.2f}' for v in args.allocator_quality_powers)}**",
        f"- Allocator utility weight grid: **{', '.join(f'{float(v):.2f}' for v in args.allocator_utility_weights)}**",
        f"- Allocator utility power grid: **{', '.join(f'{float(v):.2f}' for v in args.allocator_utility_powers)}**",
        f"- Adaptive top-up weight scale: **{float(args.topup_score_scale):.2f}**",
        f"- Leverage max grid: **{', '.join(f'{float(v):.2f}' for v in args.leverage_maxs)}**",
        f"- Leverage quality floor grid: **{', '.join(f'{float(v):.2f}' for v in args.leverage_quality_floors)}**",
        f"- Leverage quality power grid: **{', '.join(f'{float(v):.2f}' for v in args.leverage_quality_powers)}**",
        f"- Hard-abstain symbols: **{', '.join(abstain_symbols) if abstain_symbols else 'None'}**",
        f"- Hard-abstain regimes: **{', '.join(abstain_regimes) if abstain_regimes else 'None'}**",
        f"- Hard-abstain symbol:regime pairs: **{', '.join(f'{s}:{r}' for s, r in abstain_symbol_regimes) if abstain_symbol_regimes else 'None'}**",
        f"- Router confidence min grid: **{', '.join(f'{float(v):.2f}' for v in args.router_conf_mins)}**",
        f"- Expert dispersion max grid: **{', '.join(f'{float(v):.2f}' for v in args.expert_disp_maxs)}**",
        f"- Best monthly R (overall): **{ceiling:+.2f}**",
        f"- Best monthly R (robust): **{robust_ceiling:+.2f}**",
        "",
        "## Top robust policies",
        "",
        "| top% | symbols | long_only | p_min | meta_min | risk>=bps | max/ts | capR | reentryR | postCap | n | months | mean mth trades | min mth trades | trade win% | weighted win% | avg_R | mean monthly R | last10 R | worst month R |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in best_robust.iterrows():
        lines.append(
            f"| {r['top_pct']:.1f} | {r['symbols']} | "
            f"{'Y' if bool(r['long_only']) else 'N'} | {r['p_min']:.2f} | {r['meta_min']:.2f} | "
            f"{r['risk_min_bps']:.0f} | {int(r['max_per_ts'])} | "
            f"{float(r.get('month_loss_cap_r', 0.0)):.1f} | {float(r.get('reentry_drawdown_r', 0.0)):.1f} | {float(r.get('post_cap_scale', 0.0)):.2f} | "
            f"{int(r['n']):,} | {int(r['months'])} | "
            f"{r['mean_monthly_trades']:.1f} | {int(r['min_monthly_trades'])} | "
            f"{r['trade_win_rate_pct']:.1f} | {r['weighted_win_rate_pct']:.1f} | "
            f"{r['avg_r_net']:+.3f} | {r['mean_monthly_total_r']:+.2f} | {r['last10_total_r']:+.2f} | {r['worst_month_r']:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Top tail-resilient policies (last-10-month objective)",
            "",
            "| top% | symbols | long_only | p_min | meta_min | risk>=bps | max/ts | capR | reentryR | postCap | mean monthly R | last10 R | worst month R | mean mth trades |",
            "|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, r in best_tail.iterrows():
        lines.append(
            f"| {r['top_pct']:.1f} | {r['symbols']} | "
            f"{'Y' if bool(r['long_only']) else 'N'} | {r['p_min']:.2f} | {r['meta_min']:.2f} | "
            f"{r['risk_min_bps']:.0f} | {int(r['max_per_ts'])} | "
            f"{float(r.get('month_loss_cap_r', 0.0)):.1f} | {float(r.get('reentry_drawdown_r', 0.0)):.1f} | {float(r.get('post_cap_scale', 0.0)):.2f} | "
            f"{r['mean_monthly_total_r']:+.2f} | {r['last10_total_r']:+.2f} | {r['worst_month_r']:+.2f} | "
            f"{r['mean_monthly_trades']:.1f} |"
        )

    lines.extend(
        [
            "",
            f"## Monthly breakdown (selected {chosen_from} policy)",
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
