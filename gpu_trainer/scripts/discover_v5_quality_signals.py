#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.pipeline import FeatureEngineer


@dataclass
class FoldWindow:
    train_months: list[str]
    test_months: list[str]


def _to_utc_ms(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)


def _safe_spearman(x: pd.Series, y: pd.Series, min_rows: int = 200) -> float:
    mask = np.isfinite(x.values) & np.isfinite(y.values)
    if int(mask.sum()) < min_rows:
        return float("nan")
    xv = x.values[mask]
    yv = y.values[mask]
    if np.nanstd(xv) < 1e-12 or np.nanstd(yv) < 1e-12:
        return float("nan")
    xs = pd.Series(xv).rank(method="average")
    ys = pd.Series(yv).rank(method="average")
    return float(xs.corr(ys))


def _safe_mean(values: list[float], nan_if_empty: bool = False) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan") if nan_if_empty else 0.0
    return float(arr.mean())


def _safe_std(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(arr.std(ddof=0))


def _generate_regime_labels(feat: pd.DataFrame) -> pd.Series:
    trend = pd.to_numeric(feat["regime_trend"], errors="coerce").fillna(0.0)
    labels = np.where(
        trend > 0.30,
        "trending_up",
        np.where(trend < -0.30, "trending_down", "choppy"),
    )
    return pd.Series(labels, index=feat.index, name="regime")


def _monthly_folds(
    months: list[str],
    train_months: int,
    test_months: int,
    max_folds: int,
) -> list[FoldWindow]:
    folds: list[FoldWindow] = []
    if len(months) < train_months + test_months:
        return folds
    for i in range(train_months, len(months) - test_months + 1):
        train_win = months[i - train_months : i]
        test_win = months[i : i + test_months]
        folds.append(FoldWindow(train_months=train_win, test_months=test_win))
    if max_folds > 0 and len(folds) > max_folds:
        folds = folds[-max_folds:]
    return folds


def _stability_score(mean_ic: float, std_ic: float, sign_consistency: float, n_folds: int) -> float:
    if not np.isfinite(mean_ic):
        return -1e9
    if n_folds <= 0:
        return -1e9
    fold_weight = min(1.0, n_folds / 8.0)
    variance_pen = 1.0 / (1.0 + max(0.0, std_ic) * 8.0)
    return float(abs(mean_ic) * sign_consistency * variance_pen * fold_weight)


def _prepare_symbol_dataset(
    data_path: Path,
    symbol: str,
    horizons: list[int],
    lookback_months: int,
) -> pd.DataFrame:
    df = pd.read_parquet(data_path).copy()
    if "timestamp" not in df.columns:
        raise ValueError(f"{symbol}: timestamp column missing")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df[np.isfinite(df["timestamp"])].sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{symbol}: empty timestamp after cleanup")

    if lookback_months > 0:
        end_ts = int(df["timestamp"].iloc[-1])
        end_dt = _to_utc_ms(end_ts)
        # Keep one extra month for indicator warmup before the target analysis window.
        cutoff_dt = (pd.Timestamp(end_dt) - pd.DateOffset(months=lookback_months + 1)).to_pydatetime()
        cutoff_ms = int(cutoff_dt.timestamp() * 1000)
        df = df[df["timestamp"] >= cutoff_ms].reset_index(drop=True)

    fe = FeatureEngineer()
    feat = fe.compute_all_features(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    out = feat.copy()
    out["timestamp"] = df["timestamp"].values
    out["symbol"] = symbol
    out["close"] = pd.to_numeric(df["close"], errors="coerce").values
    out["regime"] = _generate_regime_labels(feat).values
    for h in horizons:
        out[f"target_logret_{h}"] = np.log(out["close"].shift(-h) / out["close"])

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _build_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# V5 Deep Signal Discovery Report")
    lines.append("")
    lines.append(f"- Generated UTC: {report.get('generated_utc', 'unknown')}")
    lines.append(f"- Symbols: {', '.join(report.get('symbols', []))}")
    lines.append(f"- Lookback months: {report.get('lookback_months', 0)}")
    lines.append(
        f"- Walk-forward: train={report.get('train_months', 0)} months, "
        f"test={report.get('test_months', 0)} month(s), max_folds={report.get('max_folds', 0)}"
    )
    best_ic = report.get("best_horizon_by_ic", {})
    if best_ic:
        lines.append(
            f"- Best horizon by OOS IC: {best_ic.get('horizon', 'n/a')} bars "
            f"(IC={float(best_ic.get('mean_ic', 0.0)):+.4f})"
        )
    best_spread = report.get("best_horizon_by_spread", {})
    if best_spread:
        lines.append(
            f"- Best horizon by decile spread: {best_spread.get('horizon', 'n/a')} bars "
            f"(spread={float(best_spread.get('decile_spread', 0.0)):+.5f})"
        )
    lines.append("")

    horizons = report.get("horizons", {})
    for horizon_key, payload in horizons.items():
        lines.append(f"## Horizon {horizon_key} bars")
        lines.append("")
        lines.append(
            f"- Rows analyzed: {payload.get('rows', 0)} | folds: {payload.get('folds', 0)}"
        )
        qm = payload.get("quality_model", {})
        lines.append(
            f"- Quality model OOS IC: {qm.get('mean_ic', 0.0):+.4f} "
            f"(std={qm.get('std_ic', 0.0):.4f}), "
            f"top-decile mean logret={qm.get('top_decile_mean_ret', 0.0):+.5f}, "
            f"top-bottom spread={qm.get('decile_spread', 0.0):+.5f}, "
            f"directional acc={100.0 * qm.get('directional_acc', 0.0):.2f}%"
        )
        lines.append("")
        lines.append("### Top stable signals (OOS)")
        lines.append("")
        lines.append("| Feature | Mean IC | Std IC | Sign Consistency | Stability |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in payload.get("top_features", [])[:12]:
            lines.append(
                f"| {row['feature']} | {row['mean_ic']:+.4f} | {row['std_ic']:.4f} | "
                f"{100.0 * row['sign_consistency']:.1f}% | {row['stability_score']:.5f} |"
            )
        lines.append("")
        lines.append("### Regime-specific best signals")
        lines.append("")
        for regime, reg_rows in payload.get("regime_top_features", {}).items():
            finite_rows = [
                r for r in reg_rows if np.isfinite(float(r.get("regime_ic", float("nan"))))
            ]
            if not finite_rows:
                lines.append(f"- **{regime}**: insufficient samples for stable ranking")
                continue
            lines.append(
                f"- **{regime}**: "
                + ", ".join(
                    f"{r['feature']} ({float(r['regime_ic']):+.3f})"
                    for r in finite_rows[:6]
                )
            )
        lines.append("")
        lines.append("### Suggested quality-signal blend")
        lines.append("")
        blend = payload.get("quality_blend", [])
        if blend:
            blend_str = " + ".join(f"{w:+.3f}*rank({f})" for f, w in blend[:8])
            lines.append(f"`quality_signal = {blend_str}`")
        else:
            lines.append("No stable blend generated.")
        lines.append("")

    lines.append("## Practical prediction guidance")
    lines.append("")
    lines.append(
        "1. Use **regime-conditional signals** first (trend regime != choppy regime) and avoid one global threshold."
    )
    lines.append(
        "2. Rank bars by the quality blend and only trade top quantiles; this improves precision at the cost of frequency."
    )
    lines.append(
        "3. Track stability drift monthly: if signal IC or consistency drops, reduce risk and retrain."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def _fit_fold_quality_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    top_n: int,
) -> tuple[dict[str, float], list[str]]:
    train_ic: dict[str, float] = {}
    for col in feature_cols:
        train_ic[col] = _safe_spearman(train_df[col], train_df[target_col], min_rows=250)

    ranked = sorted(
        ((c, v) for c, v in train_ic.items() if np.isfinite(v)),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    selected = [c for c, _ in ranked[: max(3, top_n)]]
    if not selected:
        return (
            {
                "ic": float("nan"),
                "top_decile_ret": float("nan"),
                "bottom_decile_ret": float("nan"),
                "decile_spread": float("nan"),
                "directional_acc": float("nan"),
            },
            [],
        )

    tr = train_df[selected + ["symbol", target_col]].copy().dropna()
    te = test_df[selected + ["symbol", target_col]].copy().dropna()
    if tr.empty or te.empty:
        return (
            {
                "ic": float("nan"),
                "top_decile_ret": float("nan"),
                "bottom_decile_ret": float("nan"),
                "decile_spread": float("nan"),
                "directional_acc": float("nan"),
            },
            selected,
        )

    sym_dummies_tr = pd.get_dummies(tr["symbol"], prefix="sym", dtype=float)
    sym_dummies_te = pd.get_dummies(te["symbol"], prefix="sym", dtype=float)
    sym_dummies_te = sym_dummies_te.reindex(columns=sym_dummies_tr.columns, fill_value=0.0)

    x_tr = pd.concat([tr[selected].astype(float), sym_dummies_tr], axis=1)
    x_te = pd.concat([te[selected].astype(float), sym_dummies_te], axis=1)
    y_tr = tr[target_col].astype(float)
    y_te = te[target_col].astype(float)

    med = x_tr.median(axis=0)
    iqr = (x_tr.quantile(0.75) - x_tr.quantile(0.25)).replace(0.0, 1.0)
    x_trs = (x_tr - med) / iqr
    x_tes = (x_te - med) / iqr

    model = Ridge(alpha=2.0, random_state=42)
    model.fit(x_trs, y_tr)
    pred = pd.Series(model.predict(x_tes), index=x_tes.index)

    ic = _safe_spearman(pred, y_te, min_rows=250)
    q90 = float(pred.quantile(0.90))
    q10 = float(pred.quantile(0.10))
    top_ret = float(y_te[pred >= q90].mean()) if (pred >= q90).any() else float("nan")
    bot_ret = float(y_te[pred <= q10].mean()) if (pred <= q10).any() else float("nan")
    spread = (
        float(top_ret - bot_ret)
        if np.isfinite(top_ret) and np.isfinite(bot_ret)
        else float("nan")
    )
    direction = np.sign(pred.values)
    target_direction = np.sign(y_te.values)
    valid = direction != 0
    directional_acc = (
        float((direction[valid] == target_direction[valid]).mean()) if valid.any() else float("nan")
    )
    return (
        {
            "ic": ic,
            "top_decile_ret": top_ret,
            "bottom_decile_ret": bot_ret,
            "decile_spread": spread,
            "directional_acc": directional_acc,
        },
        selected,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover robust V5 market-predictive signals.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/workspace/gpu_trainer/data_cache",
        help="Directory containing *_15m.parquet files",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT", "XRPUSDT"],
        help="Symbols for multi-asset signal discovery",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="16,32,48",
        help="Comma-separated prediction horizons in bars",
    )
    parser.add_argument("--lookback-months", type=int, default=24)
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--max-folds", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--model-top-n", type=int, default=12)
    parser.add_argument("--min-fold-rows", type=int, default=2000)
    parser.add_argument(
        "--output-json",
        type=str,
        default="/workspace/gpu_trainer/checkpoints/v5_signal_discovery_report.json",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="/workspace/gpu_trainer/checkpoints/v5_signal_discovery_report.md",
    )
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"data dir not found: {data_dir}")

    all_frames: list[pd.DataFrame] = []
    for symbol in args.symbols:
        path = data_dir / f"{symbol}_15m.parquet"
        if not path.exists():
            continue
        frame = _prepare_symbol_dataset(path, symbol, horizons, args.lookback_months)
        if not frame.empty:
            all_frames.append(frame)
    if not all_frames:
        raise SystemExit("No symbol datasets loaded.")

    full = pd.concat(all_frames, axis=0, ignore_index=True)
    full["timestamp"] = pd.to_numeric(full["timestamp"], errors="coerce")
    full = full[np.isfinite(full["timestamp"])].copy()
    full["month"] = pd.to_datetime(full["timestamp"], unit="ms", utc=True).dt.to_period("M").astype(str)
    full = full.replace([np.inf, -np.inf], np.nan)

    base_drop = {"timestamp", "symbol", "close", "regime", "month"}
    target_cols = {f"target_logret_{h}" for h in horizons}
    feature_cols = [c for c in full.columns if c not in (base_drop | target_cols)]
    if not feature_cols:
        raise SystemExit("No feature columns detected.")

    months = sorted(full["month"].dropna().unique().tolist())
    folds = _monthly_folds(months, args.train_months, args.test_months, args.max_folds)
    if not folds:
        raise SystemExit("No walk-forward folds available with current settings.")

    report: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "symbols": sorted(full["symbol"].dropna().unique().tolist()),
        "lookback_months": int(args.lookback_months),
        "train_months": int(args.train_months),
        "test_months": int(args.test_months),
        "max_folds": int(args.max_folds),
        "horizons": {},
        "best_horizon_by_ic": {},
        "best_horizon_by_spread": {},
    }

    regimes = ["trending_up", "trending_down", "choppy"]
    for h in horizons:
        target_col = f"target_logret_{h}"
        hdf = full.copy()
        hdf = hdf[np.isfinite(hdf[target_col].values)].copy()
        if hdf.empty:
            continue

        feature_fold_ic: dict[str, list[float]] = {c: [] for c in feature_cols}
        feature_regime_ic: dict[str, dict[str, list[float]]] = {
            c: {r: [] for r in regimes} for c in feature_cols
        }
        model_metrics_by_fold: list[dict[str, float]] = []
        selected_hist: dict[str, int] = {}
        used_folds = 0

        for fold in folds:
            tr_mask = hdf["month"].isin(fold.train_months).values
            te_mask = hdf["month"].isin(fold.test_months).values
            tr_df = hdf.loc[tr_mask]
            te_df = hdf.loc[te_mask]
            if len(tr_df) < args.min_fold_rows or len(te_df) < args.min_fold_rows:
                continue
            used_folds += 1

            for col in feature_cols:
                ic = _safe_spearman(te_df[col], te_df[target_col], min_rows=250)
                if np.isfinite(ic):
                    feature_fold_ic[col].append(float(ic))
                for reg in regimes:
                    reg_df = te_df[te_df["regime"] == reg]
                    ric = _safe_spearman(reg_df[col], reg_df[target_col], min_rows=80)
                    if np.isfinite(ric):
                        feature_regime_ic[col][reg].append(float(ric))

            mm, selected = _fit_fold_quality_model(
                train_df=tr_df,
                test_df=te_df,
                feature_cols=feature_cols,
                target_col=target_col,
                top_n=args.model_top_n,
            )
            model_metrics_by_fold.append(mm)
            for fcol in selected:
                selected_hist[fcol] = selected_hist.get(fcol, 0) + 1

        ranked_rows: list[dict[str, Any]] = []
        for col in feature_cols:
            arr = np.asarray(feature_fold_ic[col], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            mean_ic = float(arr.mean())
            std_ic = float(arr.std(ddof=0))
            pos_ratio = float((arr > 0.0).mean())
            sign_consistency = max(pos_ratio, 1.0 - pos_ratio)
            score = _stability_score(mean_ic, std_ic, sign_consistency, n_folds=int(arr.size))
            reg_map = {
                reg: _safe_mean(feature_regime_ic[col][reg], nan_if_empty=True)
                for reg in regimes
            }
            ranked_rows.append(
                {
                    "feature": col,
                    "mean_ic": mean_ic,
                    "std_ic": std_ic,
                    "n_folds": int(arr.size),
                    "sign_consistency": sign_consistency,
                    "stability_score": score,
                    "regime_ic": reg_map,
                }
            )

        ranked_rows.sort(key=lambda r: r["stability_score"], reverse=True)
        top_features = ranked_rows[: max(1, args.top_k)]
        top_feature_names = [r["feature"] for r in top_features]

        regime_top: dict[str, list[dict[str, Any]]] = {}
        for reg in regimes:
            reg_rows = []
            for r in ranked_rows:
                reg_ic = float(r["regime_ic"].get(reg, float("nan")))
                reg_abs = abs(reg_ic) if np.isfinite(reg_ic) else 0.0
                reg_rows.append(
                    {
                        "feature": r["feature"],
                        "regime_ic": reg_ic,
                        "global_stability": float(r["stability_score"]),
                    }
                )
            reg_rows.sort(
                key=lambda x: (
                    (abs(float(x["regime_ic"])) if np.isfinite(float(x["regime_ic"])) else 0.0)
                    * (1.0 + x["global_stability"])
                ),
                reverse=True,
            )
            regime_top[reg] = reg_rows[: max(1, min(10, args.top_k))]

        # Build a direct quality blend from top stable features.
        weights_raw = []
        for row in top_features:
            w = np.sign(row["mean_ic"]) * row["stability_score"]
            weights_raw.append((row["feature"], float(w)))
        norm = sum(abs(w) for _, w in weights_raw) or 1.0
        quality_blend = [(f, float(w / norm)) for f, w in weights_raw]

        model_payload = {
            "mean_ic": _safe_mean([m.get("ic", np.nan) for m in model_metrics_by_fold]),
            "std_ic": _safe_std([m.get("ic", np.nan) for m in model_metrics_by_fold]),
            "top_decile_mean_ret": _safe_mean([m.get("top_decile_ret", np.nan) for m in model_metrics_by_fold]),
            "bottom_decile_mean_ret": _safe_mean([m.get("bottom_decile_ret", np.nan) for m in model_metrics_by_fold]),
            "decile_spread": _safe_mean([m.get("decile_spread", np.nan) for m in model_metrics_by_fold]),
            "directional_acc": _safe_mean([m.get("directional_acc", np.nan) for m in model_metrics_by_fold]),
            "fold_count": int(len(model_metrics_by_fold)),
            "most_selected_features": sorted(
                (
                    {"feature": k, "count": int(v)}
                    for k, v in selected_hist.items()
                ),
                key=lambda x: x["count"],
                reverse=True,
            )[:15],
        }

        report["horizons"][str(h)] = {
            "rows": int(len(hdf)),
            "folds": int(used_folds),
            "feature_count": int(len(feature_cols)),
            "top_features": top_features,
            "regime_top_features": regime_top,
            "quality_blend": quality_blend[: max(1, min(20, len(quality_blend)))],
            "quality_model": model_payload,
            "top_feature_names": top_feature_names,
        }

    horizon_summary = []
    for hkey, payload in report["horizons"].items():
        qm = payload.get("quality_model", {})
        horizon_summary.append(
            {
                "horizon": int(hkey),
                "mean_ic": float(qm.get("mean_ic", float("nan"))),
                "decile_spread": float(qm.get("decile_spread", float("nan"))),
            }
        )
    ic_candidates = [r for r in horizon_summary if np.isfinite(r["mean_ic"])]
    spread_candidates = [r for r in horizon_summary if np.isfinite(r["decile_spread"])]
    if ic_candidates:
        report["best_horizon_by_ic"] = max(ic_candidates, key=lambda r: r["mean_ic"])
    if spread_candidates:
        report["best_horizon_by_spread"] = max(spread_candidates, key=lambda r: r["decile_spread"])

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, indent=2))
    out_md.write_text(_build_markdown_report(report))

    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")
    for horizon_key, payload in report.get("horizons", {}).items():
        qm = payload.get("quality_model", {})
        print(
            f"[h={horizon_key}] folds={payload.get('folds', 0)} "
            f"OOS_IC={qm.get('mean_ic', 0.0):+.4f} "
            f"Spread={qm.get('decile_spread', 0.0):+.5f} "
            f"DirAcc={100.0 * qm.get('directional_acc', 0.0):.2f}%"
        )


if __name__ == "__main__":
    main()
