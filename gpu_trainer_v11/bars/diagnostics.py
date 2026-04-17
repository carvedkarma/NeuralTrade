"""
Dollar-bar threshold sweep.

Tries N candidate thresholds on BTCUSDT and reports for each:
  - bars/day (target ~96 to match 15m cadence)
  - median, p25, p75 bar duration in minutes
  - bars per month (consistency over time)

The sweep itself does NO labeling and NO modeling — it only sizes the
bar so the locked configuration in README.md is data-driven, not
guessed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gpu_trainer_v11.bars.dollar_bars import (
    DollarBarConfig,
    bars_per_day,
    build_dollar_bars,
)


@dataclass
class SweepRow:
    threshold_dollars: float
    n_bars: int
    bars_per_day: float
    median_minutes: float
    p25_minutes: float
    p75_minutes: float
    monthly_bar_cv: float  # coefficient of variation of bars-per-month


def sweep(df_15m: pd.DataFrame, candidates: list[float]) -> pd.DataFrame:
    rows: list[SweepRow] = []
    for thr in candidates:
        cfg = DollarBarConfig(threshold_dollars=thr)
        db = build_dollar_bars(df_15m, cfg)
        if len(db) < 2:
            rows.append(SweepRow(thr, len(db), 0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        bpd = bars_per_day(db)
        durations_ms = db["timestamp"].to_numpy()[1:] - db["timestamp"].to_numpy()[:-1]
        durations_min = durations_ms / 60000.0
        med = float(np.median(durations_min))
        p25 = float(np.percentile(durations_min, 25))
        p75 = float(np.percentile(durations_min, 75))
        ts = pd.to_datetime(db["timestamp"], unit="ms", utc=True)
        per_month = ts.dt.to_period("M").value_counts()
        cv = float(per_month.std() / per_month.mean()) if per_month.mean() > 0 else 0.0
        rows.append(SweepRow(thr, len(db), bpd, med, p25, p75, cv))
    return pd.DataFrame([r.__dict__ for r in rows])


def pick_threshold(sweep_df: pd.DataFrame, target_bars_per_day: float = 96.0) -> float:
    """Pick the threshold whose bars/day is closest to target.

    Tie-breaker (rare): prefer the threshold with lower monthly_bar_cv
    (more time-stable bar count).
    """
    if len(sweep_df) == 0:
        raise ValueError("empty sweep")
    df = sweep_df.copy()
    df["dist"] = (df["bars_per_day"] - target_bars_per_day).abs()
    df = df.sort_values(["dist", "monthly_bar_cv"], ascending=[True, True]).reset_index(drop=True)
    return float(df.iloc[0]["threshold_dollars"])
