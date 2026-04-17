"""
Compose all feature blocks into one strictly-causal feature matrix.

Per locked contract:
    Group                 Count
    Frac-diff returns     4
    Base technical        ~88
    Microstructure        6 (signed_vol_z, bar_vel, rv_short, rv_med, rv_long, bar_dur_min)
    Cross-asset           4
    Regime                3 (vol_of_vol, atr_pct_bucket, adx_regime)
    Total                 ~105

`atr_14` and `adx_14` are also returned (in a separate side-DataFrame)
for downstream use by the label layer (barrier widths, primary rules).
They are NOT included in the model feature matrix.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gpu_trainer_v11.features.base_technical import compute_base_technical
from gpu_trainer_v11.features.cross_asset import compute_cross_asset
from gpu_trainer_v11.features.frac_diff import frac_diff_ffd
from gpu_trainer_v11.features.microstructure import compute_microstructure
from gpu_trainer_v11.features.regime import compute_regime

# Locked d-grid for fractional differentiation. The auto-d picker can
# select any of these per-symbol; recording the grid here keeps the
# process reproducible.
FRAC_DIFF_DS = [0.3, 0.5, 0.7, 0.9]
MODEL_FEATURE_COLS_EXCLUDED = {"atr_14", "adx_14"}  # exposed but not modeled


@dataclass
class FeatureBundle:
    features: pd.DataFrame      # columns fed to the model
    side_data: pd.DataFrame     # atr_14, adx_14, etc. — for labels & rules


def compute_features(
    bars: pd.DataFrame,
    btc_bars: pd.DataFrame | None,
    symbol: str,
) -> FeatureBundle:
    if len(bars) == 0:
        empty = pd.DataFrame(index=bars.index)
        return FeatureBundle(features=empty, side_data=empty)

    base = compute_base_technical(bars)
    micro = compute_microstructure(bars)
    regime = compute_regime(bars)
    cross = compute_cross_asset(bars, btc_bars, symbol)

    log_close = np.log(bars["close"].replace(0, np.nan))
    fd = pd.DataFrame(index=bars.index)
    for d in FRAC_DIFF_DS:
        fd[f"fd_log_close_d{int(d*10)}"] = frac_diff_ffd(log_close, d).fillna(0.0)

    # Curate microstructure to the locked count of 6 (drop signed_vol raw)
    micro_locked = micro.drop(columns=["signed_vol"])
    # Side data for downstream label/rule modules
    side = pd.DataFrame({
        "atr_14": regime["atr_14"],
        "adx_14": regime["adx_14"],
    }, index=bars.index)
    regime_model = regime.drop(columns=["atr_14", "adx_14"])

    feats = pd.concat([fd, base, micro_locked, cross, regime_model], axis=1)
    feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return FeatureBundle(features=feats, side_data=side)
