from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


DEFAULT_RESEARCH_SYMBOLS: List[str] = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "LTCUSDT",
]


@dataclass
class PromotionGateConfig:
    min_total_r: float = 0.0
    min_expectancy_r: float = 0.0
    min_total_trades: int = 40
    min_active_folds: int = 2
    min_action_accuracy: float = 0.50
    min_mu_r_correlation: float = 0.0
    min_score_disc_p90p50: float = 3.0
    max_long_pct: float = 80.0
    require_vs_baseline: bool = True
    allow_total_r_regression: float = 0.0
    allow_expectancy_regression: float = 0.0


@dataclass
class ResearchProfile:
    name: str = "crypto-10-liquid-v1"
    symbols: List[str] = field(default_factory=lambda: list(DEFAULT_RESEARCH_SYMBOLS))
    interval: str = "15m"
    min_candles: int = 20_000
    epochs: int = 40
    batch_size: int = 128
    lr: float = 3e-4
    horizon: int = 16
    tp_mult: float = 2.0
    sl_mult: float = 1.2
    train_months: int = 9
    test_months: int = 1
    walk_forward_folds: int = 3
    cooldown: int = 4
    score_lambda: float = 0.50
    target_tpd: float = 4.5
    target_tpd_tol: float = 1.0
    threshold_warmup_epochs: int = 2
    threshold_step_mult: float = 0.05
    min_threshold_floor: float = 0.004
    phase1_epochs: int = 1
    promotion_gates: PromotionGateConfig = field(default_factory=PromotionGateConfig)

    def regime_side_map(self) -> Dict[str, str]:
        return {
            "trending_up": "LONG",
            "trending_down": "SHORT",
            "choppy": "BOTH",
        }

    def shared_v5_kwargs(self) -> Dict[str, Any]:
        return {
            "tp_mult": self.tp_mult,
            "sl_mult": self.sl_mult,
            "horizon": self.horizon,
            "score_lambda": 0.50,
            "risk_proxy": "mae",
            "hold_target": 0.30,
            "mfe_min": 0.05,
            "balanced_sampling": True,
            "balanced_sampling_mode": "cap",
            "symbol_embed_dim": 8,
            "per_symbol_scaler": False,
            "short_oversample": True,
            "short_min_fraction": 0.40,
            "per_symbol_threshold": True,
            "per_side_threshold": True,
            "ema200_soft_mult": 0.50,
            "adx_gate": True,
            "adx_min": 18.0,
            # Empirically healthy runs in this stack tend to cluster around ~0.003-0.007.
            # 0.04 was an order-of-magnitude too high for this score scale.
            "min_threshold": self.min_threshold_floor,
            "trailing_sl": True,
            "trail_activation": 1.5,
            "trail_distance": 1.0,
            "corr_block": True,
            "corr_thresh": 0.90,
            "corr_same_side_only": True,
            "side_aware_scoring": True,
            "recency_weight": True,
            "recency_half_life": 60,
            "cooldown": self.cooldown,
            "min_trades": 20,
            "regime_side_map": self.regime_side_map(),
            "sweep_objective": "quality",
            "slippage_base_bps": 6.0,
        }

    def train_v5_kwargs(self) -> Dict[str, Any]:
        kwargs = self.shared_v5_kwargs()
        kwargs.update(
            {
                "target_tpd": self.target_tpd,
                "target_tpd_tol": self.target_tpd_tol,
                "phase1_epochs": self.phase1_epochs,
                "run_forward_test": True,
                "run_diagnostics": True,
                "promote_metric": "expectancy",
            }
        )
        return kwargs

    def walk_forward_kwargs(self) -> Dict[str, Any]:
        kwargs = self.shared_v5_kwargs()
        kwargs.update(
            {
                "train_months": self.train_months,
                "test_months": self.test_months,
                "max_folds": self.walk_forward_folds,
                "phase1_epochs": self.phase1_epochs,
                "warm_start": True,
                "wf_threshold_ema": True,
                "wf_threshold_ema_alpha": 0.50,
                "wf_threshold_decay": 0.50,
                "promote_metric": "expectancy",
            }
        )
        return kwargs

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_research_profile(
    symbols: Optional[List[str]] = None,
    *,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    min_candles: Optional[int] = None,
    train_months: Optional[int] = None,
    test_months: Optional[int] = None,
    walk_forward_folds: Optional[int] = None,
) -> ResearchProfile:
    profile = ResearchProfile()
    if symbols:
        profile.symbols = list(symbols)
    if epochs is not None:
        profile.epochs = epochs
    if batch_size is not None:
        profile.batch_size = batch_size
    if lr is not None:
        profile.lr = lr
    if min_candles is not None:
        profile.min_candles = min_candles
    if train_months is not None:
        profile.train_months = train_months
    if test_months is not None:
        profile.test_months = test_months
    if walk_forward_folds is not None:
        profile.walk_forward_folds = walk_forward_folds
    return profile
