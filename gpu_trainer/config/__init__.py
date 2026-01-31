"""
GPU Trainer Configuration Module

Centralized configuration for timeframes, training parameters, and defaults.
"""

from .timeframe_config import (
    TimeframeConfig,
    TIMEFRAME_CONFIGS,
    DEFAULT_TIMEFRAME,
    DEFAULT_SYMBOL,
    get_config,
    get_horizon_for_timeframe,
    get_lookback_for_timeframe,
    get_default_cost,
)

__all__ = [
    "TimeframeConfig",
    "TIMEFRAME_CONFIGS",
    "DEFAULT_TIMEFRAME",
    "DEFAULT_SYMBOL",
    "get_config",
    "get_horizon_for_timeframe",
    "get_lookback_for_timeframe",
    "get_default_cost",
]
