"""
Regression Training Targets for Institutional-Grade Trading

Instead of classification (LONG/SHORT/HOLD), we predict:
- μ (mu): Expected future return over horizon (4h default)
- σ (sigma): Uncertainty/volatility of the prediction
- P(move > cost): Probability that net move beats transaction costs
- Quantiles: p10, p50, p90 for asymmetric risk assessment

The trading signal is then computed as:
    edge = μ - cost
    confidence = edge / σ
    enter_trade = confidence > threshold
    
This approach naturally produces "strong signals only" because weak 
signals have low edge and high uncertainty.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradingCosts:
    """Transaction costs for edge calculation."""
    maker_fee: float = 0.0002  # 0.02% maker fee
    taker_fee: float = 0.0004  # 0.04% taker fee
    slippage_base: float = 0.0001  # Base slippage
    slippage_vol_mult: float = 0.5  # Slippage increases with volatility
    funding_per_8h: float = 0.0001  # Average funding rate
    
    def total_round_trip_cost(self, volatility: float = 0.01, 
                               is_taker: bool = True,
                               hold_hours: float = 4) -> float:
        """
        Calculate total round-trip transaction cost.
        
        Args:
            volatility: Current volatility (for slippage estimate)
            is_taker: Whether using taker orders
            hold_hours: Expected hold time in hours
            
        Returns:
            Total cost as a fraction (e.g., 0.001 = 0.1%)
        """
        fee = self.taker_fee if is_taker else self.maker_fee
        slippage = self.slippage_base + self.slippage_vol_mult * volatility
        
        funding_periods = hold_hours / 8
        funding_cost = abs(self.funding_per_8h) * funding_periods
        
        total = (fee * 2) + (slippage * 2) + funding_cost
        
        return total


class RegressionTargetGenerator:
    """
    Generates regression targets from price data.
    
    Targets:
    1. mu (μ): Expected return = (close_future - close_current) / close_current
    2. sigma (σ): Realized volatility over the horizon
    3. p_profitable: Probability that |move| > cost (from historical distribution)
    4. quantiles: p10, p50, p90 of return distribution
    
    The model learns to predict these directly, then we compute:
        edge = μ - cost
        confidence = edge / σ
    """
    
    def __init__(self, 
                 horizon_periods: int = 16,  # 4 hours in 15-minute candles (production default)
                 lookback_periods: int = 32,  # 8 hours = 2x horizon for volatility
                 costs: Optional[TradingCosts] = None):
        self.horizon_periods = horizon_periods
        self.lookback_periods = lookback_periods
        self.costs = costs or TradingCosts()
        
        logger.info(f"RegressionTargetGenerator: horizon={horizon_periods}, lookback={lookback_periods}")
        
    def compute_forward_returns(self, prices: pd.Series) -> pd.Series:
        """
        Compute forward returns over horizon.
        
        Returns:
            Series of forward returns (future_close / current_close - 1)
        """
        future_prices = prices.shift(-self.horizon_periods)
        returns = (future_prices - prices) / prices
        return returns
    
    def compute_realized_volatility(self, prices: pd.Series, bars_per_day: int = 96) -> pd.Series:
        """
        Compute realized volatility (standard deviation of returns).
        
        Uses rolling window of lookback periods.
        
        Args:
            bars_per_day: Number of bars per day for annualization (96 for 15m, 288 for 5m)
        """
        log_returns = np.log(prices / prices.shift(1))
        volatility = log_returns.rolling(window=self.lookback_periods).std()
        
        annualization = np.sqrt(bars_per_day)  # 96 for 15m, 288 for 5m
        volatility_annualized = volatility * annualization
        
        return volatility_annualized
    
    def compute_forward_volatility(self, prices: pd.Series, bars_per_day: int = 96) -> pd.Series:
        """
        Compute forward realized volatility (uncertainty of the prediction).
        
        This is the actual volatility that will occur during the holding period.
        
        Args:
            bars_per_day: Number of bars per day for annualization (96 for 15m, 288 for 5m)
        """
        log_returns = np.log(prices / prices.shift(1))
        
        forward_vol = log_returns.shift(-self.horizon_periods).rolling(
            window=self.horizon_periods
        ).std()
        
        forward_vol = forward_vol.shift(self.horizon_periods)
        
        # FIXED: Use correct annualization for 15m bars (was hardcoded to 288 for 5m)
        annualization = np.sqrt(bars_per_day)  # 96 for 15m, 288 for 5m
        
        return forward_vol * annualization
    
    def compute_max_favorable_excursion(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Maximum Favorable Excursion (MFE) - the best price during horizon.
        
        For long: max high during horizon relative to entry
        For short: min low during horizon relative to entry
        
        Returns the maximum potential profit during the trade.
        """
        mfe = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df) - self.horizon_periods):
            entry_close = df["close"].iloc[i]
            horizon_highs = df["high"].iloc[i+1:i+1+self.horizon_periods]
            horizon_lows = df["low"].iloc[i+1:i+1+self.horizon_periods]
            
            max_up = (horizon_highs.max() - entry_close) / entry_close
            max_down = (entry_close - horizon_lows.min()) / entry_close
            
            mfe.iloc[i] = max(max_up, max_down)
        
        return mfe
    
    def compute_max_adverse_excursion(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Maximum Adverse Excursion (MAE) - the worst drawdown during horizon.
        
        Returns the maximum potential loss (as positive number) during the trade.
        """
        mae = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df) - self.horizon_periods):
            entry_close = df["close"].iloc[i]
            horizon_lows = df["low"].iloc[i+1:i+1+self.horizon_periods]
            horizon_highs = df["high"].iloc[i+1:i+1+self.horizon_periods]
            
            max_drawdown_long = (entry_close - horizon_lows.min()) / entry_close
            max_drawdown_short = (horizon_highs.max() - entry_close) / entry_close
            
            mae.iloc[i] = min(max_drawdown_long, max_drawdown_short)
        
        return mae
    
    def compute_return_quantiles(self, prices: pd.Series,
                                  quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
                                  ) -> pd.DataFrame:
        """
        Compute rolling quantiles of forward returns.
        
        Uses expanding window to compute quantiles at each point.
        Now includes q10, q25, q50, q75, q90 for full distribution.
        """
        forward_returns = self.compute_forward_returns(prices)
        
        result = pd.DataFrame(index=prices.index)
        
        for q in quantiles:
            col_name = f"return_p{int(q*100)}"
            result[col_name] = forward_returns.expanding(min_periods=100).quantile(q)
        
        return result
    
    def generate_multihead_targets(
        self, 
        df: pd.DataFrame, 
        n_future_candles: int = 5,
        min_net_edge: float = 0.0,
        min_confidence: float = 0.40,
        use_volatility_cost: bool = False,
        fixed_cost: float = 0.0009,
        use_pure_directional: bool = False,
        directional_threshold: float = 0.0020,
        use_regime_labels: bool = False,
        trend_threshold: float = 0.0015,
        range_threshold: float = 0.0030
    ) -> pd.DataFrame:
        """
        Generate targets specifically for multi-head model training.
        
        HOLD Fix Stages:
        - Stage 1: min_confidence lowered from 0.7 to 0.40 (cost-aware mode)
        - Stage 2: use_pure_directional=True bypasses gates, uses simple return threshold
        - Stage 3: use_regime_labels=True uses ADX-based adaptive thresholds
        
        Args:
            df: DataFrame with OHLCV data
            n_future_candles: Number of future candles to predict
            min_net_edge: Minimum net edge after costs for trade signals (default 0.0 for debugging)
            min_confidence: Minimum mu/sigma ratio for trade signals (default 0.40, lowered from 0.7)
            use_volatility_cost: If True, use volatility-based cost calculation. If False, use fixed_cost
            fixed_cost: Fixed round-trip trading cost when not using volatility-based (default 0.09%)
            use_pure_directional: If True, use simple return threshold instead of cost-aware gating
            directional_threshold: Return threshold for pure directional mode (default 0.20%)
        
        Returns:
            DataFrame with:
            - mu: Actual forward return (regression target)
            - sigma: Forward volatility (for uncertainty calibration)
            - class_label: 0=SHORT, 1=HOLD, 2=LONG (for classification head)
            - forward_return: The actual return to predict (same as mu, for quantile loss)
            - entry_offset: Optimal entry offset based on volatility
            - sl_distance: Optimal stop loss distance based on ATR
            - tp_distance: Optimal take profit distance based on ATR
            - candle_delta_close_N: Future close deltas for N steps
            - candle_delta_high_N: Future high deltas for N steps  
            - candle_delta_low_N: Future low deltas for N steps
        
        Note: The model predicts quantiles of the return distribution.
        The target for quantile loss is the actual realized return.
        Pinball loss naturally learns the correct quantiles from individual returns.
        """
        prices = df["close"]
        highs = df["high"]
        lows = df["low"]
        
        # Actual forward return (the target we're predicting)
        mu = self.compute_forward_returns(prices)
        
        # Forward volatility (uncertainty target)
        current_vol = self.compute_realized_volatility(prices)
        sigma = self.compute_forward_volatility(prices)
        sigma = sigma.fillna(current_vol)
        
        # ============================================================
        # PHASE 1a: COST-AWARE CLASS LABELS
        # ============================================================
        # Derive class labels from NET EDGE after trading costs
        # edge_net = |mu| - cost(volatility, hold_hours)
        # Trade only if edge_net > min_edge AND mu/sigma > confidence_min
        # This eliminates garbage signals that don't beat costs
        
        # Compute trading cost
        hold_hours = self.horizon_periods * 0.25  # 15-minute bars -> hours
        trading_costs = pd.Series(index=df.index, dtype=float)
        
        if use_volatility_cost:
            # Volatility-based cost calculation (original behavior)
            for i in range(len(df)):
                vol = current_vol.iloc[i] if not pd.isna(current_vol.iloc[i]) else 0.01
                trading_costs.iloc[i] = self.costs.total_round_trip_cost(
                    volatility=vol, 
                    is_taker=True, 
                    hold_hours=hold_hours
                )
        else:
            # Fixed cost mode - uses the exact cost from config
            trading_costs[:] = fixed_cost
        
        # Net edge = absolute expected return - trading costs
        net_edge = mu.abs() - trading_costs
        
        # Confidence ratio: mu/sigma (signal-to-noise)
        sigma_safe = sigma.clip(lower=0.001)
        confidence_ratio = mu.abs() / sigma_safe
        
        # ============================================================
        # LABEL GENERATION - Two modes available:
        # 1. Cost-aware mode (default): Uses net_edge & confidence gates
        # 2. Pure directional mode: Uses simple return threshold
        # ============================================================
        
        valid_mu = mu.dropna()
        valid_net_edge = net_edge.dropna()
        valid_conf = confidence_ratio.dropna()
        total_samples = len(valid_mu)
        
        # Handle conflicting flags - regime labels takes precedence
        if use_regime_labels and use_pure_directional:
            logger.warning("Both use_regime_labels and use_pure_directional are True!")
            logger.warning("Regime-based labeling takes precedence over pure directional")
        
        logger.info("=" * 70)
        if use_regime_labels:
            # ============================================================
            # STAGE 3: REGIME-BASED LABELS (ADX-adaptive thresholds)
            # ============================================================
            logger.info("LABEL GENERATION: STAGE 3 - REGIME-BASED MODE")
            logger.info("=" * 70)
            logger.info(f"Config: horizon={self.horizon_periods} bars")
            logger.info(f"Thresholds: trend={trend_threshold:.4%}, range={range_threshold:.4%}")
            logger.info(f"Total samples: {total_samples:,}")
            
            # Compute ADX for regime detection
            adx = self._compute_adx(df, period=14)
            
            # Regime classification: ADX > 25 = trending, ADX < 20 = ranging
            is_trending = adx > 25
            is_ranging = adx < 20
            is_transition = ~is_trending & ~is_ranging
            
            pct_trend = is_trending.sum() / max(1, len(adx)) * 100
            pct_range = is_ranging.sum() / max(1, len(adx)) * 100
            pct_trans = is_transition.sum() / max(1, len(adx)) * 100
            logger.info(f"Regime distribution: TREND={pct_trend:.1f}%, RANGE={pct_range:.1f}%, TRANSITION={pct_trans:.1f}%")
            
            # Adaptive threshold per regime:
            # - Trending: lower threshold (0.15%) - trade momentum
            # - Ranging: higher threshold (0.30%) - only take strong mean reversion
            # - Transition: middle threshold (0.20%)
            class_label = pd.Series(1, index=df.index)  # Default HOLD
            
            # In trending regime: lower bar, trade more
            trend_long = is_trending & (mu > trend_threshold)
            trend_short = is_trending & (mu < -trend_threshold)
            
            # In ranging regime: higher bar, trade less
            range_long = is_ranging & (mu > range_threshold)
            range_short = is_ranging & (mu < -range_threshold)
            
            # In transition: use middle threshold
            trans_threshold = (trend_threshold + range_threshold) / 2
            trans_long = is_transition & (mu > trans_threshold)
            trans_short = is_transition & (mu < -trans_threshold)
            
            # Combine masks for consistency with other modes
            long_mask = trend_long | range_long | trans_long
            short_mask = trend_short | range_short | trans_short
            
            # Apply labels
            class_label[long_mask] = 2  # LONG
            class_label[short_mask] = 0  # SHORT
            
            # Log per-regime stats
            n_trend_trades = (trend_long.sum() + trend_short.sum())
            n_range_trades = (range_long.sum() + range_short.sum())
            n_trans_trades = (trans_long.sum() + trans_short.sum())
            logger.info(f"Trades by regime: TREND={n_trend_trades:,}, RANGE={n_range_trades:,}, TRANSITION={n_trans_trades:,}")
            logger.info("-" * 70)
            logger.info("NOTE: Regime-based thresholds adapt to market conditions")
            logger.info("NOTE: Lower threshold in trends (momentum), higher in range (mean reversion)")
            logger.info("-" * 70)
            
        elif use_pure_directional:
            # ============================================================
            # STAGE 2: PURE DIRECTIONAL LABELS (simple return threshold)
            # ============================================================
            logger.info("LABEL GENERATION: STAGE 2 - PURE DIRECTIONAL MODE")
            logger.info("=" * 70)
            logger.info(f"Config: horizon={self.horizon_periods} bars, directional_threshold={directional_threshold:.4%}")
            logger.info(f"Total samples: {total_samples:,}")
            logger.info(f"Stats: mean(|mu|)={valid_mu.abs().mean():.4%}, mean(sigma)={sigma_safe.dropna().mean():.4%}")
            
            # Pure directional: trade signal based only on return threshold
            # LONG if return > threshold, SHORT if return < -threshold, else HOLD
            class_label = pd.Series(1, index=df.index)  # Default HOLD
            
            long_mask = mu > directional_threshold
            short_mask = mu < -directional_threshold
            
            class_label[long_mask] = 2  # LONG
            class_label[short_mask] = 0  # SHORT
            
            pct_above_threshold = (mu.abs() > directional_threshold).sum() / max(1, total_samples) * 100
            logger.info(f"Trade density: {pct_above_threshold:.1f}% samples exceed threshold")
            logger.info("-" * 70)
            logger.info("NOTE: Gates (net_edge/confidence) are BYPASSED in pure directional mode")
            logger.info("NOTE: Gates still apply during LIVE EXECUTION - this is training only")
            logger.info("-" * 70)
        else:
            # ============================================================
            # STAGE 1: COST-AWARE LABELS (with relaxed confidence)
            # ============================================================
            logger.info("LABEL GENERATION: STAGE 1 - COST-AWARE MODE")
            logger.info("=" * 70)
            
            pct_positive_edge = (valid_net_edge > min_net_edge).sum() / max(1, total_samples) * 100
            pct_high_conf = (valid_conf > min_confidence).sum() / max(1, total_samples) * 100
            pct_both_gates = ((valid_net_edge > min_net_edge) & (valid_conf > min_confidence)).sum() / max(1, total_samples) * 100
            
            # Calculate what OLD threshold (0.7) would have produced for comparison
            old_threshold = 0.7
            pct_high_conf_old = (valid_conf > old_threshold).sum() / max(1, total_samples) * 100
            pct_both_gates_old = ((valid_net_edge > min_net_edge) & (valid_conf > old_threshold)).sum() / max(1, total_samples) * 100
            
            logger.info(f"Config: horizon={self.horizon_periods} bars, cost_mode={'volatility' if use_volatility_cost else 'fixed'}, "
                       f"avg_cost={trading_costs.mean():.4%}")
            logger.info(f"Thresholds: min_net_edge={min_net_edge:.4%}, min_confidence={min_confidence:.2f}")
            logger.info(f"Total samples: {total_samples:,}")
            logger.info(f"Stats: mean(|mu|)={valid_mu.abs().mean():.4%}, mean(sigma)={sigma_safe.dropna().mean():.4%}, "
                       f"mean(|mu|/sigma)={valid_conf.mean():.2f}")
            logger.info(f"Cost: mean(cost)={trading_costs.dropna().mean():.4%}, mean(net_edge)={valid_net_edge.mean():.4%}")
            
            # Before/After comparison
            logger.info("-" * 70)
            logger.info("BEFORE/AFTER COMPARISON (min_confidence threshold):")
            logger.info(f"  OLD (0.70): confidence_gate={pct_high_conf_old:.1f}%, trade_density={pct_both_gates_old:.1f}%")
            logger.info(f"  NEW ({min_confidence:.2f}): confidence_gate={pct_high_conf:.1f}%, trade_density={pct_both_gates:.1f}%")
            improvement = pct_both_gates - pct_both_gates_old
            logger.info(f"  IMPROVEMENT: +{improvement:.1f}% more samples will be labeled as trades")
            logger.info("-" * 70)
            
            if pct_both_gates < 5.0:
                logger.warning(f"!!! LOW TRADE DENSITY: Only {pct_both_gates:.2f}% samples pass both gates !!!")
                logger.warning(f"!!! Try: use_pure_directional=True with directional_threshold=0.0020 !!!")
            
            # Generate class labels - only trade when net_edge AND confidence are sufficient
            class_label = pd.Series(1, index=df.index)  # Default HOLD
            
            # LONG: positive return with sufficient NET edge and confidence
            long_mask = (mu > 0) & (net_edge > min_net_edge) & (confidence_ratio > min_confidence)
            class_label[long_mask] = 2  # LONG
            
            # SHORT: negative return with sufficient NET edge and confidence
            short_mask = (mu < 0) & (net_edge > min_net_edge) & (confidence_ratio > min_confidence)
            class_label[short_mask] = 0  # SHORT
        
        # Log class distribution with target range validation
        n_long = (class_label == 2).sum()
        n_short = (class_label == 0).sum()
        n_hold = (class_label == 1).sum()
        total = n_long + n_short + n_hold
        
        pct_long = n_long/max(1,total)*100
        pct_short = n_short/max(1,total)*100
        pct_hold = n_hold/max(1,total)*100
        
        logger.info(f"CLASS DISTRIBUTION RESULT:")
        logger.info(f"  SHORT: {n_short:,} ({pct_short:.1f}%) [target: 15-25%]")
        logger.info(f"  HOLD:  {n_hold:,} ({pct_hold:.1f}%) [target: 50-70%]")
        logger.info(f"  LONG:  {n_long:,} ({pct_long:.1f}%) [target: 15-25%]")
        
        # Validate against targets
        hold_ok = 50 <= pct_hold <= 70
        long_ok = 15 <= pct_long <= 25
        short_ok = 15 <= pct_short <= 25
        
        if hold_ok and long_ok and short_ok:
            logger.info("  STATUS: All targets MET!")
        else:
            issues = []
            if not hold_ok:
                issues.append(f"HOLD ({pct_hold:.1f}%)")
            if not long_ok:
                issues.append(f"LONG ({pct_long:.1f}%)")
            if not short_ok:
                issues.append(f"SHORT ({pct_short:.1f}%)")
            logger.warning(f"  STATUS: Targets NOT MET: {', '.join(issues)}")
            if pct_hold > 80 and not use_pure_directional:
                logger.warning("  RECOMMENDATION: Try use_pure_directional=True")
        logger.info("=" * 70)
        
        # Legacy edge for backward compatibility
        edge = self.compute_directional_edge(mu, sigma_safe)
        
        # === TRADING HEAD TARGETS ===
        # Entry offset: MFE-based optimal limit entry (learned from price path)
        atr = self._compute_atr(df, period=14)
        entry_offset, sl_distance, tp_distance = self._compute_mfe_trading_targets(
            df, prices, class_label, atr, n_future_candles
        )
        
        # ============================================================
        # PHASE 2: CONSTRAINED CANDLE PARAMETERIZATION
        # ============================================================
        # Instead of predicting raw close/high/low which can be inconsistent,
        # we predict: delta_close, log_range (always positive), skew in [-1, 1]
        # Then reconstruct: range = exp(log_range)
        #                   high = close + range * (0.5 + 0.5 * skew)
        #                   low = close - range * (0.5 - 0.5 * skew)
        # This guarantees: high >= low and valid candles
        
        candle_targets = {}
        for i in range(1, n_future_candles + 1):
            # Delta close: (future_close - current_close) / current_close
            future_close = prices.shift(-i)
            future_high = highs.shift(-i)
            future_low = lows.shift(-i)
            
            delta_close = (future_close - prices) / prices
            candle_targets[f"candle_delta_close_{i}"] = delta_close
            
            # Constrained parameterization:
            # Range = (high - low) / close (always positive)
            candle_range = (future_high - future_low) / prices
            candle_range = candle_range.clip(lower=0.0001)  # Prevent zero/negative
            
            # Log range for numerical stability (network predicts unbounded, we exp it)
            log_range = np.log(candle_range)
            candle_targets[f"candle_log_range_{i}"] = log_range
            
            # Skew in [-1, 1]: where is close relative to high/low
            # skew = (close - midpoint) / (range/2)
            # = 2 * (close - (high + low)/2) / (high - low)
            # = (close - low) / (high - low) * 2 - 1  (when close is at high, skew = 1)
            # Use future close position within high-low range
            range_safe = (future_high - future_low).clip(lower=prices * 0.0001)
            skew = 2 * (future_close - future_low) / range_safe - 1
            skew = skew.clip(lower=-1, upper=1)
            candle_targets[f"candle_skew_{i}"] = skew
            
            # Keep legacy targets for backward compatibility (but use constrained for new training)
            candle_targets[f"candle_delta_high_{i}"] = (future_high - prices) / prices
            candle_targets[f"candle_delta_low_{i}"] = (future_low - prices) / prices
        
        # ============================================================
        # PHASE 1b: LOG_SIGMA FOR GAUSSIAN NLL
        # ============================================================
        # For proper Gaussian NLL loss: loss = (y-mu)^2/(2*sigma^2) + log(sigma)
        # We predict log_sigma (unbounded) and exp it to get sigma (always positive)
        log_sigma = np.log(sigma_safe)
        
        # ============================================================
        # FLOW FORECAST: VOL_STATE AND ACCELERATION TARGETS
        # ============================================================
        # Vol_state: Volatility regime prediction (expansion/neutral/contraction)
        # forward_vol / current_vol ratio determines regime:
        #   < 0.9 = contraction (0)
        #   0.9 - 1.1 = neutral (1)
        #   > 1.1 = expansion (2)
        
        # Forward volatility over horizon
        forward_vol = self.compute_forward_volatility(prices)
        forward_vol_safe = forward_vol.fillna(current_vol)
        current_vol_safe = current_vol.fillna(0.01)
        
        vol_ratio = forward_vol_safe / current_vol_safe.clip(lower=0.001)
        vol_state = pd.Series(1, index=df.index, dtype=int)  # Default neutral
        vol_state[vol_ratio < 0.9] = 0   # Contraction
        vol_state[vol_ratio > 1.1] = 2   # Expansion
        
        # Log vol_state distribution
        n_contraction = (vol_state == 0).sum()
        n_neutral = (vol_state == 1).sum()
        n_expansion = (vol_state == 2).sum()
        total = max(1, n_contraction + n_neutral + n_expansion)
        logger.info(f"Vol_state distribution: CONTRACTION={n_contraction} ({n_contraction/total*100:.1f}%), "
                   f"NEUTRAL={n_neutral} ({n_neutral/total*100:.1f}%), "
                   f"EXPANSION={n_expansion} ({n_expansion/total*100:.1f}%)")
        
        # Acceleration: momentum change = momentum_forward - momentum_now
        # Momentum = 4-bar return (for 15m, this is 1 hour momentum)
        momentum_window = min(4, self.horizon_periods // 4) if self.horizon_periods >= 4 else 1
        momentum_now = (prices / prices.shift(momentum_window) - 1).fillna(0)
        
        # Forward momentum: return from horizon-4 to horizon
        forward_prices = prices.shift(-self.horizon_periods)
        forward_prices_back = prices.shift(-(self.horizon_periods - momentum_window))
        momentum_forward = (forward_prices / forward_prices_back - 1).fillna(0)
        
        # Acceleration = momentum change over horizon
        acceleration = momentum_forward - momentum_now
        
        logger.info(f"Acceleration stats: mean={acceleration.mean():.6f}, std={acceleration.std():.6f}, "
                   f"min={acceleration.min():.6f}, max={acceleration.max():.6f}")
        
        targets = pd.DataFrame({
            "mu": mu,
            "sigma": sigma,
            "log_sigma": log_sigma,  # For Gaussian NLL loss
            "class_label": class_label,
            "forward_return": mu,  # Same as mu, explicit for quantile loss
            "edge": edge,
            "net_edge": net_edge,  # Cost-aware edge
            "trading_cost": trading_costs,  # For debugging
            "confidence_ratio": confidence_ratio,  # mu/sigma
            "current_volatility": current_vol,
            "entry_offset": entry_offset,
            "sl_distance": sl_distance,
            "tp_distance": tp_distance,
            "vol_state": vol_state,  # Flow forecast: 0=contraction, 1=neutral, 2=expansion
            "acceleration": acceleration,  # Flow forecast: momentum change over horizon
            "vol_ratio": vol_ratio,  # For debugging
            **candle_targets
        })
        
        # Log cost-aware labeling stats
        avg_cost = trading_costs.mean()
        logger.info(f"Generated multihead targets (cost-aware): "
                   f"LONG={long_mask.sum()}, SHORT={short_mask.sum()}, "
                   f"HOLD={(class_label == 1).sum()}, "
                   f"avg_trading_cost={avg_cost:.4%}, "
                   f"candle_steps={n_future_candles}")
        
        return targets
    
    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range (ATR)."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr.fillna(true_range)
    
    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average Directional Index (ADX) for regime detection.
        
        ADX values interpretation:
        - ADX > 25: Strong trend (trending regime)
        - ADX < 20: Weak trend (ranging regime)
        - 20 <= ADX <= 25: Transition zone
        
        Returns:
            pd.Series: ADX values (0-100 scale)
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        # +DM and -DM
        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)
        
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]
        
        # Smoothed values (Wilder's smoothing)
        atr = true_range.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr.clip(lower=1e-10))
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr.clip(lower=1e-10))
        
        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx.fillna(20)  # Default to transition zone
    
    def _compute_mfe_trading_targets(
        self, 
        df: pd.DataFrame, 
        prices: pd.Series,
        class_label: pd.Series,
        atr: pd.Series,
        n_future_candles: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute MFE-based (Maximum Favorable Excursion) trading targets.
        
        MFE Logic:
        - For LONG: entry_offset = how far below current price did price dip 
          before moving up (optimal limit buy placement)
        - For SHORT: entry_offset = how far above current price did price spike
          before moving down (optimal limit sell placement)
        - For HOLD: entry_offset = 0 (no trade)
        
        SL/TP are derived from MAE (Maximum Adverse Excursion) and MFE:
        - SL = MAE (maximum adverse move during the trade)
        - TP = MFE (maximum favorable move during the trade)
        
        All values are normalized by ATR for cross-volatility learning.
        
        Returns:
            entry_offset: Optimal entry as percentage of price (negative = buy lower)
            sl_distance: Stop loss distance as percentage
            tp_distance: Take profit distance as percentage
        """
        highs = df["high"]
        lows = df["low"]
        
        # Initialize with zeros
        entry_offset = pd.Series(0.0, index=df.index)
        sl_distance = pd.Series(0.0, index=df.index)
        tp_distance = pd.Series(0.0, index=df.index)
        
        # Horizon for MFE/MAE calculation (same as forward return horizon)
        horizon = self.horizon_periods
        
        for i in range(len(df) - horizon):
            current_price = prices.iloc[i]
            label = class_label.iloc[i]
            current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else current_price * 0.01
            
            # Get future price path
            future_highs = highs.iloc[i+1:i+horizon+1]
            future_lows = lows.iloc[i+1:i+horizon+1]
            future_closes = prices.iloc[i+1:i+horizon+1]
            
            if len(future_highs) == 0:
                continue
            
            # MFE/MAE from the current entry point
            max_high = future_highs.max()
            min_low = future_lows.min()
            
            # LONG trade analysis
            if label == 2:  # LONG
                # Best entry = lowest price in first few candles (optimal limit buy)
                # We look at the first 1/4 of horizon for entry opportunity
                entry_window = max(1, horizon // 4)
                best_entry_price = future_lows.iloc[:entry_window].min()
                
                # Entry offset: how much lower than current could we have bought?
                # Negative means better (lower) entry
                entry_off = (best_entry_price - current_price) / current_price
                entry_offset.iloc[i] = np.clip(entry_off, -0.05, 0.0)  # Max 5% better entry
                
                # MAE (Maximum Adverse Excursion) = worst drawdown during trade
                mae = (min_low - current_price) / current_price
                sl_distance.iloc[i] = abs(mae) + (current_atr / current_price) * 0.5
                
                # MFE (Maximum Favorable Excursion) = best profit potential
                mfe = (max_high - current_price) / current_price
                tp_distance.iloc[i] = max(mfe, current_atr / current_price * 2)
                
            # SHORT trade analysis
            elif label == 0:  # SHORT
                # Best entry = highest price in first few candles (optimal limit sell)
                entry_window = max(1, horizon // 4)
                best_entry_price = future_highs.iloc[:entry_window].max()
                
                # Entry offset: how much higher than current could we have sold?
                # Positive means better (higher) entry for short
                entry_off = (best_entry_price - current_price) / current_price
                entry_offset.iloc[i] = np.clip(entry_off, 0.0, 0.05)  # Max 5% better entry
                
                # MAE for short = worst spike up during trade
                mae = (max_high - current_price) / current_price
                sl_distance.iloc[i] = abs(mae) + (current_atr / current_price) * 0.5
                
                # MFE for short = best drop potential
                mfe = (current_price - min_low) / current_price
                tp_distance.iloc[i] = max(mfe, current_atr / current_price * 2)
                
            else:  # HOLD
                # For HOLD, use ATR-based defaults (fallback)
                entry_offset.iloc[i] = 0.0
                sl_distance.iloc[i] = (current_atr / current_price) * 1.5
                tp_distance.iloc[i] = (current_atr / current_price) * 3.0
        
        # Clip to reasonable ranges
        sl_distance = sl_distance.clip(lower=0.003, upper=0.05)  # 0.3% to 5%
        tp_distance = tp_distance.clip(lower=0.005, upper=0.10)  # 0.5% to 10%
        
        # Log statistics
        long_mask = class_label == 2
        short_mask = class_label == 0
        logger.info(f"MFE Trading Targets - "
                   f"LONG entry_offset mean: {entry_offset[long_mask].mean():.4f}, "
                   f"SHORT entry_offset mean: {entry_offset[short_mask].mean():.4f}, "
                   f"SL mean: {sl_distance.mean():.4f}, TP mean: {tp_distance.mean():.4f}")
        
        return entry_offset, sl_distance, tp_distance
    
    def compute_probability_profitable(self, 
                                        prices: pd.Series,
                                        volatility: pd.Series) -> pd.Series:
        """
        Compute probability that absolute move exceeds transaction costs.
        
        P(|return| > cost) estimated from historical distribution.
        """
        forward_returns = self.compute_forward_returns(prices)
        
        costs = pd.Series(index=prices.index, dtype=float)
        for i in range(len(volatility)):
            vol = volatility.iloc[i] if not pd.isna(volatility.iloc[i]) else 0.01
            costs.iloc[i] = self.costs.total_round_trip_cost(vol)
        
        profitable = (np.abs(forward_returns) > costs).astype(float)
        
        p_profitable = profitable.rolling(window=500, min_periods=50).mean()
        
        return p_profitable
    
    def compute_directional_edge(self, 
                                  mu: pd.Series, 
                                  volatility: pd.Series) -> pd.Series:
        """
        Compute directional edge = (|μ| - cost) / σ.
        
        Positive edge means expected profit after costs.
        """
        costs = pd.Series(index=mu.index, dtype=float)
        for i in range(len(volatility)):
            vol = volatility.iloc[i] if not pd.isna(volatility.iloc[i]) else 0.01
            costs.iloc[i] = self.costs.total_round_trip_cost(vol)
        
        edge = (np.abs(mu) - costs) / volatility.clip(lower=0.001)
        
        return edge
    
    def generate_all_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all regression targets from candle DataFrame.
        
        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume
            
        Returns:
            DataFrame with target columns:
            - mu: Expected forward return
            - sigma: Forward volatility (uncertainty)
            - p_profitable: Probability of profitable trade
            - return_p10, return_p50, return_p90: Return quantiles
            - mfe: Maximum favorable excursion
            - mae: Maximum adverse excursion
            - edge: (|mu| - cost) / sigma
            - optimal_direction: 1 for long, -1 for short, 0 for hold
        """
        prices = df["close"]
        
        logger.info(f"Generating targets: horizon={self.horizon_periods}, "
                   f"lookback={self.lookback_periods}")
        
        mu = self.compute_forward_returns(prices)
        logger.info(f"  μ (forward returns): {mu.notna().sum()} valid")
        
        current_vol = self.compute_realized_volatility(prices)
        sigma = self.compute_forward_volatility(prices)
        sigma = sigma.fillna(current_vol)  # Use current vol as fallback
        logger.info(f"  σ (forward volatility): {sigma.notna().sum()} valid")
        
        p_profitable = self.compute_probability_profitable(prices, current_vol)
        logger.info(f"  P(profitable): {p_profitable.notna().sum()} valid")
        
        quantiles = self.compute_return_quantiles(prices)
        logger.info(f"  Quantiles computed")
        
        mfe = self.compute_max_favorable_excursion(df)
        mae = self.compute_max_adverse_excursion(df)
        logger.info(f"  MFE/MAE computed")
        
        edge = self.compute_directional_edge(mu, sigma)
        logger.info(f"  Edge computed")
        
        optimal_direction = pd.Series(0, index=df.index)
        edge_threshold = 0.5  # Require edge > 0.5 sigma
        optimal_direction[mu > 0] = 1   # Long
        optimal_direction[mu < 0] = -1  # Short
        optimal_direction[np.abs(edge) < edge_threshold] = 0  # No trade
        
        targets = pd.DataFrame({
            "mu": mu,
            "sigma": sigma,
            "p_profitable": p_profitable,
            "return_p10": quantiles["return_p10"],
            "return_p50": quantiles["return_p50"],
            "return_p90": quantiles["return_p90"],
            "mfe": mfe,
            "mae": mae,
            "edge": edge,
            "optimal_direction": optimal_direction,
            "current_volatility": current_vol
        })
        
        valid_count = targets.dropna().shape[0]
        total_count = len(targets)
        logger.info(f"Generated {valid_count}/{total_count} valid target rows")
        
        return targets


class MultiHorizonTargetGenerator:
    """
    Generate targets for multiple time horizons.
    
    This allows the model to predict returns at different horizons:
    - 1h (12 periods of 5m)
    - 4h (48 periods of 5m)
    - 12h (144 periods of 5m)
    - 24h (288 periods of 5m)
    
    Useful for regime-adaptive trading and timeframe agreement analysis.
    """
    
    def __init__(self, 
                 horizons: Dict[str, int] = None,
                 costs: Optional[TradingCosts] = None):
        self.horizons = horizons or {
            "1h": 12,
            "4h": 48,
            "12h": 144,
            "24h": 288
        }
        self.costs = costs or TradingCosts()
        
    def generate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate targets for all horizons.
        
        Returns DataFrame with columns like:
        - mu_1h, sigma_1h, edge_1h
        - mu_4h, sigma_4h, edge_4h
        - ...
        """
        all_targets = pd.DataFrame(index=df.index)
        
        for horizon_name, horizon_periods in self.horizons.items():
            logger.info(f"Generating {horizon_name} targets...")
            
            generator = RegressionTargetGenerator(
                horizon_periods=horizon_periods,
                lookback_periods=horizon_periods * 2,
                costs=self.costs
            )
            
            targets = generator.generate_all_targets(df)
            
            for col in ["mu", "sigma", "edge", "p_profitable", "optimal_direction"]:
                all_targets[f"{col}_{horizon_name}"] = targets[col]
        
        horizon_agreement = pd.Series(0, index=df.index)
        for _, row in all_targets.iterrows():
            directions = [row.get(f"optimal_direction_{h}", 0) for h in self.horizons.keys()]
            if all(d > 0 for d in directions if d != 0):
                horizon_agreement[row.name] = 1
            elif all(d < 0 for d in directions if d != 0):
                horizon_agreement[row.name] = -1
        
        all_targets["horizon_agreement"] = horizon_agreement
        
        return all_targets


def create_regression_dataset(
    candle_df: pd.DataFrame,
    features_df: pd.DataFrame,
    horizon_periods: int = 16,  # Default 16 bars = 4h at 15m timeframe (was 48 for 5m)
    min_edge_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training dataset for regression model.
    
    Args:
        candle_df: OHLCV candle data
        features_df: Computed feature matrix
        horizon_periods: Prediction horizon in candle periods
        min_edge_threshold: Minimum edge for trade labels
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y_regression: Regression targets [mu, sigma] (n_samples, 2)
        y_direction: Direction labels [-1, 0, 1] (n_samples,)
    """
    generator = RegressionTargetGenerator(horizon_periods=horizon_periods)
    targets = generator.generate_all_targets(candle_df)
    
    common_idx = features_df.index.intersection(targets.index)
    features_df = features_df.loc[common_idx]
    targets = targets.loc[common_idx]
    
    mask = targets[["mu", "sigma"]].notna().all(axis=1)
    features_df = features_df[mask]
    targets = targets[mask]
    
    X = features_df.values.astype(np.float32)
    
    y_regression = targets[["mu", "sigma"]].values.astype(np.float32)
    
    y_direction = targets["optimal_direction"].values.astype(np.float32)
    
    logger.info(f"Created regression dataset: X={X.shape}, y_reg={y_regression.shape}, y_dir={y_direction.shape}")
    
    return X, y_regression, y_direction


def generate_multihead_targets(
    df: pd.DataFrame, 
    horizon_periods: int = 16,  # Default 16 bars = 4h at 15m timeframe
    n_future_candles: int = 5,
    min_net_edge: float = 0.0,  # Default 0 for debugging (no edge filter)
    min_confidence: float = 0.40,  # Default 0.40 (lowered from 0.7 to reduce HOLD-heavy labels)
    use_volatility_cost: bool = False,  # Default to fixed cost mode
    fixed_cost: float = 0.0009,  # Default 0.09% round-trip (taker/taker)
    use_pure_directional: bool = False,  # Stage 2: bypass gates, use simple threshold
    directional_threshold: float = 0.0020,  # 0.20% return threshold for Stage 2
    use_regime_labels: bool = False,  # Stage 3: ADX-based adaptive thresholds
    trend_threshold: float = 0.0015,  # 0.15% threshold in trending regime
    range_threshold: float = 0.0030  # 0.30% threshold in ranging regime
) -> pd.DataFrame:
    """
    Standalone function to generate multi-head training targets.
    
    HOLD Fix Stages:
    - Stage 1: min_confidence lowered from 0.7 to 0.40 (cost-aware mode)
    - Stage 2: use_pure_directional=True bypasses gates, uses simple return threshold
    - Stage 3: use_regime_labels=True uses ADX-based adaptive thresholds
    
    Target distribution: HOLD 50-70%, LONG 15-25%, SHORT 15-25%.
    
    Args:
        df: DataFrame with OHLCV data (must have 'close' column)
        horizon_periods: Prediction horizon in candle periods (default 16 = 4h in 15m candles)
        n_future_candles: Number of future candles to predict (default 5)
        min_net_edge: Minimum net edge after costs for trade signals (default 0.0 for debugging)
        min_confidence: Minimum mu/sigma ratio for trade signals (default 0.40, lowered from 0.7)
        use_volatility_cost: If True, use volatility-based cost. If False, use fixed_cost
        fixed_cost: Fixed round-trip trading cost (default 0.09%)
        use_pure_directional: If True, use simple return threshold instead of cost-aware gating
        directional_threshold: Return threshold for pure directional mode (default 0.20%)
        use_regime_labels: If True, use ADX-based adaptive thresholds (Stage 3)
        trend_threshold: Return threshold for trending regime (default 0.15%)
        range_threshold: Return threshold for ranging regime (default 0.30%)
        
    Returns:
        DataFrame with:
        - mu: Actual forward return (regression target)
        - sigma: Forward volatility (for uncertainty calibration)
        - class_label: 0=SHORT, 1=HOLD, 2=LONG (for classification head)
        - forward_return: The actual return to predict (same as mu, for quantile loss)
        - entry_offset, sl_distance, tp_distance: Trading head targets
        - candle_delta_*: Future candle prediction targets
    """
    generator = RegressionTargetGenerator(horizon_periods=horizon_periods)
    return generator.generate_multihead_targets(
        df, 
        n_future_candles=n_future_candles,
        min_net_edge=min_net_edge,
        min_confidence=min_confidence,
        use_volatility_cost=use_volatility_cost,
        fixed_cost=fixed_cost,
        use_pure_directional=use_pure_directional,
        directional_threshold=directional_threshold,
        use_regime_labels=use_regime_labels,
        trend_threshold=trend_threshold,
        range_threshold=range_threshold
    )
