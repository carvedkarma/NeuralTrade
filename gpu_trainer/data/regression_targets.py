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
                 horizon_periods: int = 48,  # 4 hours in 5-minute candles
                 lookback_periods: int = 96,  # 8 hours for volatility
                 costs: Optional[TradingCosts] = None):
        self.horizon_periods = horizon_periods
        self.lookback_periods = lookback_periods
        self.costs = costs or TradingCosts()
        
    def compute_forward_returns(self, prices: pd.Series) -> pd.Series:
        """
        Compute forward returns over horizon.
        
        Returns:
            Series of forward returns (future_close / current_close - 1)
        """
        future_prices = prices.shift(-self.horizon_periods)
        returns = (future_prices - prices) / prices
        return returns
    
    def compute_realized_volatility(self, prices: pd.Series) -> pd.Series:
        """
        Compute realized volatility (standard deviation of returns).
        
        Uses rolling window of lookback periods.
        """
        log_returns = np.log(prices / prices.shift(1))
        volatility = log_returns.rolling(window=self.lookback_periods).std()
        
        annualization = np.sqrt(288)  # 5-min candles per day
        volatility_annualized = volatility * annualization
        
        return volatility_annualized
    
    def compute_forward_volatility(self, prices: pd.Series) -> pd.Series:
        """
        Compute forward realized volatility (uncertainty of the prediction).
        
        This is the actual volatility that will occur during the holding period.
        """
        log_returns = np.log(prices / prices.shift(1))
        
        forward_vol = log_returns.shift(-self.horizon_periods).rolling(
            window=self.horizon_periods
        ).std()
        
        forward_vol = forward_vol.shift(self.horizon_periods)
        
        annualization = np.sqrt(288)
        
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
    
    def generate_multihead_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate targets specifically for multi-head model training.
        
        Returns:
            DataFrame with:
            - mu: Actual forward return (regression target)
            - sigma: Forward volatility (for uncertainty calibration)
            - class_label: 0=SHORT, 1=HOLD, 2=LONG (for classification head)
            - forward_return: The actual return to predict (same as mu, for quantile loss)
        
        Note: The model predicts quantiles of the return distribution.
        The target for quantile loss is the actual realized return.
        Pinball loss naturally learns the correct quantiles from individual returns.
        """
        prices = df["close"]
        
        # Actual forward return (the target we're predicting)
        mu = self.compute_forward_returns(prices)
        
        # Forward volatility (uncertainty target)
        current_vol = self.compute_realized_volatility(prices)
        sigma = self.compute_forward_volatility(prices)
        sigma = sigma.fillna(current_vol)
        
        # Classification labels derived from return direction and magnitude
        edge = self.compute_directional_edge(mu, sigma.clip(lower=0.001))
        
        # Generate class labels
        class_label = pd.Series(1, index=df.index)  # Default HOLD
        
        # Edge threshold for directional trades
        edge_threshold = 0.3  # Lower threshold for more training signal
        min_return = 0.002   # Minimum 0.2% move to be directional
        
        # LONG: positive return with sufficient edge
        long_mask = (mu > min_return) & (edge > edge_threshold)
        class_label[long_mask] = 2  # LONG
        
        # SHORT: negative return with sufficient edge
        short_mask = (mu < -min_return) & (edge > edge_threshold)
        class_label[short_mask] = 0  # SHORT
        
        targets = pd.DataFrame({
            "mu": mu,
            "sigma": sigma,
            "class_label": class_label,
            "forward_return": mu,  # Same as mu, explicit for quantile loss
            "edge": edge,
            "current_volatility": current_vol
        })
        
        logger.info(f"Generated multihead targets: "
                   f"LONG={long_mask.sum()}, SHORT={short_mask.sum()}, "
                   f"HOLD={(class_label == 1).sum()}")
        
        return targets
    
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
    horizon_periods: int = 48,
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


def generate_multihead_targets(df: pd.DataFrame, horizon_periods: int = 48) -> pd.DataFrame:
    """
    Standalone function to generate multi-head training targets.
    
    Args:
        df: DataFrame with OHLCV data (must have 'close' column)
        horizon_periods: Prediction horizon in candle periods (default 48 = 4h in 5m candles)
        
    Returns:
        DataFrame with:
        - mu: Actual forward return (regression target)
        - sigma: Forward volatility (for uncertainty calibration)
        - class_label: 0=SHORT, 1=HOLD, 2=LONG (for classification head)
        - forward_return: The actual return to predict (same as mu, for quantile loss)
    """
    generator = RegressionTargetGenerator(horizon_periods=horizon_periods)
    return generator.generate_multihead_targets(df)
