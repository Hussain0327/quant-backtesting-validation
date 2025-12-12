"""
Pairs Trading Strategy (Statistical Arbitrage)

This strategy exploits mean reversion in the spread between two cointegrated assets.
Unlike simple momentum or mean reversion on a single asset, pairs trading:
1. Is market-neutral (hedged against market moves)
2. Based on statistical relationship (cointegration) not just correlation
3. Has a theoretical foundation in arbitrage pricing theory

The key insight: Two stocks in the same sector often move together. When they
temporarily diverge, we bet on convergence by going long the underperformer
and short the outperformer.
"""

import numpy as np
import pandas as pd
from .base import Strategy
from typing import Tuple, Optional


def calculate_spread(prices_a: pd.Series, prices_b: pd.Series, hedge_ratio: float = None) -> Tuple[pd.Series, float]:
    """
    Calculate the spread between two price series.

    The spread = Price_A - hedge_ratio * Price_B

    If hedge_ratio is None, estimate it using OLS regression.
    """
    if hedge_ratio is None:
        # Estimate hedge ratio via OLS: prices_a = alpha + beta * prices_b
        X = prices_b.values.reshape(-1, 1)
        y = prices_a.values
        # Simple OLS: beta = cov(X,y) / var(X)
        cov_xy = np.cov(prices_b, prices_a)[0, 1]
        var_x = np.var(prices_b)
        hedge_ratio = cov_xy / var_x if var_x != 0 else 1.0

    spread = prices_a - hedge_ratio * prices_b
    return spread, hedge_ratio


def calculate_zscore(spread: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Calculate rolling z-score of spread.

    Z-score normalizes the spread to standard deviations from mean,
    making it comparable across different pairs.
    """
    rolling_mean = spread.rolling(window=lookback).mean()
    rolling_std = spread.rolling(window=lookback).std()

    zscore = (spread - rolling_mean) / rolling_std
    return zscore.replace([np.inf, -np.inf], np.nan)


def test_cointegration(prices_a: pd.Series, prices_b: pd.Series) -> dict:
    """
    Simple cointegration test using Engle-Granger method.

    For two series to be cointegrated:
    1. Both must be non-stationary (have unit root)
    2. A linear combination must be stationary

    Returns ADF test statistic on the spread.
    Lower (more negative) values indicate stronger cointegration.
    """
    _, hedge_ratio = calculate_spread(prices_a, prices_b)
    spread = prices_a - hedge_ratio * prices_b

    # Augmented Dickey-Fuller test (simplified)
    # We compute the ADF statistic manually
    spread_diff = spread.diff().dropna()
    spread_lag = spread.shift(1).dropna()

    # Align series
    min_len = min(len(spread_diff), len(spread_lag))
    spread_diff = spread_diff.iloc[-min_len:]
    spread_lag = spread_lag.iloc[-min_len:]

    if len(spread_lag) < 20 or spread_lag.std() == 0:
        return {'adf_statistic': 0, 'hedge_ratio': hedge_ratio, 'is_cointegrated': False}

    # OLS: spread_diff = alpha + gamma * spread_lag + error
    # ADF statistic = gamma / std_err(gamma)
    cov = np.cov(spread_lag, spread_diff)[0, 1]
    var = np.var(spread_lag)
    gamma = cov / var if var != 0 else 0

    # Residuals
    residuals = spread_diff - gamma * spread_lag
    n = len(residuals)
    se_gamma = np.sqrt(np.sum(residuals**2) / (n - 2) / np.sum((spread_lag - spread_lag.mean())**2))
    adf_stat = gamma / se_gamma if se_gamma != 0 else 0

    # Critical values at 5% level approximately -2.86 for n=100
    # More negative = reject null of unit root = spread is stationary = cointegrated
    is_cointegrated = adf_stat < -2.86

    return {
        'adf_statistic': adf_stat,
        'hedge_ratio': hedge_ratio,
        'is_cointegrated': is_cointegrated,
        'spread_mean': spread.mean(),
        'spread_std': spread.std()
    }


class PairsTradingStrategy(Strategy):
    """
    Statistical Arbitrage via Pairs Trading.

    Entry rules:
    - Go long (buy asset A, short asset B) when z-score < -entry_threshold
    - Go short (short asset A, buy asset B) when z-score > entry_threshold

    Exit rules:
    - Close position when z-score crosses back through exit_threshold (typically 0)

    For this simplified version (single-asset backtest framework), we simulate
    pairs trading on a single synthetic spread. In practice, you'd need a
    multi-asset framework to properly implement this.
    """

    def __init__(
        self,
        lookback: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        stop_loss_threshold: float = 4.0
    ):
        super().__init__('Pairs Trading (Spread Mean Reversion)')
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on spread z-score.

        Since the base framework is single-asset, we create a synthetic
        "spread" using the asset vs its moving average as a mean reversion proxy.

        For proper pairs trading, extend the framework to handle two assets.
        """
        df = data.copy()

        # Create synthetic spread: price vs its long-term trend
        # This is a simplified proxy for pairs trading concept
        df['ma_long'] = df['close'].rolling(window=self.lookback * 2).mean()
        df['spread'] = df['close'] - df['ma_long']

        # Calculate z-score
        df['zscore'] = calculate_zscore(df['spread'], self.lookback)

        # Generate signals using z-score thresholds
        df['signal'] = 0
        df['position'] = 0

        position = 0
        signals = []

        for i in range(len(df)):
            z = df['zscore'].iloc[i]

            if pd.isna(z):
                signals.append(0)
                continue

            signal = 0

            # Entry logic
            if position == 0:
                if z < -self.entry_threshold:  # Spread too low, expect reversion up
                    signal = 1  # Buy
                    position = 1
                elif z > self.entry_threshold:  # Spread too high, expect reversion down
                    signal = -1  # Sell (or short in full implementation)
                    position = -1

            # Exit logic
            elif position == 1:  # Long position
                if z > -self.exit_threshold or z > self.stop_loss_threshold:
                    signal = -1  # Close long
                    position = 0

            elif position == -1:  # Short position (simulated as no position here)
                if z < self.exit_threshold or z < -self.stop_loss_threshold:
                    signal = 1  # Close short
                    position = 0

            signals.append(signal)

        df['signal'] = signals
        return df

    def get_params(self):
        return {
            'lookback': self.lookback,
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold,
            'stop_loss_threshold': self.stop_loss_threshold
        }


class SpreadMeanReversionStrategy(Strategy):
    """
    Bollinger Band Mean Reversion on Price Spread.

    Alternative implementation using Bollinger Bands instead of z-score.
    Entry when price breaks below lower band, exit at middle band.

    This captures the same mean reversion concept but with a different
    statistical framework.
    """

    def __init__(self, lookback: int = 20, num_std: float = 2.0):
        super().__init__('Spread Mean Reversion (Bollinger)')
        self.lookback = lookback
        self.num_std = num_std

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Bollinger Bands
        df['ma'] = df['close'].rolling(window=self.lookback).mean()
        df['std'] = df['close'].rolling(window=self.lookback).std()
        df['upper_band'] = df['ma'] + self.num_std * df['std']
        df['lower_band'] = df['ma'] - self.num_std * df['std']

        # Percentage distance from MA (normalized spread)
        df['spread_pct'] = (df['close'] - df['ma']) / df['ma'] * 100

        df['signal'] = 0
        position = 0
        signals = []

        for i in range(len(df)):
            if pd.isna(df['lower_band'].iloc[i]):
                signals.append(0)
                continue

            price = df['close'].iloc[i]
            lower = df['lower_band'].iloc[i]
            upper = df['upper_band'].iloc[i]
            ma = df['ma'].iloc[i]

            signal = 0

            if position == 0:
                if price < lower:  # Below lower band - oversold
                    signal = 1
                    position = 1
            elif position == 1:
                if price >= ma:  # Reverted to mean
                    signal = -1
                    position = 0
                elif price > upper:  # Stop loss - went wrong direction
                    signal = -1
                    position = 0

            signals.append(signal)

        df['signal'] = signals
        return df

    def get_params(self):
        return {'lookback': self.lookback, 'num_std': self.num_std}
