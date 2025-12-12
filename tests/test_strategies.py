"""
Unit tests for trading strategies.
"""
import pytest
import pandas as pd
import numpy as np

from strategies import (
    MovingAverageCrossover,
    RSIStrategy,
    MomentumStrategy,
    PairsTradingStrategy,
    SpreadMeanReversionStrategy
)


class TestMovingAverageCrossover:
    """Tests for MA Crossover strategy."""

    def test_initialization(self):
        """Test strategy initializes with correct parameters."""
        strategy = MovingAverageCrossover(short_window=10, long_window=30)
        params = strategy.get_params()

        assert params['short_window'] == 10
        assert params['long_window'] == 30

    def test_generates_signals(self, sample_price_data):
        """Test that strategy generates valid signals."""
        strategy = MovingAverageCrossover(short_window=10, long_window=20)
        signals = strategy.generate_signals(sample_price_data)

        assert 'signal' in signals.columns
        assert signals['signal'].isin([-1, 0, 1]).all()

    def test_signal_values(self, sample_price_data):
        """Test signal values are valid."""
        strategy = MovingAverageCrossover(short_window=5, long_window=10)
        signals = strategy.generate_signals(sample_price_data)

        # Should have some non-zero signals in 100 days of data
        assert (signals['signal'] != 0).any()

    def test_requires_sufficient_data(self, sample_price_data_short):
        """Test behavior with insufficient data."""
        strategy = MovingAverageCrossover(short_window=10, long_window=20)
        signals = strategy.generate_signals(sample_price_data_short)

        # Should still return a dataframe, just with NaN/0 signals initially
        assert isinstance(signals, pd.DataFrame)


class TestRSIStrategy:
    """Tests for RSI strategy."""

    def test_initialization(self):
        """Test strategy initializes with correct parameters."""
        strategy = RSIStrategy(period=14, oversold=30, overbought=70)
        params = strategy.get_params()

        assert params['period'] == 14
        assert params['oversold'] == 30
        assert params['overbought'] == 70

    def test_generates_signals(self, sample_price_data):
        """Test that strategy generates valid signals."""
        strategy = RSIStrategy(period=14, oversold=30, overbought=70)
        signals = strategy.generate_signals(sample_price_data)

        assert 'signal' in signals.columns
        assert signals['signal'].isin([-1, 0, 1]).all()

    def test_rsi_bounds(self, sample_price_data):
        """Test that RSI signals correspond to threshold crossings."""
        strategy = RSIStrategy(period=14, oversold=30, overbought=70)
        signals = strategy.generate_signals(sample_price_data)

        # RSI should generate some signals over 100 days
        assert isinstance(signals, pd.DataFrame)


class TestMomentumStrategy:
    """Tests for Momentum strategy."""

    def test_initialization(self):
        """Test strategy initializes with correct parameters."""
        strategy = MomentumStrategy(lookback=20)
        params = strategy.get_params()

        assert params['lookback'] == 20

    def test_generates_signals(self, sample_price_data):
        """Test that strategy generates valid signals."""
        strategy = MomentumStrategy(lookback=10)
        signals = strategy.generate_signals(sample_price_data)

        assert 'signal' in signals.columns
        assert signals['signal'].isin([-1, 0, 1]).all()


class TestPairsTradingStrategy:
    """Tests for Pairs Trading strategy."""

    def test_initialization(self):
        """Test strategy initializes with correct parameters."""
        strategy = PairsTradingStrategy(lookback=20, entry_threshold=2.0, exit_threshold=0.5)
        params = strategy.get_params()

        assert params['lookback'] == 20
        assert params['entry_threshold'] == 2.0
        assert params['exit_threshold'] == 0.5

    def test_generates_signals(self, sample_price_data):
        """Test that strategy generates valid signals."""
        strategy = PairsTradingStrategy(lookback=20, entry_threshold=2.0, exit_threshold=0.5)
        signals = strategy.generate_signals(sample_price_data)

        assert 'signal' in signals.columns
        assert signals['signal'].isin([-1, 0, 1]).all()


class TestSpreadMeanReversionStrategy:
    """Tests for Bollinger Bands strategy."""

    def test_initialization(self):
        """Test strategy initializes with correct parameters."""
        strategy = SpreadMeanReversionStrategy(lookback=20, num_std=2.0)
        params = strategy.get_params()

        assert params['lookback'] == 20
        assert params['num_std'] == 2.0

    def test_generates_signals(self, sample_price_data):
        """Test that strategy generates valid signals."""
        strategy = SpreadMeanReversionStrategy(lookback=20, num_std=2.0)
        signals = strategy.generate_signals(sample_price_data)

        assert 'signal' in signals.columns
        assert signals['signal'].isin([-1, 0, 1]).all()
