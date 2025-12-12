"""
Unit tests for analytics metrics.
"""
import pytest
import pandas as pd
import numpy as np

from analytics.metrics import sharpe_ratio, max_drawdown, win_rate, calculate_metrics


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_positive_returns(self):
        """Test Sharpe with positive returns."""
        returns = pd.Series([0.01, 0.02, 0.01, 0.015, 0.01] * 50)  # 250 days
        sharpe = sharpe_ratio(returns)

        # Should be positive for consistently positive returns
        assert sharpe > 0

    def test_negative_returns(self):
        """Test Sharpe with negative returns."""
        returns = pd.Series([-0.01, -0.02, -0.01, -0.015, -0.01] * 50)
        sharpe = sharpe_ratio(returns)

        # Should be negative for consistently negative returns
        assert sharpe < 0

    def test_zero_volatility(self):
        """Test Sharpe handles zero volatility."""
        returns = pd.Series([0.01] * 100)  # Constant returns
        sharpe = sharpe_ratio(returns)

        # Should handle without error
        assert isinstance(sharpe, (int, float))

    def test_empty_returns(self):
        """Test Sharpe handles empty series."""
        returns = pd.Series([])
        sharpe = sharpe_ratio(returns)

        # Should return 0 or handle gracefully
        assert sharpe == 0 or np.isnan(sharpe)


class TestMaxDrawdown:
    """Tests for max drawdown calculation."""

    def test_no_drawdown(self):
        """Test with monotonically increasing equity."""
        equity_df = pd.DataFrame({'equity': [100, 110, 120, 130, 140]})
        dd = max_drawdown(equity_df)

        assert dd == 0

    def test_known_drawdown(self):
        """Test with known drawdown."""
        equity_df = pd.DataFrame({'equity': [100, 110, 100, 105, 110]})  # 10/110 = 9.09% drawdown
        dd = max_drawdown(equity_df)

        assert abs(dd - (-9.09)) < 0.1  # Returns negative value

    def test_large_drawdown(self):
        """Test with large drawdown."""
        equity_df = pd.DataFrame({'equity': [100, 50, 60, 70]})  # 50% drawdown
        dd = max_drawdown(equity_df)

        assert dd == -50.0  # Returns negative value

    def test_empty_equity(self):
        """Test with empty series."""
        equity_df = pd.DataFrame({'equity': []})
        dd = max_drawdown(equity_df)

        # Should handle without error
        assert pd.isna(dd) or dd == 0


class TestWinRate:
    """Tests for win rate calculation."""

    def test_all_wins(self):
        """Test with all winning trades."""
        trades = [
            {'type': 'buy', 'price': 100},
            {'type': 'sell', 'price': 110},
            {'type': 'buy', 'price': 105},
            {'type': 'sell', 'price': 115},
        ]
        wr = win_rate(trades)

        assert wr == 100.0

    def test_all_losses(self):
        """Test with all losing trades."""
        trades = [
            {'type': 'buy', 'price': 100},
            {'type': 'sell', 'price': 90},
            {'type': 'buy', 'price': 95},
            {'type': 'sell', 'price': 85},
        ]
        wr = win_rate(trades)

        assert wr == 0.0

    def test_mixed_trades(self):
        """Test with mixed wins/losses."""
        trades = [
            {'type': 'buy', 'price': 100},
            {'type': 'sell', 'price': 110},  # Win
            {'type': 'buy', 'price': 105},
            {'type': 'sell', 'price': 100},  # Loss
        ]
        wr = win_rate(trades)

        assert wr == 50.0

    def test_no_trades(self):
        """Test with no trades."""
        trades = []
        wr = win_rate(trades)

        assert wr == 0


class TestCalculateMetrics:
    """Tests for overall metrics calculation."""

    def test_returns_all_metrics(self, sample_price_data):
        """Test that calculate_metrics returns all required fields."""
        from backtest.engine import BacktestEngine
        from strategies import MovingAverageCrossover

        engine = BacktestEngine(initial_capital=10000)
        strategy = MovingAverageCrossover(short_window=5, long_window=10)
        results = engine.run(sample_price_data, strategy, train_pct=0.7)

        metrics = calculate_metrics(results['test'])

        assert 'total_return' in metrics
        assert 'sharpe' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'num_trades' in metrics
        assert 'final_equity' in metrics

    def test_metrics_types(self, sample_price_data):
        """Test that metrics are correct types."""
        from backtest.engine import BacktestEngine
        from strategies import MovingAverageCrossover

        engine = BacktestEngine(initial_capital=10000)
        strategy = MovingAverageCrossover(short_window=5, long_window=10)
        results = engine.run(sample_price_data, strategy, train_pct=0.7)

        metrics = calculate_metrics(results['test'])

        assert isinstance(metrics['total_return'], (int, float))
        assert isinstance(metrics['sharpe'], (int, float))
        assert isinstance(metrics['max_drawdown'], (int, float))
        assert isinstance(metrics['win_rate'], (int, float))
        assert isinstance(metrics['num_trades'], int)
        assert isinstance(metrics['final_equity'], (int, float))
