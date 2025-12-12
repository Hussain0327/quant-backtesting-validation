"""
Unit tests for backtest engine.
"""
import pytest
import pandas as pd
import numpy as np

from backtest.engine import BacktestEngine
from backtest.costs import calculate_costs
from strategies import MovingAverageCrossover


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def test_initialization(self):
        """Test engine initializes with correct parameters."""
        engine = BacktestEngine(
            initial_capital=10000,
            commission=0.001,
            slippage=0.0005
        )

        assert engine.initial_capital == 10000
        assert engine.commission == 0.001
        assert engine.slippage == 0.0005

    def test_default_values(self):
        """Test default initialization values."""
        engine = BacktestEngine()

        assert engine.initial_capital == 10000
        assert engine.commission == 0.001
        assert engine.slippage == 0.0005

    def test_run_returns_results(self, sample_price_data):
        """Test that run() returns proper result structure."""
        engine = BacktestEngine(initial_capital=10000)
        strategy = MovingAverageCrossover(short_window=5, long_window=10)

        results = engine.run(sample_price_data, strategy, train_pct=0.7)

        assert 'train' in results
        assert 'test' in results
        assert 'strategy' in results
        assert 'params' in results

    def test_train_test_split(self, sample_price_data):
        """Test train/test split is applied correctly."""
        engine = BacktestEngine(initial_capital=10000)
        strategy = MovingAverageCrossover(short_window=5, long_window=10)

        results = engine.run(sample_price_data, strategy, train_pct=0.7)

        train_equity = results['train']['equity_curve']
        test_equity = results['test']['equity_curve']

        # Both should have data
        assert len(train_equity) > 0
        assert len(test_equity) > 0

    def test_equity_curve_structure(self, sample_price_data):
        """Test equity curve has required columns."""
        engine = BacktestEngine(initial_capital=10000)
        strategy = MovingAverageCrossover(short_window=5, long_window=10)

        results = engine.run(sample_price_data, strategy, train_pct=0.7)

        equity_curve = results['test']['equity_curve']

        assert 'equity' in equity_curve.columns
        assert 'price' in equity_curve.columns
        assert 'date' in equity_curve.columns

    def test_trades_recorded(self, sample_price_data):
        """Test that trades are recorded."""
        engine = BacktestEngine(initial_capital=10000)
        strategy = MovingAverageCrossover(short_window=5, long_window=10)

        results = engine.run(sample_price_data, strategy, train_pct=0.7)

        # Trades should be a list
        assert isinstance(results['train']['trades'], list)
        assert isinstance(results['test']['trades'], list)

    def test_return_calculation(self, sample_price_data):
        """Test return percentage is calculated."""
        engine = BacktestEngine(initial_capital=10000)
        strategy = MovingAverageCrossover(short_window=5, long_window=10)

        results = engine.run(sample_price_data, strategy, train_pct=0.7)

        assert 'return_pct' in results['train']
        assert 'return_pct' in results['test']
        assert isinstance(results['test']['return_pct'], (int, float))

    def test_different_capital_amounts(self, sample_price_data):
        """Test engine works with different capital amounts."""
        strategy = MovingAverageCrossover(short_window=5, long_window=10)

        for capital in [1000, 10000, 100000]:
            engine = BacktestEngine(initial_capital=capital)
            results = engine.run(sample_price_data, strategy, train_pct=0.7)

            # Final equity should be related to initial capital
            assert results['train']['final_equity'] > 0
            assert results['test']['final_equity'] > 0


class TestTransactionCosts:
    """Tests for transaction cost calculations."""

    def test_calculate_costs(self):
        """Test cost calculation."""
        trade_value = 10000
        commission = 0.001
        slippage = 0.0005

        cost = calculate_costs(trade_value, commission, slippage)

        expected = trade_value * (commission + slippage)
        assert cost == expected

    def test_zero_costs(self):
        """Test with zero commission and slippage."""
        cost = calculate_costs(10000, 0, 0)
        assert cost == 0

    def test_costs_scale_with_value(self):
        """Test costs scale linearly with trade value."""
        commission = 0.001
        slippage = 0.0005

        cost_small = calculate_costs(1000, commission, slippage)
        cost_large = calculate_costs(10000, commission, slippage)

        assert cost_large == cost_small * 10
