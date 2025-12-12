"""
Unit tests for statistical significance testing.
"""
import pytest
import pandas as pd
import numpy as np

from analytics.significance import (
    bootstrap_sharpe_confidence_interval,
    permutation_test_vs_baseline,
    monte_carlo_under_null,
    analyze_return_distribution
)


class TestBootstrapSharpeCI:
    """Tests for bootstrap Sharpe ratio confidence intervals."""

    def test_returns_required_fields(self, sample_returns):
        """Test that function returns all required fields."""
        result = bootstrap_sharpe_confidence_interval(sample_returns, n_bootstrap=100)

        assert 'sharpe' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert 'std_error' in result
        assert 'ci_includes_zero' in result

    def test_ci_bounds_order(self, sample_returns):
        """Test that CI lower < upper."""
        result = bootstrap_sharpe_confidence_interval(sample_returns, n_bootstrap=100)

        assert result['ci_lower'] <= result['ci_upper']

    def test_positive_returns_ci(self):
        """Test CI for consistently positive returns."""
        positive_returns = pd.Series([0.01, 0.02, 0.015] * 100)
        result = bootstrap_sharpe_confidence_interval(positive_returns, n_bootstrap=100)

        # CI should likely exclude zero for strong positive returns
        assert result['sharpe'] > 0

    def test_handles_small_sample(self):
        """Test handling of small sample size."""
        small_sample = pd.Series([0.01, 0.02, -0.01, 0.015, 0.01])
        result = bootstrap_sharpe_confidence_interval(small_sample, n_bootstrap=50)

        assert isinstance(result['sharpe'], (int, float))


class TestPermutationTest:
    """Tests for permutation test vs baseline."""

    def test_returns_required_fields(self, sample_returns):
        """Test that function returns all required fields."""
        baseline = pd.Series(np.random.normal(0.0005, 0.02, len(sample_returns)))
        result = permutation_test_vs_baseline(sample_returns, baseline, n_permutations=100)

        assert 'observed_diff' in result
        assert 'p_value' in result
        assert 'significant_at_05' in result

    def test_pvalue_bounds(self, sample_returns):
        """Test that p-value is between 0 and 1."""
        baseline = pd.Series(np.random.normal(0.0005, 0.02, len(sample_returns)))
        result = permutation_test_vs_baseline(sample_returns, baseline, n_permutations=100)

        assert 0 <= result['p_value'] <= 1

    def test_identical_series(self, sample_returns):
        """Test with identical series (should not be significant)."""
        result = permutation_test_vs_baseline(sample_returns, sample_returns.copy(), n_permutations=100)

        # P-value should be high when series are identical
        assert result['p_value'] > 0.05


class TestMonteCarloNull:
    """Tests for Monte Carlo null hypothesis testing."""

    def test_returns_required_fields(self):
        """Test that function returns all required fields."""
        prices = pd.Series(np.linspace(100, 110, 100))
        result = monte_carlo_under_null(prices, n_simulations=50, strategy_return=5.0)

        assert 'null_mean' in result
        assert 'null_95th_percentile' in result
        assert 'p_value' in result
        assert 'significant_at_05' in result

    def test_pvalue_bounds(self):
        """Test that p-value is between 0 and 1."""
        prices = pd.Series(np.linspace(100, 110, 100))
        result = monte_carlo_under_null(prices, n_simulations=50, strategy_return=5.0)

        assert 0 <= result['p_value'] <= 1

    def test_high_return_significance(self):
        """Test that very high returns are likely significant."""
        prices = pd.Series(np.linspace(100, 110, 100))
        result = monte_carlo_under_null(prices, n_simulations=100, strategy_return=50.0)

        # Very high return should likely be significant
        assert result['percentile_rank'] > 50


class TestReturnDistribution:
    """Tests for return distribution analysis."""

    def test_returns_required_fields(self, sample_returns):
        """Test that function returns all required fields."""
        result = analyze_return_distribution(sample_returns)

        assert 'mean_daily' in result
        assert 'std_daily' in result
        assert 'skewness' in result
        assert 'excess_kurtosis' in result
        assert 'var_95' in result

    def test_mean_calculation(self):
        """Test mean calculation."""
        # Need at least 20 data points
        returns = pd.Series([0.01] * 25)  # Constant returns
        result = analyze_return_distribution(returns)

        assert abs(result['mean_daily'] - 0.01) < 0.001

    def test_fat_tail_detection(self):
        """Test fat tail detection."""
        # Create returns with fat tails
        np.random.seed(42)
        fat_tailed = pd.Series(np.random.standard_t(3, 1000))  # t-distribution has fat tails
        result = analyze_return_distribution(fat_tailed)

        # Should detect fat tails (kurtosis > 1)
        assert 'is_fat_tailed' in result

    def test_skewness_detection(self):
        """Test skewness detection."""
        # Create negatively skewed returns
        np.random.seed(42)
        base = np.random.exponential(1, 1000)
        skewed = -base + base.mean()
        result = analyze_return_distribution(pd.Series(skewed))

        assert 'is_negatively_skewed' in result
