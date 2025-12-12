"""
Statistical Significance Testing for Backtesting Results

This module provides rigorous statistical tests to evaluate whether strategy
performance is statistically significant or likely due to random chance.

Key insight: A backtest showing 15% returns means nothing without knowing
the probability that result occurred by chance.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional


def bootstrap_sharpe_confidence_interval(
    returns: pd.Series,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculate bootstrap confidence interval for Sharpe ratio.

    The Sharpe ratio from a single backtest is a point estimate.
    This function provides confidence bounds to understand uncertainty.

    Args:
        returns: Daily returns series
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        risk_free_rate: Annual risk-free rate

    Returns:
        Dict with point estimate and confidence bounds
    """
    returns = returns.dropna()
    if len(returns) < 20:
        return {'sharpe': 0, 'ci_lower': 0, 'ci_upper': 0, 'std_error': 0}

    daily_rf = risk_free_rate / 252

    def calc_sharpe(r):
        excess = r - daily_rf
        if excess.std() == 0:
            return 0
        return np.sqrt(252) * excess.mean() / excess.std()

    point_estimate = calc_sharpe(returns)

    # Bootstrap resampling
    bootstrap_sharpes = []
    n = len(returns)
    returns_arr = returns.values

    for _ in range(n_bootstrap):
        sample = np.random.choice(returns_arr, size=n, replace=True)
        sample_series = pd.Series(sample)
        bootstrap_sharpes.append(calc_sharpe(sample_series))

    bootstrap_sharpes = np.array(bootstrap_sharpes)
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_sharpes, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_sharpes, (1 - alpha / 2) * 100)

    return {
        'sharpe': point_estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std_error': bootstrap_sharpes.std(),
        'ci_includes_zero': ci_lower <= 0 <= ci_upper
    }


def permutation_test_vs_baseline(
    strategy_returns: pd.Series,
    baseline_returns: pd.Series,
    n_permutations: int = 10000,
    metric: str = 'mean'
) -> Dict[str, float]:
    """
    Permutation test to determine if strategy outperformance is significant.

    Null hypothesis: Strategy and baseline returns come from the same distribution.

    This is more robust than t-tests because it doesn't assume normality,
    which is important since financial returns often have fat tails.

    Args:
        strategy_returns: Strategy daily returns
        baseline_returns: Baseline (e.g., buy-and-hold) daily returns
        n_permutations: Number of permutations
        metric: 'mean' for average return, 'sharpe' for risk-adjusted

    Returns:
        Dict with test statistic and p-value
    """
    strategy_returns = strategy_returns.dropna()
    baseline_returns = baseline_returns.dropna()

    # Align lengths
    min_len = min(len(strategy_returns), len(baseline_returns))
    strategy_returns = strategy_returns.iloc[:min_len]
    baseline_returns = baseline_returns.iloc[:min_len]

    if min_len < 10:
        return {'observed_diff': 0, 'p_value': 1.0, 'significant': False}

    def calc_metric(r):
        if metric == 'sharpe':
            if r.std() == 0:
                return 0
            return np.sqrt(252) * r.mean() / r.std()
        return r.mean() * 252  # Annualized mean return

    observed_diff = calc_metric(strategy_returns) - calc_metric(baseline_returns)

    # Pool all returns
    combined = np.concatenate([strategy_returns.values, baseline_returns.values])
    n_strategy = len(strategy_returns)

    # Permutation test
    count_extreme = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_strategy = pd.Series(combined[:n_strategy])
        perm_baseline = pd.Series(combined[n_strategy:])
        perm_diff = calc_metric(perm_strategy) - calc_metric(perm_baseline)

        if perm_diff >= observed_diff:
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    return {
        'observed_diff': observed_diff,
        'p_value': p_value,
        'significant_at_05': p_value < 0.05,
        'significant_at_01': p_value < 0.01
    }


def monte_carlo_under_null(
    prices: pd.Series,
    n_simulations: int = 1000,
    strategy_return: float = None
) -> Dict[str, float]:
    """
    Monte Carlo simulation to test if strategy beats random entry/exit.

    Generates random trading signals and computes distribution of returns
    under the null hypothesis that the strategy has no edge.

    Args:
        prices: Price series used in backtest
        n_simulations: Number of random strategy simulations
        strategy_return: Actual strategy return to compare against

    Returns:
        Dict with null distribution statistics and p-value
    """
    prices = prices.dropna()
    if len(prices) < 20:
        return {'p_value': 1.0, 'null_mean': 0, 'null_std': 0}

    random_returns = []

    for _ in range(n_simulations):
        # Generate random signals
        n = len(prices)
        # Random number of trades (similar to typical strategy)
        n_trades = np.random.randint(2, max(3, n // 20))

        # Random entry/exit points
        trade_points = sorted(np.random.choice(range(1, n-1), size=min(n_trades * 2, n-2), replace=False))

        # Simulate random strategy
        capital = 10000
        position = 0

        for i, idx in enumerate(trade_points):
            price = prices.iloc[idx]
            if i % 2 == 0 and position == 0:  # Buy
                position = capital / price
                capital = 0
            elif i % 2 == 1 and position > 0:  # Sell
                capital = position * price
                position = 0

        # Close any open position at end
        if position > 0:
            capital = position * prices.iloc[-1]

        final_return = (capital - 10000) / 10000 * 100
        random_returns.append(final_return)

    random_returns = np.array(random_returns)
    null_mean = random_returns.mean()
    null_std = random_returns.std()

    result = {
        'null_mean': null_mean,
        'null_std': null_std,
        'null_median': np.median(random_returns),
        'null_5th_percentile': np.percentile(random_returns, 5),
        'null_95th_percentile': np.percentile(random_returns, 95)
    }

    if strategy_return is not None:
        # One-tailed test: what fraction of random strategies beat ours?
        p_value = np.mean(random_returns >= strategy_return)
        result['strategy_return'] = strategy_return
        result['p_value'] = p_value
        result['percentile_rank'] = np.mean(random_returns <= strategy_return) * 100
        result['significant_at_05'] = p_value < 0.05

    return result


def analyze_return_distribution(returns: pd.Series) -> Dict[str, float]:
    """
    Test statistical properties of the return distribution.

    Important for understanding if standard assumptions hold and
    whether risk metrics are reliable.

    Args:
        returns: Daily returns series

    Returns:
        Dict with distribution statistics and normality tests
    """
    returns = returns.dropna()

    if len(returns) < 20:
        return {'error': 'Insufficient data for distribution analysis'}

    # Basic statistics
    mean = returns.mean()
    std = returns.std()
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)  # Excess kurtosis (normal = 0)

    # Normality tests
    if len(returns) >= 20:
        shapiro_stat, shapiro_p = stats.shapiro(returns[:min(5000, len(returns))])
    else:
        shapiro_stat, shapiro_p = 0, 1

    # Jarque-Bera test
    jb_stat, jb_p = stats.jarque_bera(returns)

    return {
        'mean_daily': mean,
        'std_daily': std,
        'skewness': skewness,
        'excess_kurtosis': kurtosis,
        'is_fat_tailed': kurtosis > 1,  # Higher than normal
        'is_negatively_skewed': skewness < -0.5,
        'shapiro_wilk_p': shapiro_p,
        'jarque_bera_p': jb_p,
        'is_normal_shapiro': shapiro_p > 0.05,
        'is_normal_jb': jb_p > 0.05,
        'var_95': np.percentile(returns, 5),  # 95% VaR
        'cvar_95': returns[returns <= np.percentile(returns, 5)].mean()  # Expected shortfall
    }


def calculate_benchmark_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate buy-and-hold benchmark returns for comparison.

    Args:
        prices: Price series

    Returns:
        Daily returns series for buy-and-hold strategy
    """
    return prices.pct_change().dropna()


def strategy_significance_report(
    strategy_results: Dict,
    prices: pd.Series,
    n_bootstrap: int = 5000,
    n_permutations: int = 5000
) -> Dict:
    """
    Generate comprehensive statistical significance report for a strategy.

    This is the main function to call after running a backtest.

    Args:
        strategy_results: Results dict from BacktestEngine
        prices: Price series used in backtest
        n_bootstrap: Bootstrap samples for confidence intervals
        n_permutations: Permutations for hypothesis tests

    Returns:
        Complete significance report
    """
    equity_curve = strategy_results['equity_curve']
    strategy_returns = equity_curve['equity'].pct_change().dropna()
    benchmark_returns = calculate_benchmark_returns(prices)

    report = {
        'sharpe_confidence': bootstrap_sharpe_confidence_interval(
            strategy_returns, n_bootstrap=n_bootstrap
        ),
        'vs_benchmark': permutation_test_vs_baseline(
            strategy_returns,
            benchmark_returns,
            n_permutations=n_permutations
        ),
        'vs_random': monte_carlo_under_null(
            prices,
            n_simulations=n_permutations,
            strategy_return=strategy_results['return_pct']
        ),
        'return_distribution': test_return_distribution(strategy_returns)
    }

    # Summary interpretation
    sharpe_significant = not report['sharpe_confidence'].get('ci_includes_zero', True)
    beats_benchmark = report['vs_benchmark'].get('significant_at_05', False)
    beats_random = report['vs_random'].get('significant_at_05', False)

    report['summary'] = {
        'sharpe_statistically_significant': sharpe_significant,
        'beats_benchmark_significantly': beats_benchmark,
        'beats_random_trading': beats_random,
        'overall_evidence': _interpret_evidence(sharpe_significant, beats_benchmark, beats_random)
    }

    return report


def _interpret_evidence(sharpe_sig: bool, beats_bench: bool, beats_random: bool) -> str:
    """Interpret the statistical evidence."""
    score = sum([sharpe_sig, beats_bench, beats_random])

    if score == 3:
        return "Strong evidence of genuine edge"
    elif score == 2:
        return "Moderate evidence - warrants further investigation"
    elif score == 1:
        return "Weak evidence - likely noise or overfitting"
    else:
        return "No statistical evidence of edge over random"
