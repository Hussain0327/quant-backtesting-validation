from data.fetcher import fetch_data
from strategies import MovingAverageCrossover, RSIStrategy, PairsTradingStrategy
from backtest.engine import BacktestEngine
from analytics.metrics import calculate_metrics
from analytics.significance import (
    bootstrap_sharpe_confidence_interval,
    permutation_test_vs_baseline,
    monte_carlo_under_null,
    analyze_return_distribution
)


def main():
    print('=' * 60)
    print('QUANTITATIVE TRADING RESEARCH FRAMEWORK')
    print('=' * 60)

    print('\nfetching market data...')
    data = fetch_data('AAPL', '2022-01-01', '2024-01-01')

    strategy = MovingAverageCrossover(short_window=20, long_window=50)
    engine = BacktestEngine(initial_capital=10000)

    print(f'running backtest with {strategy.name}...')
    results = engine.run(data, strategy)

    train_metrics = calculate_metrics(results['train'])
    test_metrics = calculate_metrics(results['test'])

    print('\n' + '-' * 40)
    print('BACKTEST RESULTS')
    print('-' * 40)

    print('\n[Training Period]')
    print(f"  Return:       {train_metrics['total_return']:>8.2f}%")
    print(f"  Sharpe:       {train_metrics['sharpe']:>8.2f}")
    print(f"  Max Drawdown: {train_metrics['max_drawdown']:>8.2f}%")
    print(f"  Win Rate:     {train_metrics['win_rate']:>8.1f}%")
    print(f"  Trades:       {train_metrics['num_trades']:>8}")

    print('\n[Test Period (Out-of-Sample)]')
    print(f"  Return:       {test_metrics['total_return']:>8.2f}%")
    print(f"  Sharpe:       {test_metrics['sharpe']:>8.2f}")
    print(f"  Max Drawdown: {test_metrics['max_drawdown']:>8.2f}%")
    print(f"  Win Rate:     {test_metrics['win_rate']:>8.1f}%")
    print(f"  Trades:       {test_metrics['num_trades']:>8}")

    # Statistical Significance Testing
    print('\n' + '-' * 40)
    print('STATISTICAL SIGNIFICANCE ANALYSIS')
    print('-' * 40)

    test_equity = results['test']['equity_curve']
    test_returns = test_equity['equity'].pct_change().dropna()
    test_prices = test_equity['price']
    benchmark_returns = test_prices.pct_change().dropna()

    print('\nRunning bootstrap analysis (5000 samples)...')
    sharpe_ci = bootstrap_sharpe_confidence_interval(test_returns, n_bootstrap=5000)
    print(f"\n[1] Sharpe Ratio Confidence Interval")
    print(f"    Point Estimate: {sharpe_ci['sharpe']:.3f}")
    print(f"    95% CI:         [{sharpe_ci['ci_lower']:.3f}, {sharpe_ci['ci_upper']:.3f}]")
    print(f"    Std Error:      {sharpe_ci['std_error']:.3f}")
    sharpe_sig = not sharpe_ci.get('ci_includes_zero', True)
    print(f"    Significant:    {'YES - CI excludes zero' if sharpe_sig else 'NO - CI includes zero'}")

    print('\nRunning permutation test vs buy-and-hold...')
    perm_test = permutation_test_vs_baseline(test_returns, benchmark_returns, n_permutations=5000)
    print(f"\n[2] Permutation Test vs Benchmark")
    print(f"    Observed Diff:  {perm_test['observed_diff']:.4f} (annualized)")
    print(f"    p-value:        {perm_test['p_value']:.4f}")
    bench_sig = perm_test.get('significant_at_05', False)
    print(f"    Significant:    {'YES - p < 0.05' if bench_sig else 'NO - p >= 0.05'}")

    print('\nRunning Monte Carlo null hypothesis test...')
    mc_test = monte_carlo_under_null(
        test_prices,
        n_simulations=2000,
        strategy_return=results['test']['return_pct']
    )
    print(f"\n[3] vs Random Trading")
    print(f"    Strategy Return:    {mc_test.get('strategy_return', 0):.2f}%")
    print(f"    Random Mean:        {mc_test['null_mean']:.2f}%")
    print(f"    Random 95th Pctl:   {mc_test['null_95th_percentile']:.2f}%")
    print(f"    Percentile Rank:    {mc_test.get('percentile_rank', 50):.1f}%")
    print(f"    p-value:            {mc_test.get('p_value', 1):.4f}")
    random_sig = mc_test.get('significant_at_05', False)
    print(f"    Significant:        {'YES - beats random' if random_sig else 'NO - not better than random'}")

    print('\nAnalyzing return distribution...')
    dist_test = analyze_return_distribution(test_returns)
    print(f"\n[4] Return Distribution")
    print(f"    Daily Mean:         {dist_test.get('mean_daily', 0)*100:.4f}%")
    print(f"    Daily Std:          {dist_test.get('std_daily', 0)*100:.4f}%")
    print(f"    Skewness:           {dist_test.get('skewness', 0):.3f}")
    print(f"    Excess Kurtosis:    {dist_test.get('excess_kurtosis', 0):.3f}")
    print(f"    95% VaR:            {dist_test.get('var_95', 0)*100:.2f}%")
    print(f"    Fat Tails:          {'YES' if dist_test.get('is_fat_tailed', False) else 'NO'}")
    print(f"    Negative Skew:      {'YES' if dist_test.get('is_negatively_skewed', False) else 'NO'}")

    # Overall Assessment
    print('\n' + '=' * 60)
    print('OVERALL ASSESSMENT')
    print('=' * 60)

    tests_passed = sum([sharpe_sig, bench_sig, random_sig])

    if tests_passed == 3:
        verdict = "STRONG EVIDENCE OF EDGE"
        detail = "Strategy passes all significance tests. Warrants further research."
    elif tests_passed == 2:
        verdict = "MODERATE EVIDENCE"
        detail = "Strategy passes 2/3 tests. Promising but not conclusive."
    elif tests_passed == 1:
        verdict = "WEAK EVIDENCE"
        detail = "Strategy passes only 1/3 tests. Likely noise or overfitting."
    else:
        verdict = "NO STATISTICAL EVIDENCE"
        detail = "Strategy fails all tests. Returns indistinguishable from random."

    print(f"\nTests Passed: {tests_passed}/3")
    print(f"Verdict:      {verdict}")
    print(f"Detail:       {detail}")

    print('\n' + '-' * 60)
    print('NOTE: These results are for research purposes only.')
    print('Past performance does not guarantee future results.')
    print('-' * 60)


if __name__ == '__main__':
    main()
