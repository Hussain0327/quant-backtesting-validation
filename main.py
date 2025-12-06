#!/usr/bin/env python3
"""
Quantitative Systems Simulator - Demo Script

This script demonstrates the core functionality of QSS.
Run with: python main.py
"""

import sys
from pathlib import Path

# Add Python package to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

import numpy as np

from qss import Orchestrator
from qss.interface import SimulationConfig, AssetData


def main():
    print("=" * 60)
    print("Quantitative Systems Simulator - Demo")
    print("=" * 60)
    print()

    # Configuration
    config = SimulationConfig(
        n_simulations=10000,
        time_horizon=252,  # 1 year
        risk_free_rate=0.02,
        seed=42  # For reproducibility
    )

    print(f"Configuration:")
    print(f"  Simulations: {config.n_simulations:,}")
    print(f"  Time Horizon: {config.time_horizon} trading days")
    print(f"  Risk-Free Rate: {config.risk_free_rate:.1%}")
    print()

    # Initialize orchestrator
    orch = Orchestrator(config)

    # Create portfolio manually
    portfolio_data = {
        "assets": [
            {"symbol": "AAPL", "weight": 0.25, "expected_return": 0.12, "volatility": 0.25},
            {"symbol": "MSFT", "weight": 0.25, "expected_return": 0.10, "volatility": 0.22},
            {"symbol": "NVDA", "weight": 0.20, "expected_return": 0.18, "volatility": 0.40},
            {"symbol": "GOOG", "weight": 0.15, "expected_return": 0.11, "volatility": 0.28},
            {"symbol": "AMZN", "weight": 0.15, "expected_return": 0.14, "volatility": 0.32},
        ]
    }

    # Create correlation matrix
    correlation = np.array([
        [1.00, 0.65, 0.55, 0.60, 0.58],
        [0.65, 1.00, 0.52, 0.58, 0.55],
        [0.55, 0.52, 1.00, 0.48, 0.52],
        [0.60, 0.58, 0.48, 1.00, 0.62],
        [0.58, 0.55, 0.52, 0.62, 1.00]
    ])

    # Get volatilities and compute covariance
    vols = np.array([a["volatility"] for a in portfolio_data["assets"]])
    cov_matrix = np.outer(vols, vols) * correlation

    # Load portfolio
    orch.load_portfolio_from_dict(portfolio_data, cov_matrix)

    print("Portfolio:")
    for asset in portfolio_data["assets"]:
        print(f"  {asset['symbol']}: {asset['weight']:.0%} weight, "
              f"{asset['expected_return']:.1%} return, {asset['volatility']:.1%} vol")
    print()

    # Run simulation
    print("Running Monte Carlo simulation...")
    result = orch.run_simulation()
    print(f"Complete! Generated {len(result.terminal_values):,} scenarios.")
    print()

    # Display metrics
    metrics = result.metrics
    print("=" * 60)
    print("PORTFOLIO RISK METRICS")
    print("=" * 60)
    print(f"Expected Annual Return:  {metrics.expected_return:>10.2%}")
    print(f"Volatility:              {metrics.volatility:>10.2%}")
    print(f"Sharpe Ratio:            {metrics.sharpe_ratio:>10.2f}")
    print()
    print(f"Value at Risk (95%):     {metrics.var_95:>10.2%}")
    print(f"Value at Risk (99%):     {metrics.var_99:>10.2%}")
    print(f"CVaR / ES (95%):         {metrics.cvar_95:>10.2%}")
    print(f"CVaR / ES (99%):         {metrics.cvar_99:>10.2%}")
    print()
    print(f"Skewness:                {metrics.skewness:>10.3f}")
    print(f"Excess Kurtosis:         {metrics.kurtosis:>10.3f}")
    print(f"Avg Max Drawdown:        {metrics.max_drawdown:>10.2%}")
    print(f"Convergence Error:       {result.convergence_error:>10.4f}")
    print("=" * 60)
    print()

    # Compute analytics
    print("Distribution Analysis:")
    analytics = orch.compute_analytics()
    dist = analytics["distribution_analysis"]
    print(f"  Is Fat-Tailed: {dist['is_fat_tailed']}")
    print(f"  Is Skewed: {dist['is_skewed']}")
    jb = dist["normality_tests"]["jarque_bera"]
    print(f"  Jarque-Bera p-value: {jb['p_value']:.4f}")
    print(f"  Normal Distribution: {'Yes' if jb['is_normal'] else 'No'}")
    print()

    # Confidence intervals
    print("95% Confidence Intervals:")
    cis = analytics["confidence_intervals"]
    for name, ci in cis.items():
        print(f"  {name}: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    print()

    # Generate report
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    report_path = output_dir / "demo_report.json"
    orch.generate_report(report_path, format="json")
    print(f"Report saved to: {report_path}")

    # Optional: Create visualizations if matplotlib is available
    try:
        from qss.visualization import Visualizer
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        viz = Visualizer(backend='matplotlib')
        returns = result.terminal_values - 1.0

        # Save distribution plot
        fig = viz.plot_return_distribution(
            returns,
            var_95=metrics.var_95,
            cvar_95=metrics.cvar_95,
            title='Portfolio Return Distribution',
            save_path=output_dir / "distribution.png",
            show=False
        )
        print(f"Distribution plot saved to: {output_dir / 'distribution.png'}")

        # Save paths plot
        fig = viz.plot_simulation_paths(
            result.simulated_paths,
            n_paths=50,
            title='Monte Carlo Simulation Paths',
            save_path=output_dir / "paths.png",
            show=False
        )
        print(f"Paths plot saved to: {output_dir / 'paths.png'}")

    except ImportError as e:
        print(f"Visualization skipped (matplotlib not available): {e}")
    except Exception as e:
        print(f"Visualization error: {e}")

    print()
    print("Demo complete!")


if __name__ == "__main__":
    main()
