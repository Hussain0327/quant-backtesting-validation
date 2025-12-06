#!/usr/bin/env python3
"""
Benchmark: C++ Core vs Pure Python Implementation

This script compares the performance of the C++ Monte Carlo engine
against the pure Python implementation, demonstrating the advantages
of each approach.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "python"))

import numpy as np


def create_test_portfolio():
    """Create a test portfolio with correlation matrix."""
    from qss.interface import AssetData

    assets = [
        AssetData(symbol="AAPL", weight=0.25, expected_return=0.12, volatility=0.25),
        AssetData(symbol="MSFT", weight=0.25, expected_return=0.10, volatility=0.22),
        AssetData(symbol="NVDA", weight=0.20, expected_return=0.18, volatility=0.40),
        AssetData(symbol="GOOG", weight=0.15, expected_return=0.11, volatility=0.28),
        AssetData(symbol="AMZN", weight=0.15, expected_return=0.14, volatility=0.32),
    ]

    correlation = np.array([
        [1.00, 0.65, 0.55, 0.60, 0.58],
        [0.65, 1.00, 0.52, 0.58, 0.55],
        [0.55, 0.52, 1.00, 0.48, 0.52],
        [0.60, 0.58, 0.48, 1.00, 0.62],
        [0.58, 0.55, 0.52, 0.62, 1.00]
    ])

    vols = np.array([a.volatility for a in assets])
    cov_matrix = np.outer(vols, vols) * correlation

    return assets, cov_matrix


def benchmark_cpp_parallel(assets, cov_matrix, n_simulations, n_runs=3):
    """Benchmark parallel C++ implementation (multi-threaded)."""
    from qss.interface import SimulationConfig

    try:
        from qss import qss_core
    except ImportError:
        return None, None, "C++ core not available"

    config = SimulationConfig(
        n_simulations=n_simulations,
        time_horizon=252,
        risk_free_rate=0.02,
        seed=42
    )

    cpp_config = qss_core.SimulationConfig()
    cpp_config.n_simulations = config.n_simulations
    cpp_config.time_horizon = config.time_horizon
    cpp_config.dt = config.dt
    cpp_config.risk_free_rate = config.risk_free_rate
    cpp_config.seed = config.seed or 0

    cpp_portfolio = qss_core.Portfolio()
    for asset in assets:
        cpp_portfolio.add_asset(
            asset.symbol,
            asset.weight,
            asset.expected_return,
            asset.volatility
        )
    cpp_portfolio.set_covariance_matrix(cov_matrix.tolist())

    # Use parallel engine (multi-threaded)
    cpp_engine = qss_core.ParallelMonteCarloEngine(cpp_config, 0)  # 0 = auto-detect threads

    # Warmup
    _ = cpp_engine.simulate_portfolio(cpp_portfolio)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = cpp_engine.simulate_portfolio(cpp_portfolio)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times), result


def benchmark_cpp_single(assets, cov_matrix, n_simulations, n_runs=3):
    """Benchmark single-threaded C++ implementation."""
    from qss.interface import SimulationConfig

    try:
        from qss import qss_core
    except ImportError:
        return None, None, "C++ core not available"

    config = SimulationConfig(
        n_simulations=n_simulations,
        time_horizon=252,
        risk_free_rate=0.02,
        seed=42
    )

    cpp_config = qss_core.SimulationConfig()
    cpp_config.n_simulations = config.n_simulations
    cpp_config.time_horizon = config.time_horizon
    cpp_config.dt = config.dt
    cpp_config.risk_free_rate = config.risk_free_rate
    cpp_config.seed = config.seed or 0

    cpp_portfolio = qss_core.Portfolio()
    for asset in assets:
        cpp_portfolio.add_asset(
            asset.symbol,
            asset.weight,
            asset.expected_return,
            asset.volatility
        )
    cpp_portfolio.set_covariance_matrix(cov_matrix.tolist())

    cpp_engine = qss_core.MonteCarloEngine(cpp_config)

    # Warmup
    _ = cpp_engine.simulate_portfolio(cpp_portfolio)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = cpp_engine.simulate_portfolio(cpp_portfolio)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times), result


def benchmark_python_vectorized(assets, cov_matrix, n_simulations, n_runs=3):
    """Benchmark vectorized NumPy implementation (optimized)."""
    from qss.interface import SimulationConfig

    config = SimulationConfig(
        n_simulations=n_simulations,
        time_horizon=252,
        risk_free_rate=0.02,
        seed=42
    )

    n_sims = config.n_simulations
    n_steps = config.time_horizon
    dt = config.dt

    weights = np.array([a.weight for a in assets])
    returns = np.array([a.expected_return for a in assets])
    vols = np.array([a.volatility for a in assets])
    L = np.linalg.cholesky(cov_matrix)

    # Warmup
    np.random.seed(config.seed)
    _ = np.random.standard_normal((100, len(assets)))

    times = []
    for _ in range(n_runs):
        np.random.seed(config.seed)
        start = time.perf_counter()

        paths = np.zeros((n_sims, n_steps + 1))
        paths[:, 0] = 1.0

        for t in range(1, n_steps + 1):
            z = np.random.standard_normal((n_sims, len(assets)))
            correlated_z = z @ L.T
            drift = (returns - 0.5 * vols ** 2) * dt
            diffusion = vols * np.sqrt(dt) * correlated_z
            asset_returns = drift + diffusion
            portfolio_return = np.sum(weights * asset_returns, axis=1)
            paths[:, t] = paths[:, t - 1] * np.exp(portfolio_return)

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times), paths[:, -1]


def benchmark_python_naive(assets, cov_matrix, n_simulations, n_runs=1):
    """Benchmark naive Python implementation (no NumPy vectorization)."""
    import random
    import math
    from qss.interface import SimulationConfig

    config = SimulationConfig(
        n_simulations=n_simulations,
        time_horizon=252,
        risk_free_rate=0.02,
        seed=42
    )

    n_sims = config.n_simulations
    n_steps = config.time_horizon
    dt = config.dt
    n_assets = len(assets)

    weights = [a.weight for a in assets]
    returns = [a.expected_return for a in assets]
    vols = [a.volatility for a in assets]

    # Simple Cholesky (for small matrices)
    L = np.linalg.cholesky(cov_matrix).tolist()

    times = []
    for _ in range(n_runs):
        random.seed(config.seed)
        start = time.perf_counter()

        terminal_values = []

        for sim in range(n_sims):
            path_value = 1.0

            for t in range(n_steps):
                # Generate independent normals
                z = [random.gauss(0, 1) for _ in range(n_assets)]

                # Apply Cholesky for correlation
                correlated_z = []
                for i in range(n_assets):
                    val = sum(L[i][j] * z[j] for j in range(i + 1))
                    correlated_z.append(val)

                # Calculate portfolio return
                port_return = 0.0
                for i in range(n_assets):
                    drift = (returns[i] - 0.5 * vols[i] ** 2) * dt
                    diffusion = vols[i] * math.sqrt(dt) * correlated_z[i]
                    port_return += weights[i] * (drift + diffusion)

                path_value *= math.exp(port_return)

            terminal_values.append(path_value)

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times), terminal_values


def run_benchmark():
    """Run comprehensive benchmark."""
    import multiprocessing
    n_cores = multiprocessing.cpu_count()

    print("=" * 75)
    print("QUANTITATIVE SYSTEMS SIMULATOR - PERFORMANCE BENCHMARK")
    print("=" * 75)
    print()
    print(f"System: {n_cores} CPU cores available")
    print()
    print("Implementations tested:")
    print("  1. C++ Parallel    - Multi-threaded C++ Monte Carlo engine")
    print("  2. C++ Single      - Single-threaded C++ implementation")
    print("  3. Python+NumPy    - Vectorized NumPy (highly optimized)")
    print("  4. Python Naive    - Pure Python loops (baseline)")
    print()

    assets, cov_matrix = create_test_portfolio()

    # First, show the naive Python baseline (small scale only)
    print("-" * 75)
    print("BASELINE: Naive Python (pure loops, no NumPy vectorization)")
    print("-" * 75)

    naive_time, _, _ = benchmark_python_naive(assets, cov_matrix, 1000, n_runs=1)
    print(f"  1,000 simulations: {naive_time:.2f}s")

    # Estimate for larger
    estimated_10k = naive_time * 10
    print(f"  10,000 simulations (estimated): {estimated_10k:.1f}s")
    print()

    # Main benchmark
    print("-" * 75)
    print("PERFORMANCE COMPARISON")
    print("-" * 75)

    simulation_counts = [1_000, 10_000, 50_000]

    print(f"{'Sims':>8} | {'C++ Parallel':>14} | {'C++ Single':>14} | {'NumPy':>14} | {'Best':>12}")
    print("-" * 75)

    for n_sims in simulation_counts:
        cpp_par_time, _, _ = benchmark_cpp_parallel(assets, cov_matrix, n_sims)
        cpp_single_time, _, _ = benchmark_cpp_single(assets, cov_matrix, n_sims)
        numpy_time, _, _ = benchmark_python_vectorized(assets, cov_matrix, n_sims)

        times = {
            "C++ Parallel": cpp_par_time,
            "C++ Single": cpp_single_time,
            "NumPy": numpy_time
        }
        best = min(times, key=times.get)

        print(f"{n_sims:>8,} | {cpp_par_time:>12.4f}s | {cpp_single_time:>12.4f}s | "
              f"{numpy_time:>12.4f}s | {best:>12}")

    print("-" * 75)
    print()

    # Large scale comparison
    print("LARGE SCALE TEST (100,000 simulations)")
    print("-" * 75)

    n_large = 100_000

    cpp_par_time, _, _ = benchmark_cpp_parallel(assets, cov_matrix, n_large, n_runs=2)
    cpp_single_time, _, _ = benchmark_cpp_single(assets, cov_matrix, n_large, n_runs=2)
    numpy_time, _, _ = benchmark_python_vectorized(assets, cov_matrix, n_large, n_runs=2)
    naive_estimated = naive_time * 100

    print(f"  C++ Parallel ({n_cores} threads): {cpp_par_time:.2f}s")
    print(f"  C++ Single-threaded:      {cpp_single_time:.2f}s")
    print(f"  Python + NumPy:           {numpy_time:.2f}s")
    print(f"  Python Naive (estimated): {naive_estimated:.1f}s")
    print()

    # Speedup analysis
    print("SPEEDUP vs NAIVE PYTHON")
    print("-" * 75)
    print(f"  C++ Parallel:  {naive_estimated/cpp_par_time:>6.0f}x faster")
    print(f"  C++ Single:    {naive_estimated/cpp_single_time:>6.0f}x faster")
    print(f"  NumPy:         {naive_estimated/numpy_time:>6.0f}x faster")
    print()

    print("=" * 75)
    print("KEY INSIGHTS")
    print("=" * 75)
    print()
    print("1. NumPy is highly optimized (BLAS/LAPACK with SIMD instructions)")
    print("   - Competitive with or faster than naive C++ implementations")
    print()
    print("2. C++ advantages emerge with:")
    print("   - Multi-threading (bypasses Python's GIL)")
    print("   - Complex algorithms not easily vectorized")
    print("   - Memory-constrained environments")
    print("   - Integration with other C++ systems")
    print()
    print("3. The hybrid architecture provides:")
    print("   - Python's ease of use for data handling and visualization")
    print("   - C++ performance for compute-intensive simulation cores")
    print("   - Flexibility to optimize hot paths as needed")
    print("=" * 75)


if __name__ == "__main__":
    run_benchmark()
