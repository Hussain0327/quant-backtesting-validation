"""
Interface module for Python-C++ communication.

This module provides a Python wrapper around the C++ simulation core,
handling data conversion and providing a Pythonic API.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

# Try to import the C++ module, fall back to pure Python implementation
try:
    # Try relative import first (when module is in qss package)
    from . import qss_core
    HAS_CPP_CORE = True
except ImportError:
    try:
        # Try top-level import (when installed as separate package)
        import qss_core
        HAS_CPP_CORE = True
    except ImportError:
        HAS_CPP_CORE = False
        print("Warning: C++ core not available. Using pure Python implementation.")


@dataclass
class AssetData:
    """Represents a single asset in the portfolio."""
    symbol: str
    weight: float
    expected_return: float
    volatility: float


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""
    n_simulations: int = 10000
    time_horizon: int = 252  # Trading days
    dt: float = 1.0 / 252.0
    risk_free_rate: float = 0.02
    use_antithetic: bool = False
    use_stratified: bool = False
    seed: Optional[int] = None


@dataclass
class PortfolioMetrics:
    """Risk and return metrics for a portfolio."""
    expected_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    max_drawdown: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "max_drawdown": self.max_drawdown,
        }


@dataclass
class SimulationResult:
    """Results from a Monte Carlo simulation."""
    simulated_paths: np.ndarray = field(default_factory=lambda: np.array([]))
    terminal_values: np.ndarray = field(default_factory=lambda: np.array([]))
    max_drawdowns: np.ndarray = field(default_factory=lambda: np.array([]))
    metrics: PortfolioMetrics = field(default_factory=PortfolioMetrics)
    convergence_error: float = 0.0


class SimulationInterface:
    """
    Interface to the C++ Monte Carlo simulation engine.

    Falls back to a pure Python implementation if C++ core is not available.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize the simulation interface."""
        self.config = config or SimulationConfig()
        self._cpp_engine = None
        self._cpp_portfolio = None

        if HAS_CPP_CORE:
            self._init_cpp_engine()

    def _init_cpp_engine(self):
        """Initialize C++ engine if available."""
        cpp_config = qss_core.SimulationConfig()
        cpp_config.n_simulations = self.config.n_simulations
        cpp_config.time_horizon = self.config.time_horizon
        cpp_config.dt = self.config.dt
        cpp_config.risk_free_rate = self.config.risk_free_rate
        cpp_config.use_antithetic = self.config.use_antithetic
        cpp_config.use_stratified = self.config.use_stratified
        cpp_config.seed = self.config.seed or 0

        self._cpp_engine = qss_core.MonteCarloEngine(cpp_config)

    def create_portfolio(
        self,
        assets: List[AssetData],
        covariance_matrix: np.ndarray
    ) -> None:
        """
        Create a portfolio from asset data.

        Parameters
        ----------
        assets : List[AssetData]
            List of assets with their properties
        covariance_matrix : np.ndarray
            Covariance matrix of asset returns
        """
        if HAS_CPP_CORE:
            self._cpp_portfolio = qss_core.Portfolio()
            for asset in assets:
                self._cpp_portfolio.add_asset(
                    asset.symbol,
                    asset.weight,
                    asset.expected_return,
                    asset.volatility
                )
            self._cpp_portfolio.set_covariance_matrix(covariance_matrix.tolist())
        else:
            self._py_assets = assets
            self._py_cov_matrix = covariance_matrix

    def run_simulation(self) -> SimulationResult:
        """
        Run Monte Carlo simulation on the portfolio.

        Returns
        -------
        SimulationResult
            Simulation results including paths and metrics
        """
        if HAS_CPP_CORE:
            return self._run_cpp_simulation()
        else:
            return self._run_python_simulation()

    def _run_cpp_simulation(self) -> SimulationResult:
        """Run simulation using C++ engine."""
        if self._cpp_portfolio is None:
            raise ValueError("Portfolio not created. Call create_portfolio first.")

        cpp_result = self._cpp_engine.simulate_portfolio(self._cpp_portfolio)

        # Convert C++ result to Python dataclass
        metrics = PortfolioMetrics(
            expected_return=cpp_result.metrics.expected_return,
            volatility=cpp_result.metrics.volatility,
            sharpe_ratio=cpp_result.metrics.sharpe_ratio,
            var_95=cpp_result.metrics.var_95,
            var_99=cpp_result.metrics.var_99,
            cvar_95=cpp_result.metrics.cvar_95,
            cvar_99=cpp_result.metrics.cvar_99,
            skewness=cpp_result.metrics.skewness,
            kurtosis=cpp_result.metrics.kurtosis,
            max_drawdown=cpp_result.metrics.max_drawdown,
        )

        return SimulationResult(
            simulated_paths=np.array(cpp_result.simulated_returns),
            terminal_values=np.array(cpp_result.terminal_values),
            max_drawdowns=np.array(cpp_result.max_drawdowns),
            metrics=metrics,
            convergence_error=cpp_result.convergence_error,
        )

    def _run_python_simulation(self) -> SimulationResult:
        """Pure Python fallback for Monte Carlo simulation."""
        n_sims = self.config.n_simulations
        n_steps = self.config.time_horizon
        dt = self.config.dt

        # Get portfolio parameters
        weights = np.array([a.weight for a in self._py_assets])
        returns = np.array([a.expected_return for a in self._py_assets])
        vols = np.array([a.volatility for a in self._py_assets])
        cov = self._py_cov_matrix

        # Cholesky decomposition for correlated samples
        L = np.linalg.cholesky(cov)

        # Initialize paths
        paths = np.zeros((n_sims, n_steps + 1))
        paths[:, 0] = 1.0  # Start with unit value

        np.random.seed(self.config.seed)

        for t in range(1, n_steps + 1):
            # Generate correlated random returns
            z = np.random.standard_normal((n_sims, len(self._py_assets)))
            correlated_z = z @ L.T

            # Calculate portfolio return
            drift = (returns - 0.5 * vols ** 2) * dt
            diffusion = vols * np.sqrt(dt) * correlated_z
            asset_returns = drift + diffusion
            portfolio_return = np.sum(weights * asset_returns, axis=1)

            paths[:, t] = paths[:, t - 1] * np.exp(portfolio_return)

        # Calculate metrics
        terminal_values = paths[:, -1]
        total_returns = terminal_values - 1.0

        # Max drawdowns
        max_drawdowns = np.zeros(n_sims)
        for i in range(n_sims):
            peak = paths[i, 0]
            max_dd = 0.0
            for v in paths[i]:
                if v > peak:
                    peak = v
                dd = (peak - v) / peak
                if dd > max_dd:
                    max_dd = dd
            max_drawdowns[i] = max_dd

        # Compute metrics
        sorted_returns = np.sort(total_returns)
        var_95 = sorted_returns[int(0.05 * n_sims)]
        var_99 = sorted_returns[int(0.01 * n_sims)]
        cvar_95 = np.mean(sorted_returns[sorted_returns <= var_95])
        cvar_99 = np.mean(sorted_returns[sorted_returns <= var_99])

        mean_ret = np.mean(total_returns)
        std_ret = np.std(total_returns, ddof=1)
        sharpe = (mean_ret - self.config.risk_free_rate) / std_ret if std_ret > 0 else 0

        # Skewness and kurtosis
        if std_ret > 0:
            skew = np.mean(((total_returns - mean_ret) / std_ret) ** 3)
            kurt = np.mean(((total_returns - mean_ret) / std_ret) ** 4) - 3.0
        else:
            skew = 0.0
            kurt = 0.0

        metrics = PortfolioMetrics(
            expected_return=mean_ret,
            volatility=std_ret,
            sharpe_ratio=sharpe,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            skewness=skew,
            kurtosis=kurt,
            max_drawdown=np.mean(max_drawdowns),
        )

        convergence_error = std_ret / np.sqrt(n_sims)

        return SimulationResult(
            simulated_paths=paths,
            terminal_values=terminal_values,
            max_drawdowns=max_drawdowns,
            metrics=metrics,
            convergence_error=convergence_error,
        )

    def simulate_gbm(
        self,
        S0: float,
        mu: float,
        sigma: float
    ) -> SimulationResult:
        """
        Simulate Geometric Brownian Motion for a single asset.

        Parameters
        ----------
        S0 : float
            Initial price
        mu : float
            Expected return (drift)
        sigma : float
            Volatility

        Returns
        -------
        SimulationResult
            Simulation results
        """
        if HAS_CPP_CORE:
            cpp_result = self._cpp_engine.simulate_gbm(S0, mu, sigma)

            metrics = PortfolioMetrics(
                expected_return=cpp_result.metrics.expected_return,
                volatility=cpp_result.metrics.volatility,
                sharpe_ratio=cpp_result.metrics.sharpe_ratio,
                var_95=cpp_result.metrics.var_95,
                var_99=cpp_result.metrics.var_99,
                cvar_95=cpp_result.metrics.cvar_95,
                cvar_99=cpp_result.metrics.cvar_99,
                skewness=cpp_result.metrics.skewness,
                kurtosis=cpp_result.metrics.kurtosis,
                max_drawdown=cpp_result.metrics.max_drawdown,
            )

            return SimulationResult(
                simulated_paths=np.array(cpp_result.simulated_returns),
                terminal_values=np.array(cpp_result.terminal_values),
                max_drawdowns=np.array(cpp_result.max_drawdowns),
                metrics=metrics,
                convergence_error=cpp_result.convergence_error,
            )
        else:
            # Pure Python GBM
            n_sims = self.config.n_simulations
            n_steps = self.config.time_horizon
            dt = self.config.dt

            np.random.seed(self.config.seed)

            paths = np.zeros((n_sims, n_steps + 1))
            paths[:, 0] = S0

            drift = (mu - 0.5 * sigma ** 2) * dt
            vol = sigma * np.sqrt(dt)

            for t in range(1, n_steps + 1):
                z = np.random.standard_normal(n_sims)
                paths[:, t] = paths[:, t - 1] * np.exp(drift + vol * z)

            terminal_values = paths[:, -1]
            returns = (terminal_values - S0) / S0

            sorted_returns = np.sort(returns)
            var_95 = sorted_returns[int(0.05 * n_sims)]
            var_99 = sorted_returns[int(0.01 * n_sims)]

            metrics = PortfolioMetrics(
                expected_return=np.mean(returns),
                volatility=np.std(returns, ddof=1),
                var_95=var_95,
                var_99=var_99,
            )

            return SimulationResult(
                simulated_paths=paths,
                terminal_values=terminal_values,
                metrics=metrics,
            )


def get_stats_module():
    """
    Get the statistics module (C++ or Python fallback).

    Returns
    -------
    module
        Statistics module with hypothesis testing and CI functions
    """
    if HAS_CPP_CORE:
        return qss_core.stats
    else:
        # Return scipy.stats as fallback
        from scipy import stats
        return stats
