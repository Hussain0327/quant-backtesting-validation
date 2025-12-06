"""
Orchestrator module - Main data pipeline and workflow coordination.

This module handles:
- Reading and cleaning portfolio data
- Coordinating simulation runs
- Managing output generation
"""

from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import pandas as pd

from .interface import (
    SimulationInterface,
    SimulationConfig,
    SimulationResult,
    AssetData,
    PortfolioMetrics,
)
from .analytics import PortfolioAnalytics, RiskMetrics
from .visualization import Visualizer


class Orchestrator:
    """
    Main orchestrator for the Quantitative Systems Simulator.

    Coordinates data ingestion, simulation, analysis, and reporting.

    Example
    -------
    >>> orch = Orchestrator()
    >>> orch.load_portfolio("data/portfolio.csv")
    >>> results = orch.run_simulation(n_simulations=10000)
    >>> orch.generate_report("output/report.json")
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize the orchestrator.

        Parameters
        ----------
        config : SimulationConfig, optional
            Simulation configuration. If None, uses defaults.
        """
        self.config = config or SimulationConfig()
        self.interface = SimulationInterface(self.config)
        self.analytics = PortfolioAnalytics()
        self.visualizer = Visualizer()

        self._assets: List[AssetData] = []
        self._covariance_matrix: Optional[np.ndarray] = None
        self._returns_data: Optional[pd.DataFrame] = None
        self._last_result: Optional[SimulationResult] = None

    def load_portfolio(
        self,
        filepath: Union[str, Path],
        returns_filepath: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Load portfolio data from CSV file.

        Parameters
        ----------
        filepath : str or Path
            Path to portfolio CSV with columns:
            symbol, weight, expected_return, volatility
        returns_filepath : str or Path, optional
            Path to historical returns CSV for covariance estimation
        """
        filepath = Path(filepath)
        df = pd.read_csv(filepath)

        required_cols = ["symbol", "weight"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Portfolio file must contain '{col}' column")

        self._assets = []
        for _, row in df.iterrows():
            asset = AssetData(
                symbol=row["symbol"],
                weight=row["weight"],
                expected_return=row.get("expected_return", 0.0),
                volatility=row.get("volatility", 0.0),
            )
            self._assets.append(asset)

        # Normalize weights
        total_weight = sum(a.weight for a in self._assets)
        for asset in self._assets:
            asset.weight /= total_weight

        # Load or compute covariance matrix
        if returns_filepath:
            self._load_returns(Path(returns_filepath))
            self._compute_covariance()
        elif "covariance" in df.columns or self._covariance_matrix is not None:
            pass  # Use provided covariance
        else:
            # Generate identity covariance scaled by volatilities
            n = len(self._assets)
            vols = np.array([a.volatility for a in self._assets])
            self._covariance_matrix = np.diag(vols ** 2)

    def load_portfolio_from_dict(
        self,
        portfolio_data: Dict[str, Any],
        covariance_matrix: Optional[np.ndarray] = None
    ) -> None:
        """
        Load portfolio from dictionary.

        Parameters
        ----------
        portfolio_data : dict
            Dictionary with 'assets' key containing list of asset dicts
        covariance_matrix : np.ndarray, optional
            Covariance matrix of returns
        """
        self._assets = []
        for asset_dict in portfolio_data.get("assets", []):
            asset = AssetData(
                symbol=asset_dict["symbol"],
                weight=asset_dict["weight"],
                expected_return=asset_dict.get("expected_return", 0.0),
                volatility=asset_dict.get("volatility", 0.0),
            )
            self._assets.append(asset)

        # Normalize weights
        total_weight = sum(a.weight for a in self._assets)
        for asset in self._assets:
            asset.weight /= total_weight

        if covariance_matrix is not None:
            self._covariance_matrix = covariance_matrix
        else:
            n = len(self._assets)
            vols = np.array([a.volatility for a in self._assets])
            self._covariance_matrix = np.diag(vols ** 2)

    def _load_returns(self, filepath: Path) -> None:
        """Load historical returns data."""
        self._returns_data = pd.read_csv(filepath, index_col=0, parse_dates=True)

    def _compute_covariance(self) -> None:
        """Compute covariance matrix from returns data."""
        if self._returns_data is None:
            raise ValueError("Returns data not loaded")

        # Filter to portfolio assets
        symbols = [a.symbol for a in self._assets]
        available = [s for s in symbols if s in self._returns_data.columns]

        if len(available) < len(symbols):
            missing = set(symbols) - set(available)
            print(f"Warning: Missing return data for {missing}")

        returns = self._returns_data[available]
        self._covariance_matrix = returns.cov().values

        # Update asset volatilities and expected returns from data
        for i, symbol in enumerate(available):
            for asset in self._assets:
                if asset.symbol == symbol:
                    asset.volatility = returns[symbol].std() * np.sqrt(252)
                    asset.expected_return = returns[symbol].mean() * 252

    def set_covariance_matrix(self, cov_matrix: np.ndarray) -> None:
        """
        Set covariance matrix directly.

        Parameters
        ----------
        cov_matrix : np.ndarray
            Covariance matrix of asset returns
        """
        if cov_matrix.shape[0] != len(self._assets):
            raise ValueError("Covariance matrix size must match number of assets")
        self._covariance_matrix = cov_matrix

    def run_simulation(
        self,
        n_simulations: Optional[int] = None,
        **kwargs
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation on the portfolio.

        Parameters
        ----------
        n_simulations : int, optional
            Number of simulations. Overrides config if provided.
        **kwargs
            Additional simulation parameters

        Returns
        -------
        SimulationResult
            Simulation results with paths and metrics
        """
        if not self._assets:
            raise ValueError("Portfolio not loaded. Call load_portfolio first.")

        if n_simulations:
            self.config.n_simulations = n_simulations

        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Recreate interface with updated config
        self.interface = SimulationInterface(self.config)
        self.interface.create_portfolio(self._assets, self._covariance_matrix)

        self._last_result = self.interface.run_simulation()
        return self._last_result

    def get_metrics(self) -> PortfolioMetrics:
        """
        Get portfolio metrics from last simulation.

        Returns
        -------
        PortfolioMetrics
            Risk and return metrics
        """
        if self._last_result is None:
            raise ValueError("No simulation run yet. Call run_simulation first.")
        return self._last_result.metrics

    def compute_analytics(self) -> Dict[str, Any]:
        """
        Compute extended analytics on simulation results.

        Returns
        -------
        dict
            Extended analytics including confidence intervals,
            hypothesis tests, and distribution analysis
        """
        if self._last_result is None:
            raise ValueError("No simulation run yet. Call run_simulation first.")

        returns = self._last_result.terminal_values - 1.0

        analytics = {
            "basic_metrics": self._last_result.metrics.to_dict(),
            "confidence_intervals": self.analytics.compute_confidence_intervals(returns),
            "distribution_analysis": self.analytics.analyze_distribution(returns),
            "tail_analysis": self.analytics.analyze_tails(returns),
        }

        return analytics

    def generate_report(
        self,
        output_path: Union[str, Path],
        format: str = "json"
    ) -> None:
        """
        Generate analysis report.

        Parameters
        ----------
        output_path : str or Path
            Output file path
        format : str
            Output format: 'json', 'html', or 'excel'
        """
        if self._last_result is None:
            raise ValueError("No simulation run yet. Call run_simulation first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        analytics = self.compute_analytics()

        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "n_simulations": self.config.n_simulations,
                "time_horizon": self.config.time_horizon,
                "risk_free_rate": self.config.risk_free_rate,
            },
            "portfolio": {
                "assets": [
                    {
                        "symbol": a.symbol,
                        "weight": a.weight,
                        "expected_return": a.expected_return,
                        "volatility": a.volatility,
                    }
                    for a in self._assets
                ],
            },
            "simulation_results": {
                "convergence_error": self._last_result.convergence_error,
            },
            "analytics": analytics,
        }

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

        elif format == "html":
            self._generate_html_report(report, output_path)

        elif format == "excel":
            self._generate_excel_report(report, output_path)

    def _generate_html_report(self, report: Dict, output_path: Path) -> None:
        """Generate HTML dashboard report."""
        # Create visualizations
        figures = self.visualizer.create_dashboard(
            self._last_result,
            self._assets
        )

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Portfolio Risk Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px;
                   background: #f0f0f0; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .metric-label {{ font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <h1>Portfolio Risk Analysis Report</h1>
    <p>Generated: {report['metadata']['generated_at']}</p>

    <h2>Key Metrics</h2>
    <div class="metrics">
        <div class="metric">
            <div class="metric-value">{report['analytics']['basic_metrics']['expected_return']:.2%}</div>
            <div class="metric-label">Expected Return</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report['analytics']['basic_metrics']['volatility']:.2%}</div>
            <div class="metric-label">Volatility</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report['analytics']['basic_metrics']['sharpe_ratio']:.2f}</div>
            <div class="metric-label">Sharpe Ratio</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report['analytics']['basic_metrics']['var_95']:.2%}</div>
            <div class="metric-label">VaR (95%)</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report['analytics']['basic_metrics']['cvar_95']:.2%}</div>
            <div class="metric-label">CVaR (95%)</div>
        </div>
    </div>

    <h2>Portfolio Composition</h2>
    <table border="1" cellpadding="5">
        <tr><th>Symbol</th><th>Weight</th><th>Expected Return</th><th>Volatility</th></tr>
        {''.join(f"<tr><td>{a['symbol']}</td><td>{a['weight']:.2%}</td><td>{a['expected_return']:.2%}</td><td>{a['volatility']:.2%}</td></tr>" for a in report['portfolio']['assets'])}
    </table>
</body>
</html>
"""
        with open(output_path, "w") as f:
            f.write(html_content)

    def _generate_excel_report(self, report: Dict, output_path: Path) -> None:
        """Generate Excel report."""
        with pd.ExcelWriter(output_path) as writer:
            # Metrics sheet
            metrics_df = pd.DataFrame([report["analytics"]["basic_metrics"]])
            metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

            # Portfolio sheet
            portfolio_df = pd.DataFrame(report["portfolio"]["assets"])
            portfolio_df.to_excel(writer, sheet_name="Portfolio", index=False)

            # Simulation paths (sample)
            if self._last_result is not None:
                n_samples = min(100, len(self._last_result.simulated_paths))
                paths_df = pd.DataFrame(
                    self._last_result.simulated_paths[:n_samples].T
                )
                paths_df.to_excel(writer, sheet_name="Sample Paths", index=True)

    def create_visualizations(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Dict[str, Any]:
        """
        Create visualization charts.

        Parameters
        ----------
        output_dir : str or Path, optional
            Directory to save figures
        show : bool
            Whether to display figures

        Returns
        -------
        dict
            Dictionary of figure objects
        """
        if self._last_result is None:
            raise ValueError("No simulation run yet. Call run_simulation first.")

        return self.visualizer.create_dashboard(
            self._last_result,
            self._assets,
            output_dir=output_dir,
            show=show
        )

    @property
    def portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        if not self._assets:
            return {}

        return {
            "n_assets": len(self._assets),
            "assets": [a.symbol for a in self._assets],
            "weights": {a.symbol: a.weight for a in self._assets},
            "total_expected_return": sum(a.weight * a.expected_return for a in self._assets),
            "weighted_volatility": sum(a.weight * a.volatility for a in self._assets),
        }
