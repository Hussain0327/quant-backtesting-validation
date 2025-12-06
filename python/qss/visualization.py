"""
Visualization module - Charts, dashboards, and reporting.

This module creates:
- Return distribution plots
- VaR/CVaR confidence intervals
- Correlation heatmaps
- Time series of simulated paths
- Interactive dashboards
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import numpy as np

# Visualization imports with fallbacks
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class Visualizer:
    """
    Visualization engine for portfolio analysis.

    Supports both matplotlib (static) and plotly (interactive) outputs.
    """

    def __init__(self, backend: str = "matplotlib"):
        """
        Initialize visualizer.

        Parameters
        ----------
        backend : str
            Visualization backend: 'matplotlib' or 'plotly'
        """
        self.backend = backend

        if backend == "matplotlib" and not HAS_MATPLOTLIB:
            raise ImportError("matplotlib not installed")
        if backend == "plotly" and not HAS_PLOTLY:
            raise ImportError("plotly not installed")

    def plot_return_distribution(
        self,
        returns: np.ndarray,
        var_95: Optional[float] = None,
        cvar_95: Optional[float] = None,
        title: str = "Portfolio Return Distribution",
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> Any:
        """
        Plot histogram and KDE of returns with VaR/CVaR lines.

        Parameters
        ----------
        returns : np.ndarray
            Array of portfolio returns
        var_95 : float, optional
            95% Value at Risk
        cvar_95 : float, optional
            95% Conditional VaR
        title : str
            Plot title
        save_path : Path, optional
            Path to save figure
        show : bool
            Whether to display the plot

        Returns
        -------
        figure
            matplotlib Figure or plotly Figure
        """
        if self.backend == "matplotlib":
            return self._plot_distribution_mpl(
                returns, var_95, cvar_95, title, save_path, show
            )
        else:
            return self._plot_distribution_plotly(
                returns, var_95, cvar_95, title, save_path, show
            )

    def _plot_distribution_mpl(
        self, returns, var_95, cvar_95, title, save_path, show
    ):
        """matplotlib implementation."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram with KDE
        if HAS_SEABORN:
            sns.histplot(returns, kde=True, ax=ax, stat="density", alpha=0.7)
        else:
            ax.hist(returns, bins=50, density=True, alpha=0.7, edgecolor='black')

        # Add VaR and CVaR lines
        if var_95 is not None:
            ax.axvline(var_95, color='red', linestyle='--', linewidth=2,
                       label=f'VaR 95%: {var_95:.2%}')
        if cvar_95 is not None:
            ax.axvline(cvar_95, color='darkred', linestyle=':', linewidth=2,
                       label=f'CVaR 95%: {cvar_95:.2%}')

        # Add mean line
        mean = np.mean(returns)
        ax.axvline(mean, color='green', linestyle='-', linewidth=2,
                   label=f'Mean: {mean:.2%}')

        ax.set_xlabel('Return')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()

        return fig

    def _plot_distribution_plotly(
        self, returns, var_95, cvar_95, title, save_path, show
    ):
        """Plotly implementation."""
        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            histnorm='probability density',
            name='Returns',
            opacity=0.7
        ))

        # VaR and CVaR lines
        if var_95 is not None:
            fig.add_vline(x=var_95, line_dash="dash", line_color="red",
                          annotation_text=f"VaR 95%: {var_95:.2%}")
        if cvar_95 is not None:
            fig.add_vline(x=cvar_95, line_dash="dot", line_color="darkred",
                          annotation_text=f"CVaR 95%: {cvar_95:.2%}")

        # Mean line
        mean = np.mean(returns)
        fig.add_vline(x=mean, line_color="green",
                      annotation_text=f"Mean: {mean:.2%}")

        fig.update_layout(
            title=title,
            xaxis_title="Return",
            yaxis_title="Density",
            showlegend=True
        )

        if save_path:
            fig.write_html(str(save_path))
        if show:
            fig.show()

        return fig

    def plot_simulation_paths(
        self,
        paths: np.ndarray,
        n_paths: int = 100,
        highlight_percentiles: bool = True,
        title: str = "Monte Carlo Simulation Paths",
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> Any:
        """
        Plot sample simulation paths with percentile bands.

        Parameters
        ----------
        paths : np.ndarray
            Simulated paths array (n_simulations x n_steps)
        n_paths : int
            Number of paths to display
        highlight_percentiles : bool
            Whether to show percentile bands
        title : str
            Plot title
        save_path : Path, optional
            Path to save figure
        show : bool
            Whether to display

        Returns
        -------
        figure
            matplotlib or plotly figure
        """
        if self.backend == "matplotlib":
            return self._plot_paths_mpl(
                paths, n_paths, highlight_percentiles, title, save_path, show
            )
        else:
            return self._plot_paths_plotly(
                paths, n_paths, highlight_percentiles, title, save_path, show
            )

    def _plot_paths_mpl(
        self, paths, n_paths, highlight_percentiles, title, save_path, show
    ):
        """matplotlib implementation."""
        fig, ax = plt.subplots(figsize=(12, 6))

        n_steps = paths.shape[1]
        x = np.arange(n_steps)

        # Sample paths
        sample_indices = np.random.choice(len(paths), min(n_paths, len(paths)), replace=False)
        for idx in sample_indices:
            ax.plot(x, paths[idx], alpha=0.1, color='blue', linewidth=0.5)

        # Percentile bands
        if highlight_percentiles:
            p5 = np.percentile(paths, 5, axis=0)
            p25 = np.percentile(paths, 25, axis=0)
            p50 = np.percentile(paths, 50, axis=0)
            p75 = np.percentile(paths, 75, axis=0)
            p95 = np.percentile(paths, 95, axis=0)

            ax.fill_between(x, p5, p95, alpha=0.2, color='blue', label='5-95%')
            ax.fill_between(x, p25, p75, alpha=0.3, color='blue', label='25-75%')
            ax.plot(x, p50, color='darkblue', linewidth=2, label='Median')

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Portfolio Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()

        return fig

    def _plot_paths_plotly(
        self, paths, n_paths, highlight_percentiles, title, save_path, show
    ):
        """Plotly implementation."""
        fig = go.Figure()

        n_steps = paths.shape[1]
        x = list(range(n_steps))

        # Sample paths
        sample_indices = np.random.choice(len(paths), min(n_paths, len(paths)), replace=False)
        for idx in sample_indices:
            fig.add_trace(go.Scatter(
                x=x, y=paths[idx],
                mode='lines',
                line=dict(width=0.5, color='rgba(0,0,255,0.1)'),
                showlegend=False
            ))

        # Percentile bands
        if highlight_percentiles:
            p5 = np.percentile(paths, 5, axis=0)
            p50 = np.percentile(paths, 50, axis=0)
            p95 = np.percentile(paths, 95, axis=0)

            fig.add_trace(go.Scatter(
                x=x, y=p95, mode='lines',
                line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=x, y=p5, mode='lines',
                fill='tonexty', fillcolor='rgba(0,0,255,0.2)',
                line=dict(width=0), name='5-95%'
            ))
            fig.add_trace(go.Scatter(
                x=x, y=p50, mode='lines',
                line=dict(width=2, color='darkblue'),
                name='Median'
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Time Step",
            yaxis_title="Portfolio Value"
        )

        if save_path:
            fig.write_html(str(save_path))
        if show:
            fig.show()

        return fig

    def plot_correlation_heatmap(
        self,
        correlation_matrix: np.ndarray,
        labels: List[str],
        title: str = "Asset Correlation Matrix",
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> Any:
        """
        Plot correlation heatmap.

        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix
        labels : List[str]
            Asset labels
        title : str
            Plot title
        save_path : Path, optional
            Save path
        show : bool
            Whether to display

        Returns
        -------
        figure
        """
        if self.backend == "matplotlib":
            fig, ax = plt.subplots(figsize=(8, 6))

            if HAS_SEABORN:
                sns.heatmap(
                    correlation_matrix,
                    annot=True,
                    fmt='.2f',
                    xticklabels=labels,
                    yticklabels=labels,
                    cmap='RdBu_r',
                    center=0,
                    ax=ax
                )
            else:
                im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)
                plt.colorbar(im, ax=ax)

            ax.set_title(title)

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
            if show:
                plt.show()

            return fig
        else:
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix,
                x=labels,
                y=labels,
                colorscale='RdBu_r',
                zmid=0,
                text=np.round(correlation_matrix, 2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))

            fig.update_layout(title=title)

            if save_path:
                fig.write_html(str(save_path))
            if show:
                fig.show()

            return fig

    def plot_var_analysis(
        self,
        returns: np.ndarray,
        confidence_levels: List[float] = [0.90, 0.95, 0.99],
        title: str = "Value at Risk Analysis",
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> Any:
        """
        Plot VaR at multiple confidence levels.
        """
        var_values = [np.percentile(returns, (1 - cl) * 100) for cl in confidence_levels]

        if self.backend == "matplotlib":
            fig, ax = plt.subplots(figsize=(10, 6))

            # Distribution
            ax.hist(returns, bins=50, density=True, alpha=0.7, edgecolor='black')

            colors = ['green', 'orange', 'red']
            for var, cl, color in zip(var_values, confidence_levels, colors):
                ax.axvline(var, color=color, linestyle='--', linewidth=2,
                           label=f'VaR {cl:.0%}: {var:.2%}')

            ax.set_xlabel('Return')
            ax.set_ylabel('Density')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
            if show:
                plt.show()

            return fig
        else:
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=returns, nbinsx=50, histnorm='probability density',
                name='Returns', opacity=0.7
            ))

            colors = ['green', 'orange', 'red']
            for var, cl, color in zip(var_values, confidence_levels, colors):
                fig.add_vline(
                    x=var, line_dash="dash", line_color=color,
                    annotation_text=f"VaR {cl:.0%}: {var:.2%}"
                )

            fig.update_layout(
                title=title,
                xaxis_title="Return",
                yaxis_title="Density"
            )

            if save_path:
                fig.write_html(str(save_path))
            if show:
                fig.show()

            return fig

    def create_dashboard(
        self,
        simulation_result: Any,
        assets: List[Any],
        output_dir: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Dict[str, Any]:
        """
        Create comprehensive visualization dashboard.

        Parameters
        ----------
        simulation_result : SimulationResult
            Results from Monte Carlo simulation
        assets : List[AssetData]
            Portfolio assets
        output_dir : str or Path, optional
            Directory to save figures
        show : bool
            Whether to display figures

        Returns
        -------
        dict
            Dictionary of figure objects
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        figures = {}
        returns = simulation_result.terminal_values - 1.0
        metrics = simulation_result.metrics

        # 1. Return Distribution
        figures["distribution"] = self.plot_return_distribution(
            returns,
            var_95=metrics.var_95,
            cvar_95=metrics.cvar_95,
            save_path=output_dir / "distribution.png" if output_dir else None,
            show=show
        )

        # 2. Simulation Paths
        figures["paths"] = self.plot_simulation_paths(
            simulation_result.simulated_paths,
            n_paths=100,
            save_path=output_dir / "paths.png" if output_dir else None,
            show=show
        )

        # 3. VaR Analysis
        figures["var_analysis"] = self.plot_var_analysis(
            returns,
            save_path=output_dir / "var_analysis.png" if output_dir else None,
            show=show
        )

        return figures

    def plot_convergence(
        self,
        terminal_values: np.ndarray,
        metric: str = "mean",
        title: str = "Convergence Analysis",
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> Any:
        """
        Plot convergence of Monte Carlo estimates.

        Parameters
        ----------
        terminal_values : np.ndarray
            Terminal values from simulation
        metric : str
            Metric to track ('mean', 'var', 'var_95')
        title : str
            Plot title
        save_path : Path, optional
            Save path
        show : bool
            Whether to display

        Returns
        -------
        figure
        """
        n = len(terminal_values)
        sample_sizes = np.logspace(2, np.log10(n), 50).astype(int)

        estimates = []
        for size in sample_sizes:
            sample = terminal_values[:size]
            if metric == "mean":
                estimates.append(np.mean(sample))
            elif metric == "var":
                estimates.append(np.std(sample))
            elif metric == "var_95":
                estimates.append(np.percentile(sample, 5))

        if self.backend == "matplotlib":
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(sample_sizes, estimates, 'b-', linewidth=2)
            ax.axhline(estimates[-1], color='r', linestyle='--',
                       label=f'Final estimate: {estimates[-1]:.4f}')
            ax.set_xscale('log')
            ax.set_xlabel('Number of Simulations')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
            if show:
                plt.show()

            return fig
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sample_sizes, y=estimates, mode='lines',
                name='Estimate'
            ))
            fig.add_hline(
                y=estimates[-1], line_dash="dash", line_color="red",
                annotation_text=f"Final: {estimates[-1]:.4f}"
            )
            fig.update_xaxes(type="log")
            fig.update_layout(
                title=title,
                xaxis_title="Number of Simulations",
                yaxis_title=metric.capitalize()
            )

            if save_path:
                fig.write_html(str(save_path))
            if show:
                fig.show()

            return fig
