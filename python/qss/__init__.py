"""
Quantitative Systems Simulator (QSS)
=====================================

A Python + C++ hybrid system for portfolio risk analysis,
Monte Carlo simulation, and statistical analytics.

Modules:
--------
- orchestrator: Main data pipeline and workflow coordination
- analytics: Statistical analysis and metrics computation
- visualization: Charts, dashboards, and reporting
- interface: Python-C++ bridge for simulation engine

Example Usage:
--------------
>>> from qss import Orchestrator
>>> orch = Orchestrator()
>>> orch.load_portfolio("data/sample_portfolio.csv")
>>> results = orch.run_simulation(n_simulations=10000)
>>> orch.generate_report("output/report.json")
"""

__version__ = "0.1.0"
__author__ = "Raja Hussain"

from .orchestrator import Orchestrator
from .analytics import PortfolioAnalytics, RiskMetrics
from .visualization import Visualizer
from .interface import SimulationInterface

__all__ = [
    "Orchestrator",
    "PortfolioAnalytics",
    "RiskMetrics",
    "Visualizer",
    "SimulationInterface",
]
