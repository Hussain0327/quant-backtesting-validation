# Quantitative Systems Simulator

**Portfolio Risk Analysis | Monte Carlo Simulation | Statistical Analytics**

*A hybrid Python + C++ system for quantitative research and portfolio risk management*

---

## Overview

The **Quantitative Systems Simulator (QSS)** is a high-performance portfolio risk analysis engine that combines:

- **C++17** for computationally intensive Monte Carlo simulations
- **Python** for data orchestration, visualization, and reporting
- **pybind11** for seamless Python-C++ integration

This architecture demonstrates real-world quantitative finance systems where performance-critical code runs in C++ while maintaining Python's ease of use for data science workflows.

---

## System Architecture

```
+-------------------------------------------------------------------------+
|                         USER INPUT LAYER                                 |
|                  (Portfolio CSV / API Data / Returns)                    |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                      PYTHON INGESTION LAYER                              |
|  +-----------+  +-----------+  +-----------+  +-----------+             |
|  |  pandas   |  |  Data     |  | Portfolio |  |  Config   |             |
|  |  Reader   |  |  Cleaning |  | Validation|  |  Setup    |             |
|  +-----------+  +-----------+  +-----------+  +-----------+             |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                    C++ SIMULATION CORE (pybind11)                        |
|  +-------------------------------------------------------------------+  |
|  |                   Monte Carlo Engine                               |  |
|  |  +-----------+  +-----------+  +-----------+                      |  |
|  |  |  Random   |  |  Cholesky |  |  GBM Path |                      |  |
|  |  |  Generator|  |  Decomp   |  | Simulation|                      |  |
|  |  +-----------+  +-----------+  +-----------+                      |  |
|  |  +-----------+  +-----------+  +-----------+                      |  |
|  |  | Variance  |  |  Multi-   |  |  Risk     |                      |  |
|  |  | Reduction |  |  Threading|  |  Metrics  |                      |  |
|  |  +-----------+  +-----------+  +-----------+                      |  |
|  +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                   STATISTICAL ANALYSIS LAYER                             |
|  +-----------+  +-----------+  +-----------+  +-----------+             |
|  | VaR/CVaR  |  | Hypothesis|  | Confidence|  |Distribution|            |
|  |Calculation|  |  Testing  |  | Intervals |  |  Fitting   |            |
|  +-----------+  +-----------+  +-----------+  +-----------+             |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                    VISUALIZATION & REPORTING                             |
|  +-----------+  +-----------+  +-----------+  +-----------+             |
|  |Distribution|  |  Path     |  |Correlation|  | JSON/HTML |            |
|  |   Plots   |  | Simulation|  |  Heatmaps |  |  Reports  |            |
|  +-----------+  +-----------+  +-----------+  +-----------+             |
+-------------------------------------------------------------------------+
```

---

## Performance Benchmark

The C++ core provides significant performance improvements over pure Python:

```
===========================================================================
QUANTITATIVE SYSTEMS SIMULATOR - PERFORMANCE BENCHMARK
===========================================================================

System: 8 CPU cores available

---------------------------------------------------------------------------
PERFORMANCE COMPARISON (100,000 simulations)
---------------------------------------------------------------------------
  C++ Parallel (8 threads):  1.50s
  C++ Single-threaded:       6.59s
  Python + NumPy:            4.91s
  Python Naive (loops):    177.00s

SPEEDUP vs NAIVE PYTHON
---------------------------------------------------------------------------
  C++ Parallel:   118x faster
  C++ Single:      27x faster
  NumPy:           36x faster
===========================================================================
```

**Key Insight**: Multi-threaded C++ bypasses Python's GIL, achieving **3.3x speedup over NumPy** and **118x over naive Python**.

---

## Getting Started

### Prerequisites

- Python 3.13 (I'm using a virtual environment)
- CMake 3.14+
- A C++17 compatible compiler (clang on macOS)
- pybind11

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quantitative-systems-simulator.git
cd quantitative-systems-simulator

# Create and activate virtual environment
python3.13 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Build the C++ extension
cd cpp_core
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR=$(python -m pybind11 --cmakedir) \
  -DPYBIND11_FINDPYTHON=ON \
  -DPython_EXECUTABLE=$(which python)
make -j4

# Copy the compiled module to the Python package
cp qss_core*.so ../../python/qss/
cd ../..
```

### Running the Demo

```bash
source venv/bin/activate
python main.py
```

### Running the Benchmark

```bash
python benchmark.py
```

### Jupyter Notebook

The notebook in `notebooks/simulation_analysis.ipynb` walks through the entire workflow with explanations.

---

## Features

### Monte Carlo Engine (C++)

| Feature | Description |
|---------|-------------|
| **Multi-threaded Simulation** | Parallel execution across all CPU cores |
| **Correlated Returns** | Cholesky decomposition for asset correlation |
| **Variance Reduction** | Antithetic and stratified sampling |
| **GBM Paths** | Geometric Brownian Motion simulation |
| **Risk Metrics** | VaR, CVaR, Sharpe, Max Drawdown |

### Statistical Analysis (Python)

| Feature | Description |
|---------|-------------|
| **Hypothesis Testing** | t-tests, Jarque-Bera, Shapiro-Wilk |
| **Confidence Intervals** | Mean, variance, Sharpe ratio CIs |
| **Distribution Fitting** | Normal, Student-t, Log-normal |
| **Power Analysis** | Sample size determination |

### Visualization

- Return distribution (histogram + KDE)
- Monte Carlo path simulations
- Correlation heatmaps
- VaR confidence bands
- Convergence analysis

---

## Project Structure

```
quantitative-systems-simulator/
├── cpp_core/
│   ├── include/
│   │   ├── random_generator.hpp    # RNG with variance reduction
│   │   ├── portfolio.hpp           # Portfolio data structures
│   │   ├── monte_carlo.hpp         # MC engine (single + parallel)
│   │   └── statistics.hpp          # Statistical functions
│   ├── src/
│   │   ├── random_generator.cpp
│   │   ├── portfolio.cpp
│   │   ├── monte_carlo.cpp
│   │   ├── statistics.cpp
│   │   └── bindings.cpp            # pybind11 Python bindings
│   └── CMakeLists.txt
├── python/
│   └── qss/
│       ├── __init__.py
│       ├── orchestrator.py         # Main workflow coordinator
│       ├── analytics.py            # Statistical analysis
│       ├── visualization.py        # Plotting and dashboards
│       └── interface.py            # C++ bridge with Python fallback
├── data/
│   ├── sample_portfolio.csv
│   └── example_results.json
├── notebooks/
│   └── simulation_analysis.ipynb   # Interactive analysis
├── output/                         # Generated reports
├── main.py                         # Demo script
├── benchmark.py                    # Performance benchmark
├── requirements.txt
├── setup.py
└── README.md
```

---

## Example Output

```
============================================================
PORTFOLIO RISK METRICS
============================================================
Expected Annual Return:      11.95%
Volatility:                  26.37%
Sharpe Ratio:                  0.38

Value at Risk (95%):        -25.87%
Value at Risk (99%):        -36.57%
CVaR / ES (95%):            -32.25%
CVaR / ES (99%):            -40.67%

Skewness:                     0.731
Excess Kurtosis:              0.990
Avg Max Drawdown:            21.05%
============================================================
```

---

## Mathematical Foundation

### Portfolio Return Simulation

For a portfolio with weights w and asset returns R:

```
L_i = w^T * R_i
```

where R_i ~ N(mu, Sigma) with covariance matrix Sigma.

### Correlated Asset Simulation

Using Cholesky decomposition Sigma = L * L^T:

```
R = mu + L * Z
```

where Z ~ N(0, I) is a vector of independent standard normals.

### Risk Metrics

**Value at Risk (VaR)**:
```
VaR_alpha = Quantile(L, 1 - alpha)
```

**Conditional VaR (Expected Shortfall)**:
```
CVaR_alpha = E[L | L > VaR_alpha]
```

---

## Technologies

| Category | Technologies |
|----------|-------------|
| **Languages** | C++17, Python 3.13 |
| **C++ Libraries** | STL, `<random>`, `<thread>`, `<future>` |
| **Python Binding** | pybind11 |
| **Data Science** | NumPy, pandas, SciPy, statsmodels |
| **Visualization** | Matplotlib, Plotly, Seaborn |
| **Build System** | CMake, setuptools |

---

## Where I Left Off

The core system is fully functional. Here's what's working:

1. **C++ Monte Carlo Engine** - Built and integrated via pybind11. The parallel engine uses all CPU cores and bypasses Python's GIL for real speedups.

2. **Benchmark Script** - Shows the C++ parallel implementation is 118x faster than naive Python and 3.3x faster than NumPy.

3. **Demo (main.py)** - Runs a full simulation and outputs risk metrics.

4. **Jupyter Notebook** - Complete walkthrough of the system.

5. **Visualizations** - Distribution plots and path simulations are generated in `/output`.

### Next Steps

- [ ] GPU acceleration (CUDA/OpenCL)
- [ ] Options pricing (Black-Scholes, Greeks)
- [ ] Bayesian updating of return distributions
- [ ] Streamlit/Dash web interface
- [ ] Real-time market data integration

---

## License

MIT License - 2025 Raja Hussain

---

**Built with C++ and Python for Quantitative Finance**
