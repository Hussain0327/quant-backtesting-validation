# Algorithmic Trading Research Platform

A modular backtesting framework for developing, testing, and evaluating trading strategies on historical market data.

![Dashboard](screenshots/dashboard.png)

---

## Overview

This platform provides a complete workflow for algorithmic trading research:

- **Data Pipeline** - Fetch and store historical price data
- **Strategy Framework** - Modular system for implementing trading strategies
- **Backtesting Engine** - Simulate trades with realistic transaction costs
- **Analytics** - Performance metrics and risk analysis
- **Visualization** - Interactive dashboard for strategy evaluation

---

## Features

| Feature | Description |
|---------|-------------|
| Data Fetching | Pull historical OHLCV data from Yahoo Finance |
| Local Storage | SQLite database for caching price data |
| Multiple Strategies | MA Crossover, RSI, Momentum (easily extensible) |
| Train/Test Split | Validate strategies against unseen data |
| Cost Modeling | Commission and slippage simulation |
| Risk Metrics | Sharpe ratio, max drawdown, win rate |
| Interactive Dashboard | Streamlit-based UI for parameter tuning |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run backtest from command line
python main.py

# Launch interactive dashboard
streamlit run app.py
```

---

## Demo Output

```
fetching data...
running backtest with MA Crossover...

--- train results ---
return: -13.08%
sharpe: -0.62
max drawdown: -29.35%
trades: 9

--- test results ---
return: 1.39%
sharpe: 0.23
max drawdown: -2.82%
trades: 3
```

---

## Dashboard

The Streamlit dashboard allows you to:

- Select any ticker symbol
- Choose and configure strategies
- Adjust train/test split ratio
- Visualize equity curves and trade execution
- Compare performance metrics

![Equity Curve](screenshots/equity_curve.png)

---

## Project Structure

```
├── main.py                 # CLI entry point
├── app.py                  # Streamlit dashboard
│
├── data/
│   ├── fetcher.py          # Yahoo Finance API wrapper
│   └── database.py         # SQLite storage layer
│
├── strategies/
│   ├── base.py             # Abstract strategy interface
│   ├── moving_average.py   # MA Crossover implementation
│   ├── rsi.py              # RSI strategy
│   └── momentum.py         # Momentum strategy
│
├── backtest/
│   ├── engine.py           # Core backtesting logic
│   └── costs.py            # Transaction cost modeling
│
├── analytics/
│   └── metrics.py          # Performance calculations
│
└── screenshots/            # Documentation images
```

---

## Strategies

### Moving Average Crossover

Classic trend-following strategy. Generates buy signals when the short-term moving average crosses above the long-term moving average, and sell signals on the opposite crossover.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `short_window` | 20 | Short MA period |
| `long_window` | 50 | Long MA period |

### RSI (Relative Strength Index)

Mean reversion strategy based on the RSI indicator. Buys when RSI indicates oversold conditions and sells when overbought.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `period` | 14 | RSI calculation period |
| `oversold` | 30 | Buy threshold |
| `overbought` | 70 | Sell threshold |

### Momentum

Trend-following strategy based on price momentum. Takes long positions when momentum is positive over the lookback period.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | 20 | Momentum calculation period |

---

## Performance Metrics

| Metric | Description |
|--------|-------------|
| **Total Return** | Overall profit/loss as percentage |
| **Sharpe Ratio** | Risk-adjusted return (annualized) |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Win Rate** | Percentage of profitable trades |

---

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Fetch Data │ ──▶ │   Strategy  │ ──▶ │  Backtest   │
│  (yfinance) │     │  (signals)  │     │  (simulate) │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Report    │ ◀── │  Analytics  │ ◀── │   Results   │
│  (metrics)  │     │  (metrics)  │     │  (trades)   │
└─────────────┘     └─────────────┘     └─────────────┘
```

1. **Data Fetching** - Historical OHLCV data is pulled from Yahoo Finance
2. **Train/Test Split** - Data is split (default 70/30) to validate strategy generalization
3. **Signal Generation** - Strategy processes price data and outputs buy/sell signals
4. **Trade Simulation** - Backtest engine executes trades with transaction costs
5. **Performance Analysis** - Metrics are calculated on both train and test periods

The train/test split is crucial for detecting overfitting. A strategy that performs well on training data but poorly on test data is likely overfit to historical patterns.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical operations |
| `pandas` | Data manipulation |
| `yfinance` | Market data API |
| `streamlit` | Interactive dashboard |
| `plotly` | Interactive charts |
| `matplotlib` | Static visualizations |

---

## Roadmap

- [ ] Bollinger Bands strategy
- [ ] Mean reversion strategy
- [ ] Parameter optimization / grid search
- [ ] Volume-based slippage model
- [ ] Strategy comparison view
- [ ] Export results to CSV
- [ ] Multi-asset portfolio support

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Free to use, modify, and distribute.
