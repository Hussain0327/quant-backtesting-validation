# Roadmap

this is the first iteration of the platform. got the core working - data fetching, backtesting, basic strategies, and the dashboard. now i want to expand it and make it more robust.

currently researching and planning the next features. this doc tracks what i'm working towards.

---

## Strategies

### Bollinger Bands
- buy when price touches lower band (oversold)
- sell when price touches upper band (overbought)
- need to implement: band calculation, signal generation
- reference: https://www.investopedia.com/terms/b/bollingerbands.asp

### MACD (Moving Average Convergence Divergence)
- signal line crossover strategy
- more complex than simple MA crossover
- need to research: fast/slow EMA periods, signal line calculation

---

## Parameter Optimization

want to add grid search to find optimal strategy parameters

idea:
```python
param_grid = {
    'short_window': [10, 15, 20, 25],
    'long_window': [40, 50, 60, 70]
}
# run backtest for each combo, find best sharpe
```

considerations:
- need to be careful about overfitting
- should optimize on train set only
- maybe add walk-forward optimization later

---

## Risk Management

this is important for real trading. current system has none.

### Stop Loss
- exit position if loss exceeds X%
- need to track entry price and check on each bar

### Take Profit
- exit position if gain exceeds X%
- same logic as stop loss but for profits

### Position Sizing
- currently going all-in on each trade (not realistic)
- want to add:
  - fixed percentage of portfolio per trade
  - kelly criterion (maybe too advanced for now)
  - volatility-based sizing

---

## Multi-Asset Portfolio

right now only supports single stock backtests

want to add:
- backtest on multiple tickers
- portfolio allocation (equal weight, custom weights)
- rebalancing logic
- correlation analysis between assets

this would be a bigger refactor but would make it way more useful

---

## Better Visualizations

### Rolling Sharpe Chart
- shows how sharpe ratio changes over time
- helps identify when strategy is working vs not
- should be easy to add with pandas rolling

### Returns Distribution
- histogram of daily/trade returns
- overlay normal distribution to check for fat tails
- add skewness/kurtosis stats

### Correlation Matrix
- for multi-asset support
- heatmap of asset correlations
- helps with portfolio construction

### Drawdown Chart
- already have this in the dashboard but could improve
- add underwater plot (time spent in drawdown)

---

## Export Results

need to save backtest results for later analysis

- CSV export of:
  - all trades (entry, exit, pnl)
  - equity curve
  - daily returns
- JSON export of:
  - strategy params
  - performance metrics
  - metadata (date range, ticker, etc)

---

## Config System

hardcoding params is annoying. want to use YAML config files.

example:
```yaml
strategy:
  name: ma_crossover
  params:
    short_window: 20
    long_window: 50

backtest:
  initial_capital: 10000
  commission: 0.001
  slippage: 0.0005

data:
  ticker: AAPL
  start: 2022-01-01
  end: 2024-01-01
```

makes it easier to run different configs without changing code

---

## Strategy Comparison

want to run multiple strategies on same data and compare

dashboard feature:
- select 2-3 strategies
- run backtests
- show side-by-side metrics
- overlay equity curves

this would be really useful for deciding which strategy to use

---

## Unit Tests

should add tests for core functions

priority:
- [ ] test metrics calculations (sharpe, drawdown, etc)
- [ ] test strategy signal generation
- [ ] test backtest engine logic
- [ ] test data fetching

using pytest. haven't set this up yet.

---

## Logging

proper logging instead of print statements

want to track:
- every trade executed
- strategy signals
- errors and warnings
- backtest run metadata

probably use python logging module, write to file

---

## Notes

- focusing on getting core features solid first
- will tackle these roughly in order of impact
- some of these might change as i learn more
- open to suggestions if anyone has ideas

last updated: dec 2024
