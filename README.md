# Trading Strategy Validation

A **research-grade** backtesting platform for testing trading strategies against historical market data with rigorous statistical validation. Built with Next.js and TypeScript for instant Vercel deployment.

---

## Quickstart

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

Visit [http://localhost:3000](http://localhost:3000) to see the dashboard.

---

## Features

| Feature                     | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| Multiple Strategies         | MA Crossover, RSI, Momentum, Pairs Trading, Bollinger Bands  |
| **Walk-Forward Validation** | Tests across multiple time periods, not just one split       |
| **Deflated Sharpe Ratio**   | Corrects for multiple testing bias (Bailey & Lopez de Prado) |
| **Block Bootstrap**         | Preserves autocorrelation in resampling (Politis & Romano)   |
| Statistical Testing         | Bootstrap CI, permutation tests, Monte Carlo simulation      |
| Cost Modeling               | Fixed, spread-based, and market impact models                |
| Dark Theme UI               | Professional dashboard with responsive design                |

---

## Strategies

| Strategy            | Type                  | Logic                                | Best For               |
| ------------------- | --------------------- | ------------------------------------ | ---------------------- |
| **MA Crossover**    | Trend-following       | Buy when short MA > long MA          | Trending markets       |
| **RSI**             | Mean reversion        | Buy oversold, sell overbought        | Range-bound markets    |
| **Momentum**        | Trend-following       | Trade in direction of recent returns | Strong trending stocks |
| **Pairs Trading**   | Statistical arbitrage | Mean reversion on z-score spread     | Correlated assets      |
| **Bollinger Bands** | Mean reversion        | Buy at lower band, sell at mean      | Volatility trading     |

---

## Statistical Analysis

Every backtest runs three significance tests:

| Test                 | Question                                               |
| -------------------- | ------------------------------------------------------ |
| **Sharpe CI**        | Is the Sharpe ratio significantly different from zero? |
| **Permutation Test** | Does it beat buy-and-hold?                             |
| **Monte Carlo**      | Does it beat random entry/exit?                        |

**Interpretation:**

- 3/3 pass → Strong evidence of edge
- 2/3 pass → Needs more investigation
- 0-1/3 pass → Likely noise

---

## Tech Stack

| Component     | Technology           |
| ------------- | -------------------- |
| Framework     | Next.js 15           |
| Language      | TypeScript           |
| Styling       | Tailwind CSS         |
| Charts        | Recharts             |
| Market Data   | yahoo-finance2       |
| UI Components | Radix UI primitives  |

---

## Project Structure

```
├── app/
│   ├── page.tsx           # Main dashboard
│   ├── layout.tsx         # Root layout
│   ├── globals.css        # Global styles
│   └── api/
│       ├── backtest/      # Backtest API endpoint
│       └── market-data/   # Market data endpoint
│
├── lib/
│   ├── math/              # Statistics (normal dist, rolling windows)
│   ├── strategies/        # All 5 trading strategies
│   ├── backtest/          # Simulation engine with costs
│   ├── analytics/         # Metrics & significance tests
│   └── data/              # Yahoo Finance fetcher
│
├── components/
│   ├── ui/                # Base UI components
│   ├── charts/            # Recharts visualizations
│   └── metrics/           # Metric cards & grids
│
└── hooks/                 # React hooks for state
```

---

## Deploy to Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/Hussain0327/quant-backtesting-validation)

Or deploy via CLI:

```bash
npx vercel
```

---

## License

MIT
