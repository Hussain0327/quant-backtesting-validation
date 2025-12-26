// Theme colors
export const COLORS = {
  background: '#0e1117',
  backgroundSecondary: '#1e293b',
  primary: '#3b82f6',
  primaryDark: '#2563eb',
  accent: '#00d4aa',
  success: '#10b981',
  error: '#ef4444',
  warning: '#f59e0b',
  surface: '#334155',
  textPrimary: '#e2e8f0',
  textSecondary: '#94a3b8',
  textMuted: '#64748b',
} as const;

// Chart colors
export const CHART_COLORS = {
  equity: '#3b82f6',
  benchmark: '#94a3b8',
  drawdown: '#ef4444',
  buy: '#10b981',
  sell: '#ef4444',
  grid: '#334155',
} as const;

// Strategy definitions
export const STRATEGIES = {
  'ma-crossover': {
    name: 'MA Crossover',
    description: 'Trend-following strategy using moving average crossovers',
    howItWorks: 'Buy when short MA crosses above long MA, sell when it crosses below',
    bestFor: 'Trending markets with clear directional moves',
    minDays: 120,
    defaultParams: {
      shortWindow: 20,
      longWindow: 50,
    },
  },
  'rsi': {
    name: 'RSI',
    description: 'Mean reversion strategy using Relative Strength Index',
    howItWorks: 'Buy when RSI indicates oversold (<30), sell when overbought (>70)',
    bestFor: 'Range-bound markets with mean-reverting behavior',
    minDays: 60,
    defaultParams: {
      period: 14,
      oversold: 30,
      overbought: 70,
    },
  },
  'momentum': {
    name: 'Momentum',
    description: 'Trend-following based on recent price momentum',
    howItWorks: 'Buy when momentum is positive, sell when negative',
    bestFor: 'Markets with strong trending periods',
    minDays: 90,
    defaultParams: {
      lookback: 20,
    },
  },
  'pairs-trading': {
    name: 'Pairs Trading',
    description: 'Statistical arbitrage using z-score deviations',
    howItWorks: 'Trade spread mean-reversion using entry/exit z-score thresholds',
    bestFor: 'Correlated assets with stable relationship',
    minDays: 90,
    defaultParams: {
      lookback: 20,
      entryThreshold: 2.0,
      exitThreshold: 0.5,
    },
  },
  'bollinger-bands': {
    name: 'Bollinger Bands',
    description: 'Mean reversion using Bollinger Band breakouts',
    howItWorks: 'Buy when price breaks below lower band, exit at middle band',
    bestFor: 'Volatile markets with mean-reverting tendencies',
    minDays: 90,
    defaultParams: {
      lookback: 20,
      numStd: 2.0,
    },
  },
} as const;

export type StrategyKey = keyof typeof STRATEGIES;

// Default backtest settings
export const DEFAULT_SETTINGS = {
  initialCapital: 10000,
  trainSplit: 0.7,
  commission: 0.001, // 0.1%
  slippage: 0.0005, // 0.05%
  riskFreeRate: 0.02, // 2% annual
  bootstrapSamples: 5000,
  tradingDaysPerYear: 252,
} as const;

// Formatting helpers
export const FORMATTERS = {
  currency: new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }),
  currencyPrecise: new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }),
  percent: new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }),
  percentPrecise: new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }),
  number: new Intl.NumberFormat('en-US', {
    minimumFractionDigits: 0,
    maximumFractionDigits: 2,
  }),
} as const;
