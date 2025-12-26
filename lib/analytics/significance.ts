/**
 * Statistical Significance Testing
 * Ported from analytics/significance.py
 */

import { mean, std, skewness, kurtosis, percentile, median } from '../math/statistics';
import { autocorrelation, pctChange, dropNaN } from '../math/rolling';
import { shuffle, randInt, choice } from '../math/random';
import {
  BootstrapResult,
  PermutationResult,
  MonteCarloResult,
  DistributionResult,
} from './types';

// =============================================================================
// ASSUMPTIONS DOCUMENTATION
// =============================================================================

export const BOOTSTRAP_ASSUMPTIONS = {
  name: 'Block Bootstrap Sharpe CI',
  assumes: [
    'Returns are stationary (distribution does not change over time)',
    'Finite variance (no infinite-variance fat tails)',
    'Block size captures relevant autocorrelation structure',
  ],
  preserves: [
    'Autocorrelation within blocks',
    'Marginal distribution of returns',
  ],
  doesNotAccountFor: [
    'Multiple testing (run many strategies, pick best)',
    'Regime changes (bull/bear market shifts)',
    'Look-ahead bias in strategy construction',
  ],
};

export const PERMUTATION_ASSUMPTIONS = {
  name: 'Permutation Test vs Benchmark',
  assumes: [
    'Strategy and benchmark returns are exchangeable under H0',
    'Returns within each series may be dependent',
    'Both series cover the same time period',
  ],
  limitations: [
    'Time-series structure (shuffles across time)',
    'Different risk profiles (vol, skew, kurtosis)',
    'Transaction costs already embedded in strategy returns',
  ],
};

export const MONTE_CARLO_ASSUMPTIONS = {
  name: 'Monte Carlo vs Random Trading',
  assumes: [
    'Random entry/exit points are uniformly distributed',
    'Number of trades similar to strategy',
    'Same capital and cost structure',
  ],
  limitations: [
    'Signal-based entry timing',
    'Correlation between strategy signals and price moves',
    'Survivorship bias in price data',
  ],
};

// =============================================================================
// BLOCK BOOTSTRAP
// =============================================================================

/**
 * Estimate optimal block size for block bootstrap
 */
export function estimateBlockSize(returns: number[]): number {
  const n = returns.length;
  if (n < 20) return Math.max(2, Math.floor(n / 4));

  // Rule of thumb for weakly dependent data
  let blockSize = Math.ceil(Math.pow(n, 1 / 3));

  // Adjust based on autocorrelation
  const acf1 = autocorrelation(returns, 1);
  if (!Number.isNaN(acf1) && Math.abs(acf1) > 0.1) {
    blockSize = Math.floor(blockSize * (1 + Math.abs(acf1)));
  }

  return Math.max(2, Math.min(blockSize, Math.floor(n / 4)));
}

/**
 * Generate one block bootstrap sample
 */
export function blockBootstrapSample(returns: number[], blockSize: number): number[] {
  const n = returns.length;
  const nBlocks = Math.ceil(n / blockSize);
  const maxStart = n - blockSize;

  if (maxStart < 0) {
    // Fall back to i.i.d. bootstrap
    return choice(returns, n);
  }

  const resampled: number[] = [];
  for (let i = 0; i < nBlocks; i++) {
    const start = randInt(0, maxStart + 1);
    resampled.push(...returns.slice(start, start + blockSize));
  }

  return resampled.slice(0, n);
}

/**
 * Calculate Sharpe ratio for bootstrap
 */
function calcSharpe(returns: number[], dailyRf: number): number {
  if (returns.length === 0) return 0;
  const excess = returns.map((r) => r - dailyRf);
  const s = std(excess);
  if (s === 0) return 0;
  return Math.sqrt(252) * mean(excess) / s;
}

/**
 * Bootstrap confidence interval for Sharpe ratio
 */
export function bootstrapSharpeCI(
  returns: number[],
  options: {
    nBootstrap?: number;
    confidenceLevel?: number;
    riskFreeRate?: number;
    blockSize?: number;
    method?: 'block' | 'iid';
  } = {}
): BootstrapResult {
  const {
    nBootstrap = 5000,
    confidenceLevel = 0.95,
    riskFreeRate = 0.02,
    method = 'block',
  } = options;
  let { blockSize } = options;

  const cleanReturns = dropNaN(returns);
  if (cleanReturns.length < 20) {
    return {
      sharpe: 0,
      ciLower: 0,
      ciUpper: 0,
      stdError: 0,
      ciIncludesZero: true,
      method,
      blockSize: null,
      nBootstrap,
      assumptions: BOOTSTRAP_ASSUMPTIONS,
    };
  }

  const dailyRf = riskFreeRate / 252;

  // Auto-estimate block size
  if (method === 'block' && !blockSize) {
    blockSize = estimateBlockSize(cleanReturns);
  }

  const pointEstimate = calcSharpe(cleanReturns, dailyRf);

  // Bootstrap resampling
  const bootstrapSharpes: number[] = [];
  for (let i = 0; i < nBootstrap; i++) {
    let sample: number[];
    if (method === 'block') {
      sample = blockBootstrapSample(cleanReturns, blockSize!);
    } else {
      sample = choice(cleanReturns, cleanReturns.length);
    }
    bootstrapSharpes.push(calcSharpe(sample, dailyRf));
  }

  const alpha = 1 - confidenceLevel;
  const ciLower = percentile(bootstrapSharpes, (alpha / 2) * 100);
  const ciUpper = percentile(bootstrapSharpes, (1 - alpha / 2) * 100);

  return {
    sharpe: pointEstimate,
    ciLower,
    ciUpper,
    stdError: std(bootstrapSharpes),
    ciIncludesZero: ciLower <= 0 && ciUpper >= 0,
    method,
    blockSize: method === 'block' ? blockSize! : null,
    nBootstrap,
    assumptions: BOOTSTRAP_ASSUMPTIONS,
  };
}

// =============================================================================
// PERMUTATION TEST
// =============================================================================

/**
 * Permutation test vs baseline
 */
export function permutationTest(
  strategyReturns: number[],
  baselineReturns: number[],
  options: {
    nPermutations?: number;
    metric?: 'mean' | 'sharpe';
  } = {}
): PermutationResult {
  const { nPermutations = 5000, metric = 'mean' } = options;

  const cleanStrategy = dropNaN(strategyReturns);
  const cleanBaseline = dropNaN(baselineReturns);

  const minLen = Math.min(cleanStrategy.length, cleanBaseline.length);
  const strat = cleanStrategy.slice(0, minLen);
  const base = cleanBaseline.slice(0, minLen);

  if (minLen < 10) {
    return {
      observedDiff: 0,
      pValue: 1.0,
      significantAt05: false,
      significantAt01: false,
      metricUsed: metric,
      nPermutations,
      assumptions: PERMUTATION_ASSUMPTIONS,
      caveat: 'i.i.d. permutation breaks time-series dependence',
    };
  }

  const calcMetric = (r: number[]): number => {
    if (metric === 'sharpe') {
      const s = std(r);
      return s === 0 ? 0 : Math.sqrt(252) * mean(r) / s;
    }
    return mean(r) * 252; // Annualized mean
  };

  const observedDiff = calcMetric(strat) - calcMetric(base);

  // Pool and permute
  const combined = [...strat, ...base];
  const nStrategy = strat.length;
  let countExtreme = 0;

  for (let i = 0; i < nPermutations; i++) {
    const shuffled = shuffle(combined);
    const permStrat = shuffled.slice(0, nStrategy);
    const permBase = shuffled.slice(nStrategy);
    const permDiff = calcMetric(permStrat) - calcMetric(permBase);

    if (permDiff >= observedDiff) countExtreme++;
  }

  const pValue = (countExtreme + 1) / (nPermutations + 1);

  return {
    observedDiff,
    pValue,
    significantAt05: pValue < 0.05,
    significantAt01: pValue < 0.01,
    metricUsed: metric,
    nPermutations,
    assumptions: PERMUTATION_ASSUMPTIONS,
    caveat: 'i.i.d. permutation breaks time-series dependence; interpret cautiously',
  };
}

// =============================================================================
// MONTE CARLO
// =============================================================================

/**
 * Monte Carlo simulation vs random trading
 */
export function monteCarloNull(
  prices: number[],
  options: {
    nSimulations?: number;
    strategyReturn?: number;
    nTradesObserved?: number;
  } = {}
): MonteCarloResult {
  const { nSimulations = 1000, strategyReturn, nTradesObserved } = options;

  const cleanPrices = dropNaN(prices);
  if (cleanPrices.length < 20) {
    return {
      nullMean: 0,
      nullStd: 0,
      nullMedian: 0,
      null5thPercentile: 0,
      null95thPercentile: 0,
      nSimulations,
      pValue: 1.0,
      assumptions: MONTE_CARLO_ASSUMPTIONS,
    };
  }

  const n = cleanPrices.length;
  const randomReturns: number[] = [];

  for (let sim = 0; sim < nSimulations; sim++) {
    // Random number of trades
    let nTrades: number;
    if (nTradesObserved !== undefined) {
      nTrades = Math.max(1, Math.floor(nTradesObserved / 2));
    } else {
      nTrades = randInt(2, Math.max(3, Math.floor(n / 20)));
    }

    // Random entry/exit points
    const nPoints = Math.min(nTrades * 2, n - 2);
    if (nPoints < 2) {
      randomReturns.push(0);
      continue;
    }

    // Generate unique random indices
    const indices = new Set<number>();
    while (indices.size < nPoints) {
      indices.add(randInt(1, n - 1));
    }
    const tradePoints = Array.from(indices).sort((a, b) => a - b);

    // Simulate random strategy
    let capital = 10000;
    let position = 0;

    for (let i = 0; i < tradePoints.length; i++) {
      const price = cleanPrices[tradePoints[i]];
      if (i % 2 === 0 && position === 0) {
        // Buy
        position = capital / price;
        capital = 0;
      } else if (i % 2 === 1 && position > 0) {
        // Sell
        capital = position * price;
        position = 0;
      }
    }

    // Close any open position
    if (position > 0) {
      capital = position * cleanPrices[n - 1];
    }

    randomReturns.push(((capital - 10000) / 10000) * 100);
  }

  const result: MonteCarloResult = {
    nullMean: mean(randomReturns),
    nullStd: std(randomReturns),
    nullMedian: median(randomReturns),
    null5thPercentile: percentile(randomReturns, 5),
    null95thPercentile: percentile(randomReturns, 95),
    nSimulations,
    assumptions: MONTE_CARLO_ASSUMPTIONS,
  };

  if (strategyReturn !== undefined) {
    const pValue = randomReturns.filter((r) => r >= strategyReturn).length / nSimulations;
    result.strategyReturn = strategyReturn;
    result.pValue = pValue;
    result.percentileRank = (randomReturns.filter((r) => r <= strategyReturn).length / nSimulations) * 100;
    result.significantAt05 = pValue < 0.05;
  }

  return result;
}

// =============================================================================
// DISTRIBUTION ANALYSIS
// =============================================================================

/**
 * Jarque-Bera test statistic
 * Tests if sample has skewness and kurtosis matching normal distribution
 */
function jarqueBera(returns: number[]): { statistic: number; pValue: number } {
  const n = returns.length;
  const s = skewness(returns);
  const k = kurtosis(returns); // excess kurtosis

  // JB = n/6 * (S^2 + K^2/4)
  const jb = (n / 6) * (s * s + (k * k) / 4);

  // Approximate p-value using chi-squared with 2 dof
  // For large JB, p-value ≈ 0; for JB ≈ 0, p-value ≈ 1
  // Using simple approximation since we don't have chi-squared CDF
  const pValue = Math.exp(-jb / 2);

  return { statistic: jb, pValue: Math.min(1, pValue) };
}

/**
 * Analyze return distribution
 */
export function analyzeDistribution(returns: number[]): DistributionResult {
  const cleanReturns = dropNaN(returns);

  if (cleanReturns.length < 20) {
    return {
      meanDaily: 0,
      stdDaily: 0,
      skewness: 0,
      excessKurtosis: 0,
      isFatTailed: false,
      isNegativelySkewed: false,
      jarqueBeraP: 1,
      isNormalJB: true,
      var95: 0,
      cvar95: 0,
    };
  }

  const m = mean(cleanReturns);
  const s = std(cleanReturns);
  const skew = skewness(cleanReturns);
  const kurt = kurtosis(cleanReturns);

  const jb = jarqueBera(cleanReturns);
  const var95 = percentile(cleanReturns, 5);
  const tailReturns = cleanReturns.filter((r) => r <= var95);
  const cvar95 = tailReturns.length > 0 ? mean(tailReturns) : var95;

  return {
    meanDaily: m,
    stdDaily: s,
    skewness: skew,
    excessKurtosis: kurt,
    isFatTailed: kurt > 1,
    isNegativelySkewed: skew < -0.5,
    jarqueBeraP: jb.pValue,
    isNormalJB: jb.pValue > 0.05,
    var95,
    cvar95,
  };
}
