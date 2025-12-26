/**
 * Performance Metrics
 * Ported from analytics/metrics.py
 */

import { mean, std } from '../math/statistics';
import { expandingMax, pctChange, dropNaN } from '../math/rolling';
import { Metrics, Trade, EquityPoint, SimulationResult } from '../backtest/types';

/**
 * Calculate Sharpe Ratio (annualized)
 * Sharpe = sqrt(252) * mean(excess_returns) / std(excess_returns)
 */
export function sharpeRatio(returns: number[], riskFreeRate: number = 0.02): number {
  const cleanReturns = dropNaN(returns);
  if (cleanReturns.length === 0) return 0;

  const dailyRf = riskFreeRate / 252;
  const excessReturns = cleanReturns.map((r) => r - dailyRf);

  const stdDev = std(excessReturns);
  if (stdDev === 0) return 0;

  return Math.sqrt(252) * mean(excessReturns) / stdDev;
}

/**
 * Calculate Maximum Drawdown (percentage)
 * Returns the largest peak-to-trough decline
 */
export function maxDrawdown(equityCurve: EquityPoint[]): number {
  if (equityCurve.length === 0) return 0;

  const equities = equityCurve.map((e) => e.equity);
  const peak = expandingMax(equities);

  let maxDD = 0;
  for (let i = 0; i < equities.length; i++) {
    const dd = (equities[i] - peak[i]) / peak[i];
    if (dd < maxDD) maxDD = dd;
  }

  return maxDD * 100; // Return as percentage (negative)
}

/**
 * Calculate Win Rate (percentage of profitable trades)
 */
export function winRate(trades: Trade[]): number {
  if (trades.length === 0) return 0;

  const profits: boolean[] = [];

  // Pair up buy/sell trades
  for (let i = 0; i < trades.length - 1; i += 2) {
    if (i + 1 < trades.length) {
      const buy = trades[i];
      const sell = trades[i + 1];
      if (buy.type === 'buy' && sell.type === 'sell') {
        profits.push(sell.price > buy.price);
      }
    }
  }

  if (profits.length === 0) return 0;
  return (profits.filter((p) => p).length / profits.length) * 100;
}

/**
 * Calculate returns from equity curve
 */
export function calculateReturns(equityCurve: EquityPoint[]): number[] {
  const equities = equityCurve.map((e) => e.equity);
  return dropNaN(pctChange(equities, 1));
}

/**
 * Calculate all metrics from simulation result
 */
export function calculateMetrics(result: SimulationResult): Metrics {
  const { equityCurve, trades, finalEquity, returnPct, numTrades } = result;

  if (equityCurve.length < 2) {
    return {
      totalReturn: 0,
      sharpe: 0,
      maxDrawdown: 0,
      winRate: 0,
      numTrades: 0,
      finalEquity: 10000,
    };
  }

  const returns = calculateReturns(equityCurve);

  return {
    totalReturn: returnPct,
    sharpe: sharpeRatio(returns),
    maxDrawdown: maxDrawdown(equityCurve),
    winRate: winRate(trades),
    numTrades,
    finalEquity,
  };
}

/**
 * Calculate buy-and-hold returns for comparison
 */
export function buyAndHoldReturn(
  prices: number[],
  initialCapital: number = 10000
): number {
  if (prices.length < 2) return 0;
  const startPrice = prices[0];
  const endPrice = prices[prices.length - 1];
  return ((endPrice - startPrice) / startPrice) * 100;
}

/**
 * Generate buy-and-hold equity curve
 */
export function buyAndHoldEquity(
  equityCurve: EquityPoint[],
  initialCapital: number = 10000
): EquityPoint[] {
  if (equityCurve.length === 0) return [];

  const startPrice = equityCurve[0].price;
  const shares = initialCapital / startPrice;

  return equityCurve.map((point) => ({
    date: point.date,
    equity: shares * point.price,
    price: point.price,
  }));
}
