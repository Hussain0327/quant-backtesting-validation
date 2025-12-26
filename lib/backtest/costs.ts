/**
 * Transaction Cost Models
 * Models real-world costs that erode backtested returns
 */

import { CostModel } from './types';

/**
 * Estimate bid-ask spread based on price level
 * Lower-priced stocks typically have wider spreads
 */
export function estimateSpread(price: number): number {
  if (price <= 0) return 0.002; // Default 0.2%

  if (price > 100) return 0.0005; // 0.05%
  if (price > 50) return 0.001; // 0.1%
  if (price > 20) return 0.0015; // 0.15%
  if (price > 10) return 0.002; // 0.2%
  if (price > 5) return 0.003; // 0.3%
  return 0.005; // 0.5% for penny stocks
}

export interface CostOptions {
  commissionPct?: number;
  slippagePct?: number;
  model?: CostModel;
  price?: number;
  volume?: number;
  avgDailyVolume?: number;
}

/**
 * Calculate transaction costs for a trade
 */
export function calculateCosts(
  tradeValue: number,
  options: CostOptions = {}
): number {
  const {
    commissionPct = 0.001,
    slippagePct = 0.0005,
    model = 'fixed',
    price,
    volume,
    avgDailyVolume,
  } = options;

  const absValue = Math.abs(tradeValue);

  switch (model) {
    case 'fixed':
      // Simple percentage-based costs
      return absValue * (commissionPct + slippagePct);

    case 'spread':
      // Estimate spread based on price level
      const spread = price ? estimateSpread(price) : 0.001;
      return absValue * spread + absValue * commissionPct;

    case 'impact':
      // Square-root market impact model for large orders
      if (volume && avgDailyVolume && avgDailyVolume > 0) {
        const participation = volume / avgDailyVolume;
        // Assume 2% daily volatility, 0.1 impact coefficient
        const impactPct = 0.02 * 0.1 * Math.sqrt(participation);
        return absValue * impactPct + absValue * commissionPct;
      }
      return absValue * (commissionPct + slippagePct);

    default:
      return absValue * (commissionPct + slippagePct);
  }
}

/**
 * Summarize total costs over multiple trades
 */
export function totalCostSummary(
  nTrades: number,
  avgTradeValue: number,
  options: CostOptions = {}
): {
  costPerTrade: number;
  totalCost: number;
  costPctOfCapital: number;
  modelUsed: CostModel;
} {
  const costPerTrade = calculateCosts(avgTradeValue, options);
  const totalCost = costPerTrade * nTrades;
  const costPct = (totalCost / (avgTradeValue * nTrades)) * 100;

  return {
    costPerTrade,
    totalCost,
    costPctOfCapital: costPct,
    modelUsed: options.model ?? 'fixed',
  };
}
