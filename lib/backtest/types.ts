/**
 * Backtest type definitions
 */

import { SignalData, StrategyParams } from '../strategies/types';

export interface Trade {
  type: 'buy' | 'sell';
  price: number;
  shares: number;
  date: Date;
  value?: number;
}

export interface EquityPoint {
  date: Date;
  equity: number;
  price: number;
}

export interface SimulationResult {
  equityCurve: EquityPoint[];
  trades: Trade[];
  finalEquity: number;
  returnPct: number;
  numTrades: number;
}

export interface BacktestResult {
  train: SimulationResult;
  test: SimulationResult;
  strategy: string;
  params: StrategyParams;
}

export interface Metrics {
  totalReturn: number;
  sharpe: number;
  maxDrawdown: number;
  winRate: number;
  numTrades: number;
  finalEquity: number;
}

export interface FoldResult {
  foldId: number;
  trainStart: Date;
  trainEnd: Date;
  testStart: Date;
  testEnd: Date;
  trainMetrics: Metrics;
  testMetrics: Metrics;
  nTrainDays: number;
  nTestDays: number;
}

export interface AggregateStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  median?: number;
}

export interface WalkForwardResult {
  foldResults: FoldResult[];
  aggregate: {
    return: AggregateStats;
    sharpe: AggregateStats;
    maxDrawdown: { mean: number; std: number; worst: number };
    winRate: { mean: number; std: number };
  };
  consistency: {
    pctPositiveReturns: number;
    pctPositiveSharpe: number;
    sharpeCoefficientOfVariation: number;
    isConsistent: boolean;
    interpretation: string;
  };
  nFolds: number;
  method: 'rolling' | 'expanding';
}

export type CostModel = 'fixed' | 'spread' | 'impact';
