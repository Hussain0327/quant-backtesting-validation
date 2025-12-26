/**
 * Backtest Engine
 * Core simulation engine for trading strategies
 */

import { Strategy, SignalData, OHLCV } from '../strategies/types';
import { calculateCosts } from './costs';
import {
  BacktestResult,
  SimulationResult,
  Trade,
  EquityPoint,
  CostModel,
} from './types';

export interface EngineConfig {
  initialCapital?: number;
  commission?: number;
  slippage?: number;
  costModel?: CostModel;
}

export class BacktestEngine {
  private initialCapital: number;
  private commission: number;
  private slippage: number;
  private costModel: CostModel;

  constructor(config: EngineConfig = {}) {
    this.initialCapital = config.initialCapital ?? 10000;
    this.commission = config.commission ?? 0.001;
    this.slippage = config.slippage ?? 0.0005;
    this.costModel = config.costModel ?? 'fixed';
  }

  /**
   * Run backtest with train/test split
   */
  run(data: OHLCV[], strategy: Strategy, trainPct: number = 0.7): BacktestResult {
    const n = data.length;
    const trainEnd = Math.floor(n * trainPct);

    const trainData = data.slice(0, trainEnd);
    const testData = data.slice(trainEnd);

    // Generate signals separately for train and test
    const trainSignals = strategy.generateSignals(trainData);
    const testSignals = strategy.generateSignals(testData);

    // Simulate both periods
    const trainResults = this.simulate(trainSignals);
    const testResults = this.simulate(testSignals);

    return {
      train: trainResults,
      test: testResults,
      strategy: strategy.name,
      params: strategy.getParams(),
    };
  }

  /**
   * Simulate trading on signal data
   */
  simulate(signals: SignalData[]): SimulationResult {
    // Filter out any signals with NaN values
    const data = signals.filter(
      (s) => !Number.isNaN(s.close) && s.signal !== undefined
    );

    let capital = this.initialCapital;
    let position = 0;
    const trades: Trade[] = [];
    const equityCurve: EquityPoint[] = [];

    for (const row of data) {
      const price = row.close;
      const signal = row.signal;

      // Buy signal and no position
      if (signal === 1 && position === 0) {
        const shares = capital / price;
        const cost = calculateCosts(capital, {
          commissionPct: this.commission,
          slippagePct: this.slippage,
          model: this.costModel,
          price,
        });
        capital -= cost;
        position = shares;
        trades.push({
          type: 'buy',
          price,
          shares,
          date: row.date,
          value: capital,
        });
      }
      // Sell signal and have position
      else if (signal === -1 && position > 0) {
        const tradeValue = position * price;
        const cost = calculateCosts(tradeValue, {
          commissionPct: this.commission,
          slippagePct: this.slippage,
          model: this.costModel,
          price,
        });
        capital = tradeValue - cost;
        trades.push({
          type: 'sell',
          price,
          shares: position,
          date: row.date,
          value: capital,
        });
        position = 0;
      }

      // Track equity
      const equity = position === 0 ? capital : position * price;
      equityCurve.push({
        date: row.date,
        equity,
        price,
      });
    }

    const finalEquity =
      equityCurve.length > 0
        ? equityCurve[equityCurve.length - 1].equity
        : this.initialCapital;

    return {
      equityCurve,
      trades,
      finalEquity,
      returnPct: ((finalEquity - this.initialCapital) / this.initialCapital) * 100,
      numTrades: trades.length,
    };
  }

  /**
   * Get engine configuration
   */
  getConfig(): EngineConfig {
    return {
      initialCapital: this.initialCapital,
      commission: this.commission,
      slippage: this.slippage,
      costModel: this.costModel,
    };
  }
}
