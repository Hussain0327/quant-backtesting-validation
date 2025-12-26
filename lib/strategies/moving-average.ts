/**
 * Moving Average Crossover Strategy
 * Trend-following strategy using SMA crossovers
 */

import { Strategy, OHLCV, SignalData, StrategyParams } from './types';
import { rollingMean, diff } from '../math/rolling';

export interface MAParams {
  shortWindow: number;
  longWindow: number;
}

export class MovingAverageCrossover implements Strategy {
  name = 'MA Crossover';
  private shortWindow: number;
  private longWindow: number;

  constructor(params: Partial<MAParams> = {}) {
    this.shortWindow = params.shortWindow ?? 20;
    this.longWindow = params.longWindow ?? 50;
  }

  generateSignals(data: OHLCV[]): SignalData[] {
    const closes = data.map((d) => d.close);

    // Calculate SMAs
    const smaShort = rollingMean(closes, this.shortWindow);
    const smaLong = rollingMean(closes, this.longWindow);

    // Generate raw position signals (1 when short > long, -1 when short < long)
    const rawSignals: number[] = smaShort.map((short, i) => {
      const long = smaLong[i];
      if (Number.isNaN(short) || Number.isNaN(long)) return 0;
      if (short > long) return 1;
      if (short < long) return -1;
      return 0;
    });

    // Find crossover points using diff
    const signalDiff = diff(rawSignals, 1);

    // Convert to actual signals
    const signals: SignalData[] = data.map((row, i) => {
      let signal: -1 | 0 | 1 = 0;
      const d = signalDiff[i];

      if (!Number.isNaN(d)) {
        if (d > 0) signal = 1; // Crossed above
        else if (d < 0) signal = -1; // Crossed below
      }

      return {
        ...row,
        signal,
        smaShort: smaShort[i],
        smaLong: smaLong[i],
      };
    });

    return signals;
  }

  getParams(): StrategyParams {
    return {
      shortWindow: this.shortWindow,
      longWindow: this.longWindow,
    };
  }
}
