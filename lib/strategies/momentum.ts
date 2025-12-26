/**
 * Momentum Strategy
 * Trend-following based on recent price momentum
 */

import { Strategy, OHLCV, SignalData, StrategyParams } from './types';
import { pctChange } from '../math/rolling';

export interface MomentumParams {
  lookback: number;
}

export class MomentumStrategy implements Strategy {
  name = 'Momentum';
  private lookback: number;

  constructor(params: Partial<MomentumParams> = {}) {
    this.lookback = params.lookback ?? 20;
  }

  generateSignals(data: OHLCV[]): SignalData[] {
    const closes = data.map((d) => d.close);

    // Calculate momentum as percent change over lookback period
    const momentum = pctChange(closes, this.lookback);

    const signals: SignalData[] = data.map((row, i) => {
      let signal: -1 | 0 | 1 = 0;
      const m = momentum[i];

      if (!Number.isNaN(m)) {
        if (m > 0) signal = 1; // Positive momentum - buy
        else if (m < 0) signal = -1; // Negative momentum - sell
      }

      return {
        ...row,
        signal,
        momentum: m,
      };
    });

    return signals;
  }

  getParams(): StrategyParams {
    return {
      lookback: this.lookback,
    };
  }
}
