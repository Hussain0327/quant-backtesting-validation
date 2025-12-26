/**
 * Bollinger Bands Strategy
 * Mean reversion using Bollinger Band breakouts
 */

import { Strategy, OHLCV, SignalData, StrategyParams } from './types';
import { rollingMean, rollingStd } from '../math/rolling';

export interface BollingerParams {
  lookback: number;
  numStd: number;
}

export class BollingerBandsStrategy implements Strategy {
  name = 'Bollinger Bands Mean Reversion';
  private lookback: number;
  private numStd: number;

  constructor(params: Partial<BollingerParams> = {}) {
    this.lookback = params.lookback ?? 20;
    this.numStd = params.numStd ?? 2.0;
  }

  generateSignals(data: OHLCV[]): SignalData[] {
    const closes = data.map((d) => d.close);

    // Calculate Bollinger Bands
    const ma = rollingMean(closes, this.lookback);
    const std = rollingStd(closes, this.lookback);

    const upperBand = ma.map((m, i) => {
      const s = std[i];
      if (Number.isNaN(m) || Number.isNaN(s)) return NaN;
      return m + this.numStd * s;
    });

    const lowerBand = ma.map((m, i) => {
      const s = std[i];
      if (Number.isNaN(m) || Number.isNaN(s)) return NaN;
      return m - this.numStd * s;
    });

    // Generate signals with position tracking
    let position = 0;
    const signals: SignalData[] = data.map((row, i) => {
      const price = closes[i];
      const lower = lowerBand[i];
      const upper = upperBand[i];
      const middle = ma[i];
      let signal: -1 | 0 | 1 = 0;

      if (Number.isNaN(lower) || Number.isNaN(upper) || Number.isNaN(middle)) {
        return {
          ...row,
          signal,
          ma: middle,
          upperBand: upper,
          lowerBand: lower,
        };
      }

      // Entry: buy when price breaks below lower band (oversold)
      if (position === 0) {
        if (price < lower) {
          signal = 1;
          position = 1;
        }
      }
      // Exit: when price reverts to mean or hits stop loss (upper band)
      else if (position === 1) {
        if (price >= middle) {
          signal = -1; // Reverted to mean - take profit
          position = 0;
        } else if (price > upper) {
          signal = -1; // Stop loss - went wrong direction
          position = 0;
        }
      }

      return {
        ...row,
        signal,
        ma: middle,
        upperBand: upper,
        lowerBand: lower,
      };
    });

    return signals;
  }

  getParams(): StrategyParams {
    return {
      lookback: this.lookback,
      numStd: this.numStd,
    };
  }
}
