/**
 * Pairs Trading Strategy (Statistical Arbitrage)
 * Mean reversion using z-score of spread
 */

import { Strategy, OHLCV, SignalData, StrategyParams } from './types';
import { rollingMean, rollingStd } from '../math/rolling';

export interface PairsTradingParams {
  lookback: number;
  entryThreshold: number;
  exitThreshold: number;
  stopLossThreshold: number;
}

/**
 * Calculate z-score of spread
 */
function calculateZScore(spread: number[], lookback: number): number[] {
  const rollingM = rollingMean(spread, lookback);
  const rollingS = rollingStd(spread, lookback);

  return spread.map((s, i) => {
    const mean = rollingM[i];
    const std = rollingS[i];
    if (Number.isNaN(mean) || Number.isNaN(std) || std === 0) return NaN;
    const z = (s - mean) / std;
    // Replace infinities with NaN
    if (!Number.isFinite(z)) return NaN;
    return z;
  });
}

export class PairsTradingStrategy implements Strategy {
  name = 'Pairs Trading (Spread Mean Reversion)';
  private lookback: number;
  private entryThreshold: number;
  private exitThreshold: number;
  private stopLossThreshold: number;

  constructor(params: Partial<PairsTradingParams> = {}) {
    this.lookback = params.lookback ?? 20;
    this.entryThreshold = params.entryThreshold ?? 2.0;
    this.exitThreshold = params.exitThreshold ?? 0.5;
    this.stopLossThreshold = params.stopLossThreshold ?? 4.0;
  }

  generateSignals(data: OHLCV[]): SignalData[] {
    const closes = data.map((d) => d.close);

    // Create synthetic spread: price vs its long-term trend
    const maLong = rollingMean(closes, this.lookback * 2);
    const spread = closes.map((c, i) => {
      const ma = maLong[i];
      if (Number.isNaN(ma)) return NaN;
      return c - ma;
    });

    // Calculate z-score
    const zscore = calculateZScore(spread, this.lookback);

    // Generate signals with position tracking
    let position = 0;
    const signals: SignalData[] = data.map((row, i) => {
      const z = zscore[i];
      let signal: -1 | 0 | 1 = 0;

      if (Number.isNaN(z)) {
        return { ...row, signal, zscore: z, spread: spread[i] };
      }

      // Entry logic
      if (position === 0) {
        if (z < -this.entryThreshold) {
          signal = 1; // Spread too low, expect reversion up - buy
          position = 1;
        } else if (z > this.entryThreshold) {
          signal = -1; // Spread too high, expect reversion down - sell
          position = -1;
        }
      }
      // Exit logic for long position
      else if (position === 1) {
        if (z > -this.exitThreshold || z > this.stopLossThreshold) {
          signal = -1; // Close long
          position = 0;
        }
      }
      // Exit logic for short position
      else if (position === -1) {
        if (z < this.exitThreshold || z < -this.stopLossThreshold) {
          signal = 1; // Close short
          position = 0;
        }
      }

      return { ...row, signal, zscore: z, spread: spread[i] };
    });

    return signals;
  }

  getParams(): StrategyParams {
    return {
      lookback: this.lookback,
      entryThreshold: this.entryThreshold,
      exitThreshold: this.exitThreshold,
      stopLossThreshold: this.stopLossThreshold,
    };
  }
}
