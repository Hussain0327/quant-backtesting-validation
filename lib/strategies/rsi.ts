/**
 * RSI (Relative Strength Index) Strategy
 * Mean reversion strategy based on overbought/oversold levels
 */

import { Strategy, OHLCV, SignalData, StrategyParams } from './types';
import { rollingMean, diff } from '../math/rolling';

export interface RSIParams {
  period: number;
  oversold: number;
  overbought: number;
}

/**
 * Calculate RSI indicator
 * RSI = 100 - (100 / (1 + RS))
 * where RS = Average Gain / Average Loss
 */
function calculateRSI(closes: number[], period: number): number[] {
  const deltas = diff(closes, 1);

  // Separate gains and losses
  const gains: number[] = deltas.map((d) => (Number.isNaN(d) ? NaN : d > 0 ? d : 0));
  const losses: number[] = deltas.map((d) => (Number.isNaN(d) ? NaN : d < 0 ? -d : 0));

  // Calculate rolling averages
  const avgGain = rollingMean(gains, period);
  const avgLoss = rollingMean(losses, period);

  // Calculate RSI
  const rsi: number[] = avgGain.map((gain, i) => {
    const loss = avgLoss[i];
    if (Number.isNaN(gain) || Number.isNaN(loss)) return NaN;
    if (loss === 0) return 100;
    const rs = gain / loss;
    return 100 - 100 / (1 + rs);
  });

  return rsi;
}

export class RSIStrategy implements Strategy {
  name = 'RSI';
  private period: number;
  private oversold: number;
  private overbought: number;

  constructor(params: Partial<RSIParams> = {}) {
    this.period = params.period ?? 14;
    this.oversold = params.oversold ?? 30;
    this.overbought = params.overbought ?? 70;
  }

  generateSignals(data: OHLCV[]): SignalData[] {
    const closes = data.map((d) => d.close);
    const rsi = calculateRSI(closes, this.period);

    const signals: SignalData[] = data.map((row, i) => {
      let signal: -1 | 0 | 1 = 0;
      const r = rsi[i];

      if (!Number.isNaN(r)) {
        if (r < this.oversold) signal = 1; // Oversold - buy
        else if (r > this.overbought) signal = -1; // Overbought - sell
      }

      return {
        ...row,
        signal,
        rsi: r,
      };
    });

    return signals;
  }

  getParams(): StrategyParams {
    return {
      period: this.period,
      oversold: this.oversold,
      overbought: this.overbought,
    };
  }
}
