/**
 * Strategy exports
 */

export * from './types';
export * from './moving-average';
export * from './rsi';
export * from './momentum';
export * from './pairs-trading';
export * from './bollinger-bands';

import { Strategy, StrategyType } from './types';
import { MovingAverageCrossover, MAParams } from './moving-average';
import { RSIStrategy, RSIParams } from './rsi';
import { MomentumStrategy, MomentumParams } from './momentum';
import { PairsTradingStrategy, PairsTradingParams } from './pairs-trading';
import { BollingerBandsStrategy, BollingerParams } from './bollinger-bands';

export type StrategyParamsUnion =
  | MAParams
  | RSIParams
  | MomentumParams
  | PairsTradingParams
  | BollingerParams;

/**
 * Factory function to create strategy instance
 */
export function createStrategy(
  type: StrategyType,
  params: Partial<StrategyParamsUnion> = {}
): Strategy {
  switch (type) {
    case 'ma-crossover':
      return new MovingAverageCrossover(params as Partial<MAParams>);
    case 'rsi':
      return new RSIStrategy(params as Partial<RSIParams>);
    case 'momentum':
      return new MomentumStrategy(params as Partial<MomentumParams>);
    case 'pairs-trading':
      return new PairsTradingStrategy(params as Partial<PairsTradingParams>);
    case 'bollinger-bands':
      return new BollingerBandsStrategy(params as Partial<BollingerParams>);
    default:
      throw new Error(`Unknown strategy type: ${type}`);
  }
}
