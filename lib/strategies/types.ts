/**
 * Strategy type definitions
 */

export interface OHLCV {
  date: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface SignalData extends OHLCV {
  signal: -1 | 0 | 1;
  [key: string]: unknown; // Allow additional strategy-specific columns
}

export interface StrategyParams {
  [key: string]: number | string | boolean;
}

export interface Strategy {
  name: string;
  generateSignals(data: OHLCV[]): SignalData[];
  getParams(): StrategyParams;
}

export type StrategyType =
  | 'ma-crossover'
  | 'rsi'
  | 'momentum'
  | 'pairs-trading'
  | 'bollinger-bands';
