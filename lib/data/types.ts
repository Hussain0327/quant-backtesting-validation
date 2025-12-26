/**
 * Data layer type definitions
 */

export interface OHLCV {
  date: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface FetchOptions {
  ticker: string;
  startDate: Date;
  endDate: Date;
}

export interface FetchResult {
  data: OHLCV[];
  ticker: string;
  startDate: Date;
  endDate: Date;
  dataPoints: number;
}

export interface DataError {
  message: string;
  ticker: string;
  type: 'not_found' | 'invalid_dates' | 'api_error' | 'insufficient_data';
}
