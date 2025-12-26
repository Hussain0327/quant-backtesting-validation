/**
 * Market Data Fetcher
 * Fetches historical OHLCV data from Yahoo Finance
 */

import { OHLCV, FetchOptions, FetchResult } from './types';

/**
 * Fetch historical market data
 * Note: This is designed to work with the API route
 * Direct usage requires the yahoo-finance2 package on server-side
 */
export async function fetchMarketData(options: FetchOptions): Promise<FetchResult> {
  const { ticker, startDate, endDate } = options;

  // Call our API route
  const params = new URLSearchParams({
    ticker: ticker.toUpperCase(),
    start: startDate.toISOString().split('T')[0],
    end: endDate.toISOString().split('T')[0],
  });

  const response = await fetch(`/api/market-data?${params}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.message || 'Failed to fetch market data');
  }

  const result = await response.json();

  // Convert date strings back to Date objects
  const data: OHLCV[] = result.data.map((row: Record<string, unknown>) => ({
    ...row,
    date: new Date(row.date as string),
  }));

  return {
    data,
    ticker: result.ticker,
    startDate: new Date(result.startDate),
    endDate: new Date(result.endDate),
    dataPoints: data.length,
  };
}

/**
 * Validate ticker symbol
 */
export function isValidTicker(ticker: string): boolean {
  // Basic validation: 1-5 uppercase letters
  return /^[A-Z]{1,5}$/.test(ticker.toUpperCase());
}

/**
 * Calculate approximate trading days between dates
 */
export function estimateTradingDays(start: Date, end: Date): number {
  const msPerDay = 24 * 60 * 60 * 1000;
  const calendarDays = Math.ceil((end.getTime() - start.getTime()) / msPerDay);
  // Approximately 252 trading days per 365 calendar days
  return Math.floor(calendarDays * (252 / 365));
}

/**
 * Data quality warnings
 */
export const DATA_WARNINGS = {
  survivorshipBias:
    'Historical data may exclude delisted stocks, inflating average returns.',
  adjustedPrices:
    'Prices are adjusted for splits and dividends. Historical values may differ from original.',
  freeDataQuality:
    'Free data sources may contain errors. Cross-validate with premium providers for production use.',
};
