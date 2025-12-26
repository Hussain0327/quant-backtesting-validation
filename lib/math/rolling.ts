/**
 * Rolling window operations
 * Ported from pandas rolling/expanding methods
 */

import { mean, std } from './statistics';

/**
 * Calculate rolling mean (simple moving average)
 * Returns array of same length with NaN for initial values
 */
export function rollingMean(arr: number[], window: number): number[] {
  const result: number[] = new Array(arr.length).fill(NaN);

  if (window <= 0 || window > arr.length) return result;

  // First value at index window-1
  let sum = 0;
  for (let i = 0; i < window; i++) {
    sum += arr[i];
  }
  result[window - 1] = sum / window;

  // Sliding window for remaining values
  for (let i = window; i < arr.length; i++) {
    sum = sum - arr[i - window] + arr[i];
    result[i] = sum / window;
  }

  return result;
}

/**
 * Calculate rolling standard deviation
 */
export function rollingStd(arr: number[], window: number, ddof: number = 0): number[] {
  const result: number[] = new Array(arr.length).fill(NaN);

  if (window <= ddof || window > arr.length) return result;

  for (let i = window - 1; i < arr.length; i++) {
    const slice = arr.slice(i - window + 1, i + 1);
    result[i] = std(slice, ddof);
  }

  return result;
}

/**
 * Calculate rolling sum
 */
export function rollingSum(arr: number[], window: number): number[] {
  const result: number[] = new Array(arr.length).fill(NaN);

  if (window <= 0 || window > arr.length) return result;

  let sum = 0;
  for (let i = 0; i < window; i++) {
    sum += arr[i];
  }
  result[window - 1] = sum;

  for (let i = window; i < arr.length; i++) {
    sum = sum - arr[i - window] + arr[i];
    result[i] = sum;
  }

  return result;
}

/**
 * Calculate expanding maximum (cumulative max)
 * Used for drawdown calculations
 */
export function expandingMax(arr: number[]): number[] {
  const result: number[] = [];
  let currentMax = -Infinity;

  for (const val of arr) {
    currentMax = Math.max(currentMax, val);
    result.push(currentMax);
  }

  return result;
}

/**
 * Calculate expanding minimum (cumulative min)
 */
export function expandingMin(arr: number[]): number[] {
  const result: number[] = [];
  let currentMin = Infinity;

  for (const val of arr) {
    currentMin = Math.min(currentMin, val);
    result.push(currentMin);
  }

  return result;
}

/**
 * Calculate percentage change
 * pct_change = (current - previous) / previous
 */
export function pctChange(arr: number[], periods: number = 1): number[] {
  const result: number[] = new Array(arr.length).fill(NaN);

  for (let i = periods; i < arr.length; i++) {
    const prev = arr[i - periods];
    if (prev !== 0) {
      result[i] = (arr[i] - prev) / prev;
    }
  }

  return result;
}

/**
 * Calculate difference between consecutive values
 */
export function diff(arr: number[], periods: number = 1): number[] {
  const result: number[] = new Array(arr.length).fill(NaN);

  for (let i = periods; i < arr.length; i++) {
    result[i] = arr[i] - arr[i - periods];
  }

  return result;
}

/**
 * Calculate autocorrelation at specified lag
 * Used for block bootstrap block size estimation
 */
export function autocorrelation(arr: number[], lag: number = 1): number {
  if (lag >= arr.length) return 0;

  const n = arr.length;
  const m = mean(arr);

  let numerator = 0;
  let denominator = 0;

  for (let i = 0; i < n - lag; i++) {
    numerator += (arr[i] - m) * (arr[i + lag] - m);
  }

  for (let i = 0; i < n; i++) {
    denominator += (arr[i] - m) ** 2;
  }

  if (denominator === 0) return 0;
  return numerator / denominator;
}

/**
 * Calculate cumulative sum
 */
export function cumsum(arr: number[]): number[] {
  const result: number[] = [];
  let sum = 0;

  for (const val of arr) {
    sum += val;
    result.push(sum);
  }

  return result;
}

/**
 * Calculate cumulative product
 */
export function cumprod(arr: number[]): number[] {
  const result: number[] = [];
  let prod = 1;

  for (const val of arr) {
    prod *= val;
    result.push(prod);
  }

  return result;
}

/**
 * Fill NaN values with a specified value
 */
export function fillNaN(arr: number[], value: number = 0): number[] {
  return arr.map((v) => (Number.isNaN(v) ? value : v));
}

/**
 * Drop NaN values from array
 */
export function dropNaN(arr: number[]): number[] {
  return arr.filter((v) => !Number.isNaN(v));
}
