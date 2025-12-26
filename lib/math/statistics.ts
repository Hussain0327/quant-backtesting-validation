/**
 * Core statistical functions
 * Ported from numpy/scipy
 */

/**
 * Calculate the arithmetic mean of an array
 */
export function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((sum, val) => sum + val, 0) / arr.length;
}

/**
 * Calculate the standard deviation
 * @param ddof Delta degrees of freedom (0 = population, 1 = sample)
 */
export function std(arr: number[], ddof: number = 0): number {
  if (arr.length <= ddof) return 0;
  const m = mean(arr);
  const squaredDiffs = arr.map((val) => (val - m) ** 2);
  return Math.sqrt(squaredDiffs.reduce((sum, val) => sum + val, 0) / (arr.length - ddof));
}

/**
 * Calculate variance
 */
export function variance(arr: number[], ddof: number = 0): number {
  const s = std(arr, ddof);
  return s * s;
}

/**
 * Calculate covariance between two arrays
 */
export function covariance(x: number[], y: number[]): number {
  const n = Math.min(x.length, y.length);
  if (n === 0) return 0;

  const meanX = mean(x.slice(0, n));
  const meanY = mean(y.slice(0, n));

  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += (x[i] - meanX) * (y[i] - meanY);
  }

  return sum / n;
}

/**
 * Calculate Pearson correlation coefficient
 */
export function correlation(x: number[], y: number[]): number {
  const n = Math.min(x.length, y.length);
  if (n === 0) return 0;

  const cov = covariance(x, y);
  const stdX = std(x.slice(0, n));
  const stdY = std(y.slice(0, n));

  if (stdX === 0 || stdY === 0) return 0;
  return cov / (stdX * stdY);
}

/**
 * Calculate skewness (third standardized moment)
 * Using scipy.stats.skew formula (Fisher's definition)
 */
export function skewness(arr: number[]): number {
  const n = arr.length;
  if (n < 3) return 0;

  const m = mean(arr);
  const s = std(arr, 0);

  if (s === 0) return 0;

  let sum = 0;
  for (const val of arr) {
    sum += ((val - m) / s) ** 3;
  }

  return sum / n;
}

/**
 * Calculate excess kurtosis (fourth standardized moment - 3)
 * Using scipy.stats.kurtosis formula (Fisher's definition)
 * Normal distribution = 0
 */
export function kurtosis(arr: number[]): number {
  const n = arr.length;
  if (n < 4) return 0;

  const m = mean(arr);
  const s = std(arr, 0);

  if (s === 0) return 0;

  let sum = 0;
  for (const val of arr) {
    sum += ((val - m) / s) ** 4;
  }

  // Excess kurtosis (subtract 3 so normal = 0)
  return sum / n - 3;
}

/**
 * Calculate percentile using linear interpolation
 * Matches numpy's default interpolation
 * @param p Percentile (0-100)
 */
export function percentile(arr: number[], p: number): number {
  if (arr.length === 0) return 0;
  if (arr.length === 1) return arr[0];

  const sorted = [...arr].sort((a, b) => a - b);
  const index = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);

  if (lower === upper) {
    return sorted[lower];
  }

  const fraction = index - lower;
  return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
}

/**
 * Calculate multiple percentiles at once
 */
export function percentiles(arr: number[], ps: number[]): number[] {
  return ps.map((p) => percentile(arr, p));
}

/**
 * Calculate the sum of an array
 */
export function sum(arr: number[]): number {
  return arr.reduce((acc, val) => acc + val, 0);
}

/**
 * Find minimum value
 */
export function min(arr: number[]): number {
  if (arr.length === 0) return 0;
  return Math.min(...arr);
}

/**
 * Find maximum value
 */
export function max(arr: number[]): number {
  if (arr.length === 0) return 0;
  return Math.max(...arr);
}

/**
 * Calculate the median
 */
export function median(arr: number[]): number {
  return percentile(arr, 50);
}

/**
 * Linear regression: y = a + bx
 * Returns [intercept, slope]
 */
export function linearRegression(x: number[], y: number[]): [number, number] {
  const n = Math.min(x.length, y.length);
  if (n < 2) return [0, 0];

  const meanX = mean(x.slice(0, n));
  const meanY = mean(y.slice(0, n));

  let numerator = 0;
  let denominator = 0;

  for (let i = 0; i < n; i++) {
    numerator += (x[i] - meanX) * (y[i] - meanY);
    denominator += (x[i] - meanX) ** 2;
  }

  if (denominator === 0) return [meanY, 0];

  const slope = numerator / denominator;
  const intercept = meanY - slope * meanX;

  return [intercept, slope];
}
