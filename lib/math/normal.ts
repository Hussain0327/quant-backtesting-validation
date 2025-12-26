/**
 * Normal distribution functions
 * Implements scipy.stats.norm.cdf and norm.ppf
 * Using Abramowitz & Stegun approximations
 */

/**
 * Error function approximation
 * Abramowitz & Stegun formula 7.1.26
 * Maximum error: 1.5 × 10^-7
 */
function erf(x: number): number {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const sign = x < 0 ? -1 : 1;
  const absX = Math.abs(x);

  const t = 1.0 / (1.0 + p * absX);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-absX * absX);

  return sign * y;
}

/**
 * Standard normal cumulative distribution function (CDF)
 * P(X <= x) for X ~ N(0, 1)
 * Equivalent to scipy.stats.norm.cdf(x)
 */
export function normCdf(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

/**
 * Standard normal probability density function (PDF)
 * f(x) for X ~ N(0, 1)
 */
export function normPdf(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

/**
 * Standard normal quantile function (inverse CDF / PPF)
 * Returns x such that P(X <= x) = p for X ~ N(0, 1)
 * Equivalent to scipy.stats.norm.ppf(p)
 *
 * Uses Acklam's algorithm with very high accuracy
 * Maximum relative error < 1.15 × 10^-9
 */
export function normPpf(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  if (p === 0.5) return 0;

  // Coefficients for rational approximation
  const a = [
    -3.969683028665376e+01,
    2.209460984245205e+02,
    -2.759285104469687e+02,
    1.383577518672690e+02,
    -3.066479806614716e+01,
    2.506628277459239e+00,
  ];

  const b = [
    -5.447609879822406e+01,
    1.615858368580409e+02,
    -1.556989798598866e+02,
    6.680131188771972e+01,
    -1.328068155288572e+01,
  ];

  const c = [
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
    4.374664141464968e+00,
    2.938163982698783e+00,
  ];

  const d = [
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e+00,
    3.754408661907416e+00,
  ];

  const pLow = 0.02425;
  const pHigh = 1 - pLow;

  let q: number;
  let r: number;

  if (p < pLow) {
    // Rational approximation for lower region
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p <= pHigh) {
    // Rational approximation for central region
    q = p - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  } else {
    // Rational approximation for upper region
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
}

/**
 * Standard normal survival function
 * P(X > x) = 1 - CDF(x)
 */
export function normSf(x: number): number {
  return 1 - normCdf(x);
}

/**
 * Normal CDF with custom mean and standard deviation
 */
export function normalCdf(x: number, mean: number = 0, std: number = 1): number {
  return normCdf((x - mean) / std);
}

/**
 * Normal PPF with custom mean and standard deviation
 */
export function normalPpf(p: number, mean: number = 0, std: number = 1): number {
  return mean + std * normPpf(p);
}
