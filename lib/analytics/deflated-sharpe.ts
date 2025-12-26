/**
 * Deflated Sharpe Ratio
 * Adjusts for multiple testing bias
 * Ported from analytics/deflated_sharpe.py
 */

import { normPpf, normCdf } from '../math/normal';
import { DeflatedSharpeResult } from './types';

/**
 * Calculate expected maximum Sharpe ratio from N independent trials
 * Under the null hypothesis (no skill)
 */
export function expectedMaxSharpe(
  nTrials: number,
  sampleLength: number,
  skewness: number = 0,
  kurtosis: number = 3
): number {
  if (nTrials <= 0) return 0;
  if (nTrials === 1) return 0;

  // Euler-Mascheroni constant
  const gamma = 0.5772156649;

  // Expected max of N standard normals
  // E[max(Z_1, ..., Z_N)] ≈ (1 - γ) * Φ^{-1}(1 - 1/N) + γ * Φ^{-1}(1 - 1/(N*e))
  const eMax =
    (1 - gamma) * normPpf(1 - 1 / nTrials) +
    gamma * normPpf(1 - 1 / (nTrials * Math.E));

  // Adjust for sample size and non-normality
  // Standard error of Sharpe ratio
  const srStd = Math.sqrt(
    (1 + 0.5 * skewness * skewness + (kurtosis - 3) / 4) / sampleLength
  );

  return eMax * srStd;
}

/**
 * Calculate Deflated Sharpe Ratio
 *
 * This tells you: "What's the probability that my observed Sharpe ratio
 * is due to skill rather than luck from testing many strategies?"
 */
export function deflatedSharpeRatio(
  observedSharpe: number,
  nTrials: number,
  sampleLength: number,
  options: {
    skewness?: number;
    kurtosis?: number;
    sharpeBenchmark?: number;
  } = {}
): DeflatedSharpeResult {
  const { skewness = 0, kurtosis = 3, sharpeBenchmark = 0 } = options;

  if (nTrials < 1) nTrials = 1;

  if (sampleLength < 10) {
    return {
      observedSharpe,
      deflatedSharpe: observedSharpe,
      haircut: 0,
      pValue: 1.0,
      eMaxSharpe: 0,
      nTrials,
      sampleLength,
      isSignificant: false,
      interpretation: 'Insufficient data for DSR calculation',
    };
  }

  // Expected maximum Sharpe under the null (no skill)
  const eMax = expectedMaxSharpe(nTrials, sampleLength, skewness, kurtosis);

  // Standard error of Sharpe ratio
  const srStd = Math.sqrt(
    (1 + 0.5 * skewness * skewness + (kurtosis - 3) / 4) / sampleLength
  );

  // Z-score
  const zScore = srStd > 0 ? (observedSharpe - sharpeBenchmark) / srStd : 0;

  // P-value with multiple testing adjustment (Sidak correction)
  let pValue: number;
  if (nTrials > 1) {
    const pSingle = 1 - normCdf(zScore);
    pValue = Math.min(1.0, 1 - Math.pow(1 - pSingle, nTrials));
  } else {
    pValue = 1 - normCdf(zScore);
  }

  // Deflated Sharpe = observed - expected_max_under_null
  const deflated = Math.max(0, observedSharpe - eMax);

  // Haircut percentage
  const haircut = observedSharpe > 0 ? (1 - deflated / observedSharpe) * 100 : 0;

  // Interpretation
  let interpretation: string;
  if (pValue < 0.01) {
    interpretation = `Strong evidence of skill. After ${nTrials} trials, p=${pValue.toFixed(3)}`;
  } else if (pValue < 0.05) {
    interpretation = `Moderate evidence. After ${nTrials} trials, p=${pValue.toFixed(3)}. Needs more validation.`;
  } else if (pValue < 0.1) {
    interpretation = `Weak evidence. After ${nTrials} trials, ${(pValue * 100).toFixed(0)}% chance this is luck.`;
  } else {
    interpretation = `Likely luck. After ${nTrials} trials, Sharpe of ${observedSharpe.toFixed(2)} has ${(pValue * 100).toFixed(0)}% chance of being noise.`;
  }

  return {
    observedSharpe,
    deflatedSharpe: deflated,
    haircut,
    pValue,
    eMaxSharpe: eMax,
    nTrials,
    sampleLength,
    isSignificant: pValue < 0.05,
    interpretation,
  };
}

/**
 * Estimate number of trials from parameter grid
 */
export function estimateNTrials(
  paramGrid?: Record<string, unknown[]>,
  nStrategies: number = 1,
  nParamCombos?: number
): number {
  if (nParamCombos !== undefined) {
    return nParamCombos * nStrategies;
  }

  if (!paramGrid) {
    return nStrategies;
  }

  let nCombos = 1;
  for (const values of Object.values(paramGrid)) {
    nCombos *= values.length;
  }

  return nCombos * nStrategies;
}
