/**
 * Analytics type definitions
 */

export interface BootstrapResult {
  sharpe: number;
  ciLower: number;
  ciUpper: number;
  stdError: number;
  ciIncludesZero: boolean;
  method: 'block' | 'iid';
  blockSize: number | null;
  nBootstrap: number;
  assumptions: {
    assumes: string[];
    preserves: string[];
    doesNotAccountFor: string[];
  };
}

export interface PermutationResult {
  observedDiff: number;
  pValue: number;
  significantAt05: boolean;
  significantAt01: boolean;
  metricUsed: 'mean' | 'sharpe';
  nPermutations: number;
  assumptions: {
    assumes: string[];
    limitations: string[];
  };
  caveat: string;
}

export interface MonteCarloResult {
  nullMean: number;
  nullStd: number;
  nullMedian: number;
  null5thPercentile: number;
  null95thPercentile: number;
  nSimulations: number;
  strategyReturn?: number;
  pValue?: number;
  percentileRank?: number;
  significantAt05?: boolean;
  assumptions: {
    assumes: string[];
    limitations: string[];
  };
}

export interface DistributionResult {
  meanDaily: number;
  stdDaily: number;
  skewness: number;
  excessKurtosis: number;
  isFatTailed: boolean;
  isNegativelySkewed: boolean;
  jarqueBeraP: number;
  isNormalJB: boolean;
  var95: number;
  cvar95: number;
}

export interface DeflatedSharpeResult {
  observedSharpe: number;
  deflatedSharpe: number;
  haircut: number;
  pValue: number;
  eMaxSharpe: number;
  nTrials: number;
  sampleLength: number;
  isSignificant: boolean;
  interpretation: string;
}

export interface SignificanceReport {
  sharpeConfidence: BootstrapResult;
  vsBenchmark: PermutationResult;
  vsRandom: MonteCarloResult;
  returnDistribution: DistributionResult;
  summary: {
    sharpeStatisticallySignificant: boolean;
    beatsBenchmarkSignificantly: boolean;
    beatsRandomTrading: boolean;
    overallEvidence: string;
  };
}
