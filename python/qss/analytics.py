"""
Analytics module - Statistical analysis and metrics computation.

This module handles:
- Risk metrics calculation
- Statistical hypothesis testing
- Distribution analysis
- Confidence interval computation
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class RiskMetrics:
    """Container for risk-related metrics."""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    expected_shortfall: float
    tail_ratio: float


class PortfolioAnalytics:
    """
    Statistical analytics for portfolio simulation results.

    Provides methods for computing confidence intervals,
    hypothesis tests, and distribution analysis.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize analytics.

        Parameters
        ----------
        confidence_level : float
            Default confidence level for intervals (default: 0.95)
        """
        self.confidence_level = confidence_level

    def compute_risk_metrics(self, returns: np.ndarray) -> RiskMetrics:
        """
        Compute comprehensive risk metrics.

        Parameters
        ----------
        returns : np.ndarray
            Array of portfolio returns

        Returns
        -------
        RiskMetrics
            Container with all risk metrics
        """
        sorted_returns = np.sort(returns)
        n = len(returns)

        # Value at Risk
        var_95 = sorted_returns[int(0.05 * n)]
        var_99 = sorted_returns[int(0.01 * n)]

        # Conditional VaR (Expected Shortfall)
        cvar_95 = np.mean(sorted_returns[sorted_returns <= var_95])
        cvar_99 = np.mean(sorted_returns[sorted_returns <= var_99])

        # Tail ratio (upside vs downside)
        upper_tail = np.percentile(returns, 95)
        lower_tail = np.percentile(returns, 5)
        tail_ratio = abs(upper_tail / lower_tail) if lower_tail != 0 else np.inf

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=0.0,  # Computed from paths
            expected_shortfall=cvar_95,
            tail_ratio=tail_ratio,
        )

    def compute_confidence_intervals(
        self,
        returns: np.ndarray,
        confidence: Optional[float] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute confidence intervals for key metrics.

        Parameters
        ----------
        returns : np.ndarray
            Array of returns
        confidence : float, optional
            Confidence level (default: self.confidence_level)

        Returns
        -------
        dict
            Confidence intervals for mean, variance, VaR, etc.
        """
        conf = confidence or self.confidence_level
        alpha = 1 - conf

        n = len(returns)
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        se = std / np.sqrt(n)

        # Mean CI
        t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
        mean_ci = {
            "estimate": mean,
            "lower": mean - t_crit * se,
            "upper": mean + t_crit * se,
            "confidence": conf,
        }

        # Variance CI (chi-squared based)
        var = std ** 2
        chi2_lower = stats.chi2.ppf(alpha / 2, n - 1)
        chi2_upper = stats.chi2.ppf(1 - alpha / 2, n - 1)
        var_ci = {
            "estimate": var,
            "lower": (n - 1) * var / chi2_upper,
            "upper": (n - 1) * var / chi2_lower,
            "confidence": conf,
        }

        # Sharpe ratio CI (bootstrap approximation)
        sharpe = mean / std if std > 0 else 0
        sharpe_se = np.sqrt((1 + 0.5 * sharpe ** 2) / n)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        sharpe_ci = {
            "estimate": sharpe,
            "lower": sharpe - z_crit * sharpe_se,
            "upper": sharpe + z_crit * sharpe_se,
            "confidence": conf,
        }

        # VaR CI (order statistics approach)
        sorted_returns = np.sort(returns)
        p = 0.05  # For 95% VaR
        var_idx = int(p * n)
        var_estimate = sorted_returns[var_idx]

        # Approximate CI using binomial
        k_lower = max(0, int(stats.binom.ppf(alpha / 2, n, p)))
        k_upper = min(n - 1, int(stats.binom.ppf(1 - alpha / 2, n, p)))
        var_ci_result = {
            "estimate": var_estimate,
            "lower": sorted_returns[k_lower],
            "upper": sorted_returns[k_upper],
            "confidence": conf,
        }

        return {
            "mean": mean_ci,
            "variance": var_ci,
            "sharpe_ratio": sharpe_ci,
            "var_95": var_ci_result,
        }

    def analyze_distribution(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the distribution of returns.

        Parameters
        ----------
        returns : np.ndarray
            Array of returns

        Returns
        -------
        dict
            Distribution analysis including normality tests,
            moments, and best-fit distribution
        """
        # Moments
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)  # Excess kurtosis

        # Normality tests
        jarque_bera = stats.jarque_bera(returns)
        shapiro = stats.shapiro(returns[:5000])  # Shapiro limited to 5000

        # Anderson-Darling test
        anderson = stats.anderson(returns, dist='norm')

        # Fit distributions and compare
        distributions = self._fit_distributions(returns)

        return {
            "moments": {
                "mean": mean,
                "std": std,
                "skewness": skew,
                "kurtosis": kurt,
            },
            "normality_tests": {
                "jarque_bera": {
                    "statistic": jarque_bera.statistic,
                    "p_value": jarque_bera.pvalue,
                    "is_normal": jarque_bera.pvalue > 0.05,
                },
                "shapiro_wilk": {
                    "statistic": shapiro.statistic,
                    "p_value": shapiro.pvalue,
                    "is_normal": shapiro.pvalue > 0.05,
                },
                "anderson_darling": {
                    "statistic": anderson.statistic,
                    "critical_values": dict(zip(
                        [15, 10, 5, 2.5, 1],
                        anderson.critical_values
                    )),
                },
            },
            "fitted_distributions": distributions,
            "is_fat_tailed": kurt > 0,
            "is_skewed": abs(skew) > 0.5,
        }

    def _fit_distributions(self, returns: np.ndarray) -> Dict[str, Dict]:
        """Fit various distributions and return parameters and goodness-of-fit."""
        results = {}

        # Normal
        loc, scale = stats.norm.fit(returns)
        ks_stat, ks_p = stats.kstest(returns, 'norm', args=(loc, scale))
        results["normal"] = {
            "params": {"loc": loc, "scale": scale},
            "ks_statistic": ks_stat,
            "ks_p_value": ks_p,
        }

        # Student-t
        df, t_loc, t_scale = stats.t.fit(returns)
        ks_stat, ks_p = stats.kstest(returns, 't', args=(df, t_loc, t_scale))
        results["student_t"] = {
            "params": {"df": df, "loc": t_loc, "scale": t_scale},
            "ks_statistic": ks_stat,
            "ks_p_value": ks_p,
        }

        # Log-normal (for positive returns only)
        positive_returns = returns[returns > 0]
        if len(positive_returns) > 100:
            s, ln_loc, ln_scale = stats.lognorm.fit(positive_returns)
            results["lognormal"] = {
                "params": {"s": s, "loc": ln_loc, "scale": ln_scale},
                "note": "Fitted to positive returns only",
            }

        return results

    def analyze_tails(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Analyze tail behavior of the return distribution.

        Parameters
        ----------
        returns : np.ndarray
            Array of returns

        Returns
        -------
        dict
            Tail analysis including extreme value metrics
        """
        # Left tail (losses)
        left_tail = returns[returns < np.percentile(returns, 10)]
        right_tail = returns[returns > np.percentile(returns, 90)]

        # Extreme value analysis
        extremes = np.abs(returns[np.abs(returns) > 2 * np.std(returns)])
        extreme_freq = len(extremes) / len(returns)

        # Expected vs observed extremes under normality
        expected_extreme_freq = 2 * (1 - stats.norm.cdf(2))
        extreme_ratio = extreme_freq / expected_extreme_freq if expected_extreme_freq > 0 else np.inf

        return {
            "left_tail": {
                "mean": np.mean(left_tail),
                "std": np.std(left_tail),
                "min": np.min(left_tail),
                "count": len(left_tail),
            },
            "right_tail": {
                "mean": np.mean(right_tail),
                "std": np.std(right_tail),
                "max": np.max(right_tail),
                "count": len(right_tail),
            },
            "extreme_events": {
                "frequency": extreme_freq,
                "expected_normal": expected_extreme_freq,
                "excess_ratio": extreme_ratio,
                "interpretation": "Fat tails" if extreme_ratio > 1.5 else "Normal-like tails",
            },
            "percentiles": {
                "p1": np.percentile(returns, 1),
                "p5": np.percentile(returns, 5),
                "p10": np.percentile(returns, 10),
                "p25": np.percentile(returns, 25),
                "p50": np.percentile(returns, 50),
                "p75": np.percentile(returns, 75),
                "p90": np.percentile(returns, 90),
                "p95": np.percentile(returns, 95),
                "p99": np.percentile(returns, 99),
            },
        }

    def hypothesis_test_sharpe(
        self,
        returns1: np.ndarray,
        returns2: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Test if two strategies have different Sharpe ratios.

        Parameters
        ----------
        returns1 : np.ndarray
            Returns from first strategy
        returns2 : np.ndarray
            Returns from second strategy
        alpha : float
            Significance level

        Returns
        -------
        dict
            Test results including statistic and p-value
        """
        n1, n2 = len(returns1), len(returns2)
        mu1, mu2 = np.mean(returns1), np.mean(returns2)
        s1, s2 = np.std(returns1, ddof=1), np.std(returns2, ddof=1)

        sr1 = mu1 / s1 if s1 > 0 else 0
        sr2 = mu2 / s2 if s2 > 0 else 0

        # Jobson-Korkie test statistic (simplified)
        se1 = np.sqrt((1 + 0.5 * sr1 ** 2) / n1)
        se2 = np.sqrt((1 + 0.5 * sr2 ** 2) / n2)
        se_diff = np.sqrt(se1 ** 2 + se2 ** 2)

        z_stat = (sr1 - sr2) / se_diff if se_diff > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return {
            "sharpe_ratio_1": sr1,
            "sharpe_ratio_2": sr2,
            "difference": sr1 - sr2,
            "z_statistic": z_stat,
            "p_value": p_value,
            "reject_null": p_value < alpha,
            "conclusion": (
                "Sharpe ratios are significantly different"
                if p_value < alpha
                else "No significant difference in Sharpe ratios"
            ),
        }

    def power_analysis(
        self,
        effect_size: float,
        sample_size: int,
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """
        Compute statistical power for detecting an effect.

        Parameters
        ----------
        effect_size : float
            Cohen's d effect size
        sample_size : int
            Sample size
        alpha : float
            Significance level

        Returns
        -------
        dict
            Power analysis results
        """
        from scipy.stats import nct

        df = sample_size - 1
        critical_t = stats.t.ppf(1 - alpha / 2, df)
        ncp = effect_size * np.sqrt(sample_size)

        power = 1 - nct.cdf(critical_t, df, ncp) + nct.cdf(-critical_t, df, ncp)

        return {
            "effect_size": effect_size,
            "sample_size": sample_size,
            "alpha": alpha,
            "power": power,
            "interpretation": (
                "Adequate power (>80%)" if power > 0.8 else "May need larger sample"
            ),
        }

    def required_sample_size(
        self,
        effect_size: float,
        power: float = 0.8,
        alpha: float = 0.05
    ) -> int:
        """
        Compute required sample size for desired power.

        Parameters
        ----------
        effect_size : float
            Cohen's d effect size
        power : float
            Desired power (default: 0.8)
        alpha : float
            Significance level

        Returns
        -------
        int
            Required sample size
        """
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_power = stats.norm.ppf(power)

        n = ((z_alpha + z_power) / effect_size) ** 2

        return int(np.ceil(n))
