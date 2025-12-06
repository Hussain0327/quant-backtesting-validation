#include "statistics.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

namespace qss {
namespace stats {

double mean(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    return std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(data.size());
}

double median(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    if (n % 2 == 0) {
        return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }
    return sorted[n / 2];
}

double variance(const std::vector<double>& data, bool sample) {
    if (data.size() < 2) return 0.0;
    double m = mean(data);
    double sum = 0.0;
    for (double x : data) {
        sum += (x - m) * (x - m);
    }
    return sum / static_cast<double>(sample ? data.size() - 1 : data.size());
}

double stddev(const std::vector<double>& data, bool sample) {
    return std::sqrt(variance(data, sample));
}

double skewness(const std::vector<double>& data) {
    if (data.size() < 3) return 0.0;
    double m = mean(data);
    double s = stddev(data, true);
    if (s == 0) return 0.0;

    double n = static_cast<double>(data.size());
    double sum = 0.0;
    for (double x : data) {
        double z = (x - m) / s;
        sum += z * z * z;
    }
    // Adjusted Fisher-Pearson standardized moment coefficient
    return (n / ((n - 1.0) * (n - 2.0))) * sum;
}

double kurtosis(const std::vector<double>& data) {
    if (data.size() < 4) return 0.0;
    double m = mean(data);
    double s = stddev(data, true);
    if (s == 0) return 0.0;

    double n = static_cast<double>(data.size());
    double sum = 0.0;
    for (double x : data) {
        double z = (x - m) / s;
        sum += z * z * z * z;
    }
    // Excess kurtosis with bias correction
    double g2 = (sum / n) - 3.0;
    return ((n + 1.0) * n * g2 + 6.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0)) * (n - 1.0);
}

double quantile(const std::vector<double>& data, double p) {
    if (data.empty()) return 0.0;
    if (p <= 0) return *std::min_element(data.begin(), data.end());
    if (p >= 1) return *std::max_element(data.begin(), data.end());

    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());

    double index = p * static_cast<double>(sorted.size() - 1);
    size_t lower = static_cast<size_t>(std::floor(index));
    size_t upper = static_cast<size_t>(std::ceil(index));

    if (lower == upper) return sorted[lower];

    double frac = index - static_cast<double>(lower);
    return sorted[lower] * (1.0 - frac) + sorted[upper] * frac;
}

double covariance(const std::vector<double>& x, const std::vector<double>& y, bool sample) {
    if (x.size() != y.size() || x.size() < 2) return 0.0;

    double mean_x = mean(x);
    double mean_y = mean(y);
    double sum = 0.0;

    for (size_t i = 0; i < x.size(); ++i) {
        sum += (x[i] - mean_x) * (y[i] - mean_y);
    }

    return sum / static_cast<double>(sample ? x.size() - 1 : x.size());
}

double correlation(const std::vector<double>& x, const std::vector<double>& y) {
    double cov = covariance(x, y);
    double std_x = stddev(x);
    double std_y = stddev(y);

    if (std_x == 0 || std_y == 0) return 0.0;
    return cov / (std_x * std_y);
}

std::vector<std::vector<double>> covariance_matrix(const std::vector<std::vector<double>>& data) {
    size_t n = data.size();
    std::vector<std::vector<double>> cov(n, std::vector<double>(n));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i; j < n; ++j) {
            cov[i][j] = covariance(data[i], data[j]);
            cov[j][i] = cov[i][j];
        }
    }
    return cov;
}

std::vector<std::vector<double>> correlation_matrix(const std::vector<std::vector<double>>& data) {
    size_t n = data.size();
    std::vector<std::vector<double>> corr(n, std::vector<double>(n));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i; j < n; ++j) {
            corr[i][j] = correlation(data[i], data[j]);
            corr[j][i] = corr[i][j];
        }
    }
    return corr;
}

// Normal distribution functions
namespace normal {

double pdf(double x, double mean, double stddev) {
    double z = (x - mean) / stddev;
    return std::exp(-0.5 * z * z) / (stddev * std::sqrt(2.0 * M_PI));
}

double cdf(double x, double mean, double stddev) {
    double z = (x - mean) / stddev;
    return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
}

double ppf(double p, double mean, double stddev) {
    // Rational approximation for the inverse normal CDF
    if (p <= 0) return -std::numeric_limits<double>::infinity();
    if (p >= 1) return std::numeric_limits<double>::infinity();

    double t = (p < 0.5) ? std::sqrt(-2.0 * std::log(p))
                         : std::sqrt(-2.0 * std::log(1.0 - p));

    // Coefficients for rational approximation
    double c0 = 2.515517;
    double c1 = 0.802853;
    double c2 = 0.010328;
    double d1 = 1.432788;
    double d2 = 0.189269;
    double d3 = 0.001308;

    double z = t - (c0 + c1 * t + c2 * t * t) /
                   (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if (p < 0.5) z = -z;

    return mean + stddev * z;
}

} // namespace normal

namespace student_t {

double pdf(double x, double df) {
    double coef = std::tgamma((df + 1.0) / 2.0) /
                  (std::sqrt(df * M_PI) * std::tgamma(df / 2.0));
    return coef * std::pow(1.0 + x * x / df, -(df + 1.0) / 2.0);
}

double cdf(double x, double df) {
    // Approximation using regularized incomplete beta function
    double t2 = x * x;
    double p = 0.5 * std::exp(std::lgamma((df + 1.0) / 2.0) - std::lgamma(df / 2.0)) /
               std::sqrt(df * M_PI);

    // Simple numerical integration for CDF (could be improved)
    if (std::abs(x) < 1e-10) return 0.5;

    // Use normal approximation for large df
    if (df > 100) {
        return normal::cdf(x, 0.0, 1.0);
    }

    // Continued fraction approximation
    double sum = 0.0;
    double term = x;
    for (int k = 1; k <= 100; ++k) {
        double prev = term;
        term *= -t2 * (df + 2.0 * k - 3.0) / ((2.0 * k - 1.0) * (df + 2.0 * k - 2.0));
        sum += term / (2.0 * k + 1.0);
        if (std::abs(term) < 1e-15 * std::abs(sum)) break;
    }

    return 0.5 + p * (x + x * sum);
}

} // namespace student_t

// Hypothesis testing
TestResult t_test_one_sample(const std::vector<double>& sample, double mu0) {
    TestResult result;
    size_t n = sample.size();
    double m = mean(sample);
    double s = stddev(sample, true);

    result.statistic = (m - mu0) / (s / std::sqrt(static_cast<double>(n)));

    // Two-tailed p-value
    double df = static_cast<double>(n - 1);
    double t_abs = std::abs(result.statistic);
    result.p_value = 2.0 * (1.0 - student_t::cdf(t_abs, df));

    result.reject_null = result.p_value < 0.05;
    result.description = "One-sample t-test";

    return result;
}

TestResult t_test_two_sample(const std::vector<double>& sample1,
                             const std::vector<double>& sample2,
                             bool equal_variance) {
    TestResult result;
    size_t n1 = sample1.size();
    size_t n2 = sample2.size();
    double m1 = mean(sample1);
    double m2 = mean(sample2);
    double v1 = variance(sample1, true);
    double v2 = variance(sample2, true);

    double df, se;

    if (equal_variance) {
        // Pooled variance
        double sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2);
        se = std::sqrt(sp2 * (1.0 / n1 + 1.0 / n2));
        df = static_cast<double>(n1 + n2 - 2);
    } else {
        // Welch's t-test
        se = std::sqrt(v1 / n1 + v2 / n2);
        double num = std::pow(v1 / n1 + v2 / n2, 2);
        double den = std::pow(v1 / n1, 2) / (n1 - 1) + std::pow(v2 / n2, 2) / (n2 - 1);
        df = num / den;
    }

    result.statistic = (m1 - m2) / se;
    double t_abs = std::abs(result.statistic);
    result.p_value = 2.0 * (1.0 - student_t::cdf(t_abs, df));
    result.reject_null = result.p_value < 0.05;
    result.description = equal_variance ? "Two-sample t-test (pooled)" : "Welch's t-test";

    return result;
}

TestResult jarque_bera_test(const std::vector<double>& data) {
    TestResult result;
    size_t n = data.size();
    double S = skewness(data);
    double K = kurtosis(data);  // Excess kurtosis

    // JB statistic
    result.statistic = (static_cast<double>(n) / 6.0) * (S * S + K * K / 4.0);

    // Chi-squared distribution with 2 degrees of freedom
    // P(X > x) for chi-squared(2) = exp(-x/2)
    result.p_value = std::exp(-result.statistic / 2.0);
    result.reject_null = result.p_value < 0.05;
    result.description = "Jarque-Bera normality test";

    return result;
}

TestResult shapiro_wilk_test(const std::vector<double>& data) {
    // Simplified Shapiro-Wilk approximation
    TestResult result;
    size_t n = data.size();

    if (n < 3 || n > 5000) {
        result.statistic = 0;
        result.p_value = 1.0;
        result.reject_null = false;
        result.description = "Shapiro-Wilk test (sample size out of range)";
        return result;
    }

    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());

    double m = mean(data);
    double ss = 0.0;
    for (double x : data) {
        ss += (x - m) * (x - m);
    }

    // Use normal scores approximation
    double sum_a = 0.0;
    for (size_t i = 0; i < n / 2; ++i) {
        double p = (static_cast<double>(i) + 1.0 - 0.375) / (static_cast<double>(n) + 0.25);
        double a = normal::ppf(p, 0.0, 1.0);
        sum_a += a * (sorted[n - 1 - i] - sorted[i]);
    }

    result.statistic = (sum_a * sum_a) / ss;

    // Approximate p-value (simplified)
    double z = std::sqrt(-2.0 * std::log(1.0 - result.statistic));
    result.p_value = 2.0 * (1.0 - normal::cdf(z, 0.0, 1.0));

    result.reject_null = result.p_value < 0.05;
    result.description = "Shapiro-Wilk normality test";

    return result;
}

// Confidence intervals
ConfidenceInterval mean_ci(const std::vector<double>& data, double confidence) {
    ConfidenceInterval ci;
    ci.confidence_level = confidence;
    ci.point_estimate = mean(data);

    double se = stddev(data, true) / std::sqrt(static_cast<double>(data.size()));
    double alpha = 1.0 - confidence;
    double z = normal::ppf(1.0 - alpha / 2.0, 0.0, 1.0);

    ci.lower = ci.point_estimate - z * se;
    ci.upper = ci.point_estimate + z * se;

    return ci;
}

ConfidenceInterval variance_ci(const std::vector<double>& data, double confidence) {
    ConfidenceInterval ci;
    ci.confidence_level = confidence;
    ci.point_estimate = variance(data, true);

    size_t n = data.size();
    double df = static_cast<double>(n - 1);
    double alpha = 1.0 - confidence;

    // Chi-squared quantiles approximation
    double chi2_lower = df * std::pow(1.0 - 2.0 / (9.0 * df) +
                        normal::ppf(alpha / 2.0, 0.0, 1.0) * std::sqrt(2.0 / (9.0 * df)), 3);
    double chi2_upper = df * std::pow(1.0 - 2.0 / (9.0 * df) +
                        normal::ppf(1.0 - alpha / 2.0, 0.0, 1.0) * std::sqrt(2.0 / (9.0 * df)), 3);

    ci.lower = df * ci.point_estimate / chi2_upper;
    ci.upper = df * ci.point_estimate / chi2_lower;

    return ci;
}

ConfidenceInterval proportion_ci(size_t successes, size_t trials, double confidence) {
    ConfidenceInterval ci;
    ci.confidence_level = confidence;
    ci.point_estimate = static_cast<double>(successes) / static_cast<double>(trials);

    double p = ci.point_estimate;
    double n = static_cast<double>(trials);
    double alpha = 1.0 - confidence;
    double z = normal::ppf(1.0 - alpha / 2.0, 0.0, 1.0);

    // Wilson score interval
    double denom = 1.0 + z * z / n;
    double center = (p + z * z / (2.0 * n)) / denom;
    double margin = z * std::sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom;

    ci.lower = center - margin;
    ci.upper = center + margin;

    return ci;
}

// Regression
RegressionResult linear_regression(const std::vector<double>& x, const std::vector<double>& y) {
    RegressionResult result;
    size_t n = x.size();

    if (n != y.size() || n < 3) {
        throw std::invalid_argument("Invalid input for regression");
    }

    double mean_x = mean(x);
    double mean_y = mean(y);

    double ss_xx = 0.0, ss_yy = 0.0, ss_xy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        ss_xx += dx * dx;
        ss_yy += dy * dy;
        ss_xy += dx * dy;
    }

    result.beta = ss_xy / ss_xx;
    result.alpha = mean_y - result.beta * mean_x;
    result.r_squared = (ss_xy * ss_xy) / (ss_xx * ss_yy);

    // Residual standard error
    double ss_res = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double resid = y[i] - (result.alpha + result.beta * x[i]);
        ss_res += resid * resid;
    }
    double mse = ss_res / (n - 2);
    double se = std::sqrt(mse);

    result.std_error_beta = se / std::sqrt(ss_xx);
    result.std_error_alpha = se * std::sqrt(1.0 / n + mean_x * mean_x / ss_xx);

    result.t_stat_alpha = result.alpha / result.std_error_alpha;
    result.t_stat_beta = result.beta / result.std_error_beta;

    double df = static_cast<double>(n - 2);
    result.p_value_alpha = 2.0 * (1.0 - student_t::cdf(std::abs(result.t_stat_alpha), df));
    result.p_value_beta = 2.0 * (1.0 - student_t::cdf(std::abs(result.t_stat_beta), df));

    return result;
}

// Power analysis
double power_t_test(double effect_size, size_t sample_size, double alpha) {
    double se = 1.0 / std::sqrt(static_cast<double>(sample_size));
    double critical_value = normal::ppf(1.0 - alpha / 2.0, 0.0, 1.0);
    double ncp = effect_size / se;  // Non-centrality parameter

    // Power = P(reject H0 | H1 true)
    double power = 1.0 - normal::cdf(critical_value - ncp, 0.0, 1.0) +
                   normal::cdf(-critical_value - ncp, 0.0, 1.0);

    return power;
}

size_t required_sample_size(double effect_size, double power, double alpha) {
    double z_alpha = normal::ppf(1.0 - alpha / 2.0, 0.0, 1.0);
    double z_power = normal::ppf(power, 0.0, 1.0);

    double n = std::pow((z_alpha + z_power) / effect_size, 2);

    return static_cast<size_t>(std::ceil(n));
}

} // namespace stats
} // namespace qss
