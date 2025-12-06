#ifndef STATISTICS_HPP
#define STATISTICS_HPP

#include <vector>
#include <utility>
#include <optional>

namespace qss {
namespace stats {

// Descriptive statistics
double mean(const std::vector<double>& data);
double median(const std::vector<double>& data);
double variance(const std::vector<double>& data, bool sample = true);
double stddev(const std::vector<double>& data, bool sample = true);
double skewness(const std::vector<double>& data);
double kurtosis(const std::vector<double>& data);  // Excess kurtosis
double quantile(const std::vector<double>& data, double p);

// Covariance and correlation
double covariance(const std::vector<double>& x, const std::vector<double>& y, bool sample = true);
double correlation(const std::vector<double>& x, const std::vector<double>& y);
std::vector<std::vector<double>> covariance_matrix(const std::vector<std::vector<double>>& data);
std::vector<std::vector<double>> correlation_matrix(const std::vector<std::vector<double>>& data);

// Distribution functions
namespace normal {
    double pdf(double x, double mean = 0.0, double stddev = 1.0);
    double cdf(double x, double mean = 0.0, double stddev = 1.0);
    double ppf(double p, double mean = 0.0, double stddev = 1.0);  // Inverse CDF
}

namespace student_t {
    double pdf(double x, double df);
    double cdf(double x, double df);
}

// Hypothesis testing
struct TestResult {
    double statistic;
    double p_value;
    bool reject_null;  // At alpha = 0.05
    std::string description;
};

TestResult t_test_one_sample(const std::vector<double>& sample, double mu0);
TestResult t_test_two_sample(const std::vector<double>& sample1,
                             const std::vector<double>& sample2,
                             bool equal_variance = true);
TestResult jarque_bera_test(const std::vector<double>& data);  // Normality test
TestResult shapiro_wilk_test(const std::vector<double>& data); // Normality test

// Confidence intervals
struct ConfidenceInterval {
    double lower;
    double upper;
    double confidence_level;
    double point_estimate;
};

ConfidenceInterval mean_ci(const std::vector<double>& data, double confidence = 0.95);
ConfidenceInterval variance_ci(const std::vector<double>& data, double confidence = 0.95);
ConfidenceInterval proportion_ci(size_t successes, size_t trials, double confidence = 0.95);

// Regression
struct RegressionResult {
    double alpha;           // Intercept
    double beta;            // Slope
    double r_squared;
    double std_error_alpha;
    double std_error_beta;
    double t_stat_alpha;
    double t_stat_beta;
    double p_value_alpha;
    double p_value_beta;
};

RegressionResult linear_regression(const std::vector<double>& x, const std::vector<double>& y);

// Power analysis
double power_t_test(double effect_size, size_t sample_size, double alpha = 0.05);
size_t required_sample_size(double effect_size, double power = 0.8, double alpha = 0.05);

} // namespace stats
} // namespace qss

#endif // STATISTICS_HPP
