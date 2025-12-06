#include "random_generator.hpp"
#include <stdexcept>
#include <algorithm>

namespace qss {

RandomGenerator::RandomGenerator(unsigned int seed) : engine_(seed) {}

void RandomGenerator::set_seed(unsigned int seed) {
    engine_.seed(seed);
}

double RandomGenerator::uniform(double min, double max) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(engine_);
}

double RandomGenerator::normal(double mean, double stddev) {
    std::normal_distribution<double> dist(mean, stddev);
    return dist(engine_);
}

double RandomGenerator::lognormal(double mean, double stddev) {
    std::lognormal_distribution<double> dist(mean, stddev);
    return dist(engine_);
}

double RandomGenerator::student_t(double degrees_of_freedom) {
    std::student_t_distribution<double> dist(degrees_of_freedom);
    return dist(engine_);
}

std::vector<double> RandomGenerator::normal_vector(size_t n, double mean, double stddev) {
    std::vector<double> result(n);
    std::normal_distribution<double> dist(mean, stddev);
    for (size_t i = 0; i < n; ++i) {
        result[i] = dist(engine_);
    }
    return result;
}

std::vector<double> RandomGenerator::uniform_vector(size_t n, double min, double max) {
    std::vector<double> result(n);
    std::uniform_real_distribution<double> dist(min, max);
    for (size_t i = 0; i < n; ++i) {
        result[i] = dist(engine_);
    }
    return result;
}

std::vector<std::vector<double>> RandomGenerator::cholesky_decompose(
    const std::vector<std::vector<double>>& matrix
) {
    size_t n = matrix.size();
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }
            if (i == j) {
                double diag = matrix[i][i] - sum;
                if (diag <= 0) {
                    throw std::runtime_error("Matrix is not positive definite");
                }
                L[i][j] = std::sqrt(diag);
            } else {
                L[i][j] = (matrix[i][j] - sum) / L[j][j];
            }
        }
    }
    return L;
}

std::vector<std::vector<double>> RandomGenerator::correlated_normals(
    size_t n_samples,
    const std::vector<std::vector<double>>& correlation_matrix
) {
    size_t n_assets = correlation_matrix.size();
    auto L = cholesky_decompose(correlation_matrix);

    std::vector<std::vector<double>> result(n_samples, std::vector<double>(n_assets));

    for (size_t s = 0; s < n_samples; ++s) {
        // Generate independent standard normals
        auto z = normal_vector(n_assets, 0.0, 1.0);

        // Transform to correlated normals: X = L * Z
        for (size_t i = 0; i < n_assets; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j <= i; ++j) {
                sum += L[i][j] * z[j];
            }
            result[s][i] = sum;
        }
    }

    return result;
}

std::vector<double> RandomGenerator::antithetic_normal(size_t n, double mean, double stddev) {
    size_t half = n / 2;
    std::vector<double> result(n);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < half; ++i) {
        double z = dist(engine_);
        result[i] = mean + stddev * z;
        result[half + i] = mean - stddev * z;  // Antithetic pair
    }

    // Handle odd n
    if (n % 2 != 0) {
        result[n - 1] = dist(engine_) * stddev + mean;
    }

    return result;
}

std::vector<double> RandomGenerator::stratified_normal(size_t n_strata, double mean, double stddev) {
    std::vector<double> result(n_strata);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

    for (size_t i = 0; i < n_strata; ++i) {
        // Sample uniformly within each stratum
        double u = (static_cast<double>(i) + uniform_dist(engine_)) / static_cast<double>(n_strata);
        // Transform to normal using inverse CDF (Box-Muller approximation for efficiency)
        // Using Beasley-Springer-Moro algorithm for inverse normal CDF
        double t = u - 0.5;
        double r, x;
        if (std::abs(t) < 0.42) {
            r = t * t;
            x = t * (((2.50662823884 * r + -18.61500062529) * r + 41.39119773534) * r + -25.44106049637) /
                    ((((r + -8.47351093090) * r + 23.08336743743) * r + -21.06224101826) * r + 3.13082909833);
        } else {
            r = (t > 0) ? 1.0 - u : u;
            r = std::log(-std::log(r));
            x = 0.3374754822726147 + r * (0.9761690190917186 + r * (0.1607979714918209 +
                r * (0.0276438810333863 + r * (0.0038405729373609 + r * (0.0003951896511919 +
                r * (0.0000321767881768))))));
            if (t < 0) x = -x;
        }
        result[i] = mean + stddev * x;
    }

    return result;
}

} // namespace qss
