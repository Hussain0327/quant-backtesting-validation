#ifndef RANDOM_GENERATOR_HPP
#define RANDOM_GENERATOR_HPP

#include <random>
#include <vector>
#include <cmath>

namespace qss {

class RandomGenerator {
public:
    explicit RandomGenerator(unsigned int seed = std::random_device{}());

    // Basic distributions
    double uniform(double min = 0.0, double max = 1.0);
    double normal(double mean = 0.0, double stddev = 1.0);
    double lognormal(double mean = 0.0, double stddev = 1.0);
    double student_t(double degrees_of_freedom);

    // Vector generation
    std::vector<double> normal_vector(size_t n, double mean = 0.0, double stddev = 1.0);
    std::vector<double> uniform_vector(size_t n, double min = 0.0, double max = 1.0);

    // Correlated random samples using Cholesky decomposition
    std::vector<std::vector<double>> correlated_normals(
        size_t n_samples,
        const std::vector<std::vector<double>>& correlation_matrix
    );

    // Variance reduction techniques
    std::vector<double> antithetic_normal(size_t n, double mean = 0.0, double stddev = 1.0);
    std::vector<double> stratified_normal(size_t n_strata, double mean = 0.0, double stddev = 1.0);

    void set_seed(unsigned int seed);

private:
    std::mt19937_64 engine_;

    // Cholesky decomposition helper
    std::vector<std::vector<double>> cholesky_decompose(
        const std::vector<std::vector<double>>& matrix
    );
};

} // namespace qss

#endif // RANDOM_GENERATOR_HPP
