#ifndef MONTE_CARLO_HPP
#define MONTE_CARLO_HPP

#include "portfolio.hpp"
#include "random_generator.hpp"
#include <vector>
#include <functional>

namespace qss {

struct SimulationConfig {
    size_t n_simulations = 10000;
    size_t time_horizon = 252;  // Trading days in a year
    double dt = 1.0 / 252.0;    // Time step
    double risk_free_rate = 0.02;
    bool use_antithetic = false;
    bool use_stratified = false;
    unsigned int seed = 0;      // 0 = random seed
};

struct SimulationResult {
    std::vector<std::vector<double>> simulated_returns;  // [simulation][time_step]
    std::vector<double> terminal_values;
    std::vector<double> path_maxima;
    std::vector<double> path_minima;
    std::vector<double> max_drawdowns;
    PortfolioMetrics metrics;
    double convergence_error;
};

class MonteCarloEngine {
public:
    explicit MonteCarloEngine(const SimulationConfig& config = SimulationConfig{});

    // Core simulation methods
    SimulationResult simulate_portfolio(const Portfolio& portfolio);
    SimulationResult simulate_gbm(double S0, double mu, double sigma);

    // Path generation
    std::vector<std::vector<double>> generate_paths(
        const Portfolio& portfolio,
        size_t n_paths
    );

    // Risk metrics calculation
    static double calculate_var(const std::vector<double>& returns, double confidence);
    static double calculate_cvar(const std::vector<double>& returns, double confidence);
    static double calculate_sharpe(const std::vector<double>& returns, double risk_free_rate);
    static double calculate_max_drawdown(const std::vector<double>& path);

    // Statistical moments
    static double mean(const std::vector<double>& data);
    static double variance(const std::vector<double>& data);
    static double stddev(const std::vector<double>& data);
    static double skewness(const std::vector<double>& data);
    static double kurtosis(const std::vector<double>& data);

    // Configuration
    void set_config(const SimulationConfig& config);
    const SimulationConfig& config() const { return config_; }

private:
    SimulationConfig config_;
    RandomGenerator rng_;

    // Internal simulation helpers
    std::vector<double> simulate_single_path(
        const Portfolio& portfolio,
        const std::vector<std::vector<double>>& cholesky_L
    );

    PortfolioMetrics compute_metrics(const std::vector<double>& terminal_values);
};

// Parallel simulation support
class ParallelMonteCarloEngine {
public:
    ParallelMonteCarloEngine(const SimulationConfig& config, size_t n_threads = 0);

    SimulationResult simulate_portfolio(const Portfolio& portfolio);

private:
    SimulationConfig config_;
    size_t n_threads_;
};

} // namespace qss

#endif // MONTE_CARLO_HPP
