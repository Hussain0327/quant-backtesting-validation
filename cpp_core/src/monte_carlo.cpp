#include "monte_carlo.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <thread>
#include <future>

namespace qss {

MonteCarloEngine::MonteCarloEngine(const SimulationConfig& config)
    : config_(config)
    , rng_(config.seed == 0 ? std::random_device{}() : config.seed)
{}

void MonteCarloEngine::set_config(const SimulationConfig& config) {
    config_ = config;
    if (config.seed != 0) {
        rng_.set_seed(config.seed);
    }
}

double MonteCarloEngine::mean(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    return std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(data.size());
}

double MonteCarloEngine::variance(const std::vector<double>& data) {
    if (data.size() < 2) return 0.0;
    double m = mean(data);
    double sum = 0.0;
    for (double x : data) {
        sum += (x - m) * (x - m);
    }
    return sum / static_cast<double>(data.size() - 1);
}

double MonteCarloEngine::stddev(const std::vector<double>& data) {
    return std::sqrt(variance(data));
}

double MonteCarloEngine::skewness(const std::vector<double>& data) {
    if (data.size() < 3) return 0.0;
    double m = mean(data);
    double s = stddev(data);
    if (s == 0) return 0.0;

    double sum = 0.0;
    for (double x : data) {
        double z = (x - m) / s;
        sum += z * z * z;
    }
    return sum / static_cast<double>(data.size());
}

double MonteCarloEngine::kurtosis(const std::vector<double>& data) {
    if (data.size() < 4) return 0.0;
    double m = mean(data);
    double s = stddev(data);
    if (s == 0) return 0.0;

    double sum = 0.0;
    for (double x : data) {
        double z = (x - m) / s;
        sum += z * z * z * z;
    }
    return sum / static_cast<double>(data.size()) - 3.0;  // Excess kurtosis
}

double MonteCarloEngine::calculate_var(const std::vector<double>& returns, double confidence) {
    if (returns.empty()) return 0.0;

    std::vector<double> sorted_returns = returns;
    std::sort(sorted_returns.begin(), sorted_returns.end());

    size_t index = static_cast<size_t>((1.0 - confidence) * static_cast<double>(sorted_returns.size()));
    index = std::min(index, sorted_returns.size() - 1);

    return sorted_returns[index];
}

double MonteCarloEngine::calculate_cvar(const std::vector<double>& returns, double confidence) {
    if (returns.empty()) return 0.0;

    double var = calculate_var(returns, confidence);

    double sum = 0.0;
    size_t count = 0;
    for (double ret : returns) {
        if (ret <= var) {
            sum += ret;
            ++count;
        }
    }

    return count > 0 ? sum / static_cast<double>(count) : var;
}

double MonteCarloEngine::calculate_sharpe(const std::vector<double>& returns, double risk_free_rate) {
    if (returns.empty()) return 0.0;

    double mean_ret = mean(returns);
    double std_ret = stddev(returns);

    if (std_ret == 0) return 0.0;

    return (mean_ret - risk_free_rate) / std_ret;
}

double MonteCarloEngine::calculate_max_drawdown(const std::vector<double>& path) {
    if (path.empty()) return 0.0;

    double max_drawdown = 0.0;
    double peak = path[0];

    for (double value : path) {
        if (value > peak) {
            peak = value;
        }
        double drawdown = (peak - value) / peak;
        if (drawdown > max_drawdown) {
            max_drawdown = drawdown;
        }
    }

    return max_drawdown;
}

std::vector<double> MonteCarloEngine::simulate_single_path(
    const Portfolio& portfolio,
    const std::vector<std::vector<double>>& cholesky_L
) {
    size_t n_assets = portfolio.num_assets();
    size_t n_steps = config_.time_horizon;
    auto weights = portfolio.get_weights();
    auto expected_returns = portfolio.get_expected_returns();
    auto volatilities = portfolio.get_volatilities();

    std::vector<double> path(n_steps + 1);
    path[0] = 1.0;  // Start with unit value

    for (size_t t = 1; t <= n_steps; ++t) {
        // Generate correlated random returns
        auto z = rng_.normal_vector(n_assets, 0.0, 1.0);

        // Apply Cholesky transformation for correlation
        std::vector<double> correlated_z(n_assets, 0.0);
        for (size_t i = 0; i < n_assets; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                correlated_z[i] += cholesky_L[i][j] * z[j];
            }
        }

        // Calculate portfolio return for this step
        double portfolio_return = 0.0;
        for (size_t i = 0; i < n_assets; ++i) {
            double drift = (expected_returns[i] - 0.5 * volatilities[i] * volatilities[i]) * config_.dt;
            double diffusion = volatilities[i] * std::sqrt(config_.dt) * correlated_z[i];
            double asset_return = drift + diffusion;
            portfolio_return += weights[i] * asset_return;
        }

        path[t] = path[t - 1] * std::exp(portfolio_return);
    }

    return path;
}

PortfolioMetrics MonteCarloEngine::compute_metrics(const std::vector<double>& terminal_values) {
    PortfolioMetrics metrics{};

    // Convert terminal values to returns
    std::vector<double> returns(terminal_values.size());
    for (size_t i = 0; i < terminal_values.size(); ++i) {
        returns[i] = terminal_values[i] - 1.0;  // Return from initial value of 1.0
    }

    metrics.expected_return = mean(returns);
    metrics.volatility = stddev(returns);
    metrics.sharpe_ratio = calculate_sharpe(returns, config_.risk_free_rate);
    metrics.var_95 = calculate_var(returns, 0.95);
    metrics.var_99 = calculate_var(returns, 0.99);
    metrics.cvar_95 = calculate_cvar(returns, 0.95);
    metrics.cvar_99 = calculate_cvar(returns, 0.99);
    metrics.skewness = skewness(returns);
    metrics.kurtosis = kurtosis(returns);

    return metrics;
}

SimulationResult MonteCarloEngine::simulate_portfolio(const Portfolio& portfolio) {
    SimulationResult result;
    result.simulated_returns.resize(config_.n_simulations);
    result.terminal_values.resize(config_.n_simulations);
    result.max_drawdowns.resize(config_.n_simulations);

    // Get correlation matrix and compute Cholesky decomposition
    auto corr_matrix = portfolio.correlation_matrix();
    std::vector<std::vector<double>> cholesky_L;

    // Compute Cholesky decomposition
    size_t n = corr_matrix.size();
    cholesky_L.resize(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += cholesky_L[i][k] * cholesky_L[j][k];
            }
            if (i == j) {
                cholesky_L[i][j] = std::sqrt(std::max(0.0, corr_matrix[i][i] - sum));
            } else {
                cholesky_L[i][j] = (cholesky_L[j][j] > 0) ?
                    (corr_matrix[i][j] - sum) / cholesky_L[j][j] : 0.0;
            }
        }
    }

    // Run simulations
    for (size_t sim = 0; sim < config_.n_simulations; ++sim) {
        auto path = simulate_single_path(portfolio, cholesky_L);
        result.simulated_returns[sim] = path;
        result.terminal_values[sim] = path.back();
        result.max_drawdowns[sim] = calculate_max_drawdown(path);
    }

    // Compute aggregate metrics
    result.metrics = compute_metrics(result.terminal_values);

    // Calculate max drawdown metric
    result.metrics.max_drawdown = mean(result.max_drawdowns);

    // Estimate convergence error (standard error of mean)
    result.convergence_error = stddev(result.terminal_values) /
                               std::sqrt(static_cast<double>(config_.n_simulations));

    return result;
}

SimulationResult MonteCarloEngine::simulate_gbm(double S0, double mu, double sigma) {
    SimulationResult result;
    result.simulated_returns.resize(config_.n_simulations);
    result.terminal_values.resize(config_.n_simulations);
    result.max_drawdowns.resize(config_.n_simulations);

    double dt = config_.dt;
    double drift = (mu - 0.5 * sigma * sigma) * dt;
    double vol = sigma * std::sqrt(dt);

    for (size_t sim = 0; sim < config_.n_simulations; ++sim) {
        std::vector<double> path(config_.time_horizon + 1);
        path[0] = S0;

        std::vector<double> z;
        if (config_.use_antithetic) {
            z = rng_.antithetic_normal(config_.time_horizon);
        } else if (config_.use_stratified) {
            z = rng_.stratified_normal(config_.time_horizon);
        } else {
            z = rng_.normal_vector(config_.time_horizon);
        }

        for (size_t t = 1; t <= config_.time_horizon; ++t) {
            path[t] = path[t - 1] * std::exp(drift + vol * z[t - 1]);
        }

        result.simulated_returns[sim] = path;
        result.terminal_values[sim] = path.back();
        result.max_drawdowns[sim] = calculate_max_drawdown(path);
    }

    // Compute returns for metrics
    std::vector<double> returns(config_.n_simulations);
    for (size_t i = 0; i < config_.n_simulations; ++i) {
        returns[i] = (result.terminal_values[i] - S0) / S0;
    }

    result.metrics.expected_return = mean(returns);
    result.metrics.volatility = stddev(returns);
    result.metrics.sharpe_ratio = calculate_sharpe(returns, config_.risk_free_rate);
    result.metrics.var_95 = calculate_var(returns, 0.95);
    result.metrics.var_99 = calculate_var(returns, 0.99);
    result.metrics.cvar_95 = calculate_cvar(returns, 0.95);
    result.metrics.cvar_99 = calculate_cvar(returns, 0.99);
    result.metrics.skewness = skewness(returns);
    result.metrics.kurtosis = kurtosis(returns);
    result.metrics.max_drawdown = mean(result.max_drawdowns);

    result.convergence_error = stddev(result.terminal_values) /
                               std::sqrt(static_cast<double>(config_.n_simulations));

    return result;
}

std::vector<std::vector<double>> MonteCarloEngine::generate_paths(
    const Portfolio& portfolio,
    size_t n_paths
) {
    auto original_n = config_.n_simulations;
    config_.n_simulations = n_paths;

    auto result = simulate_portfolio(portfolio);

    config_.n_simulations = original_n;

    return result.simulated_returns;
}

// Parallel Monte Carlo Engine
ParallelMonteCarloEngine::ParallelMonteCarloEngine(const SimulationConfig& config, size_t n_threads)
    : config_(config)
    , n_threads_(n_threads == 0 ? std::thread::hardware_concurrency() : n_threads)
{}

SimulationResult ParallelMonteCarloEngine::simulate_portfolio(const Portfolio& portfolio) {
    size_t sims_per_thread = config_.n_simulations / n_threads_;
    size_t remainder = config_.n_simulations % n_threads_;

    std::vector<std::future<SimulationResult>> futures;

    for (size_t t = 0; t < n_threads_; ++t) {
        size_t n_sims = sims_per_thread + (t < remainder ? 1 : 0);
        unsigned int seed = (config_.seed == 0) ?
            std::random_device{}() : config_.seed + static_cast<unsigned int>(t);

        futures.push_back(std::async(std::launch::async, [=]() {
            SimulationConfig thread_config = config_;
            thread_config.n_simulations = n_sims;
            thread_config.seed = seed;

            MonteCarloEngine engine(thread_config);
            return engine.simulate_portfolio(portfolio);
        }));
    }

    // Aggregate results
    SimulationResult final_result;
    final_result.terminal_values.reserve(config_.n_simulations);

    for (auto& future : futures) {
        auto result = future.get();
        for (const auto& path : result.simulated_returns) {
            final_result.simulated_returns.push_back(path);
        }
        for (double tv : result.terminal_values) {
            final_result.terminal_values.push_back(tv);
        }
        for (double dd : result.max_drawdowns) {
            final_result.max_drawdowns.push_back(dd);
        }
    }

    // Recompute metrics on aggregated results
    std::vector<double> returns(final_result.terminal_values.size());
    for (size_t i = 0; i < final_result.terminal_values.size(); ++i) {
        returns[i] = final_result.terminal_values[i] - 1.0;
    }

    final_result.metrics.expected_return = MonteCarloEngine::mean(returns);
    final_result.metrics.volatility = MonteCarloEngine::stddev(returns);
    final_result.metrics.sharpe_ratio = MonteCarloEngine::calculate_sharpe(returns, config_.risk_free_rate);
    final_result.metrics.var_95 = MonteCarloEngine::calculate_var(returns, 0.95);
    final_result.metrics.var_99 = MonteCarloEngine::calculate_var(returns, 0.99);
    final_result.metrics.cvar_95 = MonteCarloEngine::calculate_cvar(returns, 0.95);
    final_result.metrics.cvar_99 = MonteCarloEngine::calculate_cvar(returns, 0.99);
    final_result.metrics.skewness = MonteCarloEngine::skewness(returns);
    final_result.metrics.kurtosis = MonteCarloEngine::kurtosis(returns);
    final_result.metrics.max_drawdown = MonteCarloEngine::mean(final_result.max_drawdowns);

    final_result.convergence_error = MonteCarloEngine::stddev(final_result.terminal_values) /
                                     std::sqrt(static_cast<double>(config_.n_simulations));

    return final_result;
}

} // namespace qss
