#ifndef PORTFOLIO_HPP
#define PORTFOLIO_HPP

#include <vector>
#include <string>
#include <unordered_map>

namespace qss {

struct Asset {
    std::string symbol;
    double weight;
    double expected_return;
    double volatility;
};

struct PortfolioMetrics {
    double expected_return;
    double volatility;
    double sharpe_ratio;
    double var_95;
    double var_99;
    double cvar_95;
    double cvar_99;
    double skewness;
    double kurtosis;
    double max_drawdown;
};

class Portfolio {
public:
    Portfolio() = default;

    // Asset management
    void add_asset(const std::string& symbol, double weight,
                   double expected_return, double volatility);
    void set_weights(const std::vector<double>& weights);
    void normalize_weights();

    // Getters
    size_t num_assets() const { return assets_.size(); }
    const std::vector<Asset>& assets() const { return assets_; }
    std::vector<double> get_weights() const;
    std::vector<double> get_expected_returns() const;
    std::vector<double> get_volatilities() const;

    // Covariance matrix
    void set_covariance_matrix(const std::vector<std::vector<double>>& cov_matrix);
    const std::vector<std::vector<double>>& covariance_matrix() const { return covariance_matrix_; }
    std::vector<std::vector<double>> correlation_matrix() const;

    // Portfolio calculations
    double portfolio_return() const;
    double portfolio_variance() const;
    double portfolio_volatility() const;

private:
    std::vector<Asset> assets_;
    std::vector<std::vector<double>> covariance_matrix_;
};

} // namespace qss

#endif // PORTFOLIO_HPP
