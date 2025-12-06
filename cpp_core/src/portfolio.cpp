#include "portfolio.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

namespace qss {

void Portfolio::add_asset(const std::string& symbol, double weight,
                          double expected_return, double volatility) {
    assets_.push_back({symbol, weight, expected_return, volatility});
}

void Portfolio::set_weights(const std::vector<double>& weights) {
    if (weights.size() != assets_.size()) {
        throw std::invalid_argument("Weight vector size must match number of assets");
    }
    for (size_t i = 0; i < weights.size(); ++i) {
        assets_[i].weight = weights[i];
    }
}

void Portfolio::normalize_weights() {
    double total = 0.0;
    for (const auto& asset : assets_) {
        total += asset.weight;
    }
    if (total > 0) {
        for (auto& asset : assets_) {
            asset.weight /= total;
        }
    }
}

std::vector<double> Portfolio::get_weights() const {
    std::vector<double> weights(assets_.size());
    for (size_t i = 0; i < assets_.size(); ++i) {
        weights[i] = assets_[i].weight;
    }
    return weights;
}

std::vector<double> Portfolio::get_expected_returns() const {
    std::vector<double> returns(assets_.size());
    for (size_t i = 0; i < assets_.size(); ++i) {
        returns[i] = assets_[i].expected_return;
    }
    return returns;
}

std::vector<double> Portfolio::get_volatilities() const {
    std::vector<double> vols(assets_.size());
    for (size_t i = 0; i < assets_.size(); ++i) {
        vols[i] = assets_[i].volatility;
    }
    return vols;
}

void Portfolio::set_covariance_matrix(const std::vector<std::vector<double>>& cov_matrix) {
    if (cov_matrix.size() != assets_.size()) {
        throw std::invalid_argument("Covariance matrix size must match number of assets");
    }
    covariance_matrix_ = cov_matrix;
}

std::vector<std::vector<double>> Portfolio::correlation_matrix() const {
    size_t n = assets_.size();
    std::vector<std::vector<double>> corr(n, std::vector<double>(n));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double var_i = covariance_matrix_[i][i];
            double var_j = covariance_matrix_[j][j];
            if (var_i > 0 && var_j > 0) {
                corr[i][j] = covariance_matrix_[i][j] / (std::sqrt(var_i) * std::sqrt(var_j));
            } else {
                corr[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }
    }
    return corr;
}

double Portfolio::portfolio_return() const {
    double ret = 0.0;
    for (const auto& asset : assets_) {
        ret += asset.weight * asset.expected_return;
    }
    return ret;
}

double Portfolio::portfolio_variance() const {
    if (covariance_matrix_.empty()) {
        throw std::runtime_error("Covariance matrix not set");
    }

    size_t n = assets_.size();
    double variance = 0.0;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            variance += assets_[i].weight * assets_[j].weight * covariance_matrix_[i][j];
        }
    }

    return variance;
}

double Portfolio::portfolio_volatility() const {
    return std::sqrt(portfolio_variance());
}

} // namespace qss
