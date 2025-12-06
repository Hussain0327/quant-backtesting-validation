#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "random_generator.hpp"
#include "portfolio.hpp"
#include "monte_carlo.hpp"
#include "statistics.hpp"

namespace py = pybind11;

PYBIND11_MODULE(qss_core, m) {
    m.doc() = "Quantitative Systems Simulator - C++ Core Module";

    // RandomGenerator class
    py::class_<qss::RandomGenerator>(m, "RandomGenerator")
        .def(py::init<unsigned int>(), py::arg("seed") = 0)
        .def("set_seed", &qss::RandomGenerator::set_seed)
        .def("uniform", &qss::RandomGenerator::uniform,
             py::arg("min") = 0.0, py::arg("max") = 1.0)
        .def("normal", &qss::RandomGenerator::normal,
             py::arg("mean") = 0.0, py::arg("stddev") = 1.0)
        .def("lognormal", &qss::RandomGenerator::lognormal,
             py::arg("mean") = 0.0, py::arg("stddev") = 1.0)
        .def("student_t", &qss::RandomGenerator::student_t)
        .def("normal_vector", &qss::RandomGenerator::normal_vector,
             py::arg("n"), py::arg("mean") = 0.0, py::arg("stddev") = 1.0)
        .def("uniform_vector", &qss::RandomGenerator::uniform_vector,
             py::arg("n"), py::arg("min") = 0.0, py::arg("max") = 1.0)
        .def("correlated_normals", &qss::RandomGenerator::correlated_normals)
        .def("antithetic_normal", &qss::RandomGenerator::antithetic_normal,
             py::arg("n"), py::arg("mean") = 0.0, py::arg("stddev") = 1.0)
        .def("stratified_normal", &qss::RandomGenerator::stratified_normal,
             py::arg("n_strata"), py::arg("mean") = 0.0, py::arg("stddev") = 1.0);

    // Asset struct
    py::class_<qss::Asset>(m, "Asset")
        .def(py::init<>())
        .def_readwrite("symbol", &qss::Asset::symbol)
        .def_readwrite("weight", &qss::Asset::weight)
        .def_readwrite("expected_return", &qss::Asset::expected_return)
        .def_readwrite("volatility", &qss::Asset::volatility);

    // PortfolioMetrics struct
    py::class_<qss::PortfolioMetrics>(m, "PortfolioMetrics")
        .def(py::init<>())
        .def_readwrite("expected_return", &qss::PortfolioMetrics::expected_return)
        .def_readwrite("volatility", &qss::PortfolioMetrics::volatility)
        .def_readwrite("sharpe_ratio", &qss::PortfolioMetrics::sharpe_ratio)
        .def_readwrite("var_95", &qss::PortfolioMetrics::var_95)
        .def_readwrite("var_99", &qss::PortfolioMetrics::var_99)
        .def_readwrite("cvar_95", &qss::PortfolioMetrics::cvar_95)
        .def_readwrite("cvar_99", &qss::PortfolioMetrics::cvar_99)
        .def_readwrite("skewness", &qss::PortfolioMetrics::skewness)
        .def_readwrite("kurtosis", &qss::PortfolioMetrics::kurtosis)
        .def_readwrite("max_drawdown", &qss::PortfolioMetrics::max_drawdown);

    // Portfolio class
    py::class_<qss::Portfolio>(m, "Portfolio")
        .def(py::init<>())
        .def("add_asset", &qss::Portfolio::add_asset)
        .def("set_weights", &qss::Portfolio::set_weights)
        .def("normalize_weights", &qss::Portfolio::normalize_weights)
        .def("num_assets", &qss::Portfolio::num_assets)
        .def("assets", &qss::Portfolio::assets)
        .def("get_weights", &qss::Portfolio::get_weights)
        .def("get_expected_returns", &qss::Portfolio::get_expected_returns)
        .def("get_volatilities", &qss::Portfolio::get_volatilities)
        .def("set_covariance_matrix", &qss::Portfolio::set_covariance_matrix)
        .def("covariance_matrix", &qss::Portfolio::covariance_matrix)
        .def("correlation_matrix", &qss::Portfolio::correlation_matrix)
        .def("portfolio_return", &qss::Portfolio::portfolio_return)
        .def("portfolio_variance", &qss::Portfolio::portfolio_variance)
        .def("portfolio_volatility", &qss::Portfolio::portfolio_volatility);

    // SimulationConfig struct
    py::class_<qss::SimulationConfig>(m, "SimulationConfig")
        .def(py::init<>())
        .def_readwrite("n_simulations", &qss::SimulationConfig::n_simulations)
        .def_readwrite("time_horizon", &qss::SimulationConfig::time_horizon)
        .def_readwrite("dt", &qss::SimulationConfig::dt)
        .def_readwrite("risk_free_rate", &qss::SimulationConfig::risk_free_rate)
        .def_readwrite("use_antithetic", &qss::SimulationConfig::use_antithetic)
        .def_readwrite("use_stratified", &qss::SimulationConfig::use_stratified)
        .def_readwrite("seed", &qss::SimulationConfig::seed);

    // SimulationResult struct
    py::class_<qss::SimulationResult>(m, "SimulationResult")
        .def(py::init<>())
        .def_readwrite("simulated_returns", &qss::SimulationResult::simulated_returns)
        .def_readwrite("terminal_values", &qss::SimulationResult::terminal_values)
        .def_readwrite("path_maxima", &qss::SimulationResult::path_maxima)
        .def_readwrite("path_minima", &qss::SimulationResult::path_minima)
        .def_readwrite("max_drawdowns", &qss::SimulationResult::max_drawdowns)
        .def_readwrite("metrics", &qss::SimulationResult::metrics)
        .def_readwrite("convergence_error", &qss::SimulationResult::convergence_error);

    // MonteCarloEngine class
    py::class_<qss::MonteCarloEngine>(m, "MonteCarloEngine")
        .def(py::init<const qss::SimulationConfig&>(),
             py::arg("config") = qss::SimulationConfig{})
        .def("simulate_portfolio", &qss::MonteCarloEngine::simulate_portfolio)
        .def("simulate_gbm", &qss::MonteCarloEngine::simulate_gbm)
        .def("generate_paths", &qss::MonteCarloEngine::generate_paths)
        .def("set_config", &qss::MonteCarloEngine::set_config)
        .def("config", &qss::MonteCarloEngine::config)
        .def_static("calculate_var", &qss::MonteCarloEngine::calculate_var)
        .def_static("calculate_cvar", &qss::MonteCarloEngine::calculate_cvar)
        .def_static("calculate_sharpe", &qss::MonteCarloEngine::calculate_sharpe)
        .def_static("calculate_max_drawdown", &qss::MonteCarloEngine::calculate_max_drawdown)
        .def_static("mean", &qss::MonteCarloEngine::mean)
        .def_static("variance", &qss::MonteCarloEngine::variance)
        .def_static("stddev", &qss::MonteCarloEngine::stddev)
        .def_static("skewness", &qss::MonteCarloEngine::skewness)
        .def_static("kurtosis", &qss::MonteCarloEngine::kurtosis);

    // ParallelMonteCarloEngine class
    py::class_<qss::ParallelMonteCarloEngine>(m, "ParallelMonteCarloEngine")
        .def(py::init<const qss::SimulationConfig&, size_t>(),
             py::arg("config"), py::arg("n_threads") = 0)
        .def("simulate_portfolio", &qss::ParallelMonteCarloEngine::simulate_portfolio);

    // Statistics submodule
    auto stats = m.def_submodule("stats", "Statistical functions");

    stats.def("mean", &qss::stats::mean);
    stats.def("median", &qss::stats::median);
    stats.def("variance", &qss::stats::variance, py::arg("data"), py::arg("sample") = true);
    stats.def("stddev", &qss::stats::stddev, py::arg("data"), py::arg("sample") = true);
    stats.def("skewness", &qss::stats::skewness);
    stats.def("kurtosis", &qss::stats::kurtosis);
    stats.def("quantile", &qss::stats::quantile);
    stats.def("covariance", &qss::stats::covariance,
              py::arg("x"), py::arg("y"), py::arg("sample") = true);
    stats.def("correlation", &qss::stats::correlation);
    stats.def("covariance_matrix", &qss::stats::covariance_matrix);
    stats.def("correlation_matrix", &qss::stats::correlation_matrix);

    // Test result struct
    py::class_<qss::stats::TestResult>(stats, "TestResult")
        .def(py::init<>())
        .def_readwrite("statistic", &qss::stats::TestResult::statistic)
        .def_readwrite("p_value", &qss::stats::TestResult::p_value)
        .def_readwrite("reject_null", &qss::stats::TestResult::reject_null)
        .def_readwrite("description", &qss::stats::TestResult::description);

    // Confidence interval struct
    py::class_<qss::stats::ConfidenceInterval>(stats, "ConfidenceInterval")
        .def(py::init<>())
        .def_readwrite("lower", &qss::stats::ConfidenceInterval::lower)
        .def_readwrite("upper", &qss::stats::ConfidenceInterval::upper)
        .def_readwrite("confidence_level", &qss::stats::ConfidenceInterval::confidence_level)
        .def_readwrite("point_estimate", &qss::stats::ConfidenceInterval::point_estimate);

    // Regression result struct
    py::class_<qss::stats::RegressionResult>(stats, "RegressionResult")
        .def(py::init<>())
        .def_readwrite("alpha", &qss::stats::RegressionResult::alpha)
        .def_readwrite("beta", &qss::stats::RegressionResult::beta)
        .def_readwrite("r_squared", &qss::stats::RegressionResult::r_squared)
        .def_readwrite("std_error_alpha", &qss::stats::RegressionResult::std_error_alpha)
        .def_readwrite("std_error_beta", &qss::stats::RegressionResult::std_error_beta)
        .def_readwrite("t_stat_alpha", &qss::stats::RegressionResult::t_stat_alpha)
        .def_readwrite("t_stat_beta", &qss::stats::RegressionResult::t_stat_beta)
        .def_readwrite("p_value_alpha", &qss::stats::RegressionResult::p_value_alpha)
        .def_readwrite("p_value_beta", &qss::stats::RegressionResult::p_value_beta);

    // Hypothesis tests
    stats.def("t_test_one_sample", &qss::stats::t_test_one_sample);
    stats.def("t_test_two_sample", &qss::stats::t_test_two_sample,
              py::arg("sample1"), py::arg("sample2"), py::arg("equal_variance") = true);
    stats.def("jarque_bera_test", &qss::stats::jarque_bera_test);
    stats.def("shapiro_wilk_test", &qss::stats::shapiro_wilk_test);

    // Confidence intervals
    stats.def("mean_ci", &qss::stats::mean_ci, py::arg("data"), py::arg("confidence") = 0.95);
    stats.def("variance_ci", &qss::stats::variance_ci, py::arg("data"), py::arg("confidence") = 0.95);
    stats.def("proportion_ci", &qss::stats::proportion_ci,
              py::arg("successes"), py::arg("trials"), py::arg("confidence") = 0.95);

    // Regression
    stats.def("linear_regression", &qss::stats::linear_regression);

    // Power analysis
    stats.def("power_t_test", &qss::stats::power_t_test,
              py::arg("effect_size"), py::arg("sample_size"), py::arg("alpha") = 0.05);
    stats.def("required_sample_size", &qss::stats::required_sample_size,
              py::arg("effect_size"), py::arg("power") = 0.8, py::arg("alpha") = 0.05);

    // Normal distribution submodule
    auto normal = stats.def_submodule("normal", "Normal distribution functions");
    normal.def("pdf", &qss::stats::normal::pdf,
               py::arg("x"), py::arg("mean") = 0.0, py::arg("stddev") = 1.0);
    normal.def("cdf", &qss::stats::normal::cdf,
               py::arg("x"), py::arg("mean") = 0.0, py::arg("stddev") = 1.0);
    normal.def("ppf", &qss::stats::normal::ppf,
               py::arg("p"), py::arg("mean") = 0.0, py::arg("stddev") = 1.0);

    // Student-t distribution submodule
    auto student = stats.def_submodule("student_t", "Student-t distribution functions");
    student.def("pdf", &qss::stats::student_t::pdf);
    student.def("cdf", &qss::stats::student_t::cdf);
}
