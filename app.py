import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from data.fetcher import fetch_data
from strategies import (
    MovingAverageCrossover, RSIStrategy, MomentumStrategy,
    PairsTradingStrategy, SpreadMeanReversionStrategy
)
from backtest.engine import BacktestEngine
from analytics.metrics import calculate_metrics
from analytics.significance import (
    bootstrap_sharpe_confidence_interval,
    permutation_test_vs_baseline,
    monte_carlo_under_null,
    test_return_distribution
)


st.set_page_config(page_title='Quantitative Research Framework', layout='wide')
st.title('Quantitative Trading Research Framework')

st.markdown("""
<style>
.disclaimer {
    background-color: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 5px;
    padding: 10px;
    margin-bottom: 20px;
}
.significant { color: #28a745; font-weight: bold; }
.not-significant { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
<strong>Research Framework Disclaimer:</strong> This platform provides baseline strategy implementations
for educational and research purposes. Backtest results do not guarantee future performance.
Statistical significance tests help identify potential edges, but all results require
out-of-sample validation before any real-world application.
</div>
""", unsafe_allow_html=True)

# sidebar
st.sidebar.header('Research Parameters')

ticker = st.sidebar.text_input('Ticker Symbol', value='AAPL')
start_date = st.sidebar.date_input('Start Date', value=datetime.now() - timedelta(days=365*2))
end_date = st.sidebar.date_input('End Date', value=datetime.now())

strategy_name = st.sidebar.selectbox(
    'Strategy',
    ['MA Crossover', 'RSI', 'Momentum', 'Pairs Trading', 'Mean Reversion (Bollinger)']
)

# strategy params
if strategy_name == 'MA Crossover':
    st.sidebar.markdown("*Trend-following: trades crossovers of two moving averages*")
    short_window = st.sidebar.slider('Short MA Window', 5, 50, 20)
    long_window = st.sidebar.slider('Long MA Window', 20, 200, 50)
    strategy = MovingAverageCrossover(short_window, long_window)
elif strategy_name == 'RSI':
    st.sidebar.markdown("*Mean reversion: buys oversold, sells overbought*")
    period = st.sidebar.slider('RSI Period', 5, 30, 14)
    oversold = st.sidebar.slider('Oversold Threshold', 10, 40, 30)
    overbought = st.sidebar.slider('Overbought Threshold', 60, 90, 70)
    strategy = RSIStrategy(period, oversold, overbought)
elif strategy_name == 'Momentum':
    st.sidebar.markdown("*Trend-following: trades in direction of recent price movement*")
    lookback = st.sidebar.slider('Lookback Period', 5, 60, 20)
    strategy = MomentumStrategy(lookback)
elif strategy_name == 'Pairs Trading':
    st.sidebar.markdown("*Statistical arbitrage: trades z-score of spread vs trend*")
    lookback = st.sidebar.slider('Lookback Period', 10, 50, 20)
    entry_z = st.sidebar.slider('Entry Z-Score', 1.0, 3.0, 2.0)
    exit_z = st.sidebar.slider('Exit Z-Score', 0.0, 1.5, 0.5)
    strategy = PairsTradingStrategy(lookback, entry_z, exit_z)
else:
    st.sidebar.markdown("*Mean reversion: trades Bollinger Band breakouts*")
    lookback = st.sidebar.slider('Lookback Period', 10, 50, 20)
    num_std = st.sidebar.slider('Num Std Deviations', 1.0, 3.0, 2.0)
    strategy = SpreadMeanReversionStrategy(lookback, num_std)

initial_capital = st.sidebar.number_input('Initial Capital ($)', value=10000)
train_split = st.sidebar.slider('Train/Test Split', 0.5, 0.9, 0.7)

st.sidebar.markdown("---")
st.sidebar.subheader("Statistical Testing")
run_significance = st.sidebar.checkbox('Run Significance Tests', value=True)
n_bootstrap = st.sidebar.slider('Bootstrap Samples', 1000, 10000, 5000, help="More samples = more accurate confidence intervals")

if st.sidebar.button('Run Backtest', type='primary'):
    with st.spinner('Fetching market data...'):
        data = fetch_data(ticker, str(start_date), str(end_date))

    if data is None or len(data) == 0:
        st.error('No data found for this ticker/date range')
    else:
        engine = BacktestEngine(initial_capital=initial_capital)
        results = engine.run(data, strategy, train_pct=train_split)

        train_metrics = calculate_metrics(results['train'])
        test_metrics = calculate_metrics(results['test'])

        # Main results
        st.header('Backtest Results')

        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Training Period')
            st.metric('Total Return', f"{train_metrics.get('total_return', 0):.2f}%")
            st.metric('Sharpe Ratio', f"{train_metrics.get('sharpe', 0):.2f}")
            st.metric('Max Drawdown', f"{train_metrics.get('max_drawdown', 0):.2f}%")
            st.metric('Win Rate', f"{train_metrics.get('win_rate', 0):.1f}%")
            st.metric('Number of Trades', train_metrics.get('num_trades', 0))

        with col2:
            st.subheader('Test Period (Out-of-Sample)')
            st.metric('Total Return', f"{test_metrics.get('total_return', 0):.2f}%")
            st.metric('Sharpe Ratio', f"{test_metrics.get('sharpe', 0):.2f}")
            st.metric('Max Drawdown', f"{test_metrics.get('max_drawdown', 0):.2f}%")
            st.metric('Win Rate', f"{test_metrics.get('win_rate', 0):.1f}%")
            st.metric('Number of Trades', test_metrics.get('num_trades', 0))

        # Statistical Significance Testing
        if run_significance:
            st.header('Statistical Significance Analysis')
            st.markdown("""
            *These tests help determine if strategy performance is statistically significant
            or likely due to random chance. A strategy should pass multiple tests before
            being considered for further research.*
            """)

            with st.spinner('Running statistical tests...'):
                test_equity = results['test']['equity_curve']
                test_returns = test_equity['equity'].pct_change().dropna()
                test_prices = test_equity['price']

                # Benchmark returns (buy and hold)
                benchmark_returns = test_prices.pct_change().dropna()

                # Bootstrap Sharpe CI
                sharpe_ci = bootstrap_sharpe_confidence_interval(
                    test_returns, n_bootstrap=n_bootstrap
                )

                # Permutation test vs benchmark
                perm_test = permutation_test_vs_baseline(
                    test_returns, benchmark_returns, n_permutations=n_bootstrap
                )

                # Monte Carlo null
                mc_test = monte_carlo_under_null(
                    test_prices,
                    n_simulations=min(n_bootstrap, 2000),
                    strategy_return=results['test']['return_pct']
                )

                # Return distribution
                dist_test = test_return_distribution(test_returns)

            # Display results in columns
            col1, col2 = st.columns(2)

            with col1:
                st.subheader('1. Sharpe Ratio Confidence Interval')
                sharpe_sig = not sharpe_ci.get('ci_includes_zero', True)

                st.markdown(f"""
                - **Point Estimate:** {sharpe_ci['sharpe']:.3f}
                - **95% CI:** [{sharpe_ci['ci_lower']:.3f}, {sharpe_ci['ci_upper']:.3f}]
                - **Standard Error:** {sharpe_ci['std_error']:.3f}
                """)

                if sharpe_sig:
                    st.success("CI excludes zero - Sharpe is statistically significant")
                else:
                    st.warning("CI includes zero - cannot reject null hypothesis")

                st.subheader('2. vs Buy-and-Hold Benchmark')
                bench_sig = perm_test.get('significant_at_05', False)

                st.markdown(f"""
                - **Observed Difference:** {perm_test['observed_diff']:.4f} (annualized)
                - **p-value:** {perm_test['p_value']:.4f}
                """)

                if bench_sig:
                    st.success(f"p < 0.05 - Strategy significantly outperforms buy-and-hold")
                else:
                    st.warning("p >= 0.05 - No significant difference from buy-and-hold")

            with col2:
                st.subheader('3. vs Random Trading')
                random_sig = mc_test.get('significant_at_05', False)

                st.markdown(f"""
                - **Strategy Return:** {mc_test.get('strategy_return', 0):.2f}%
                - **Random Mean:** {mc_test['null_mean']:.2f}%
                - **Random 95th Percentile:** {mc_test['null_95th_percentile']:.2f}%
                - **Percentile Rank:** {mc_test.get('percentile_rank', 50):.1f}%
                - **p-value:** {mc_test.get('p_value', 1):.4f}
                """)

                if random_sig:
                    st.success("Strategy outperforms random trading significantly")
                else:
                    st.warning("Strategy does not significantly beat random trading")

                st.subheader('4. Return Distribution')
                st.markdown(f"""
                - **Daily Mean:** {dist_test.get('mean_daily', 0)*100:.4f}%
                - **Daily Std:** {dist_test.get('std_daily', 0)*100:.4f}%
                - **Skewness:** {dist_test.get('skewness', 0):.3f}
                - **Excess Kurtosis:** {dist_test.get('excess_kurtosis', 0):.3f}
                - **95% VaR:** {dist_test.get('var_95', 0)*100:.2f}%
                """)

                if dist_test.get('is_fat_tailed', False):
                    st.info("Fat tails detected - extreme events more likely than normal")
                if dist_test.get('is_negatively_skewed', False):
                    st.info("Negative skew - larger downside risk")

            # Overall Assessment
            st.header('Overall Statistical Assessment')

            tests_passed = sum([
                sharpe_sig,
                bench_sig,
                random_sig
            ])

            if tests_passed == 3:
                st.success("""
                **Strong Evidence of Edge** - Strategy passes all three significance tests.
                This warrants further investigation with walk-forward analysis and
                additional out-of-sample testing.
                """)
            elif tests_passed == 2:
                st.warning("""
                **Moderate Evidence** - Strategy passes 2/3 tests. Results are promising
                but not conclusive. Consider parameter sensitivity analysis and
                longer test periods.
                """)
            elif tests_passed == 1:
                st.error("""
                **Weak Evidence** - Strategy passes only 1/3 tests. Performance is likely
                due to noise or overfitting. Significant modifications needed.
                """)
            else:
                st.error("""
                **No Statistical Evidence** - Strategy fails all significance tests.
                Returns are indistinguishable from random chance. Do not use.
                """)

        # Equity curve chart
        st.header('Equity Curve')
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.7, 0.3],
            subplot_titles=('Portfolio Equity', 'Asset Price with Trade Signals')
        )

        test_equity = results['test']['equity_curve']

        # Equity curve
        fig.add_trace(go.Scatter(
            x=test_equity['date'],
            y=test_equity['equity'],
            name='Strategy Equity',
            line=dict(color='blue')
        ), row=1, col=1)

        # Buy and hold comparison
        initial_shares = initial_capital / test_equity['price'].iloc[0]
        buy_hold_equity = initial_shares * test_equity['price']
        fig.add_trace(go.Scatter(
            x=test_equity['date'],
            y=buy_hold_equity,
            name='Buy & Hold',
            line=dict(color='gray', dash='dash')
        ), row=1, col=1)

        # Price chart
        fig.add_trace(go.Scatter(
            x=test_equity['date'],
            y=test_equity['price'],
            name='Price',
            line=dict(color='black')
        ), row=2, col=1)

        # Trade markers
        trades = results['test']['trades']
        buys = [t for t in trades if t['type'] == 'buy']
        sells = [t for t in trades if t['type'] == 'sell']

        if buys:
            fig.add_trace(go.Scatter(
                x=[t['date'] for t in buys],
                y=[t['price'] for t in buys],
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='green'),
                name='Buy Signal'
            ), row=2, col=1)

        if sells:
            fig.add_trace(go.Scatter(
                x=[t['date'] for t in sells],
                y=[t['price'] for t in sells],
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='red'),
                name='Sell Signal'
            ), row=2, col=1)

        fig.update_layout(height=700, showlegend=True)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Trades table
        st.header('Trade Log')
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.write('No trades executed in test period')

        # Strategy parameters
        with st.expander("Strategy Parameters"):
            st.json(results['params'])
