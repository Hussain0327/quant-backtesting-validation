import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from data.fetcher import fetch_data
from strategies import MovingAverageCrossover, RSIStrategy, MomentumStrategy
from backtest.engine import BacktestEngine
from analytics.metrics import calculate_metrics


st.set_page_config(page_title='trading backtest', layout='wide')
st.title('algorithmic trading research platform')

# sidebar
st.sidebar.header('settings')

ticker = st.sidebar.text_input('ticker', value='AAPL')
start_date = st.sidebar.date_input('start date', value=datetime.now() - timedelta(days=365*2))
end_date = st.sidebar.date_input('end date', value=datetime.now())

strategy_name = st.sidebar.selectbox('strategy', ['MA Crossover', 'RSI', 'Momentum'])

# strategy params
if strategy_name == 'MA Crossover':
    short_window = st.sidebar.slider('short window', 5, 50, 20)
    long_window = st.sidebar.slider('long window', 20, 200, 50)
    strategy = MovingAverageCrossover(short_window, long_window)
elif strategy_name == 'RSI':
    period = st.sidebar.slider('rsi period', 5, 30, 14)
    oversold = st.sidebar.slider('oversold', 10, 40, 30)
    overbought = st.sidebar.slider('overbought', 60, 90, 70)
    strategy = RSIStrategy(period, oversold, overbought)
else:
    lookback = st.sidebar.slider('lookback', 5, 60, 20)
    strategy = MomentumStrategy(lookback)

initial_capital = st.sidebar.number_input('initial capital', value=10000)
train_split = st.sidebar.slider('train/test split', 0.5, 0.9, 0.7)

if st.sidebar.button('run backtest'):
    with st.spinner('fetching data...'):
        data = fetch_data(ticker, str(start_date), str(end_date))

    if data is None or len(data) == 0:
        st.error('no data found')
    else:
        engine = BacktestEngine(initial_capital=initial_capital)
        results = engine.run(data, strategy, train_pct=train_split)

        train_metrics = calculate_metrics(results['train'])
        test_metrics = calculate_metrics(results['test'])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader('train results')
            st.metric('return', f"{train_metrics.get('total_return', 0):.2f}%")
            st.metric('sharpe', f"{train_metrics.get('sharpe', 0):.2f}")
            st.metric('max drawdown', f"{train_metrics.get('max_drawdown', 0):.2f}%")
            st.metric('win rate', f"{train_metrics.get('win_rate', 0):.1f}%")

        with col2:
            st.subheader('test results')
            st.metric('return', f"{test_metrics.get('total_return', 0):.2f}%")
            st.metric('sharpe', f"{test_metrics.get('sharpe', 0):.2f}")
            st.metric('max drawdown', f"{test_metrics.get('max_drawdown', 0):.2f}%")
            st.metric('win rate', f"{test_metrics.get('win_rate', 0):.1f}%")

        # equity curve chart
        st.subheader('equity curve')
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])

        test_equity = results['test']['equity_curve']
        fig.add_trace(go.Scatter(x=test_equity['date'], y=test_equity['equity'], name='equity'), row=1, col=1)
        fig.add_trace(go.Scatter(x=test_equity['date'], y=test_equity['price'], name='price'), row=2, col=1)

        # add trade markers
        trades = results['test']['trades']
        buys = [t for t in trades if t['type'] == 'buy']
        sells = [t for t in trades if t['type'] == 'sell']

        if buys:
            fig.add_trace(go.Scatter(
                x=[t['date'] for t in buys],
                y=[t['price'] for t in buys],
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                name='buy'
            ), row=2, col=1)

        if sells:
            fig.add_trace(go.Scatter(
                x=[t['date'] for t in sells],
                y=[t['price'] for t in sells],
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                name='sell'
            ), row=2, col=1)

        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # trades table
        st.subheader('trades')
        if trades:
            st.dataframe(pd.DataFrame(trades))
        else:
            st.write('no trades')


# TODO: add parameter optimization
# TODO: add strategy comparison feature
