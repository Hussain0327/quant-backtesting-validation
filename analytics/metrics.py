import numpy as np
import pandas as pd


def sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252
    if excess_returns.std() == 0:
        return 0
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def max_drawdown(equity_curve):
    peak = equity_curve['equity'].expanding().max()
    drawdown = (equity_curve['equity'] - peak) / peak
    return drawdown.min() * 100


def win_rate(trades):
    if not trades:
        return 0

    profits = []
    for i in range(0, len(trades) - 1, 2):
        if i + 1 < len(trades):
            buy = trades[i]
            sell = trades[i + 1]
            profit = sell['price'] - buy['price']
            profits.append(profit > 0)

    if not profits:
        return 0
    return sum(profits) / len(profits) * 100


def calculate_metrics(results):
    equity_df = results['equity_curve']
    trades = results['trades']

    if len(equity_df) < 2:
        return {}

    equity_df['returns'] = equity_df['equity'].pct_change()

    return {
        'total_return': results['return_pct'],
        'sharpe': sharpe_ratio(equity_df['returns'].dropna()),
        'max_drawdown': max_drawdown(equity_df),
        'win_rate': win_rate(trades),
        'num_trades': len(trades),
        'final_equity': results['final_equity']
    }


# TODO: add sortino ratio
# TODO: add calmar ratio
