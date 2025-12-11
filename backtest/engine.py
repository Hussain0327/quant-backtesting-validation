import pandas as pd
import numpy as np
from .costs import calculate_costs


class BacktestEngine:
    def __init__(self, initial_capital=10000, commission=0.001, slippage=0.0005):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(self, data, strategy, train_pct=0.7):
        df = data.copy()
        n = len(df)
        train_end = int(n * train_pct)

        train_data = df.iloc[:train_end]
        test_data = df.iloc[train_end:]

        # generate signals on train, evaluate on test
        train_signals = strategy.generate_signals(train_data)
        test_signals = strategy.generate_signals(test_data)

        train_results = self._simulate(train_signals)
        test_results = self._simulate(test_signals)

        return {
            'train': train_results,
            'test': test_results,
            'strategy': strategy.name,
            'params': strategy.get_params()
        }

    def _simulate(self, data):
        df = data.copy()
        df = df.dropna()

        capital = self.initial_capital
        position = 0
        trades = []
        equity_curve = []

        for i, row in df.iterrows():
            price = row['close']
            signal = row['signal']

            if signal == 1 and position == 0:
                # buy
                shares = capital / price
                cost = calculate_costs(capital, self.commission, self.slippage)
                capital -= cost
                position = shares
                trades.append({'type': 'buy', 'price': price, 'shares': shares, 'date': row.get('date', i)})

            elif signal == -1 and position > 0:
                # sell
                trade_value = position * price
                cost = calculate_costs(trade_value, self.commission, self.slippage)
                capital = trade_value - cost
                trades.append({'type': 'sell', 'price': price, 'shares': position, 'date': row.get('date', i)})
                position = 0

            # track equity
            equity = capital if position == 0 else position * price
            equity_curve.append({'date': row.get('date', i), 'equity': equity, 'price': price})

        equity_df = pd.DataFrame(equity_curve)
        final_equity = equity_curve[-1]['equity'] if equity_curve else self.initial_capital

        return {
            'equity_curve': equity_df,
            'trades': trades,
            'final_equity': final_equity,
            'return_pct': (final_equity - self.initial_capital) / self.initial_capital * 100,
            'num_trades': len(trades)
        }


# TODO: add short selling support
# TODO: position sizing options
