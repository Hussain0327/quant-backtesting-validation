import pandas as pd
from .base import Strategy


class MovingAverageCrossover(Strategy):
    def __init__(self, short_window=20, long_window=50):
        super().__init__('MA Crossover')
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        df = data.copy()

        df['sma_short'] = df['close'].rolling(window=self.short_window).mean()
        df['sma_long'] = df['close'].rolling(window=self.long_window).mean()

        df['signal'] = 0

        # buy when short crosses above long
        df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 1
        # sell when short crosses below long
        df.loc[df['sma_short'] < df['sma_long'], 'signal'] = -1

        # only keep actual crossover points
        df['signal'] = df['signal'].diff()
        df['signal'] = df['signal'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        return df

    def get_params(self):
        return {
            'short_window': self.short_window,
            'long_window': self.long_window
        }
