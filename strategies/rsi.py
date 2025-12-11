import pandas as pd
import numpy as np
from .base import Strategy


class RSIStrategy(Strategy):
    def __init__(self, period=14, oversold=30, overbought=70):
        super().__init__('RSI')
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def calculate_rsi(self, prices):
        delta = prices.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def generate_signals(self, data):
        df = data.copy()

        df['rsi'] = self.calculate_rsi(df['close'])
        df['signal'] = 0

        # buy when oversold
        df.loc[df['rsi'] < self.oversold, 'signal'] = 1
        # sell when overbought
        df.loc[df['rsi'] > self.overbought, 'signal'] = -1

        return df

    def get_params(self):
        return {
            'period': self.period,
            'oversold': self.oversold,
            'overbought': self.overbought
        }
