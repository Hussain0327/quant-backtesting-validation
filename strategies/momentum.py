import pandas as pd
import numpy as np
from .base import Strategy


class MomentumStrategy(Strategy):
    def __init__(self, lookback=20):
        super().__init__('Momentum')
        self.lookback = lookback

    def generate_signals(self, data):
        df = data.copy()

        # calculate momentum as percent change over lookback period
        df['momentum'] = df['close'].pct_change(periods=self.lookback)

        df['signal'] = 0
        df.loc[df['momentum'] > 0, 'signal'] = 1
        df.loc[df['momentum'] < 0, 'signal'] = -1

        # TODO: add threshold so we dont trade on tiny momentum
        # TODO: maybe add volume confirmation

        return df

    def get_params(self):
        return {'lookback': self.lookback}


# TODO: implement mean reversion strategy
# TODO: bollinger bands strategy
