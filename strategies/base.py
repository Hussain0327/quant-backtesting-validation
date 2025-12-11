from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def generate_signals(self, data):
        """returns df with 'signal' column: 1 = buy, -1 = sell, 0 = hold"""
        pass

    def get_params(self):
        return {}
