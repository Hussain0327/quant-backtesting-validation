import sqlite3
import pandas as pd
from pathlib import Path


class Database:
    def __init__(self, db_path='data/trading.db'):
        self.db_path = db_path
        Path(db_path).parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(db_path)

    def save_prices(self, df, ticker):
        df = df.copy()
        df['ticker'] = ticker
        df.to_sql('prices', self.conn, if_exists='append', index=False)

    def load_prices(self, ticker):
        query = f"SELECT * FROM prices WHERE ticker = '{ticker}'"
        try:
            df = pd.read_sql(query, self.conn)
            return df
        except:
            return None

    def get_tickers(self):
        try:
            df = pd.read_sql("SELECT DISTINCT ticker FROM prices", self.conn)
            return df['ticker'].tolist()
        except:
            return []

    def close(self):
        self.conn.close()


# TODO: add method to delete old data
# TODO: handle duplicates better
