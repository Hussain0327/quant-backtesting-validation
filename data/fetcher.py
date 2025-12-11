import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def fetch_data(ticker, start, end):
    """gets stock data from yfinance"""
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.reset_index()
    # flatten multi-index columns if needed
    if isinstance(df.columns[0], tuple):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df


def fetch_multiple(tickers, start, end):
    data = {}
    for t in tickers:
        data[t] = fetch_data(t, start, end)
    return data


# TODO: add caching so we dont hit the api every time
