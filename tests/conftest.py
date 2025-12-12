import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_price_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_days = 100

    dates = pd.date_range(start="2025-01-01", periods=n_days, freq="D")

    # Generate realistic price movement
    initial_price = 100
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = initial_price * np.cumprod(1 + returns)

    # Create OHLCV data
    data = pd.DataFrame(
        {
            "open": prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
            "high": prices * (1 + np.random.uniform(0, 0.02, n_days)),
            "low": prices * (1 - np.random.uniform(0, 0.02, n_days)),
            "close": prices,
            "volume": np.random.randint(1000000, 10000000, n_days),
        },
        index=dates,
    )

    return data


@pytest.fixture
def sample_price_data_short():
    """Generate short sample data (30 days) for edge case testing."""
    np.random.seed(42)
    n_days = 30

    dates = pd.date_range(start="2025-01-01", periods=n_days, freq="D")
    initial_price = 100
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = initial_price * np.cumprod(1 + returns)

    data = pd.DataFrame(
        {
            "open": prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
            "high": prices * (1 + np.random.uniform(0, 0.02, n_days)),
            "low": prices * (1 - np.random.uniform(0, 0.02, n_days)),
            "close": prices,
            "volume": np.random.randint(1000000, 10000000, n_days),
        },
        index=dates,
    )

    return data


@pytest.fixture
def sample_returns():
    """Generate sample daily returns for statistical testing."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.001, 0.02, 100))


@pytest.fixture
def sample_equity_curve():
    """Generate sample equity curve for metrics testing."""
    np.random.seed(42)
    n_days = 100
    initial_capital = 10000

    returns = np.random.normal(0.001, 0.02, n_days)
    equity = initial_capital * np.cumprod(1 + returns)

    return pd.Series(equity)
