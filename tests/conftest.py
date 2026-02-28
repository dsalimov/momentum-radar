"""
conftest.py – Shared pytest fixtures for Momentum Signal Radar tests.
"""

import pandas as pd
import pytest
from datetime import datetime, timezone


@pytest.fixture
def sample_daily_bars() -> pd.DataFrame:
    """20 days of synthetic daily OHLCV data."""
    import numpy as np

    rng = pd.date_range("2024-01-02", periods=20, freq="B")
    np.random.seed(42)
    closes = 100 + np.cumsum(np.random.randn(20))
    data = {
        "open": closes - 0.5,
        "high": closes + 1.0,
        "low": closes - 1.0,
        "close": closes,
        "volume": np.random.randint(2_000_000, 5_000_000, size=20).astype(float),
    }
    return pd.DataFrame(data, index=rng)


@pytest.fixture
def sample_intraday_bars() -> pd.DataFrame:
    """60 minutes of synthetic 1-min OHLCV intraday data."""
    import numpy as np

    rng = pd.date_range("2024-01-15 09:30", periods=60, freq="1min")
    np.random.seed(7)
    closes = 100 + np.cumsum(np.random.randn(60) * 0.1)
    data = {
        "open": closes - 0.05,
        "high": closes + 0.15,
        "low": closes - 0.15,
        "close": closes,
        "volume": np.random.randint(5_000, 50_000, size=60).astype(float),
    }
    return pd.DataFrame(data, index=rng)


@pytest.fixture
def sample_fundamentals() -> dict:
    """Fundamentals dict with short interest data."""
    return {
        "float_shares": 50_000_000,
        "short_ratio": 4.5,
        "short_percent_of_float": 0.20,
        "shares_outstanding": 60_000_000,
    }


@pytest.fixture
def sample_options() -> dict:
    """Options activity dict with elevated call volume."""
    return {
        "call_volume": 9000,
        "put_volume": 1000,
        "avg_call_volume": 2000,
        "avg_put_volume": 1000,
    }
