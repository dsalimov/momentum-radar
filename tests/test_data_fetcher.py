"""
test_data_fetcher.py – Unit tests for data_fetcher helpers and YFinanceDataFetcher.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from momentum_radar.data.data_fetcher import (
    YFinanceDataFetcher,
    _normalise_yf_columns,
)


# ---------------------------------------------------------------------------
# _normalise_yf_columns
# ---------------------------------------------------------------------------

def test_normalise_yf_columns_plain_strings():
    """Plain string columns are lowercased in-place."""
    df = pd.DataFrame([[1, 2, 3, 4, 5]], columns=["Open", "High", "Low", "Close", "Volume"])
    result = _normalise_yf_columns(df)
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]


def test_normalise_yf_columns_already_lowercase():
    """Already-lowercase string columns are unchanged."""
    df = pd.DataFrame([[1]], columns=["close"])
    result = _normalise_yf_columns(df)
    assert list(result.columns) == ["close"]


def test_normalise_yf_columns_multiindex_tuples():
    """Tuple columns (MultiIndex-style from yfinance) are flattened to first element, lowercased."""
    df = pd.DataFrame(
        [[1, 2, 3, 4, 5]],
        columns=[("Close", "AAPL"), ("High", "AAPL"), ("Low", "AAPL"), ("Open", "AAPL"), ("Volume", "AAPL")],
    )
    result = _normalise_yf_columns(df)
    assert list(result.columns) == ["close", "high", "low", "open", "volume"]


def test_normalise_yf_columns_returns_df():
    """_normalise_yf_columns returns the same DataFrame (in-place mutation)."""
    df = pd.DataFrame([[1]], columns=["Close"])
    result = _normalise_yf_columns(df)
    assert result is df


# ---------------------------------------------------------------------------
# YFinanceDataFetcher.get_daily_bars
# ---------------------------------------------------------------------------

def _make_multiindex_df():
    """Return a DataFrame with MultiIndex columns simulating yfinance output."""
    return pd.DataFrame(
        [[150.0, 152.0, 149.0, 151.0, 1_000_000]],
        columns=[("Close", "AAPL"), ("High", "AAPL"), ("Low", "AAPL"), ("Open", "AAPL"), ("Volume", "AAPL")],
    )


def test_get_daily_bars_normalises_multiindex_columns():
    """get_daily_bars must not crash and must return lowercase columns for MultiIndex DataFrames."""
    fetcher = YFinanceDataFetcher()
    with patch("yfinance.download", return_value=_make_multiindex_df()):
        df = fetcher.get_daily_bars("AAPL")

    assert df is not None
    assert "close" in df.columns
    assert "open" in df.columns


def test_get_intraday_bars_normalises_multiindex_columns():
    """get_intraday_bars must not crash and must return lowercase columns for MultiIndex DataFrames."""
    fetcher = YFinanceDataFetcher()
    with patch("yfinance.download", return_value=_make_multiindex_df()):
        df = fetcher.get_intraday_bars("AAPL")

    assert df is not None
    assert "close" in df.columns
    assert "open" in df.columns


def test_get_daily_bars_returns_none_on_empty():
    """get_daily_bars returns None when yfinance returns an empty DataFrame."""
    fetcher = YFinanceDataFetcher()
    with patch("yfinance.download", return_value=pd.DataFrame()):
        df = fetcher.get_daily_bars("AAPL")

    assert df is None


def test_get_intraday_bars_returns_none_on_empty():
    """get_intraday_bars returns None when yfinance returns an empty DataFrame."""
    fetcher = YFinanceDataFetcher()
    with patch("yfinance.download", return_value=pd.DataFrame()):
        df = fetcher.get_intraday_bars("AAPL")

    assert df is None
