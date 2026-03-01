"""
test_universe_builder.py – Unit tests for fetch_index_constituents().
"""

from unittest.mock import patch

import pandas as pd
import pytest

from momentum_radar.data.universe_builder import (
    _SEED_UNIVERSE,
    fetch_index_constituents,
)


def _sp500_df() -> pd.DataFrame:
    """Minimal fake S&P 500 Wikipedia table."""
    return pd.DataFrame({"Symbol": ["AAPL", "MSFT", "BRK.B", "GOOG"]})


def _ndx_df() -> pd.DataFrame:
    """Minimal fake NASDAQ-100 Wikipedia table."""
    return pd.DataFrame({"Ticker": ["AAPL", "MSFT", "AMZN", "TSLA"]})


def test_fetch_index_constituents_success():
    """Returns merged, deduplicated tickers when both fetches succeed."""
    with patch(
        "momentum_radar.data.universe_builder.pd.read_html",
        side_effect=[[_sp500_df()], [_ndx_df()]],
    ):
        result = fetch_index_constituents()

    # BRK.B → BRK-B conversion
    assert "BRK-B" in result
    assert "BRK.B" not in result
    # All tickers present
    for t in ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]:
        assert t in result
    # No duplicates
    assert len(result) == len(set(result))


def test_fetch_index_constituents_sp500_failure_returns_seed():
    """Falls back to _SEED_UNIVERSE when S&P 500 fetch fails."""
    with patch(
        "momentum_radar.data.universe_builder.pd.read_html",
        side_effect=Exception("network error"),
    ):
        result = fetch_index_constituents()

    assert result is _SEED_UNIVERSE


def test_fetch_index_constituents_ndx_failure_uses_sp500_only():
    """Uses S&P 500 tickers only when NASDAQ-100 fetch fails."""

    def read_html_side_effect(url, **kwargs):
        if "S%26P" in url:
            return [_sp500_df()]
        raise Exception("ndx error")

    with patch(
        "momentum_radar.data.universe_builder.pd.read_html",
        side_effect=read_html_side_effect,
    ):
        result = fetch_index_constituents()

    assert "AAPL" in result
    assert "BRK-B" in result
    # AMZN only in NASDAQ-100 mock – should be absent
    assert "AMZN" not in result


def test_fetch_index_constituents_deduplicates():
    """Tickers appearing in both indices are returned only once."""
    with patch(
        "momentum_radar.data.universe_builder.pd.read_html",
        side_effect=[[_sp500_df()], [_ndx_df()]],
    ):
        result = fetch_index_constituents()

    assert result.count("AAPL") == 1
    assert result.count("MSFT") == 1
