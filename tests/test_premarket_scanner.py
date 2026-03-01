"""
test_premarket_scanner.py – Unit tests for the pre-market scanner module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock


def _make_daily(n: int = 35, vol_multiplier: float = 1.0) -> pd.DataFrame:
    """Return a minimal daily OHLCV DataFrame."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    np.random.seed(0)
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    volumes = np.random.randint(1_000_000, 3_000_000, size=n).astype(float)
    volumes[-1] *= vol_multiplier  # today's volume
    return pd.DataFrame(
        {
            "open": closes - 0.5,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": volumes,
        },
        index=rng,
    )


def _make_fetcher(daily: pd.DataFrame, options: dict = None):
    fetcher = MagicMock()
    fetcher.get_daily_bars.return_value = daily
    fetcher.get_fundamentals.return_value = {
        "float_shares": 30_000_000,
        "short_percent_of_float": 0.25,
        "short_ratio": 4.5,
        "shares_outstanding": 40_000_000,
    }
    fetcher.get_options_volume.return_value = options or {
        "call_volume": 6_000,
        "put_volume": 2_000,
        "avg_call_volume": 2_000,
        "avg_put_volume": 1_000,
    }
    return fetcher


# ---------------------------------------------------------------------------
# scan_unusual_volume
# ---------------------------------------------------------------------------

class TestScanUnusualVolume:
    def test_returns_high_rvol_tickers(self):
        from momentum_radar.premarket.scanner import scan_unusual_volume

        daily = _make_daily(vol_multiplier=5.0)  # RVOL ≈ 5
        fetcher = _make_fetcher(daily)
        results = scan_unusual_volume(["AAPL"], fetcher, min_rvol=2.0)

        assert len(results) == 1
        assert results[0]["ticker"] == "AAPL"
        assert results[0]["rvol"] >= 2.0

    def test_filters_low_rvol_tickers(self):
        from momentum_radar.premarket.scanner import scan_unusual_volume

        # today_vol = 0.1× avg → rvol=0.1 (< 2.0) AND vol_ratio=0.1 (< 0.20) → filtered
        daily = _make_daily(vol_multiplier=0.1)
        fetcher = _make_fetcher(daily)
        results = scan_unusual_volume(["XYZ"], fetcher, min_rvol=2.0)

        assert len(results) == 0

    def test_returns_correct_fields(self):
        from momentum_radar.premarket.scanner import scan_unusual_volume

        daily = _make_daily(vol_multiplier=4.0)
        fetcher = _make_fetcher(daily)
        results = scan_unusual_volume(["TSLA"], fetcher, min_rvol=2.0)

        assert len(results) == 1
        r = results[0]
        for key in ("ticker", "rvol", "pct_change", "gap_pct", "last_close", "avg_volume", "today_volume"):
            assert key in r

    def test_sorted_by_rvol_descending(self):
        from momentum_radar.premarket.scanner import scan_unusual_volume

        fetcher = MagicMock()
        daily_high = _make_daily(vol_multiplier=8.0)
        daily_low = _make_daily(vol_multiplier=3.0)
        fetcher.get_daily_bars.side_effect = [daily_high, daily_low]
        fetcher.get_fundamentals.return_value = None

        results = scan_unusual_volume(["A", "B"], fetcher, min_rvol=2.0)
        if len(results) >= 2:
            assert results[0]["rvol"] >= results[1]["rvol"]

    def test_handles_empty_daily(self):
        from momentum_radar.premarket.scanner import scan_unusual_volume

        fetcher = MagicMock()
        fetcher.get_daily_bars.return_value = None
        fetcher.get_fundamentals.return_value = None
        results = scan_unusual_volume(["NONE"], fetcher)
        assert results == []

    def test_respects_top_n(self):
        from momentum_radar.premarket.scanner import scan_unusual_volume

        tickers = [f"T{i}" for i in range(20)]
        fetcher = MagicMock()
        fetcher.get_daily_bars.return_value = _make_daily(vol_multiplier=5.0)
        fetcher.get_fundamentals.return_value = None

        results = scan_unusual_volume(tickers, fetcher, top_n=5, min_rvol=0.1)
        assert len(results) <= 5


# ---------------------------------------------------------------------------
# scan_most_active
# ---------------------------------------------------------------------------

class TestScanMostActive:
    def test_returns_four_categories(self):
        from momentum_radar.premarket.scanner import scan_most_active

        daily = _make_daily()
        fetcher = _make_fetcher(daily)
        result = scan_most_active(["AAPL", "MSFT"], fetcher)

        for key in ("highest_volume", "highest_dollar_volume", "top_gainers", "top_losers"):
            assert key in result

    def test_top_gainers_sorted_descending(self):
        from momentum_radar.premarket.scanner import scan_most_active

        fetcher = MagicMock()
        fetcher.get_daily_bars.side_effect = [
            _make_daily(n=5),
            _make_daily(n=5),
        ]
        result = scan_most_active(["A", "B"], fetcher)
        gainers = result["top_gainers"]
        if len(gainers) >= 2:
            assert gainers[0]["pct_change"] >= gainers[1]["pct_change"]

    def test_handles_missing_data(self):
        from momentum_radar.premarket.scanner import scan_most_active

        fetcher = MagicMock()
        fetcher.get_daily_bars.return_value = None
        result = scan_most_active(["AAPL"], fetcher)
        for cat in result.values():
            assert isinstance(cat, list)

    def test_result_dicts_contain_required_fields(self):
        from momentum_radar.premarket.scanner import scan_most_active

        daily = _make_daily()
        fetcher = _make_fetcher(daily)
        result = scan_most_active(["AAPL"], fetcher)

        for cat_list in result.values():
            for item in cat_list:
                for key in ("ticker", "last_close", "pct_change", "today_volume", "dollar_volume"):
                    assert key in item


# ---------------------------------------------------------------------------
# scan_options_spikes
# ---------------------------------------------------------------------------

class TestScanOptionsSpikes:
    def test_detects_call_spike(self):
        from momentum_radar.premarket.scanner import scan_options_spikes

        fetcher = MagicMock()
        fetcher.get_options_volume.return_value = {
            "call_volume": 10_000,
            "put_volume": 1_000,
            "avg_call_volume": 2_000,
            "avg_put_volume": 500,
        }
        results = scan_options_spikes(["AAPL"], fetcher, min_multiplier=2.0)
        assert len(results) == 1
        assert results[0]["bias"] == "BULLISH"

    def test_detects_put_spike(self):
        from momentum_radar.premarket.scanner import scan_options_spikes

        fetcher = MagicMock()
        fetcher.get_options_volume.return_value = {
            "call_volume": 500,
            "put_volume": 8_000,
            "avg_call_volume": 500,
            "avg_put_volume": 1_000,
        }
        results = scan_options_spikes(["SPY"], fetcher, min_multiplier=2.0)
        assert len(results) == 1
        assert results[0]["bias"] == "BEARISH"

    def test_filters_normal_volume(self):
        from momentum_radar.premarket.scanner import scan_options_spikes

        fetcher = MagicMock()
        fetcher.get_options_volume.return_value = {
            "call_volume": 1_000,
            "put_volume": 1_000,
            "avg_call_volume": 1_000,
            "avg_put_volume": 1_000,
        }
        results = scan_options_spikes(["FLAT"], fetcher, min_multiplier=2.0)
        assert len(results) == 0

    def test_handles_none_options(self):
        from momentum_radar.premarket.scanner import scan_options_spikes

        fetcher = MagicMock()
        fetcher.get_options_volume.return_value = None
        results = scan_options_spikes(["X"], fetcher)
        assert results == []
