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


# ---------------------------------------------------------------------------
# scan_swing_trade_setups
# ---------------------------------------------------------------------------

def _make_double_bottom_daily(n: int = 60) -> pd.DataFrame:
    """Return a daily DataFrame shaped like a double bottom."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, 100.0)
    # First bottom around bar 15
    closes[10:20] = 90.0
    # Intermediate high around bar 30
    closes[25:35] = 105.0
    # Second bottom around bar 45
    closes[40:50] = 91.0
    # Recovery to breakout level
    closes[50:] = 106.0
    highs = closes + 2.0
    lows = closes - 2.0
    volumes = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=rng,
    )


def _make_flag_daily(n: int = 30) -> pd.DataFrame:
    """Return a daily DataFrame shaped like a bullish flag."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, 100.0)
    # Strong upward move in prior bars (bars 10–20)
    closes[10:20] = np.linspace(100.0, 115.0, 10)
    # Tight consolidation in final bars
    closes[20:] = np.linspace(114.0, 113.5, n - 20)
    highs = closes + 1.0
    lows = closes - 1.0
    volumes = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=rng,
    )


class TestScanSwingTradeSetups:
    """Tests for the premarket swing trade setup scanner."""

    def test_returns_list(self):
        from momentum_radar.premarket.scanner import scan_swing_trade_setups

        fetcher = MagicMock()
        fetcher.get_daily_bars.return_value = _make_double_bottom_daily()
        results = scan_swing_trade_setups(["AAPL"], fetcher)
        assert isinstance(results, list)

    def test_respects_top_n(self):
        from momentum_radar.premarket.scanner import scan_swing_trade_setups

        fetcher = MagicMock()
        fetcher.get_daily_bars.return_value = _make_double_bottom_daily()
        tickers = [f"T{i}" for i in range(20)]
        results = scan_swing_trade_setups(tickers, fetcher, top_n=5)
        assert len(results) <= 5

    def test_returns_at_most_ten_by_default(self):
        from momentum_radar.premarket.scanner import scan_swing_trade_setups

        fetcher = MagicMock()
        fetcher.get_daily_bars.return_value = _make_double_bottom_daily()
        tickers = [f"T{i}" for i in range(30)]
        results = scan_swing_trade_setups(tickers, fetcher)
        assert len(results) <= 10

    def test_result_fields_present(self):
        from momentum_radar.premarket.scanner import scan_swing_trade_setups

        fetcher = MagicMock()
        fetcher.get_daily_bars.return_value = _make_double_bottom_daily()
        results = scan_swing_trade_setups(["TSLA"], fetcher, min_confidence=0)
        if results:
            r = results[0]
            assert "ticker" in r
            assert "pattern_name" in r
            assert "pattern_confidence" in r
            assert "current_price" in r
            assert "key_level" in r
            assert "strategy_type" in r
            assert r["strategy_type"] == "SWING TRADE"
            assert r["timeframe"] == "Daily"

    def test_strategy_type_is_swing_trade(self):
        from momentum_radar.premarket.scanner import scan_swing_trade_setups

        fetcher = MagicMock()
        fetcher.get_daily_bars.return_value = _make_double_bottom_daily()
        results = scan_swing_trade_setups(["MSFT"], fetcher, min_confidence=0)
        for r in results:
            assert r["strategy_type"] == "SWING TRADE"

    def test_filters_by_confidence(self):
        from momentum_radar.premarket.scanner import scan_swing_trade_setups

        fetcher = MagicMock()
        fetcher.get_daily_bars.return_value = _make_double_bottom_daily()
        results_high = scan_swing_trade_setups(["X"], fetcher, min_confidence=95)
        results_low = scan_swing_trade_setups(["X"], fetcher, min_confidence=0)
        # High confidence filter should return fewer or equal results
        assert len(results_high) <= len(results_low)

    def test_empty_data_returns_empty(self):
        from momentum_radar.premarket.scanner import scan_swing_trade_setups

        fetcher = MagicMock()
        fetcher.get_daily_bars.return_value = None
        results = scan_swing_trade_setups(["NONE"], fetcher)
        assert results == []

    def test_insufficient_data_returns_empty(self):
        from momentum_radar.premarket.scanner import scan_swing_trade_setups

        import pandas as pd

        tiny = pd.DataFrame(
            {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [1e6]},
            index=pd.date_range("2024-01-01", periods=1, freq="B"),
        )
        fetcher = MagicMock()
        fetcher.get_daily_bars.return_value = tiny
        results = scan_swing_trade_setups(["TINY"], fetcher)
        assert results == []

    def test_sorted_by_confidence_descending(self):
        from momentum_radar.premarket.scanner import scan_swing_trade_setups

        fetcher = MagicMock()
        fetcher.get_daily_bars.return_value = _make_double_bottom_daily()
        tickers = [f"T{i}" for i in range(10)]
        results = scan_swing_trade_setups(tickers, fetcher, min_confidence=0)
        confidences = [r["pattern_confidence"] for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_fetcher_error_handled_gracefully(self):
        from momentum_radar.premarket.scanner import scan_swing_trade_setups

        fetcher = MagicMock()
        fetcher.get_daily_bars.side_effect = Exception("network error")
        results = scan_swing_trade_setups(["ERR"], fetcher)
        assert results == []
