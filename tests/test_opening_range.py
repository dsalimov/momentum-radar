"""
tests/test_opening_range.py – Tests for the opening_range module.
"""

import numpy as np
import pandas as pd
import pytest


def _make_intraday(n=60, interval_minutes=1, base_price=150.0):
    """Synthetic 1-min intraday bars."""
    rng = pd.date_range("2024-01-15 09:30:00", periods=n, freq="1min", tz="America/New_York")
    closes = np.full(n, base_price)
    highs = closes + 0.5
    lows = closes - 0.5
    volumes = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {"open": closes - 0.1, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=rng,
    )


def _make_orb_bullish(base_price=150.0, n=60):
    """Bars where price breaks out above the OR high with strong volume."""
    df = _make_intraday(n=n, base_price=base_price)
    or_high = base_price + 0.5

    # Breakout bar: last bar closes well above OR high with big volume and body
    df.iloc[-1, df.columns.get_loc("open")] = or_high + 0.1
    df.iloc[-1, df.columns.get_loc("close")] = or_high + 1.5
    df.iloc[-1, df.columns.get_loc("high")] = or_high + 1.8
    df.iloc[-1, df.columns.get_loc("low")] = or_high - 0.1
    df.iloc[-1, df.columns.get_loc("volume")] = 3_000_000.0  # 3× avg

    # Follow-through bar (bar before last)
    df.iloc[-2, df.columns.get_loc("open")] = or_high - 0.1
    df.iloc[-2, df.columns.get_loc("close")] = or_high + 0.5
    df.iloc[-2, df.columns.get_loc("volume")] = 2_000_000.0

    return df


def _make_orb_bearish(base_price=150.0, n=60):
    """Bars where price breaks down below the OR low with strong volume."""
    df = _make_intraday(n=n, base_price=base_price)
    or_low = base_price - 0.5

    df.iloc[-1, df.columns.get_loc("open")] = or_low - 0.1
    df.iloc[-1, df.columns.get_loc("close")] = or_low - 1.5
    df.iloc[-1, df.columns.get_loc("high")] = or_low + 0.1
    df.iloc[-1, df.columns.get_loc("low")] = or_low - 1.8
    df.iloc[-1, df.columns.get_loc("volume")] = 3_000_000.0

    df.iloc[-2, df.columns.get_loc("open")] = or_low + 0.1
    df.iloc[-2, df.columns.get_loc("close")] = or_low - 0.5
    df.iloc[-2, df.columns.get_loc("volume")] = 2_000_000.0

    return df


class TestComputeOpeningRange:
    def test_returns_correct_keys(self):
        from momentum_radar.services.opening_range import compute_opening_range

        bars = _make_intraday()
        result = compute_opening_range(bars, minutes=15)
        assert result is not None
        for key in ("high", "low", "range"):
            assert key in result

    def test_high_gt_low(self):
        from momentum_radar.services.opening_range import compute_opening_range

        bars = _make_intraday()
        result = compute_opening_range(bars, minutes=15)
        assert result["high"] >= result["low"]

    def test_empty_bars_returns_none(self):
        from momentum_radar.services.opening_range import compute_opening_range

        assert compute_opening_range(pd.DataFrame(), minutes=15) is None

    def test_range_equals_high_minus_low(self):
        from momentum_radar.services.opening_range import compute_opening_range

        bars = _make_intraday()
        result = compute_opening_range(bars, minutes=15)
        assert abs(result["range"] - (result["high"] - result["low"])) < 1e-6


class TestDetectOrb:
    def test_bullish_orb_triggers(self):
        from momentum_radar.services.opening_range import compute_opening_range, detect_orb

        bars = _make_orb_bullish()
        orb = compute_opening_range(bars, minutes=15)
        result = detect_orb(bars, orb)
        assert result["triggered"] is True
        assert result["direction"] == "bullish"

    def test_bearish_orb_triggers(self):
        from momentum_radar.services.opening_range import compute_opening_range, detect_orb

        bars = _make_orb_bearish()
        orb = compute_opening_range(bars, minutes=15)
        result = detect_orb(bars, orb)
        assert result["triggered"] is True
        assert result["direction"] == "bearish"

    def test_no_breakout_not_triggered(self):
        from momentum_radar.services.opening_range import compute_opening_range, detect_orb

        bars = _make_intraday()  # price stays flat
        orb = compute_opening_range(bars, minutes=15)
        result = detect_orb(bars, orb)
        assert result["triggered"] is False

    def test_none_orb_not_triggered(self):
        from momentum_radar.services.opening_range import detect_orb

        bars = _make_intraday()
        result = detect_orb(bars, None)
        assert result["triggered"] is False

    def test_score_in_range(self):
        from momentum_radar.services.opening_range import compute_opening_range, detect_orb

        bars = _make_orb_bullish()
        orb = compute_opening_range(bars, minutes=15)
        result = detect_orb(bars, orb)
        assert 0 <= result["score"] <= 100

    def test_returns_expected_keys(self):
        from momentum_radar.services.opening_range import compute_opening_range, detect_orb

        bars = _make_intraday()
        orb = compute_opening_range(bars, minutes=15)
        result = detect_orb(bars, orb)
        for key in ("triggered", "direction", "score", "breakout_level", "details"):
            assert key in result
