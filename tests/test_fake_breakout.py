"""
tests/test_fake_breakout.py – Tests for the fake_breakout module.
"""

import numpy as np
import pandas as pd
import pytest


def _make_bars(n=30, last_close=150.0, last_open=149.0,
               last_high=151.0, last_low=148.5, last_vol=800_000.0):
    """Synthetic OHLCV bars."""
    rng = pd.date_range("2024-01-01", periods=n, freq="5min")
    closes = np.linspace(148, last_close, n)
    opens = np.linspace(148, last_open, n)
    highs = np.maximum(closes, opens) + 0.5
    lows = np.minimum(closes, opens) - 0.5
    volumes = np.full(n, 1_000_000.0)
    # Override last bar
    closes[-1] = last_close
    opens[-1] = last_open
    highs[-1] = last_high
    lows[-1] = last_low
    volumes[-1] = last_vol
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=rng,
    )


class TestIsFakeBreakout:
    def test_returns_expected_keys(self):
        from momentum_radar.services.fake_breakout import is_fake_breakout

        bars = _make_bars()
        result = is_fake_breakout(bars, level=150.0, direction="above")
        for key in ("is_fake", "reasons", "score"):
            assert key in result

    def test_insufficient_bars_not_fake(self):
        from momentum_radar.services.fake_breakout import is_fake_breakout

        bars = _make_bars(n=2)
        result = is_fake_breakout(bars, level=150.0, direction="above")
        assert result["is_fake"] is False

    def test_low_volume_breakout_is_fake(self):
        from momentum_radar.services.fake_breakout import is_fake_breakout

        # Volume on last bar is only 500k vs avg ~1M → low volume
        bars = _make_bars(last_close=151.0, last_open=149.0,
                          last_high=152.0, last_low=148.8, last_vol=500_000.0)
        result = is_fake_breakout(bars, level=150.5, direction="above")
        assert "low_volume_breakout" in result["reasons"]

    def test_wick_dominant_candle_flagged(self):
        from momentum_radar.services.fake_breakout import is_fake_breakout

        # Wide range candle with tiny body (doji-like)
        bars = _make_bars(
            last_close=150.1,   # tiny body
            last_open=150.0,
            last_high=153.0,   # big upper wick
            last_low=147.0,    # big lower wick
            last_vol=2_000_000.0,
        )
        result = is_fake_breakout(bars, level=149.5, direction="above")
        assert "wick_dominant_candle" in result["reasons"]

    def test_no_follow_through_flagged(self):
        from momentum_radar.services.fake_breakout import is_fake_breakout

        # Price did NOT break above the level (close < level)
        bars = _make_bars(last_close=149.0, last_open=148.0,
                          last_high=150.5, last_low=147.5, last_vol=1_200_000.0)
        result = is_fake_breakout(bars, level=150.0, direction="above")
        assert "no_follow_through" in result["reasons"]

    def test_is_fake_requires_two_reasons(self):
        from momentum_radar.services.fake_breakout import is_fake_breakout

        # Craft a scenario with exactly 1 reason
        bars = _make_bars(
            last_close=151.0,  # did break above 150
            last_open=150.5,
            last_high=151.5,
            last_low=150.2,
            last_vol=2_000_000.0,  # strong volume
        )
        result = is_fake_breakout(bars, level=150.0, direction="above")
        # may or may not be fake; just verify the flag is consistent with reason count
        assert result["is_fake"] == (len(result["reasons"]) >= 2)

    def test_score_proportional_to_reasons(self):
        from momentum_radar.services.fake_breakout import is_fake_breakout

        bars = _make_bars(last_close=149.0, last_vol=400_000.0,
                          last_open=148.0, last_high=153.0, last_low=146.0)
        result = is_fake_breakout(bars, level=150.0, direction="above")
        assert result["score"] == int(min(len(result["reasons"]) / 5.0 * 100.0, 100.0))
