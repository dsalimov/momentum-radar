"""
tests/test_zone_retest.py – Tests for the zone_retest module.
"""

import numpy as np
import pandas as pd
import pytest

from momentum_radar.signals.supply_demand import SupplyDemandZone


def _make_zone(zone_type="demand", score=80.0, low=100.0, high=105.0):
    return SupplyDemandZone(
        ticker="TEST",
        timeframe="daily",
        zone_type=zone_type,
        zone_high=high,
        zone_low=low,
        strength_score=score,
    )


def _make_bars(n=20, last_close=102.0, last_open=101.0, last_vol=1_500_000.0):
    """Synthetic OHLCV bars for retest testing."""
    rng = pd.date_range("2024-01-01", periods=n, freq="5min")
    closes = np.linspace(100, last_close, n)
    opens = np.linspace(100, last_open, n)
    highs = np.maximum(closes, opens) + 0.5
    lows = np.minimum(closes, opens) - 0.5
    volumes = np.full(n, 1_000_000.0)
    volumes[-1] = last_vol
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=rng,
    )


class TestVolumeReaction:
    def test_high_volume_triggers(self):
        from momentum_radar.signals.zone_retest import _check_volume_reaction

        bars = _make_bars(last_vol=2_000_000.0)
        # avg is ~1_000_000 for first 19 bars
        assert _check_volume_reaction(bars, avg_volume=1_000_000.0)

    def test_low_volume_does_not_trigger(self):
        from momentum_radar.signals.zone_retest import _check_volume_reaction

        bars = _make_bars(last_vol=500_000.0)
        assert not _check_volume_reaction(bars, avg_volume=1_000_000.0)

    def test_empty_bars_returns_false(self):
        from momentum_radar.signals.zone_retest import _check_volume_reaction

        assert not _check_volume_reaction(pd.DataFrame(), avg_volume=1_000_000.0)


class TestRejectionWick:
    def test_demand_lower_wick_triggers(self):
        from momentum_radar.signals.zone_retest import _check_rejection_wick

        bars = _make_bars()
        # Manually set last bar with large lower wick
        bars.iloc[-1, bars.columns.get_loc("open")] = 103.0
        bars.iloc[-1, bars.columns.get_loc("close")] = 103.5
        bars.iloc[-1, bars.columns.get_loc("low")] = 100.0   # big lower wick
        bars.iloc[-1, bars.columns.get_loc("high")] = 104.0
        assert _check_rejection_wick(bars, "demand")

    def test_supply_upper_wick_triggers(self):
        from momentum_radar.signals.zone_retest import _check_rejection_wick

        bars = _make_bars()
        bars.iloc[-1, bars.columns.get_loc("open")] = 103.0
        bars.iloc[-1, bars.columns.get_loc("close")] = 102.5
        bars.iloc[-1, bars.columns.get_loc("high")] = 106.0  # big upper wick
        bars.iloc[-1, bars.columns.get_loc("low")] = 102.0
        assert _check_rejection_wick(bars, "supply")

    def test_no_wick_demand_returns_false(self):
        from momentum_radar.signals.zone_retest import _check_rejection_wick

        bars = _make_bars()
        # Marubozu (no wicks)
        bars.iloc[-1, bars.columns.get_loc("open")] = 100.0
        bars.iloc[-1, bars.columns.get_loc("close")] = 105.0
        bars.iloc[-1, bars.columns.get_loc("high")] = 105.0
        bars.iloc[-1, bars.columns.get_loc("low")] = 100.0
        assert not _check_rejection_wick(bars, "demand")


class TestEngulfing:
    def test_bullish_engulfing_demand(self):
        from momentum_radar.signals.zone_retest import _check_engulfing

        bars = _make_bars(n=5)
        # prev: bearish small body
        bars.iloc[-2, bars.columns.get_loc("open")] = 102.5
        bars.iloc[-2, bars.columns.get_loc("close")] = 101.5
        # curr: bullish large body
        bars.iloc[-1, bars.columns.get_loc("open")] = 101.0
        bars.iloc[-1, bars.columns.get_loc("close")] = 104.0
        assert _check_engulfing(bars, "demand")

    def test_bearish_engulfing_supply(self):
        from momentum_radar.signals.zone_retest import _check_engulfing

        bars = _make_bars(n=5)
        # prev: bullish small body
        bars.iloc[-2, bars.columns.get_loc("open")] = 101.5
        bars.iloc[-2, bars.columns.get_loc("close")] = 102.5
        # curr: bearish large body
        bars.iloc[-1, bars.columns.get_loc("open")] = 104.0
        bars.iloc[-1, bars.columns.get_loc("close")] = 100.0
        assert _check_engulfing(bars, "supply")

    def test_insufficient_bars_returns_false(self):
        from momentum_radar.signals.zone_retest import _check_engulfing

        assert not _check_engulfing(pd.DataFrame(), "demand")


class TestFakeBreakoutReclaim:
    def test_demand_spike_and_reclaim(self):
        from momentum_radar.signals.zone_retest import _check_fake_breakout_reclaim

        zone = _make_zone(low=100.0, high=105.0)
        bars = _make_bars(n=6)
        bars.iloc[-3, bars.columns.get_loc("low")] = 98.0   # spiked below 100
        bars.iloc[-1, bars.columns.get_loc("close")] = 102.0  # back inside
        assert _check_fake_breakout_reclaim(bars, zone)

    def test_supply_spike_and_reclaim(self):
        from momentum_radar.signals.zone_retest import _check_fake_breakout_reclaim

        zone = _make_zone(zone_type="supply", low=100.0, high=105.0)
        bars = _make_bars(n=6)
        bars.iloc[-2, bars.columns.get_loc("high")] = 108.0  # spiked above 105
        bars.iloc[-1, bars.columns.get_loc("close")] = 103.0  # back inside
        assert _check_fake_breakout_reclaim(bars, zone)

    def test_no_spike_returns_false(self):
        from momentum_radar.signals.zone_retest import _check_fake_breakout_reclaim

        # Zone low is 98.0 so bars with lows ~99.5 never breach it
        zone = _make_zone(low=98.0, high=103.0)
        bars = _make_bars(n=6)
        # All lows stay above zone_low (98.0); no spike, so returns False
        assert not _check_fake_breakout_reclaim(bars, zone)


class TestEvaluateRetest:
    def test_empty_bars_not_confirmed(self):
        from momentum_radar.signals.zone_retest import evaluate_retest

        zone = _make_zone()
        result = evaluate_retest(zone, pd.DataFrame(), atr=2.0)
        assert result["confirmed"] is False

    def test_low_score_zone_not_confirmed(self):
        from momentum_radar.signals.zone_retest import evaluate_retest

        zone = _make_zone(score=50.0)  # below RETEST_MIN_ZONE_SCORE=75
        bars = _make_bars(n=20, last_vol=3_000_000.0)
        result = evaluate_retest(zone, bars, atr=2.0)
        assert result["zone_score_ok"] is False
        assert result["confirmed"] is False

    def test_returns_expected_keys(self):
        from momentum_radar.signals.zone_retest import evaluate_retest

        zone = _make_zone(score=80.0)
        bars = _make_bars()
        result = evaluate_retest(zone, bars, atr=2.0)
        for key in ("confirmed", "confirmations", "score", "zone_score_ok"):
            assert key in result

    def test_score_is_in_range(self):
        from momentum_radar.signals.zone_retest import evaluate_retest

        zone = _make_zone(score=80.0)
        bars = _make_bars()
        result = evaluate_retest(zone, bars, atr=2.0)
        assert 0 <= result["score"] <= 100
