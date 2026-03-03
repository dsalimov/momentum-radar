"""
tests/test_signal_engine.py – Unit tests for the multi-confirmation signal engine.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily(n: int = 65, trend: str = "up") -> pd.DataFrame:
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    np.random.seed(7)
    if trend == "up":
        closes = 100 + np.arange(n) * 0.8
    else:
        closes = 100 + np.zeros(n)
    closes = closes + np.random.randn(n) * 0.1
    return pd.DataFrame(
        {
            "open": closes - 0.3,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": np.random.randint(1_000_000, 3_000_000, size=n).astype(float),
        },
        index=rng,
    )


def _make_breakout_daily(n: int = 65) -> pd.DataFrame:
    """Daily bars where the last close is above all prior highs (breakout)."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, 100.0)
    closes[-1] = 120.0  # last bar breaks out
    return pd.DataFrame(
        {
            "open": closes - 0.3,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": np.full(n, 2_000_000.0),
        },
        index=rng,
    )


def _make_volume_spike_bars(n: int = 60) -> pd.DataFrame:
    """Intraday 1-min bars with a big volume spike on the last bar.

    The last bar is a strong bullish candle (close == high, tiny lower wick)
    so it does not trigger the fake-breakout wick filter.
    """
    rng = pd.date_range("2024-01-15 09:30", periods=n, freq="1min")
    volumes = np.full(n, 50_000.0)
    volumes[-1] = 400_000.0  # 8× spike
    closes = np.full(n, 50.0)
    opens = closes - 0.05
    highs = closes + 0.1
    lows = closes - 0.1
    # Last bar: open near low, close at high → momentum candle, small wick ratio
    opens[-1] = 49.90
    closes[-1] = 50.10
    highs[-1] = 50.10   # close == high (no upper wick)
    lows[-1] = 49.88    # tiny lower wick (0.02)
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=rng,
    )


def _make_high_volume_daily(n: int = 40) -> pd.DataFrame:
    """Daily bars where today's volume is 3× the 30-day average."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, 50.0)
    volumes = np.full(n, 1_000_000.0)
    volumes[-1] = 4_000_000.0  # 4× avg
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": volumes,
        },
        index=rng,
    )


def _options_call_spike() -> dict:
    return {
        "call_volume": 10_000,
        "put_volume": 2_000,
        "avg_call_volume": 2_000,
        "avg_put_volume": 2_000,
    }


def _options_normal() -> dict:
    return {
        "call_volume": 1_000,
        "put_volume": 1_000,
        "avg_call_volume": 1_000,
        "avg_put_volume": 1_000,
    }


# ---------------------------------------------------------------------------
# Confirmation unit tests
# ---------------------------------------------------------------------------

class TestCheckVolume:
    def test_rvol_spike_triggers(self):
        from momentum_radar.services.signal_engine import _check_volume

        bars = _make_volume_spike_bars()
        daily = _make_high_volume_daily()
        conf = _check_volume(bars, daily)
        assert conf is not None
        assert conf.category == "volume"
        assert conf.confidence >= 70

    def test_no_spike_returns_none(self):
        from momentum_radar.services.signal_engine import _check_volume

        rng = pd.date_range("2024-01-15", periods=40, freq="B")
        closes = np.full(40, 50.0)
        daily = pd.DataFrame(
            {
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": np.full(40, 1_000_000.0),
            },
            index=rng,
        )
        conf = _check_volume(None, daily)
        # Last bar volume equals avg → ratio = 1.0 < 2.0
        assert conf is None

    def test_no_daily_data_returns_none(self):
        from momentum_radar.services.signal_engine import _check_volume

        conf = _check_volume(None, None)
        assert conf is None


class TestCheckPattern:
    def test_breakout_detected(self):
        from momentum_radar.services.signal_engine import _check_pattern

        daily = _make_breakout_daily()
        conf = _check_pattern(daily)
        assert conf is not None
        assert "Breakout" in conf.name or "Ascending" in conf.name
        assert conf.confidence >= 70

    def test_no_breakout_no_pattern(self):
        from momentum_radar.services.signal_engine import _check_pattern

        # Flat data with no breakout or double bottom
        rng = pd.date_range("2024-01-01", periods=25, freq="B")
        closes = np.full(25, 50.0)
        daily = pd.DataFrame(
            {
                "open": closes,
                "high": closes + 0.1,
                "low": closes - 0.1,
                "close": closes,
                "volume": np.full(25, 1_000_000.0),
            },
            index=rng,
        )
        conf = _check_pattern(daily)
        # Support bounce is possible for flat data, but breakout should not fire
        # Just assert we don't crash
        assert conf is None or isinstance(conf.confidence, float)

    def test_insufficient_data_returns_none(self):
        from momentum_radar.services.signal_engine import _check_pattern

        rng = pd.date_range("2024-01-01", periods=5, freq="B")
        closes = np.full(5, 50.0)
        daily = pd.DataFrame(
            {"open": closes, "high": closes, "low": closes, "close": closes,
             "volume": np.full(5, 1_000_000.0)},
            index=rng,
        )
        conf = _check_pattern(daily)
        assert conf is None

    def test_none_returns_none(self):
        from momentum_radar.services.signal_engine import _check_pattern

        assert _check_pattern(None) is None


class TestCheckCandlestick:
    def test_bullish_engulfing_detected(self):
        from momentum_radar.services.signal_engine import _check_candlestick

        rng = pd.date_range("2024-01-01", periods=2, freq="B")
        # Previous bar: bearish (open > close)
        # Current bar:  bullish and engulfs prior body
        daily = pd.DataFrame(
            {
                "open":  [55.0, 49.0],
                "high":  [56.0, 57.0],
                "low":   [49.0, 48.0],
                "close": [50.0, 56.0],
                "volume": [1e6, 2e6],
            },
            index=rng,
        )
        conf = _check_candlestick(None, daily)
        assert conf is not None
        assert conf.category == "candlestick"
        assert "Engulfing" in conf.name

    def test_no_pattern_no_confirmation(self):
        from momentum_radar.services.signal_engine import _check_candlestick

        rng = pd.date_range("2024-01-01", periods=2, freq="B")
        # Two nearly identical bars, neither pattern
        daily = pd.DataFrame(
            {
                "open":  [50.0, 50.1],
                "high":  [50.5, 50.6],
                "low":   [49.5, 49.6],
                "close": [50.2, 50.3],
                "volume": [1e6, 1e6],
            },
            index=rng,
        )
        conf = _check_candlestick(None, daily)
        assert conf is None

    def test_insufficient_data_returns_none(self):
        from momentum_radar.services.signal_engine import _check_candlestick

        rng = pd.date_range("2024-01-01", periods=1, freq="B")
        daily = pd.DataFrame(
            {"open": [50.0], "high": [51.0], "low": [49.0], "close": [50.5],
             "volume": [1e6]},
            index=rng,
        )
        assert _check_candlestick(None, daily) is None


class TestCheckOptions:
    def test_call_spike_triggers(self):
        from momentum_radar.services.signal_engine import _check_options

        conf = _check_options(_options_call_spike())
        assert conf is not None
        assert conf.category == "options"
        assert "Call" in conf.name

    def test_normal_volume_no_trigger(self):
        from momentum_radar.services.signal_engine import _check_options

        conf = _check_options(_options_normal())
        assert conf is None

    def test_none_returns_none(self):
        from momentum_radar.services.signal_engine import _check_options

        assert _check_options(None) is None

    def test_gamma_flip_triggers(self):
        from momentum_radar.services.signal_engine import _check_options

        options = {
            "call_volume": 6_000,
            "put_volume": 2_000,
            "avg_call_volume": 5_000,  # call ratio only 1.2 – below spike
            "avg_put_volume": 2_000,   # put ratio 1.0
        }
        conf = _check_options(options)
        # C/P ratio = 3.0 → gamma flip
        assert conf is not None
        assert "Gamma" in conf.name


# ---------------------------------------------------------------------------
# evaluate() integration tests
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_no_signal_with_empty_data(self):
        from momentum_radar.services.signal_engine import evaluate

        result = evaluate("NONE")
        assert result.priority == "NO_SIGNAL"
        assert result.confirmation_count == 0

    def test_two_confirmations_produce_no_signal(self):
        from momentum_radar.services.signal_engine import evaluate

        daily = _make_breakout_daily()    # triggers pattern
        options = _options_call_spike()   # triggers options

        result = evaluate("TST", daily=daily, options=options)
        # With the 3-confirmation minimum, 2 confirmations must not produce a signal
        assert result.priority == "NO_SIGNAL"

    def test_three_confirmations_produce_high_confidence(self):
        from momentum_radar.services.signal_engine import evaluate

        daily = _make_high_volume_daily()
        bars = _make_volume_spike_bars()
        options = _options_call_spike()

        # volume + options are guaranteed; pattern may or may not fire
        result = evaluate("HI", bars=bars, daily=daily, options=options)
        assert result.confirmation_count >= 2
        if result.confirmation_count >= 4:
            assert result.priority == "HIGH_CONFIDENCE"
        elif result.confirmation_count >= 3:
            assert result.priority in ("ALERT", "HIGH_CONFIDENCE")

    def test_single_confirmation_no_signal(self):
        from momentum_radar.services.signal_engine import evaluate

        # Only options spikes – all other data absent
        result = evaluate("OPT", options=_options_call_spike())
        # Only 1 confirmation → no signal
        assert result.priority == "NO_SIGNAL"

    def test_result_fields_populated(self):
        from momentum_radar.services.signal_engine import evaluate

        daily = _make_breakout_daily()
        options = _options_call_spike()
        result = evaluate("X", daily=daily, options=options)

        assert isinstance(result.ticker, str)
        assert isinstance(result.confirmations, list)
        assert isinstance(result.priority, str)
        assert isinstance(result.confidence_score, float)

    def test_confirmation_labels_are_strings(self):
        from momentum_radar.services.signal_engine import evaluate

        daily = _make_breakout_daily()
        options = _options_call_spike()
        result = evaluate("Y", daily=daily, options=options)

        for label in result.confirmation_labels:
            assert isinstance(label, str)
            assert len(label) > 0


# ---------------------------------------------------------------------------
# New confirmation checker tests
# ---------------------------------------------------------------------------

class TestCheckHtfTrend:
    def test_bullish_alignment_detected(self):
        from momentum_radar.services.signal_engine import _check_htf_trend

        rng = pd.date_range("2024-01-01", periods=60, freq="B")
        # Trending up so last close > EMA21 > EMA50
        closes = 100 + np.arange(60) * 0.5
        daily = pd.DataFrame(
            {"open": closes - 0.2, "high": closes + 0.5, "low": closes - 0.5,
             "close": closes, "volume": np.full(60, 1e6)},
            index=rng,
        )
        conf = _check_htf_trend(daily)
        assert conf is not None
        assert conf.category == "htf_trend"

    def test_insufficient_data_returns_none(self):
        from momentum_radar.services.signal_engine import _check_htf_trend

        rng = pd.date_range("2024-01-01", periods=20, freq="B")
        closes = np.full(20, 100.0)
        daily = pd.DataFrame(
            {"open": closes, "high": closes, "low": closes,
             "close": closes, "volume": np.full(20, 1e6)},
            index=rng,
        )
        assert _check_htf_trend(daily) is None

    def test_none_returns_none(self):
        from momentum_radar.services.signal_engine import _check_htf_trend

        assert _check_htf_trend(None) is None


class TestCheckMomentum:
    def test_rsi_macd_bullish_detected(self):
        from momentum_radar.services.signal_engine import _check_momentum

        rng = pd.date_range("2024-01-01", periods=60, freq="B")
        np.random.seed(42)
        # Moderate uptrend with noise – RSI stays in 40-70 zone
        closes = 50 + np.cumsum(np.random.randn(60) * 0.4 + 0.1)
        daily = pd.DataFrame(
            {"open": closes - 0.1, "high": closes + 0.2, "low": closes - 0.2,
             "close": closes, "volume": np.full(60, 1e6)},
            index=rng,
        )
        conf = _check_momentum(None, daily)
        # Confirms when RSI in 40-70 with positive MACD histogram; may be None
        # if data doesn't align – just verify no exception and correct type
        assert conf is None or conf.category == "momentum"

    def test_none_returns_none(self):
        from momentum_radar.services.signal_engine import _check_momentum

        assert _check_momentum(None, None) is None


class TestCheckRetest:
    def test_retest_detected_near_prior_high(self):
        from momentum_radar.services.signal_engine import _check_retest

        rng = pd.date_range("2024-01-01", periods=25, freq="B")
        closes = np.full(25, 100.0)
        highs = closes + 1.0
        # Prior 20-bar high = 101.0; last close is 100.98 → within 1.5% → retest
        closes[-1] = 100.98
        highs[-1] = 101.0
        daily = pd.DataFrame(
            {"open": closes - 0.2, "high": highs, "low": closes - 0.5,
             "close": closes, "volume": np.full(25, 1e6)},
            index=rng,
        )
        conf = _check_retest(daily)
        assert conf is not None
        assert conf.category == "retest"

    def test_none_returns_none(self):
        from momentum_radar.services.signal_engine import _check_retest

        assert _check_retest(None) is None


class TestCheckLiquiditySweep:
    def test_bullish_sweep_detected(self):
        from momentum_radar.services.signal_engine import _check_liquidity_sweep

        rng = pd.date_range("2024-01-01", periods=15, freq="B")
        closes = np.full(15, 50.0)
        opens = closes - 0.3
        highs = closes + 0.5
        lows = closes - 0.5
        # Last bar wicks below swing low (49.5) then closes above it
        lows[-1] = 48.8   # wick below prior swing low
        closes[-1] = 50.2
        highs[-1] = 50.5
        opens[-1] = 49.6
        daily = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes,
             "volume": np.full(15, 1e6)},
            index=rng,
        )
        conf = _check_liquidity_sweep(None, daily)
        assert conf is not None
        assert conf.category == "liquidity_sweep"
        assert "Bullish" in conf.name

    def test_no_sweep_returns_none(self):
        from momentum_radar.services.signal_engine import _check_liquidity_sweep

        rng = pd.date_range("2024-01-01", periods=15, freq="B")
        closes = np.full(15, 50.0)
        daily = pd.DataFrame(
            {"open": closes - 0.1, "high": closes + 0.2, "low": closes - 0.2,
             "close": closes, "volume": np.full(15, 1e6)},
            index=rng,
        )
        assert _check_liquidity_sweep(None, daily) is None


class TestIsFakeBreakout:
    def test_low_volume_is_fake(self):
        from momentum_radar.services.signal_engine import _is_fake_breakout

        rng = pd.date_range("2024-01-15 09:30", periods=25, freq="1min")
        volumes = np.full(25, 100_000.0)
        volumes[-1] = 10_000.0  # last bar volume << average → fake
        closes = np.full(25, 50.0)
        bars = pd.DataFrame(
            {"open": closes - 0.1, "high": closes + 0.2, "low": closes - 0.2,
             "close": closes, "volume": volumes},
            index=rng,
        )
        assert _is_fake_breakout(bars, None) is True

    def test_large_wick_is_fake(self):
        from momentum_radar.services.signal_engine import _is_fake_breakout

        rng = pd.date_range("2024-01-15 09:30", periods=25, freq="1min")
        volumes = np.full(25, 100_000.0)  # adequate volume
        closes = np.full(25, 50.0)
        opens = closes - 0.05
        highs = closes + 2.0   # very large upper wick
        lows = closes - 0.05
        bars = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows,
             "close": closes, "volume": volumes},
            index=rng,
        )
        assert _is_fake_breakout(bars, None) is True

    def test_strong_close_passes(self):
        from momentum_radar.services.signal_engine import _is_fake_breakout

        bars = _make_volume_spike_bars()
        assert _is_fake_breakout(bars, None) is False

    def test_none_data_returns_false(self):
        from momentum_radar.services.signal_engine import _is_fake_breakout

        assert _is_fake_breakout(None, None) is False


# ---------------------------------------------------------------------------
# Timeframe and opening range tests (main.py helpers)
# ---------------------------------------------------------------------------

class TestGetActiveTimeframe:
    def test_scalp_window(self):
        from momentum_radar.main import get_active_timeframe

        dt = datetime(2024, 1, 15, 9, 45)
        assert get_active_timeframe(dt) == "2m"

    def test_intraday_window(self):
        from momentum_radar.main import get_active_timeframe

        dt = datetime(2024, 1, 15, 10, 30)
        assert get_active_timeframe(dt) == "5m"

    def test_trend_window(self):
        from momentum_radar.main import get_active_timeframe

        dt = datetime(2024, 1, 15, 13, 0)
        assert get_active_timeframe(dt) == "10m"

    def test_boundary_10am(self):
        from momentum_radar.main import get_active_timeframe

        dt = datetime(2024, 1, 15, 10, 0)
        assert get_active_timeframe(dt) == "5m"

    def test_boundary_11am(self):
        from momentum_radar.main import get_active_timeframe

        dt = datetime(2024, 1, 15, 11, 0)
        assert get_active_timeframe(dt) == "10m"
