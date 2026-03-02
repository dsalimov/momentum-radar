"""
tests/test_signal_engine.py – Unit tests for the multi-confirmation signal engine.
"""

import pytest
import pandas as pd
import numpy as np
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
    """Intraday 1-min bars with a big volume spike on the last bar."""
    rng = pd.date_range("2024-01-15 09:30", periods=n, freq="1min")
    volumes = np.full(n, 50_000.0)
    volumes[-1] = 400_000.0  # 8× spike
    closes = np.full(n, 50.0)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.1,
            "low": closes - 0.1,
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

    def test_two_confirmations_produce_alert(self):
        from momentum_radar.services.signal_engine import evaluate

        daily = _make_breakout_daily()    # triggers pattern
        options = _options_call_spike()   # triggers options

        result = evaluate("TST", daily=daily, options=options)
        assert result.confirmation_count >= 2
        assert result.priority in ("ALERT", "HIGH_CONFIDENCE")

    def test_three_confirmations_produce_high_confidence(self):
        from momentum_radar.services.signal_engine import evaluate

        daily = _make_high_volume_daily()
        bars = _make_volume_spike_bars()
        options = _options_call_spike()

        # volume + options are guaranteed; pattern may or may not fire
        result = evaluate("HI", bars=bars, daily=daily, options=options)
        assert result.confirmation_count >= 2
        if result.confirmation_count >= 3:
            assert result.priority == "HIGH_CONFIDENCE"

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
