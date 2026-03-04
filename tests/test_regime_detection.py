"""
tests/test_regime_detection.py – Tests for the regime_detection module.
"""

import numpy as np
import pandas as pd
import pytest


def _make_trending_daily(n=80, direction="up"):
    """Daily bars with a clear trend."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    if direction == "up":
        closes = np.linspace(100, 130, n) + np.random.randn(n) * 0.3
    else:
        closes = np.linspace(130, 100, n) + np.random.randn(n) * 0.3
    highs = closes + 1.0
    lows = closes - 1.0
    return pd.DataFrame(
        {"open": closes - 0.5, "high": highs, "low": lows, "close": closes,
         "volume": np.full(n, 1_000_000.0)},
        index=rng,
    )


def _make_ranging_daily(n=80):
    """Daily bars oscillating in a flat range."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    np.random.seed(0)
    closes = 100.0 + np.sin(np.linspace(0, 4 * np.pi, n)) * 2.0
    highs = closes + 0.3
    lows = closes - 0.3
    return pd.DataFrame(
        {"open": closes - 0.1, "high": highs, "low": lows, "close": closes,
         "volume": np.full(n, 1_000_000.0)},
        index=rng,
    )


def _make_expanding_daily(n=80):
    """Daily bars with suddenly expanding ATR."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, 100.0, dtype=float)
    highs = closes + 0.5
    lows = closes - 0.5
    # Last 20 bars: huge range
    highs[-20:] = closes[-20:] + 10.0
    lows[-20:] = closes[-20:] - 10.0
    return pd.DataFrame(
        {"open": closes - 0.2, "high": highs, "low": lows, "close": closes,
         "volume": np.full(n, 1_000_000.0)},
        index=rng,
    )


def _make_compressing_daily(n=80):
    """Daily bars with shrinking ATR."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, 100.0, dtype=float)
    # First bars have normal range; last 20 bars are squeezed
    highs = closes + 2.0
    lows = closes - 2.0
    highs[-20:] = closes[-20:] + 0.1
    lows[-20:] = closes[-20:] - 0.1
    return pd.DataFrame(
        {"open": closes - 0.2, "high": highs, "low": lows, "close": closes,
         "volume": np.full(n, 1_000_000.0)},
        index=rng,
    )


class TestDetectRegime:
    def test_returns_string(self):
        from momentum_radar.services.regime_detection import detect_regime

        daily = _make_ranging_daily()
        result = detect_regime(daily)
        assert isinstance(result, str)
        assert result in ("trending", "ranging", "expanding", "compressing")

    def test_none_data_returns_ranging(self):
        from momentum_radar.services.regime_detection import detect_regime

        assert detect_regime(None) == "ranging"

    def test_insufficient_data_returns_ranging(self):
        from momentum_radar.services.regime_detection import detect_regime

        rng = pd.date_range("2024-01-01", periods=10, freq="B")
        tiny = pd.DataFrame(
            {"open": [100.0]*10, "high": [101.0]*10, "low": [99.0]*10,
             "close": [100.5]*10, "volume": [1e6]*10},
            index=rng,
        )
        assert detect_regime(tiny) == "ranging"

    def test_expanding_regime_detected(self):
        from momentum_radar.services.regime_detection import detect_regime

        daily = _make_expanding_daily()
        result = detect_regime(daily)
        assert result == "expanding"

    def test_compressing_regime_detected(self):
        from momentum_radar.services.regime_detection import detect_regime

        daily = _make_compressing_daily()
        result = detect_regime(daily)
        assert result == "compressing"


class TestGetRegimeContext:
    def test_returns_expected_keys(self):
        from momentum_radar.services.regime_detection import get_regime_context

        daily = _make_ranging_daily()
        ctx = get_regime_context(daily)
        for key in ("regime", "atr", "trend_direction", "is_trending",
                    "is_ranging", "is_expanding", "is_compressing"):
            assert key in ctx

    def test_bool_flags_consistent_with_regime(self):
        from momentum_radar.services.regime_detection import get_regime_context

        daily = _make_ranging_daily()
        ctx = get_regime_context(daily)
        active_flags = [ctx["is_trending"], ctx["is_ranging"],
                        ctx["is_expanding"], ctx["is_compressing"]]
        # Exactly one flag should be True
        assert sum(active_flags) == 1

    def test_none_data_returns_ranging(self):
        from momentum_radar.services.regime_detection import get_regime_context

        ctx = get_regime_context(None)
        assert ctx["regime"] == "ranging"
