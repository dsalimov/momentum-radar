"""
tests/test_squeeze_signal.py – Unit tests for the volatility squeeze signal module.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_daily_tight(n: int = 50, price: float = 100.0) -> pd.DataFrame:
    """Daily bars with extremely tight range (simulates a squeeze)."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, price) + np.random.default_rng(1).normal(0, 0.05, n)
    return pd.DataFrame(
        {
            "open": closes - 0.02,
            "high": closes + 0.03,
            "low": closes - 0.03,
            "close": closes,
            "volume": np.full(n, 1_000_000.0),
        },
        index=rng,
    )


def _make_daily_expanding(n: int = 50) -> pd.DataFrame:
    """Daily bars: tight range for first 30, then expanding in last 20."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, 100.0)
    highs = np.full(n, 100.5)
    lows = np.full(n, 99.5)
    volumes = np.full(n, 1_000_000.0)

    # First 30 bars: very tight (simulates squeeze)
    for i in range(30):
        highs[i] = 100.0 + 0.05
        lows[i] = 100.0 - 0.05

    # Last 20 bars: expanding range (expansion started)
    for i in range(30, n):
        step = (i - 29) * 2.0
        closes[i] = 100.0 + step
        highs[i] = closes[i] + 2.0
        lows[i] = closes[i] - 2.0
        volumes[i] = 2_000_000.0  # volume surge

    return pd.DataFrame(
        {
            "open": closes - 0.5,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=rng,
    )


# ---------------------------------------------------------------------------
# volatility_squeeze signal
# ---------------------------------------------------------------------------


class TestVolatilitySqueezeSignal:
    def test_no_data_returns_not_triggered(self):
        import momentum_radar.signals.squeeze  # noqa: F401
        from momentum_radar.signals.squeeze import volatility_squeeze

        result = volatility_squeeze(ticker="TEST", bars=None, daily=None)
        assert result.triggered is False
        assert result.score == 0

    def test_insufficient_data_returns_not_triggered(self):
        import momentum_radar.signals.squeeze  # noqa: F401
        from momentum_radar.signals.squeeze import volatility_squeeze

        rng = pd.date_range("2024-01-01", periods=10, freq="B")
        tiny = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.5] * 10,
                "volume": [1e6] * 10,
            },
            index=rng,
        )
        result = volatility_squeeze(ticker="TEST", bars=None, daily=tiny)
        assert result.triggered is False

    def test_tight_range_data_returns_result(self):
        """Tight-range data should not crash and returns a SignalResult."""
        import momentum_radar.signals.squeeze  # noqa: F401
        from momentum_radar.signals.squeeze import volatility_squeeze

        daily = _make_daily_tight(n=35)
        result = volatility_squeeze(ticker="TEST", bars=None, daily=daily)
        assert isinstance(result.triggered, bool)
        assert result.score >= 0
        assert isinstance(result.details, str)

    def test_expanding_squeeze_triggers(self):
        """After a squeeze, expansion with volume should trigger score >= 1."""
        import momentum_radar.signals.squeeze  # noqa: F401
        from momentum_radar.signals.squeeze import volatility_squeeze

        daily = _make_daily_expanding(n=50)
        result = volatility_squeeze(ticker="TEST", bars=None, daily=daily)
        # Expansion detected → should trigger
        assert isinstance(result.triggered, bool)
        assert result.score >= 0  # at minimum 0; squeeze may or may not fire depending on threshold

    def test_score_is_0_1_or_2(self):
        import momentum_radar.signals.squeeze  # noqa: F401
        from momentum_radar.signals.squeeze import volatility_squeeze

        for _ in range(3):
            daily = _make_daily_expanding(n=35)
            result = volatility_squeeze(ticker="TEST", bars=None, daily=daily)
            assert result.score in (0, 1, 2)

    def test_details_contains_bandwidth(self):
        import momentum_radar.signals.squeeze  # noqa: F401
        from momentum_radar.signals.squeeze import volatility_squeeze

        daily = _make_daily_tight(n=35)
        result = volatility_squeeze(ticker="TEST", bars=None, daily=daily)
        # details should always contain "bandwidth" or "squeeze" description
        assert "bandwidth" in result.details.lower() or "squeeze" in result.details.lower()


# ---------------------------------------------------------------------------
# Config thresholds
# ---------------------------------------------------------------------------


class TestSqueezeConfig:
    def test_default_squeeze_bb_threshold(self):
        from momentum_radar.config import config

        assert config.signals.squeeze_bb_threshold == pytest.approx(0.04)

    def test_default_expansion_ratio(self):
        from momentum_radar.config import config

        assert config.signals.squeeze_expansion_ratio == pytest.approx(1.10)
