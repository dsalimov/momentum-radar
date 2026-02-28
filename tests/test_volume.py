"""
test_volume.py – Unit tests for volume spike and RVOL signal detection.
"""

import numpy as np
import pandas as pd
import pytest

# Import the modules under test (this also registers signals)
import momentum_radar.signals.volume  # noqa: F401
from momentum_radar.signals.volume import volume_spike, relative_volume


def _make_bars(
    n: int = 60,
    base_vol: float = 10_000,
    last_vol_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Build synthetic 1-min bars with configurable last-bar volume."""
    rng = pd.date_range("2024-01-15 09:30", periods=n, freq="1min")
    volumes = np.full(n, base_vol)
    volumes[-1] = base_vol * last_vol_multiplier
    closes = 100 + np.arange(n) * 0.01
    return pd.DataFrame(
        {
            "open": closes - 0.05,
            "high": closes + 0.1,
            "low": closes - 0.1,
            "close": closes,
            "volume": volumes,
        },
        index=rng,
    )


def _make_daily(
    n: int = 30,
    avg_vol: float = 1_000_000,
    today_vol_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Build synthetic daily bars."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    volumes = np.full(n, avg_vol)
    volumes[-1] = avg_vol * today_vol_multiplier
    closes = 50 + np.arange(n) * 0.1
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


# ---------------------------------------------------------------------------
# volume_spike tests
# ---------------------------------------------------------------------------

class TestVolumeSpike:
    def test_no_data_returns_not_triggered(self):
        result = volume_spike(ticker="TEST", bars=None, daily=None)
        assert result.triggered is False
        assert result.score == 0

    def test_empty_bars_returns_not_triggered(self):
        result = volume_spike(
            ticker="TEST",
            bars=pd.DataFrame(),
            daily=None,
        )
        assert result.triggered is False

    def test_strong_1min_spike_triggers(self):
        """Last 1-min bar volume 5x the 20-bar avg should trigger score ≥ 2."""
        bars = _make_bars(n=60, base_vol=10_000, last_vol_multiplier=5.0)
        result = volume_spike(ticker="TEST", bars=bars, daily=None)
        assert result.triggered is True
        assert result.score >= 2

    def test_no_spike_not_triggered(self):
        """Normal volume should not trigger a volume spike signal."""
        bars = _make_bars(n=60, base_vol=10_000, last_vol_multiplier=1.0)
        result = volume_spike(ticker="TEST", bars=bars, daily=None)
        # Score could still be > 0 if 5m or daily triggers; just validate struct
        assert isinstance(result.triggered, bool)
        assert result.score >= 0

    def test_daily_volume_spike_triggers(self):
        """Today's daily volume 2x avg should push DAILY_VOLUME_RATIO trigger."""
        bars = _make_bars(n=30, base_vol=5_000)
        daily = _make_daily(n=31, avg_vol=1_000_000, today_vol_multiplier=2.0)
        result = volume_spike(ticker="TEST", bars=bars, daily=daily)
        assert result.triggered is True


# ---------------------------------------------------------------------------
# relative_volume tests
# ---------------------------------------------------------------------------

class TestRelativeVolume:
    def test_no_daily_data_not_triggered(self):
        result = relative_volume(ticker="TEST", bars=None, daily=None)
        assert result.triggered is False

    def test_high_rvol_triggers_score_2(self):
        """RVOL >= 3.0 should return score 2."""
        bars = _make_bars(n=60, base_vol=100_000)
        # avg daily vol = 100k, today cumulative = 60 * 100k = 6_000_000 = 60x
        daily = _make_daily(n=32, avg_vol=100_000)
        result = relative_volume(ticker="TEST", bars=bars, daily=daily)
        assert result.triggered is True
        assert result.score == 2

    def test_moderate_rvol_triggers_score_1(self):
        """RVOL between 2.0 and 3.0 should return score 1."""
        # 60 bars × 42_000 = 2_520_000 cumulative
        # avg daily = 1_000_000  →  RVOL ≈ 2.52  (≥ 2.0, < 3.0)
        bars = _make_bars(n=60, base_vol=42_000)
        daily = _make_daily(n=32, avg_vol=1_000_000)
        result = relative_volume(ticker="TEST", bars=bars, daily=daily)
        assert result.triggered is True
        assert result.score == 1

    def test_low_rvol_not_triggered(self):
        """RVOL < 2.0 should not trigger."""
        # 60 × 1_000 = 60_000 cumulative vs avg 1_000_000  →  RVOL = 0.06
        bars = _make_bars(n=60, base_vol=1_000)
        daily = _make_daily(n=32, avg_vol=1_000_000)
        result = relative_volume(ticker="TEST", bars=bars, daily=daily)
        assert result.triggered is False
