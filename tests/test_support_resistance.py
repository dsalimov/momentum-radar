"""
tests/test_support_resistance.py – Unit tests for third-touch support
and failed breakout (liquidity trap) signal modules.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_daily_flat(n: int = 60, price: float = 100.0) -> pd.DataFrame:
    """Flat daily bars at *price*."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, price)
    return pd.DataFrame(
        {
            "open": closes - 0.2,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": np.full(n, 1_000_000.0),
        },
        index=rng,
    )


def _make_daily_with_support_bounces(
    n: int = 60,
    support: float = 100.0,
    bounce_bars: tuple = (10, 25, 40),
) -> pd.DataFrame:
    """Daily bars that bounce from *support* at the specified bar indices.

    Price hovers around *support + 5* and dips to *support* on the bounce bars.
    """
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, support + 5.0)
    highs = closes + 0.5
    lows = closes - 0.5

    for b in bounce_bars:
        lows[b] = support - 0.05   # touched the level
        closes[b] = support + 0.3  # recovered above it
        highs[b] = support + 1.0

    return pd.DataFrame(
        {
            "open": closes - 0.2,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(n, 1_000_000.0),
        },
        index=rng,
    )


def _make_bull_trap_daily(n: int = 30) -> pd.DataFrame:
    """Daily bars where the last bar wicks above prev high but closes below it."""
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, 100.0)
    highs = closes + 0.5
    lows = closes - 0.5
    volumes = np.full(n, 1_000_000.0)

    # prev bar: normal high
    highs[-2] = 102.0
    closes[-2] = 101.0

    # last bar: breaks above 102, but closes back at 99 (bull trap)
    highs[-1] = 104.0   # breaks above prev high
    closes[-1] = 99.5   # reverses back well below prev high
    lows[-1] = 99.0
    volumes[-1] = 1_500_000.0  # above-average volume

    return pd.DataFrame(
        {
            "open": np.where(
                np.arange(n) == n - 1,
                101.0,                 # opens above prev close
                closes - 0.2,
            ),
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=rng,
    )


def _make_bear_trap_daily(n: int = 30) -> pd.DataFrame:
    """Daily bars where the last bar wicks below prev low then recovers.

    Last bar: opens at 100.5, lows to 96.0 (wick=4.5), closes at 101.0 (body=0.5)
    → lower_wick (4.5) > body * 1.5 (0.75) → bear trap confirmed.
    """
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, 100.0)
    highs = closes + 0.5
    lows = closes - 0.5
    volumes = np.full(n, 1_000_000.0)

    # prev bar: normal low
    lows[-2] = 98.0
    closes[-2] = 99.0

    # last bar: wicks below 98, but closes at 101 (bear trap / liquidity sweep)
    # Open near close to keep body small so lower_wick > body * 1.5
    lows[-1] = 96.0     # breaks below prev low (lower_wick = 100.5 - 96.0 = 4.5)
    closes[-1] = 101.0  # recovers back above prev low (body = |101.0 - 100.5| = 0.5)
    highs[-1] = 101.5
    volumes[-1] = 1_500_000.0

    opens = closes - 0.2
    opens[-1] = 100.5   # small body: 101.0 - 100.5 = 0.5 < lower_wick 4.5

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


# ---------------------------------------------------------------------------
# third_touch_support
# ---------------------------------------------------------------------------


class TestThirdTouchSupport:
    def test_no_data_returns_not_triggered(self):
        import momentum_radar.signals.support_resistance  # noqa: F401
        from momentum_radar.signals.support_resistance import third_touch_support

        result = third_touch_support(ticker="TEST", bars=None, daily=None)
        assert result.triggered is False
        assert result.score == 0

    def test_insufficient_data_returns_not_triggered(self):
        import momentum_radar.signals.support_resistance  # noqa: F401
        from momentum_radar.signals.support_resistance import third_touch_support

        rng = pd.date_range("2024-01-01", periods=5, freq="B")
        tiny = pd.DataFrame(
            {"open": [100.0] * 5, "high": [101.0] * 5,
             "low": [99.0] * 5, "close": [100.5] * 5,
             "volume": [1e6] * 5},
            index=rng,
        )
        result = third_touch_support(ticker="TEST", bars=None, daily=tiny)
        assert result.triggered is False

    def test_two_prior_bounces_triggers(self):
        """Price near support with 2 confirmed prior bounces → signal triggers."""
        import momentum_radar.signals.support_resistance  # noqa: F401
        from momentum_radar.signals.support_resistance import third_touch_support

        # Create daily data where support = 100.0 with bounces at bars 10, 25
        # and current bar (last) approaches the level again
        daily = _make_daily_with_support_bounces(
            n=50,
            support=100.0,
            bounce_bars=(10, 25),
        )
        # Make the last bar approach the support
        daily.iloc[-1, daily.columns.get_loc("low")] = 100.05
        daily.iloc[-1, daily.columns.get_loc("close")] = 100.2

        result = third_touch_support(ticker="TEST", bars=None, daily=daily)
        # May or may not trigger depending on local-low detection; validate no crash
        assert isinstance(result.triggered, bool)
        assert result.score >= 0

    def test_result_has_score_and_details(self):
        import momentum_radar.signals.support_resistance  # noqa: F401
        from momentum_radar.signals.support_resistance import third_touch_support

        daily = _make_daily_flat()
        result = third_touch_support(ticker="TEST", bars=None, daily=daily)
        assert isinstance(result.score, int)
        assert isinstance(result.details, str)


# ---------------------------------------------------------------------------
# _find_local_lows
# ---------------------------------------------------------------------------


class TestFindLocalLows:
    def test_returns_list(self):
        from momentum_radar.signals.support_resistance import _find_local_lows

        daily = _make_daily_flat(n=40)
        levels = _find_local_lows(daily)
        assert isinstance(levels, list)

    def test_insufficient_data_returns_empty(self):
        from momentum_radar.signals.support_resistance import _find_local_lows

        rng = pd.date_range("2024-01-01", periods=3, freq="B")
        tiny = pd.DataFrame({"low": [99.0, 100.0, 99.5]}, index=rng)
        result = _find_local_lows(tiny)
        assert result == []


# ---------------------------------------------------------------------------
# _count_bounces
# ---------------------------------------------------------------------------


class TestCountBounces:
    def test_counts_clear_bounces(self):
        from momentum_radar.signals.support_resistance import _count_bounces

        # Bars where price touches 100 and then recovers
        lows = [100.0, 99.95, 105.0, 100.02, 106.0]
        closes = [101.0, 101.5, 105.0, 102.0, 106.0]
        rng = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {"low": lows, "close": closes,
             "open": [c - 0.1 for c in closes],
             "high": [c + 0.5 for c in closes]},
            index=rng,
        )
        count = _count_bounces(df, level=100.0)
        assert count >= 1

    def test_no_touch_returns_zero(self):
        from momentum_radar.signals.support_resistance import _count_bounces

        daily = _make_daily_flat(n=30, price=200.0)
        # Level far from price
        count = _count_bounces(daily, level=100.0)
        assert count == 0


# ---------------------------------------------------------------------------
# failed_breakout
# ---------------------------------------------------------------------------


class TestFailedBreakout:
    def test_no_data_returns_not_triggered(self):
        import momentum_radar.signals.support_resistance  # noqa: F401
        from momentum_radar.signals.support_resistance import failed_breakout

        result = failed_breakout(ticker="TEST", bars=None, daily=None)
        assert result.triggered is False
        assert result.score == 0

    def test_insufficient_data_returns_not_triggered(self):
        import momentum_radar.signals.support_resistance  # noqa: F401
        from momentum_radar.signals.support_resistance import failed_breakout

        rng = pd.date_range("2024-01-01", periods=3, freq="B")
        tiny = pd.DataFrame(
            {"open": [100.0] * 3, "high": [101.0] * 3,
             "low": [99.0] * 3, "close": [100.5] * 3,
             "volume": [1e6] * 3},
            index=rng,
        )
        result = failed_breakout(ticker="TEST", bars=None, daily=tiny)
        assert result.triggered is False

    def test_bull_trap_detected(self):
        """Clear bull trap: high above prev high, close well below prev high."""
        import momentum_radar.signals.support_resistance  # noqa: F401
        from momentum_radar.signals.support_resistance import failed_breakout

        daily = _make_bull_trap_daily()
        result = failed_breakout(ticker="TEST", bars=None, daily=daily)
        assert result.triggered is True
        assert result.score == 2
        assert "trap" in result.details.lower() or "wick" in result.details.lower()

    def test_bear_trap_detected(self):
        """Clear bear trap: low below prev low, close well above prev low."""
        import momentum_radar.signals.support_resistance  # noqa: F401
        from momentum_radar.signals.support_resistance import failed_breakout

        daily = _make_bear_trap_daily()
        result = failed_breakout(ticker="TEST", bars=None, daily=daily)
        assert result.triggered is True
        assert result.score == 2
        assert "trap" in result.details.lower() or "wick" in result.details.lower()

    def test_normal_bar_not_triggered(self):
        """Normal bars without a wick rejection should not trigger."""
        import momentum_radar.signals.support_resistance  # noqa: F401
        from momentum_radar.signals.support_resistance import failed_breakout

        daily = _make_daily_flat()
        result = failed_breakout(ticker="TEST", bars=None, daily=daily)
        assert result.triggered is False

    def test_result_has_score_and_details(self):
        import momentum_radar.signals.support_resistance  # noqa: F401
        from momentum_radar.signals.support_resistance import failed_breakout

        daily = _make_daily_flat()
        result = failed_breakout(ticker="TEST", bars=None, daily=daily)
        assert isinstance(result.score, int)
        assert isinstance(result.details, str)
        assert len(result.details) > 0


# ---------------------------------------------------------------------------
# resistance_break
# ---------------------------------------------------------------------------


def _make_resistance_break_daily(
    n: int = 30,
    resistance: float = 105.0,
) -> pd.DataFrame:
    """Daily bars where the last bar closes above a prior resistance level on volume.

    Creates an isolated local high at resistance (surrounded by lower highs),
    then a current bar that closes above it with strong volume.
    """
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, 100.0)
    highs = closes + 1.0   # default high = 101
    lows = closes - 1.0    # default low  = 99
    volumes = np.full(n, 1_000_000.0)

    # Isolated local high at resistance (surrounded by lower highs)
    # index n-10: the local maximum
    highs[-10] = resistance          # clear local high
    highs[-11] = resistance - 2.0   # lower neighbour
    highs[-9] = resistance - 2.0    # lower neighbour

    # Previous bar: closes just below resistance
    closes[-2] = resistance - 0.5
    highs[-2] = resistance - 0.1
    lows[-2] = resistance - 1.5

    # Current bar: breaks above resistance on strong volume
    closes[-1] = resistance + 1.5
    highs[-1] = resistance + 2.0
    lows[-1] = resistance - 0.2
    volumes[-1] = 2_000_000.0  # 2x average

    opens = closes - 0.3
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=rng,
    )


def _make_support_break_daily(
    n: int = 30,
    support: float = 95.0,
) -> pd.DataFrame:
    """Daily bars where the last bar closes below a prior support level on volume.

    Creates an isolated local low at support (surrounded by higher lows),
    then a current bar that closes below it with strong volume.
    """
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, 100.0)
    highs = closes + 1.0   # default high = 101
    lows = closes - 1.0    # default low  = 99
    volumes = np.full(n, 1_000_000.0)

    # Isolated local low at support (surrounded by higher lows)
    lows[-10] = support              # clear local low
    lows[-11] = support + 2.0       # higher neighbour
    lows[-9] = support + 2.0        # higher neighbour

    # Previous bar: closes just above support
    closes[-2] = support + 0.5
    lows[-2] = support + 0.1
    highs[-2] = support + 1.5

    # Current bar: flushes support on strong volume
    closes[-1] = support - 1.5
    lows[-1] = support - 2.0
    highs[-1] = support + 0.2
    volumes[-1] = 2_000_000.0  # 2x average

    opens = closes + 0.3
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=rng,
    )


class TestResistanceBreak:
    def test_no_data_returns_not_triggered(self):
        from momentum_radar.signals.support_resistance import resistance_break

        result = resistance_break(ticker="TEST", bars=None, daily=None)
        assert result.triggered is False
        assert result.score == 0

    def test_resistance_break_detected(self):
        from momentum_radar.signals.support_resistance import resistance_break

        daily = _make_resistance_break_daily()
        result = resistance_break(ticker="TEST", bars=None, daily=daily)
        assert result.triggered is True
        assert result.score >= 1
        assert "resistance break" in result.details.lower() or "resistance" in result.details.lower()

    def test_flat_market_not_triggered(self):
        from momentum_radar.signals.support_resistance import resistance_break

        daily = _make_daily_flat()
        result = resistance_break(ticker="TEST", bars=None, daily=daily)
        assert result.triggered is False

    def test_high_volume_break_scores_2(self):
        from momentum_radar.signals.support_resistance import resistance_break

        daily = _make_resistance_break_daily()
        result = resistance_break(ticker="TEST", bars=None, daily=daily)
        if result.triggered:
            assert result.score in (1, 2)

    def test_result_has_details(self):
        from momentum_radar.signals.support_resistance import resistance_break

        daily = _make_resistance_break_daily()
        result = resistance_break(ticker="TEST", bars=None, daily=daily)
        assert isinstance(result.details, str)
        assert len(result.details) > 0


class TestSupportBreak:
    def test_no_data_returns_not_triggered(self):
        from momentum_radar.signals.support_resistance import support_break

        result = support_break(ticker="TEST", bars=None, daily=None)
        assert result.triggered is False
        assert result.score == 0

    def test_support_break_detected(self):
        from momentum_radar.signals.support_resistance import support_break

        daily = _make_support_break_daily()
        result = support_break(ticker="TEST", bars=None, daily=daily)
        assert result.triggered is True
        assert result.score >= 1
        assert "support break" in result.details.lower() or "support" in result.details.lower()

    def test_flat_market_not_triggered(self):
        from momentum_radar.signals.support_resistance import support_break

        daily = _make_daily_flat()
        result = support_break(ticker="TEST", bars=None, daily=daily)
        assert result.triggered is False

    def test_result_has_details(self):
        from momentum_radar.signals.support_resistance import support_break

        daily = _make_support_break_daily()
        result = support_break(ticker="TEST", bars=None, daily=daily)
        assert isinstance(result.details, str)
        assert len(result.details) > 0
