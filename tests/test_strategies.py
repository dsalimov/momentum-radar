"""
tests/test_strategies.py – Unit tests for the five strategy engines.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trending_daily(n: int = 65) -> pd.DataFrame:
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = 100 + np.arange(n) * 0.8
    return pd.DataFrame(
        {
            "open":   closes - 0.3,
            "high":   closes + 1.0,
            "low":    closes - 1.0,
            "close":  closes,
            "volume": np.full(n, 1_500_000.0),
        },
        index=rng,
    )


def _make_breakout_daily(n: int = 65) -> pd.DataFrame:
    rng = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = np.full(n, 100.0)
    closes[-1] = 120.0
    return pd.DataFrame(
        {
            "open":   closes - 0.3,
            "high":   closes + 1.0,
            "low":    closes - 1.0,
            "close":  closes,
            "volume": np.full(n, 2_000_000.0),
        },
        index=rng,
    )


def _make_volume_spike_bars(n: int = 40) -> pd.DataFrame:
    rng = pd.date_range("2024-01-15 09:30", periods=n, freq="1min")
    volumes = np.full(n, 50_000.0)
    volumes[-1] = 400_000.0
    closes = np.full(n, 50.0)
    opens  = closes - 0.05
    highs  = closes + 0.1
    lows   = closes - 0.1
    opens[-1]  = 49.90
    closes[-1] = 50.10
    highs[-1]  = 50.10
    lows[-1]   = 49.88
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows,
         "close": closes, "volume": volumes},
        index=rng,
    )


# ---------------------------------------------------------------------------
# StrategySignal base dataclass
# ---------------------------------------------------------------------------

class TestStrategySignalBase:
    def test_confirmation_count_property(self):
        from momentum_radar.strategies.base import StrategySignal

        s = StrategySignal(ticker="T", strategy="scalp")
        assert s.confirmation_count == 0

        s.confirmations = ["A", "B", "C"]
        assert s.confirmation_count == 3

    def test_default_valid_is_false(self):
        from momentum_radar.strategies.base import StrategySignal

        s = StrategySignal(ticker="X", strategy="swing")
        assert s.valid is False

    def test_strategy_type_auto_derived_scalp(self):
        from momentum_radar.strategies.base import StrategySignal

        s = StrategySignal(ticker="T", strategy="scalp")
        assert s.strategy_type == "SCALP TRADE"

    def test_strategy_type_auto_derived_intraday(self):
        from momentum_radar.strategies.base import StrategySignal

        s = StrategySignal(ticker="T", strategy="intraday")
        assert s.strategy_type == "DAY TRADE"

    def test_strategy_type_auto_derived_swing(self):
        from momentum_radar.strategies.base import StrategySignal

        s = StrategySignal(ticker="T", strategy="swing")
        assert s.strategy_type == "SWING TRADE"

    def test_strategy_type_auto_derived_chart_pattern(self):
        from momentum_radar.strategies.base import StrategySignal

        s = StrategySignal(ticker="T", strategy="chart_pattern")
        assert s.strategy_type == "SWING TRADE"

    def test_strategy_type_auto_derived_unusual_volume(self):
        from momentum_radar.strategies.base import StrategySignal

        s = StrategySignal(ticker="T", strategy="unusual_volume")
        assert s.strategy_type == "DAY TRADE"

    def test_strategy_type_explicit_override(self):
        from momentum_radar.strategies.base import StrategySignal

        s = StrategySignal(ticker="T", strategy="scalp", strategy_type="DAY TRADE")
        assert s.strategy_type == "DAY TRADE"

    def test_target2_default_zero(self):
        from momentum_radar.strategies.base import StrategySignal

        s = StrategySignal(ticker="T", strategy="scalp")
        assert s.target2 == 0.0

    def test_options_flow_label_default_empty(self):
        from momentum_radar.strategies.base import StrategySignal

        s = StrategySignal(ticker="T", strategy="scalp")
        assert s.options_flow_label == ""


# ---------------------------------------------------------------------------
# scalp_strategy
# ---------------------------------------------------------------------------

class TestScalpStrategy:
    def test_returns_strategy_signal(self):
        from momentum_radar.strategies.scalp_strategy import evaluate
        from momentum_radar.strategies.base import StrategySignal

        result = evaluate("TEST")
        assert isinstance(result, StrategySignal)

    def test_strategy_name(self):
        from momentum_radar.strategies.scalp_strategy import evaluate

        result = evaluate("SPY")
        assert result.strategy == "scalp"

    def test_empty_data_not_valid(self):
        from momentum_radar.strategies.scalp_strategy import evaluate

        result = evaluate("NONE")
        assert result.valid is False

    def test_score_range(self):
        from momentum_radar.strategies.scalp_strategy import evaluate

        result = evaluate("TEST", bars=_make_volume_spike_bars(), daily=_make_trending_daily())
        assert 0 <= result.score <= 100

    def test_grade_is_valid(self):
        from momentum_radar.strategies.scalp_strategy import evaluate

        result = evaluate("TEST", bars=_make_volume_spike_bars(), daily=_make_trending_daily())
        assert result.grade in ("A+", "A", "B", "C")

    def test_ticker_preserved(self):
        from momentum_radar.strategies.scalp_strategy import evaluate

        result = evaluate("AAPL")
        assert result.ticker == "AAPL"

    def test_timeframe_is_set(self):
        from momentum_radar.strategies.scalp_strategy import evaluate

        result = evaluate("TEST")
        assert result.timeframe in ("2m", "5m")

    def test_entry_stop_target_populated_when_data_given(self):
        from momentum_radar.strategies.scalp_strategy import evaluate

        result = evaluate("TEST", bars=_make_volume_spike_bars(), daily=_make_trending_daily())
        if result.entry > 0:
            assert result.stop > 0
            assert result.target > 0


# ---------------------------------------------------------------------------
# intraday_strategy
# ---------------------------------------------------------------------------

class TestIntradayStrategy:
    def test_returns_strategy_signal(self):
        from momentum_radar.strategies.intraday_strategy import evaluate
        from momentum_radar.strategies.base import StrategySignal

        assert isinstance(evaluate("TEST"), StrategySignal)

    def test_strategy_name(self):
        from momentum_radar.strategies.intraday_strategy import evaluate

        assert evaluate("X").strategy == "intraday"

    def test_empty_data_not_valid(self):
        from momentum_radar.strategies.intraday_strategy import evaluate

        assert evaluate("NONE").valid is False

    def test_score_range(self):
        from momentum_radar.strategies.intraday_strategy import evaluate

        result = evaluate("T", bars=_make_volume_spike_bars(), daily=_make_trending_daily())
        assert 0 <= result.score <= 100

    def test_direction_default_buy(self):
        from momentum_radar.strategies.intraday_strategy import evaluate

        assert evaluate("T").direction == "BUY"


# ---------------------------------------------------------------------------
# swing_strategy
# ---------------------------------------------------------------------------

class TestSwingStrategy:
    def test_returns_strategy_signal(self):
        from momentum_radar.strategies.swing_strategy import evaluate
        from momentum_radar.strategies.base import StrategySignal

        assert isinstance(evaluate("TEST"), StrategySignal)

    def test_strategy_name(self):
        from momentum_radar.strategies.swing_strategy import evaluate

        assert evaluate("X").strategy == "swing"

    def test_timeframe_is_1h(self):
        from momentum_radar.strategies.swing_strategy import evaluate

        assert evaluate("X").timeframe == "1H"

    def test_empty_data_not_valid(self):
        from momentum_radar.strategies.swing_strategy import evaluate

        assert evaluate("NONE").valid is False

    def test_score_with_breakout_data(self):
        from momentum_radar.strategies.swing_strategy import evaluate

        result = evaluate("SPY", daily=_make_breakout_daily())
        assert 0 <= result.score <= 100

    def test_liquidity_sweep_confirmed(self):
        """Bars with wick below prior low and close above it fire liquidity sweep."""
        from momentum_radar.strategies.swing_strategy import _check_liquidity_sweep

        n = 15
        rng = pd.date_range("2024-01-01", periods=n, freq="B")
        closes = np.full(n, 50.0)
        lows = np.full(n, 49.5)
        # Last bar wicks below prior low but closes above
        lows[-1] = 48.0
        closes[-1] = 50.2
        daily = pd.DataFrame(
            {"open": closes - 0.3, "high": closes + 0.5,
             "low": lows, "close": closes, "volume": np.full(n, 1e6)},
            index=rng,
        )
        result = _check_liquidity_sweep(daily)
        assert result is not None
        assert "Liquidity" in result

    def test_strategy_type_is_swing(self):
        """Swing strategy sets strategy_type to SWING TRADE."""
        from momentum_radar.strategies.swing_strategy import evaluate

        result = evaluate("X")
        assert result.strategy_type == "SWING TRADE"

    def test_target2_populated_when_data_given(self):
        """target2 is set when daily data is available (swing has 2 targets)."""
        from momentum_radar.strategies.swing_strategy import evaluate

        result = evaluate("SPY", daily=_make_breakout_daily())
        if result.entry > 0:
            assert result.target2 > 0


# ---------------------------------------------------------------------------
# chart_pattern_strategy
# ---------------------------------------------------------------------------

class TestChartPatternStrategy:
    def test_returns_strategy_signal(self):
        from momentum_radar.strategies.chart_pattern_strategy import evaluate
        from momentum_radar.strategies.base import StrategySignal

        assert isinstance(evaluate("TEST"), StrategySignal)

    def test_strategy_name(self):
        from momentum_radar.strategies.chart_pattern_strategy import evaluate

        assert evaluate("X").strategy == "chart_pattern"

    def test_timeframe_is_daily(self):
        from momentum_radar.strategies.chart_pattern_strategy import evaluate

        assert evaluate("X").timeframe == "Daily"

    def test_empty_data_not_valid(self):
        from momentum_radar.strategies.chart_pattern_strategy import evaluate

        assert evaluate("NONE").valid is False

    def test_volume_contraction_detected(self):
        from momentum_radar.strategies.chart_pattern_strategy import _check_volume_contraction

        n = 30
        rng = pd.date_range("2024-01-01", periods=n, freq="B")
        # Volume declining in last 10 bars
        volumes = np.concatenate([np.full(20, 2_000_000.0), np.full(10, 800_000.0)])
        daily = pd.DataFrame(
            {"open": np.full(n, 50.0), "high": np.full(n, 51.0),
             "low": np.full(n, 49.0), "close": np.full(n, 50.5),
             "volume": volumes},
            index=rng,
        )
        result = _check_volume_contraction(daily)
        assert result is not None

    def test_volume_expansion_detected(self):
        from momentum_radar.strategies.chart_pattern_strategy import _check_volume_expansion

        n = 20
        rng = pd.date_range("2024-01-01", periods=n, freq="B")
        volumes = np.full(n, 1_000_000.0)
        volumes[-1] = 3_000_000.0  # 3× spike
        daily = pd.DataFrame(
            {"open": np.full(n, 50.0), "high": np.full(n, 51.0),
             "low": np.full(n, 49.0), "close": np.full(n, 50.5),
             "volume": volumes},
            index=rng,
        )
        result = _check_volume_expansion(daily)
        assert result is not None

    def test_strategy_type_is_swing(self):
        """Chart-pattern strategy sets strategy_type to SWING TRADE."""
        from momentum_radar.strategies.chart_pattern_strategy import evaluate

        result = evaluate("X")
        assert result.strategy_type == "SWING TRADE"

    def test_target2_populated_when_data_given(self):
        """target2 is set when daily data is available (chart_pattern has 2 targets)."""
        from momentum_radar.strategies.chart_pattern_strategy import evaluate

        result = evaluate("SPY", daily=_make_breakout_daily())
        if result.entry > 0:
            assert result.target2 > 0


# ---------------------------------------------------------------------------
# unusual_volume_strategy
# ---------------------------------------------------------------------------

class TestUnusualVolumeStrategy:
    def test_returns_strategy_signal(self):
        from momentum_radar.strategies.unusual_volume_strategy import evaluate
        from momentum_radar.strategies.base import StrategySignal

        assert isinstance(evaluate("TEST"), StrategySignal)

    def test_strategy_name(self):
        from momentum_radar.strategies.unusual_volume_strategy import evaluate

        assert evaluate("X").strategy == "unusual_volume"

    def test_timeframe_is_daily(self):
        from momentum_radar.strategies.unusual_volume_strategy import evaluate

        assert evaluate("X").timeframe == "Daily"

    def test_empty_data_not_valid(self):
        from momentum_radar.strategies.unusual_volume_strategy import evaluate

        assert evaluate("NONE").valid is False

    def test_volume_spike_detected(self):
        from momentum_radar.strategies.unusual_volume_strategy import _check_volume

        n = 30
        rng = pd.date_range("2024-01-01", periods=n, freq="B")
        volumes = np.full(n, 1_000_000.0)
        volumes[-1] = 3_000_000.0  # 3× average
        daily = pd.DataFrame(
            {"open": np.full(n, 50.0), "high": np.full(n, 51.0),
             "low": np.full(n, 49.0), "close": np.full(n, 50.5),
             "volume": volumes},
            index=rng,
        )
        result = _check_volume(daily)
        assert result is not None
        assert "3." in result or "Spike" in result

    def test_level_break_detected(self):
        from momentum_radar.strategies.unusual_volume_strategy import _check_level_break

        daily = _make_breakout_daily()
        result = _check_level_break(daily)
        assert result is not None
        assert "Resistance" in result

    def test_strong_body_close_detected(self):
        from momentum_radar.strategies.unusual_volume_strategy import _check_close_strength

        n = 5
        rng = pd.date_range("2024-01-01", periods=n, freq="B")
        # Last bar: open 48, close 53 (body=5), high 53, low 47.5 (range=5.5)
        opens  = np.full(n, 50.0); opens[-1]  = 48.0
        closes = np.full(n, 50.5); closes[-1] = 53.0
        highs  = np.full(n, 51.0); highs[-1]  = 53.0
        lows   = np.full(n, 49.0); lows[-1]   = 47.5
        daily = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows,
             "close": closes, "volume": np.full(n, 1e6)},
            index=rng,
        )
        result = _check_close_strength(daily)
        assert result is not None
