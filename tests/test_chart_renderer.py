"""
tests/test_chart_renderer.py – Unit tests for the chart renderer dispatcher.

Tests cover:
  * ``render_pattern_chart`` – routes to ``generate_pattern_chart`` when
    pattern geometry is provided; falls back to ``render_signal_chart``.
  * ``generate_signal_chart`` – unified dispatcher that routes chart_pattern
    strategy signals through the pattern renderer.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from momentum_radar.strategies.base import StrategySignal
from momentum_radar.ui.chart_renderer import (
    generate_signal_chart,
    render_pattern_chart,
    render_signal_chart,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 30, close: float = 100.0) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with a DatetimeIndex."""
    rng = pd.date_range("2024-01-02", periods=n, freq="B")
    np.random.seed(0)
    closes = close + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open": closes - 0.3,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": np.random.randint(100_000, 500_000, size=n).astype(float),
        },
        index=rng,
    )


def _make_signal(strategy: str = "swing", score: int = 80) -> StrategySignal:
    return StrategySignal(
        ticker="AAPL",
        strategy=strategy,
        direction="BUY",
        timeframe="1H",
        score=score,
        grade="A",
        entry=150.0,
        stop=147.0,
        target=156.0,
        rr=2.0,
        valid=True,
    )


# ---------------------------------------------------------------------------
# render_pattern_chart – geometry-aware routing
# ---------------------------------------------------------------------------

class TestRenderPatternChart:
    """render_pattern_chart should use generate_pattern_chart when geometry is given."""

    def test_falls_back_to_render_signal_chart_when_no_geometry(self, tmp_path):
        """Without pattern geometry render_signal_chart should be called."""
        signal = _make_signal("chart_pattern")
        df = _make_ohlcv()
        out = str(tmp_path / "out.png")

        with patch(
            "momentum_radar.ui.chart_renderer.render_signal_chart",
            return_value=out,
        ) as mock_render:
            result = render_pattern_chart(signal, df, output_path=out)

        mock_render.assert_called_once()
        assert result == out

    def test_routes_to_generate_pattern_chart_with_breakout_level(self, tmp_path):
        """When a breakout level is supplied generate_pattern_chart is used."""
        signal = _make_signal("chart_pattern")
        df = _make_ohlcv()
        out = str(tmp_path / "pattern.png")

        with patch(
            "momentum_radar.patterns.charts.generate_pattern_chart",
            return_value=out,
        ) as mock_gen:
            result = render_pattern_chart(
                signal,
                df,
                output_path=out,
                pattern_name="Ascending Triangle",
                breakout_level=152.0,
            )

        mock_gen.assert_called_once()
        assert result == out

    def test_routes_to_generate_pattern_chart_with_trendlines(self, tmp_path):
        """When trendlines are supplied generate_pattern_chart is used."""
        signal = _make_signal("chart_pattern")
        df = _make_ohlcv()
        out = str(tmp_path / "trendline.png")

        highs = [("2024-01-02", 101.0), ("2024-01-05", 102.0)]
        lows = [("2024-01-02", 99.0), ("2024-01-05", 100.0)]

        with patch(
            "momentum_radar.patterns.charts.generate_pattern_chart",
            return_value=out,
        ) as mock_gen:
            result = render_pattern_chart(
                signal,
                df,
                output_path=out,
                pattern_name="Wedge",
                trendline_highs=highs,
                trendline_lows=lows,
            )

        mock_gen.assert_called_once()
        # Verify the lines argument was passed correctly
        call_kwargs = mock_gen.call_args
        pattern_result = call_kwargs.args[2] if len(call_kwargs.args) >= 3 else call_kwargs.kwargs.get("pattern_result", {})
        assert "lines" in pattern_result
        assert result == out

    def test_falls_back_on_generate_pattern_chart_exception(self, tmp_path):
        """If generate_pattern_chart raises an exception the fallback is used."""
        signal = _make_signal("chart_pattern")
        df = _make_ohlcv()
        out = str(tmp_path / "fallback.png")

        with (
            patch(
                "momentum_radar.patterns.charts.generate_pattern_chart",
                side_effect=RuntimeError("chart error"),
            ),
            patch(
                "momentum_radar.ui.chart_renderer.render_signal_chart",
                return_value=out,
            ) as mock_fallback,
        ):
            result = render_pattern_chart(
                signal,
                df,
                output_path=out,
                breakout_level=152.0,
            )

        mock_fallback.assert_called_once()
        assert result == out


# ---------------------------------------------------------------------------
# generate_signal_chart – unified dispatcher
# ---------------------------------------------------------------------------

class TestGenerateSignalChart:
    """generate_signal_chart routes to the correct renderer."""

    def test_raises_for_empty_df(self):
        signal = _make_signal()
        with pytest.raises(ValueError, match="empty"):
            generate_signal_chart(signal, pd.DataFrame())

    def test_swing_signal_uses_render_signal_chart(self, tmp_path):
        """Swing signals (no pattern_result) should use render_signal_chart."""
        signal = _make_signal("swing")
        df = _make_ohlcv()
        out = str(tmp_path / "swing.png")

        with patch(
            "momentum_radar.ui.chart_renderer.render_signal_chart",
            return_value=out,
        ) as mock_render:
            result = generate_signal_chart(signal, df, output_path=out)

        mock_render.assert_called_once()
        assert result == out

    def test_intraday_signal_uses_render_signal_chart(self, tmp_path):
        """Intraday signals use render_signal_chart."""
        signal = _make_signal("intraday")
        df = _make_ohlcv()
        out = str(tmp_path / "intraday.png")

        with patch(
            "momentum_radar.ui.chart_renderer.render_signal_chart",
            return_value=out,
        ) as mock_render:
            result = generate_signal_chart(signal, df, output_path=out)

        mock_render.assert_called_once()
        assert result == out

    def test_chart_pattern_signal_without_pattern_result_uses_render_signal_chart(self, tmp_path):
        """chart_pattern without a pattern_result falls back to render_signal_chart."""
        signal = _make_signal("chart_pattern")
        df = _make_ohlcv()
        out = str(tmp_path / "cp_no_result.png")

        with patch(
            "momentum_radar.ui.chart_renderer.render_signal_chart",
            return_value=out,
        ) as mock_render:
            result = generate_signal_chart(signal, df, output_path=out)

        mock_render.assert_called_once()
        assert result == out

    def test_chart_pattern_signal_with_pattern_result_uses_render_pattern_chart(self, tmp_path):
        """chart_pattern + pattern_result routes through render_pattern_chart."""
        signal = _make_signal("chart_pattern")
        df = _make_ohlcv()
        out = str(tmp_path / "cp_with_result.png")

        pattern_result = {
            "pattern": "Cup and Handle",
            "confidence": 82,
            "state": None,
            "breakout_level_upper": 153.0,
            "breakout_level_lower": None,
            "lines": [],
            "key_points": [],
            "pattern_type": "structure",
        }

        with patch(
            "momentum_radar.ui.chart_renderer.render_pattern_chart",
            return_value=out,
        ) as mock_pattern:
            result = generate_signal_chart(
                signal, df, output_path=out, pattern_result=pattern_result
            )

        mock_pattern.assert_called_once()
        # Confirm breakout_level was extracted from pattern_result and passed through
        call_kwargs = mock_pattern.call_args.kwargs
        assert call_kwargs.get("breakout_level") == 153.0
        assert result == out

    def test_chart_pattern_with_trendlines_passes_lines(self, tmp_path):
        """Trendline segments from pattern_result are forwarded."""
        signal = _make_signal("chart_pattern")
        df = _make_ohlcv()
        out = str(tmp_path / "trendline.png")

        highs = [("2024-01-02", 101.5), ("2024-01-04", 102.0)]
        lows = [("2024-01-02", 99.0), ("2024-01-04", 99.5)]
        pattern_result = {
            "pattern": "Wedge",
            "breakout_level_upper": 102.5,
            "lines": [highs, lows],
        }

        with patch(
            "momentum_radar.ui.chart_renderer.render_pattern_chart",
            return_value=out,
        ) as mock_pattern:
            generate_signal_chart(signal, df, output_path=out, pattern_result=pattern_result)

        _, kwargs = mock_pattern.call_args
        assert kwargs.get("trendline_highs") == highs
        assert kwargs.get("trendline_lows") == lows
