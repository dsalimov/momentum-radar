"""
ui/chart_renderer.py – Strategy-aware dark-theme chart renderer.

Generates annotated PNG charts for Telegram / Discord alerts.

Chart style
-----------
* Pure black background
* Green candles for bullish bars, red for bearish
* Entry line  → green
* Stop line   → red
* Target line → blue
* S&D zones / pattern highlights → yellow dashed lines
* Volume panel below price

File naming: ``{SYMBOL}_{TIMEFRAME}_{TIMESTAMP}.png``

Minimal indicator overlay – no RSI / MACD clutter.

Functions
---------
* :func:`render_signal_chart`      – chart for a legacy :class:`StrategySignal`
* :func:`render_trade_setup_chart` – chart for a :class:`TradeSetup` (new setup
  detector); includes entry, stop, target markers and a VWAP overlay
* :func:`render_pattern_chart`     – chart with pattern overlays (trendlines,
  breakout levels, state annotations) drawn via
  :func:`~momentum_radar.patterns.charts.generate_pattern_chart`
* :func:`generate_signal_chart`    – unified dispatcher that routes any
  :class:`StrategySignal` to the correct renderer based on its strategy type
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from typing import Optional

import pandas as pd

from momentum_radar.strategies.base import StrategySignal

logger = logging.getLogger(__name__)

# Color constants (matplotlib named colors or hex)
_COLOR_ENTRY  = "#27AE60"   # green
_COLOR_STOP   = "#E74C3C"   # red
_COLOR_TARGET = "#2980B9"   # blue
_COLOR_ZONE   = "#FFD700"   # yellow (S&D zones / pattern highlights)
_COLOR_BG     = "#000000"   # pure black background

# Number of bars shown in the generated chart (keeps the image clean and fast)
_MAX_CHART_BARS: int = 60


def _build_output_path(ticker: str, timeframe: str) -> str:
    """Build a timestamped file path in the system temp directory."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ticker}_{timeframe}_{ts}.png".replace("/", "-")
    return os.path.join(tempfile.gettempdir(), name)


def render_signal_chart(
    signal: StrategySignal,
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    zone_high: Optional[float] = None,
    zone_low: Optional[float] = None,
) -> str:
    """Generate a dark-theme annotated chart for a strategy signal.

    Args:
        signal:      Evaluated strategy signal (provides entry/stop/target).
        df:          OHLCV DataFrame to plot.
        output_path: Optional path to save the PNG.  Temp file used if ``None``.
        zone_high:   Upper boundary of the supply/demand zone to shade (optional).
        zone_low:    Lower boundary of the supply/demand zone to shade (optional).

    Returns:
        Absolute path to the generated PNG file.

    Raises:
        ImportError: If *mplfinance* or *matplotlib* is not installed.
        ValueError:  If *df* is empty.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import mplfinance as mpf
    except ImportError as exc:
        raise ImportError(
            "mplfinance and matplotlib are required for chart rendering. "
            "Install with: pip install mplfinance matplotlib"
        ) from exc

    if df is None or df.empty:
        raise ValueError("DataFrame is empty – cannot render chart")

    path = output_path or _build_output_path(signal.ticker, signal.timeframe)

    # Slice to last _MAX_CHART_BARS bars for a clean, readable chart
    plot_df = df.iloc[-_MAX_CHART_BARS:].copy() if len(df) > _MAX_CHART_BARS else df.copy()

    # Ensure required columns
    for col in ("open", "high", "low", "close", "volume"):
        if col not in plot_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Build horizontal line annotations (entry, stop, target)
    hlines: dict = {}
    hline_colors: list = []

    lines_to_draw = []
    if signal.entry > 0:
        lines_to_draw.append((signal.entry, _COLOR_ENTRY))
    if signal.stop > 0:
        lines_to_draw.append((signal.stop, _COLOR_STOP))
    if signal.target > 0:
        lines_to_draw.append((signal.target, _COLOR_TARGET))

    level_prices = [p for p, _ in lines_to_draw]
    level_colors = [c for _, c in lines_to_draw]

    # mplfinance style
    mc = mpf.make_marketcolors(
        up="#27AE60",    # bullish candle
        down="#E74C3C",  # bearish candle
        edge="inherit",
        wick={"up": "#27AE60", "down": "#E74C3C"},
        volume={"up": "#1A6634", "down": "#7B1E1E"},
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        facecolor=_COLOR_BG,
        gridcolor="#2d2d4e",
        gridstyle="--",
        gridaxis="both",
    )

    # Additional plots for zone shading (yellow dashed, consistent with pattern highlights)
    add_plots = []
    if zone_high is not None and zone_low is not None:
        fill = mpf.make_addplot(
            [zone_high] * len(plot_df),
            type="line",
            color=_COLOR_ZONE,
            linestyle="--",
            width=1.2,
            alpha=0.85,
        )
        fill_low = mpf.make_addplot(
            [zone_low] * len(plot_df),
            type="line",
            color=_COLOR_ZONE,
            linestyle="--",
            width=1.2,
            alpha=0.85,
        )
        add_plots.extend([fill, fill_low])

    title = (
        f"{signal.ticker}  |  {signal.strategy.upper()}  |  {signal.timeframe}  "
        f"|  Score {signal.score}  |  R:R {signal.rr:.1f}"
    )

    fig, axes = mpf.plot(
        plot_df,
        type="candle",
        style=style,
        title=title,
        ylabel="Price",
        volume=True,
        addplot=add_plots if add_plots else None,
        hlines=dict(hlines=level_prices, colors=level_colors, linestyle="-", linewidths=1.2) if level_prices else None,
        figsize=(14, 8),
        returnfig=True,
        tight_layout=True,
    )

    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor=_COLOR_BG)
    plt.close(fig)
    logger.info("Chart saved: %s", path)
    return path


def render_pattern_chart(
    signal: StrategySignal,
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    pattern_name: Optional[str] = None,
    breakout_level: Optional[float] = None,
    trendline_highs: Optional[list] = None,
    trendline_lows: Optional[list] = None,
) -> str:
    """Generate a chart annotated with pattern-specific overlays.

    When *pattern_name* or pattern geometry (*breakout_level*, *trendline_highs*,
    *trendline_lows*) is provided the chart is rendered via
    :func:`~momentum_radar.patterns.charts.generate_pattern_chart` so that
    trendlines, breakout levels, and pattern state labels are drawn directly on
    the candlestick chart.

    Falls back to :func:`render_signal_chart` when no pattern geometry is
    available.

    Args:
        signal:          Evaluated strategy signal.
        df:              OHLCV DataFrame.
        output_path:     Optional save path.
        pattern_name:    Pattern name used in the chart title and overlays.
        breakout_level:  Horizontal resistance/support level drawn as a dashed
                         yellow line on the chart.
        trendline_highs: List of ``(date, price)`` tuples defining the upper
                         trendline.  Passed directly to
                         :func:`~momentum_radar.patterns.charts.generate_pattern_chart`.
        trendline_lows:  List of ``(date, price)`` tuples defining the lower
                         trendline.

    Returns:
        Absolute path to the generated PNG file.
    """
    has_pattern_geometry = (
        pattern_name is not None
        or breakout_level is not None
        or trendline_highs
        or trendline_lows
    )

    if has_pattern_geometry:
        try:
            from momentum_radar.patterns.charts import generate_pattern_chart

            # Build line segments list expected by generate_pattern_chart
            lines = []
            if trendline_highs and len(trendline_highs) >= 2:
                lines.append(trendline_highs)
            if trendline_lows and len(trendline_lows) >= 2:
                lines.append(trendline_lows)

            pattern_result = {
                "pattern": pattern_name or "Pattern",
                "confidence": signal.score,
                "state": None,
                "compression_ratio": None,
                "breakout_level_upper": breakout_level,
                "breakout_level_lower": None,
                "distance_to_breakout": None,
                "lines": lines,
                "key_points": [],
                "pattern_type": "structure",
                "candle_indices": [],
                "bias": "bullish" if signal.direction == "BUY" else "bearish",
            }
            return generate_pattern_chart(
                signal.ticker,
                df,
                pattern_result,
                output_path=output_path,
            )
        except Exception as exc:
            logger.warning(
                "Pattern chart generation failed for %s (%s); falling back to standard chart",
                signal.ticker,
                exc,
            )

    return render_signal_chart(signal, df, output_path=output_path)


def generate_signal_chart(
    signal: StrategySignal,
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    pattern_result: Optional[dict] = None,
) -> str:
    """Unified chart dispatcher – generate the appropriate chart for *signal*.

    Routing logic:

    * **chart_pattern strategy** with *pattern_result* provided → full pattern
      chart with trendlines, breakout levels, and pattern state annotation via
      :func:`render_pattern_chart`.
    * **chart_pattern strategy** without *pattern_result* → standard signal
      chart with entry/stop/target levels via :func:`render_signal_chart`.
    * **All other strategies** → standard signal chart with entry/stop/target
      levels and VWAP overlay.

    The generated chart always includes:

    * Dark nightclouds theme
    * Entry (green), stop (red), and target (blue) horizontal lines
    * Volume panel below price bars
    * Ticker, strategy type, timeframe, score, and R:R in the chart title

    Args:
        signal:         A :class:`~momentum_radar.strategies.base.StrategySignal`
                        produced by any strategy engine.
        df:             OHLCV DataFrame for the signal's timeframe.
        output_path:    Optional path to save the PNG file.  A timestamped temp
                        file is used when ``None``.
        pattern_result: Optional result dict from
                        :func:`~momentum_radar.patterns.detector.detect_pattern`.
                        When supplied for chart-pattern signals, pattern overlays
                        (trendlines, breakout levels, key-point markers) are
                        drawn directly on the chart.

    Returns:
        Absolute path to the generated PNG file.

    Raises:
        ImportError: If *mplfinance* or *matplotlib* is not installed.
        ValueError:  If *df* is empty or missing required OHLCV columns.

    Example::

        from momentum_radar.ui.chart_renderer import generate_signal_chart

        path = generate_signal_chart(signal, df)
        path = generate_signal_chart(signal, df, pattern_result=detector_output)
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty – cannot generate chart")

    is_pattern = signal.strategy == "chart_pattern"

    if is_pattern and pattern_result:
        breakout_level = pattern_result.get("breakout_level_upper")
        trendline_highs = None
        trendline_lows = None
        lines = pattern_result.get("lines", [])
        if len(lines) >= 1:
            trendline_highs = lines[0]
        if len(lines) >= 2:
            trendline_lows = lines[1]

        return render_pattern_chart(
            signal,
            df,
            output_path=output_path,
            pattern_name=pattern_result.get("pattern"),
            breakout_level=breakout_level,
            trendline_highs=trendline_highs,
            trendline_lows=trendline_lows,
        )

    return render_signal_chart(signal, df, output_path=output_path)


def render_trade_setup_chart(
    setup,
    df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> str:
    """Generate a dark-theme annotated chart for a :class:`TradeSetup`.

    Draws candlestick price bars with:

    * Green horizontal line  – entry price
    * Red horizontal line    – stop-loss price
    * Blue horizontal line   – profit target price
    * Orange dashed line     – VWAP overlay (computed from *df* if available)
    * Volume bars in lower panel

    Args:
        setup:       :class:`~momentum_radar.signals.setup_detector.TradeSetup`
                     object from the setup detector.
        df:          Intraday OHLCV DataFrame (1-min or active timeframe).
        output_path: Optional path to save the PNG.  A timestamped temp file
                     is used when ``None``.

    Returns:
        Absolute path to the generated PNG file.

    Raises:
        ImportError: If *mplfinance* or *matplotlib* is not installed.
        ValueError:  If *df* is empty.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import mplfinance as mpf
    except ImportError as exc:
        raise ImportError(
            "mplfinance and matplotlib are required for chart rendering. "
            "Install with: pip install mplfinance matplotlib"
        ) from exc

    if df is None or df.empty:
        raise ValueError("DataFrame is empty – cannot render trade-setup chart")

    timeframe = "intraday"
    path = output_path or _build_output_path(setup.ticker, timeframe)

    # Limit to last _MAX_CHART_BARS bars
    plot_df = df.iloc[-_MAX_CHART_BARS:].copy() if len(df) > _MAX_CHART_BARS else df.copy()

    for col in ("open", "high", "low", "close", "volume"):
        if col not in plot_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Compute VWAP overlay
    add_plots = []
    try:
        from momentum_radar.utils.indicators import compute_vwap
        typical = (plot_df["high"] + plot_df["low"] + plot_df["close"]) / 3
        cum_vol = plot_df["volume"].cumsum()
        cum_tp_vol = (typical * plot_df["volume"]).cumsum()
        vwap_series = cum_tp_vol / cum_vol.replace(0, float("nan"))
        add_plots.append(
            mpf.make_addplot(
                vwap_series,
                color="#F39C12",   # orange
                linestyle="--",
                width=1.5,
                label="VWAP",
            )
        )
    except Exception:
        pass  # VWAP overlay is optional

    # Horizontal level lines: entry (green), stop (red), target (blue)
    level_prices = []
    level_colors = []
    for price, color in [
        (setup.entry,  _COLOR_ENTRY),
        (setup.stop,   _COLOR_STOP),
        (setup.target, _COLOR_TARGET),
    ]:
        if price > 0:
            level_prices.append(price)
            level_colors.append(color)

    mc = mpf.make_marketcolors(
        up="#27AE60",
        down="#E74C3C",
        edge="inherit",
        wick={"up": "#27AE60", "down": "#E74C3C"},
        volume={"up": "#1A6634", "down": "#7B1E1E"},
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        facecolor=_COLOR_BG,
        gridcolor="#2d2d4e",
        gridstyle="--",
        gridaxis="both",
    )

    title = (
        f"{setup.ticker}  |  {setup.setup_type.value}  |  {setup.direction.value}"
        f"  |  RR 1:{setup.risk_reward:.1f}  |  RVOL {setup.rvol:.1f}x"
        f"  |  Confidence: {setup.confidence}"
    )

    hlines_arg = (
        dict(hlines=level_prices, colors=level_colors, linestyle="-", linewidths=1.5)
        if level_prices else None
    )

    fig, axes = mpf.plot(
        plot_df,
        type="candle",
        style=style,
        title=title,
        ylabel="Price",
        volume=True,
        addplot=add_plots if add_plots else None,
        hlines=hlines_arg,
        figsize=(14, 8),
        returnfig=True,
        tight_layout=True,
    )

    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor=_COLOR_BG)
    plt.close(fig)
    logger.info("Trade-setup chart saved: %s", path)
    return path
