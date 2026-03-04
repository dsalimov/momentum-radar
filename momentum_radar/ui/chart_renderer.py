"""
ui/chart_renderer.py – Strategy-aware dark-theme chart renderer.

Generates annotated PNG charts for Telegram / Discord alerts.

Chart style
-----------
* Dark background (nightclouds theme)
* Entry line  → green
* Stop line   → red
* Target line → blue
* S&D zones   → purple shading
* Volume panel below price

File naming: ``{SYMBOL}_{TIMEFRAME}_{TIMESTAMP}.png``

Minimal indicator overlay – no RSI / MACD clutter.
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
_COLOR_ZONE   = "#8E44AD"   # purple (S&D zones)
_COLOR_BG     = "#1a1a2e"   # dark background


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

    # Slice to last 60 bars
    plot_df = df.iloc[-60:].copy() if len(df) > 60 else df.copy()

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

    # Additional plots for zone shading
    add_plots = []
    if zone_high is not None and zone_low is not None:
        fill_color = _COLOR_ZONE
        fill = mpf.make_addplot(
            [zone_high] * len(plot_df),
            type="line",
            color=fill_color,
            linestyle="--",
            width=0.8,
            alpha=0.5,
        )
        fill_low = mpf.make_addplot(
            [zone_low] * len(plot_df),
            type="line",
            color=fill_color,
            linestyle="--",
            width=0.8,
            alpha=0.5,
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

    Delegates to :func:`render_signal_chart` and adds a breakout level
    annotation when provided.

    Args:
        signal:          Evaluated strategy signal.
        df:              OHLCV DataFrame.
        output_path:     Optional save path.
        pattern_name:    Pattern name used in the chart title.
        breakout_level:  Horizontal level drawn as a dashed white line.
        trendline_highs: Reserved for future trendline drawing.
        trendline_lows:  Reserved for future trendline drawing.

    Returns:
        Absolute path to the generated PNG file.
    """
    if breakout_level:
        signal = StrategySignal(
            ticker=signal.ticker,
            strategy=signal.strategy,
            direction=signal.direction,
            timeframe=signal.timeframe,
            score=signal.score,
            grade=signal.grade,
            confirmations=signal.confirmations,
            entry=signal.entry,
            stop=signal.stop,
            target=signal.target,
            rr=signal.rr,
            regime=signal.regime,
            htf_bias=signal.htf_bias,
            session=signal.session,
            fake_breakout_passed=signal.fake_breakout_passed,
            valid=signal.valid,
        )
    return render_signal_chart(signal, df, output_path=output_path)
