"""
charts.py - Candlestick chart generation with pattern annotations.

Generates publication-quality PNG charts using *mplfinance* with:
- Trendlines drawn as solid lines
- Compression zone shaded between trendlines
- Pattern state label ("FORMING" / "NEAR BREAK")
- Breakout level shown as a horizontal dashed line
- Dark nightclouds theme
"""

import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def generate_pattern_chart(
    ticker: str,
    df: pd.DataFrame,
    pattern_result: Dict,
    output_path: Optional[str] = None,
) -> str:
    """Generate a candlestick chart with pattern annotations.

    Args:
        ticker:         Stock ticker symbol.
        df:             OHLCV DataFrame (daily bars).
        pattern_result: Result dict from
                        :func:`~momentum_radar.patterns.detector.detect_pattern`.
        output_path:    Optional path to save the PNG.  If ``None``, a
                        temporary file is created via :func:`tempfile.mkstemp`.

    Returns:
        Absolute path to the generated PNG file.

    Raises:
        ImportError: If *mplfinance* or *matplotlib* is not installed.
    """
    try:
        import mplfinance as mpf
        import matplotlib
        try:
            matplotlib.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as exc:
        raise ImportError(
            "mplfinance and matplotlib are required for chart generation."
        ) from exc

    from momentum_radar.patterns.detector import PatternState

    pattern_name = pattern_result.get("pattern", "Pattern")
    confidence = pattern_result.get("confidence", 0)
    key_points: List[Tuple] = pattern_result.get("key_points", [])
    state = pattern_result.get("state")
    compression_ratio = pattern_result.get("compression_ratio")
    breakout_level_upper = pattern_result.get("breakout_level_upper")
    breakout_level_lower = pattern_result.get("breakout_level_lower")
    distance_to_breakout = pattern_result.get("distance_to_breakout")
    lines = pattern_result.get("lines", [])

    # Use last 30 bars at most for readability
    plot_df = df.tail(30).copy()

    # Ensure column names are lowercase
    plot_df.columns = [c.lower() if isinstance(c, str) else (c[0].lower() if c else '') for c in plot_df.columns]

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(plot_df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Build chart title
    if state is not None and compression_ratio is not None:
        state_str = (
            state.value.upper().replace("_", " ")
            if hasattr(state, "value")
            else str(state).upper().replace("_", " ")
        )
        title = (
            f"{ticker} - {pattern_name} [{state_str}] - "
            f"Compression: {compression_ratio * 100:.0f}%"
        )
    else:
        title = f"{ticker} - {pattern_name} (Confidence: {confidence}%)"

    # Build alines for trendlines (list of line segments)
    alines_list = []
    aline_colors = []
    for line_seg in lines:
        if len(line_seg) >= 2:
            # Convert each point to (pd.Timestamp, price)
            pts = []
            for pt in line_seg:
                ts = pd.Timestamp(pt[0])
                pts.append((ts, float(pt[1])))
            alines_list.append(pts)
            aline_colors.append("cyan")

    # Build scatter markers for key points
    addplots = []
    for kp in key_points:
        date, price, label = kp[0], kp[1], kp[2]
        lbl_lower = label.lower()
        if "trough" in lbl_lower or "bottom" in lbl_lower or "handle" in lbl_lower:
            color = "lime"
            shape = "^"
        elif "peak" in lbl_lower or "top" in lbl_lower or "head" in lbl_lower:
            color = "red"
            shape = "v"
        elif "shoulder" in lbl_lower:
            color = "orange"
            shape = "v"
        else:
            color = "cyan"
            shape = "o"

        scatter_series = pd.Series(float("nan"), index=plot_df.index)
        try:
            target_date = pd.Timestamp(date)
            if target_date in plot_df.index:
                scatter_series[target_date] = price
            else:
                diffs = abs(plot_df.index - target_date)
                nearest = plot_df.index[diffs.argmin()]
                scatter_series[nearest] = price
        except Exception:
            pass
        if scatter_series.notna().any():
            addplots.append(
                mpf.make_addplot(
                    scatter_series,
                    type="scatter",
                    markersize=120,
                    marker=shape,
                    color=color,
                    panel=0,
                )
            )

    # Add breakout levels as horizontal lines via addplot
    if breakout_level_upper is not None:
        upper_line = pd.Series(float(breakout_level_upper), index=plot_df.index)
        addplots.append(
            mpf.make_addplot(
                upper_line,
                linestyle="--",
                color="yellow",
                width=1.2,
                panel=0,
            )
        )
    if breakout_level_lower is not None:
        lower_line = pd.Series(float(breakout_level_lower), index=plot_df.index)
        addplots.append(
            mpf.make_addplot(
                lower_line,
                linestyle="--",
                color="yellow",
                width=1.2,
                panel=0,
            )
        )

    # Determine output file path
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png", prefix=f"chart_{ticker}_")
        os.close(fd)

    # Use nightclouds dark style
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        rc={"axes.titlesize": 10, "axes.titlecolor": "white"},
    )

    fig_kwargs: Dict = dict(
        type="candle",
        style=style,
        title=title,
        volume=True,
        figsize=(14, 8),
        returnfig=True,
        warn_too_much_data=500,
    )
    if addplots:
        fig_kwargs["addplot"] = addplots
    if alines_list:
        fig_kwargs["alines"] = dict(alines=alines_list, colors=aline_colors, linewidths=1.5)

    try:
        fig, axes = mpf.plot(plot_df, **fig_kwargs)
        ax = axes[0]

        # Shade the compression zone between trendlines
        if (
            breakout_level_upper is not None
            and breakout_level_lower is not None
            and breakout_level_upper > breakout_level_lower
        ):
            x_range = range(len(plot_df))
            ax.fill_between(
                x_range,
                float(breakout_level_lower),
                float(breakout_level_upper),
                alpha=0.10,
                color="cyan",
                label="Compression zone",
            )

        # Add state annotation label
        if state is not None:
            state_str = (
                state.value.upper().replace("_", " ")
                if hasattr(state, "value")
                else str(state).upper().replace("_", " ")
            )
            if state == PatternState.NEAR_BREAK or (
                isinstance(state, str) and state == "near_break"
            ):
                label_color = "red"
            else:
                label_color = "yellow"
            ax.annotate(
                state_str,
                xy=(0.02, 0.95),
                xycoords="axes fraction",
                fontsize=12,
                fontweight="bold",
                color=label_color,
                va="top",
            )

        # Show distance to breakout if available
        if distance_to_breakout is not None:
            ax.annotate(
                f"Dist to breakout: ${distance_to_breakout:.2f}",
                xy=(0.02, 0.88),
                xycoords="axes fraction",
                fontsize=9,
                color="white",
                va="top",
            )

        # Highlight candlestick pattern candles
        pattern_type = pattern_result.get("pattern_type", "structure")
        candle_indices = pattern_result.get("candle_indices", [])
        if pattern_type == "candlestick" and candle_indices:
            for idx in candle_indices:
                if 0 <= idx < len(plot_df):
                    ax.axvspan(idx - 0.4, idx + 0.4, alpha=0.15, color="yellow")

            # Add bias arrow after the last pattern candle
            bias = pattern_result.get("bias", "neutral")
            last_idx = max(candle_indices)
            if last_idx < len(plot_df):
                arrow_price = float(plot_df["close"].iloc[last_idx])
                if bias == "bullish":
                    ax.annotate(
                        "▲",
                        xy=(last_idx + 1, arrow_price),
                        fontsize=20,
                        color="#00ff88",
                        ha="center",
                        va="bottom",
                    )
                elif bias == "bearish":
                    ax.annotate(
                        "▼",
                        xy=(last_idx + 1, arrow_price),
                        fontsize=20,
                        color="#ff4444",
                        ha="center",
                        va="top",
                    )

        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.error("Failed to generate chart for %s: %s", ticker, exc)
        raise

    logger.info("Chart saved: %s", output_path)
    return output_path
