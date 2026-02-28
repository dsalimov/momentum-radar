"""
charts.py – Candlestick chart generation with pattern annotations.

Generates publication-quality PNG charts using *mplfinance* with the detected
pattern key points overlaid as markers and lines.
"""

import logging
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
        pattern_result: Result dict from :func:`~momentum_radar.patterns.detector.detect_pattern`.
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
        # Set non-interactive backend before importing pyplot.
        # This is a no-op if pyplot has already been imported elsewhere.
        try:
            matplotlib.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "mplfinance and matplotlib are required for chart generation."
        ) from exc

    pattern_name = pattern_result.get("pattern", "Pattern")
    confidence = pattern_result.get("confidence", 0)
    key_points: List[Tuple] = pattern_result.get("key_points", [])

    # Use last 120 bars at most for readability
    plot_df = df.tail(120).copy()

    # Ensure column names are lowercase
    plot_df.columns = [c.lower() for c in plot_df.columns]

    # mplfinance requires specific column names
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(plot_df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Build scatter markers for key points
    # Map dates from key_points back to plot_df index positions
    marker_dates = []
    marker_prices = []
    marker_colors = []
    marker_shapes = []
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
        marker_dates.append(date)
        marker_prices.append(price)
        marker_colors.append(color)
        marker_shapes.append(shape)

    # Build addplot scatter series for each marker group by shape/color
    addplots = []
    for shape in set(marker_shapes):
        idxs = [i for i, s in enumerate(marker_shapes) if s == shape]
        for i in idxs:
            target_date = marker_dates[i]
            price = marker_prices[i]
            color = marker_colors[i]
            # Build a series aligned to plot_df with NaN everywhere except the marker
            scatter_series = pd.Series(float("nan"), index=plot_df.index)
            # Find nearest date in plot_df
            try:
                if target_date in plot_df.index:
                    scatter_series[target_date] = price
                else:
                    # Find closest date
                    diffs = abs(plot_df.index - pd.Timestamp(target_date))
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

    # Chart title
    title = f"{ticker} — {pattern_name} (Confidence: {confidence}%)"

    # Determine output file path
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png", prefix=f"chart_{ticker}_")
        import os
        os.close(fd)

    # Plot
    style = mpf.make_mpf_style(base_mpf_style="charles", rc={"axes.titlesize": 11})
    fig_kwargs = dict(
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

    try:
        fig, axes = mpf.plot(plot_df, **fig_kwargs)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.error("Failed to generate chart for %s: %s", ticker, exc)
        raise

    logger.info("Chart saved: %s", output_path)
    return output_path
