"""
heatmap.py – Market sector heatmap generator.

Generates a colour-coded PNG heatmap showing S&P 500 sector performance
for the current trading day.  Data is fetched via yfinance using sector
ETFs (XLK, XLF, XLV, etc.) as proxies.

Usage::

    from momentum_radar.utils.heatmap import generate_market_heatmap
    path, summary = generate_market_heatmap()
"""

import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sector ETF definitions
# ---------------------------------------------------------------------------

_SECTOR_ETFS: Dict[str, str] = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Consumer Disc.": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Comm. Services": "XLC",
    "S&P 500": "SPY",
    "Nasdaq 100": "QQQ",
    "Russell 2000": "IWM",
}

# Representative tickers to display per sector (for the detailed view)
_SECTOR_STOCKS: Dict[str, List[str]] = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AMD", "TSLA"],
    "Financials": ["JPM", "BAC", "GS", "MS", "V"],
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK"],
    "Energy": ["XOM", "CVX", "SLB", "OXY", "COP"],
    "Consumer Disc.": ["AMZN", "HD", "NKE", "MCD", "SBUX"],
}


def _fetch_sector_performance() -> List[Tuple[str, str, float]]:
    """Fetch intraday performance for each sector ETF.

    Returns:
        List of (sector_name, ticker, pct_change) tuples sorted by pct_change.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance is required for heatmap generation.")
        return []

    results: List[Tuple[str, str, float]] = []
    tickers_list = list(_SECTOR_ETFS.values())

    try:
        data = yf.download(
            " ".join(tickers_list),
            period="2d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )
    except Exception as exc:
        logger.error("Failed to download sector data: %s", exc)
        return []

    for sector, etf in _SECTOR_ETFS.items():
        try:
            if len(tickers_list) > 1:
                if etf not in data.columns.get_level_values(0):
                    continue
                closes = data[etf]["Close"].dropna()
            else:
                closes = data["Close"].dropna()

            if len(closes) < 2:
                continue

            prev_close = float(closes.iloc[-2])
            last_close = float(closes.iloc[-1])
            pct = ((last_close - prev_close) / prev_close) * 100 if prev_close != 0 else 0.0
            results.append((sector, etf, round(pct, 2)))
        except Exception as exc:
            logger.debug("Skipping %s (%s): %s", sector, etf, exc)

    results.sort(key=lambda x: x[2], reverse=True)
    return results


def _pct_to_color(pct: float) -> Tuple[float, float, float]:
    """Map a percentage change to an RGB colour (green/red gradient)."""
    if pct >= 0:
        # Green: scale from light green (0%) to dark green (3%+)
        intensity = min(pct / 3.0, 1.0)
        return (0.1 + 0.3 * (1 - intensity), 0.5 + 0.4 * intensity, 0.1 + 0.2 * (1 - intensity))
    else:
        # Red: scale from light red (0%) to dark red (-3%+)
        intensity = min(abs(pct) / 3.0, 1.0)
        return (0.5 + 0.4 * intensity, 0.1 + 0.3 * (1 - intensity), 0.1 + 0.2 * (1 - intensity))


def generate_market_heatmap(output_path: Optional[str] = None) -> Tuple[Optional[str], str]:
    """Generate a market sector heatmap PNG image.

    Args:
        output_path: Optional path to save the PNG.  If ``None``, a temp file
                     is created in the system temp directory.

    Returns:
        Tuple of (``file_path``, ``text_summary``).
        ``file_path`` is ``None`` if matplotlib is unavailable.
        ``text_summary`` is always a formatted string.
    """
    performance = _fetch_sector_performance()

    if not performance:
        return None, "⚠️ Unable to fetch sector data for heatmap."

    # Build text summary first (works even without matplotlib)
    lines = ["📊 Market Sector Heatmap", ""]
    for sector, etf, pct in performance:
        arrow = "🟢" if pct >= 0 else "🔴"
        sign = "+" if pct >= 0 else ""
        lines.append(f"{arrow} {sector:16s} ({etf})  {sign}{pct:.2f}%")

    text_summary = "\n".join(lines)

    # Try to generate chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not available; returning text summary only.")
        return None, text_summary

    try:
        n = len(performance)
        cols = 4
        rows = (n + cols - 1) // cols

        fig, ax = plt.subplots(figsize=(14, max(4, rows * 2 + 1)))
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.axis("off")

        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        for i, (sector, etf, pct) in enumerate(performance):
            row = rows - 1 - (i // cols)
            col = i % cols
            color = _pct_to_color(pct)
            sign = "+" if pct >= 0 else ""
            rect = mpatches.FancyBboxPatch(
                (col + 0.05, row + 0.05),
                0.88,
                0.88,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor="#1a1a2e",
                linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.text(
                col + 0.49,
                row + 0.60,
                sector,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
            )
            ax.text(
                col + 0.49,
                row + 0.35,
                f"{sign}{pct:.2f}%",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
            )
            ax.text(
                col + 0.49,
                row + 0.18,
                etf,
                ha="center",
                va="center",
                fontsize=7,
                color="#cccccc",
            )

        plt.title(
            "Market Sector Heatmap",
            color="white",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )
        plt.tight_layout(pad=0.5)

        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".png", prefix="heatmap_")
            os.close(fd)

        plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return output_path, text_summary

    except Exception as exc:
        logger.error("Heatmap chart generation failed: %s", exc)
        return None, text_summary
