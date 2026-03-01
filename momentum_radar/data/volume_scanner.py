"""
volume_scanner.py – Volume spike scanner for unusual stock volume vs 30-day average.

Provides :func:`scan_volume_spikes` which iterates over a list of tickers,
computes each stock's relative volume (RVOL = current volume / 30-day avg),
and returns the top movers sorted by RVOL.

Designed for the ``/volspike`` Telegram command.
"""

import logging
from typing import List, Optional

import pandas as pd

from momentum_radar.data.data_fetcher import BaseDataFetcher

logger = logging.getLogger(__name__)

# Minimum RVOL to be included in the spike list
_MIN_RVOL = 1.5


def scan_volume_spikes(
    tickers: List[str],
    fetcher: BaseDataFetcher,
    top_n: int = 15,
    min_rvol: float = _MIN_RVOL,
) -> List[dict]:
    """Scan *tickers* for unusual volume vs. 30-day average.

    For each ticker we compute::

        rvol = today_volume / avg_30d_volume

    Only tickers with ``rvol >= min_rvol`` are returned, sorted descending.

    Args:
        tickers:   List of stock ticker symbols to scan.
        fetcher:   Data fetcher used to retrieve daily bars.
        top_n:     Maximum number of results to return.
        min_rvol:  Minimum relative-volume ratio to qualify as a spike.

    Returns:
        List of dicts with keys:
        ``ticker``, ``rvol``, ``today_volume``, ``avg_30d_volume``,
        ``last_close``, ``pct_change``.
    """
    results: List[dict] = []

    for ticker in tickers:
        try:
            daily = fetcher.get_daily_bars(ticker, period="35d")
            if daily is None or daily.empty or "volume" not in daily.columns:
                continue
            if len(daily) < 5:
                continue

            # 30-day average excluding today
            hist_vols = daily["volume"].iloc[-31:-1]
            avg_30d = float(hist_vols.mean()) if len(hist_vols) >= 5 else 0.0
            if avg_30d <= 0:
                continue

            today_vol = float(daily["volume"].iloc[-1])
            rvol = today_vol / avg_30d

            if rvol < min_rvol:
                continue

            last_close = float(daily["close"].iloc[-1])
            prev_close = float(daily["close"].iloc[-2]) if len(daily) >= 2 else last_close
            pct_change = ((last_close - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

            results.append(
                {
                    "ticker": ticker,
                    "rvol": round(rvol, 2),
                    "today_volume": int(today_vol),
                    "avg_30d_volume": int(avg_30d),
                    "last_close": round(last_close, 2),
                    "pct_change": round(pct_change, 2),
                }
            )
        except Exception as exc:
            logger.debug("Volume spike scan skipped %s: %s", ticker, exc)
            continue

    results.sort(key=lambda r: r["rvol"], reverse=True)
    return results[:top_n]


def generate_volume_spike_chart(
    spikes: List[dict],
    output_path: Optional[str] = None,
) -> str:
    """Generate a horizontal bar chart showing the top volume spikes.

    Args:
        spikes:      List of spike dicts from :func:`scan_volume_spikes`.
        output_path: Optional file path for the PNG.  A temp file is
                     created when ``None``.

    Returns:
        Absolute path to the generated PNG file.
    """
    try:
        import matplotlib
        try:
            matplotlib.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        import tempfile
    except ImportError as exc:
        raise ImportError("matplotlib is required for volume spike chart.") from exc

    if not spikes:
        raise ValueError("No spike data to chart.")

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png", prefix="volspike_")
        os.close(fd)

    tickers = [s["ticker"] for s in spikes]
    rvols = [s["rvol"] for s in spikes]
    pct_changes = [s["pct_change"] for s in spikes]

    # Color bars green if price up, red if down
    bar_colors = ["#00c853" if p >= 0 else "#ff1744" for p in pct_changes]

    fig, ax = plt.subplots(figsize=(12, max(5, len(tickers) * 0.5 + 2)))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    y_pos = np.arange(len(tickers))
    bars = ax.barh(y_pos, rvols, color=bar_colors, alpha=0.85, height=0.6)

    # Reference line at 1.0x (normal volume)
    ax.axvline(x=1.0, color="#888888", linestyle="--", linewidth=1, label="1x Avg Vol")
    # Strong spike reference at 2.0x
    ax.axvline(x=2.0, color="yellow", linestyle=":", linewidth=1, label="2x Avg Vol")

    # Add value labels inside bars
    for bar, spike in zip(bars, spikes):
        width = bar.get_width()
        pct = spike["pct_change"]
        vol_str = f"{spike['today_volume'] / 1e6:.1f}M" if spike['today_volume'] >= 1_000_000 else f"{spike['today_volume'] / 1e3:.0f}K"
        label = f"  {width:.1f}x  |  {pct:+.1f}%  |  ${spike['last_close']:.2f}  |  Vol {vol_str}"
        ax.text(
            width + 0.05, bar.get_y() + bar.get_height() / 2,
            label,
            va="center", ha="left", color="white", fontsize=8,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tickers, color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="white")
    ax.set_xlabel("Relative Volume (RVOL vs 30-Day Avg)", color="white", fontsize=10)
    ax.set_title(
        "Volume Spike Scanner — Unusual Volume vs 30-Day Average",
        color="white", fontsize=13, fontweight="bold",
    )
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    # Subtitle / timestamp
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M ET")
    fig.text(
        0.99, 0.01, f"Generated {ts}",
        ha="right", va="bottom", color="#888888", fontsize=7,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0e1117")
    plt.close(fig)

    logger.info("Volume spike chart saved: %s", output_path)
    return output_path
