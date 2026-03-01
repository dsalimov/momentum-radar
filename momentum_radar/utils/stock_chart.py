"""
stock_chart.py – Professional multi-panel stock analysis chart generator.

Produces a dark-themed publication-quality PNG with three panels:

1. **Price panel** – candlestick bars with 20/50 EMA overlays and a key
   metrics annotation box (RVOL, ATR, % change, signal scores)
2. **Volume panel** – volume bars colour-coded by price direction, 30-day
   average line
3. **RVOL gauge** – relative volume bar per day

Used by the ``/analyze TICKER`` Telegram bot command.
"""

import logging
import os
import tempfile
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def generate_analysis_chart(
    ticker: str,
    daily: pd.DataFrame,
    output_path: Optional[str] = None,
    *,
    rvol: Optional[float] = None,
    score: Optional[int] = None,
    signals: Optional[list] = None,
    short_interest: Optional[float] = None,
    float_shares: Optional[float] = None,
) -> str:
    """Generate a professional multi-panel stock analysis chart.

    Args:
        ticker:         Stock ticker symbol.
        daily:          Daily OHLCV DataFrame (at least 60 bars recommended).
        output_path:    Optional file path for PNG.  Temp file used if ``None``.
        rvol:           Pre-computed relative volume (optional).
        score:          Momentum score 0-10 (optional).
        signals:        List of triggered signal description strings (optional).
        short_interest: Short-percent-of-float as decimal e.g. 0.12 (optional).
        float_shares:   Float share count (optional).

    Returns:
        Absolute path to the generated PNG file.

    Raises:
        ValueError: If *daily* is ``None`` or empty.
        ImportError: If *matplotlib* is not installed.
    """
    try:
        import matplotlib
        try:
            matplotlib.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
    except ImportError as exc:
        raise ImportError("matplotlib is required for stock chart generation.") from exc

    if daily is None or daily.empty:
        raise ValueError("daily DataFrame is required for chart generation.")

    # Normalise column names
    df = daily.copy()
    df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]

    # Use last 60 bars for the chart
    df = df.tail(60)

    closes = df["close"].values.astype(float)
    highs = df["high"].values.astype(float) if "high" in df.columns else closes
    lows = df["low"].values.astype(float) if "low" in df.columns else closes
    opens = df["open"].values.astype(float) if "open" in df.columns else closes
    volumes = df["volume"].values.astype(float) if "volume" in df.columns else None

    n = len(closes)
    x = range(n)

    # --- EMA computation ---
    def _ema(values, period):
        s = pd.Series(values)
        return s.ewm(span=period, adjust=False).mean().values

    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)

    # --- Avg volume ---
    avg_vol_30d: Optional[float] = None
    if volumes is not None and len(volumes) > 5:
        lookback = volumes[:-1] if len(volumes) >= 31 else volumes[:-1]
        avg_vol_30d = float(lookback[-30:].mean()) if len(lookback) >= 1 else None

    # --- RVOL ---
    if rvol is None and avg_vol_30d and avg_vol_30d > 0 and volumes is not None:
        rvol = float(volumes[-1]) / avg_vol_30d

    # --- % change (last close vs prev) ---
    pct_change: Optional[float] = None
    if len(closes) >= 2 and closes[-2] > 0:
        pct_change = (closes[-1] - closes[-2]) / closes[-2] * 100

    # --- ATR (14-day) ---
    atr: Optional[float] = None
    if n >= 15 and "high" in df.columns and "low" in df.columns:
        tr = pd.Series(
            [
                max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
                for i in range(1, n)
            ]
        )
        atr = float(tr.rolling(14).mean().iloc[-1])

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png", prefix=f"analysis_{ticker}_")
        os.close(fd)

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#0e1117")

    gs = gridspec.GridSpec(
        3, 1,
        figure=fig,
        height_ratios=[3, 1, 0.6],
        hspace=0.08,
    )

    ax_price = fig.add_subplot(gs[0])
    ax_vol = fig.add_subplot(gs[1], sharex=ax_price)
    ax_rvol = fig.add_subplot(gs[2], sharex=ax_price)

    for ax in (ax_price, ax_vol, ax_rvol):
        ax.set_facecolor("#0d1117")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.yaxis.label.set_color("#aaaaaa")

    # -------------------------------------------------------------------------
    # Panel 1 – Price (candlesticks approximated as OHLC lines + fills)
    # -------------------------------------------------------------------------
    for i in x:
        color = "#00c853" if closes[i] >= opens[i] else "#ff1744"
        ax_price.plot([i, i], [lows[i], highs[i]], color=color, linewidth=0.8, alpha=0.8)
        ax_price.bar(i, abs(closes[i] - opens[i]), bottom=min(opens[i], closes[i]),
                     color=color, alpha=0.85, width=0.7)

    ax_price.plot(list(x), ema20, color="#ffd740", linewidth=1.2, label="EMA 20", alpha=0.9)
    ax_price.plot(list(x), ema50, color="#40c4ff", linewidth=1.2, label="EMA 50", alpha=0.9)

    ax_price.set_ylabel("Price ($)", color="#aaaaaa")
    ax_price.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8, loc="upper left")

    # Title
    pct_str = f"{pct_change:+.2f}%" if pct_change is not None else ""
    score_str = f"  |  Score: {score}/10" if score is not None else ""
    ax_price.set_title(
        f"{ticker}  ${closes[-1]:.2f}  {pct_str}{score_str}",
        color="white", fontsize=14, fontweight="bold", pad=8,
    )
    plt.setp(ax_price.get_xticklabels(), visible=False)

    # -------------------------------------------------------------------------
    # Panel 2 – Volume
    # -------------------------------------------------------------------------
    if volumes is not None:
        vol_colors = [
            "#00c853" if closes[i] >= opens[i] else "#ff1744" for i in x
        ]
        ax_vol.bar(list(x), volumes, color=vol_colors, alpha=0.7, width=0.8)
        if avg_vol_30d:
            ax_vol.axhline(avg_vol_30d, color="yellow", linestyle="--", linewidth=1,
                           label=f"30d Avg: {int(avg_vol_30d):,}")
        ax_vol.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7, loc="upper left")
        # Format y-axis in millions / thousands
        ax_vol.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K")
        )
    ax_vol.set_ylabel("Volume", color="#aaaaaa")
    plt.setp(ax_vol.get_xticklabels(), visible=False)

    # -------------------------------------------------------------------------
    # Panel 3 – RVOL gauge
    # -------------------------------------------------------------------------
    if rvol is not None and volumes is not None and avg_vol_30d:
        rvol_per_bar = [
            volumes[i] / avg_vol_30d if avg_vol_30d > 0 else 0
            for i in range(len(volumes))
        ]
        rvol_colors = [
            "#ff9100" if v >= 2.0 else ("#ffd740" if v >= 1.5 else "#607d8b")
            for v in rvol_per_bar
        ]
        ax_rvol.bar(list(x), rvol_per_bar, color=rvol_colors, alpha=0.8, width=0.8)
        ax_rvol.axhline(1.0, color="#888888", linestyle="--", linewidth=0.8)
        ax_rvol.axhline(2.0, color="#ff9100", linestyle=":", linewidth=0.8)
    ax_rvol.set_ylabel("RVOL", color="#aaaaaa")

    # X-axis date labels on bottom panel
    date_step = max(1, n // 10)
    if isinstance(df.index, pd.DatetimeIndex):
        tick_positions = list(range(0, n, date_step))
        tick_labels = [df.index[i].strftime("%b %d") for i in tick_positions]
        ax_rvol.set_xticks(tick_positions)
        ax_rvol.set_xticklabels(tick_labels, rotation=30, ha="right", color="#aaaaaa", fontsize=7)

    # -------------------------------------------------------------------------
    # Key metrics annotation on price panel
    # -------------------------------------------------------------------------
    metrics_lines = []
    if rvol is not None:
        rvol_label = "** UNUSUAL **" if rvol >= 2.0 else ("^ ELEVATED" if rvol >= 1.5 else "NORMAL")
        metrics_lines.append(f"RVOL: {rvol:.2f}x  [{rvol_label}]")
    if atr is not None:
        metrics_lines.append(f"ATR(14): ${atr:.2f}")
    if pct_change is not None:
        metrics_lines.append(f"Day Chg: {pct_change:+.2f}%")
    if short_interest is not None:
        metrics_lines.append(f"Short %: {short_interest*100:.1f}%")
    if float_shares is not None:
        f_str = f"{float_shares/1e6:.0f}M" if float_shares >= 1e6 else f"{float_shares/1e3:.0f}K"
        metrics_lines.append(f"Float: {f_str}")
    if signals:
        metrics_lines.append("Signals:")
        for sig in signals[:4]:
            metrics_lines.append(f"  ✓ {sig}")

    if metrics_lines:
        ax_price.text(
            0.99, 0.98, "\n".join(metrics_lines),
            transform=ax_price.transAxes,
            color="white", fontsize=8,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#1a1a2e",
                alpha=0.88,
                edgecolor="#444444",
            ),
        )

    # Timestamp footer
    ts = datetime.now().strftime("%Y-%m-%d %H:%M ET")
    fig.text(0.99, 0.005, f"Generated {ts}", ha="right", va="bottom",
             color="#555555", fontsize=7)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0e1117")
    plt.close(fig)

    logger.info("Analysis chart saved: %s", output_path)
    return output_path
