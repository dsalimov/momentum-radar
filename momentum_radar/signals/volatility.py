"""
volatility.py – Volatility expansion signal detection.

Registered signals
------------------
- ``volatility_expansion`` – compares the current day's range to the 14-day ATR
"""

import logging
from typing import Dict, Optional

import pandas as pd

from momentum_radar.config import config
from momentum_radar.signals.base import SignalResult
from momentum_radar.signals.scoring import register_signal
from momentum_radar.utils.indicators import compute_atr

logger = logging.getLogger(__name__)


@register_signal("volatility_expansion")
def volatility_expansion(
    ticker: str,
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
    **kwargs,
) -> SignalResult:
    """Detect volatility expansion relative to the 14-day ATR.

    Conditions:
    - Current day range ≥ ``ATR_RATIO_STRONG`` × 14-day ATR → +2
    - Current day range ≥ ``ATR_RATIO_MODERATE`` × 14-day ATR → +1

    The current day range is derived from today's intraday bars when available,
    otherwise from the last row of the daily DataFrame.

    Args:
        ticker: Stock symbol.
        bars: Intraday 1-min OHLCV DataFrame.
        daily: Daily OHLCV DataFrame (at least 15 days recommended).

    Returns:
        :class:`~momentum_radar.signals.base.SignalResult`
    """
    cfg = config.signals

    if daily is None or daily.empty:
        return SignalResult(triggered=False, score=0, details="No daily data for ATR")

    atr = compute_atr(daily, period=14)
    if atr is None or atr <= 0:
        return SignalResult(triggered=False, score=0, details="Could not compute ATR")

    # Current day range
    if bars is not None and not bars.empty and "high" in bars.columns and "low" in bars.columns:
        day_high = float(bars["high"].max())
        day_low = float(bars["low"].min())
    elif "high" in daily.columns and "low" in daily.columns:
        day_high = float(daily["high"].iloc[-1])
        day_low = float(daily["low"].iloc[-1])
    else:
        return SignalResult(triggered=False, score=0, details="Missing high/low data")

    day_range = day_high - day_low
    if day_range <= 0:
        return SignalResult(triggered=False, score=0, details="Day range is zero")

    ratio = day_range / atr

    if ratio >= cfg.atr_ratio_strong:
        return SignalResult(
            triggered=True,
            score=2,
            details=f"Day range {ratio:.2f}x ATR (ATR={atr:.2f}, range={day_range:.2f})",
        )
    if ratio >= cfg.atr_ratio_moderate:
        return SignalResult(
            triggered=True,
            score=1,
            details=f"Day range {ratio:.2f}x ATR (ATR={atr:.2f}, range={day_range:.2f})",
        )
    return SignalResult(
        triggered=False,
        score=0,
        details=f"Day range {ratio:.2f}x ATR – below threshold",
    )
