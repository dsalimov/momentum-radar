"""
core/regime_engine.py – Market regime facade for the multi-strategy architecture.

Delegates to :mod:`momentum_radar.services.regime_detection` and adds
a human-readable HTF bias label used by strategy engines.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from momentum_radar.services.regime_detection import detect_regime

logger = logging.getLogger(__name__)

# Mapping from internal regime string to display label
_REGIME_DISPLAY = {
    "trending":    "Trending",
    "ranging":     "Ranging",
    "expanding":   "Expanding",
    "compressing": "Compressing",
}


def get_regime(daily: Optional[pd.DataFrame]) -> str:
    """Return the current market regime.

    Args:
        daily: Daily OHLCV DataFrame.

    Returns:
        One of ``"trending"`` / ``"ranging"`` / ``"expanding"`` / ``"compressing"``.
    """
    return detect_regime(daily)


def get_regime_display(daily: Optional[pd.DataFrame]) -> str:
    """Return a human-readable regime label for alert display.

    Args:
        daily: Daily OHLCV DataFrame.

    Returns:
        Capitalised regime string (e.g. ``"Trending"``).
    """
    regime = detect_regime(daily)
    return _REGIME_DISPLAY.get(regime, regime.capitalize())


def get_htf_bias(daily: Optional[pd.DataFrame]) -> str:
    """Return the higher-timeframe directional bias label.

    The bias is derived from EMA alignment on daily bars:
    - price > EMA21 > EMA50 → ``"Bullish"``
    - price < EMA21 < EMA50 → ``"Bearish"``
    - Otherwise              → ``"Neutral"``

    Args:
        daily: Daily OHLCV DataFrame.

    Returns:
        One of ``"Bullish"`` / ``"Bearish"`` / ``"Neutral"``.
    """
    if daily is None or daily.empty or "close" not in daily.columns:
        return "Neutral"
    closes = daily["close"]
    if len(closes) < 50:
        return "Neutral"
    last = float(closes.iloc[-1])
    ema21 = float(closes.ewm(span=21, adjust=False).mean().iloc[-1])
    ema50 = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
    if last > ema21 and ema21 > ema50:
        return "Bullish"
    if last < ema21 and ema21 < ema50:
        return "Bearish"
    return "Neutral"
