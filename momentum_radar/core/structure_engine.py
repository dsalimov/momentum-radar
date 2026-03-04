"""
core/structure_engine.py – Break of structure (BOS) and key level engine.

Detects:
* Break of structure (BOS) – close above prior swing high (bullish) or below
  prior swing low (bearish).
* Higher highs / higher lows – bullish market structure.
* Key horizontal levels – 20-day high/low, prior swing points.

Usage::

    from momentum_radar.core.structure_engine import detect_structure_break, get_key_levels

    result = detect_structure_break(daily)
    if result.confirmed:
        print(result.direction, result.broken_level)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Minimum price move (fraction of price) to qualify as a genuine BOS
_BOS_MIN_MOVE: float = 0.005   # 0.5 %
# Lookback window for swing high/low
_SWING_LOOKBACK: int = 10


@dataclass
class StructureResult:
    """Result from break-of-structure detection.

    Attributes:
        direction:    ``"bullish"`` / ``"bearish"`` / ``"none"``.
        broken_level: Price level that was broken.
        break_pct:    Percentage move beyond the broken level.
        confirmed:    True when the close confirms the break (not a wick).
    """

    direction: str = "none"
    broken_level: float = 0.0
    break_pct: float = 0.0
    confirmed: bool = False


def detect_structure_break(
    daily: Optional[pd.DataFrame],
    lookback: int = _SWING_LOOKBACK,
) -> StructureResult:
    """Detect a break of structure in daily price data.

    A **bullish BOS** occurs when the last close exceeds the prior *n*-bar swing high.
    A **bearish BOS** occurs when the last close falls below the prior *n*-bar swing low.

    Args:
        daily:    Daily OHLCV DataFrame (at least ``lookback + 2`` bars required).
        lookback: Number of prior bars to compute the swing high/low from.

    Returns:
        :class:`StructureResult`.
    """
    result = StructureResult()
    if daily is None or len(daily) < lookback + 2:
        return result
    if "close" not in daily.columns or "high" not in daily.columns:
        return result

    closes = daily["close"]
    highs  = daily["high"]
    lows   = daily["low"]

    last_close  = float(closes.iloc[-1])
    prior_high  = float(highs.iloc[-(lookback + 1):-1].max())
    prior_low   = float(lows.iloc[-(lookback + 1):-1].min())

    if prior_high <= 0 or prior_low <= 0:
        return result

    # Bullish BOS
    if last_close > prior_high:
        pct = (last_close - prior_high) / prior_high
        if pct >= _BOS_MIN_MOVE:
            result.direction = "bullish"
            result.broken_level = round(prior_high, 4)
            result.break_pct = round(pct * 100, 2)
            result.confirmed = True
            return result

    # Bearish BOS
    if last_close < prior_low:
        pct = (prior_low - last_close) / prior_low
        if pct >= _BOS_MIN_MOVE:
            result.direction = "bearish"
            result.broken_level = round(prior_low, 4)
            result.break_pct = round(pct * 100, 2)
            result.confirmed = True
            return result

    return result


def get_key_levels(
    daily: Optional[pd.DataFrame],
    lookback: int = 20,
) -> Dict[str, float]:
    """Return key price levels for chart annotation and risk calculation.

    Args:
        daily:    Daily OHLCV DataFrame.
        lookback: Lookback window for high/low detection.

    Returns:
        Dict with keys ``"resistance"``, ``"support"``, ``"20d_high"``, ``"20d_low"``.
    """
    if daily is None or daily.empty:
        return {}

    n = min(lookback, len(daily) - 1)
    prior = daily.iloc[-n - 1:-1] if n > 0 else daily.iloc[:0]

    return {
        "resistance": round(float(prior["high"].max()), 4) if len(prior) > 0 else 0.0,
        "support":    round(float(prior["low"].min()), 4) if len(prior) > 0 else 0.0,
        "20d_high":   round(float(daily["high"].iloc[-n:].max()), 4),
        "20d_low":    round(float(daily["low"].iloc[-n:].min()), 4),
    }


def has_bullish_structure(
    daily: Optional[pd.DataFrame],
    lookback: int = 6,
) -> bool:
    """Return True if recent bars show higher highs and higher lows.

    Args:
        daily:    Daily OHLCV DataFrame.
        lookback: Number of bars to inspect.

    Returns:
        True if bullish market structure is present.
    """
    if daily is None or len(daily) < lookback + 1:
        return False
    highs = list(daily["high"].iloc[-lookback:])
    lows  = list(daily["low"].iloc[-lookback:])
    hh = len(highs) >= 3 and highs[-1] > highs[-2] > highs[-3]
    hl = len(lows) >= 3 and lows[-1] > lows[-2] > lows[-3]
    return hh and hl
