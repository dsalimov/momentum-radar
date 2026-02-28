"""
structure.py – Price structure break signal detection.

Registered signals
------------------
- ``structure_break`` – bullish/bearish breaks of key price levels
"""

import logging
from typing import Dict, Optional

import pandas as pd

from momentum_radar.config import config
from momentum_radar.signals.base import SignalResult
from momentum_radar.signals.scoring import register_signal
from momentum_radar.utils.indicators import compute_vwap

logger = logging.getLogger(__name__)


def _is_strong_break(bars: pd.DataFrame, level: float, direction: str) -> bool:
    """Check whether the last bar confirms a break of *level* with volume.

    A break is considered 'strong' when the closing price has crossed the level
    **and** the volume on that bar is above the 20-bar average.

    Args:
        bars: Intraday 1-min OHLCV DataFrame.
        level: Price level to check.
        direction: ``"above"`` or ``"below"``.

    Returns:
        ``True`` if a strong break is confirmed.
    """
    if bars is None or bars.empty:
        return False
    last_close = float(bars["close"].iloc[-1])
    crossed = (
        last_close > level if direction == "above" else last_close < level
    )
    if not crossed:
        return False
    avg_vol = bars["volume"].iloc[-21:-1].mean() if len(bars) > 1 else 0
    last_vol = float(bars["volume"].iloc[-1])
    return last_vol >= avg_vol


@register_signal("structure_break")
def structure_break(
    ticker: str,
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
    **kwargs,
) -> SignalResult:
    """Detect bullish or bearish price structure breaks.

    Bullish triggers:
    - Break of previous day high
    - Break of 5-min opening range high (first 30 min)

    Bearish triggers:
    - Break below previous day low
    - Loss of VWAP with volume

    Score: +2 for a confirmed break with volume, +1 for a weak break.

    Args:
        ticker: Stock symbol.
        bars: Intraday 1-min OHLCV DataFrame.
        daily: Daily OHLCV DataFrame.

    Returns:
        :class:`~momentum_radar.signals.base.SignalResult`
    """
    if bars is None or bars.empty:
        return SignalResult(triggered=False, score=0, details="No intraday data")
    if daily is None or len(daily) < 2:
        return SignalResult(triggered=False, score=0, details="Insufficient daily data")

    score = 0
    reasons: list = []

    prev_high = float(daily["high"].iloc[-2])
    prev_low = float(daily["low"].iloc[-2])
    last_close = float(bars["close"].iloc[-1])

    # ------------------------------------------------------------------
    # Previous day high break (bullish)
    # ------------------------------------------------------------------
    if last_close > prev_high:
        if _is_strong_break(bars, prev_high, "above"):
            score = max(score, 2)
            reasons.append("Break of prev-day high (strong)")
        else:
            score = max(score, 1)
            reasons.append("Break of prev-day high (weak)")

    # ------------------------------------------------------------------
    # Previous day low break (bearish)
    # ------------------------------------------------------------------
    if last_close < prev_low:
        if _is_strong_break(bars, prev_low, "below"):
            score = max(score, 2)
            reasons.append("Break below prev-day low (strong)")
        else:
            score = max(score, 1)
            reasons.append("Break below prev-day low (weak)")

    # ------------------------------------------------------------------
    # 5-min opening range breakout (first 30 bars)
    # ------------------------------------------------------------------
    if len(bars) >= 30:
        opening_range = bars.iloc[:30]
        or_high = float(opening_range["high"].max())
        or_low = float(opening_range["low"].min())
        if last_close > or_high:
            if _is_strong_break(bars, or_high, "above"):
                score = max(score, 2)
                reasons.append("Opening range breakout (strong)")
            else:
                score = max(score, 1)
                reasons.append("Opening range breakout (weak)")
        elif last_close < or_low:
            if _is_strong_break(bars, or_low, "below"):
                score = max(score, 2)
                reasons.append("Opening range breakdown (strong)")
            else:
                score = max(score, 1)
                reasons.append("Opening range breakdown (weak)")

    # ------------------------------------------------------------------
    # VWAP loss with volume (bearish)
    # ------------------------------------------------------------------
    try:
        vwap = compute_vwap(bars)
        if vwap is not None and last_close < vwap:
            if _is_strong_break(bars, vwap, "below"):
                score = max(score, 2)
                reasons.append(f"Loss of VWAP {vwap:.2f} with volume")
    except Exception as exc:
        logger.debug("VWAP calculation failed for %s: %s", ticker, exc)

    triggered = score > 0
    return SignalResult(
        triggered=triggered,
        score=score,
        details="; ".join(reasons) if reasons else "No structure break detected",
    )
