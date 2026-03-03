"""
signals/trend.py – EMA trend and RSI+MACD confirmation signal modules.

Registers:
- ema_trend: Bullish when price > EMA9 > EMA21 > EMA200
- rsi_macd: RSI in 40–70 range with positive MACD histogram
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from momentum_radar.signals.scoring import register_signal
from momentum_radar.signals.base import SignalResult
from momentum_radar.utils.indicators import compute_ema, compute_rsi, compute_macd

logger = logging.getLogger(__name__)


@register_signal("ema_trend")
def ema_trend(
    ticker: str,
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
    fundamentals: Optional[Dict],
    options: Optional[Dict],
    **kwargs,
) -> SignalResult:
    """Fire when price is above EMA9, EMA21, and EMA200 (bullish alignment)."""
    if daily is None or daily.empty or "close" not in daily.columns:
        return SignalResult(triggered=False, score=0, details="No daily data")

    closes = daily["close"]
    if len(closes) < 200:
        return SignalResult(triggered=False, score=0, details="Insufficient history for EMA200")

    ema9 = compute_ema(closes, 9)
    ema21 = compute_ema(closes, 21)
    ema200 = compute_ema(closes, 200)

    if ema9 is None or ema21 is None or ema200 is None:
        return SignalResult(triggered=False, score=0, details="EMA computation failed")

    last_close = float(closes.iloc[-1])
    e9 = float(ema9.iloc[-1])
    e21 = float(ema21.iloc[-1])
    e200 = float(ema200.iloc[-1])

    # Full bullish alignment: price > EMA9 > EMA21 > EMA200
    if last_close > e9 > e21 > e200:
        return SignalResult(
            triggered=True,
            score=2,
            details=f"Bullish EMA alignment: price ${last_close:.2f} > EMA9 {e9:.2f} > EMA21 {e21:.2f} > EMA200 {e200:.2f}",
        )

    # Partial bullish: price > EMA21 > EMA200
    if last_close > e21 > e200:
        return SignalResult(
            triggered=True,
            score=1,
            details=f"Partial bullish alignment: price > EMA21 {e21:.2f} > EMA200 {e200:.2f}",
        )

    return SignalResult(triggered=False, score=0, details="No EMA bullish alignment")


@register_signal("rsi_macd")
def rsi_macd(
    ticker: str,
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
    fundamentals: Optional[Dict],
    options: Optional[Dict],
    **kwargs,
) -> SignalResult:
    """Fire when RSI is in a bullish range (40–70) with positive MACD histogram."""
    if daily is None or daily.empty or "close" not in daily.columns:
        return SignalResult(triggered=False, score=0, details="No daily data")

    closes = daily["close"]

    rsi = compute_rsi(closes, period=14)
    macd_data = compute_macd(closes)

    if rsi is None or macd_data is None:
        return SignalResult(triggered=False, score=0, details="RSI/MACD computation failed")

    histogram = macd_data["histogram"]
    macd_val = macd_data["macd"]
    signal_val = macd_data["signal"]

    # Bullish: RSI in 40-70 (momentum zone, not overbought) + positive MACD histogram
    if 40 <= rsi <= 70 and histogram > 0:
        return SignalResult(
            triggered=True,
            score=2,
            details=f"RSI {rsi:.1f} in momentum zone, MACD histogram +{histogram:.4f} (bullish)",
        )

    # RSI oversold bounce potential
    if rsi < 35 and histogram > 0:
        return SignalResult(
            triggered=True,
            score=1,
            details=f"RSI {rsi:.1f} oversold bounce + MACD turning positive",
        )

    return SignalResult(triggered=False, score=0, details=f"RSI {rsi:.1f}, MACD hist {histogram:.4f} – no signal")
