"""
signals/vwap_signal.py – VWAP proximity signal module.

Registers:
- vwap_proximity: Bullish when price is above VWAP with recent bounce
"""

import logging
from typing import Dict, Optional

import pandas as pd

from momentum_radar.signals.scoring import register_signal
from momentum_radar.signals.base import SignalResult
from momentum_radar.utils.indicators import compute_vwap

logger = logging.getLogger(__name__)


@register_signal("vwap_proximity")
def vwap_proximity(
    ticker: str,
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
    fundamentals: Optional[Dict],
    options: Optional[Dict],
    **kwargs,
) -> SignalResult:
    """Fire when intraday price is above VWAP and within a close proximity."""
    if bars is None or bars.empty:
        return SignalResult(triggered=False, score=0, details="No intraday bars")

    required = {"high", "low", "close", "volume"}
    if not required.issubset(bars.columns):
        return SignalResult(triggered=False, score=0, details="Missing OHLCV columns")

    vwap = compute_vwap(bars)
    if vwap is None or vwap <= 0:
        return SignalResult(triggered=False, score=0, details="VWAP computation failed")

    last_close = float(bars["close"].iloc[-1])
    pct_from_vwap = (last_close - vwap) / vwap

    # Price is above VWAP by 0–2% (just cleared VWAP – bullish momentum)
    if 0 < pct_from_vwap <= 0.02:
        return SignalResult(
            triggered=True,
            score=1,
            details=f"Price ${last_close:.2f} just above VWAP ${vwap:.2f} (+{pct_from_vwap:.1%})",
        )

    # Price is significantly above VWAP (>2%) – strong momentum
    if pct_from_vwap > 0.02:
        return SignalResult(
            triggered=True,
            score=2,
            details=f"Price ${last_close:.2f} well above VWAP ${vwap:.2f} (+{pct_from_vwap:.1%})",
        )

    return SignalResult(triggered=False, score=0, details=f"Price below VWAP (${last_close:.2f} vs ${vwap:.2f})")
