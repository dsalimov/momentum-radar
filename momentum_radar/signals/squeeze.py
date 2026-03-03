"""
signals/squeeze.py – Volatility squeeze detection signal module.

Registered signals
------------------
- ``volatility_squeeze`` – detects Bollinger Band compression followed by expansion

A volatility squeeze occurs when the Bollinger Bands narrow significantly
(compressed volatility), followed by a sudden expansion.  This pattern
often precedes large directional moves.
"""

import logging
from typing import Dict, Optional

import pandas as pd

from momentum_radar.signals.base import SignalResult
from momentum_radar.signals.scoring import register_signal
from momentum_radar.utils.indicators import compute_bollinger_bands

logger = logging.getLogger(__name__)

# BB bandwidth (upper−lower)/price below this → squeeze in progress
_SQUEEZE_THRESHOLD: float = 0.04
# BB bandwidth expanding by at least this fraction vs. prior window → expansion started
_EXPANSION_RATIO: float = 1.10


@register_signal("volatility_squeeze")
def volatility_squeeze(
    ticker: str,
    bars: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
    **kwargs,
) -> SignalResult:
    """Detect a Bollinger Band volatility squeeze and subsequent expansion.

    Logic:
    1. Compute BB bandwidth (upper − lower) / mid-price for the full series.
    2. Compute BB bandwidth for the series 5 bars ago (prior state).
    3. **Squeeze active**:  prior_bandwidth < ``_SQUEEZE_THRESHOLD``
    4. **Expansion begun**: current_bandwidth > prior_bandwidth × ``_EXPANSION_RATIO``
    5. Volume confirmation: recent 3-bar avg volume > prior 18-bar avg.

    Score:
    - +2 squeeze expansion confirmed with volume
    - +1 squeeze expansion without volume confirmation (or squeeze still active)

    Args:
        ticker: Stock symbol.
        bars:   Intraday 1-min OHLCV DataFrame (unused; daily preferred).
        daily:  Daily OHLCV DataFrame (at least 25 bars).

    Returns:
        :class:`~momentum_radar.signals.base.SignalResult`
    """
    if daily is None or len(daily) < 25:
        return SignalResult(triggered=False, score=0, details="Insufficient daily data for squeeze")

    closes = daily["close"]
    current_price = float(closes.iloc[-1])
    if current_price <= 0:
        return SignalResult(triggered=False, score=0, details="Invalid price data")

    # Current Bollinger Bands
    bb_current = compute_bollinger_bands(closes, period=20)
    if bb_current is None:
        return SignalResult(triggered=False, score=0, details="Could not compute Bollinger Bands")

    current_bandwidth = (bb_current["upper"] - bb_current["lower"]) / current_price

    # Prior Bollinger Bands (5 bars ago) for comparison
    prior_bandwidth = current_bandwidth
    if len(closes) >= 30:
        bb_prior = compute_bollinger_bands(closes.iloc[:-5], period=20)
        prior_price = float(closes.iloc[-6])
        if bb_prior is not None and prior_price > 0:
            prior_bandwidth = (bb_prior["upper"] - bb_prior["lower"]) / prior_price

    is_squeeze = prior_bandwidth < _SQUEEZE_THRESHOLD
    is_expanding = current_bandwidth > prior_bandwidth * _EXPANSION_RATIO

    if is_squeeze and is_expanding:
        # Volume confirmation: recent 3 bars vs. prior 18-bar average
        vol_expanding = False
        if "volume" in daily.columns and len(daily) >= 22:
            recent_vol = float(daily["volume"].iloc[-3:].mean())
            prior_vol_avg = float(daily["volume"].iloc[-21:-3].mean())
            vol_expanding = prior_vol_avg > 0 and recent_vol > prior_vol_avg * 1.10

        if vol_expanding:
            return SignalResult(
                triggered=True,
                score=2,
                details=(
                    f"Volatility squeeze expansion with volume: "
                    f"BB bandwidth {prior_bandwidth:.1%} → {current_bandwidth:.1%} "
                    f"— momentum burst expected"
                ),
            )
        return SignalResult(
            triggered=True,
            score=1,
            details=(
                f"Volatility squeeze expansion: "
                f"BB bandwidth {prior_bandwidth:.1%} → {current_bandwidth:.1%} "
                f"— watch for directional move"
            ),
        )

    if is_squeeze:
        return SignalResult(
            triggered=True,
            score=1,
            details=(
                f"Volatility squeeze active: BB bandwidth {current_bandwidth:.1%} "
                f"— range compression, pending expansion"
            ),
        )

    return SignalResult(
        triggered=False,
        score=0,
        details=f"No volatility squeeze (BB bandwidth {current_bandwidth:.1%})",
    )
