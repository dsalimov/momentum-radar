"""
services/regime_detection.py – Market regime classification engine.

Classifies the current market environment into one of four regimes so that
the signal engine can adapt its strategy accordingly.

Regimes
-------
- ``"trending"``  – Directional move with momentum confirmation
- ``"ranging"``   – Price oscillating between clear support and resistance
- ``"expanding"`` – Volatility expanding: breakout conditions
- ``"compressing"`` – Volatility contracting: squeeze / pre-breakout setup

Usage::

    from momentum_radar.services.regime_detection import detect_regime

    regime = detect_regime(daily)
    print(regime)  # "trending" | "ranging" | "expanding" | "compressing"
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from momentum_radar.utils.indicators import compute_atr

logger = logging.getLogger(__name__)

# EMA periods for trend detection
_EMA_FAST: int = 20
_EMA_SLOW: int = 50

# ATR lookback for volatility regime
_ATR_PERIOD: int = 14
_ATR_LONG_PERIOD: int = 50

# ADX-like threshold – ratio of directional strength
_TRENDING_SLOPE_MULT: float = 0.5   # EMA slope relative to ATR
_RANGING_SLOPE_MULT: float = 0.15   # Below this = ranging

# Volatility expansion / compression thresholds
_EXPANDING_RATIO: float = 1.25      # Short ATR > long ATR × this ratio
_COMPRESSING_RATIO: float = 0.75    # Short ATR < long ATR × this ratio


def _compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Compute an exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def detect_regime(daily: Optional[pd.DataFrame]) -> str:
    """Classify the current market regime from daily OHLCV data.

    Algorithm
    ---------
    1. Compute a 20-bar EMA and 50-bar EMA.
    2. Measure the slope of the fast EMA over the last 5 bars, normalised
       by ATR.  A steep slope indicates a trending market.
    3. Compare the short ATR (14 bars) to the long ATR (50 bars) to detect
       volatility expansion or compression.
    4. If neither trending nor vol-regime, classify as ranging.

    Args:
        daily: Daily OHLCV DataFrame with at least 60 bars.

    Returns:
        Regime string: ``"trending"``, ``"ranging"``, ``"expanding"``, or
        ``"compressing"``.  Returns ``"ranging"`` when data is insufficient.
    """
    if daily is None or len(daily) < _ATR_LONG_PERIOD + 5:
        return "ranging"

    closes = daily["close"].astype(float)

    # --- ATR regime ---
    atr_short = compute_atr(daily, period=_ATR_PERIOD) or 0.0
    # Long ATR approximation: average True Range over the last _ATR_LONG_PERIOD bars
    highs = daily["high"].astype(float)
    lows = daily["low"].astype(float)
    prev_closes = closes.shift(1).fillna(closes)
    true_ranges = pd.concat(
        [highs - lows, (highs - prev_closes).abs(), (lows - prev_closes).abs()],
        axis=1,
    ).max(axis=1)
    atr_long = float(true_ranges.rolling(_ATR_LONG_PERIOD).mean().iloc[-1])

    if atr_long > 0:
        vol_ratio = atr_short / atr_long
        if vol_ratio >= _EXPANDING_RATIO:
            return "expanding"
        if vol_ratio <= _COMPRESSING_RATIO:
            return "compressing"

    # --- Trend regime ---
    ema_fast = _compute_ema(closes, _EMA_FAST)
    last_ema = float(ema_fast.iloc[-1])
    prior_ema = float(ema_fast.iloc[-6]) if len(ema_fast) >= 6 else last_ema

    ema_slope = abs(last_ema - prior_ema)  # positive slope magnitude
    normalised_slope = ema_slope / atr_short if atr_short > 0 else 0.0

    if normalised_slope >= _TRENDING_SLOPE_MULT:
        # Confirm with EMA alignment
        ema_slow = _compute_ema(closes, _EMA_SLOW)
        if last_ema > float(ema_slow.iloc[-1]):
            return "trending"
        # Downtrend also qualifies
        return "trending"

    return "ranging"


def get_regime_context(daily: Optional[pd.DataFrame]) -> dict:
    """Return a full regime context dict for downstream consumption.

    Args:
        daily: Daily OHLCV DataFrame.

    Returns:
        Dict with keys:

        - ``regime``         – one of ``"trending" | "ranging" | "expanding" | "compressing"``
        - ``atr``            – 14-bar ATR
        - ``trend_direction`` – ``"up"`` | ``"down"`` | ``"flat"``
        - ``is_trending``    – convenience bool
        - ``is_ranging``     – convenience bool
        - ``is_expanding``   – convenience bool
        - ``is_compressing`` – convenience bool
    """
    regime = detect_regime(daily)

    atr_val = compute_atr(daily) if daily is not None else None

    # Trend direction via EMA20 slope
    direction = "flat"
    if daily is not None and len(daily) >= 25:
        closes = daily["close"].astype(float)
        ema = _compute_ema(closes, _EMA_FAST)
        slope = float(ema.iloc[-1]) - float(ema.iloc[-6])
        if slope > 0:
            direction = "up"
        elif slope < 0:
            direction = "down"

    return {
        "regime": regime,
        "atr": round(atr_val, 4) if atr_val else None,
        "trend_direction": direction,
        "is_trending": regime == "trending",
        "is_ranging": regime == "ranging",
        "is_expanding": regime == "expanding",
        "is_compressing": regime == "compressing",
    }
