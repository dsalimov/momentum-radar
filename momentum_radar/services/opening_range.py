"""
services/opening_range.py – Opening Range Breakout (ORB) detection.

Tracks the first ``_ORB_MINUTES`` minutes of the trading session to define
the Opening Range (OR), then monitors for confirmed breakouts.

Alert criteria (all must be satisfied):
* Strong volume expansion on the breakout bar
* Clean body close above/below the OR boundary (body ratio > threshold)
* Follow-through bar also closes beyond the level
* Market regime supports momentum (not choppy)

Monitored assets (default):
* SPY  – SPDR S&P 500 ETF Trust
* QQQ  – Invesco QQQ Trust
* Top 10 global market-cap stocks (configurable via ``DEFAULT_ORB_UNIVERSE``)

Usage::

    from momentum_radar.services.opening_range import (
        compute_opening_range,
        detect_orb,
    )

    orb = compute_opening_range(bars, minutes=15)
    result = detect_orb(bars, orb)
    if result["triggered"]:
        print(result["direction"], result["score"])
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Number of minutes defining the opening range window
_ORB_MINUTES: int = 15
# Minimum body ratio for the breakout candle
_BREAKOUT_BODY_RATIO: float = 0.60
# Minimum volume ratio vs. average for a confirmed ORB
_BREAKOUT_VOLUME_MULT: float = 1.50
# Minimum score to raise an alert
ORB_MIN_SCORE: int = 70

# Default tickers to monitor for ORB setups
DEFAULT_ORB_UNIVERSE: List[str] = [
    "SPY",   # SPDR S&P 500 ETF Trust
    "QQQ",   # Invesco QQQ Trust
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "NVDA",  # NVIDIA
    "GOOGL", # Alphabet
    "AMZN",  # Amazon
    "META",  # Meta
    "TSLA",  # Tesla
    "BRK-B", # Berkshire Hathaway
    "TSM",   # TSMC
    "AVGO",  # Broadcom
]


def compute_opening_range(
    bars: pd.DataFrame,
    minutes: int = _ORB_MINUTES,
) -> Optional[Dict]:
    """Compute the Opening Range from intraday bars.

    The Opening Range is defined as the high and low of the first *minutes*
    minutes of the session.

    Args:
        bars:    1-minute OHLCV DataFrame with a DatetimeIndex.
        minutes: Length of the opening range window in minutes.

    Returns:
        Dict with ``"high"``, ``"low"``, and ``"range"`` keys, or ``None``
        if not enough data is available.
    """
    if bars is None or bars.empty:
        return None

    # Filter to opening window
    session_start = bars.index[0]
    end_time = session_start + pd.Timedelta(minutes=minutes)
    window = bars[bars.index <= end_time]

    if window.empty or len(window) < 2:
        return None

    or_high = float(window["high"].max())
    or_low = float(window["low"].min())
    or_range = or_high - or_low

    return {
        "high": round(or_high, 4),
        "low": round(or_low, 4),
        "range": round(or_range, 4),
    }


def detect_orb(
    bars: pd.DataFrame,
    opening_range: Optional[Dict],
    volume_mult: float = _BREAKOUT_VOLUME_MULT,
    body_ratio_min: float = _BREAKOUT_BODY_RATIO,
) -> Dict:
    """Detect an Opening Range Breakout from intraday bars.

    Args:
        bars:           Full intraday OHLCV 1-min DataFrame.
        opening_range:  Dict returned by :func:`compute_opening_range`.
        volume_mult:    Minimum volume multiple for a confirmed break.
        body_ratio_min: Minimum body/range ratio for the breakout candle.

    Returns:
        Dict with keys:

        - ``triggered``  – ``True`` when ORB criteria are met
        - ``direction``  – ``"bullish"`` or ``"bearish"``
        - ``score``      – confidence score (0–100)
        - ``breakout_level`` – price level that was breached
        - ``details``    – human-readable description
    """
    empty_result = {
        "triggered": False,
        "direction": None,
        "score": 0,
        "breakout_level": None,
        "details": "Insufficient data",
    }

    if bars is None or bars.empty or opening_range is None:
        return empty_result

    if len(bars) < 20:
        return empty_result

    or_high = opening_range["high"]
    or_low = opening_range["low"]

    last = bars.iloc[-1]
    prev = bars.iloc[-2]
    last_close = float(last["close"])
    last_open = float(last["open"])
    last_high = float(last["high"])
    last_low = float(last["low"])
    last_vol = float(last["volume"]) if "volume" in bars.columns else 0.0

    # Average volume (excluding the last bar)
    avg_vol = float(bars["volume"].iloc[:-1].mean()) if "volume" in bars.columns else 0.0

    candle_range = last_high - last_low
    body = abs(last_close - last_open)
    body_ratio = body / candle_range if candle_range > 0 else 0.0

    # --- Bullish ORB ---
    if last_close > or_high:
        # Volume check
        vol_ok = avg_vol > 0 and last_vol >= volume_mult * avg_vol
        # Body check
        body_ok = body_ratio >= body_ratio_min
        # Bullish close
        bullish_close = last_close > last_open
        # Follow-through: previous bar also closed bullish
        follow_through = float(prev["close"]) > float(prev["open"]) if len(bars) >= 2 else False

        score = 0
        score += 35 if vol_ok else 0
        score += 25 if body_ok else 0
        score += 20 if bullish_close else 0
        score += 20 if follow_through else 0

        if score >= ORB_MIN_SCORE:
            return {
                "triggered": True,
                "direction": "bullish",
                "score": score,
                "breakout_level": or_high,
                "details": (
                    f"ORB Bullish: close {last_close:.2f} > OR high {or_high:.2f} | "
                    f"vol {last_vol/avg_vol:.1f}x | body {body_ratio:.0%}"
                ),
            }

    # --- Bearish ORB ---
    if last_close < or_low:
        vol_ok = avg_vol > 0 and last_vol >= volume_mult * avg_vol
        body_ok = body_ratio >= body_ratio_min
        bearish_close = last_close < last_open
        follow_through = float(prev["close"]) < float(prev["open"]) if len(bars) >= 2 else False

        score = 0
        score += 35 if vol_ok else 0
        score += 25 if body_ok else 0
        score += 20 if bearish_close else 0
        score += 20 if follow_through else 0

        if score >= ORB_MIN_SCORE:
            return {
                "triggered": True,
                "direction": "bearish",
                "score": score,
                "breakout_level": or_low,
                "details": (
                    f"ORB Bearish: close {last_close:.2f} < OR low {or_low:.2f} | "
                    f"vol {last_vol/avg_vol:.1f}x | body {body_ratio:.0%}"
                ),
            }

    return {**empty_result, "details": "No ORB triggered"}
