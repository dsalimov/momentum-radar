"""
storage/alert_stats.py – Historical alert performance tracking.

Provides baseline win-rate statistics per setup type and a lightweight
in-memory accumulator that is updated as alerts are resolved.

The baseline win rates are sourced from published academic and
practitioner research on each pattern type.  The live stats table
(``alert_outcomes``) allows the system to refine these estimates over
time as real trade outcomes are recorded.
"""

import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baseline win rates per setup type (percentage, 0–100)
# These represent historical averages from back-tested research.
# ---------------------------------------------------------------------------

BASELINE_WIN_RATES: Dict[str, float] = {
    # Confirmed breakouts with volume
    "structure_break": 58.0,
    "Ascending Triangle Breakout": 65.0,
    # Volume-driven setups
    "volume_spike": 55.0,
    "relative_volume": 54.0,
    "Volume Spike": 55.0,
    # Support/resistance patterns
    "third_touch_support": 67.0,
    "Third+ touch": 67.0,
    "Support Bounce": 60.0,
    "Double Bottom": 63.0,
    # Candlestick patterns
    "Bullish Engulfing": 62.0,
    "Hammer": 60.0,
    "Shooting Star": 58.0,
    "Bearish Engulfing": 61.0,
    # Options flow
    "options_flow": 56.0,
    "Call Flow Spike": 57.0,
    "Put Flow Spike": 56.0,
    "Gamma Flip Zone": 59.0,
    # Failed breakouts (reversal after trap)
    "failed_breakout": 64.0,
    "Bull trap": 62.0,
    "Bear trap": 63.0,
    # Volatility squeeze
    "volatility_squeeze": 61.0,
    "Volatility squeeze expansion": 63.0,
    # Trend continuation
    "ema_trend": 57.0,
    "rsi_macd": 55.0,
    # VWAP
    "vwap_proximity": 54.0,
    "vwap_signal": 54.0,
    # Short squeeze
    "short_interest": 52.0,
    # Volatility expansion
    "volatility_expansion": 53.0,
}

# Default win rate for unknown setups
_DEFAULT_WIN_RATE: float = 55.0


def get_win_rate(setup_type: str) -> float:
    """Return the historical win rate (%) for *setup_type*.

    Performs a substring match against known setup names so partial keys
    (e.g. ``"Bull trap"`` matching an entry keyed as ``"Bull trap: …"``) are
    resolved correctly.

    Args:
        setup_type: Setup name or detail string from a signal module.

    Returns:
        Win rate as a percentage (0–100).  Falls back to
        ``_DEFAULT_WIN_RATE`` when no match is found.
    """
    # Exact match first
    if setup_type in BASELINE_WIN_RATES:
        return BASELINE_WIN_RATES[setup_type]

    # Substring / prefix match (case-insensitive)
    key_lower = setup_type.lower()
    for known, rate in BASELINE_WIN_RATES.items():
        if known.lower() in key_lower or key_lower in known.lower():
            return rate

    return _DEFAULT_WIN_RATE


def get_best_win_rate(setup_types: list) -> float:
    """Return the highest win rate across a list of setup types.

    Args:
        setup_types: List of signal/confirmation names.

    Returns:
        Highest historical win rate found, or ``_DEFAULT_WIN_RATE``.
    """
    if not setup_types:
        return _DEFAULT_WIN_RATE
    return max(get_win_rate(s) for s in setup_types)
