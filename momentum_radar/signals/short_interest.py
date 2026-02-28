"""
short_interest.py – Short interest and float-based signal detection.

Registered signals
------------------
- ``short_interest`` – identifies high-short-interest, low-float stocks
"""

import logging
from typing import Dict, Optional

import pandas as pd

from momentum_radar.config import config
from momentum_radar.signals.base import SignalResult
from momentum_radar.signals.scoring import register_signal

logger = logging.getLogger(__name__)


@register_signal("short_interest")
def short_interest(
    ticker: str,
    fundamentals: Optional[Dict],
    **kwargs,
) -> SignalResult:
    """Identify stocks with high short interest and small float.

    Trigger conditions (ALL must be met):
    - Short interest ≥ ``SHORT_INTEREST_MIN`` (15 %)
    - Days-to-cover ≥ ``DAYS_TO_COVER_MIN`` (3)
    - Float < ``FLOAT_MAX`` (200 M shares)

    Score: +1 if all criteria are satisfied.

    Args:
        ticker: Stock symbol.
        fundamentals: Dict from
            :meth:`~momentum_radar.data.data_fetcher.BaseDataFetcher.get_fundamentals`.

    Returns:
        :class:`~momentum_radar.signals.base.SignalResult`
    """
    cfg = config.signals

    if fundamentals is None:
        return SignalResult(triggered=False, score=0, details="No fundamental data")

    short_pct = fundamentals.get("short_percent_of_float")
    days_to_cover = fundamentals.get("short_ratio")
    float_shares = fundamentals.get("float_shares")

    missing = []
    if short_pct is None:
        missing.append("short_percent_of_float")
    if days_to_cover is None:
        missing.append("short_ratio")
    if float_shares is None:
        missing.append("float_shares")

    if missing:
        return SignalResult(
            triggered=False,
            score=0,
            details=f"Missing data: {', '.join(missing)}",
        )

    short_pct = float(short_pct)
    days_to_cover = float(days_to_cover)
    float_shares = float(float_shares)

    if (
        short_pct >= cfg.short_interest_min
        and days_to_cover >= cfg.days_to_cover_min
        and float_shares < cfg.float_max
    ):
        return SignalResult(
            triggered=True,
            score=1,
            details=(
                f"Short {short_pct:.1%}, DTC {days_to_cover:.1f}, "
                f"Float {float_shares / 1e6:.0f}M"
            ),
        )

    return SignalResult(
        triggered=False,
        score=0,
        details=(
            f"Short {short_pct:.1%}, DTC {days_to_cover:.1f}, "
            f"Float {float_shares / 1e6:.0f}M – below threshold"
        ),
    )
